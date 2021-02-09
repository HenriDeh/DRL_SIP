using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, CSV, DataFrames, ProgressMeter, BSON, Query
include(srcdir("DDPG.jl"))

"""
````
	experiment(variant::String = "backlog"; iterations = 300000, 
		twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, 
		expected_reward::Bool = true, N = 20, 
		critic_lr = 1f-4, actor_lr = critic_lr/8, actor_clip = 1f-6, discount = 0.99, 
		softsync_rate = 0.001, batchsize = 128, replaysize = 2^15, width = 64, 
		epsilon = 0.005, folder = "main_experiments", overwrite = true, kwargs...)
```
Run the experiments of the specified `variant` with the specified components.  

Use the keyword arguments to customize the experiments:
* `variant = "backlog"` change to "leadtime" or "lostsales" to run the respective experiments.
* `iterations = 300000` number of training iterations per agent.
* `twin = true` set to `false` to disable the twin critic.
* `annealing = 75000` set to `0` to disable the linear annealing of ``K``, set to any value to change the rate of annealing. 
* `hybrid = true` set to false to disable the discrete check.
* `expected_reward = true` set false to make the environment return a "normal" sampled reward instead of the expectation.
* `N = 20` set the number of agents trained for each environment parameter combination.
* `critic_lr = 1f-4` set the learning rate of the critic ``α_C``.
* `actor_lr = critic_lr/8` set the learning rate of the actor ``α_A``.
* `actor_clip = 1f-6` set the maximum norm of the gradient for clipping.
* `discount = 0.99` set the discount factor. 
* `softsync_rate = 0.001` set the synchronization rate ``β``.
* `batchsize = 128` set the batch size ``s_B``.
* `replaysize = 2^15` set the experience replay size ``s_R``.
* `width = 64` set the number of neurons per hidden layer`.
* `epsilon = 0.005` set the probability `ϵ` to use the exploration strategy.
* `folder = "main_experiments"` change the output folder where the experiments' CSV are recorded.
* `overwrite = true` set to false to append instead of erasing the old experiement CSV.
* `experimentname` give a user-defined experiment name for the CSV files, automatically generated if not provided.
* Use kwargs to define custom _environment parameter_ sets. Example:
    ```
    experiment("backlog",
    stockouts = [10],
    CVs = [0.4],
    order_costs = [1280],
    leadtimes = [0])
    ```
    will only experiment on one instance. Unspecified environment parameter sets will be defaulted to the respective values of the `variant` argument.
    The accepted keywords are the four above.
"""
function experiment(variant::String = "backlog"; iterations = 300000, twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, expected_reward::Bool = true, 
	N = 20, critic_lr = 1f-4, actor_lr = critic_lr/8, actor_clip = 1f-6, discount = 0.99, softsync_rate = 0.001, batchsize = 128, replaysize = 2^15, width = 64, epsilon = 0.005,
	folder = "main_experiments", overwrite = true, kwargs...)

	experimentname = get(kwargs, :experimentname, experiment_name(variant, twin = twin, annealing = annealing, hybrid = hybrid, expected_reward = expected_reward))
	path = "data/exp_raw/$folder/$(experimentname)"
	exists = isfile("$(path).csv") && !overwrite
	exists_det = isfile("$(path)_details.csv") && !overwrite
	exists_ret = isfile("$(path)_returns.csv") && !overwrite
	CSV.write("$(path).csv", DataFrame(agent_ID = [], parameters_ID = Int[], stockout = Int[], CV = [], order_cost = Int[], leadtime = Int[], mean_gap = [], time = []), append = exists)
    CSV.write("$(path)_details.csv", DataFrame(agent_ID = Int[], forecast_ID = Int[], gap = []), append = exists_det)
    CSV.write("$(path)_returns.csv", DataFrame(agent_ID = Int[], returns = []), append = exists_ret)

	i = 0
	#environment parameter sets
	holding = 1
	custom_experiment = maximum([:stockouts, :CVs, :order_costs, :leadtimes] .∈ [keys(kwargs)])
    if variant == "backlog"
        stockouts = get(kwargs, :stockouts, [5,10])
        CVs = get(kwargs, :CVs, [0.2,0.4])
        order_costs = get(kwargs, :order_costs, [80, 360, 1280])
        leadtimes = get(kwargs, :leadtimes, [0])
    elseif variant == "leadtime"
        stockouts = get(kwargs, :stockouts, [10])
        CVs = get(kwargs, :CVs, [0.4])
        order_costs = get(kwargs, :order_costs, [80,360,1280])
        leadtimes = get(kwargs, :leadtimes, [0,1,2,4,8])
    elseif variant == "lostsales"
        stockouts = get(kwargs, :stockouts, [10,25,50,100])
        CVs = get(kwargs, :CVs, [0.4])
        order_costs = get(kwargs, :order_costs, [80,360,1280])
        leadtimes = get(kwargs, :leadtimes, [0])
    end
	production = 0
	H = 52
	T = 52*2
	μ = 10

	forecasts = CSV.File("data/instances.csv") |> DataFrame!
	dataset = CSV.File("data/instances_solved.csv") |> @filter(_.Experiment == variant) |> DataFrame!
	allagents = []
	for order_cost in order_costs, stockout in stockouts, CV in CVs, leadtime in leadtimes
		order_costlow = stockout
		i += 1
		if !custom_experiment
			opt_values = []
			dataset_ID = dataset |> @filter(_.parameters_ID == i) |> DataFrame!
			for k in 1:nrow(forecasts)
				forecast = eval(Meta.parse(forecasts[k,:].forecast))
				#build test environment
				reseter = CVNormal.(forecast, CV)
				sup = Supplier(fixed_linear_cost(order_cost,production), zeros(leadtime))
				pro = ProductInventory(linear_holding_cost(holding), sup, Normal(0, 0))
				ma = Market(linear_stockout_cost(stockout), pro, CVNormal, variant != "lostsales", reseter, expected_reward = false, horizon = H)
				test_envi = InventoryProblem([sup, pro, ma])
				#collect optimal policy value
				opt_policy_value = dataset_ID.opt[k]
				push!(opt_values, (test_envi, opt_policy_value))
			end
		end
			#exploration strategy
			EOQ = sqrt(μ*2*order_cost/holding)
			explorer = EpsilonGreedy(action -> [rand() > 0.5 ? 0.0 : rand(Uniform(0, EOQ*2)) for _ in eachindex(action)], epsilon)
			#build learning environment
			reseter = repeat([(Uniform(0.1μ,1.9μ), Normal(CV,0))], T)
			sup = Supplier(fixed_linear_cost(order_cost,production), fill(Uniform(0, EOQ), leadtime), test_reset_orders = zeros(leadtime))
			pro = ProductInventory(expected_reward ? expected_holding_cost(holding) : linear_holding_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
			ma = Market(expected_reward ? expected_stockout_cost(stockout) : linear_stockout_cost(stockout), pro, CVNormal, variant != "lostsales", reseter, expected_reward = expected_reward, horizon = H, test_reset_forecasts = repeat([CVNormal(μ, CV)], T))
			envi = InventoryProblem([sup, pro, ma])
			kprog = [fixed_linear_cost(round(Int,K), production) for K in LinRange(order_costlow,order_cost,annealing÷H)]
			supdy = DynamicEnvi(sup, Dict([(:order_cost, kprog)]))	
			agents = []
			df_details = DataFrame(agent_ID = Int[], forecast_ID = Int[], gap = [])
			df_med = DataFrame(agent_ID = Int[], parameters_ID = Int[], stockout = Int[], CV = [], order_cost = Int[], leadtime = [], mean_gap = [], time = [])
		for j in 1:N
			reset!(supdy)
			agent_ID = (i-1)*N + j
			println("-----------------------------------")
			println("Training agent $agent_ID...")
			#build agent
            hp = DDPG_HP(
				replaysize = replaysize,
				batchsize = batchsize,
				softsync_rate = softsync_rate,
				discount = discount,
				optimiser_critic = ADAM(critic_lr),
				optimiser_actor = Flux.Optimiser(Flux.ClipNorm(actor_clip), ADAM(actor_lr))
			)
            if twin
                if hybrid
                    agent_type = TD3QN_Agent
                else
                    agent_type = TD3_Agent
                end
            else
                if hybrid
                    agent_type = DDPGQN_Agent
                else
                    agent_type = DDPG_Agent
                end
            end
			agent = agent_type(
			        Chain(  Dense(observation_size(envi), width, relu),
			                Dense(width, width, relu),
			                Dense(width, width, relu),
			                Dense(width, action_size(envi), relu)) |> gpu,
			        Chain(  Dense((observation_size(envi) + action_size(envi)), width, relu),
			                Dense(width, width, relu),
			                Dense(width, width, relu),
			                Dense(width, 1)) |> gpu,
			        explorer
			)
			#train agent
			returns, time = train!(agent, envi, hp, maxit =  iterations, progress_bar = true, test_freq = 3000, dynamic_envis = [supdy])
			CSV.write("$(path)_returns.csv", DataFrame(agent_ID = agent_ID, returns = [returns]), append = true)
			#benchmark on instance dataset
			if !custom_experiment
				println("Benchmarking on dataset...")
				gaps = Float64[]
				@showprogress for (k,(test_envi, opt_policy_value)) in enumerate(opt_values)
					agent_value = test_agent(agent, test_envi, 1000)
					gap = agent_value/opt_policy_value - 1
					push!(gaps, gap)
					push!(df_details, [agent_ID, k, gap])
				end
				mgap = mean(gaps)
				push!(df_med, [agent_ID, i, stockout, CV, order_cost, leadtime, mgap, time])
				println("mean gap = $mgap")
				println()
			end
			push!(agents, agent)
			push!(allagents, cpu(agent))
		end
		if !custom_experiment
			CSV.write("$(path).csv", df_med, append = true)
			CSV.write("$(path)_details.csv", df_details, append = true)
		end
	end
	BSON.@save "data/pretrained_agents/$folder/$(experimentname)_agents_experiment.bson" allagents
	return nothing
end

"""
`experiment_name(variant::String = "backlog"; iterations = 300000, twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, expected_reward::Bool = true)``

Generate an automatic name for the experiment's CSV files. Not called if user defined name is provided to `experiment()`.
"""
function experiment_name(variant::String = "backlog"; iterations = 300000, twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, expected_reward::Bool = true)
    @assert variant ∈ ("backlog", "leadtime", "lostsales") "variant must be one of: backlog, leadtime, lostsales"
    experimentname = variant * (hybrid ? "-hybrid" : "-continuous")* (expected_reward ? "-expected" : "-sample")*(annealing == 0 ? "-Kfixed" : "-Kannealed") *(twin ? "-twin" : "-notwin")    
    return experimentname
end