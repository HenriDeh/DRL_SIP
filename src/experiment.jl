using InventoryModels, CSV, DataFrames, ProgressMeter, BSON, Query
include(srcdir("DDPG.jl"))

function experiment(;variant::String = "backlog", iterations = 300000, twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, expected_reward::Bool = true)
    @assert variant ∈ ("backlog", "leadtime", "lostsales") "variant must be one of: backlog, leadtime, lostsales"
    if twin == false && annealing == 0 && hybrid == false && expected_reward == false
        experimentname = variant*"_vanilla"
    else
        experimentname = variant * (hybrid ? "-hybrid" : "-continuous")* (expected_reward ? "-expected" : "-sample")*(annealing == 0 ? "-Kfixed": "-Kannealed") *(twin ? "-twin" : "-notwin")
    end
    CSV.write("data/exp_raw/main_experiments/$(experimentname).csv", DataFrame(agent_ID = [], parameters_ID = Int[], stockout = Int[], CV = [], order_cost = Int[], leadtime = Int[], mean_gap = [], time = []))
    CSV.write("data/exp_raw/main_experiments/$(experimentname)_details.csv", DataFrame(agent_ID = Int[], forecast_ID = Int[], gap = []))
    CSV.write("data/exp_raw/main_experiments/$(experimentname)_returns.csv", DataFrame(agent_ID = Int[], returns = []))

    N = 20 #number of agents trained per environment parameterization
	i = 0
	#environment parameter sets
    holding = 1
    if variant == "backlog"
        stockouts = [5,10]
        CVs = [0.2,0.4]
        order_costs = [80, 360, 1280]
        leadtimes = [0]
    elseif variant == "leadtime"
        stockouts = [10]
        CVs = [0.4]
        order_costs = [80,360,1280]
        leadtimes = [0,1,2,4,8]
    elseif variant == "lostsales"
        stockouts = [10,25,50,100]
        CVs = [0.4]
        order_costs = [80,360,1280]
        leadtimes = [0]
    end
	production = 0
	H = 52
	T = 52*2
	μ = 10
	#agent hyperparameters
	epsilon = 0.005
	width = 64
	forecasts = CSV.File("data/instances.csv") |> DataFrame!
	dataset = CSV.File("data/instances_solved.csv") |> @filter(_.Experiment == variant) |> DataFrame!
	allagents = []
	for order_cost in order_costs, stockout in stockouts, CV in CVs, leadtime in leadtimes
		order_costlow = stockout
		i += 1
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
		#exploration strategy
		EOQ = sqrt(μ*2*order_cost/holding)
		explorer = EpsilonGreedy(action -> [rand() > 0.5 ? 0.0 : rand(Uniform(0, EOQ*2)) for _ in eachindex(action)], epsilon)
		#build learning environment
		reseter = repeat([(Uniform(0.1μ,1.9μ), Normal(CV,0))], T)
        sup = Supplier(fixed_linear_cost(order_cost,production), fill(Uniform(0, EOQ), leadtime), test_reset_orders = zeros(leadtime))
		pro = ProductInventory(expected_reward ? expected_holding_cost(holding): linear_holding_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
		ma = Market(expected_reward ? expected_stockout_cost(stockout) : linear_stockout_cost(stockout), pro, CVNormal, variant != "lostsales", reseter, expected_reward = expected_reward, horizon = H, test_reset_forecasts = repeat([CVNormal(μ, CV)], T))
		envi = InventoryProblem([sup, pro, ma])
		kprog = [fixed_linear_cost(round(Int,K), production) for K in LinRange(order_costlow,order_cost,annealing÷H)]
		supdy = DynamicEnvi(sup, Dict([(:order_cost, kprog)]))	
		agents = []
		df_details = DataFrame(agent_ID = Int[], forecast_ID = Int[], gap = [])
		df_med = DataFrame(agent_ID = Int[], parameters_ID = Int[], stockout = Int[], CV = [], order_cost = Int[], mean_gap = [], time = [])
		for j in 1:N
			reset!(supdy)
			agent_ID = (i-1)*N + j
			println("-----------------------------------")
			println("Training agent $agent_ID...")
			#build agent
            hp = DDPG_HP()
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
			CSV.write("data/exp_raw/main_experiments/$(experimentname)_returns.csv", DataFrame(agent_ID = agent_ID, returns = [returns]), append = true)
			#benchmark on instance dataset
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
			push!(agents, agent)
			push!(allagents, cpu(agent))
		end
		CSV.write("data/exp_raw/main_experiments/$(experimentname).csv", df_med, append = true)
		CSV.write("data/exp_raw/main_experiments/$(experimentname)_details.csv", df_details, append = true)
	end
	BSON.@save "data/pretrained_agents/$(experimentname)_agents_experiment.bson" allagents
	return nothing
end

function experiment_name(variant::String = "backlog"; iterations = 300000, twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, expected_reward::Bool = true)
    @assert variant ∈ ("backlog", "leadtime", "lostsales") "variant must be one of: backlog, leadtime, lostsales"
    if twin == false && annealing == 0 && hybrid == false && expected_reward == false
        experimentname = variant*"_vanilla"
    else
        experimentname = variant * (hybrid ? "-hybrid" : "-continuous")* (expected_reward ? "-expected" : "-sample")*(annealing == 0 ? "-Kfixed" : "-Kannealed") *(twin ? "-twin" : "-notwin")
    end
    return experimentname
end