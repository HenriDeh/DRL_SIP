using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, CSV, DataFrames, Hyperopt

include(srcdir("TD3QN.jl"))
const n = 1000

CSV.write(datadir("tuning.csv"), DataFrame(run_ID = Int[], actor_lr = Float64[], critic_lr = Float64[], replay_size = Int[], batch_size = Int[], delay = Int[], epsilon = Int[], exploration = String[], target_actor = Bool[], gap = Float64[]))
CSV.write(datadir("convergence.csv"), DataFrame(run_Id = Int[], returns = Vector{Float64}[]))
CSV.write(datadir("instances.csv"), DataFrame(run_Id = Int[], backorder = Float64[], CV = Float64[], setup = Float64[], gap = Float64[]))

function tune()
    ho = @hyperopt for i in n,
                        sampler in CLHSampler(dims = [Continuous(),Continuous(),Continuous(),Categorical(6),Categorical(5),Continuous(),Categorical(3),Categorical(4),Categorical(2)]),
                        actor_learning_rate in Exp10.(LinRange(-6, -2, n)),
                        critic_learning_rate in Exp10.(LinRange(-6, -2, n)),
                        replay_size in LinRange(10000, 100000, n),
                        batch_size in Int.(exp2.(5:10)),
                        delay = 1:5,
                        epsilon in Exp10.(LinRange(-2, -1, n)),
                        exploration in [:none, :ϵ, :ϵ0],
                        width in Int.(exp2.(5:8)),
                        target_actor in (true,false)

        μ = 10.0
        holding = 1
        backorder_cost = rand(Uniform(2, 15))
        CV = rand(Uniform(0.1, 0.4))
        setup = rand(Uniform(0, 1300))
        production = 0
        T = 52
        H = 70
        EOQ = sqrt(μ * 2 * maximum(setup) / holding)

        reseter = repeat([(Uniform(0,2μ), Normal(CV,0))], H)
        test_forecasts = CVNormal.([rand(Uniform(0, 2μ)) for _ in 1:H], CV)

        sup = Supplier(fixed_linear_cost(setup,production))
        pro = ProductInventory(linear_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
        ma = Market(expected_hold_stockout_CVNormal(holding,backorder, CV), pro, CVNormal, true, reseter, test_reset_forecasts = test_forecasts, expected_reward = true, visibility = T)
        envi = InventoryProblem([sup, pro, ma])

        x_size = observation_size(envi)
        a_size = action_size(envi)

        test_reset!(envi)
        instance = Instance(envi)
        Scarf.backward_SDP(instance)
        opt_policy_value = test_policy(envi, instance.S, instance.s)
        dummy_policy_value = test_policy(envi, fill(-Inf, H), fill(-Inf, H))

        hp = TD3QN_HP(
            replaysize = replay_size,
            batchsize = batch_size,
            delay = delay
            optimiser_actor = Flux.Optimiser(ADAM(actor_learning_rate), ClipNorm(1))
            optimiser_critic = Flux.Optimiser(ADAM(critic_learning_rate), ClipNorm(1))
        )

        if exploration == :none
            epsilon = 0.0
        end
        if exploration == :ϵ
            zero_epsilon = 0.0
        else
            zero_epsilon = 0.5
        end
        explore(action) = rand() > epsilon ? action : [rand() > zero_epsilon ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

        agent = TD3QN_Agent(
                Chain(Dense(x_size, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, a_size, action_squashing_function(envi))) |> gpu,
                Chain(Dense((x_size + a_size), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                Chain(Dense((x_size + a_size), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                explore,
                target_actor = target_actor
        )
        returns, time = train!(agent, envi, hp, maxit = 300000, test_freq = 3000)

        gap = test_agent(agent, envi, 3000)/opt_policy_value -1

        tuning = DataFrame(run_ID = [i], actor_lr = [actor_learning_rate], critic_lr = [critic_learning_rate], replay_size = [replay_size], batch_size = [batch_size], delay = [delay], epsilon = [epsilon], exploration = [String(exploration)], target_actor = [target_actor], gap = [gap])
        convergence = DataFrame(run_Id = [i], returns = [returns])
        instances = DataFrame(run_Id = [i], backorder = Float64[backorder_cost], CV = Float64[CV], setup = Float64[setup], gap = Float64[gap])

        CSV.write(datadir("tuning.csv"), tuning, append = true)
        CSV.write(datadir("convergence.csv"), convergence, append = true)
        CSV.write(datadir("instances.csv"), instances, append = true)

        gap
    end
    return ho
end

CSV.write(datadir("test.csv"), DataFrame(h1 = [4,5,6], h2 = ['a', 'b', 'c']))
