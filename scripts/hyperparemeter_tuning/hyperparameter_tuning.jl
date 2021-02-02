using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, CSV, DataFrames

include(srcdir("DDPG.jl"))

CSV.write(datadir("exp_raw/tuning/tuning.csv"), DataFrame(run_ID = Int[],
                                                actor_lr = Float64[],
                                                critic_lr = Float64[],
                                                replay_size = Int[],
                                                batch_size = Int[],
                                                actor_ratio = Int[],
                                                gap = Float64[],
                                                time = Float64[]))
function tune()
    μ = 10.0
    holding = 1
    backorder = 10
    CV = 0.2
    setup = 1280
    production = 0
    H = 52
    T = 2*H
    EOQ = sqrt(μ*2*setup/holding)
    setuplow = backorder

    reseter = repeat([(Uniform(0.1μ,1.9μ), Normal(CV,0))], T)
    test_forecasts = [CVNormal(μ, CV) for i in 1:T]

    sup = Supplier(fixed_linear_cost(setup,production))
    pro = ProductInventory(expected_holding_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
    ma = Market(expected_stockout_cost(backorder), pro, CVNormal, true, reseter, test_reset_forecasts = test_forecasts, expected_reward = true, horizon = H)
    envi = InventoryProblem([sup, pro, ma])

    test_reset!(envi)
    instance = Instance(envi)
    Scarf.backward_SDP(instance, 0.1)
    opt_policy_value = test_Scarf_policy(envi, instance.S, instance.s)

    clrs = exp10.(LinRange(-5, -3, 3))
    actor_ratios = [1,2,4,8,16]
    rss = Int.(exp2.(14:17))
    bss = Int.(exp2.(5:8))
    i=0
    for critic_learning_rate in clrs, actor_ratio in actor_ratios, replay_size in rss, batch_size in bss
        i += 1
        #using setup cost annealing
        kprog = [fixed_linear_cost(round(Int,K), production) for K in LinRange(setuplow,setup,75000÷52)]
        supdy = DynamicEnvi(sup, Dict([(:order_cost, kprog)]))
        reset!(supdy)

        actor_learning_rate = critic_learning_rate/actor_ratio
        ϵ = 0.01
        width = 32
        hidden_af = relu
        output_af = relu
        x_size = observation_size(envi)
        a_size = action_size(envi)

        hp = DDPG_HP(
            replaysize = replay_size,
            batchsize = batch_size,
            optimiser_actor = Flux.Optimiser(Flux.ClipNorm(actor_learning_rate), ADAM(actor_learning_rate)),
            optimiser_critic = ADAM(critic_learning_rate)
        )

        explorer = EpsilonGreedy(action -> [rand() > 0.5 ? 0.0 : rand(Uniform(0, EOQ*2)) for _ in eachindex(action)], ϵ)

        agent = DDPGQN_Agent(
            Chain(Dense(x_size, width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, a_size, output_af)) |> gpu,
            Chain(Dense((x_size + a_size), width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, 1)) |> gpu,
            explorer
        )
        println("#$i -- Hyperparameters:")
        show((actor_learning_rate, critic_learning_rate, replay_size, batch_size, actor_ratio))
        println()
        print("Training...")
        returns, time = train!(agent, envi, hp, maxit = 300000, progress_bar = true, dynamic_envis = [supdy])
        gap = test_agent(agent, envi, 10000)/opt_policy_value -1
        println("done. Gap = $gap")
        println()

        tuning = DataFrame(run_ID = i,
                            actor_lr = actor_learning_rate,
                            critic_lr = critic_learning_rate,
                            replay_size = replay_size,
                            batch_size = batch_size,
                            actor_ratio = actor_ratio,
                            gap = gap,
                            time = time)
        CSV.write(datadir("exp_raw/tuning/tuning.csv"), tuning, append = true)
    end
end

tune()
