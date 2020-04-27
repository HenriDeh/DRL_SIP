using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, BSON, StatsPlots, CSV, DataFrames, Hyperopt

include(srcdir("TD3QN.jl"))

ho = @hyperopt for i in 1000,
    sampler in CLHSampler(
        dims = [
            Continuous(),
            Continuous(),
            Continuous(),
            Categorical(6),
            Categorical(5),
            Categorical(2),
            Continuous(),
            Categorical(3),
            Categorical(2),
            Categorical(4),
        ],
    ),
    actor_learning_rate in Exp10.(LinRange(-6, -1, 1000)),
    critic_learning_rate in Exp10.(LinRange(-6, -1, 1000)),
    replay_size in LinRange(10000, 100000, 1000),
    batch_size in Int.(exp2.(5:10)),
    delay = 1:5,
    twins in [true, false],
    epsilon in Exp10.(LinRange(-2, -1, 1000)),
    exploration in [:none, :ϵ, :ϵ0],
    TD3QN_agent in [true, false],
    hidden_layers_width in Int.(exp2.(5:8)),
    target_actor in [true, false]

    μ = 10.0
    holding = 1
    backorder_cost = rand(Uniform(2, 10))
    CV = rand(Uniform(0.1, 0.4))
    setup = rand(Uniform(0, 1300))
    production = 0
    T = 52
    H = 70
    EOQ = sqrt(μ * 2 * maximum(setup) / holding)
    Dt = MultiDist(Uniform(0, 2 * μ), H)
    test_μs = rand(Dt)
    on_hand_dist = Uniform(-μ, 2 * μ)
    end_prod = Inventory(holding, setup, production, on_hand_dist, on_hand = 0)
    envi = MultiEchelon(
        Dt,
        backorder_cost,
        CV,
        [end_prod],
        simulation_horizon = H,
        expected_demand = test_μs,
    )

    test_reset!(envi)
    instance = Instance(envi)
    Scarf.backward_SDP(instance)
    opt_policy_value = test_policy(envi, instance.S, instance.s)
    dummy_policy_value = 0.95 * test_policy(envi, fill(-Inf, H), fill(-Inf, H))
    hp = (
        actorlr = actor_learning_rate,
        criticlr = critic_learning_rate,
        replaysize = replay_size,
        batchsize = batch_size,
        layerwidth = hidden_layers_width,
        softsync_rate = 0.01,
        maxit = 500000,
        delay = delay,
        target_actor = target_actor
    )
    actorlr = 1f-4, criticlr = 1f-3, replaysize = 30000, batchsize = 512, softsync_rate = 0.01, delay::Int = 1, layerwidth = 64, target_actor = false,
    maxit = 500000, test_freq::Int = hp.maxit, verbose = false
    if exploration == :none
        epsilon = 0.0
    end
    if exploration == :ϵ
        zero_epsilon = 0.0
    else
        zero_epsilon = 0.5
    end
    explore(action, envi::MultiEchelon) = rand() > epsilon ? action : [rand() > zero_epsilon ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

    lastrun = TD3QN(envi; hp..., test_freq = div(hp.maxit,100))
end
