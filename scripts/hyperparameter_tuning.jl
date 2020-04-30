using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, BSON, StatsPlots, CSV, DataFrames, Hyperopt

include(srcdir("TD3QN.jl"))
function tune()
    ho = @hyperopt for i in 1000,
        sampler in CLHSampler(dims = [
                Continuous(),
                Continuous(),
                Continuous(),
                Categorical(6),
                Categorical(5),
                Continuous(),
                Categorical(3),
                Categorical(4)]),
        actor_learning_rate in Exp10.(LinRange(-6, -1, 1000)),
        critic_learning_rate in Exp10.(LinRange(-6, -1, 1000)),
        replay_size in LinRange(10000, 100000, 1000),
        batch_size in Int.(exp2.(5:10)),
        delay = 1:5,
        epsilon in Exp10.(LinRange(-2, -1, 1000)),
        exploration in [:none, :ϵ, :ϵ0],
        width in Int.(exp2.(5:8))

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
        envi = MultiEchelon(Dt, backorder_cost, CV, [end_prod], simulation_horizon = H, expected_demand = test_μs)
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
        explore(action, envi::MultiEchelon) = rand() > epsilon ? action : [rand() > zero_epsilon ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

        agent = TD3QN_Agent(
                Chain(Dense(x_size, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, a_size, action_squashing_function(envi))) |> gpu,
                Chain(Dense((x_size + a_size), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                Chain(Dense((x_size + a_size), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                explore
        )
        returns, time = train!(agent, envi, hp, maxit = 300000, test_freq = 3000)

        gap = test_agent(agent, envi, 3000)/opt_policy_value -1
    end
    return ho
end
