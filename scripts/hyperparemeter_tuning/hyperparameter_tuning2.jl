using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, CSV, DataFrames

include(srcdir("DDPG.jl"))

CSV.write(datadir("exp_raw/tuning/tuning2.csv"), DataFrame(run_ID = Int[],
                                                epsilon = Float64[],
                                                width = Int[],
                                                clip = [],
                                                gap = Float64[]))                                        
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

    ϵs = [0.05,0.025,0.01, 0.005, 0.0025]
    widths = [32,64,128,256]
    clips = [Inf, 1, 1f-4, 1.25f-5, 1f-6]
    i=0
    for ϵ in ϵs, width in widths, clip in clips
        i += 1

        kprog = [fixed_linear_cost(round(Int,K), production) for K in LinRange(setuplow,setup,75000÷52)]
        supdy = DynamicEnvi(sup, Dict([(:order_cost, kprog)]))
        reset!(supdy)

        hidden_af = relu
        output_af = relu

        x_size = observation_size(envi)
        a_size = action_size(envi)

        hp = DDPG_HP(
            optimiser_actor = Flux.Optimiser(Flux.ClipNorm(clip), ADAM(1.25f-5))
        )
        
        explorer = EpsilonGreedy(action -> [rand() > 0.5 ? 0.0 : rand(Uniform(0, EOQ*2)) for _ in eachindex(action)], ϵ)

        agent = DDPGQN_Agent(
            Chain(Dense(x_size, width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, a_size, output_af)) |> gpu,
            Chain(Dense((x_size + a_size), width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, 1)) |> gpu,
            explorer
        )
        println("#$i -- Hyperparameters:")
        show((ϵ, width, clip))
        println()
        print("Training...")
        returns, time = train!(agent, envi, hp, maxit = 300000, progress_bar = true, test_freq = 3000, dynamic_envis = [supdy])
        gap = test_agent(agent, envi, 10000)/opt_policy_value -1
        println("done. Gap = $gap")
        println()

        tuning = DataFrame(run_ID = i,
                            epsilon = ϵ,
                            width = width,
                            clip = clip,
                            gap = [gap])
        CSV.write(datadir("exp_raw/tuning/tuning2.csv"), tuning, append = true)
    end
end

tune()
