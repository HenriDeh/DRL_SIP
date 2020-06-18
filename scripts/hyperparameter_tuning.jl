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
                                                gap = Float64[]))
CSV.write(datadir("exp_raw/tuning/convergence.csv"), DataFrame(run_ID = Int[], returns = Vector{Float64}[]))
CSV.write(datadir("exp_raw/tuning/instances.csv"), DataFrame(run_ID = Int[], backorder = Float64[], CV = Float64[], setup = Float64[], c = Float64[], gap = Float64[]))
CSV.write(datadir("exp_raw/tuning/test_forecasts.csv"), DataFrame(run_ID = Int[], expected_demands = Vector{Float64}[]))
function tune()
        clrs = exp10.(LinRange(-5, -3, 3))
        actor_ratios = [1,2,4,10]
        rss = Int.(exp2.(14:17))
        bss = Int.(exp2.(6:9))
        i=0
        for critic_learning_rate in clrs
        for actor_ratio in actor_ratios
        for replay_size in rss
        for batch_size in bss
                i += 1
                actor_learning_rate = critic_learning_rate/actor_ratio
                ϵ = 0.1
                width = 64
                β = 0.001
                hidden_af = elu
                output_af = abs

                μ = 10.0
                holding = 1
                backorder = rand(Uniform(2, 15))
                CV = rand(Uniform(0.1, 0.4))
                setup = rand(Uniform(0, 1300))
                production = rand(Uniform(0, backorder/2))
                H = 52
                EOQ = sqrt(μ * 2 * maximum(setup) / holding)

                reseter = repeat([(Uniform(0,2μ), Normal(CV,0))], H)
                test_forecasts = CVNormal.([rand(Uniform(0.1, 1.9μ)) for _ in 1:H], CV)

                sup = Supplier(fixed_linear_cost(setup,production))
                pro = ProductInventory(linear_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
                ma = Market(expected_hold_stockout_CVNormal(holding,backorder, CV), pro, CVNormal, true, reseter, test_reset_forecasts = test_forecasts, expected_reward = true)
                envi = InventoryProblem([sup, pro, ma])

                test_reset!(envi)
                instance = Instance(envi)
                Scarf.backward_SDP(instance, 0.1)
                opt_policy_value = test_Scarf_policy(envi, instance.S, instance.s)
                dummy_policy_value = test_Scarf_policy(envi, fill(-Inf, H), fill(-Inf, H))

                x_size = observation_size(envi)
                a_size = action_size(envi)

                hp = DDPG_HP(
                    replaysize = replay_size,
                    batchsize = batch_size,
                    optimiser_actor = ADAM(actor_learning_rate),
                    optimiser_critic = ADAM(critic_learning_rate),
                    softsync_rate = β
                )

                explore(action) = rand() > ϵ ? action : [rand() > 0.5 ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

                agent = DDPGQN_Agent(
                    Chain(Dense(x_size, width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, a_size, output_af)) |> gpu,
                    Chain(Dense((x_size + a_size), width, hidden_af), Dense(width, width, hidden_af), Dense(width, width, hidden_af), Dense(width, 1)) |> gpu,
                    explore
                )
                println("#$i -- Hyperparameters:")
                show((actor_learning_rate, critic_learning_rate, replay_size, batch_size, actor_ratio))
                println()
                println("Instance parameters:")
                show((holding, backorder, CV, setup, production))
                println()
                print("Training...")
                returns, time = train!(agent, envi, hp, maxit = 300000, test_freq = 3000)
                gap = test_agent(agent, envi, 10000)/opt_policy_value -1
                println("done. Gap = $gap")
                println()

                tuning = DataFrame(run_ID = [i],
                                    actor_lr = actor_learning_rate,
                                    critic_lr = critic_learning_rate,
                                    replay_size = replay_size,
                                    batch_size = batch_size,
                                    actor_ratio = actor_ratio,
                                    gap = [gap])
                convergence = DataFrame(run_Id = [i], returns = [returns])
                instances = DataFrame(run_Id = [i], backorder = Float64[backorder], CV = Float64[CV], setup = Float64[setup], c = [production], gap = Float64[gap])
                forecasts = DataFrame(run_ID = i, expected_demand = [mean.(test_forecasts)])
                CSV.write(datadir("exp_raw/tuning/tuning.csv"), tuning, append = true)
                CSV.write(datadir("exp_raw/tuning/convergence.csv"), convergence, append = true)
                CSV.write(datadir("exp_raw/tuning/instances.csv"), instances, append = true)
                CSV.write(datadir("exp_raw/tuning/test_forecasts.csv"), forecasts, append = true)
        end
        end
        end
        end
end

tune()
