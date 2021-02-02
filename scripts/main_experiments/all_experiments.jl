using DrWatson
@quickactivate "DRL_SIP"
include(srcdir("experiment.jl"))

#experiment("backlog", twin = true, annealing = 75000, hybrid = true, expected_reward = true) # Full
#experiment("backlog", twin = true, annealing = 0, hybrid = true, expected_reward = true) # No Annealing
experiment("backlog", twin = true, annealing = 0, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # Vanilla
experiment("backlog", twin = true, annealing = 75000, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # No discrete check
#=
#experiment("backlog", twin = false, annealing = 75000, hybrid = true, expected_reward = true) # No TD3
experiment("backlog", twin = false, annealing = 0, hybrid = false, expected_reward = false) # True Vanilla
experiment("backlog", twin = true, annealing = 75000, hybrid = true, expected_reward = false) # No expected reward
=#

#experiment("leadtime", twin = true, annealing = 75000, hybrid = true, expected_reward = true) # Full
#experiment("leadtime", twin = true, annealing = 0, hybrid = true, expected_reward = true) # No Annealing
experiment("leadtime", twin = true, annealing = 0, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # Vanilla
experiment("leadtime", twin = true, annealing = 75000, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # No discrete check
#=
#experiment("leadtime", twin = false, annealing = 75000, hybrid = true, expected_reward = true) # No TD3
experiment("leadtime", twin = false, annealing = 0, hybrid = false, expected_reward = false) # True Vanilla
experiment("leadtime", twin = true, annealing = 75000, hybrid = true, expected_reward = false) # No expected reward
=#
#experiment("lostsales", twin = true, annealing = 75000, hybrid = true, expected_reward = true) # Full
#experiment("lostsales", twin = true, annealing = 0, hybrid = true, expected_reward = true) # No Annealing
experiment("lostsales", twin = true, annealing = 0, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # Vanilla
experiment("lostsales", twin = true, annealing = 75000, hybrid = false, expected_reward = true, actor_clip = Inf, batchsize = 64) # No discrete check
#=
#experiment("lostsales", twin = false, annealing = 75000, hybrid = true, expected_reward = true) # No TD3
experiment("lostsales", twin = false, annealing = 0, hybrid = false, expected_reward = false) # True Vanilla
experiment("lostsales", twin = true, annealing = 75000, hybrid = true, expected_reward = false) # No expected reward
=#
