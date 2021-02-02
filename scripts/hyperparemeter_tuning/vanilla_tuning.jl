using DrWatson
@quickactivate "DRL_SIP"
include(srcdir("experiment.jl"))


function tune()
    actor_ratios = [1/4,1/8]
    actor_clips = [Inf]
    batchsizes = [64,128]
    replaysizes = [2^15]
    i = 0
    for actor_ratio in actor_ratios, actor_clip in actor_clips, batchsize in batchsizes, replaysize in replaysizes
        i += 1
        experiment("backlog", annealing = 0, hybrid = false, N = 12, critic_lr = 1f-4, actor_lr = 1f-4*actor_ratio, actor_clip = actor_clip, batchsize = 128, replaysize = 2^15, 
            folder = "tuning", experimentname = "vanilla2", overwrite = false, stockouts = [10], order_costs = [1280], CVs = [0.2])
        CSV.write("data/exp_raw/tuning/vanilla_hps2.csv", DataFrame(ID = i, actor_ratio = actor_ratio, actor_clip = actor_clip, batchsize = batchsize, replaysize = replaysize), append = true)
    end
end 

tune()

