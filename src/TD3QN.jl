include("../ReplayBuffer.jl")
include("Agent.jl")
using ProgressMeter, LinearAlgebra, Zygote
println("check onedrive 5")
mutable struct ClipNorm{T}
    thresh::T
end

function Flux.Optimise.apply!(o::ClipNorm, x, Δ)
    Δnrm = norm(Δ)
    if Δnrm > o.thresh
        rmul!(Δ, o.thresh / Δnrm)
    end
    return Δ
end

function twin_target(   reward::T,
                        next_state::T,
                        done::T,
                        agent::TD3QN_Agent) where T <: AbstractArray
    next_action = agent(next_state)
    Qprime1 = agent.target_critic1(vcat(next_state, next_action)::T) .* done
    Qprime2 = agent.target_critic2(vcat(next_state, next_action)::T) .* done
    return reward .+ min.(Qprime1, Qprime2)
end

function huber_loss(ŷ, y;  δ=eltype(ŷ)(1))
   abs_error = abs.(ŷ .- y)
   temp = Flux.Zygote.dropgrad(abs_error .<  δ)
   x = eltype(ŷ)(0.5)
   hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end

function critic_loss(   state::T,
                        action::T,
                        y::T,
                        critic) where T <: AbstractArray
    Q = critic(vcat(state, action)::T)
    #δ = Q .- y
    #losses = δ .^2
    return Flux.mse(Q, y)
end

function actor_loss(state::T,
                    agent::TD3QN_Agent) where T <: AbstractArray
    return -mean(agent.critic1(vcat(state, agent.actor(state))::T))
end

mutable struct Run
    actor
    critic
    envi
    returns
    kwargs
    fails::Int
    time::Float64
end

function softsync!(net::C, target_net::C, α) where C <: Chain
    @inbounds for i in eachindex(net.layers)
        net.layers[i] isa Dense || continue
        W = net.layers[i].W
        b = net.layers[i].b
        tW = target_net.layers[i].W
        tb = target_net.layers[i].b
        tW .= ((1 - α) .* tW) .+ (α .* W)
        tb .= ((1 - α) .* tb) .+ (α .* b)
    end
    return nothing
end

function TD3QN( envi;
                actorlr = 1f-4, criticlr = 1f-3, replaysize = 30000, batchsize = 512, layerwidth = 64, softsync_rate = 0.01,
                maxit = 500000, pretrain_critic = true, test_freq::Int = hp.maxit, fails::Int = 0, max_fails::Int = 10, failure_threshold = -Inf,
                delay::Int = 2, verbose = false)
    starttime = Base.time()
    if fails > 0
        println("Run failed. Retrying...")
    end
    optA = Flux.Optimiser([ADAM(actorlr), ClipNorm(one(actorlr))])
    optC1 = Flux.Optimiser([ADAM(criticlr), ClipNorm(one(actorlr))])
    optC2 = Flux.Optimiser([ADAM(criticlr), ClipNorm(one(actorlr))])

    actor = Chain(  Dense(observation_size(envi), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, action_size(envi))
                ) |> gpu
    critic1 = Chain( Dense((observation_size(envi) + action_size(envi)), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, 1)
                ) |> gpu
    critic2 = Chain( Dense((observation_size(envi) + action_size(envi)), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, 1)
                ) |> gpu

    agent = TD3QN_Agent(actor, deepcopy(actor), critic1, deepcopy(critic1), critic2, deepcopy(critic2))
    T = Transition{eltype(observe(envi))}
    replaybuffer = ReplayBuffer(replaysize, batchsize, T)
    fillbuffer!(replaybuffer, agent, envi)
    if pretrain_critic
        for i = 1:2000
            batch = ksample(replaybuffer)
            y = twin_target(batch[3:5]..., agent)
            b = (batch[1:2]...,y)
            Flux.train!(
                (d...) -> critic_loss(d..., agent.critic1),
                Flux.params(agent.critic1),
                Iterators.repeated(b, 1),
                optC1
            )
            batch = ksample(replaybuffer)
            y = twin_target(batch[3:5]..., agent)
            b = (batch[1:2]...,y)
            Flux.train!(
                (d...) -> critic_loss(d..., agent.critic2),
                Flux.params(agent.critic2),
                Iterators.repeated(b, 1),
                optC2
            )
            softsync!(agent.critic1, agent.target_critic1, softsync_rate)
            softsync!(agent.critic2, agent.target_critic2, softsync_rate)
        end
    end
    returns = typeof(failure_threshold)[]
    @showprogress for it = 1:maxit
        batch = ksample(replaybuffer)
        y = twin_target(batch[3:5]..., agent)
        b = (batch[1:2]...,y)
        Flux.train!(
            (d...) -> critic_loss(d..., agent.critic1),
            Flux.params(agent.critic1),
            Iterators.repeated(b, 1),
            optC1
        )
        batch = ksample(replaybuffer)
        y = twin_target(batch[3:5]..., agent)
        b = (batch[1:2]...,y)
        Flux.train!(
            (d...) -> critic_loss(d..., agent.critic2),
            Flux.params(agent.critic2),
            Iterators.repeated(b, 1),
            optC2
        )

        if it % delay == 0
            batch = ksample(replaybuffer)
            Flux.train!(
                (d...) -> actor_loss(d[1], agent),
                Flux.params(agent.actor),
                Iterators.repeated(batch, 1),
                optA
            )
            softsync!(agent.actor, agent.target_actor, softsync_rate)
            softsync!(agent.critic1, agent.target_critic1, softsync_rate)
            softsync!(agent.critic2, agent.target_critic2, softsync_rate)
        end
        addTransitions!(replaybuffer, [transition!(agent, envi)])
        isdone(envi) && reset!(envi)
        if it % test_freq == 0
            push!(returns, testAgent(envi, agent))
            verbose && println(returns[end])
            if isnan(returns[end]) || returns[end] <= failure_threshold
                fails += 1
                if fails >= max_fails break end
                return TD3QN(envi; actorlr = actorlr, criticlr = criticlr, replaysize = replaysize, batchsize = batchsize, layerwidth = layerwidth, softsync_rate = softsync_rate,
                        maxit = maxit, pretrain_critic = pretrain_critic, test_freq = test_freq, failure_threshold = failure_threshold, fails = fails, max_fails = max_fails, delay = delay)
            end
        end
    end
    if fails >= max_fails
        println("Run failed $fails times. Terminating.")
    end
    #println("Return: ", returns[end])
    runtime = (Base.time() - starttime)/60
    kwargs = (actorlr = actorlr, criticlr = criticlr, replaysize = replaysize, batchsize = batchsize, layerwidth = layerwidth, softsync_rate = softsync_rate,
            maxit = maxit, pretrain_critic = pretrain_critic, test_freq = test_freq, failure_threshold = failure_threshold, max_fails = max_fails)
    return Run(agent.actor |> cpu, agent.critic1 |> cpu, envi, returns, kwargs, fails, runtime)
end

#check onedrive 3
