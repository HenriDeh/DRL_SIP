using ProgressMeter, LinearAlgebra, Flux, Distributions, CuArrays, Parameters
CuArrays.allowscalar(false)

include("ReplayBuffer.jl")
include("Agent.jl")
include("utilities.jl")

function twin_target(reward::T, next_state::T, done::T, agent::TD3QN_Agent) where T <: AbstractArray
    next_action = agent(next_state, actor = agent.tactor)
    Qprime1 = agent.tcritic(vcat(next_state, next_action))
    Qprime2 = agent.ttwin_critic(vcat(next_state, next_action))
    return reward .+ (min.(Qprime1, Qprime2) .* done)
end

function critic_loss(state::T, action::T, y::T, critic) where T <: AbstractArray
    Q = critic(vcat(state, action))
    return Flux.mse(Q, y)
end

function actor_loss(state::T, agent::TD3QN_Agent) where T <: AbstractArray
    return -mean(agent.critic(vcat(state, agent.actor(state))))
end

@with_kw struct TD3QN_HP
    replaysize::Int = 30000
    batchsize::Int = 128
    softsync_rate::Float64 = 0.01
    delay::Int = 1
    optimiser_actor
    optimiser_critic
end

function train!(agent::TD3QN_Agent, envi, hyperparameters::TD3QN_HP; maxit::Int = 500000, test_freq::Int = maxit)
    starttime = Base.time()
    @unpack replaysize, batchsize, delay, softsync_rate, optimiser_actor, optimiser_critic = hyperparameters

    test_envi = deepcopy(envi)
    test_reset!(test_envi)
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)
    returns = Float64[]
    @showprogress for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = twin_target(r,ns,d, agent)
        b = (s,a,y)
        if it % 2 == 0
            Flux.train!(
                (d...) -> critic_loss(d..., agent.critic),
                Flux.params(agent.critic),
                [b],
                optimiser_critic
            )
            softsync!(agent.critic, agent.tcritic, softsync_rate)
        else
            Flux.train!(
                (d...) -> critic_loss(d..., agent.twin_critic),
                Flux.params(agent.twin_critic),
                [b],
                optimiser_critic
            )
            softsync!(agent.twin_critic, agent.ttwin_critic, softsync_rate)
        end
        if it % delay == 0
            s,a,r,ns,d = ksample(replaybuffer)
            Flux.train!(
                s -> actor_loss(s, agent),
                Flux.params(agent.actor),
                [s],
                optimiser_actor
            )
            softsync!(agent.actor, agent.tactor, softsync_rate)
        end
        addTransition!(replaybuffer, transition!(agent, envi))
        isdone(envi) && reset!(envi)
        if it % test_freq == 0
            push!(returns, test_agent(agent, test_envi))
        end
    end
    runtime = (Base.time() - starttime)/60
    return returns, runtime
end
