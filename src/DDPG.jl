using ProgressMeter, LinearAlgebra, Flux, Distributions, CuArrays, Parameters
CuArrays.allowscalar(false)

include("ReplayBuffer.jl")
include("DRL_utilities.jl")

struct DDPG_Agent{A, C, F}
	actor::A
	critic::C
	explore::F
end

function (agent::DDPG_Agent)(state)
	s = state |> gpu
	agent.actor(s)
end

struct DDPGQN_Agent{A, C, F}
	actor::A
	critic::C
	explore::F
end

function (agent::DDPGQN_Agent)(state)
	actor = agent.actor
    critic = agent.critic
    s = state |> gpu
    a = actor(s)
    a0 = zeros(size(a)) |> gpu
    best = critic(vcat(s, a)) .> critic(vcat(s, a0))
    a .* best
end

@with_kw struct DDPG_HP
	replaysize::Int = 2^16
	batchsize::Int = 128
	softsync_rate::Float64 = 0.001
	optimiser_critic = ADAM(1f-4)
	optimiser_actor = ADAM(2.5f-5)
end

function train!(agent, envi, hyperparameters::DDPG_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false)
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, optimiser_actor, optimiser_critic = hyperparameters
	target_agent = deepcopy(agent)
    test_envi = deepcopy(envi)
    test_reset!(test_envi)
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)
    returns = Float64[]
	for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = target(r,ns,d, target_agent)
        b = (s,a,y)
        Flux.train!(
            (d...) -> critic_loss(d..., agent.critic),
            Flux.params(agent.critic),
            [b],
            optimiser_critic
        )
        softsync!(agent.critic, target_agent.critic, softsync_rate)
        s,a,r,ns,d = ksample(replaybuffer)
        Flux.train!(
            s -> actor_loss(s, agent),
            Flux.params(agent.actor),
            [s],
            optimiser_actor
        )
		softsync!(agent.actor, target_agent.actor, softsync_rate)
        addTransition!(replaybuffer, transition!(agent, envi))
        isdone(envi) && reset!(envi)
        if it % test_freq == 0
            push!(returns, test_agent(agent, test_envi, 1000))
            verbose && print(Int(round(returns[end])), " ")
        end
    end
    runtime = (Base.time() - starttime)/60
    return returns, runtime
end