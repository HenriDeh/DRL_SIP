using ProgressMeter, LinearAlgebra, Flux, Distributions, CUDA, Parameters
CUDA.allowscalar(false)

include("ReplayBuffer.jl")
include("DRL_utilities.jl")

struct DDPG_Agent{A, C, F}
	actor::A
	critic::C
	explorer::F
end

function (agent::DDPG_Agent)(state)
	s = state |> gpu
	agent.actor(s)
end

struct DDPGQN_Agent{A, C, F}
	actor::A
	critic::C
	explorer::F
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

struct EpsilonGreedy{T,F}
	explore::F
	ϵ::T
end

(eg::EpsilonGreedy)(action) = rand() > eg.ϵ ? action : eg.explore(action)

@with_kw struct DDPG_HP
	replaysize::Int = 2^16
	batchsize::Int = 128
	softsync_rate::Float64 = 0.001
	optimiser_critic = ADAM(1f-4)
	optimiser_actor = ADAM(1.66f-5)
	nstep = 1
end

function train!(agent, envi, hyperparameters::DDPG_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false, progress_bar = false)
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, optimiser_actor, optimiser_critic, nstep = hyperparameters
	target_agent = deepcopy(agent)
    test_envi = deepcopy(envi)
    test_reset!(test_envi)
	T = eltype(observe(envi))
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{T})
    fillbuffer!(replaybuffer, agent, envi)
    returns = Float64[]
	nsteps = Vector{Transition{T}}()
	period = 1
	progress = Progress(maxit)
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
		push!(nsteps, transition!(agent, envi))
		if period >= nstep
			r = zero(T)
			for t in nsteps
				r += t.r
			end
			addTransition!(replaybuffer, Transition(first(nsteps).s, first(nsteps).a, r, last(nsteps).ns, isdone(envi) ? zero(T) : one(T)))
			popfirst!(nsteps)
		end
        if isdone(envi)
			while !isempty(nsteps)
				r = zero(T)
				for t in nsteps
					r += t.r
				end
				addTransition!(replaybuffer, Transition(first(nsteps).s, first(nsteps).a, r, last(nsteps).ns, zero(T)))
				popfirst!(nsteps)
			end
			period = 0
			reset!(envi)
		end
		period += 1
        if it % test_freq == 0
            push!(returns, test_agent(agent, test_envi, 1000))
            verbose && print(Int(round(returns[end])), " ")
        end
		progress_bar && next!(progress)
    end
    runtime = (Base.time() - starttime)/60
    return returns, runtime
end

struct Agent_Bag{A}
	agents::Vector{A}
end

function (ab::Agent_Bag)(state)
	N = length(ab.agents)
	actions = [agent(state)|>cpu for agent in ab.agents]
	actions_b = hcat(actions...)
	SA = vcat(repeat(state, outer = [1,N]), actions_b) |> gpu
	values = vcat((agent.critic(SA) for agent in ab.agents)...) |> cpu
	means = reshape(median(values,dims = 1), :, N)
	return vcat(actions...)'[argmax(means, dims = 2)]' |> gpu
end
