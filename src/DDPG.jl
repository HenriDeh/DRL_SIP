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

@with_kw struct DDPG_HP
	replaysize::Int = 2^16
	batchsize::Int = 128
	softsync_rate::Float64 = 0.001
	optimiser_critic = ADAM(1f-4)
	optimiser_actor = ADAM(1.66f-5)
end

Flux.gpu(a::DDPGQN_Agent) = DDPGQN_Agent(a.actor |> gpu, a.critic |> gpu, a.explorer) 
Flux.cpu(a::DDPGQN_Agent) = DDPGQN_Agent(a.actor |> cpu, a.critic |> cpu, a.explorer)

function train!(agent, envi, hyperparameters::DDPG_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false, progress_bar = false)
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, optimiser_actor, optimiser_critic = hyperparameters
	target_agent = deepcopy(agent)
    test_envi = deepcopy(envi)
    test_reset!(test_envi)
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)
    returns = Float64[]
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
        addTransition!(replaybuffer, transition!(agent, envi))
        isdone(envi) && reset!(envi)
        if it % test_freq == 0
            push!(returns, test_agent(agent, test_envi, 1000))
            verbose && print(Int(round(returns[end])), " ")
        end
		progress_bar && next!(progress, showvalues = [(:it, it), ("Last return", isempty(returns) ? "NA" : last(returns))])
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
