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
	replaysize::Int = 2^15
	batchsize::Int = 128
    softsync_rate::Float64 = 0.001
    discount::Float32 = 0.99f0
	optimiser_critic = ADAM(1f-4)
	optimiser_actor = Flux.Optimiser(Flux.ClipNorm(1f-6), ADAM(1.25f-5))
end

Flux.gpu(a::DDPGQN_Agent) = DDPGQN_Agent(a.actor |> gpu, a.critic |> gpu, a.explorer) 
Flux.cpu(a::DDPGQN_Agent) = DDPGQN_Agent(a.actor |> cpu, a.critic |> cpu, a.explorer)

function train!(agent, envi, hyperparameters::DDPG_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false, progress_bar = false, dynamic_envis = [])
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, discount, optimiser_actor, optimiser_critic = hyperparameters
	target_agent = deepcopy(agent)
    
    
    test_envi = deepcopy(envi)
    test_reset!(test_envi)
    
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)

    returns = Float64[]
	progress = Progress(maxit)
	for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = target(r,ns,d,discount, target_agent)
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
        if isdone(envi) 
            reset!(envi)
            map(x->x(), dynamic_envis)
            test_envi = deepcopy(envi)
        end
        if it % test_freq == 0
            push!(returns, test_agent(agent, test_envi, 1000))
            verbose && print(Int(round(returns[end])), " ")
        end
		progress_bar && next!(progress, showvalues = [("Last return", isempty(returns) ? "NA" : last(returns))])
    end
    runtime = (Base.time() - starttime)/60
    return returns, runtime
end

function target(reward::T, next_state::T, done::T, discount::Float32, target_agent) where T <: AbstractArray
    next_action = target_agent(next_state)
    Qprime = target_agent.critic(vcat(next_state, next_action)) #.* done
    return reward .+ Qprime .* discount
end

function critic_loss(state::T, action::T, y::T, critic) where T <: AbstractArray
    Q = critic(vcat(state, action))
    return Flux.mse(Q, y)
end

function actor_loss(state::T, agent) where T <: AbstractArray
    a = agent.actor(state)
    return -mean(agent.critic(vcat(state, a)))
end

struct Agent_Bag{A}
	agents::Vector{A}
end

function (ab::Agent_Bag)(state)
	N = length(ab.agents)
	actions = [agent(state)|>cpu for agent in ab.agents]
	actions_b = reduce(hcat, actions)
	SA = vcat(repeat(state, outer = [1,N]), actions_b) |> gpu
	values = reduce(hcat, [agent.critic(SA) |> cpu for agent in ab.agents]) 
	medians = reshape(median(values,dims = 2), :, N)
	return actions[argmax(medians, dims = 1)] |> gpu
end

include("TD3.jl")