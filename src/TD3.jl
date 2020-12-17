struct TD3_Agent{A, C, F}
	actor::A
    critic::C
    twin::C
	explorer::F
end

function TD3_Agent(actor, critic, explorer)
    TD3_Agent(actor, critic, deepcopy(critic), explorer)
end

function (agent::TD3_Agent)(state)
	s = state |> gpu
	agent.actor(s)
end

Flux.gpu(a::TD3_Agent) = TD3_Agent(a.actor |> gpu, a.critic |> gpu, a.twin |> gpu, a.explorer) 
Flux.cpu(a::TD3_Agent) = TD3_Agent(a.actor |> cpu, a.critic |> cpu, a.twin |> cpu, a.explorer)

struct TD3QN_Agent{A, C, F}
	actor::A
    critic::C
    twin::C
	explorer::F
end

function TD3QN_Agent(actor, critic, explorer)
    TD3QN_Agent(actor, critic, deepcopy(critic), explorer)
end

function (agent::TD3QN_Agent)(state)
	actor = agent.actor
    critic = agent.critic
    twin = agent.twin
    s = state |> gpu
    a = actor(s)
    a0 = zeros(size(a)) |> gpu
    sa = vcat(s, a)
    sa0 = vcat(s, a0)
    best = min.(critic(sa), twin(sa)) .> min.(critic(sa0), twin(sa0))
    a .* best
end

Flux.gpu(a::TD3QN_Agent) = TD3QN_Agent(a.actor |> gpu, a.critic |> gpu, a.twin |> gpu, a.explorer) 
Flux.cpu(a::TD3QN_Agent) = TD3QN_Agent(a.actor |> cpu, a.critic |> cpu, a.twin |> cpu, a.explorer)


function Flux.trainmode!(agent::Union{TD3_Agent, TD3QN_Agent})
    trainmode!(agent.actor)
    trainmode!(agent.critic)
    trainmode!(agent.twin) 
end

function Flux.testmode!(agent::Union{TD3_Agent, TD3QN_Agent})
    testmode!(agent.actor)
    testmode!(agent.critic)
    testmode!(agent.twin)
end

function train!(agent::TD3QN_Agent, envi, hyperparameters::DDPG_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false, progress_bar = false, dynamic_envis = [])
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, discount, optimiser_actor, optimiser_critic = hyperparameters
    target_agent = deepcopy(agent)
    
    test_envi = deepcopy(envi)
    test_reset!(test_envi)

    testmode!(agent)
    testmode!(target_agent)
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)

    returns = Float64[]
	progress = Progress(maxit)
	for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = target_twin(r,ns,d,discount, target_agent)
        b = (s,a,y)
        trainmode!(agent)
        if it%2 == 1
            Flux.train!(
                (d...) -> critic_loss(d..., agent.critic),
                Flux.params(agent.critic),
                [b],
                optimiser_critic
            )
            softsync!(agent.critic, target_agent.critic, softsync_rate)
        else
            Flux.train!(
                (d...) -> critic_loss(d..., agent.twin),
                Flux.params(agent.twin),
                [b],
                optimiser_critic
            )
            softsync!(agent.twin, target_agent.twin, softsync_rate)
        end
        testmode!(agent.critic)
        testmode!(agent.twin)
        s,a,r,ns,d = ksample(replaybuffer)
        Flux.train!(
            s -> actor_loss(s, agent),
            Flux.params(agent.actor),
            [s],
            optimiser_actor
        )
        softsync!(agent.actor, target_agent.actor, softsync_rate)
        testmode!(agent)
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

function target_twin(reward::T, next_state::T, done::T, discount::Float32, target_agent) where T <: AbstractArray
    next_action = target_agent(next_state)
    sa = vcat(next_state, next_action)
    Qprime = target_agent.critic(sa)
    Qprimetwin = target_agent.twin(sa)
    return reward .+ min.(Qprime, Qprimetwin) .* discount
end