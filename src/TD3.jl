@with_kw struct TD3_HP
	replaysize::Int = 2^15
	batchsize::Int = 128
    softsync_rate::Float64 = 0.001
    discount::Float32 = 0.99f0
	optimiser_critic = ADAM(1f-4)
    optimiser_actor = Flux.Optimiser(Flux.ClipNorm(1f-6), ADAM(1.25f-5))
end

function train!(agent, envi, hyperparameters::TD3_HP; maxit::Int = 500000, test_freq::Int = maxit, verbose = false, progress_bar = false, dynamic_envis = [])
    starttime = Base.time()
    @unpack replaysize, batchsize, softsync_rate, discount, optimiser_actor, optimiser_critic, td3 = hyperparameters
    target_agent = deepcopy(agent)
    
    twin = deepcopy(agent.critic)
    target_twin = deepcopy(twin)
    
    test_envi = deepcopy(envi)
    test_reset!(test_envi)
    
    replaybuffer = ReplayBuffer(replaysize, batchsize, Transition{eltype(observe(envi))})
    fillbuffer!(replaybuffer, agent, envi)

    returns = Float64[]
	progress = Progress(maxit)
	for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = target(r,ns,d,discount, target_agent,target_twin)
        b = (s,a,y)
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
                (d...) -> critic_loss(d..., twin),
                Flux.params(twin),
                [b],
                optimiser_critic
            )
            softsync!(twin, target_twin, softsync_rate)
        end
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

function target(reward::T, next_state::T, done::T, discount::Float32, target_agent, twin) where T <: AbstractArray
    next_action = target_agent(next_state)
    sa = vcat(next_state, next_action)
    Qprime = target_agent.critic(sa)
    Qprimetwin = twin(sa)
    return reward .+ min.(Qprime, Qprimetwin) .* discount
end