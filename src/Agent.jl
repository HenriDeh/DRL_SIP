struct TD3QN_Agent{A,C,F}
    actor::A
    critic::C
    twin_critic::C
    tactor::A
    tcritic::C
    ttwin_critic::C
    explore::F
end

TD3QN_Agent(actor, critic, twin_critic, explore = (a,e) -> a) = TD3QN_Agent(actor, critic, twin_critic, deepcopy(actor), deepcopy(critic), deepcopy(twin_critic), explore)

function (agent::TD3QN_Agent)(state; actor = agent.actor, critic = agent.critic)
    s = state |> gpu
    a = actor(s)
    a0 = zeros(size(a)) |> gpu
    best = critic(vcat(s, a)) .> critic(vcat(s, a0))
    a .* best |> cpu
end

function transition!(agent::TD3QN_Agent, envi)
    state = observe(envi)
    action = agent.explore(agent(state), envi)
    reward = expected_reward(envi, action)
    action!(envi, action)
    next_state = observe(envi)
    Transition(state, action, reward, next_state, isdone(envi) ? 0 : 1)
end

function test_agent(agent::TD3QN_Agent, envi, n = 1000)
    test_reset!(envi)
    envis = [deepcopy(envi) for _ in 1:n]
    cumreward = 0.0
    for t in 1:envi.T
        actions = agent(hcat(observe.(envis)...))
        cumreward += sum([action!(envis[i], actions[:,i]) for i in 1:n])
    end
    return cumreward/n
end
