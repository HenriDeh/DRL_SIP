struct TD3QN_Agent{A,C}
    actor::A
    critic::C
    twin_critic::C
    tactor::A
    tcritic::C
    ttwin_critic::C
end

explore(a,e) = a #fallback explore function

function (agent::TD3QN_Agent)(state)
    s = state #|> gpu
    a = agent.actor(s)
    a0 = zeros(size(a)) #|> gpu
    best = agent.critic(vcat(s, a)) .> agent.critic(vcat(s, a0))
    a .* best |> cpu
end

function transition!(agent::TD3QN_Agent, envi; state = observe(envi))
    action = explore(agent(state), envi)
    reward = expected_reward(envi, action)
    action!(envi, action)
    next_state = observe(envi)
    Transition(state, action, reward, next_state, isdone(envi) ? 0 : 1)
end

function simulatePolicy(agent::TD3QN_Agent, envi)
    cost = 0
    for t in 1:envi.T
        action = agent(observe(envi))
        cost += action!(envi, action)
    end
    return cost
end

function testAgent(envi, agent::TD3QN_Agent, n = 500)
    test_reset!(envi)
    envis = [deepcopy(envi) for _ in 1:n]
    cumreward = 0.0
    for t in 1:envi.T
        actions = agent(hcat(observe.(envis)...))
        cumreward += sum([action!(envis[i], actions[:,i]) for i in 1:n])
    end
    return cumreward/n
end
