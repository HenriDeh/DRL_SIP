using Flux, CuArrays
CuArrays.allowscalar(false)
struct TD3QN_Agent{A, C}
    actor::A
    target_actor::A
    critic1::C
    target_critic1::C
    critic2::C
    target_critic2::C
end

TD3QN_Agent(actor, critic) = TD3QN_Agent(actor, actor, critic, critic, critic, critic)

explore(a,e) = a #fallback explore function

function (agent::TD3QN_Agent)(state)
    s = cu(state)
    a = agent.actor(s)
    best = agent.critic1(vcat(s, a)) .> agent.critic1(vcat(s, cu(zeros(size(a)))))
    cpu(a .* best)
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
