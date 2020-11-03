function softsync!(net::C, target_net::C, β) where C <: Chain
    @inbounds for i in eachindex(net.layers)
        net.layers[i] isa Dense || continue
        W = net.layers[i].W
        b = net.layers[i].b
        tW = target_net.layers[i].W
        tb = target_net.layers[i].b
        tW .= ((1 - β) .* tW) .+ (β .* W)
        tb .= ((1 - β) .* tb) .+ (β .* b)
    end
    return nothing
end

function transition!(agent, envi)
    state = observe(envi)
    action = agent.explorer(agent(state) |> cpu)
    reward = envi(action)
    next_state = observe(envi)
    Transition(state, action, reward, next_state, isdone(envi) ? 0 : 1)
end

function transition_exp!(agent, envi)
    state = observe(envi)
    action = agent.explorer.explore(agent(state) |> cpu)
    reward = envi(action)
    next_state = observe(envi)
    Transition(state, action, reward, next_state, isdone(envi) ? 0 : 1)
end

function test_agent(agent, envi, n = 10000)
    test_reset!(envi)
    envis = [deepcopy(envi) for _ in 1:n]
    cumreward = 0.0
    while !isdone(first(envis))
        observations = observe.(envis)
        input = reduce(hcat, observations)
        actions = agent(input) |> cpu
        cumreward += sum([envis[i](actions[:,i]) for i in 1:n])
    end
    return cumreward/n
end

struct EpsilonGreedy{T,F}
	explore::F
	ϵ::T
end

(eg::EpsilonGreedy)(action) = rand() > eg.ϵ ? action : eg.explore(action)
