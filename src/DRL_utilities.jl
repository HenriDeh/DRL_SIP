function softsync!(net::C, target_net::C, α) where C <: Chain
    @inbounds for i in eachindex(net.layers)
        net.layers[i] isa Dense || continue
        W = net.layers[i].W
        b = net.layers[i].b
        tW = target_net.layers[i].W
        tb = target_net.layers[i].b
        tW .= ((1 - α) .* tW) .+ (α .* W)
        tb .= ((1 - α) .* tb) .+ (α .* b)
    end
    return nothing
end

mutable struct ClipNorm{T}
    thresh::T
end

function Flux.Optimise.apply!(o::ClipNorm, x, Δ)
    Δnrm = norm(Δ)
    if Δnrm > o.thresh
        rmul!(Δ, o.thresh / Δnrm)
    end
    return Δ
end

function transition!(agent, envi)
    state = observe(envi)
    action = agent.explore(agent(state))
    reward = envi(action)
    next_state = observe(envi)
    Transition(state, action, reward, next_state, isdone(envi) ? 0 : 1)
end

function test_agent(agent, envi, n = 1000)
    test_reset!(envi)
    envis = [deepcopy(envi) for _ in 1:n]
    cumreward = 0.0
    while !isdone(first(envis))
        observations = observe.(envis)
        input = hcat(observations...)
        actions = agent(input)
        cumreward += sum([envis[i](actions[:,i]) for i in 1:n])
    end
    return cumreward/n
end
