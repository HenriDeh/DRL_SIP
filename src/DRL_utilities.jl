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
        input = hcat(observations...)
        actions = agent(input) |> cpu
        cumreward += sum([envis[i](actions[:,i]) for i in 1:n])
    end
    return cumreward/n
end

function target(reward::T, next_state::T, done::T, target_agent) where T <: AbstractArray
    next_action = target_agent(next_state)
    Qprime = target_agent.critic(vcat(next_state, next_action)) .* done
    return reward .+ Qprime
end

function critic_loss(state::T, action::T, y::T, critic) where T <: AbstractArray
    Q = critic(vcat(state, action))
    return Flux.mse(Q, y)
end

function actor_loss(state::T, agent) where T <: AbstractArray
    return -mean(agent.critic(vcat(state, agent.actor(state))))
end
