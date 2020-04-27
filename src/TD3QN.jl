using ProgressMeter, LinearAlgebra, Zygote, Flux, Distributions# CuArrays
#CuArrays.allowscalar(false)

include("ReplayBuffer.jl")
include("Agent.jl")

mutable struct Run
    actor
    critic
    envi
    returns
    kwargs
    time
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

function target(reward::T,
    next_state::T,
    done::T,
    agent::TD3QN_Agent) where T <: AbstractArray
    next_action = agent(next_state)
    Qprime = agent.tcritic(vcat(next_state, next_action)::T)
    return reward .+ (Qprime .* done)
end
function twin_target(   reward::T,
                        next_state::T,
                        done::T,
                        agent::TD3QN_Agent) where T <: AbstractArray
    next_action = agent(next_state)
    Qprime1 = agent.tcritic(vcat(next_state, next_action)::T)
    Qprime2 = agent.ttwin_critic(vcat(next_state, next_action)::T)
    return reward .+ (min.(Qprime1, Qprime2) .* done)
end


function critic_loss(   state::T,
                        action::T,
                        y::T,
                        critic) where T <: AbstractArray
    Q = critic(vcat(state, action)::T)
    #δ = Q .- y
    #losses = δ .^2
    return Flux.mse(Q, y)
end

function actor_loss(state::T,
                    agent::TD3QN_Agent) where T <: AbstractArray
    return -mean(agent.critic(vcat(state, agent.actor(state))::T))
end

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

function TD3QN( envi;
                actorlr = 1f-4, criticlr = 1f-3, replaysize = 30000, batchsize = 512, softsync_rate = 0.01, delay::Int = 1, layerwidth = 64,
                maxit = 500000, test_freq::Int = maxit, verbose = false)
    starttime = Base.time()

    opt_actor = Flux.Optimiser(ADAM(actorlr), ClipNorm(one(actorlr)))
    opt_critic = Flux.Optimiser(ADAM(criticlr), ClipNorm(one(actorlr)))


    actor = Chain(  Dense(observation_size(envi), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, action_size(envi), action_squashing_function(envi))
                ) #|> gpu
    tactor = deepcopy(actor)
    critic = Chain( Dense((observation_size(envi) + action_size(envi)), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, 1)
                ) #|> gpu
    tcritic = deepcopy(critic)
    twin_critic = Chain( Dense((observation_size(envi) + action_size(envi)), layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, layerwidth, leakyrelu),
                    Dense(layerwidth, 1)
                ) #|> gpu
    ttwin_critic = deepcopy(twin_critic)
    agent = TD3QN_Agent(actor, critic, twin_critic, tactor, tcritic, ttwin_critic)
    T = Transition{eltype(observe(envi))}
    replaybuffer = ReplayBuffer(replaysize, batchsize, T)
    fillbuffer!(replaybuffer, agent, envi)
    returns = Real[]
    @showprogress for it = 1:maxit
        s,a,r,ns,d = ksample(replaybuffer)
        y = twin_target(r,ns,d, agent)
        b = (s,a,y)
        Flux.train!(
            (d...) -> critic_loss(d..., agent.critic),
            Flux.params(agent.critic),
            [b],
            opt_critic
        )
        s,a,r,ns,d = ksample(replaybuffer)
        y = twin_target(r,ns,d, agent)
        b = (s,a,y)
        Flux.train!(
            (d...) -> critic_loss(d..., agent.twin_critic),
            Flux.params(agent.twin_critic),
            [b],
            opt_critic
        )
        if it % delay == 0
            s,a,r,ns,d = ksample(replaybuffer)
            Flux.train!(
                s -> actor_loss(s, agent),
                Flux.params(agent.actor),
                [s],
                opt_actor
            )
            softsync!(actor, tactor, softsync_rate)
            softsync!(critic, tcritic, softsync_rate)
            softsync!(twin_critic, ttwin_critic, softsync_rate)
        end
        addTransitions!(replaybuffer, [transition!(agent, envi)])
        isdone(envi) && reset!(envi)
        if it % test_freq == 0
            push!(returns, testAgent(envi, agent))
        end
    end
    runtime = (Base.time() - starttime)/60
    kwargs = (actorlr = actorlr, criticlr = criticlr, replaysize = replaysize, batchsize = batchsize, layerwidth = layerwidth, softsync_rate = softsync_rate,
            maxit = maxit, test_freq = test_freq)
    return Run(agent.actor |> cpu, agent.critic |> cpu, envi, returns, kwargs, runtime)
end

#check onedrive 4
