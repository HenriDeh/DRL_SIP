#Include this file to train an Agent one time on a single predefined MDP. Periodic tests are performed with the same intial state.
using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels
using BSON, StatsPlots
include(srcdir("TD3QN.jl"))

const μ = 10.0
holding = 1
backorder_cost = 10
CV = 0.4
setup = 1280
production = 0
T = 52
H = 70
const EOQ = sqrt(μ * 2 * maximum(setup) / holding)
Dt = MultiDist(Uniform(0, 2 * μ), H)
test_μs = rand(Dt)
on_hand_dist = Uniform(-μ, 2 * μ)
end_prod = Inventory(holding, setup, production, on_hand_dist, on_hand = 0)
envi = MultiEchelon(Dt, backorder_cost, CV, [end_prod], simulation_horizon = H, expected_demand = test_μs)
test_reset!(envi)
instance = Instance(envi)
Scarf.backward_SDP(instance)
opt_policy_value = test_policy(envi, instance.S, instance.s)
dummy_policy_value = test_policy(envi, fill(-Inf, H), fill(-Inf, H))

hp = TD3QN_HP(
    replaysize = 30000,
    batchsize = 128,
    delay = 2,
    optimiser_actor = Flux.Optimiser(ADAM(1f-5), ClipNorm(1)),
    optimiser_critic = Flux.Optimiser(ADAM(1f-3), ClipNorm(1))
)

const epsilon = 0.05
const zero_epsilon = 0.5

explore(action, envi::MultiEchelon) = rand() > epsilon ? action : [rand() > zero_epsilon ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

width = 64
for i in 1:30
    println(i)
    agent = TD3QN_Agent(
                Chain(Dense(observation_size(envi), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, action_size(envi), action_squashing_function(envi))) |> gpu,
                Chain(Dense((observation_size(envi) + action_size(envi)), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                Chain(Dense((observation_size(envi) + action_size(envi)), width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, width, leakyrelu), Dense(width, 1)) |> gpu,
                explore
        )

    returns, time = train!(agent, envi, hp, maxit = 300000, test_freq = 3000)
    println(test_agent(agent, envi, 3000)/opt_policy_value -1)
end
#=
lastrun = runs[2]
#BSON.@load "td3qn.bson" lastrun

agent = TD3QN_Agent(gpu(lastrun.actor), gpu(lastrun.critic))
pol(x) = agent([test_μs...,x])[1]
plot(pol, -50:200, label = "Agent");
plot!(x -> lastrun.actor([test_μs...,x])[1], label= "actor");
plot!(x -> x < instance.s[1] ? instance.S[1] - x : 0, -50:200, label = "Scarf")

z = -30
Q(q) = -lastrun.critic([test_μs...,z, q])[1]
plot(Q , -20:2*EOQ);
scatter!([pol(z)], [Q(pol(z))])

plot(-lastrun.returns)
=##3
