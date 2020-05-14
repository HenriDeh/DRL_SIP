#Include this file to train an Agent one time on a single predefined MDP. Periodic tests are performed with the same intial state.
using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels
#using BSON, Plots
include(srcdir("TD3QN.jl"))

const μ = 10.0
holding = 1
backorder = 10
CV = 0.4
setup = 1280
production = 0
T = 52
H = 70
const EOQ = sqrt(μ * 2 * maximum(setup) / holding)

reseter = repeat([(Uniform(0,2μ), Normal(CV,0))], H)
test_forecasts = CVNormal.([rand(Uniform(0, 2μ)) for _ in 1:H], CV)

sup = Supplier(fixed_linear_cost(setup,production))
pro = ProductInventory(linear_cost(holding), sup, Uniform(-μ, 2μ), test_reset_level = 0.0)
ma = Market(expected_hold_stockout_CVNormal(holding,backorder, CV), pro, CVNormal, true, reseter, test_reset_forecasts = test_forecasts, expected_reward = true, visibility = T)
envi = InventoryProblem([sup, pro, ma])

test_reset!(envi)
instance = Instance(envi)
Scarf.backward_SDP(instance)
opt_policy_value = test_Scarf_policy(envi, instance.S, instance.s)
dummy_policy_value = test_Scarf_policy(envi, fill(-Inf, H), fill(-Inf, H))

hp = TD3QN_HP(
    replaysize = 30000,
    batchsize = 256,
    delay = 4,
    optimiser_actor = Flux.Optimiser(ADAM(1f-5), ClipNorm(1)),
    optimiser_critic = Flux.Optimiser(ADAM(1f-3), ClipNorm(1))
)

const epsilon = 0.05
const zero_epsilon = 0.5

explore(action) = rand() > epsilon ? action : [rand() > zero_epsilon ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

width = 64

agents = []
for i in 1:10
    println(i)
    agent = TD3QN_Agent(
                Chain(  Dense(observation_size(envi), width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, action_size(envi), action_squashing_function(envi))) |> gpu,
                Chain(  Dense((observation_size(envi) + action_size(envi)), width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, 1)) |> gpu,
                Chain(  Dense((observation_size(envi) + action_size(envi)), width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, width, leakyrelu),
                        Dense(width, 1)) |> gpu,
                explore,
                target_actor = i < 5
        )

    returns, time = train!(agent, envi, hp, maxit = 200000, test_freq = 2000)
    println(test_agent(agent, envi, 3000)/opt_policy_value -1)
    push!(agents, agent)
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
