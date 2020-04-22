#Include this file to train an Agent one time on a single predefined MDP. Periodic tests are performed with the same intial state.
println("Check OneDrive 2")
include("TD3QN.jl")
include("../MultiEchelonModels.jl")
include("../policy_tests.jl")
using .MultiEchelonModels
using BSON, StatsPlots

#test_instance_parameters
const μ = 10.0
holding = 1
backorder_cost = 5
CV = 0.2
setup = 1280
production = 0
T = 52
H = 70
const EOQ = sqrt(μ*2*maximum(setup)/holding)
test_μs = [(1+0.25sin(4t*pi/T))*μ for t = 1:H]

#Inventory
on_hand_dist = Uniform(-μ, 2*μ)
end_prod = Inventory(holding, setup, production, on_hand_dist, on_hand = 0)
#MultiEchelon
Dt = MultiDist(Uniform(0, 2*μ), H)
envi = MultiEchelon(Dt, backorder_cost, CV, [end_prod], simulation_horizon = H, expected_demand = test_μs)
test_reset!(envi)
instance = Instance(envi)
#instance.H = T
SDP.backward_SDP(instance)
opt = test_policy(envi, instance.S, instance.s)
println("opt = ", opt)

hp = (  actorlr = 1f-5, criticlr = 1f-4, replaysize = 30000, batchsize = 128, layerwidth = 64,
        softsync_rate = 0.01, maxit = 30000, pretrain_critic = false, delay = 4)

ft = 0.95*test_policy(envi, fill(-Inf, H), fill(-Inf,H))
explore(action, envi::MultiEchelon) = rand() > 0.05 ? action : [rand() > 0.5 ? rand(Uniform(0, EOQ*2)) : zero(eltype(action)) for _ in eachindex(action)]

runs = Run[]
for i in 1:30
    println(i)
    lastrun = TD3QN(envi; hp..., test_freq = div(hp.maxit,10), verbose = false, max_fails = 0 )
    push!(runs, lastrun)
    perf = testAgent(envi, TD3QN_Agent(gpu(lastrun.actor), gpu(lastrun.critic)), 1000)
    println("perf = ", perf)
    println("gap =", perf/opt - 1)
    fail = false
    for r in lastrun.returns if r < ft fail =true end end
    println("Would've failed = ", fail)
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
=#
