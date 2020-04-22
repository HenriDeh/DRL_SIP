module MultiEchelonModels
using SpecialFunctions, Reexport
import MacroTools.@forward, Flux.relu
export MultiDist, Inventory, MultiEchelon, observation_size, observe, reset!, action_size, action!, expected_reward, test_reset!, isdone, action_noised, action_squashing_function
@reexport using Distributions
"""
A custom distributions useful for the initial states. It is simply a wrapper into a single Multivariate Distribution to be used instead of calling rand(<:Distribution, ::Int).
"""

struct MultiDist{S <: ValueSupport, D <: UnivariateDistribution{S}} <: MultivariateDistribution{S}
    dist::D
    n::Int
end

Base.rand(x::MultiDist) = rand(x.dist, x.n)
Base.rand(x::MultiDist, n::Int) = rand(x.dist, x.n, n)
Base.length(x::MultiDist) = x.n
Distributions.support(m::MultiDist) = support(m.dist)

"""
# The environment struct is coded to be able to simulate a multi-level system with lead times. The single-level problem is a special case.
"""

mutable struct Inventory{T <: Real}
    holding::T
    setup::T
    production::T
    on_hand::T
    on_order::Vector{T}
    position::T
    lead_time::Int
    components::Vector{Inventory{T}}
    BOM_index::Int
    initial_state_distributions::Dict{Symbol, Distribution}
    state_fields::Vector{Symbol}
    test_kwargs
end

function Inventory( holding,
                    setup,
                    production,
                    on_hand,
                    on_order = Real[],
                    components::Vector{Inventory} = Inventory[];
                    test_kwargs...)
    isd = Dict{Symbol, Union{Distribution, Array{Distribution}}}()
    sf = [:on_hand]
    if !(holding isa Real); isd[:holding] = holding; push!(sf, :holding); holding = rand(holding) end
    if !(setup isa Real); isd[:setup] = setup; push!(sf, :setup); setup = rand(setup) end
    if !(production isa Real); isd[:production] = production; push!(sf, :production); production = rand(production) end
    if on_hand isa Real
        isd[:on_hand] = Normal(on_hand, 0)
    else
        isd[:on_hand] = on_hand; on_hand = rand(on_hand)
    end
    if !(on_order isa Array{<:Real})
        @assert minimum(support(on_order)) >= 0 "on_order must have positive support"
        isd[:on_order] = on_order
        on_order = rand(on_order)
        push!(sf, :on_order)
    elseif !isempty(on_order)
        @assert  minimum(on_order) >= 0 "on_order must be positive"
        push!(sf, :on_order)
    end
    lead_time = length(on_order)
    position = sum(on_order) + on_hand
    T = eltype([holding, setup, production, on_hand, on_order...])
    Inventory{T}(holding, setup, production, on_hand, on_order, position, lead_time, components, 0, isd, sf, test_kwargs)
end

function observation_size(inv::Inventory)
    sum(Int[length(getfield(inv, f)) for f in inv.state_fields])
end

function observe(inv::Inventory{T}) where T<:Real
    state = T[]
    for f in inv.state_fields
        push!(state, getproperty(inv, f)...)
    end
    return state
end

function subDemand!(inv::Inventory, d)
    inv.on_hand -= d
    inv.position -= d
    return nothing
end

function addOrder!(inv::Inventory, o)
    push!(inv.on_order, o)
    inv.position += o
    return nothing
end

function addProduction!(inv::Inventory, q)
    inv.on_hand += q
    inv.position += q
    return nothing
end

#Resets according to initial_state_distributions, then with the values specified as keyword arguments
function reset!(inv::Inventory{T}; initvalues...) where T<:Real
    for f in keys(inv.initial_state_distributions)
        setfield!(inv, f, rand(inv.initial_state_distributions[f]))
    end
    for key in keys(initvalues)
        setfield!(inv, key, fieldtype(typeof(inv), key)(initvalues[key]))
    end
    @assert inv.lead_time == length(inv.on_order)
    inv.position = sum(inv.on_order) + inv.on_hand
end

function test_reset!(inv::Inventory)
    reset!(inv; inv.test_kwargs...)
end

"""
############# MultiEchelon ################
"""

"""
MultiEchelon is the master structure. It is the one that interacts with the agent, that is charged to return the states, compute rewards and simulate the transitions.
It simulates the Markov Decision Process, the BOM is the production tree made of `Inventory`'s, each of which represent one production operation.

"""
mutable struct MultiEchelon{E <: Real}
    expected_demand::Vector{E}
    demand_dist::Vector{Normal{E}}
    backorder_cost::E
    CV::E
    BOM::Array{Inventory{E}, 1} #first is end-product
    t::Int #current time index
    T::Int #max time index
    M::Int #number of inventories
    V::Int #forecast visibility horizon (number of forecast periods returned in observations)
    initial_state_distributions::Dict{Symbol, Distribution}
    state_fields::Vector{Symbol}
    test_kwargs
end

@forward MultiEchelon.BOM Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

"""
Creates an Environement instance.
test_inventory_kwargs must be pairs mapping inventory IDs to NamedTuple.
test_environment_kwargs must be kw arguments (a NamedTuple)
example:
MultiEchelon(MultiDist(Uniform(0,100), 4), 10, 0.4, [an_inventory_instance, a_second_inventory],
            1 => (on_hand = 0,), 2 => (on_hand = 10, on_order = [0]),
            expected_demand = [20,40,60,40], simulation_horizon = 3)
these kwargs are used as a fixed reset state with test_reset!
"""

function MultiEchelon(  expected_demand,
                        backorder_cost,
                        CV,
                        BOM::Vector{Inventory{E}};
                        simulation_horizon::Int = length(expected_demand),
                        visibility_horizon::Int = simulation_horizon,
                        test_kwargs...) where E <: Real
    #Should add a check that the BOM is a tree
    isd = Dict{Symbol, Distribution}()
    sf = [:expected_demand]
    if !(expected_demand isa Vector{<:Real}); isd[:expected_demand] = expected_demand; expected_demand = rand(expected_demand) end
    if !(backorder_cost isa Real); isd[:backorder_cost] = backorder_cost; push!(sf, :backorder_cost); backorder_cost = rand(backorder_cost) end
    if !(CV isa Real); isd[:CV] = CV; push!(sf, :CV); CV = rand(CV) end
    T = simulation_horizon
    M = length(BOM)
    forecasts = Normal.(expected_demand, expected_demand .* CV)
    for i in eachindex(BOM)
        BOM[i].BOM_index = i
    end
    MultiEchelon{E}(expected_demand, forecasts, backorder_cost, CV, BOM, 1, T, M, visibility_horizon, isd, sf, test_kwargs)
end

observation_size(envi::MultiEchelon) = sum(observation_size.(envi.BOM)) + sum(Int[length(getfield(envi, f)) for f in envi.state_fields]) - length(envi.expected_demand) + envi.V

action_size(envi::MultiEchelon) = envi.M

#Return the current state vector s_t
function observe(envi::MultiEchelon{E}) where E<:Real
    state = zeros(E, envi.V)
    for i in eachindex(envi.t:envi.V)
        state[i] = envi.expected_demand[envi.t+i-1]
    end
    for f in envi.state_fields
        f == :expected_demand && continue
        push!(state, getproperty(envi, f)...)
    end
    return vcat(state, observe.(envi.BOM)...)
end

isdone(envi::MultiEchelon) = envi.t > envi.T

"""
Resets an environment according to its inital state distribution. Keyword arguments can provide fixed values for environment fields and for inventories.
Use the keyword inventory_values and a dict mapping BOM indices to keyword arguments for respective inventories
example:
reset!(envi, 1 => (setup = 80,), 3 => (on_hand = 0, holding = 1), expected_demand = [20,30,40,60])
will reset the forecast and the first, third inventories in the BOM with the specified values.
"""
#TODO: add a check that struct values are in their valid space
function reset!(envi::MultiEchelon{E}, inventory_kwarg_pairs...; initvalues...) where E <: Real
    for f in keys(envi.initial_state_distributions)
        setfield!(envi, f, rand(envi.initial_state_distributions[f]))
    end
    for f in keys(initvalues)
        setfield!(envi, f, fieldtype(typeof(envi), f)(initvalues[f]))
    end

    inventory_kwarg_pairs = Dict(inventory_kwarg_pairs)
    for i in 1:envi.M
        in(i, keys(inventory_kwarg_pairs)) ? reset!(envi.BOM[i]; inventory_kwarg_pairs[i]...) : reset!(envi.BOM[i])
    end
    envi.demand_dist .= Normal.(envi.expected_demand , envi.expected_demand .* envi.CV)
    envi.t = 1
    return nothing
end

function test_reset!(envi::MultiEchelon)
    for f in keys(envi.initial_state_distributions)
        setfield!(envi, f, rand(envi.initial_state_distributions[f]))
    end
    for f in keys(envi.test_kwargs)
        setfield!(envi, f, fieldtype(typeof(envi), f)(envi.test_kwargs[f]))
    end
    for inv in envi
        test_reset!(inv)
    end
    envi.demand_dist .= Normal.(envi.expected_demand , envi.expected_demand .* envi.CV)
    envi.t = 1
    return nothing
end
"""
#Carry out a transition with a_t = productionsQ. Return r_t, the immediate inventory and production costs that occured.
Order of events:
for each Inventory i, from lowest to end product in BOM:
    1) Check if enough components are available on_hand. If not, clip productionQ[i] to feasible quantity.
    2) Reduce on_hand of components by production quantity
    3) Add ProductionQ[i] to order queue
    4) First Order in queue is moved to on_hand
5) realize demand in end_product inventory
6) compute reward
"""

function action!(envi::MultiEchelon{E}, productionsQ::AbstractArray{<: Real}) where E <: Real
    productionsQ = Vector{E}(vec(productionsQ))
    M = envi.M
    @assert isdone(envi) == 0 "MultiEchelon is at terminal state"
    @assert length(productionsQ) == M "Action vector size incorrect"
    @assert productionsQ >= zeros(length(productionsQ)) "Cannot produce negative quantities"
    productionsQ = Vector{E}(productionsQ)
    total_prod_cost = zero(E)
    for (i,inv) in Iterators.reverse(enumerate(envi.BOM))
        if !isempty(inv.components)
            #1)
            productionsQ[i] = max(min(productionsQ[i], minimum([comp.on_hand for comp in inv.components])), zero(E))
            #2)
            subDemand!.(inv.components, productionsQ[i])
        end
        #3)
        addOrder!(inv, productionsQ[i])
        #4)
        inv.on_hand += popfirst!(inv.on_order)
        total_prod_cost += productionsQ[i] == 0 ? zero(E) : inv.setup + productionsQ[i] * inv.production
    end
    #5)
    dem = max(0, rand(envi.demand_dist[envi.t]))
    subDemand!(envi[1], dem)
    #6)
    holding_cost = sum([inv.holding for inv in envi] .* max.(0, [inv.position for inv in envi]))
    backorder_cost = - min(0, envi[1].on_hand) * envi.backorder_cost
    envi.t += 1
    return -(total_prod_cost  +  holding_cost + backorder_cost)
end

"""
#this function computes the expected reward without changing the state of the MultiEchelon
Order of events:
for each Inventory i, from lowest to end product in BOM:
    1) Check if enough components are available on_hand. If not, clip productionQ[i] to feasible quantity.
    2) Reduce on_hand of components by production quantity
    3) Add ProductionQ[i] to order queue
    4) First Order in queue is moved to on_hand
5) realize demand in end_product inventory
6) compute reward
"""
function expected_reward(envi::MultiEchelon{E}, productionsQ::AbstractArray{<: Real}) where E <: Real
    productionsQ = Vector{E}(vec(productionsQ))
    @assert isdone(envi) == 0 "MultiEchelon is at terminal state"
    @assert length(productionsQ) == envi.M "Action vector size incorrect"
    @assert productionsQ >= zeros(length(productionsQ)) "Cannot produce negative quantities"
    total_prod_cost = zero(E)
    on_hands = [inv.on_hand for inv in envi]
    positions = [inv.position for inv in envi]
    for (i,inv) in Iterators.reverse(enumerate(envi.BOM))
        if !isempty(inv.components)
            #1)
            components_idx = [comp.BOM_index for comp in inv.components]
            productionsQ[i] = max(min(productionsQ[i], minimum(@view on_hands[components_idx])), zero(E))
            #2)
            (@view on_hands[components_idx]) .-= productionsQ[i]
            (@view positions[components_idx]) .-= productionsQ[i]
        end
        #3)
        on_hands[i] += inv.lead_time == 0 ? productionsQ[i] : first(inv.on_order)
        positions[i] += productionsQ[i]
        #4)
        total_prod_cost += productionsQ[i] == 0 ? zero(E) : inv.setup + productionsQ[i] * inv.production
    end
    holding_costs = [inv.holding for inv in envi]
    holding_costs_comp = sum(holding_costs[2:end] .* positions[2:end])

    y = positions[1]
    h = envi[1].holding
    b = envi.backorder_cost
    μ = mean(envi.demand_dist[envi.t])
    v = var(envi.demand_dist[envi.t])
    e = MathConstants.e
    π = MathConstants.pi
    fh = zero(E)
    fb = zero(E)
    if y >= 0
        fh = (((e^(-((y - μ)^2)/2v) -e^(-(μ^2)/2v))*sqrt(v))/(sqrt(2π))) + 1//2*(y-μ)*(erf((y-μ)/(sqrt(2)*sqrt(v))) + erf(μ/(sqrt(2)*sqrt(v))))
        fh *= h
        fb = ((e^(-((y - μ)^2)/2v))*v + sqrt(π/2) * (μ - y) * sqrt(v) * erfc((y-μ)/(sqrt(2)*sqrt(v))))/(sqrt(2π)*sqrt(v))
        fb *= b
    else
        fb = (e^(-(μ^2)/2v)*v + sqrt(π/2) * (μ - y) * sqrt(v) * (1 + erf(μ/(sqrt(2)*sqrt(v)))))/(sqrt(2π)*sqrt(v))
        fb *= b
    end
    return -(total_prod_cost + holding_costs_comp + fh + fb)
end
action_squashing_function(envi::MultiEchelon) = relu

end
