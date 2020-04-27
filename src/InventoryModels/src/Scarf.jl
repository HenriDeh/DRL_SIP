module Scarf

using Distributions, SpecialFunctions
export Instance, backward_SDP
mutable struct Instance{T <: Real}
    holding_cost::T
    backorder_cost::T
    setup_cost::T
    production_cost::T
    demand_forecasts::Array{Normal{T},1}
    H::Int
    s::Array{T,1}
    S::Array{T,1}
end

function Instance(h,b,K,c,CV,demands)
    T = typeof(CV)
    dists = Normal{T}[]
    for d in demands
        push!(dists, Normal(d, CV*d))
    end
    Instance{T}(h,b,K,c,dists,length(dists),Array{T,1}(undef,length(dists)),Array{T,1}(undef,length(dists)))
end

mutable struct Pwla{T}
    breakpoints::Array{T,1}
    range::StepRangeLen{T,T,T}
end

function Pwla(stepsize)
    T = typeof(stepsize)
    Pwla{T}(Array{T,1}(), stepsize:stepsize:stepsize)
end

function (pwla::Pwla)(y)
    if length(pwla.breakpoints) > 0
        return (@view pwla.breakpoints[end:-1:1])[Int(round((y - pwla.range[1])/pwla.range.step)+1)]
    else
        return zero(y)
    end
end

function production_cost(instance::Instance, q)
    if q > 0
        return instance.production_cost*q + instance.setup_cost
    else
        return zero(q)
    end
end

function L(instance::Instance, y, t::Int)
    h = instance.holding_cost
    b = instance.backorder_cost
    μ = mean(instance.demand_forecasts[t])
    v = var(instance.demand_forecasts[t])
    e = MathConstants.e
    π = MathConstants.pi
    fh = zero(y)
    fb = zero(y)
    if y >= 0
        fh = (((e^(-((y - μ)^2)/2v) -e^(-(μ^2)/2v))*sqrt(v))/(sqrt(2π))) + 1//2*(y-μ)*(erf((y-μ)/(sqrt(2)*sqrt(v))) + erf(μ/(sqrt(2)*sqrt(v))))
        fh *= h
        fb = ((e^(-((y - μ)^2)/2v))*v + sqrt(π/2) * (μ - y) * sqrt(v) * erfc((y-μ)/(sqrt(2)*sqrt(v))))/(sqrt(2π)*sqrt(v))
        fb *= b
    else
        fb = (e^(-(μ^2)/2v)*v + sqrt(π/2) * (μ - y) * sqrt(v) * (1 + erf(μ/(sqrt(2)*sqrt(v)))))/(sqrt(2π)*sqrt(v))
        fb *= b
    end
    return fb + fh
end

function G(instance::Instance, y, t::Int)
    instance.production_cost*y + L(instance, y, t)
end

function C(instance::Instance, x, t::Int, pwla::Pwla)
    S = instance.S[t]
    s = instance.s[t]
    q = x <= s ? S - x : 0.0
    return production_cost(instance, q) + L(instance, x + q, t) + pwla(x + q)
end

function expected_future_cost(instance::Instance, y, t::Int, pwla::Pwla)
    if t >= instance.H
        return zero(y)
    else
        df = instance.demand_forecasts[t]
        ub = quantile(df, 0.99999)
        ξ = pwla.range.step:pwla.range.step:ub
        x = y .- ξ
        p = cdf.(df, ξ .+ ξ.step.hi/2) .- cdf.(df, ξ .- ξ.step.hi/2)#::Array{Float64,1}
        c(x) = C(instance, x, t+1, pwla)
        return sum(c.(x) .* p)
    end
end

function backward_SDP(instance::Instance{T}, stepsize::T = one(T)) where T <: Real
    meandemand = max(stepsize, mean(mean.(instance.demand_forecasts)))
    critical_ratio = 1# (instance.backorder_cost-instance.holding_cost)/instance.backorder_cost
    EOQ = sqrt(2*meandemand*instance.setup_cost/(critical_ratio*instance.holding_cost))
    upperbound = 2*EOQ
    H = instance.H
    pwla = Pwla(stepsize)
    for t in H:-1:1
        newpwla = Pwla(stepsize)
        descending = true
        y = upperbound
        EFC = expected_future_cost(instance, y, t, pwla)
        g = G(instance, y, t) + EFC
        push!(newpwla.breakpoints, EFC)
        instance.S[t] = y
        ming = g
        while g <= ming + instance.setup_cost || descending
            y -= stepsize
            EFC = expected_future_cost(instance, y, t, pwla)
            push!(newpwla.breakpoints, EFC)
            g = G(instance, y, t) + EFC
            if g < ming
                ming = g
                instance.S[t] = y
            elseif descending
                descending = false
            end
        end
        instance.s[t] = y
        newpwla.range = y:stepsize:upperbound
        pwla = newpwla
    end
end

end #module
