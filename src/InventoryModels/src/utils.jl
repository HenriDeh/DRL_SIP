struct CVNormal{T} <: ContinuousUnivariateDistribution
    normal::Normal{T}
end

function CVNormal{T}(μ, CV) where T
    n=Normal{T}(μ, CV*μ)
    CVNormal{T}(n)
end

function CVNormal(μ, CV)
    n=Normal(μ, CV*μ)
    CVNormal(n)
end

@forward CVNormal.normal Distributions.mean, Base.rand, Distributions.sampler, Distributions.pdf, Distributions.logpdf, Distributions.cdf, Distributions.quantile, Distributions.minimum,
    Distributions.maximum, Distributions.insupport, Distributions.var, Distributions.std, Distributions.modes, Distributions.mode, Distributions.skewness, Distributions.kurtosis,
    Distributions.entropy, Distributions.mgf, Distributions.cf, Base.eltype
Distributions.params(d::CVNormal) = mean(d)

Base.show(n::CVNormal) = print(typeof(n),"(",params(n.normal),")")

struct linear_holding_backorder{T1,T2}
    h::T1
    b::T2
end
(f::linear_holding_backorder)(y, par...) = h*max(zero(y), y) - b*min(zero(y), y)


struct fixed_linear_cost{T1,T2}
    K::T1
    c::T2
end
(f::fixed_linear_cost)(q) = q <= 0 ? zero(q) : f.K + f.c*q

linear_cost(c) = fixed_linear_cost(zero(c), c)

struct expected_hold_stockout_Normal{T1,T2}
    h::T1
    b::T2
end

function (f::expected_hold_stockout_Normal)(y, μ, σ)
    σ == 0 && return linear_holding_backorder(f.h,f.b)(y-μ)
    h = f.h
    b = f.b
    v = σ^2
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
    return fh + fb
end

struct expected_hold_stockout_CVNormal{T1,T2,T3}
    h::T1
    b::T2
    CV::T3
end

function (f::expected_hold_stockout_CVNormal)(y, μ)
    f.CV == 0 && return linear_holding_backorder(f.h,f.b)(y-μ)
    h = f.h
    b = f.b
    v = (f.CV*μ)^2
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
    return fh + fb
end
