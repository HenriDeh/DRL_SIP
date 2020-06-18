#script used to generate instances.csv
using Distributions, CSV, DataFrames
test_forecasts = []
T = 52
H = 52
μ = 10
"""Uncorrelated"""

for i in 1:50
    push!(test_forecasts, rand(Uniform(0,2μ),T))
end

"""Seasonal"""

ρs = [0.1, 0.3]
fs = [1, 2]
for ρ in ρs
    for f in fs
        for i in 1:50
            spread = ρ*sqrt(12)*μ/2
            μt = [(1 + 0.5*sin(2f*t*π/H))*max(0, rand(Uniform(μ-spread, μ + spread))) for t in 1:T]
            push!(test_forecasts, μt)
        end
    end
end

"""Trend"""

ρ = [0.1, 0.3]
for ρ in ρs
    for i in 1:50
        spread = ρ*sqrt(12)*μ/4
        μt = [(1+2t/H)*max(0, rand(Uniform(μ/2-spread, μ/2 + spread))) for t in 1:T]
        push!(test_forecasts, μt)
        μt = [(3-2t/H)*max(0, rand(Uniform(μ/2-spread, μ/2 + spread))) for t in 1:T]
        push!(test_forecasts, μt)
    end
end

CSV.write("instances.csv", DataFrame(hcat(test_forecasts...)'), writeheader = false)
