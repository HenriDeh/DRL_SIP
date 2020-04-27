
#creates a problem instance compatible with the SDP.jl module
function Instance(envi::MultiEchelon)
    @assert length(envi) == 1
    h = envi[1].holding
    p = envi.backorder_cost
    K = envi[1].setup
    c = envi[1].production
    dem = envi.expected_demand
    CV = envi.CV
    return Scarf.Instance(h, p, K, c, CV, dem)
end

#Monte Carlo test of a (S,s) policy. Only works on single-level SIP
function test_policy(envi::MultiEchelon{E}, S, s, n = 10000) where E <: Real
    @assert length(envi) == 1
    totReward = zero(E)
    test_reset!(envi)
    for _ in 1:n
        reward = zero(E)
        for t in 1:envi.T
            y = envi[1].position
            q = zero(E)
            if y < s[t]
                q = S[t] - y
            end
            reward += action!(envi, [q])
        end
        totReward += reward
        test_reset!(envi)
    end
    totReward /= n
    #println("Optimal policy cost: $totReward")
    return totReward
end
