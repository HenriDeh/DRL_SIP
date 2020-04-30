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
