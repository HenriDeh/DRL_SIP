struct Transition{T<:Real}
    s::Vector{T}
    a::Vector{T}
    r::T
    ns::Vector{T}
    d::T
    Transition(s,a,r,ns,d) = new{typeof(r)}(s,a,r,ns,d)
end

mutable struct ReplayBuffer{T<:Transition}
    capacity::Int
    buffer::Vector{T}
    sample_size::Int
    current::Int
end

ReplayBuffer(capacity, sample_size, type) = ReplayBuffer(capacity, Vector{type}(undef, capacity), sample_size, 1)


function fillbuffer!(rb::ReplayBuffer, agent, envi)
    T = envi.T
    for i in 1:rb.capacity
        addTransition!(rb, transition!(agent, envi))
        isdone(envi) && reset!(envi)
    end
    return nothing
end

function addTransition!(rb::ReplayBuffer{T}, t::T) where T <: Transition
    rb.buffer[rb.current] = t
    rb.current += 1
    if rb.current > rb.capacity
        rb.current = 1
    end
end

function ksample(rb::ReplayBuffer)
    batch = rand(rb.buffer, rb.sample_size)
    l = length(batch)
    t = batch[1]
    T = typeof(t.r)
    s = Array{T}(undef, length(t.s), l)
    a = Array{T}(undef, length(t.a), l)
    r = Array{T}(undef, 1, l)
    ns = similar(s)
    d = Array{T}(undef, length(t.d), l)
    @inbounds for i in 1:l
        s[:,i] = batch[i].s
        a[:,i] = batch[i].a
        r[i] = batch[i].r
        ns[:,i] = batch[i].ns
        d[i] = batch[i].d
    end
    return (s,a,r,ns,d) |> gpu
end
