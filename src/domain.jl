function map_to_std_range(θ::T) where {T}
    y = rem2pi(θ, RoundNearest)
    return y
end

"""
    Arc(h::T) <: Domain{SVector{2, T}}

Represents the part of an arc on a unit circle where `x ≥ h`.
"""
struct Arc{T} <: Domain{T}
    h::T
end
Arc{T}() where {T} = Arc{T}(zero(T))
Arc() = Arc{Float64}()
Arc(a::Arc) = a
Arc{T}(a::Arc) where {T} = Arc{T}(a.h)
show(io::IO, a::Arc) = summary(io, a)
summary(io::IO, a::Arc{T}) where {T} = print(io, "Arc{$T}($(a.h))")
==(a::Arc, b::Arc) = a.h == b.h
function in(θ, d::Arc)
    x = cos(θ)
    return x ≥ d.h
end

function checkpoints(a::Arc{T}) where {T}
    h = a.h
    _, x1, x2, _ = LinRange(h, one(h), 4)
    θ1, θ2 = acos(x1), acos(x2)
    return [-θ1, -θ2, θ2, θ1]
end

cardinality(::Arc) = ℵ₁

# Domain definition 
const ArcInclusion{T} = Inclusion{T,Arc{T}}
ArcInclusion{T}() where {T} = ArcInclusion{T}(Arc{T}())
ArcInclusion{T}(a::Number) where {T} = ArcInclusion{T}(Arc{T}(a))
ArcInclusion(a::T) where {T} = ArcInclusion{T}(a)