module PeriodicLinearSplines

import Base: require_one_based_indexing, OneTo, show, axes, getindex, ==
import ContinuumArrays: Basis, grid, grammatrix_layout
import ..ArcPolynomials: map_to_std_range
import ..CyclicBandedMatrices: CyclicBandedMatrix
import DomainSets: ℝ
import QuasiArrays: Inclusion
import LinearAlgebra: Symmetric

export PeriodicLinearSpline

struct PeriodicLinearSpline{T,P<:AbstractVector{T}} <: Basis{T}
    points::P
    function PeriodicLinearSpline(points::P) where {P}
        require_one_based_indexing(points)
        @assert isapprox(points[end] - points[1], 2π) "The points must span 2π"
        _points = map(map_to_std_range, points)
        return new{eltype(P),typeof(_points)}(_points)
    end
end
function show(io::IO, S::PeriodicLinearSpline)
    print(io, "PeriodicLinearSpline(")
    print(IOContext(io, :limit => true), S.points)
    print(io, ")")
end
function axes(S::PeriodicLinearSpline{T}) where {T}
    ax1 = Inclusion(ℝ)
    ax2 = OneTo(length(S.points) - 1)
    return (ax1, ax2)
end
function getindex(S::PeriodicLinearSpline{T}, θ::Number, k::Int) where {T}
    p = S.points
    n = length(p) - 1
    @boundscheck (θ ∈ axes(S, 1) && 1 ≤ k ≤ n)
    θ = map_to_std_range(θ)
    θ == p[k] && return one(T)
    if k == 1 && θ ≤ p[2]
        return (θ - p[k+1]) / (p[k] - p[k+1])
    elseif k == 1 && θ > p[n]
        return (θ - p[n]) / (p[n+1] - p[n])
    else
        ((k > 1 && θ ≤ p[k-1]) || (k < n && θ ≥ p[k+1])) && return zero(T)
        if θ < p[k]
            return (θ - p[k-1]) / (p[k] - p[k-1])
        else
            return (θ - p[k+1]) / (p[k] - p[k+1])
        end
    end
end

grid(S::PeriodicLinearSpline, ::Integer) = S.points
==(S::PeriodicLinearSpline, T::PeriodicLinearSpline) = S.points == T.points

function _grammatrix_diag(S::PeriodicLinearSpline, i)
    p = S.points
    n = length(p) - 1
    Δθᵢ₋₁ = p[i] - p[mod1(i - 1, n)]
    i == 1 && (Δθᵢ₋₁ += 2π)
    Δθᵢ = p[i+1] - p[i]
    return Δθᵢ₋₁ / 3 + Δθᵢ / 3
end
function _grammatrix_offdiag(S::PeriodicLinearSpline, i)
    p = S.points
    Δθᵢ = p[i+1] - p[i]
    return Δθᵢ / 6
end
function grammatrix_layout(_, S::PeriodicLinearSpline{T}) where {T}
    p = S.points
    n = length(p) - 1
    B = CyclicBandedMatrix{T}(undef, (n, n), (1, 1))
    for i in 1:n
        B[i, i] = _grammatrix_diag(S, i)
        B[i, i+1] = _grammatrix_offdiag(S, i)
    end
    if n == 2
        B[1, 2] *= 2
    end
    return Symmetric(B)
end

end # module PeriodicLinearSplines