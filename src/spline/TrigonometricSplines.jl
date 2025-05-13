module TrigonometricSplines

import Base: require_one_based_indexing, OneTo, axes, show, getindex, ==
import ContinuumArrays: Basis, grammatrix_layout, grid
import ..ArcPolynomials: ArcInclusion, map_to_std_range
import ..CyclicBandedMatrices: CyclicBandedMatrix
import StaticArrays: StaticVector
import LinearAlgebra: Symmetric

export TrigonometricSpline

struct TrigonometricSpline{T,P<:AbstractVector{T}} <: Basis{T} # Based on ContinuumArrays.jl's LinearSpline
    points::P
    function TrigonometricSpline(points::P) where {P}
        require_one_based_indexing(points)
        @assert isapprox(points[end] - points[1], 2π) "The points must span 2π"
        _points = map(map_to_std_range, points)
        return new{eltype(P),typeof(_points)}(_points)
    end
end
function show(io::IO, S::TrigonometricSpline)
    print(io, "TrigonometricSpline(")
    print(IOContext(io, :limit => true), S.points)
    print(io, ")")
end
function axes(S::TrigonometricSpline{T}) where {T}
    ax1 = ArcInclusion{T}(-1)
    ax2 = OneTo(length(S.points) - 1)
    return (ax1, ax2)
end
function getindex(S::TrigonometricSpline{T}, θ::Number, k::Int) where {T}
    p = S.points
    n = length(p) - 1
    @boundscheck ((θ ∈ axes(S, 1) && 1 ≤ k ≤ n)) || throw(BoundsError(S, (θ, k)))
    θ == p[k] && return one(T)
    θ == T(π) && k == 1 && return one(T)
    θ == T(π) && k == n && return zero(T)
    if k == 1 && θ ≤ p[2]
        a, b, u, v = p[1], p[2], 1, -1 
    elseif k == 1 && θ > p[n]
        a, b, u, v = p[n], p[1], -1, 1
    else
        ((k > 1 && θ ≤ p[k-1]) || (k < n && θ ≥ p[k+1])) && return zero(T)
        if θ < p[k]
            a, b, u, v = p[k-1], p[k], -1, 1
        else
            a, b, u, v = p[k], p[k+1], 1, -1
        end
    end
    β = csc(b - a)
    sᵢ, cᵢ = sincos(a)
    sⱼ, cⱼ = sincos(b)
    s, c = sincos(θ)
    A = u * β * (sᵢ + sⱼ)
    B = v * β * (cᵢ + cⱼ)
    return one(T) / 2 * (one(T) + A * c + B * s)
end

grid(S::TrigonometricSpline, ::Integer) = S.points
==(S::TrigonometricSpline, T::TrigonometricSpline) = S.points == T.points

function _grammatrix_diag(S::TrigonometricSpline, i)
    p = S.points
    n = length(p) - 1
    Δθᵢ₋₁ = p[i] - p[mod1(i - 1, n)]
    i == 1 && (Δθᵢ₋₁ += 2π)
    Δθᵢ = p[i+1] - p[i]
    sᵢ₋₁, cᵢ₋₁ = sincos(Δθᵢ₋₁)
    sᵢ, cᵢ = sincos(Δθᵢ)
    α = -1 / (4 * (1 - cᵢ₋₁)) * (sᵢ₋₁ - 2Δθᵢ₋₁ + Δθᵢ₋₁ * cᵢ₋₁)
    β = -1 / (4 * (1 - cᵢ)) * (sᵢ - 2Δθᵢ + Δθᵢ * cᵢ)
    return α + β
end
function _grammatrix_offdiag(S::TrigonometricSpline, i)
    p = S.points
    Δθᵢ = p[i+1] - p[i]
    sᵢ, cᵢ = sincos(Δθᵢ)
    return 1 / (4 * (1 - cᵢ)) * (sᵢ - Δθᵢ * cᵢ)
end
function grammatrix_layout(_, S::TrigonometricSpline{T}) where {T}
    p = S.points
    n = length(p) - 1
    B = CyclicBandedMatrix{T}(undef, (n, n), (1, 1))
    for i in 1:n
        B[i, i] = _grammatrix_diag(S, i)
        B[i, i + 1] = _grammatrix_offdiag(S, i)
    end
    if n == 2 
        B[1, 2] *= 2
    end
    return Symmetric(B)
end

end # module TrigonometricSplines