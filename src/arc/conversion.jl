struct SemiclassicalJacobiArcConversionData{T} <: LazyMatrix{T} # BandedMatrix data for SemiclassicalJacobiArc(h, b + 1) \ SemiclassicalJacobiArc(h, b)
    A::AbstractMatrix{T}    # SemiclassicalJacobi(t, -1/2, b + 1, -1/2) \ SemiclassicalJacobi(t, -1/2, b, -1/2)
    B::AbstractMatrix{T}    # SemiclassicalJacobi(t,  1/2, b + 1,  1/2) \ SemiclassicalJacobi(t,  1/2, b,  1/2)
end
size(::SemiclassicalJacobiArcConversionData) = (3, ∞)
function SemiclassicalJacobiArcConversionData(R1::SemiclassicalJacobiArc{T}, R2::SemiclassicalJacobiArc{T}) where {T}
    A = R1.P \ R2.P
    B = R1.Q \ R2.Q
    return SemiclassicalJacobiArcConversionData{T}(A, B)
end
function getindex(M::SemiclassicalJacobiArcConversionData{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(Bool, M, i, j) || throw(BoundsError(M, (i, j)))
    i == 2 && return zero(T) 
    @inbounds begin
        if j == 1
            return i == 3 ? M.A[1, 1] : zero(T)
        elseif j == 2
            return i == 3 ? M.B[1, 1] : zero(T)
        elseif isodd(j)
            j′ = j ÷ 2
            return i == 1 ? M.A[j′, j′+1] : M.A[j′+1, j′+1] # already know i ≠ 2
        else # iseven(j) 
            j′ = j ÷ 2
            return i == 1 ? M.B[j′-1, j′] : M.B[j′, j′] # already know i ≠ 2
        end
    end
end
function \(R1::SemiclassicalJacobiArc, R2::SemiclassicalJacobiArc)
    @assert R1.h == R2.h "Can only convert between polynomials on the same interval."
    R1.b == R2.b && return SquareEye{eltype(R1)}((axes(R1, 2),))
    @assert R1.b == R2.b + 1 "Raising is currently only implemented for zero-step or single-step increments."
    data = SemiclassicalJacobiArcConversionData(R1, R2)
    return BlockedArray(_BandedMatrix(data, 1:∞, 0, 2), (axes(R1, 2), axes(R1, 2)))
end 