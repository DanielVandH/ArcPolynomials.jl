function semiclassicaljacobiarc_diffp(R::SemiclassicalJacobiArc) # dp/dθ = q(b+1) * D 
    h = R.h
    D = arguments(diff(R.P))[2]
    return 1 / (1 - h) * D
end

function semiclassicaljacobiarc_diffq(R::SemiclassicalJacobiArc) # dq/dθ = p(b+1) * D 
    h = R.h
    Q1, D1 = arguments(diff(R.Q))
    D1 = D1
    LP = SemiclassicalJacobi(R.Q.t, R.Q.a, Q1.b, R.Q.c, R.Q)
    L1 = Weighted(LP) \ Weighted(Q1)
    UP = SemiclassicalJacobi(R.P.t, R.P.a, Q1.b, R.P.c, R.P)
    U1 = UP \ LP
    U2 = UP \ R.Q
    J = jacobimatrix(R.Q)
    D = U2 * ((h - 1) * J + I) + (1 - h) * U1 * L1 * D1 
    return Bidiagonal(D[band(0)], D[band(-1)], :L)
end

struct SemiclassicalJacobiArcDiffData{T} <: LazyMatrix{T} 
    D⁻::AbstractMatrix{T}
    D⁺::AbstractMatrix{T}
end
function SemiclassicalJacobiArcDiffData(R::SemiclassicalJacobiArc)
    D⁻ = semiclassicaljacobiarc_diffp(R)
    D⁺ = semiclassicaljacobiarc_diffq(R)
    return SemiclassicalJacobiArcDiffData(D⁻, D⁺)
end
size(::SemiclassicalJacobiArcDiffData) = (5, ∞)
function getindex(D::SemiclassicalJacobiArcDiffData{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(Bool, D, i, j) || throw(BoundsError(D, (i, j)))
    iseven(i) && return zero(T)
    D⁻, D⁺ = D.D⁻, D.D⁺
    @inbounds begin 
        if j == 1 
            return zero(T) 
        elseif j == 2 
            return i == 3 ? D⁺[1, 1] : i == 5 ? D⁺[2, 1] : zero(T)
        elseif j == 3 
            return i == 3 ? D⁻[1, 2] : zero(T)
        elseif iseven(j)
            j′ = j ÷ 2 
            return i == 3 ? D⁺[j′, j′] : i == 5 ? D⁺[j′+1, j′] : zero(T)
        else # isodd(j)
            j′ = j ÷ 2
            return i == 1 ? D⁻[j′-1, j′+1] : i == 3 ? D⁻[j′, j′+1] : i == 5 ? D⁻[j′+1, j′+1] : zero(T)
        end
    end
end
function diff(R::SemiclassicalJacobiArc; dims=1) # dR/dθ
    D = SemiclassicalJacobiArcDiffData(R)
    R1 = SemiclassicalJacobiArc(R.b + 1, R)
    return R1 * BlockedArray(_BandedMatrix(D, 1:∞, 1, 3), (axes(R, 2), axes(R, 2)))
end
