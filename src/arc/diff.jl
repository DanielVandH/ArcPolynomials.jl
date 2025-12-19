function semiclassicaljacobiarc_diffp(R::SemiclassicalJacobiArc{T}) where {T} # dp/dθ = q(b+1) * D 
    h = R.h
    D = diff(get_P(R)).args[2]
    return 1 / (1 - h) * D
end

function semiclassicaljacobiarc_diffq(R::SemiclassicalJacobiArc) # dq/dθ = p(b+1) * D 
    T = eltype(R)
    h = R.h
    Q = get_Q(R)
    Q1 = get_Q1(R)
    if R.b ≠ -1
        DD1 = divdiff(Q1, Q)
    else
        P1 = HalfWeighted{:b}(get_P1(R))
        HQ = HalfWeighted{:b}(Q1)
        Dmat = divdiff(HQ, P1)
        b2 = Vcat(zero(T), zero(T), Dmat.ev)
        b1 = Vcat(zero(T), Dmat.dv)
        data = Hcat(b2, b1)'
        DD1 = _BandedMatrix(data, ∞, -1, 2)
    end
    LP = get_LP(R)
    L1 = Weighted(LP) \ Weighted(Q1)
    UP = get_UP(R)
    J = jacobimatrix(Q)
    X = (h - 1) * J + I
    R1 = LP \ Q
    R2 = LP \ UP
    D1 = BidiagonalConjugation(R2, R1, X, 'L')
    U = LP \ UP
    X = L1
    V = DD1
    D2 = (1 - h) * BidiagonalConjugation(U, X, V, 'L')
    Dq = D1 + D2 # U2 * ((h - 1) * J + I) + (1 - h) * U1 * L1 * D1
    return Dq
end

struct SemiclassicalJacobiArcDiffData{T,A,B} <: LazyMatrix{T}
    D⁻::A
    D⁺::B
end
SemiclassicalJacobiArcDiffData{T}(D⁻::A, D⁺::B) where {T,A,B} = SemiclassicalJacobiArcDiffData{T,A,B}(D⁻, D⁺)
function SemiclassicalJacobiArcDiffData(R::SemiclassicalJacobiArc{T}) where {T}
    D⁻ = semiclassicaljacobiarc_diffp(R)
    D⁺ = semiclassicaljacobiarc_diffq(R)
    return SemiclassicalJacobiArcDiffData{T}(D⁻, D⁺)
end
size(::SemiclassicalJacobiArcDiffData) = (5, ∞)
function getindex(D::SemiclassicalJacobiArcDiffData{T}, i::Int, j::Int)::T where {T}
    @boundscheck checkbounds(Bool, D, i, j) || throw(BoundsError(D, (i, j)))
    iseven(i) && return zero(T)
    D⁻, D⁺ = D.D⁻, D.D⁺
    if j == 1
        return zero(T)
    elseif j == 2
        return i == 3 ? D⁺[1, 1] : i == 5 ? D⁺[2, 1] : zero(T)
    elseif j == 3
        return i == 3 ? D⁻[1, 2] : zero(T)
    elseif iseven(j)
        j′ = j >>> 1
        return i == 3 ? D⁺[j′, j′] : i == 5 ? D⁺[j′+1, j′] : zero(T)
    else # isodd(j)
        j′ = j >>> 1
        return i == 1 ? D⁻[j′-1, j′+1] : i == 3 ? D⁻[j′, j′+1] : i == 5 ? D⁻[j′+1, j′+1] : zero(T)
    end
end
function diff(R::SemiclassicalJacobiArc; dims=1) # dR/dθ
    D = SemiclassicalJacobiArcDiffData(R)
    R1 = SemiclassicalJacobiArc(R.b + 1, R)
    return R1 * BlockedArray(_BandedMatrix(D, 1:∞, 1, 3), (axes(R, 2), axes(R, 2)))
end
