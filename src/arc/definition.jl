abstract type AlgebraicOrthogonalPolynomial{d,T} <: MultivariateOrthogonalPolynomial{d,T} end

"""
    ArcWeight(h::T, b::T) <: Weight{T}

Represents the weight `w(θ) = (x - h)^b` on an `Arc{T}`, where `x = cos(θ)`.
"""
struct ArcWeight{T} <: Weight{T}
    h::T
    b::T
end
ArcWeight(h::T, b::V) where {T,V} = ArcWeight(promote(h, b)...)
getindex(w::ArcWeight, θ) = (cos(θ) - w.h)^w.b
axes(w::ArcWeight) = (ArcInclusion(w.h),)
show(io::IO, w::ArcWeight) = summary(io, w)
function summary(io::IO, w::ArcWeight{T}) where {T}
    if isone(w.b)
        print(io, "x - $(w.h) on ", Arc{T})
    elseif iszero(w.b)
        print(io, "1 on ", Arc{T})
    else
        print(io, "(x - $(w.h))^$(w.b) on ", Arc{T})
    end
end

"""
    SemiclassicalJacobiArc(h::T, b::T) <: AlgebraicOrthogonalPolynomial{2, T}

Defines orthogonal polynomials with respect to the weight `(x - h)^b` on 
the part of the unit circle `x^2 + y^2 = 1` where `x ≥ h`.
"""
struct SemiclassicalJacobiArc{T,PP,QQ,LL,UU,QQ1,PP1} <: AlgebraicOrthogonalPolynomial{2,T}
    h::T
    b::T
    P::PP    # SemiclassicalJacobi(t, -1/2, b,     -1/2)
    Q::QQ    # SemiclassicalJacobi(t,  1/2, b,      1/2)
    LP::LL   # SemiclassicalJacobi(t,  1/2, b + 1,  1/2)
    UP::UU   # SemiclassicalJacobi(t, -1/2, b + 1, -1/2)
    Q1::QQ1  # SemiclassicalJacobi(t,  3/2, b + 1,  3/2)
    P1::PP1  # SemiclassicalJacobi(t,  1/2, 1,      1/2)
end

function resizedata!(P::SemiclassicalJacobiArc, ::Colon, n::Int)
    resizedata!(P.P, :, n)
    resizedata!(P.Q, :, n)
    resizedata!(P.LP, :, n)
    resizedata!(P.UP, :, n)
    resizedata!(P.Q1, :, n)
    resizedata!(P.P1, :, n)
    return P
end

SemiclassicalJacobiArc{T}(h, b, P::PP, Q::QQ, LP::LL, UP::UU, Q1::QQ1, P1::PP1) where {T,PP,QQ,LL,UU,QQ1,PP1} = SemiclassicalJacobiArc{T,PP,QQ,LL,UU,QQ1,PP1}(h, b, P, Q, LP, UP, Q1, P1)
function SemiclassicalJacobiArc{T}(h, b) where {T}
    _h, _b = convert(T, h), convert(T, b)
    t = 2 / (1 - _h)
    if !isone(t)
        P = SemiclassicalJacobi(t, -one(T) / 2, _b, -one(T) / 2)
        Q = SemiclassicalJacobi(t, one(T) / 2, _b, one(T) / 2, P)
        LP = SemiclassicalJacobi(t, one(T) / 2, _b + 1, one(T) / 2, Q)
        UP = SemiclassicalJacobi(t, -one(T) / 2, _b + 1, -one(T) / 2, P)
        Q1 = SemiclassicalJacobi(t, 3one(T) / 2, _b + 1, 3one(T) / 2, Q)
        P1 = SemiclassicalJacobi(t, one(T) / 2, one(T), one(T) / 2)
    else
        _b < -one(T) / 2 && throw(ArgumentError("b must be at least -1/2 when h = -1"))
        P = SemiclassicalJacobi(t, -one(T) / 2, _b - one(T) / 2, zero(T))
        Q = SemiclassicalJacobi(t, one(T) / 2, _b + one(T) / 2, zero(T), P)
        LP = SemiclassicalJacobi(t, one(T) / 2, _b + 3one(T) / 2, zero(T), Q)
        UP = SemiclassicalJacobi(t, -one(T) / 2, _b + 3one(T) / 2, zero(T), P)
        Q1 = SemiclassicalJacobi(t, 3one(T) / 2, _b + 3one(T) / 2, one(T), Q)
        P1 = P
    end
    return SemiclassicalJacobiArc{T}(_h, _b, P, Q, LP, UP, Q1, P1)
end
SemiclassicalJacobiArc(h::T, b::V) where {T,V} = SemiclassicalJacobiArc{real(float(promote_type(T, V)))}(h, b)
function SemiclassicalJacobiArc(b, R::SemiclassicalJacobiArc{T}) where {T}
    _b = convert(T, b)
    Pold, Qold = get_P(R), get_Q(R)
    LPold, UPold = get_LP(R), get_UP(R)
    Q1old = get_Q1(R)
    P1old = get_P1(R)
    Pnew = SemiclassicalJacobi(Pold.t, Pold.a, _b, Pold.c, Pold)
    Qnew = SemiclassicalJacobi(Qold.t, Qold.a, _b, Qold.c, Qold)
    LPnew = SemiclassicalJacobi(LPold.t, LPold.a, _b, LPold.c, LPold)
    UPnew = SemiclassicalJacobi(UPold.t, UPold.a, _b, UPold.c, UPold)
    Q1new = SemiclassicalJacobi(Q1old.t, Q1old.a, _b, Q1old.c, Q1old)
    P1new = SemiclassicalJacobi(P1old.t, P1old.a, _b, P1old.c, P1old)
    return SemiclassicalJacobiArc{T}(R.h, _b, Pnew, Qnew, LPnew, UPnew, Q1new, P1new)
end

get_P(R::SemiclassicalJacobiArc) = R.P
get_Q(R::SemiclassicalJacobiArc) = R.Q
get_LP(R::SemiclassicalJacobiArc) = R.LP
get_UP(R::SemiclassicalJacobiArc) = R.UP
get_Q1(R::SemiclassicalJacobiArc) = R.Q1
get_P1(R::SemiclassicalJacobiArc) = R.P1

orthogonalityweight(P::SemiclassicalJacobiArc{T}) where {T} = ArcWeight{T}(P.h, P.b)

show(io::IO, R::SemiclassicalJacobiArc) = summary(io, R)
summary(io::IO, R::SemiclassicalJacobiArc{T}) where {T} = print(io, "SemiclassicalJacobiArc{$T} with weight ", ArcWeight{T}(R.h, R.b))

axes(R::SemiclassicalJacobiArc) = (ArcInclusion(R.h), _BlockedUnitRange(1:2:∞))
# FIXME: Change to BlockedOneTo. It messes up the expand tests for some reason inside plan_grid_transform...

sub_materialize(_, V::SubArray{<:Any,1,<:SemiclassicalJacobiArc}, ::Tuple{<:IdentityUnitRange{<:BlockedUnitRange{Int,<:InfStepRange}}}) = V # this fixes e.g. R[0.2, :]. Needed since axes(R, 2) isn't an InfiniteArrays.InfAxes

# unitrange(ax::IdentityUnitRange{<:BlockedUnitRange{Int, <:InfStepRange}}) = InfUnitRange(1:∞) # FIXME: Piracy (and also just incorrect). This is needed to make print_matrix work for e.g. R[0.2, :]. See Line 171 in base/arrayshow.jl (_print_matrix(io, inferencebarrier(X), pre, sep, post, hdots, vdots, ddots, hmod, vmod, unitrange(axes(X,1)), unitrange(axes(X,2))))

isassigned(R::SemiclassicalJacobiArc, θ::Number, JR::Integer) = θ ∈ axes(R, 1) && JR ∈ axes(R, 2)

(==)(P::SemiclassicalJacobiArc, Q::SemiclassicalJacobiArc) = (P.h == Q.h) && (P.b == Q.b)

_mapx(R::SemiclassicalJacobiArc{T}, x) where {T} = clamp((x - 1) / (R.h - 1), zero(T), one(T)) # (x-1)/(h-1) could be e.g. 1.00000000003 which gives a domain error in SemiclassicalJacobi
@propagate_inbounds _eval_jacobi(P::SemiclassicalJacobi{T}, x, inds::UnitRange) where {T} = P[x, inds]::Vector{T}
@propagate_inbounds _eval_jacobi(P::SemiclassicalJacobi{T}, x, n::Int) where {T} = P[x, n]::T
@propagate_inbounds _eval_P(R::SemiclassicalJacobiArc{T}, x, idx::UnitRange) where {T} = _eval_jacobi(get_P(R), x, idx)::Vector{T}
@propagate_inbounds _eval_P(R::SemiclassicalJacobiArc{T}, x, n::Int) where {T} = _eval_jacobi(get_P(R), x, n)::T
@propagate_inbounds _eval_Q(R::SemiclassicalJacobiArc{T}, x, idx::UnitRange) where {T} = _eval_jacobi(get_Q(R), x, idx)::Vector{T}
@propagate_inbounds _eval_Q(R::SemiclassicalJacobiArc{T}, x, n::Int) where {T} = _eval_jacobi(get_Q(R), x, n)::T
function getindex(R::SemiclassicalJacobiArc{V}, θ::Number, JR::BlockOneTo) where {V}
    T = promote_type(V, typeof(θ))
    ret = BlockedArray{T}(undef, (axes(R, 2)[JR],))
    isempty(ret) && return ret
    y, x = sincos(θ)
    xt = _mapx(R, x)
    idx = Int(last(JR))
    P = _eval_P(R, xt, 1:idx)
    Q = _eval_Q(R, xt, 1:(idx-1))
    ret[1] = P[1]
    for J in Block(2):JR[end]
        _idx = Int(J)
        ret[J] .= ((y * Q[_idx-1])::T, P[_idx]::T)
    end
    return ret
end
function getindex(R::SemiclassicalJacobiArc{V}, θ::Number, j::Int) where {V}
    T = promote_type(V, typeof(θ))
    y, x = sincos(θ)
    xt = _mapx(R, x)
    if isodd(j)
        jd = 1 + (j ÷ 2)
        return _eval_P(R, xt, jd)::T
    else
        jd = j ÷ 2
        return (y * _eval_Q(R, xt, jd))::T
    end
end
function getindex(R::SemiclassicalJacobiArc{V}, θ::Number, Kk::BlockIndex{1}) where {V}
    T = promote_type(V, typeof(θ))
    K, k = block(Kk), blockindex(Kk)
    @boundscheck (Int(K) ≥ 1 && (K == 1 ? k == 1 : k ∈ (1, 2))) || throw(BoundsError(R, (θ, Kk)))
    j = axes(R, 2)[Kk]
    return R[θ, j]::T
end
function getindex(R::SemiclassicalJacobiArc, θ::Number, JR::Union{Block{1},BlockRange{1}})
    JRn = Block.(OneTo(Int(maximum(JR))))
    return R[θ, JRn][JR]
end