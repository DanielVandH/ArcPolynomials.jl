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
getindex(w::ArcWeight, θ) = @inbounds (cos(θ) - w.h)^w.b
axes(w::ArcWeight) = (ArcInclusion(w.h),)
show(io::IO, w::ArcWeight)= summary(io, w)
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
struct SemiclassicalJacobiArc{T,PP,QQ} <: AlgebraicOrthogonalPolynomial{2,T}
    h::T
    b::T
    P::PP
    Q::QQ
end
SemiclassicalJacobiArc{T}(h, b, P::PP, Q::QQ) where {T,PP,QQ} = SemiclassicalJacobiArc{T,PP,QQ}(h, b, P, Q)
function SemiclassicalJacobiArc{T}(h, b) where {T}
    _h, _b = convert(T, h), convert(T, b) 
    t = 2 / (1 - _h)
    if !isone(t) 
        P = SemiclassicalJacobi(t, -one(T) / 2, _b, -one(T) / 2)
        Q = SemiclassicalJacobi(t, one(T) / 2, _b, one(T) / 2, P)
    else
        _b < -one(T) / 2 && throw(ArgumentError("b must be at least -1/2 when h = -1"))
        P = SemiclassicalJacobi(t, -one(T) / 2, _b - one(T) / 2, zero(T))
        Q = SemiclassicalJacobi(t, one(T) / 2, _b + one(T) / 2, zero(T), P)
    end
    return SemiclassicalJacobiArc{T}(_h, _b, P, Q)
end
SemiclassicalJacobiArc(h::T, b::V) where {T,V} = SemiclassicalJacobiArc{real(float(promote_type(T, V)))}(h, b)
function SemiclassicalJacobiArc(b, R::SemiclassicalJacobiArc{T}) where {T}
    _b = convert(T, b)
    Pold, Qold = R.P, R.Q 
    Pnew = SemiclassicalJacobi(Pold.t, Pold.a, _b, Pold.c, Pold) 
    Qnew = SemiclassicalJacobi(Qold.t, Qold.a, _b, Qold.c, Qold) 
    return SemiclassicalJacobiArc{T}(R.h, _b, Pnew, Qnew)
end

orthogonalityweight(P::SemiclassicalJacobiArc{T}) where {T} = ArcWeight{T}(P.h, P.b)

show(io::IO, R::SemiclassicalJacobiArc) = summary(io, R)
summary(io::IO, R::SemiclassicalJacobiArc{T}) where {T} = print(io, "SemiclassicalJacobiArc{$T} with weight ", ArcWeight{T}(R.h, R.b))

axes(R::SemiclassicalJacobiArc) = (ArcInclusion(R.h), _BlockedUnitRange(1:2:∞))
# FIXME: Change to BlockedOneTo. It messes up the expand tests for some reason inside plan_grid_transform...

sub_materialize(_, V::SubArray{<:Any, 1, <:SemiclassicalJacobiArc}, ::Tuple{<:IdentityUnitRange{<:BlockedUnitRange{Int, <:InfStepRange}}}) = V # this fixes e.g. R[0.2, :]. Needed since axes(R, 2) isn't an InfiniteArrays.InfAxes

# unitrange(ax::IdentityUnitRange{<:BlockedUnitRange{Int, <:InfStepRange}}) = InfUnitRange(1:∞) # FIXME: Piracy (and also just incorrect). This is needed to make print_matrix work for e.g. R[0.2, :]. See Line 171 in base/arrayshow.jl (_print_matrix(io, inferencebarrier(X), pre, sep, post, hdots, vdots, ddots, hmod, vmod, unitrange(axes(X,1)), unitrange(axes(X,2))))

isassigned(R::SemiclassicalJacobiArc, θ::Number, JR::Integer) = θ ∈ axes(R, 1) && JR ∈ axes(R, 2)

(==)(P::SemiclassicalJacobiArc, Q::SemiclassicalJacobiArc) = (P.h == Q.h) && (P.b == Q.b)

_mapx(R::SemiclassicalJacobiArc{T}, x) where {T} = clamp((x - 1) / (R.h - 1), zero(T), one(T)) # (x-1)/(h-1) could be e.g. 1.00000000003 which gives a domain error in SemiclassicalJacobi
function getindex(R::SemiclassicalJacobiArc{V}, θ::Number, JR::BlockOneTo) where {V}
    T = promote_type(V, typeof(θ))
    ret = BlockedArray{T}(undef, (axes(R, 2)[JR],))
    isempty(ret) && return ret
    y, x = sincos(θ)
    xt = _mapx(R, x)
    idx = Int(last(JR))
    P = @inbounds R.P[xt, 1:idx]::Vector{T}
    Q = @inbounds R.Q[xt, 1:(idx-1)]::Vector{T}
    ret[1] = P[1]
    @inbounds for J in Block(2):JR[end]
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
        return R.P[xt, jd]::T
    else
        jd = j ÷ 2
        return (y * R.Q[xt, jd])::T
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