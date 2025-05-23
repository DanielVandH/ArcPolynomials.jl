function multiplication_blocks(a::ApplyQuasiVector{T,typeof(*),<:Tuple{<:SemiclassicalJacobiArc,<:Any}}) where {T}
    P, 𝐚 = arguments(a)
    𝐚data = paddeddata(𝐚)
    𝐚₂ = 𝐚data[1:2:end]
    𝐚₁ = 𝐚data[2:2:end]
    P⁻ = P.P
    P⁺ = P.Q
    J⁻ = jacobimatrix(P⁻)
    J⁺ = jacobimatrix(P⁺)
    b1⁻ = Clenshaw(𝐚₁, recurrencecoefficients(P⁺)..., J⁻, _p0(P⁺))
    b1⁺ = Clenshaw(𝐚₁, recurrencecoefficients(P⁺)..., J⁺, _p0(P⁺))
    b2⁻ = Clenshaw(𝐚₂, recurrencecoefficients(P⁻)..., J⁻, _p0(P⁻))
    b2⁺ = Clenshaw(𝐚₂, recurrencecoefficients(P⁻)..., J⁺, _p0(P⁻))
    _R = P⁺ \ P⁻ # This has some layout issues when b = -1, so we force this to be a banded matrix by reconstructing it below
    V = eltype(_R)
    r0, r1, r2 = _R[band(0)], Vcat(zero(V), _R[band(1)]), Vcat(zero(V), zero(V), _R[band(2)])
    R = _BandedMatrix(Vcat(r2', r1', r0'), axes(_R, 1), 0, 2)
    L = Weighted(P⁻) \ Weighted(P⁺)
    h = P.h
    M₂₂ = b2⁻
    M₂₁ = b2⁺
    M₁₂ = R * b1⁻
    M₁₁ = (1 - h)^2 * L * b1⁺
    return M₂₂, M₂₁, M₁₂, M₁₁, length(𝐚₁), max(0, length(𝐚₂) - 1)
end

struct ArcMultiplicationMatrix{T,U,V,W,X} <: AbstractBandedMatrix{T}
    M₂₂::U
    M₂₁::V
    M₁₂::W
    M₁₁::X
    bandwidth::Int
    function ArcMultiplicationMatrix(M₂₂::U, M₂₁::V, M₁₂::W, M₁₁::X, bandwidth::Int) where {U,V,W,X}
        T = promote_type(eltype(M₂₂), eltype(M₂₁), eltype(M₁₂), eltype(M₁₁))
        return new{T,U,V,W,X}(M₂₂, M₂₁, M₁₂, M₁₁, bandwidth)
    end
end

function ArcMultiplicationMatrix(a::ApplyQuasiVector{T,typeof(*),<:Tuple{<:SemiclassicalJacobiArc,<:Any}}) where {T}
    M₂₂, M₂₁, M₁₂, M₁₁, α₁, α₂ = multiplication_blocks(a)
    bw = 2max(α₁, α₂) + (α₁ ≥ α₂)
    return ArcMultiplicationMatrix(M₂₂, M₂₁, M₁₂, M₁₁, bw)
end

Base.array_summary(io::IO, J::ArcMultiplicationMatrix{T}, inds::Tuple{Vararg{OneToInf{Int}}}) where {T} =
    print(io, Base.dims2string(length.(inds)), " ArcMultiplicationMatrix{$T} with bandwidths $(bandwidths(J))")

@inline copy(J::ArcMultiplicationMatrix) = J
@inline size(J::ArcMultiplicationMatrix) = (∞, ∞)
@inline axes(J::ArcMultiplicationMatrix) = (OneToInf(), OneToInf())
@inline bandwidths(J::ArcMultiplicationMatrix) = (J.bandwidth, J.bandwidth)

struct ArcMultiplicationLayout <: AbstractLazyBandedLayout end
@inline MemoryLayout(::Type{<:ArcMultiplicationMatrix}) = ArcMultiplicationLayout()
@inline sublayout(::ArcMultiplicationLayout, ::Type{<:NTuple{2,AbstractUnitRange{Int}}}) = ArcMultiplicationLayout()
@inline sublayout(::ArcMultiplicationLayout, ::Type{<:Tuple{AbstractUnitRange{Int},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
@inline sublayout(::ArcMultiplicationLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},AbstractUnitRange{Int}}}) = LazyBandedLayout()
@inline sublayout(::ArcMultiplicationLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
@inline transposelayout(::ArcMultiplicationLayout) = LazyBandedLayout()
@inline sub_materialize(::ArcMultiplicationMatrix, J) = BandedMatrix(J)

@inline @propagate_inbounds function inbands_getindex(J::ArcMultiplicationMatrix{T}, i::Int, j::Int)::T where {T}
    @boundscheck checkbounds(Bool, J, i, j) || throw(BoundsError(J, (i, j)))
    if isodd(j)
        j′ = 1 + (j ÷ 2)
        if isodd(i)
            i′ = 1 + (i ÷ 2)
            return J.M₂₂[i′, j′]
        else
            i′ = i ÷ 2
            return J.M₁₂[i′, j′]
        end
    else
        j′ = j ÷ 2
        if isodd(i)
            i′ = 1 + (i ÷ 2)
            return J.M₁₁[i′, j′]
        else
            i′ = i ÷ 2
            return J.M₂₁[i′, j′]
        end
    end
end

@inline @propagate_inbounds function getindex(J::ArcMultiplicationMatrix{T}, i::Int, j::Int)::T where {T}
    @boundscheck checkbounds(Bool, J, i, j) || throw(BoundsError(J, (i, j)))
    ℓ, u = bandwidths(J)
    if -ℓ ≤ j - i ≤ u
        return @inbounds inbands_getindex(J, i, j)
    else
        return zero(T)
    end
end

function layout_broadcasted(::Tuple{<:ExpansionLayout, <:Any}, ::typeof(*), a, P::SemiclassicalJacobiArc) 
    @assert basis(a) == P
    J = ArcMultiplicationMatrix(a)
    return P * BlockedArray(J, (axes(P, 2), axes(P, 2)))
end