struct PeriodicContinuousPolynomial{O,T,G} <: AbstractPiecewisePolynomial{O,T,G}
    points::G
end
PeriodicContinuousPolynomial{O,T}(points::G) where {O,T,G} = PeriodicContinuousPolynomial{O,T,G}(points)
PeriodicContinuousPolynomial{O}(points) where {O} = PeriodicContinuousPolynomial{O,Float64}(points)
PeriodicContinuousPolynomial{O,T}(P::PeriodicContinuousPolynomial) where {O,T} = PeriodicContinuousPolynomial{O,T}(P.points)
PeriodicContinuousPolynomial{O}(P::PeriodicContinuousPolynomial) where {O} = PeriodicContinuousPolynomial{O,eltype(P)}(P.points)
PeriodicContinuousPolynomial{O}(r::Integer) where {O} = PeriodicContinuousPolynomial{O}(LinRange(-π, π, r + 1))
axes(P::PeriodicContinuousPolynomial) = (Inclusion(ℝ), blockedrange(Fill(length(P.points) - 1, ∞)))

==(P::PeriodicContinuousPolynomial{O1}, Q::PeriodicContinuousPolynomial{O2}) where {O1,O2} = P.points == Q.points && O1 == O2

show(io::IO, P::PeriodicContinuousPolynomial) = summary(io, P)
summary(io::IO, P::PeriodicContinuousPolynomial{O}) where {O} = print(io, "PeriodicContinuousPolynomial{$O}($(P.points))")

@inline function find_element(P::PeriodicContinuousPolynomial, θ)
    θ = map_to_std_range(θ)
    points = P.points
    element = if θ == map_to_std_range(points[1])
        1
    elseif θ == map_to_std_range(points[end])
        length(points) - 1
    else
        searchsortedfirst(points, θ, by=map_to_std_range) - 1
    end
    (element < 1 || element ≥ length(points)) && error(lazy"Point $θ not in domain.") # Could also just return nothing and then check for isnothing later.
    return element::Int
end

function getindex(P::PeriodicContinuousPolynomial, θ::Number, JR::Union{Block{1},BlockRange{1}})
    JRn = Block.(OneTo(Int(maximum(JR))))
    return P[θ, JRn][JR]
end
function getindex(P::PeriodicContinuousPolynomial{O,T}, θ::Number, Kk::BlockIndex{1}) where {O,T}
    K, k = block(Kk), blockindex(Kk)
    element = find_element(P, θ)
    if O == 1 && Int(K) == 1
        return PeriodicLinearSpline(P.points)[θ, k]
    elseif element == k
        return P[θ, Block.(OneTo(Int(K)))][Kk]
    else
        return zero(T)
    end
end
function getindex(P::PeriodicContinuousPolynomial, θ::Number, k::Int)
    Kk = findblockindex(axes(P, 2), k)
    return P[θ, Kk]
end

function getindex(P::PeriodicContinuousPolynomial{0,T}, θ::Number, JR::BlockOneTo) where {T}
    ret = BlockedArray{T}(undef, (axes(P, 2)[JR],))
    isempty(ret) && return ret
    fill!(ret, zero(T))
    element = find_element(P, θ)::Int
    θ = map_to_std_range(θ)
    a, b = convert(T, P.points[element]), convert(T, P.points[element+1])
    θ = (2θ - a - b) / (b - a)
    if θ < -1 && isapprox(θ, -1, atol=1e-12)
        θ = -one(T)
    elseif θ > 1 && isapprox(θ, 1, atol=1e-12)
        θ = one(T)
    end # @inbounds doesn't want to propagate into Ultraspherical in tests
    vals = @inbounds Legendre{T}()[θ, OneTo(Int(maximum(JR)))]
    nel = length(P.points) - 1
    ret[element:nel:end] .= vals
    return ret
end

function getindex(P::PeriodicContinuousPolynomial{1,T}, θ::Number, JR::BlockRange{1}) where {T}
    ret = BlockedArray{T}(undef, (axes(P, 2)[JR],))
    isempty(ret) && return ret
    fill!(ret, zero(T))
    element = find_element(P, θ)::Int
    θ = map_to_std_range(θ)
    a, b = convert(T, P.points[element]), convert(T, P.points[element+1])
    θ′ = (2θ - a - b) / (b - a)
    if θ′ < -1 && isapprox(θ′, -1, atol=1e-12)
        θ′ = -one(T)
    elseif θ > 1 && isapprox(θ′, 1, atol=1e-12)
        θ′ = one(T)
    end # @inbounds doesn't want to propagate into Ultraspherical in tests

    d = Int(maximum(JR))
    ret[Block(1)] .= PeriodicLinearSpline(P.points)[θ, :]
    if d > 1
        JR′ = 3:(d+1)
        vals = @inbounds Ultraspherical{T}(-one(T) / 2)[θ′, JR′]
        nel = length(P.points) - 1
        ret[(element+nel):nel:end] .= vals
    end
    return ret
end