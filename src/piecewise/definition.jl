struct PiecewiseArcPolynomial{O,T,G,PP,PP0} <: AbstractPiecewisePolynomial{O,T,G}
    P::PP
    P0::PP0
    points::G
    function PiecewiseArcPolynomial{O}(P::PP, P0::PP0, points::G) where {O,G,PP,PP0}
        T = eltype(points)
        @assert points[1] == -T(π) && points[end] == T(π) && issorted(points) "The points must be sorted and in the range [-π, π]."
        new{O,T,G,PP,PP0}(P, P0, points)
    end
end
function PiecewiseArcPolynomial{O}(points::AbstractRange) where {O}
    @assert length(points) - 1 ≥ 2 "There must be at least two elements."
    require_one_based_indexing(points)
    h = step(points)
    ϕ = cos(h / 2)
    T = eltype(points)
    P = SemiclassicalJacobiArc(T(ϕ), -one(T))
    P0 = SemiclassicalJacobiArc(zero(T), P)
    return PiecewiseArcPolynomial{O}(P, P0, points)
end
function PiecewiseArcPolynomial{O}(points) where {O}
    @assert length(points) - 1 ≥ 2 "There must be at least two elements."
    require_one_based_indexing(points)
    h = diff(points)
    ϕ = cos.(h / 2)
    T = eltype(points)
    P = SemiclassicalJacobiArc.(T.(ϕ), -one(T))
    P0 = SemiclassicalJacobiArc.(zero(T), P)
    return PiecewiseArcPolynomial{O}(P, P0, points)
end
function PiecewiseArcPolynomial{O}(n::Int) where {O}
    @assert n ≥ 2 "There must be at least two elements."
    points = LinRange(-π, π, n + 1)
    return PiecewiseArcPolynomial{O}(points)
end
function PiecewiseArcPolynomial{O}(P::PiecewiseArcPolynomial{OO}) where {O,OO}
    return PiecewiseArcPolynomial{O}(P.P, P.P0, P.points)
end

@inline has_equal_spacing(::PiecewiseArcPolynomial{O,T,G,PP}) where {O,T,G,PP} = !(PP <: Vector)

@inline get_P(P::PiecewiseArcPolynomial, element) = has_equal_spacing(P) ? P.P : P.P[element]
@inline get_P0(P::PiecewiseArcPolynomial, element) = has_equal_spacing(P) ? P.P0 : P.P0[element]

==(P::PiecewiseArcPolynomial{O1}, Q::PiecewiseArcPolynomial{O2}) where {O1,O2} = P.P == Q.P && P.P0 == Q.P0 && P.points == Q.points && O1 == O2

show(io::IO, P::PiecewiseArcPolynomial) = summary(io, P)
summary(io::IO, P::PiecewiseArcPolynomial{O}) where {O} = print(io, "PiecewiseArcPolynomial{$O}($(P.points))")

@inline function find_element(P::PiecewiseArcPolynomial, θ)
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

function getindex(P::PiecewiseArcPolynomial, θ::Number, JR::Union{Block{1},BlockRange{1}})
    JRn = Block.(OneTo(Int(maximum(JR))))
    return P[θ, JRn][JR]
end
function getindex(P::PiecewiseArcPolynomial{O,T}, θ::Number, Kk::BlockIndex{1}) where {O,T}
    K, k = block(Kk), blockindex(Kk)
    element = find_element(P, θ)
    if O == 1 && Int(K) == 1
        return TrigonometricSpline(P.points)[θ, k]
    elseif element == k
        return P[θ, Block.(OneTo(Int(K)))][Kk]
    else
        return zero(T)
    end
end
function getindex(P::PiecewiseArcPolynomial, θ::Number, k::Int)
    Kk = findblockindex(axes(P, 2), k)
    return P[θ, Kk]
end

function getindex(P::PiecewiseArcPolynomial{0,T}, θ::Number, JR::BlockOneTo) where {T}
    ## Setup and figure out where we are 
    ret = BlockedArray{T}(undef, (axes(P, 2)[JR],))
    isempty(ret) && return ret
    fill!(ret, zero(T))
    element = find_element(P, θ)::Int

    ## Map to the associated element 
    a, b = convert(T, P.points[element]), convert(T, P.points[element+1])
    if θ == a
        θ′ = (a - b) / 2
    elseif θ == b
        θ′ = (b - a) / 2
    else
        θ′ = θ - (a + b) / 2 # map to [-ϕ, ϕ]
    end

    #=
    We need to also compute what block we need from P.P. Here, recall that the polynomials are defined by 
    [
        p[1; 1] p[1; 2] ... p[1; n] |       <-- (1, 1)
        q[1; 1] q[1; 2] ... q[1; n] |       <-- (2, 2)
        p[2; 1] p[2; 2] ... p[2; n] |       <-- (3, 2)
        q[2; 1] q[2; 2] ... q[2; n] |       <-- (4, 3) 
        p[3; 1] p[3; 2] ... p[3; n] |       <-- (5, 3)
        q[3; 1] q[3; 2] ... q[3; n] |       <-- (6, 4)
        p[4; 1] p[4; 2] ... p[4; n] |       <-- (7, 4)
        q[4; 1] q[4; 2] ... q[4; n] |       <-- (8, 5)
        p[5; 1] p[5; 2] ... p[5; n] |       <-- (9, 5)
        ⋮
    ]
    The tuples next to each row give the block index relative to P, 
    and the relevant block each polynomial comes from inside P.P. In particular, see 
    that for a given `Block(i)` relative to `P`, the associated block from `P.P` is 
    given by `Block(j)`, where `j = (i ÷ 2) + 1`.
    =#
    d = Int(last(JR))
    JR′ = Block(1):Block((d ÷ 2) + 1)
    nel = length(P.points) - 1
    vals = get_P0(P, element)[θ′, JR′]
    if isodd(d)
        ret[element:nel:end] .= vals
    else
        ret[element:nel:end] .= vals[begin:end-1]
    end
    return ret
end

function getindex(P::PiecewiseArcPolynomial{1,T}, θ::Number, JR::BlockOneTo) where {T}
    ## Setup and figure out where we are 
    ret = BlockedArray{T}(undef, (axes(P, 2)[JR],))
    isempty(ret) && return ret
    fill!(ret, zero(T))
    element = find_element(P, θ)

    ## Map to the associated element 
    a, b = convert(T, P.points[element]), convert(T, P.points[element+1])
    θ′ = θ - (a + b) / 2 # map to [-ϕ, ϕ]

    #=
    Now we need to consider how to fill in ret. The block structure here is now 
        [
        h[1]    h[2]    ... h[n]           |     <-- 1
        p[2; 1] p[2; 2] ... p[2; n]        |     <-- (2, 2)
        q[3; 1] q[3; 2] ... q[3; n]        |     <-- (3, 3) 
        p[3; 1] p[3; 2] ... p[3; n]        |     <-- (4, 3)
        q[4; 1] q[4; 2] ... q[4; n]        |     <-- (5, 4)
        p[4; 1] p[4; 2] ... p[4; n]        |     <-- (6, 4)
        q[5; 1] q[5; 2] ... q[5; n]        |     <-- (7, 5)
        p[5; 1] p[5; 2] ... p[5; n]        |     <-- (8, 5)
        ⋮
    ]
    The h functions are the hat functions. Here we see that, for a given 
    `Block(i)` relative to `P`, the hat functions correspond to `Block(1)`.
    Otherwise, the associated block from `P.P` is given by `Block(j)`, 
    where `j = ((i + 1) ÷ 2) + 1.
    =#
    d = Int(last(JR))
    ret[Block(1)] .= TrigonometricSpline(P.points)[θ, :]
    nel = length(P.points) - 1
    if d > 1
        JR′ = Block(2):Block(((d + 1) ÷ 2) + 1)
        vals = get_P(P, element)[θ′, JR′]
        if iseven(d)
            ret[(nel+element):nel:end] .= vals[begin+1:end]
        else
            ret[(nel+element):nel:end] .= vals[begin+1:end-1]
        end
    end
    return ret
end

function axes(P::PiecewiseArcPolynomial{O,T}) where {O,T}
    ax1 = ArcInclusion{T}(-1)
    nel = length(P.points) - 1
    ax2 = blockedrange(Fill(nel, ∞))
    return (ax1, ax2)
end