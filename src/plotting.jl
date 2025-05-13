_compute_m(n) = min(max(3, 20(n ÷ 2)), 500)

function basis_layout(::LazyLayout, v::SubQuasiArray{T,1,<:SemiclassicalJacobiArc}) where {T}
    return parent(v)
end
function plotgrid(P::SemiclassicalJacobiArc, n::Integer)
    m = _compute_m(n)
    θ = grid(P, m)
    unique!(sort!(θ))
    return θ
end

basis_layout(::LazyLayout, v::SubQuasiArray{T,1,<:Union{<:TrigonometricSpline,<:PeriodicLinearSpline,<:PiecewiseArcPolynomial,<:PeriodicContinuousPolynomial}}) where {T} = parent(v)

plotgrid(spl::Union{<:TrigonometricSpline,<:PeriodicLinearSpline}, n::Integer) = LinRange(extrema(spl.points)..., _compute_m(n))
plotgrid(P::Union{<:PiecewiseArcPolynomial,<:PeriodicContinuousPolynomial}, n::Integer=250) = LinRange(extrema(P.points)..., _compute_m(n))
plotgrid(P::SubQuasiArray{T,1,<:Union{<:PiecewiseArcPolynomial,<:PeriodicContinuousPolynomial}}) where {T} = plotgrid(parent(P), 250)

plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::SubQuasiArray{T,1,<:TrigonometricSpline}, x=plotgrid(g)) where {T} = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::SubQuasiArray{T,1,<:PeriodicLinearSpline}, x=plotgrid(g)) where {T} = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::SubQuasiArray{T,1,<:Union{<:PiecewiseArcPolynomial, <:SemiclassicalJacobiArc}}, x=plotgrid(g)) where {T} = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::SubQuasiArray{T,1,<:PeriodicContinuousPolynomial}, x=plotgrid(g)) where {T} = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{TrigonometricSpline,AbstractVector}}, x=plotgrid(g)) = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{PeriodicLinearSpline,AbstractVector}}, x=plotgrid(g)) = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{<:Union{<:PiecewiseArcPolynomial,<:SemiclassicalJacobiArc},AbstractVector}}, x=plotgrid(g)) = g[x]
plotvalues_size(::Tuple{InfiniteCardinal{1}}, g::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{PeriodicContinuousPolynomial,AbstractVector}}, x=plotgrid(g)) = g[x]