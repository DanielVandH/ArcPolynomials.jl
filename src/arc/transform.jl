increasingtruncations(::BlockedUnitRange{Int,InfStepRange{Int,Int}}) = broadcast(n -> _BlockedUnitRange(1:2:n), (2 .^ (4:∞))) # piracy
function grid(R::SemiclassicalJacobiArc{T}, n::Int) where {T}
    if isone(-R.b)
        R0 = SemiclassicalJacobiArc(zero(T), R)
        return grid(R0, n)
    else
        t, _ = gaussradau(R.P, n - 1, zero(T))
        s = R.h - 1
        θ = Vector{T}(undef, 2n - 1)
        θ[1] = zero(T)
        for i in 2:n
            x = t[i] * s + 1
            y = sqrt(1 - x^2)
            ψ = atan(y, x)
            θ[2(i-1)] = ψ
            θ[2i-1] = -ψ 
        end
        return θ
    end
end
function _fix_broadcast!(N, w, Rvals)
    # Can't do @. N += w * Rvals^2, see https://github.com/JuliaArrays/BlockArrays.jl/issues/411
    for i in eachindex(N)
        N[i] += w * Rvals[i]^2
    end
    return N
end

struct ArcPlan{T,S,D} <: Plan{T}
    X::Matrix{T}
    sz::S
    dims::D # dims is unused
end

*(P::ArcPlan, f::AbstractArray) = P.X * f

function plan_transform(R::SemiclassicalJacobiArc{T}, (n,)::Tuple{Int}, dims::Int) where {T}
    if isone(-R.b)
        R0 = SemiclassicalJacobiArc(zero(T), R)
        P = plan_transform(R0, (n,), dims)
        # C = view(R0 \ R, 1:(2n-1), 1:(2n-1))
        C = (R0\R)[1:(2n-1), 1:(2n-1)] # doing view means we have to define strides() for some reason, which just leads to "conversion to pointer not defined for SemiclassicalJacobiArcConversionData{Float64}"
        return ArcPlan(C \ P.X, n, dims)
    else
        t, w = gaussradau(R.P, n - 1, zero(T))
        s = R.h - 1
        P = zeros(T, 2n - 1, 2n - 1)
        N = zeros(T, 2n - 1)
        blocks = Block(1):Block(n)
        Rvals = R[zero(T), blocks]
        @. P[:, 1] = 2w[1] * Rvals
        _fix_broadcast!(N, 2w[1], Rvals)
        for i in 2:n
            x = t[i] * s + 1
            y = sqrt(1 - x^2)
            ψ = atan(y, x)
            Rvals = R[ψ, blocks]
            @. P[:, 2(i-1)] = w[i] * Rvals
            _fix_broadcast!(N, w[i], Rvals)
            Rvals = R[-ψ, blocks] # TODO: This could be easily computed from R above by noting that the p's are even and the q's are odd
            @. P[:, 2i-1] = w[i] * Rvals
            _fix_broadcast!(N, w[i], Rvals)
        end
        for p in eachcol(P)
            @. p /= N
        end
        return ArcPlan(P, n, dims)
    end
end
function plan_transform(R::SemiclassicalJacobiArc, (n, m)::Tuple{Int,Int}, dims::Int)
    @assert dims == 1
    P = plan_transform(R, (n,), dims)
    return ArcPlan(P.X, (n, m), dims)
end

function getindex(f::ApplyQuasiVector{T,typeof(*),<:Tuple{SemiclassicalJacobiArc,AbstractVector}}, θ::Number)::T where {T}
    R, c = arguments(f)
    N = blockcolsupport(c)
    return dot(view(c, N), R[θ::T, N])
end