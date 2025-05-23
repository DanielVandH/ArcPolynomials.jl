get_element_grid(ref_grid::Vector{<:Number}, element) = ref_grid
get_element_grid(ref_grid::Vector{<:Vector{<:Number}}, element) = ref_grid[element]

function repeatgrid(axes, ref_grid, points)
    ret = Matrix{eltype(get_element_grid(ref_grid, 1))}(undef, length(get_element_grid(ref_grid, 1)), length(points) - 1)
    for j in 1:(length(points)-1)
        shift = (points[j] + points[j+1]) / 2
        ret[:, j] .= map(get_element_grid(ref_grid, j)) do θ
            θ += shift
            return θ
        end
    end
    return ret
end

function grid(P::PiecewiseArcPolynomial{0,T}, N::Block{1}) where {T}
    L = P.P0
    if has_equal_spacing(P)
        ref_grid = grid(L, Int(N))
    else
        ref_grid = grid.(L, Int(N))
    end
    return repeatgrid(axes(L, 1), ref_grid, P.points)
end
function grid(P::PiecewiseArcPolynomial{1,T}, N::Block{1}) where {T}
    U = P.P
    ref_grid = grid(U, Int(N))
    return repeatgrid(axes(U, 1), ref_grid, P.points)
end
grid(P::PiecewiseArcPolynomial, n::Integer) = grid(P, findblock(axes(P, 2), n))

struct ApplyPlan{T,FF,FAC,Args} <: Plan{T}
    f::FF
    F::FAC
    args::Args
    ApplyPlan{T}(f::FF, F::FAC, args::Args) where {T,FF,FAC,Args} = new{T,FF,FAC,Args}(f, F, args)
end

function *(P::ApplyPlan{T,FF}, f::AbstractArray) where {T,FF}
    if P.F isa ArcPlan
        return P.f(P.F * f, P.args...)
    else
        mul = zeros(T, size(f))
        for j in eachindex(P.F)
            mul[:, j] = P.F[j] * view(f, :, j)
        end
        return P.f(mul, P.args...)
    end
end

function to_blockvec(X::AbstractMatrix, dims=1)
    # Converts a matrix into a blocked-vector, where each block is
    # a row from the matrix
    @assert dims == 1 || dims == (1,) || dims == 1:1
    return BlockVec(transpose(X))
end

function to_mat(X::BlockVec, dims=1) # inverse of to_blockvec
    @assert dims == 1 || dims == (1,) || dims == 1:1
    return transpose(only(X.args))
end

function \(F::ApplyPlan{T,typeof(to_blockvec)}, X::AbstractArray) where {T}
    rhs = to_mat(X, only(F.args))
    if F.F isa ArcPlan
        return F.F \ rhs
    else
        mat = zeros(T, size(rhs))
        for j in eachindex(F.F)
            mat[:, j] = F.F[j] \ view(rhs, :, j)
        end
        return mat
    end
end

function plan_transform(P::PiecewiseArcPolynomial{0,T}, Ns::NTuple{N,Block{1}}, dims=ntuple(identity, Val(N))) where {N,T}
    @assert dims == 1:N || dims == ntuple(identity, Val(N)) || (N == dims == 1)
    if has_equal_spacing(P)
        F = plan_transform(P.P0, (Int.(Ns)..., length(P.points) - 1), 1)
    else
        F = [plan_transform(get_P0(P, i), (Int.(Ns)..., 1), 1) for i in 1:length(P.points)-1]
    end
    return ApplyPlan{T}(to_blockvec, F, (dims,))
end

function adaptivetransform_ldiv(Q::PiecewiseArcPolynomial{1,T}, f::AbstractQuasiVector) where {T}
    C₀ = PiecewiseArcPolynomial{0}(Q)
    M = length(Q.points) - 1

    c = C₀ \ f
    c̃ = paddeddata(c)
    N = max(2, length(c̃) ÷ M)
    if has_equal_spacing(Q)
        P = Q.P0
        W = Q.P
        h = Q.P0.h
        scale = 1 / sqrt(1 - h^2)
        R̃ = Hcat(Vcat(T[1 1; -scale scale] / 2, Zeros{T}(∞, 2)), (P\W)[:, 3:end]) # Avoiding an annoying ambiguity with [[T[1 1; -1 1] / 2; Zeros{T}(∞, 2)] (P\W)[:, 4:end]]
        dat = R̃[1:N, 1:N] \ reshape(pad(c̃, M * N), M, N)'
    else
        dat = zeros(T, N, M)
        coeffs = reshape(pad(c̃, M * N), M, N)' 
        for j in 1:M
            P = get_P0(Q, j)
            W = get_P(Q, j)
            h = P.h
            scale = 1 / sqrt(1 - h^2)
            R̃ = Hcat(Vcat(T[1 1; -scale scale] / 2, Zeros{T}(∞, 2)), (P\W)[:, 3:end]) # Avoiding an annoying ambiguity with [[T[1 1; -1 1] / 2; Zeros{T}(∞, 2)] (P\W)[:, 4:end]]
            dat[:, j] = R̃[1:N, 1:N] \ view(coeffs, :, j)
        end
    end
    cfs = T[]
    if size(dat, 1) ≥ 1
        for j in 1:M
            c1 = dat[1, j]
            c2 = dat[2, mod1(j - 1, M)]
            push!(cfs, (c1 + c2) / 2)
            #=
            For non-periodic functions, note that dat[1, 1] and data[2, M]
            will not be the same.
            =#
        end
    end
    return pad(append!(cfs, vec(dat[3:end, :]')), axes(Q, 2))
end

function getindex(f::ApplyQuasiVector{T,typeof(*),<:Tuple{PiecewiseArcPolynomial,AbstractVector}}, θ::Number)::T where {T}
    R, c = arguments(f)
    N = blockcolsupport(c)
    return dot(c[N], R[θ, N])' # ' so that this forms the sum of the individual products instead of a complex inner product when c[N] is complex, noting that R[...] is real.
end