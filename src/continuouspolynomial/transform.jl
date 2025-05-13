function repeatgrid(axes, ref_grid, points)
    ret = Matrix{eltype(ref_grid)}(undef, length(ref_grid), length(points) - 1)
    for j in 1:(length(points)-1)
        ret[:, j] .= affine(axes, points[j] .. points[j+1])[ref_grid]
    end
    return ret
end

function grid(P::PeriodicContinuousPolynomial{0,T}, N::Block{1}) where {T}
    L = Legendre{T}()
    ref_grid = grid(L, Int(N))
    return repeatgrid(axes(L, 1), ref_grid, P.points)
end
function grid(P::PeriodicContinuousPolynomial{1,T}, N::Block{1}) where {T}
    U = Ultraspherical{T}(-one(T) / 2)
    ref_grid = grid(U, Int(N))
    return repeatgrid(axes(U, 1), ref_grid, P.points)
end
grid(P::PeriodicContinuousPolynomial, n::Integer) = grid(P, findblock(axes(P, 2), n))

struct ApplyPlan{T,FF,FAC<:Plan{T},Args} <: Plan{T}
    f::FF
    F::FAC
    args::Args
end

*(P::ApplyPlan, f::AbstractArray) = P.f(P.F * f, P.args...)

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

function \(F::ApplyPlan{<:Any,typeof(to_blockvec)}, X::AbstractArray)
    lhs = F.F
    rhs = to_mat(X, only(F.args))
    return lhs \ rhs
end

function plan_transform(P::PeriodicContinuousPolynomial{0,T}, Ns::NTuple{N,Block{1}}, dims=ntuple(identity, Val(N))) where {N,T}
    @assert dims == 1:N || dims == ntuple(identity, Val(N)) || (N == dims == 1)
    F = plan_transform(Legendre{T}(), (Int.(Ns)..., length(P.points) - 1), 1)
    return ApplyPlan(to_blockvec, F, (dims,))
end

checkpoints(P::PeriodicContinuousPolynomial) = checkpoints(P.points[1] .. P.points[end])

function adaptivetransform_ldiv(Q::PeriodicContinuousPolynomial{1,V}, f::AbstractQuasiVector) where {V}
    T = promote_type(V, eltype(f))
    C₀ = PeriodicContinuousPolynomial{0,V}(Q)
    M = length(Q.points) - 1

    c = C₀ \ f # Piecewise Legendre transform
    c̃ = paddeddata(c)
    N = max(2, length(c̃) ÷ M)
    P = Legendre{T}()
    W = Ultraspherical{T}(-one(T) / 2)

    R̃ = [[T[1 1; -1 1] / 2; Zeros{T}(∞, 2)] (P\W)[:, 3:end]]

    #=
    The first two rows of v store the coefficients for the hat functions in the form

        φ1[2] φ2[2] φ3[2] ... φn-1[2] φn[2]
        φ2[1] φ3[1] φ4[1] ... φn[1]   φ1[1]

    We assume that the coefficients for the hat functions on either side are the same.
    =#
    dat = R̃[1:N, 1:N] \ reshape(pad(c̃, M * N), M, N)'
    cfs = T[]
    if size(dat, 1) ≥ 1
        for j in 1:M 
            c1 = dat[1, j]
            c2 = dat[2, mod1(j-1, M)]
            push!(cfs, (c1 + c2) / 2)
            #=
            For non-periodic functions, note that dat[1, 1] and data[2, M]
            will not be the same.
            =#
        end
    end
    return pad(append!(cfs, vec(dat[3:end, :]')), axes(Q, 2))
end

function getindex(f::ApplyQuasiVector{T,typeof(*),<:Tuple{PeriodicContinuousPolynomial,AbstractVector}}, xy::StaticVector{2})::T where {T}
    R, c = arguments(f)
    N = blockcolsupport(c)
    return dot(c[N], R[xy, N])' # ' so that this forms the sum of the individual products instead of a complex inner product when c[N] is complex, noting that R[...] is real.
end