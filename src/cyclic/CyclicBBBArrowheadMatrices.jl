module CyclicBBBArrowheadMatrices

import LazyArrays: LazyArrays, LazyMatrix, Vcat, AbstractLazyLayout, LazyArrayStyle
import LinearAlgebra: LinearAlgebra, mul!, Symmetric, cholesky, cholesky!, cholcopy, choltype, BlasInt, ishermitian, Cholesky, PosDefException, checksquare, copymutable_oftype, Hermitian, UpperOrLowerTriangular, Diagonal, SymTridiagonal, eigvals, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular, checkpositivedefinite, Adjoint, Transpose, parent
import Base: similar, broadcasted, copyto!, setindex!, size, axes, getindex, +, -, *, /, ==, \, OneTo, oneto, replace_in_print_matrix, copy, diff, getproperty, adjoint, transpose, tail, _sum, inv, show, summary, replace_with_centered_mark
import Base.Broadcast: materialize!, BroadcastStyle, AbstractArrayStyle, Broadcasted
import BlockArrays: blockedrange, blocksizes, blockrowstart, AbstractBlockedUnitRange, blockrowstop, blocklasts, blocksize, mortar, blockaxes, BlockSlice, BlockRange, block, _show_typeof, BlockIndex, blockindex, Block, findblockindex, AbstractBlockMatrix, AbstractBlockLayout, BlockedArray
import FillArrays: Fill, AbstractFill, Zeros
import ..CyclicBandedMatrices: CyclicBandedMatrix, _bpbandwidths, diagonal_subrmul!, diagonal_rinv!, mat_sub_MMᵀ!
import BandedMatrices: bandwidths, BandedMatrix
import ArrayLayouts: ArrayLayouts, CNoPivot, cholesky_layout, cholesky!_layout, layout_getindex, MatLdivVec, MatMulMatAdd, muladd!, LayoutVector, AbstractColumnMajor, MatMulVecAdd, _fill_lmul!, SymmetricLayout, layout_replace_in_print_matrix, HermitianLayout, DiagonalLayout, TriangularLayout, symmetriclayout, MemoryLayout, sublayout, sub_materialize, AbstractStridedLayout, triangulardata, AbstractBandedLayout
import InfiniteArrays: OneToInf
import BlockBandedMatrices: AbstractBandedBlockBandedMatrix, AbstractBandedBlockBandedLayout, blockbandwidths, subblockbandwidths
import MatrixFactorizations: ReverseCholesky, reversecholesky, reversecholesky!, reversecholesky_layout, reversecholesky_layout!
import LazyBandedMatrices: AbstractLazyBandedBlockBandedLayout
import SparseArrays: sparse
const _LazyArraysBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBandedMatricesExt)
const _LazyArraysBlockBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBlockBandedMatricesExt)
const BandedLazyLayouts = _LazyArraysBandedMatricesExt.BandedLazyLayouts
const AbstractLazyBlockBandedLayout = _LazyArraysBlockBandedMatricesExt.AbstractLazyBlockBandedLayout

export CyclicBBBArrowheadMatrix, principal_submatrix

struct InterlacedMatrix{T,M<:AbstractVector,I} <: AbstractBandedBlockBandedMatrix{T}
    D::M
    n::I
    ℓ::Int
    u::Int
    function InterlacedMatrix(D::AbstractVector)
        T = eltype(D[1])
        ℓ, u = bandwidths(D[1])
        n = size(D[1], 1)
        @assert all(D -> size(D) == (n, n), D)
        @assert all(D -> bandwidths(D) == (ℓ, u), D)
        return new{T,typeof(D),typeof(n)}(D, n, ℓ, u)
    end
end
function layout_getindex(A::InterlacedMatrix, I::Block{2})
    # Could use sub_materialize for this but I don't know the correct typing to guarantee that it only gets called for Block(i, j) and not e.g. Block.(1:10, 1:10)
    arr = view(A, I)
    return Diagonal(arr)
end
blockbandwidths(A::InterlacedMatrix) = (A.ℓ, A.u)
subblockbandwidths(A::InterlacedMatrix) = (0, 0)
function getindex(A::InterlacedMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where {T}
    K, k = block(Kk), blockindex(Kk)
    J, j = block(Jj), blockindex(Jj)
    if k ≠ j
        return zero(eltype(A))
    else
        return A.D[k][Int(K), Int(J)]
    end
end
function getindex(A::InterlacedMatrix{T}, k::Int, j::Int)::T where {T}
    ax, bx = axes(A)
    A[findblockindex(ax, k), findblockindex(bx, j)]
end
function setindex!(A::InterlacedMatrix, v, Kk::BlockIndex{1}, Jj::BlockIndex{1})
    K, k = block(Kk), blockindex(Kk)
    J, j = block(Jj), blockindex(Jj)
    if k == j
        A.D[k][Int(K), Int(J)] = v
    elseif !iszero(v)
        throw(ArgumentError("Cannot set off-diagonal elements in a block to a non-zero value"))
    end
    return v
end
function setindex!(A::InterlacedMatrix, v, k::Int, j::Int)
    ax, bx = axes(A)
    setindex!(A, v, findblockindex(ax, k), findblockindex(bx, j))
end
axes(A::InterlacedMatrix) = (blockedrange(Fill(length(A.D), A.n)), blockedrange(Fill(length(A.D), A.n)))
function similar(A::InterlacedMatrix, ::Type{T}=eltype(A), dims::Tuple{Int,Int}=size(A)) where {T}
    dims == size(A) || throw(ArgumentError("Cannot change size of InterlacedMatrix"))
    D = similar.(A.D, T)
    return InterlacedMatrix(D)
end

for adj in (:adjoint, :transpose)
    @eval $adj(A::InterlacedMatrix) = InterlacedMatrix(map($adj, A.D))
end

function copy(A::InterlacedMatrix)
    return InterlacedMatrix(map(copy, A.D))
end
function copyto!(dest::InterlacedMatrix, src::InterlacedMatrix)
    for (d, s) in zip(dest.D, src.D)
        copyto!(d, s)
    end
    return dest
end

cholcopy(A::Symmetric{<:Real,<:InterlacedMatrix}) = copyto!(similar(A, choltype(A)), A)
cholesky(A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = cholesky_layout(MemoryLayout(A), axes(A), A, v; check)
cholesky!(A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = cholesky!_layout(MemoryLayout(A), axes(A), A, v; check)
reversecholesky(A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout(MemoryLayout(A), axes(A), A, v; check)
reversecholesky!(A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout!(MemoryLayout(A), A, A.uplo == 'U' ? UpperTriangular : LowerTriangular; check)
cholesky_layout(::AbstractBandedBlockBandedLayout, axes, A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = cholesky!(cholcopy(A), v; check)
reversecholesky_layout(::AbstractBandedBlockBandedLayout, axes, A::Symmetric{<:Real,<:InterlacedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky!(cholcopy(A), v; check)
function cholesky!_layout(::ML, ::NTuple{2,AbstractBlockedUnitRange}, SA::Symmetric{T,S}, piv::CNoPivot=CNoPivot(); check::Bool=true) where {ML,T<:Real,S<:InterlacedMatrix}
    A = parent(SA)
    info = 0
    for D in A.D
        Ds = Symmetric(D, SA.uplo == 'U' ? :U : :L)
        chol = cholesky!_layout(MemoryLayout(Ds), axes(Ds), Ds, piv; check)
        check && checkpositivedefinite(chol.info)
        info = info != 0 ? info : oftype(info, chol.info)
    end
    return Cholesky(UpperTriangular(A), 'U', convert(BlasInt, info))
end
function reversecholesky_layout!(::ML, SA::Symmetric{T,S}, ::Type{UpperTriangular}; check::Bool=true) where {ML,T<:Real,S<:InterlacedMatrix}
    A = parent(SA)
    info = 0
    for Ds in A.D
        _, inf = reversecholesky_layout!(MemoryLayout(Ds), Ds, SA.uplo == 'U' ? UpperTriangular : LowerTriangular)
        check && checkpositivedefinite(inf)
        info = info != 0 ? info : oftype(info, inf)
    end
    return ReverseCholesky(UpperTriangular(A), 'U', convert(BlasInt, info))
end

struct InterlacedMatrixStyle <: AbstractArrayStyle{2} end
InterlacedMatrixStyle(::Val{2}) = InterlacedMatrixStyle()
BroadcastStyle(::Type{<:InterlacedMatrix}) = InterlacedMatrixStyle()
function copy(bc::Broadcasted{<:InterlacedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix}})
    (A,) = bc.args
    return InterlacedMatrix(broadcast(bc.f, A.D))
end
function copy(bc::Broadcasted{<:InterlacedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix,Number}})
    (A, x) = bc.args
    return InterlacedMatrix(broadcast(bc.f, A.D, x))
end
function copy(bc::Broadcasted{<:InterlacedMatrixStyle,<:Any,<:Any,<:Tuple{Number,AbstractMatrix}})
    (x, A) = bc.args
    return InterlacedMatrix(broadcast(bc.f, x, A.D))
end
function copy(bc::Broadcasted{<:InterlacedMatrixStyle,<:Any,<:Any,<:Tuple{InterlacedMatrix,InterlacedMatrix}})
    (A, B) = bc.args
    return InterlacedMatrix(broadcast(bc.f, A.D, B.D))
end

"""
    CyclicBBBArrowheadMatrix(A, B, C, D)

Constructs a `CyclicBBBArrowheadMatrix`, a variant of a `BBBArrowheadMatrix` where the blocks
`A`, `B`, and `C` may be `CyclicBandedMatrices` instead of `BandedMatrices`.
"""
struct CyclicBBBArrowheadMatrix{T,AA,BB,CC,DD<:InterlacedMatrix} <: AbstractBandedBlockBandedMatrix{T}
    A::AA # (1, 1) block
    B::BB # First row blocks
    C::CC # First column blocks
    D::DD # Interlaced diagonals 
end
function CyclicBBBArrowheadMatrix{T}(A::AbstractMatrix, B, C, D) where {T}
    Dlace = InterlacedMatrix(D)
    return CyclicBBBArrowheadMatrix{T,typeof(A),typeof(B),typeof(C),typeof(Dlace)}(A, B, C, Dlace)
end

_show_typeof(io::IO, B::CyclicBBBArrowheadMatrix{T}) where {T} = print(io, "CyclicBBBArrowheadMatrix{$T}")

const CyclicArrowheadMatrices = Union{
    CyclicBBBArrowheadMatrix,
    Symmetric{<:Any,<:CyclicBBBArrowheadMatrix},
    Hermitian{<:Any,<:CyclicBBBArrowheadMatrix},
    UpperOrLowerTriangular{<:Any,<:CyclicBBBArrowheadMatrix},
}

function axes(L::CyclicBBBArrowheadMatrix)
    ξ, n = size(L.A)
    m = length(L.D.D)
    μ, ν = size(L.D.D[1])
    blockedrange(Vcat(ξ, Fill(m, μ))), blockedrange(Vcat(n, Fill(m, ν)))
end

copy(A::CyclicBBBArrowheadMatrix) = CyclicBBBArrowheadMatrix(copy(A.A), map(copy, A.B), map(copy, A.C), copy(A.D).D)

function blockbandwidths(A::CyclicBBBArrowheadMatrix)
    ℓ, u = bandwidths(A.D.D[1])
    return max(ℓ, length(A.C)), max(u, length(A.B))
end

function getindex(L::CyclicBBBArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where {T}
    K, k = block(Kk), blockindex(Kk)
    J, j = block(Jj), blockindex(Jj)
    J == K == Block(1) && return L.A[k, j]
    if K == Block(1)
        return Int(J) - 1 ≤ length(L.B) ? L.B[Int(J)-1][k, j] : zero(T)
    end
    if J == Block(1)
        return Int(K) - 1 ≤ length(L.C) ? L.C[Int(K)-1][k, j] : zero(T)
    end
    k ≠ j && return zero(T)
    return L.D[BlockIndex(K .- 1, k), BlockIndex(J .- 1, j)]
end
function getindex(L::CyclicBBBArrowheadMatrix{T}, k::Int, j::Int)::T where {T}
    ax, bx = axes(L)
    return L[findblockindex(ax, k), findblockindex(bx, j)]
end

@inline _bandwidths(A) = bandwidths(A)
@inline _bandwidths(A::CyclicBandedMatrix) = _bpbandwidths(A)
@inline _bandwidths(A::Union{<:Adjoint{<:Any,<:CyclicBandedMatrix},<:Transpose{<:Any,<:CyclicBandedMatrix}}) = reverse(_bandwidths(parent(A)))
function _check_args(A, B, C, D)
    ξ, n = size(A)
    m = length(D)
    μ, v = size(D[1])

    λ, μ = _bandwidths(A)
    @assert -1 ≤ λ ≤ 1 && -1 ≤ μ ≤ 1 lazy"Provided sub-bandwidths for A, $λ, $μ, are not in the set (-1, 0, 1)."

    # We restrict the bandwidths of B and C so that 
    #   - The reverse Cholesky algorithm can be simplified,
    #   - and so that MM' has a cyclic-banded structure with unit corner bandwidths.
    foreach(B) do op
        @assert size(op) == (ξ, m)
        λ, μ = _bandwidths(op)
        @assert (max(λ, 0), max(μ, 0)) == (1, 0) || (max(λ, 0), max(μ, 0)) == (0, 1) || (max(λ, 0), max(μ, 0)) == (0, 0) lazy"Each block of B must be a bidiagonal matrix, but got sub-bandwidths $(λ, μ)."
    end
    foreach(C) do op
        @assert size(op) == (m, n)
        λ, μ = _bandwidths(op)
        @assert (max(λ, 0), max(μ, 0)) == (1, 0) || (max(λ, 0), max(μ, 0)) == (0, 1) || (max(λ, 0), max(μ, 0)) == (0, 0) lazy"Each block of C must be a bidiagonal matrix, but got sub-bandwidths $(λ, μ)."
    end

    ℓ, u = _bandwidths(D[1])
    foreach(D) do op
        @assert _bandwidths(op) == (ℓ, u)
    end
    return nothing
end
function CyclicBBBArrowheadMatrix(A, B, C, D)
    _check_args(A, B, C, D)
    T = promote_type(
        eltype(A),
        mapreduce(eltype, promote_type, B; init=eltype(A)),
        mapreduce(eltype, promote_type, C; init=eltype(A)),
        mapreduce(eltype, promote_type, D; init=eltype(A))
    )
    return CyclicBBBArrowheadMatrix{T}(A, B, C, D)
end

for adj in (:adjoint, :transpose)
    @eval $adj(A::CyclicBBBArrowheadMatrix{T}) where {T} = CyclicBBBArrowheadMatrix{T}($adj(A.A), map($adj, A.C), map($adj, A.B), $adj(A.D).D)
end

struct CyclicArrowheadLayout <: AbstractBandedBlockBandedLayout end
struct LazyCyclicArrowheadLayout <: AbstractLazyBandedBlockBandedLayout end
const CyclicArrowheadLayouts = Union{
    CyclicArrowheadLayout,LazyCyclicArrowheadLayout,
    SymmetricLayout{CyclicArrowheadLayout},SymmetricLayout{LazyCyclicArrowheadLayout},
    HermitianLayout{CyclicArrowheadLayout},HermitianLayout{LazyCyclicArrowheadLayout},
    TriangularLayout{'U','N',CyclicArrowheadLayout},TriangularLayout{'L','N',CyclicArrowheadLayout},
    TriangularLayout{'U','U',CyclicArrowheadLayout},TriangularLayout{'L','U',CyclicArrowheadLayout},
    TriangularLayout{'U','N',LazyCyclicArrowheadLayout},TriangularLayout{'L','N',LazyCyclicArrowheadLayout},
    TriangularLayout{'U','U',LazyCyclicArrowheadLayout},TriangularLayout{'L','U',LazyCyclicArrowheadLayout},
}
cyclicarrowheadlayout(_) = CyclicArrowheadLayout()
cyclicarrowheadlayout(::BandedLazyLayouts) = LazyCyclicArrowheadLayout()
cyclicarrowheadlayout(::DiagonalLayout{<:AbstractLazyLayout}) = LazyCyclicArrowheadLayout()
symmetriclayout(lay::CyclicArrowheadLayouts) = SymmetricLayout{typeof(lay)}()

MemoryLayout(::Type{<:CyclicBBBArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:InterlacedMatrix{<:Any,<:AbstractVector{D}}}}) where {D} = cyclicarrowheadlayout(MemoryLayout(D))
MemoryLayout(::Type{<:CyclicBBBArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:InterlacedMatrix{<:Any,<:AbstractVector{<:BandedMatrix{<:Any, <:AbstractFill{<:Any, 2, <:Tuple{<:OneTo, <:OneToInf}}}}}}}) = LazyCyclicArrowheadLayout()
MemoryLayout(::Type{<:CyclicBBBArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:InterlacedMatrix{<:Any,<:AbstractVector{<:Diagonal{<:Any,<:AbstractFill{<:Any,1,<:Tuple{OneToInf}}}}}}}) = LazyCyclicArrowheadLayout()

function sublayout(::CyclicArrowheadLayouts, ::Type{<:NTuple{2,BlockSlice{<:BlockRange{1,Tuple{OneTo{Int}}}}}})
    throw("...")
    return CyclicArrowheadLayout()
end

function sub_materialize(::CyclicArrowheadLayout, V::AbstractMatrix)
    throw("...")
    KR, JR = parentindices(V)
    P = parent(V)
    M, N = KR.block[end], JR.block[end]
    return CyclicBBBArrowheadMatrix(
        P.A,
        P.B,
        P.C,
        layout_getindex.(
            P.D,
            (oneto(Int(M) - 1),),
            (oneto(Int(N) - 1),)
        )
    )
end

function layout_replace_in_print_matrix(::CyclicArrowheadLayouts, A::CyclicBBBArrowheadMatrix, k, j, s)
    bi = findblockindex.(axes(A), (k, j))
    K, J = block.(bi)
    k, j = blockindex.(bi)
    K == J == Block(1) && return replace_in_print_matrix(A.A, k, j, s)
    if K == Block(1)
        return Int(J) - 1 ≤ length(A.B) ? replace_in_print_matrix(A.B[Int(J)-1], k, j, s) : replace_with_centered_mark(s)
    end
    if J == Block(1)
        return Int(K) - 1 ≤ length(A.C) ? replace_in_print_matrix(A.C[Int(K)-1], k, j, s) : replace_with_centered_mark(s)
    end
    k ≠ j && return replace_with_centered_mark(s)
    return replace_in_print_matrix(A.D.D[k], Int(K) - 1, Int(J) - 1, s)
end

## MUL
function materialize!(M::MatMulVecAdd{<:CyclicArrowheadLayouts,<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, A, x_in, β, y_in = M.α, M.A, M.B, M.β, M.C
    x = BlockedArray(x_in, (axes(A, 2),))
    y = BlockedArray(y_in, (axes(A, 1),))
    m, n = size(A.A)

    _fill_lmul!(β, y)

    mul!(view(y, Block(1)), A.A, view(x, Block(1)), α, one(α))
    for k = 1:length(A.B)
        mul!(view(y, Block(1)), A.B[k], view(x, Block(k + 1)), α, one(α))
    end
    for k = 1:length(A.C)
        mul!(view(y, Block(k + 1)), A.C[k], view(x, Block(1)), α, one(α))
    end

    d = length(A.D.D)
    for k = 1:d
        mul!(view(y, m+k:d:size(y, 1)), A.D.D[k], view(x, n+k:d:size(x, 1)), α, one(α))
    end
    return y_in
end

function materialize!(M::MatMulMatAdd{<:CyclicArrowheadLayouts,<:AbstractColumnMajor,<:AbstractColumnMajor})
    α, A, X_in, β, Y_in = M.α, M.A, M.B, M.β, M.C
    X = BlockedArray(X_in, (axes(A, 2), axes(X_in, 2)))
    Y = BlockedArray(Y_in, (axes(A, 1), axes(X_in, 2)))
    m, n = size(A.A)

    _fill_lmul!(β, Y)
    for J = blockaxes(X, 2)
        mul!(view(Y, Block(1), J), A.A, view(X, Block(1), J), α, one(α))
        for k = 1:min(length(A.B), blocksize(X, 1) - 1)
            mul!(view(Y, Block(1), J), A.B[k], view(X, Block(k + 1), J), α, one(α))
        end
        for k = 1:min(length(A.C), blocksize(Y, 1) - 1)
            mul!(view(Y, Block(k + 1), J), A.C[k], view(X, Block(1), J), α, one(α))
        end
    end
    d = length(A.D.D)
    for k = 1:d
        mul!(view(Y, m+k:d:size(Y, 1), :), A.D.D[k], view(X, n+k:d:size(Y, 1), :), α, one(α))
    end
    return Y_in
end

function materialize!(M::MatMulMatAdd{<:AbstractColumnMajor,<:CyclicArrowheadLayouts,<:AbstractColumnMajor})
    α, X_in, A, β, Y_in = M.α, M.A, M.B, M.β, M.C
    X = BlockedArray(X_in, (axes(X_in, 1), axes(A, 1)))
    Y = BlockedArray(Y_in, (axes(Y_in, 1), axes(A, 2)))
    m, n = size(A.A)

    _fill_lmul!(β, Y)
    for K = blockaxes(X, 1)
        mul!(view(Y, K, Block(1)), view(X, K, Block(1)), A.A, α, one(α))
        for k = 1:length(A.C)
            mul!(view(Y, K, Block(1)), view(X, K, Block(k + 1)), A.C[k], α, one(α))
        end
        for k = 1:length(A.B)
            mul!(view(Y, K, Block(k + 1)), view(X, K, Block(1)), A.B[k], α, one(α))
        end
    end
    d = length(A.D.D)
    for k = 1:d
        mul!(view(Y, :, n+k:d:size(Y, 2)), view(X, :, m+k:d:size(Y, 2)), A.D.D[k], α, one(α))
    end
    return Y_in
end

## CHOLESKY 
function reversecholcopy(S::Symmetric{<:Any,<:CyclicBBBArrowheadMatrix})
    T = choltype(S)
    A = parent(S)
    copied = CyclicBBBArrowheadMatrix(
        copymutable_oftype(A.A, T),
        copymutable_oftype.(A.B, T),
        copymutable_oftype.(A.C, T),
        copymutable_oftype.(A.D.D, T)
    )
    return Symmetric(copied)
end

reversecholesky(A::Symmetric{<:Real,<:CyclicBBBArrowheadMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout(MemoryLayout(A), axes(A), A, v; check)
function reversecholesky(A::CyclicBBBArrowheadMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return reversecholesky(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
reversecholesky!(A::Symmetric{<:Real,<:CyclicBBBArrowheadMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout!(MemoryLayout(A), A, A.uplo == 'U' ? UpperTriangular : LowerTriangular; check)
function reversecholesky!(A::CyclicBBBArrowheadMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return reversecholesky!(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
reversecholesky_layout(::CyclicArrowheadLayouts, axes, A::Symmetric{<:Real,<:CyclicBBBArrowheadMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky!(reversecholcopy(A), v; check)

function reversecholesky_layout!(::SymmetricLayout{ML}, SA::Symmetric{T,S}, ::Type{UpperTriangular}; check::Bool=true) where {ML<:CyclicArrowheadLayouts,T<:Real,S<:CyclicBBBArrowheadMatrix}
    K = parent(SA)
    A, B, D = K.A, K.B, K.D
    chol = reversecholesky!(Symmetric(D); check)
    info = chol.info
    check && checkpositivedefinite(info)
    ℓ = length(B)
    for b in ℓ:-1:1
        for j in (b+1):ℓ
            diagonal_subrmul!(B[b], @view(D[Block(b, j)]), B[j])
        end
        diagonal_rinv!(B[b], @view(D[Block(b, b)]))
        mat_sub_MMᵀ!(A, B[b])
    end
    chol2 = reversecholesky!(Symmetric(A); check)
    info2 = chol2.info
    check && checkpositivedefinite(info2)
    Kc = UpperTriangular(CyclicBBBArrowheadMatrix{T}(chol2.factors, B, (), D.D))
    return ReverseCholesky(Kc, 'U', convert(BlasInt, max(info, info2)))
end

## OPERATIONS 
tupleop(::typeof(+), ::Tuple{}, ::Tuple{}) = ()
tupleop(::typeof(-), ::Tuple{}, ::Tuple{}) = ()
tupleop(::typeof(+), A::Tuple, B::Tuple{}) = A
tupleop(::typeof(-), A::Tuple, B::Tuple{}) = A
tupleop(::typeof(+), A::Tuple{}, B::Tuple) = B
tupleop(::typeof(-), A::Tuple{}, B::Tuple) = map(-, B)
tupleop(op, A::Tuple, B::Tuple) = (op(first(A), first(B)), tupleop(op, tail(A), tail(B))...)

for op in (:+, :-)
    @eval $op(A::CyclicBBBArrowheadMatrix, B::CyclicBBBArrowheadMatrix) = CyclicBBBArrowheadMatrix($op(A.A, B.A), tupleop($op, A.B, B.B), tupleop($op, A.C, B.C), $op(A.D, B.D).D)
end
-(A::CyclicBBBArrowheadMatrix) = CyclicBBBArrowheadMatrix(-A.A, map(-, A.B), map(-, A.C), -A.D.D)

for op in (:*, :\)
    @eval $op(c::Number, A::CyclicBBBArrowheadMatrix) = CyclicBBBArrowheadMatrix($op(c, A.A), broadcast($op, c, A.B), broadcast($op, c, A.C), $op(c, A.D).D)
end

for op in (:*, :/)
    @eval $op(A::CyclicBBBArrowheadMatrix, c::Number) = CyclicBBBArrowheadMatrix($op(A.A, c), broadcast($op, A.B, c), broadcast($op, A.C, c), $op(A.D, c).D)
end

BroadcastStyle(::Type{<:CyclicBBBArrowheadMatrix}) = LazyArrayStyle{2}()
broadcasted(::LazyArrayStyle, ::typeof(*), c::Number, A::CyclicBBBArrowheadMatrix) = c * A
broadcasted(::LazyArrayStyle, ::typeof(*), A::CyclicBBBArrowheadMatrix, c::Number) = A * c
broadcasted(::LazyArrayStyle, ::typeof(/), A::CyclicBBBArrowheadMatrix, c::Number) = A / c

## TRIANGULAR 
for (UNIT, Tri) in (('U', UnitUpperTriangular), ('N', UpperTriangular))
    @eval @inline function materialize!(M::MatLdivVec{<:TriangularLayout{'U',$UNIT,CyclicArrowheadLayout},<:AbstractStridedLayout})
        U, dest = M.A, M.B
        T = eltype(dest)
        P = triangulardata(U)

        ξ, n = size(P.A)
        A, B, D = P.A, P.B, P.D.D
        m = length(D)

        for k = 1:m
            ArrayLayouts.ldiv!($Tri(D[k]), view(dest, n+k:m:length(dest)))
        end

        N = blocksize(P, 1)

        # impose block structure
        b = BlockedArray(dest, (axes(P, 1),))
        b̃_1 = view(b, Block(1))

        for K = 1:min(N - 1, length(B))
            muladd!(-one(T), B[K], view(b, Block(K + 1)), one(T), b̃_1)
        end

        ArrayLayouts.ldiv!($Tri(A), b̃_1)

        dest
    end
end
for (UNIT, Tri) in (('U', UnitLowerTriangular), ('N', LowerTriangular))
    @eval @inline function materialize!(M::MatLdivVec{<:TriangularLayout{'L',$UNIT,CyclicArrowheadLayout},<:AbstractStridedLayout})
        U, dest = M.A, M.B
        T = eltype(dest)

        P = triangulardata(U)
        ξ, n = size(P.A)
        A, C, D = P.A, P.C, P.D.D
        m = length(D)

        # impose block structure
        b = BlockedArray(dest, (axes(P, 1),))
        b̃_1 = view(b, Block(1))
        ArrayLayouts.ldiv!($Tri(A), b̃_1)

        N = blocksize(P, 1)
        for K = 1:min(N - 1, length(C))
            muladd!(-one(T), C[K], b̃_1, one(T), view(b, Block(K + 1)))
        end


        for k = 1:length(D)
            ArrayLayouts.ldiv!($Tri(D[k]), view(dest, n+k:m:length(dest)))
        end

        dest
    end
end


for Tri in (:UpperTriangular, :UnitUpperTriangular)
    @eval function getproperty(F::$Tri{<:Any,<:CyclicBBBArrowheadMatrix}, d::Symbol)
        P = getfield(F, :data)
        if d == :A
            return $Tri(P.A)
        elseif d == :B
            return P.B
        elseif d == :C
            return ()
        elseif d == :D
            return $Tri.(P.D)
        else
            return getfield(F, d)
        end
    end
end

for Tri in (:LowerTriangular, :UnitLowerTriangular)
    @eval function getproperty(F::$Tri{<:Any,<:CyclicBBBArrowheadMatrix}, d::Symbol)
        P = getfield(F, :data)
        if d == :A
            return $Tri(P.A)
        elseif d == :B
            return ()
        elseif d == :C
            return P.C
        elseif d == :D
            return $Tri.(P.D)
        else
            getfield(F, d)
        end
    end
end

function to_interlace(D)
    # Given a block matrix D, returns the interlaced formulation for the diagonals, compatible 
    # with the layout necessary for a CyclicBBBArrowheadMatrix
    ax1, ax2 = axes(D)
    @assert ax1 === ax2
    n = step(blocklasts(ax1))
    λ, u = blockbandwidths(D)
    nb = blocksize(D, 1)
    Dint = [BandedMatrix{eltype(D)}(undef, (nb, nb), (λ, u)) for _ in 1:n]
    for j in blockaxes(D, 2)
        for i in blockrowstart(D, j):blockrowstop(D, j)
            Dij = D[i, j]
            for k in 1:n
                Dint[k][Int(i), Int(j)] = Dij[k, k]
            end
        end
    end
    return Dint
end

function principal_submatrix(M, K::Block{1})
    # Returns the principal submatrix of M up to block K, returning 
    # the result as another CyclicBBBArrowheadMatrix
    k = Int(K)
    to_cbm(A) = begin
        a1n, an1 = A[1, end], A[end, 1]
        A[1, end] = A[end, 1] = zero(eltype(A))
        return CyclicBandedMatrix(BandedMatrix(sparse(A)), a1n, an1) # BandedMatrix(sparse(A)) because M[Block(1, 1)] creates a matrix with excess bandwidth 
    end
    λ, u = blockbandwidths(M)
    A = to_cbm(M[Block(1), Block(1)])
    B = [to_cbm(M[Block(1), Block(i)]) for i in 2:min(u + 1, k)]
    C = [to_cbm(M[Block(i), Block(1)]) for i in 2:min(λ + 1, k)]

    diagonal_block(i, j) = begin
        if !((j < i - λ) || (j > i + u))
            return M[Block(i, j)]
        else
            return BandedMatrix(Zeros{eltype(M)}(blocksizes(M, 1)[i], blocksizes(M, 2)[j]))
        end
    end
    MD = mortar(diagonal_block.(2:k, (2:k)'), blocksizes(M,1)[2:k], blocksizes(M,2)[2:k])
    D = to_interlace(MD)
    return CyclicBBBArrowheadMatrix(A, B, C, D)
end

## SOLVE
function ArrayLayouts.ldiv!(K::ReverseCholesky{<:Any,<:UpperTriangular{<:Any,<:CyclicBBBArrowheadMatrix}}, b::AbstractVector)
    A = K.factors
    ArrayLayouts.ldiv!(A, b)
    ArrayLayouts.ldiv!(A', b)
    b
end
LinearAlgebra.ldiv!(K::ReverseCholesky{<:Any,<:UpperTriangular{<:Any,<:CyclicBBBArrowheadMatrix}}, b::AbstractVector) = ArrayLayouts.ldiv!(K, b)
LinearAlgebra.ldiv!(K::ReverseCholesky{<:Any,<:UpperTriangular{<:Any,<:CyclicBBBArrowheadMatrix}}, b::LayoutVector) = ArrayLayouts.ldiv!(K, b) # ambiguity

end