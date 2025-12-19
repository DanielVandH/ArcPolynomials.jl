module CyclicBandedMatrices

import Base: getindex, DimensionMismatch, OneTo, setindex!, convert, size, axes, parent, similar, copy, copyto!, @propagate_inbounds, replace_with_centered_mark, require_one_based_indexing
import ArrayLayouts: copymutable_oftype_layout, CNoPivot, LayoutMatrix, LayoutVector, MemoryLayout, layout_replace_in_print_matrix, cholesky_layout, cholesky!_layout, symmetriclayout, SymmetricLayout, materialize!, Lmul, Rmul, DiagonalLayout, diagonaldata, colsupport, rowsupport
import BandedMatrices: _BandedMatrix, BandedMatrix, inbands_setindex!, AbstractBandedMatrix, bandwidth, bandeddata, bandwidths, isbanded, inbands_getindex
import LinearAlgebra: BlasInt, SingularException, norm, Symmetric, Cholesky, cholesky, cholcopy, ishermitian, choltype, PosDefException, LowerTriangular, UpperTriangular, checksquare, checkpositivedefinite, adjoint, transpose, Adjoint, Transpose
import MatrixFactorizations: ReverseCholesky, reversecholesky, reversecholesky!, reversecholesky_layout, reversecholesky_layout!
import SemiseparableMatrices: AlmostBandedMatrix
import FillArrays: OneElement, Ones, Zeros
import LazyArrays: ApplyMatrix
import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle, broadcasted

export CyclicBandedMatrix

# definition
struct CyclicBandedMatrix{T,CONTAINER} <: LayoutMatrix{T}
    data::CONTAINER
    ℓ::Int
    u::Int
    extended_ℓu::NTuple{2,Int}
    global function _CyclicBandedMatrix(data::AbstractMatrix{T}, ℓ, u, (_ℓ, _u)) where {T}
        require_one_based_indexing(data)
        n = size(data, 1)
        if n > 2 && (size(data, 1) ≠ _ℓ + _u + 1 && !(size(data, 1) == 0 && -_ℓ > _u))
            throw(ArgumentError("Data matrix must have number of rows equal to the number of bands"))
        else
            return new{T,typeof(data)}(data, ℓ, u, (_ℓ, _u))
        end
    end
end

function _maybeextend(A::BandedMatrix{T}) where {T}
    ℓ, u = bandwidths(A)
    ℓ, u = max(ℓ, 1), max(u, 1)
    n, m = size(A)
    ret = _BandedMatrix(zeros(T, max(0, ℓ + u + 1), m), n, ℓ, u)
    for j = 1:m, k = max(1, j - u):min(n, j + ℓ)
        inbands_setindex!(ret, A[k, j], k, j)
    end
    return ret # This is basically return BandedMatrix(A, (max(ℓ, 1), max(u, 1))), but with zeros instead of an UndefInitializer
end

@inline function _placecorners!(data, bdata, a, b, u)
    n = checksquare(data)
    if n == 2
        data[1, 2] = a
        data[2, 1] = b
    else
        bdata[u, 1] = a
        bdata[u+2, end] = b
    end
    return data
end

function _builddata(data::BandedMatrix{T}, a::A, b::B) where {T,A,B}
    data = _maybeextend(data)
    V = promote_type(T, A, B)
    _data = convert(BandedMatrix{V}, data)
    _a, _b = convert(V, a), convert(V, b)
    bdata = bandeddata(_data)
    ℓ, u = bandwidths(_data)
    _placecorners!(_data, bdata, _a, _b, u)
    return bdata, ℓ, u
end

function CyclicBandedMatrix(data::BandedMatrix{T}, a::A, b::B) where {T,A,B}
    ℓ, u = bandwidths(data)
    require_one_based_indexing(data)
    bdata, _ℓ, _u = _builddata(data, a, b)
    return _CyclicBandedMatrix(bdata, ℓ, u, (_ℓ, _u))
end

function CyclicBandedMatrix{T}(::UndefInitializer, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}) where {T}
    return CyclicBandedMatrix(BandedMatrix{T}(undef, nm, ab), zero(T), zero(T))
end

# bandedpart
parent(C::CyclicBandedMatrix) = C.data
_bpbandwidths(C::CyclicBandedMatrix) = (C.ℓ, C.u)
_bpbandwidths(C::Adjoint{<:Any,<:CyclicBandedMatrix}) = reverse(_bpbandwidths(parent(C)))
_bpbandwidths(C::Transpose{<:Any,<:CyclicBandedMatrix}) = reverse(_bpbandwidths(parent(C)))
_bpbandwidths(C, i::Int) = _bpbandwidths(C)[i]
_extended_bpbandwidths(C::CyclicBandedMatrix) = C.extended_ℓu
_extended_bpbandwidths(C::Adjoint{<:Any,<:CyclicBandedMatrix}) = reverse(_extended_bpbandwidths(parent(C)))
_extended_bpbandwidths(C::Transpose{<:Any,<:CyclicBandedMatrix}) = reverse(_extended_bpbandwidths(parent(C)))
_extended_bpbandwidths(C, i::Int) = _extended_bpbandwidths(C)[i]
bandedpart(C::CyclicBandedMatrix) = _BandedMatrix(copy(parent(C)), axes(C, 1), _extended_bpbandwidths(C)...) # TODO: Remove the copy. The issue is tests with Cholesky randomly failing.
bandedpart(C::Adjoint{<:Any,<:CyclicBandedMatrix}) = adjoint(bandedpart(parent(C)))
bandedpart(C::Transpose{<:Any,<:CyclicBandedMatrix}) = transpose(bandedpart(parent(C)))

function _aliased_bandedpart(C::CyclicBandedMatrix)
    _ℓ, _u = _extended_bpbandwidths(C)
    return _BandedMatrix(C.data, axes(C, 1), _ℓ, _u)
end
function _aliased_bandedpart(C::Adjoint{<:Any,<:CyclicBandedMatrix})
    _ℓ, _u = _extended_bpbandwidths(C)
    Cbp_parent = _BandedMatrix(parent(C).data, axes(parent(C), 1), reverse((_ℓ, _u))...)
    return adjoint(Cbp_parent)
end
function _aliased_bandedpart(C::Transpose{<:Any,<:CyclicBandedMatrix})
    _ℓ, _u = _extended_bpbandwidths(C)
    Cbp_parent = _BandedMatrix(parent(C).data, axes(parent(C), 1), reverse((_ℓ, _u))...)
    return transpose(Cbp_parent)
end

# axes
_ncols(C::AbstractMatrix) = size(C, 2)
_ncols(C::CyclicBandedMatrix) = _ncols(parent(C))
_raxis(C::CyclicBandedMatrix) = OneTo(_ncols(C))
axes(C::CyclicBandedMatrix) = (_raxis(C), _raxis(C))
size(C::CyclicBandedMatrix) = (_ncols(C), _ncols(C))

# layout
struct CyclicBandedLayout{ML} <: MemoryLayout end # ML is the memory layout of the banded part
_bptype(::Type{CyclicBandedMatrix{T,CONTAINER}}) where {T,CONTAINER} = BandedMatrix{T,CONTAINER,OneTo{Int}}
function MemoryLayout(::Type{C}) where {C<:CyclicBandedMatrix}
    ML = MemoryLayout(_bptype(C))
    return CyclicBandedLayout{typeof(ML)}()
end
@inline symmetriclayout(::ML) where {ML<:CyclicBandedLayout} = SymmetricLayout{ML}()
@inline MemoryLayout(::Type{Symmetric{T,S}}) where {T,S<:CyclicBandedMatrix} = symmetriclayout(MemoryLayout(S))

_colstart(C::CyclicBandedMatrix, j::Integer) = (j == firstindex(C, 2) || j == lastindex(C, 2)) ? firstindex(C, 1) : max(j - _bpbandwidths(C, 2), 1) + firstindex(C, 1) - 1
_colstop(C::CyclicBandedMatrix, j::Integer) = (j == firstindex(C, 2) || j == lastindex(C, 2)) ? lastindex(C, 1) : clamp(j + _bpbandwidths(C, 1), 0, _ncols(C)) + firstindex(C, 1) - 1
_rowstart(C::CyclicBandedMatrix, i::Integer) = (i == firstindex(C, 1) || i == lastindex(C, 1)) ? firstindex(C, 2) : max(i - _bpbandwidths(C, 1), 1) + firstindex(C, 2) - 1
_rowstop(C::CyclicBandedMatrix, i::Integer) = (i == firstindex(C, 1) || i == lastindex(C, 1)) ? lastindex(C, 2) : clamp(i + _bpbandwidths(C, 2), 0, _ncols(C)) + firstindex(C, 2) - 1
colsupport(_, C::CyclicBandedMatrix, j::Integer) = _colstart(C, j):_colstop(C, j)
rowsupport(_, C::CyclicBandedMatrix, i::Integer) = _rowstart(C, i):_rowstop(C, i)
colsupport(_, C::CyclicBandedMatrix, j) = isempty(j) ? (1:0) : _colstart(C, firstindex(C, 2) in j || lastindex(C, 2) in j ? firstindex(C, 2) : minimum(j)):_colstop(C, firstindex(C, 2) in j || lastindex(C, 2) in j ? lastindex(C, 2) : maximum(j))
rowsupport(_, C::CyclicBandedMatrix, i) = isempty(i) ? (1:0) : _rowstart(C, firstindex(C, 1) in i || lastindex(C, 1) in i ? firstindex(C, 1) : minimum(i)):_rowstop(C, firstindex(C, 1) in i || lastindex(C, 1) in i ? lastindex(C, 1) : maximum(i))

colsupport(lay, C::Adjoint{<:Any,<:CyclicBandedMatrix}, j::Integer) = rowsupport(lay, parent(C), j)
rowsupport(lay, C::Adjoint{<:Any,<:CyclicBandedMatrix}, i::Integer) = colsupport(lay, parent(C), i)
colsupport(lay, C::Adjoint{<:Any,<:CyclicBandedMatrix}, j) = rowsupport(lay, parent(C), j)
rowsupport(lay, C::Adjoint{<:Any,<:CyclicBandedMatrix}, i) = colsupport(lay, parent(C), i)

# convert/similar
function convert(::Type{CyclicBandedMatrix{V}}, C::CyclicBandedMatrix{V}) where {V}
    return C
end
function convert(::Type{CyclicBandedMatrix{V}}, C::CyclicBandedMatrix) where {V}
    return _CyclicBandedMatrix(convert(AbstractMatrix{V}, parent(C)), _bpbandwidths(C)..., _extended_bpbandwidths(C))
end
function similar(C::CyclicBandedMatrix, ::Type{T}=eltype(C), dims::Tuple{Int,Int}=size(C)) where {T}
    data = parent(C)
    m, n = dims
    m == n || throw(ArgumentError("Matrix must be square"))
    return _CyclicBandedMatrix(similar(data, T, size(data, 1), m), _bpbandwidths(C)..., _extended_bpbandwidths(C))
end

# getindex
_dataindex(u::Integer, k::Integer, j::Integer, n::Integer) = (u + k - j + 1, mod1(j, n))
@propagate_inbounds function _inbands_getindex(data::AbstractMatrix, u::Integer, k::Integer, j::Integer)
    row, col = _dataindex(u, k, j, _ncols(data))
    return data[row, col]
end
function _isinband(ℓ::Integer, u::Integer, k::Integer, j::Integer)
    return -ℓ ≤ j - k ≤ u
end
function _isincorner(n::Integer, k::Integer, j::Integer)
    return (k, j) == (1, n) || (k, j) == (n, 1)
end
function _isnonzero(n::Integer, ℓ::Integer, u::Integer, k::Integer, j::Integer) # check structural zero
    return _isinband(ℓ, u, k, j) || _isincorner(n, k, j)
end
@propagate_inbounds function _banded_getindex(data::AbstractMatrix{T}, ℓ::Integer, u::Integer, k::Integer, j::Integer, ℓr::Integer, ur::Integer) where {T}
    n = _ncols(data)
    if _isincorner(n, k, j) && n > 2
        k′, j′ = (k, j) == (1, n) ? (n, n + 1) : (n + 1, n)
        return _inbands_getindex(data, u, k′, j′)
    elseif n == 2 && (k, j) == (1, 2)
        return data[1, 2]
    elseif n == 2 && (k, j) == (2, 1)
        return data[end, 1]
    elseif _isinband(ℓr, ur, k, j)
        return _inbands_getindex(data, u, k, j)
    else
        return zero(T)
    end
end
@propagate_inbounds function getindex(C::CyclicBandedMatrix, i::Integer, j::Integer)
    n = _ncols(C)
    @boundscheck (checkindex(Bool, OneTo(n + 1), i) && checkindex(Bool, OneTo(n + 1), j)) || throw(BoundsError(C, (i, j)))
    r = _banded_getindex(parent(C), _extended_bpbandwidths(C)..., i, j, _bpbandwidths(C)...)
    return r
end

# setindex!
@propagate_inbounds function _inbands_setindex!(data::AbstractMatrix, u::Integer, v, k::Integer, j::Integer)
    n = _ncols(data)
    row, col = _dataindex(u, k, j, n)
    data[row, col] = v
end
@propagate_inbounds function _banded_setindex!(data::AbstractMatrix, ℓ::Integer, u::Integer, v, k::Integer, j::Integer, ℓr::Integer, ur::Integer)
    n = _ncols(data)
    if _isincorner(n, k, j) && n > 2
        k′, j′ = (k, j) == (1, n) ? (n, n + 1) : (n + 1, n)
        _inbands_setindex!(data, u, v, k′, j′)
    elseif n == 2 && (k, j) == (1, 2)
        data[1, 2] = v
    elseif n == 2 && (k, j) == (2, 1)
        data[end, 1] = v
    elseif _isinband(ℓr, ur, k, j)
        _inbands_setindex!(data, u, v, k, j)
    elseif !iszero(v)
        throw(ArgumentError("Cannot set a structural zero to a nonzero value"))
    else
        v
    end
end
@propagate_inbounds function setindex!(C::CyclicBandedMatrix, v, k::Integer, j::Integer)
    n = _ncols(C)
    @boundscheck (checkindex(Bool, OneTo(n + 1), k) && checkindex(Bool, OneTo(n + 1), j)) || throw(BoundsError(C, (k, j)))
    _banded_setindex!(parent(C), _extended_bpbandwidths(C)..., v, k, j, _bpbandwidths(C)...)
    C
end

# copy
@inline function copy(C::CyclicBandedMatrix)
    return _CyclicBandedMatrix(copy(parent(C)), _bpbandwidths(C)..., _extended_bpbandwidths(C))
end
@inline function copy(C::Adjoint{<:Any,<:CyclicBandedMatrix})
    return adjoint(copy(parent(C)))
end
@inline function copy(C::Transpose{<:Any,<:CyclicBandedMatrix})
    return transpose(copy(parent(C)))
end
copymutable_oftype_layout(_, A::CyclicBandedMatrix{T}, ::Type{T}) where {T} = copy(A)
copymutable_oftype_layout(_, A::Adjoint{<:Any,<:CyclicBandedMatrix{T}}, ::Type{T}) where {T} = copy(A)
copymutable_oftype_layout(_, A::Transpose{<:Any,<:CyclicBandedMatrix{T}}, ::Type{T}) where {T} = copy(A)

@inline function copyto!(dest::CyclicBandedMatrix, src::CyclicBandedMatrix)
    copyto!(parent(dest), parent(src))
    return dest
end

# show 
@inline function layout_replace_in_print_matrix(::CyclicBandedLayout, C::CyclicBandedMatrix, i, j, s)
    n, ℓ, u = _ncols(C), _bpbandwidths(C)...
    if _isnonzero(n, ℓ, u, i, j)
        return s
    else
        return replace_with_centered_mark(s)
    end
end

# mul
function copy(M::Lmul{<:DiagonalLayout,<:CyclicBandedLayout})
    D = M.A
    C = M.B
    B = bandedpart(C)
    sub = Lmul(D, B)
    X = copy(sub)
    n = _ncols(C)
    if n > 2
        a, b = C[1, end], C[end, 1]
        diag = diagonaldata(D)
        ã, b̃ = a * diag[1], b * diag[end]
    else
        ã, b̃ = X[1, end], X[end, 1]
    end
    u = _extended_bpbandwidths(C, 2)
    _placecorners!(X, bandeddata(X), ã, b̃, u)
    Y = _CyclicBandedMatrix(bandeddata(X), _bpbandwidths(C)..., _extended_bpbandwidths(C))
    return Y
end
function copy(M::Rmul{<:CyclicBandedLayout,<:DiagonalLayout})
    D = M.B
    C = M.A
    B = bandedpart(C)
    sub = Rmul(B, D)
    X = copy(sub)
    n = _ncols(C)
    if n > 2
        a, b = C[1, end], C[end, 1]
        diag = diagonaldata(D)
        ã, b̃ = a * diag[end], b * diag[1]
    else
        ã, b̃ = X[1, end], X[end, 1]
    end
    u = _extended_bpbandwidths(C, 2)
    _placecorners!(X, bandeddata(X), ã, b̃, u)
    Y = _CyclicBandedMatrix(bandeddata(X), _bpbandwidths(C)..., _extended_bpbandwidths(C))
    return Y
end

function MMᵀ(M::CyclicBandedMatrix)
    # Computes M * M' 
    ℓ, u = _bpbandwidths(M)
    α, β = M[1, end], M[end, 1]
    if (ℓ, u) == (0, 0) || size(M, 1) == 2
        B = bandedpart(M)
        BBᵀ = B * B'
        if size(M, 1) > 2
            BBᵀ[1, 1] += α^2
            BBᵀ[end, end] += β^2
        end
        bdata, _ℓ, _u = _builddata(BBᵀ, B[1, 1] * β + B[end, end] * α, B[1, 1] * β + B[end, end] * α)
        return _CyclicBandedMatrix(bdata, 0, 0, (_ℓ, _u))
    elseif (ℓ, u) == (1, 0)
        @assert iszero(β) "Cannot compute MMᵀ for a cyclic banded matrix with a non-zero lower-left corner entry."
        B = bandedpart(M)
        BBᵀ = B * B'
        BBᵀ[1, 1] += α^2
        bdata, _ℓ, _u = _builddata(BBᵀ, B[end, end] * α, B[end, end] * α)
        return _CyclicBandedMatrix(bdata, 1, 1, (_ℓ, _u))
    elseif (ℓ, u) == (0, 1)
        @assert iszero(α) "Cannot compute MMᵀ for a cyclic banded matrix with a non-zero upper-right corner entry."
        B = bandedpart(M)
        BBᵀ = B * B'
        BBᵀ[end, end] += β^2
        bdata, _ℓ, _u = _builddata(BBᵀ, B[1, 1] * β, B[1, 1] * β)
        return _CyclicBandedMatrix(bdata, 1, 1, (_ℓ, _u))
    else
        throw(ArgumentError("Cannot compute MMᵀ for a cyclic banded matrix with sub-bandwidths ($ℓ, $u)."))
    end
end

function mat_sub_MMᵀ!(A::AbstractMatrix, M::AbstractMatrix)
    # Computes A - MM', where A and M are cyclic banded matrices with the same 
    # sub-bandwidths. The result is placed into A. 
    # Only allows bidiagonal matrices.
    require_one_based_indexing(A, M)

    Abp = _aliased_bandedpart(A)
    Mbp = _aliased_bandedpart(M)

    ℓ_M, u_M = _bpbandwidths(M)
    ℓ_A, u_A = _bpbandwidths(A)
    m, n = size(A)
    m′, n′ = size(M)
    m == m′ || throw(DimensionMismatch(lazy"expect A to have $m rows, but got $m′"))
    n == n′ || throw(DimensionMismatch(lazy"expect A to have $n columns, but got $n′"))

    # Special case for 2×2 matrices: all entries are in the banded part
    if n == 2
        for j in 1:2, i in 1:2
            for k in 1:2
                Abp[i, j] -= Mbp[i, k] * Mbp[j, k]
            end
        end
        return A
    end

    # General case for n > 2
    for j in 1:n
        i_start = max(1, j - u_A)
        i_stop = min(n, j + ℓ_A)
        for i in i_start:i_stop
            k_start = max(max(1, i - ℓ_M), max(1, j - ℓ_M))
            k_stop = min(min(n, i + u_M), min(n, j + u_M))
            for k in k_start:k_stop
                Abp[i, j] -= Mbp[i, k] * Mbp[j, k]
            end
        end
    end

    m11, mnn = M[1, 1], M[end, end]
    α, β = M[1, end], M[end, 1]
    if (max(ℓ_M, 0), max(u_M, 0)) == (0, 0)
        A[1, 1] -= α^2
        A[end, end] -= β^2
        sub = α * mnn + β * m11
        A[1, end] -= sub
        A[end, 1] -= sub
    elseif (max(ℓ_M, 0), max(u_M, 0)) == (1, 0)
        @assert iszero(β) "Cannot compute MMᵀ for a cyclic banded matrix with a non-zero lower-left corner entry."
        A[1, 1] -= α^2
        A[1, end] -= mnn * α
        A[end, 1] -= mnn * α
    elseif (max(ℓ_M, 0), max(u_M, 0)) == (0, 1)
        @assert iszero(α) "Cannot compute MMᵀ for a cyclic banded matrix with a non-zero upper-right corner entry."
        A[end, end] -= β^2
        A[1, end] -= m11 * β
        A[end, 1] -= m11 * β
    else
        throw(ArgumentError("Cannot compute MMᵀ for a cyclic banded matrix with sub-bandwidths ($ℓ_M, $u_M)."))
    end
    return A
end

function diagonal_rinv!(C::AbstractMatrix, D::AbstractMatrix)
    # Computes C = C * inv(D), where D is a diagonal matrix 
    require_one_based_indexing(C, D)
    d = checksquare(D)
    m, n = size(C)
    m′, n′ = size(D)
    m == d || throw(DimensionMismatch(lazy"right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch(lazy"expect output to be $m by $n, but got $m′ by $n′"))

    Cbp = _aliased_bandedpart(C)

    for j in 1:n
        iszero(D[j, j]) && throw(SingularException(j))
        for i in colsupport(Cbp, j)
            Cbp[i, j] /= D[j, j]
        end
    end
    if m > 2
        C[1, end] /= D[end, end]
        C[end, 1] /= D[1, 1]
    end
    return C
end

function diagonal_subrmul!(C::AbstractMatrix, D::AbstractMatrix, M::AbstractMatrix)
    # Computes C = C - M * D, where D is a diagonal matrix. The result is placed into C
    # C and M should have the same banded structure 
    require_one_based_indexing(C, D, M)
    d = checksquare(D)
    m, n = size(C)
    m′, n′ = size(D)
    m′′, n′′ = size(M)
    m == d || throw(DimensionMismatch(lazy"right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch(lazy"expect output to be $m by $n, but got $m′ by $n′"))
    (m, n) == (m′′, n′′) || throw(DimensionMismatch(lazy"expect output to be $m by $n, but got $m′′ by $n′′"))

    Cbp = _aliased_bandedpart(C)
    Mbp = _aliased_bandedpart(M)

    ℓ, u = bandwidths(Cbp)
    ℓ′, u′ = bandwidths(Mbp)
    ℓ == ℓ′ || throw(ArgumentError(lazy"expect the lower sub-bandwidth of C to be $ℓ, but got $ℓ′"))
    u == u′ || throw(ArgumentError(lazy"expect the upper sub-bandwidth of C to be $u, but got $u′"))
    for j in 1:n
        Dj = D[j, j]
        iszero(Dj) && continue
        for i in colsupport(C, j)
            iszero(Mbp[i, j]) || (Cbp[i, j] -= Mbp[i, j] * Dj)
        end
    end
    if n > 2
        C[1, end] -= M[1, end] * D[end, end]
        C[end, 1] -= M[end, 1] * D[1, 1]
    end
    return C
end

function MᵀM(M::CyclicBandedMatrix)
    # Computes M' * M 
    ℓ, u = _bpbandwidths(M)
    β, α = M[1, end], M[end, 1]
    if (ℓ, u) == (0, 0) || size(M, 1) == 2
        B = bandedpart(M)
        BᵀB = B' * B
        if size(M, 1) > 2
            BᵀB[1, 1] += α^2
            BᵀB[end, end] += β^2
        end
        bdata, _ℓ, _u = _builddata(BᵀB, B[1, 1] * β + B[end, end] * α, B[1, 1] * β + B[end, end] * α)
        return _CyclicBandedMatrix(bdata, 0, 0, (_ℓ, _u))
    elseif (ℓ, u) == (1, 0)
        @assert iszero(α) "Cannot compute MᵀM for a cyclic banded matrix with a non-zero lower-left corner entry."
        B = bandedpart(M)
        BᵀB = B' * B
        BᵀB[end, end] += β^2
        bdata, _ℓ, _u = _builddata(BᵀB, B[1, 1] * β, B[1, 1] * β)
        return _CyclicBandedMatrix(bdata, 1, 1, (_ℓ, _u))
    elseif (ℓ, u) == (0, 1)
        @assert iszero(β) "Cannot compute MᵀM for a cyclic banded matrix with a non-zero upper-right corner entry."
        B = bandedpart(M)
        BᵀB = B' * B
        BᵀB[1, 1] += α^2
        bdata, _ℓ, _u = _builddata(BᵀB, B[end, end] * α, B[end, end] * α)
        return _CyclicBandedMatrix(bdata, 1, 1, (_ℓ, _u))
    else
        throw(ArgumentError("Cannot compute MᵀM for a cyclic banded matrix with sub-bandwidths ($ℓ, $u)."))
    end
end

# cholesky/reversecholesky
cholcopy(A::Symmetric{<:Real,<:CyclicBandedMatrix}) = copyto!(similar(A, choltype(A)), A)
cholesky(A::Symmetric{<:Real,<:CyclicBandedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = cholesky_layout(MemoryLayout(A), axes(A), A, v; check)
function cholesky(A::CyclicBandedMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return cholesky(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
cholesky!(A::Symmetric{<:Real,<:CyclicBandedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = cholesky!_layout(MemoryLayout(A), axes(A), A, v; check)
function cholesky!(A::CyclicBandedMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return cholesky!(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
reversecholesky(A::Symmetric{<:Real,<:CyclicBandedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout(MemoryLayout(A), axes(A), A, v; check)
function reversecholesky(A::CyclicBandedMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return reversecholesky(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
reversecholesky!(A::Symmetric{<:Real,<:CyclicBandedMatrix}, v::CNoPivot=CNoPivot(); check::Bool=true) = reversecholesky_layout!(MemoryLayout(A), A, A.uplo == 'U' ? UpperTriangular : LowerTriangular; check)
function reversecholesky!(A::CyclicBandedMatrix, v::CNoPivot=CNoPivot(); check::Bool=true)
    if ishermitian(A)
        return reversecholesky!(Symmetric(A), v; check)
    else
        throw(ArgumentError("Matrix must be Hermitian"))
    end
end
function cholesky!_layout(::SymmetricLayout{ML}, _axes, SA::Symmetric{T,S}, piv::CNoPivot=CNoPivot(); check::Bool=true) where {ML<:CyclicBandedLayout,T<:Real,S<:CyclicBandedMatrix}
    #=
    This function overwrites the data in A so that the upper triangular part (excluding the corner)
    is the banded part of the Cholesky factor. A[band(-1)] contains entries in the last column of A,
    except for the last u entries where u is the bandwidth. In particular, A[band(-1)][begin:(end-u)]
    contains the entries in chol.U[begin:(end-u), end]. It would be better to work with chol.factors from the 
    output, which represents this Cholesky factor as an AlmostBandedMatrix.
    =#

    A = parent(SA)
    n = checksquare(A)
    storage = A.data

    # Save value before it is destroyed
    ann = SA[end, end]

    # Get the Cholesky of the banded (n-1) principal submatrix
    _ℓ, _u = _extended_bpbandwidths(A)
    BP_direct = _BandedMatrix(storage, axes(A, 1), _ℓ, _u)
    B = @views Symmetric(BP_direct[begin:end-1, begin:end-1], SA.uplo == 'U' ? :U : :L)
    chol = cholesky!_layout(MemoryLayout(B), axes(B), B, CNoPivot(); check)
    check && checkpositivedefinite(chol.info)
    @assert chol.uplo == 'U'

    # Compute the rightmost column of the Cholesky factor
    u_ext = _u  # extended bandwidth
    a = @view SA[begin:end-1, end]
    L = A'
    v = @view storage[u_ext+2, begin:end-1]
    fill!(v, zero(eltype(v)))
    _lcbm_ldiv!(v, L, a, LowerTriangular)
    val = ann - norm(v)^2
    if check
        val > 0 || throw(PosDefException(n))
    end
    storage[u_ext+1, end] = sqrt(val)

    # Fix the overlapping parts
    for i in (n-u_ext):(n-1)
        A[i, end] = v[i]
        v[i] = 0
    end
    storage[u_ext+2, end] = 0

    # Represent as an AlmostBandedMatrix
    extv = view(storage, u_ext + 2, :)
    U = _BandedMatrix(view(storage, 1:u_ext+1, :), axes(A, 1), 0, u_ext)
    B = ApplyMatrix(*, extv, OneElement(one(eltype(extv)), (1, n), (1, n)))
    Ufact = AlmostBandedMatrix(U, B)
    return Cholesky(Ufact, 'U', convert(BlasInt, chol.info != 0 ? chol.info : val > 0 ? 0 : 1))
end
function _lcbm_ldiv!(v, L, a, ::Type{LowerTriangular})
    #= 
    Solves Lv = a, where
        - L is a lower triangular matrix with bandwidth u 
        - a is a vector where the only non-zero entries are a[1] and a[n-u+1:n]
    =#
    require_one_based_indexing(v, L, a)
    n = length(v)
    u = bandwidth(L, 1)
    v[1] = a[1] / L[1, 1]
    for i in 2:(n-u)
        range = max(1, i - u):(i-1)
        v[i] = 0.0
        for j in range
            v[i] += L[i, j] * v[j]
        end
        v[i] *= -inv(L[i, i])
    end
    for i in (n-u+1):n
        range = max(1, i - u):(i-1)
        v[i] = a[i]
        for j in range
            v[i] -= L[i, j] * v[j]
        end
        v[i] /= L[i, i]
    end
    return v
end

function reversecholesky_layout!(::SymmetricLayout{ML}, SA::Symmetric{T,S}, ::Type{UpperTriangular}; check::Bool=true) where {ML<:CyclicBandedLayout,T<:Real,S<:CyclicBandedMatrix}

    # The same comments from cholesky!_layout apply here

    A = parent(SA)
    n = checksquare(A)
    storage = A.data

    # Work directly with A.data to avoid the copy from bandedpart
    _ℓ, _u = _extended_bpbandwidths(A)
    BP = _BandedMatrix(storage, axes(A, 1), _ℓ, _u)

    a11 = SA[begin, begin]
    B = @views BP[begin+1:end, begin+1:end]
    check = true
    U, info = reversecholesky_layout!(MemoryLayout(B), B, SA.uplo == 'U' ? UpperTriangular : LowerTriangular)
    check && checkpositivedefinite(info)

    a = @view SA[begin+1:end, begin]
    v = @view storage[_u+2, begin+1:end]
    fill!(v, zero(eltype(v)))
    _lcbm_ldiv!(v, U, a, UpperTriangular)
    val = a11 - norm(v)^2
    if check
        val > 0 || throw(PosDefException(1))
    end
    storage[_u+1, begin] = sqrt(val)

    for i in 1:_u
        # The work-around here is to avoid the !iszero branch in _banded_setindex!, forcing the setindex! to allow 
        # storage outside the bandwidth and instead use the internal-bandwidth (recall, the public bandwidth may differ
        # from the extended bandwidth in cases where we needed to expand the bandwidth to accomodate the corner entries).
        _inbands_setindex!(storage, _u, v[i], 1, i + 1) # A[begin, i+1] = v[i]
        v[i] = 0
    end
    storage[_u+2, begin] = 0

    extv = view(storage, _u + 2, :)
    U = _BandedMatrix(view(storage, 1:_u+1, :), axes(A, 1), 0, _u)
    B = ApplyMatrix(*, OneElement(one(eltype(extv)), 1, n), extv')
    Ufact = AlmostBandedMatrix(U, B)
    return ReverseCholesky(Ufact, 'U', info != 0 ? info : val > 0 ? 0 : 1)
end

function _lcbm_ldiv!(v, U, a, ::Type{UpperTriangular})
    #= 
    Solves Uv = a, where
        - U is an upper triangular matrix with bandwidth u 
        - a is a vector where the only non-zero entries are a[1:u] and a[n]
    =#
    require_one_based_indexing(v, U, a)
    n = length(v)
    u = bandwidth(U, 2)
    v[n] = a[n] / U[n, n]
    for i in (n-1):-1:(u+1)
        range = (i+1):min(n, i + u)
        v[i] = 0.0
        for j in range
            v[i] += U[i, j] * v[j]
        end
        v[i] *= -inv(U[i, i])
    end
    for i in u:-1:1
        range = (i+1):min(n, i + u)
        v[i] = a[i]
        for j in range
            v[i] -= U[i, j] * v[j]
        end
        v[i] /= U[i, i]
    end
    return v
end

# broadcast  
struct CyclicBandedMatrixStyle <: AbstractArrayStyle{2} end
CyclicBandedMatrixStyle(::Val{2}) = CyclicBandedMatrixStyle()
BroadcastStyle(::Type{<:CyclicBandedMatrix}) = CyclicBandedMatrixStyle()
BroadcastStyle(::Type{<:Adjoint{<:Any,Mat}}) where {Mat<:CyclicBandedMatrix} = BroadcastStyle(Mat)
BroadcastStyle(::Type{<:Transpose{<:Any,Mat}}) where {Mat<:CyclicBandedMatrix} = BroadcastStyle(Mat)
BroadcastStyle(::Type{<:SubArray{<:Any,2,<:CyclicBandedMatrix,<:NTuple{2,AbstractUnitRange{Int}}}}) = BandedStyle()
BroadcastStyle(a::DefaultArrayStyle{2}, ::CyclicBandedMatrixStyle) = CyclicBandedMatrixStyle() ####### (Ignore following) So that e.g. CyclicBanded - Matrix works. Could maybe also use ArrayConflict
BroadcastStyle(::CyclicBandedMatrixStyle, a::DefaultArrayStyle{2}) = CyclicBandedMatrixStyle()
function copy(bc::Broadcasted{<:CyclicBandedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix}})
    (A,) = bc.args
    bp = bandedpart(A)
    bp_op = broadcast(bc.f, bp)
    a, b = A[1, end], A[end, 1]
    a_op = bc.f(a)
    b_op = bc.f(b)
    bdata = _builddata(bp_op, a_op, b_op)[1]
    return _CyclicBandedMatrix(bdata, _bpbandwidths(A)..., _extended_bpbandwidths(A))
end
function copy(bc::Broadcasted{<:CyclicBandedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix,Number}})
    (A, x) = bc.args
    bp = bandedpart(A)
    bp_op = broadcast(bc.f, bp, x)
    a, b = A[1, end], A[end, 1]
    a_op = bc.f(a, x)
    b_op = bc.f(b, x)
    bdata = _builddata(bp_op, a_op, b_op)[1]
    return _CyclicBandedMatrix(bdata, _bpbandwidths(A)..., _extended_bpbandwidths(A))
end
function copy(bc::Broadcasted{<:CyclicBandedMatrixStyle,<:Any,<:Any,<:Tuple{Number,AbstractMatrix}})
    (x, A) = bc.args
    bp = bandedpart(A)
    bp_op = broadcast(bc.f, x, bp)
    a, b = A[1, end], A[end, 1]
    a_op = bc.f(x, a)
    b_op = bc.f(x, b)
    bdata = _builddata(bp_op, a_op, b_op)[1]
    return _CyclicBandedMatrix(bdata, _bpbandwidths(A)..., _extended_bpbandwidths(A))
end


datasize(A::CyclicBandedMatrix) = size(A.data)
datasize(A::Adjoint{<:Any,<:CyclicBandedMatrix}) = datasize(parent(A))
datasize(A::Transpose{<:Any,<:CyclicBandedMatrix}) = datasize(parent(A))
datasize(A) = size(A)  # fallback for other types
same_datasize(A, B) = datasize(A) == datasize(B)

function compatible_cbm(A::Zeros, B)
    # Returns a CyclicBandedMatrix with the same size and bandwidths as B, with Zero data
    return _CyclicBandedMatrix(Zeros(datasize(B)), _bpbandwidths(B)..., _extended_bpbandwidths(B))
end
function compatible_cbm(A, B::Zeros)
    # Returns a CyclicBandedMatrix with the same size and bandwidths as A, with Zero data
    return _CyclicBandedMatrix(Zeros(datasize(A)), _bpbandwidths(A)..., _extended_bpbandwidths(A))
end

combine_bpbandwidths(A, B) = (max(_bpbandwidths(A, 1), _bpbandwidths(B, 1)), max(_bpbandwidths(A, 2), _bpbandwidths(B, 2)))
combine_extended_bpbandwidths(A, B) = (max(_extended_bpbandwidths(A, 1), _extended_bpbandwidths(B, 1)), max(_extended_bpbandwidths(A, 2), _extended_bpbandwidths(B, 2)))

# This method has a bunch of bandaid fixes. Could be improved later.
function copy(bc::Broadcasted{<:CyclicBandedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix,AbstractMatrix}})
    (A, B) = bc.args
    if A isa Zeros
        A = compatible_cbm(A, B)
    end
    if B isa Zeros
        B = compatible_cbm(B, A)
    end
    if bc.f(zero(eltype(A)), zero(eltype(B))) ≠ 0
        return broadcast(bc.f, Matrix(A), Matrix(B))
    elseif same_datasize(A, B)
        if A isa CyclicBandedMatrix && B isa CyclicBandedMatrix
            return _CyclicBandedMatrix(broadcast(bc.f, A.data, B.data), combine_bpbandwidths(A, B)..., combine_extended_bpbandwidths(A, B))
        elseif A isa Adjoint && B isa Adjoint
            _A, _B = parent(A), parent(B)
            C = _CyclicBandedMatrix(broadcast(bc.f, _A.data, _B.data), combine_bpbandwidths(_A, _B)..., combine_extended_bpbandwidths(_A, _B))
            return adjoint(C)
        elseif A isa Adjoint && B isa CyclicBandedMatrix
            _A = CyclicBandedMatrix(BandedMatrix(bandedpart(A), _bpbandwidths(A)), A[1, end], A[end, 1])
            return _CyclicBandedMatrix(broadcast(bc.f, _A.data, B.data), combine_bpbandwidths(_A, B)..., combine_extended_bpbandwidths(_A, B))
        elseif A isa CyclicBandedMatrix && B isa Adjoint
            _B = CyclicBandedMatrix(BandedMatrix(bandedpart(B), _bpbandwidths(B)), B[1, end], B[end, 1])
            return _CyclicBandedMatrix(broadcast(bc.f, A.data, _B.data), combine_bpbandwidths(A, _B)..., combine_extended_bpbandwidths(A, _B))
        else
            throw("Not implemented for types $(typeof(A)) and $(typeof(B))")
        end
    else
        A_bp, B_bp = bandedpart(A), bandedpart(B)
        op_bp = broadcast(bc.f, A_bp, B_bp)
        corner_ops = broadcast(bc.f, A[1, end], B[1, end]), broadcast(bc.f, A[end, 1], B[end, 1])
        bdata = _builddata(op_bp, corner_ops...)[1]
        ℓ, u = combine_bpbandwidths(A, B)
        _ℓ, _u = combine_extended_bpbandwidths(A, B)
        return _CyclicBandedMatrix(bdata, ℓ, u, (_ℓ, _u))
    end
end

function similar(bc::Broadcasted{<:CyclicBandedMatrixStyle}, ::Type{T}) where {T}
    A, x = bc.args
    bp = bandedpart(A)
    f = bc.f
    bc2 = broadcasted(f, bp, x)
    dest = similar(bc2, T)
    if bandwidths(dest) == bandwidths(bp)
        return _CyclicBandedMatrix(_builddata(dest, zero(T), zero(T))[1], _bpbandwidths(A)..., _extended_bpbandwidths(A))
    else
        return CyclicBandedMatrix(dest, zero(T), zero(T)) # If the bandwidth had to be increased to accommodate a non-zero f(0, 0), we just recreate it
    end
end

function copyto!(dest::AbstractArray, bc::Broadcasted{<:CyclicBandedMatrixStyle,<:Any,<:Any,<:Tuple{AbstractMatrix,AbstractVector}})
    f = bc.f
    (A, x) = bc.args
    a1, a2 = A[1, end], A[end, 1]
    bp = bandedpart(A)

    # Work directly with dest's internal storage instead of bandedpart(dest)
    _ℓ, _u = _extended_bpbandwidths(dest)
    bpdest = _BandedMatrix(dest.data, axes(dest, 1), _ℓ, _u)

    copyto!(bpdest, broadcasted(f, bp, x))
    dest[1, end] = f(a1, x[1])
    dest[end, 1] = f(a2, x[end])
    dest
end

end # module CyclicBandedMatrices