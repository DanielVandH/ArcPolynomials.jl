struct SemiclassicalJacobiArcJacobiXData{T} <: LazyMatrix{T}
    PX::AbstractMatrix{T}
    QX::AbstractMatrix{T}
end
struct SemiclassicalJacobiArcJacobiYData{T} <: LazyMatrix{T}
    Φ::AbstractMatrix{T} # P = QΦ 
    Ψ::AbstractMatrix{T} # y^2Q = PΨ
end
size(::SemiclassicalJacobiArcJacobiYData) = (7, ∞)
size(::SemiclassicalJacobiArcJacobiXData) = (5, ∞)
function SemiclassicalJacobiArcJacobiXData(R::SemiclassicalJacobiArc{T}) where {T}
    PX = (R.h - 1) * jacobimatrix(R.P)
    QX = (R.h - 1) * jacobimatrix(R.Q)
    return SemiclassicalJacobiArcJacobiXData(PX, QX)
end
function SemiclassicalJacobiArcJacobiYData(R::SemiclassicalJacobiArc{T}) where {T}
    Φ = R.Q \ R.P
    M = Weighted(R.P) \ Weighted(R.Q)
    return SemiclassicalJacobiArcJacobiYData(Φ, (one(R.h) - R.h)^2 * M)
end
function getindex(X::SemiclassicalJacobiArcJacobiXData{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(Bool, X, i, j) || throw(BoundsError(X, (i, j)))
    i == 2 && return zero(T)
    i == 4 && return zero(T)
    a⁻, b⁻, c⁻ = supdiagonaldata(X.PX), diagonaldata(X.PX), subdiagonaldata(X.PX)
    a⁺, b⁺, c⁺ = supdiagonaldata(X.QX), diagonaldata(X.QX), subdiagonaldata(X.QX)
    @inbounds begin
        if j == 1 
            return i == 3 ? b⁻[1] + 1 : i == 5 ? c⁻[1] : zero(T)
        elseif j == 2 
            return i == 3 ? b⁺[1] + 1 : i == 5 ? c⁺[1] : zero(T)
        elseif isodd(j) 
            j′ = j ÷ 2
            return i == 1 ? a⁻[j′] : i == 3 ? b⁻[j′+1] + 1 : i == 5 ? c⁻[j′+1] : zero(T)
        else # iseven(j) 
            j′ = j ÷ 2
            return i == 1 ? a⁺[j′-1] : i == 3 ? b⁺[j′] + 1 : i == 5 ? c⁺[j′] : zero(T)
        end
    end
end
function getindex(Y::SemiclassicalJacobiArcJacobiYData{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(Bool, Y, i, j) || throw(BoundsError(Y, (i, j)))
    Φ, Ψ = Y.Φ, Y.Ψ
    iseven(i) && return zero(T)
    @inbounds begin 
        if j == 1
            return i == 5 ? Φ[1, 1] : zero(T)
        elseif j == 2 
            return i == 3 ? Ψ[1, 1] : i == 5 ? Ψ[2, 1] : i == 7 ? Ψ[3, 1] : zero(T)
        elseif j == 3 
            return i == 3 ? Φ[1, 2] : i == 5 ? Φ[2, 2] : zero(T)
        elseif iseven(j)
            j′ = j ÷ 2
            return i == 3 ? Ψ[j′, j′] : i == 5 ? Ψ[j′+1, j′] : i == 7 ? Ψ[j′+2, j′] : zero(T)
        else # isodd(j)
            j′ = j ÷ 2
            return i == 1 ? Φ[j′-1, j′+1] : i == 3 ? Φ[j′, j′+1] : i == 5 ? Φ[j′+1, j′+1] : zero(T)
        end
    end
end

function jacobimatrix(::Val{1}, R::SemiclassicalJacobiArc{T}) where {T}
    data = SemiclassicalJacobiArcJacobiXData(R)
    return BlockedArray(_BandedMatrix(data, 1:∞, 2, 2), (axes(R, 2), axes(R, 2)))
end
function jacobimatrix(::Val{2}, R::SemiclassicalJacobiArc{T}) where {T}
    data = SemiclassicalJacobiArcJacobiYData(R)
    return BlockedArray(_BandedMatrix(data, 1:∞, 3, 3), (axes(R, 2), axes(R, 2)))
end