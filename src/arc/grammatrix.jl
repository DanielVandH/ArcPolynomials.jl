struct SemiclassicalJacobiArcGrammatrixData{T} <: LazyVector{T}
    PG::T
    QG::T
end
size(::SemiclassicalJacobiArcGrammatrixData) = (∞,)
function getindex(M::SemiclassicalJacobiArcGrammatrixData, i::Int)
    @boundscheck checkbounds(Bool, M, i) || throw(BoundsError(M, i))
    return isodd(i) ? M.PG : M.QG
end
function SemiclassicalJacobiArcGrammatrixData(R::SemiclassicalJacobiArc{T}) where {T}
    PG = 2sum(orthogonalityweight(R.P))
    QG = 2(1-R.h)^2*sum(orthogonalityweight(R.Q))
    return SemiclassicalJacobiArcGrammatrixData(PG, QG)
end

function grammatrix_layout(_, R::SemiclassicalJacobiArc{T}) where {T}
    b = R.b
    if iszero(b)
        return Diagonal(SemiclassicalJacobiArcGrammatrixData(R))
    else
        R0 = SemiclassicalJacobiArc(zero(R.b), R)
        U = R0 \ R
        G = grammatrix(R0)
        return Symmetric(U' * G * U)
    end
end 