module SemiclassicalArcPolynomials  

import ..ArcPolynomials: ArcInclusion, Arc
import MultivariateOrthogonalPolynomials: MultivariateOrthogonalPolynomial
import ContinuumArrays: Weight, grid, plan_transform, grammatrix_layout, grammatrix, ExpansionLayout, basis
import HarmonicOrthogonalPolynomials: BlockOneTo
import LazyArrays: Hcat, LazyMatrix, ApplyArray, LazyVector, arguments, paddeddata, Vcat, AbstractLazyBandedLayout, LazyBandedLayout, paddeddata, resizedata!
import BlockArrays: BlockedUnitRange, _BlockedUnitRange, BlockedArray, Block, blockcolsupport, BlockRange, findblockindex, block, blockindex, BlockIndex
import LinearAlgebra: dot, Bidiagonal, Diagonal, Symmetric, I
import InfiniteArrays: InfStepRange, OneToInf, AbstractInfUnitRange, InfUnitRange
import QuasiArrays: ApplyQuasiVector, layout_broadcasted, layout_broadcasted
import Base: getindex, axes, show, summary, ==, size, diff, \, *, copy, Slice, @propagate_inbounds, OneTo, IdentityUnitRange, unitrange, isassigned
import Infinities: ∞, ℵ₀
import SemiclassicalOrthogonalPolynomials: SemiclassicalJacobi, divdiff
import ClassicalOrthogonalPolynomials: HalfWeighted, orthogonalityweight, increasingtruncations, gaussradau, jacobimatrix, Weighted, recurrencecoefficients, _p0
import FillArrays: SquareEye 
import BandedMatrices: _BandedMatrix, BandedMatrix, band, AbstractBandedMatrix, bandwidths, isbanded, inbands_getindex, bandeddata
import ArrayLayouts: MemoryLayout, supdiagonaldata, subdiagonaldata, diagonaldata, sublayout, sub_materialize, sublayout, sub_materialize, transposelayout
import AbstractFFTs: Plan
import RecurrenceRelationshipArrays: Clenshaw
import LazyBandedMatrices: LazyBandedMatrices
import InfiniteLinearAlgebra: BidiagonalConjugation

include("definition.jl")
include("conversion.jl")
include("transform.jl")
include("jacobi.jl")
include("diff.jl")
include("grammatrix.jl")
include("multiplication.jl")

export SemiclassicalJacobiArc

end # module SemiclassicalArcPolynomials