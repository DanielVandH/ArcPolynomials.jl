module PiecewiseArcPolynomials 

import ..ArcPolynomials: ArcInclusion, map_to_std_range
import ..TrigonometricSplines: TrigonometricSpline
import ..CyclicBBBArrowheadMatrices: CyclicBBBArrowheadMatrix, InterlacedMatrix
import ..CyclicBandedMatrices: CyclicBandedMatrix, _CyclicBandedMatrix, MᵀM
import ..SemiclassicalArcPolynomials: SemiclassicalJacobiArc, semiclassicaljacobiarc_diffp, semiclassicaljacobiarc_diffq, ArcPlan
import PiecewiseOrthogonalPolynomials: AbstractPiecewisePolynomial, PiecewisePolynomialLayout
import SemiclassicalOrthogonalPolynomials: _linear_coefficients
import BlockArrays: Block, mortar, findblockindex, findblock, BlockRange, blockindex, block, blockedrange, BlockIndex, AbstractBlockVector, BlockedArray
import QuasiArrays: ApplyQuasiVector, AbstractQuasiVector, layout_broadcasted
import HarmonicOrthogonalPolynomials: BlockOneTo
import LazyArrays: LazyVector, Vcat, Hcat, arguments, paddeddata
import Base: require_one_based_indexing, OneTo, oneto, diff, ==, show, size, summary, getindex, axes, \, *
import FillArrays: Fill, Ones, Zeros
import Infinities: ∞
import ContinuumArrays: affine, grammatrix, grammatrix_layout, transform, grid, plan_transform, basis, ExpansionLayout, weaklaplacian_layout
import LinearAlgebra: Diagonal, Symmetric, dot, transpose
import BandedMatrices: BandedMatrix, _BandedMatrix, band
import ClassicalOrthogonalPolynomials: orthogonalityweight, adaptivetransform_ldiv
import ArrayLayouts: colsupport, ldiv, MemoryLayout
import BlockBandedMatrices: blockcolsupport
import AbstractFFTs: Plan
import LazyBandedMatrices: BlockVec, BlockBroadcastMatrix, unitblocks
import IntervalSets: var".."
import InfiniteLinearAlgebra: pad

include("definition.jl")
include("conversion.jl")
include("grammatrix.jl")
include("diff.jl")
include("transform.jl")
include("multiplication.jl")

export PiecewiseArcPolynomial 

end # module PiecewiseArcPolynomials