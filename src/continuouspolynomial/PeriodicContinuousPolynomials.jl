module PeriodicContinuousPolynomials

import ..ArcPolynomials: map_to_std_range
import ..PeriodicLinearSplines: PeriodicLinearSpline
import ..CyclicBBBArrowheadMatrices: CyclicBBBArrowheadMatrix, InterlacedMatrix
import ..CyclicBandedMatrices: CyclicBandedMatrix, MᵀM
import PiecewiseOrthogonalPolynomials: AbstractPiecewisePolynomial, PiecewisePolynomialLayout
import ClassicalOrthogonalPolynomials: Legendre, Ultraspherical, adaptivetransform_ldiv
import DomainSets: ℝ
import QuasiArrays: Inclusion, AbstractQuasiVector, ApplyQuasiVector, layout_broadcasted
import Base: axes, ==, OneTo, oneto, require_one_based_indexing, show, summary, getindex, \, diff, *
import FillArrays: Fill, Ones, Zeros # can remove Zeros after implementing \ for CyclicBBB
import Infinities: ∞
import BlockArrays: blockedrange, BlockRange, mortar, block, blockindex, Block, findblock, findblockindex, BlockIndex, BlockedArray
import HarmonicOrthogonalPolynomials: BlockOneTo
import ContinuumArrays: grammatrix_layout, grammatrix, grid, affine, plan_transform, checkpoints, InvPlan, basis, ExpansionLayout, weaklaplacian_layout
import LinearAlgebra: Symmetric, Diagonal, transpose
import BandedMatrices: _BandedMatrix, band, BandedMatrix
import LazyArrays: Vcat, paddeddata, arguments, Hcat
import AbstractFFTs: Plan
import LazyBandedMatrices: BlockVec, BlockBroadcastMatrix, unitblocks
import IntervalSets: var".."
import InfiniteLinearAlgebra: pad
import ArrayLayouts: ldiv, colsupport 
import StaticArrays: StaticVector

export PeriodicContinuousPolynomial

include("definition.jl")
include("grammatrix.jl")
include("conversion.jl")
include("diff.jl")
include("transform.jl")
include("multiplication.jl")

end # module PeriodicContinuousPolynomials