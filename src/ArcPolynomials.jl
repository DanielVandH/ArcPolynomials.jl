module ArcPolynomials

import QuasiArrays: Inclusion, cardinality
import DomainSets: Domain
import IntervalSets: var".."
import LazyArrays: LazyLayout
import QuasiArrays: SubQuasiArray, ApplyQuasiVector, Inclusion
import Infinities: InfiniteCardinal, ℵ₁
import Base: getindex, show, summary, ==, in, checkindex
import ContinuumArrays: checkpoints, plotgrid, basis_layout, grid, plotvalues_size

include("domain.jl")

include("arc/SemiclassicalArcPolynomials.jl")
using .SemiclassicalArcPolynomials

include("cyclic/CyclicBandedMatrices.jl")
using .CyclicBandedMatrices
include("cyclic/CyclicBBBArrowheadMatrices.jl")
using .CyclicBBBArrowheadMatrices

include("spline/TrigonometricSplines.jl") 
using .TrigonometricSplines # Could just define a PeriodicSplines module with AbstractPeriodicSpline
include("spline/PeriodicLinearSplines.jl")
using .PeriodicLinearSplines

include("piecewise/PiecewiseArcPolynomials.jl")
using .PiecewiseArcPolynomials 
include("continuouspolynomial/PeriodicContinuousPolynomials.jl")
using .PeriodicContinuousPolynomials
 
export SemiclassicalJacobiArc
export TrigonometricSpline, PeriodicLinearSpline
export PiecewiseArcPolynomial
export PeriodicContinuousPolynomial
export principal_submatrix

include("plotting.jl")

end # module ArcPolynomials
