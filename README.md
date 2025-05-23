# ArcPolynomials

Implements the bases introduced in the paper _A sparse hp-finite element method for piecewise-smooth differential equations with periodic boundary conditions_ by Daniel VandenHeuvel and Sheehan Olver.

The code used for producing the results in the paper is in the `paper/` directory, which includes a READMD file with instructions on how to run the code.

# Installation

To use the package, do

```julia
using Pkg
Pkg.add("https://github.com/DanielVandH/ArcPolynomials.jl")
using ArcPolynomials
```

# Overview

The package implements the bases:
- `SemiclassicalJacobiArc(h, b)`: This is the $\boldsymbol P^{(b, h)}(x, y)$ basis from the paper, i.e. the arc polynomial basis.
- `PiecewiseArcPolynomial{O}(points)`: This is the $\boldsymbol P^{(b), \boldsymbol\theta}$ basis from the paper, where $\boldsymbol\theta$ is the `points` vector and `O = 0` or `O = 1` for $b = 0$ and $b = -1$, respectively, i.e. the piecewise arc polynomial basis.
- `PeriodicContinuousPolynomial{O}(points)`: This is the $\boldsymbol W^{(b),\boldsymbol\theta}$ basis from the paper, where $\boldsymbol\theta$ is the `points` vector and `O = 0` or `O = 1` for $b = 0$ and $b = -1$, respectively, i.e. the periodic piecewise integrated Legendre basis. 

The hat functions for the piecewise bases are also implemented in:
- `PeriodicLinearSpline(points)`: These are the $\boldsymbol H_L$ functions, in particular a periodic version of the usual linear hat functions for points on $[-\pi, \pi]$ that wraps around to be $2\pi$-periodic. Note that this is simply a periodic version of the `LinearSpline` function from [`ContinuumArrays.jl`](https://github.com/JuliaApproximation/ContinuumArrays.jl).
- `TrigonometricSpline(points)`: These are the $\boldsymbol H^{\boldsymbol\theta}$ functions, in particular a version of `PeriodicLinearSpline(points)` that, instead of being linear in $\theta$, is linear in $\cos(\theta)$ and $\sin(\theta)$. 

Finally, we also implement:
- `CyclicBandedMatrix`: This is a struct that is analogous to `BandedMatrix` from [`BandedMatrices.jl`](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) except allowing zeros in the lower-left and upper-right corners of the matrix. This is the implementation of the cyclic banded matrix definition given in the paper.
- `CyclicBBBArrowheadMatrix`: This is the implementation of the $CB^3$-arrowhead matrix defined in the paper, and is a cyclic version of the `BBBArrowheadMatrix` from [`PiecewiseOrthogonalPolynomials.jl`](https://github.com/JuliaApproximation/PiecewiseOrthogonalPolynomials.jl).

# Examples 

```julia
julia> # SemiclassicalJacobiArc

julia> h, b = 0.5, -1.0
(0.5, -1.0)

julia> P = SemiclassicalJacobiArc(h, b)
SemiclassicalJacobiArc{Float64} with weight (x - 0.5)^-1.0 on ArcPolynomials.Arc{Float64}

julia> P[0.2, 1] # p₀(cos(0.2), sin(0.2))
1.0

julia> P[0.5, Block(3)] # (q₂(cos(0.5), sin(0.5)), p₂(cos(0.5), sin(0.5)))
2-element Vector{Float64}:
 0.36204544620369356
 0.13332710949428817

julia> P'P # mass matrix
ℵ₀×ℵ₀ LinearAlgebra.Symmetric{Float64, LazyArrays.ApplyArray{Float64, 2, typeof(*), Tuple{LinearAlgebra.Adjoint{Float64, BlockedMatrix{Float64, BandedMatrices.BandedMatrix{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcConversionData{Float64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}, BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}}}, LinearAlgebra.Diagonal{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcGrammatrixData{Float64}}, BlockedMatrix{Float64, BandedMatrices.BandedMatrix{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcConversionData{Float64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}, BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}}}}} with indices 1:1:ℵ₀×1:1:ℵ₀:
 2.0944   │  0.0        1.36971  │    ⋅           ⋅        │    ⋅           ⋅        │    ⋅
 ─────────┼──────────────────────┼─────────────────────────┼─────────────────────────┼────────────  …
 0.0      │  0.614185   0.0      │   0.251841     ⋅        │    ⋅           ⋅        │    ⋅
 1.36971  │  0.0        1.08703  │   0.0        -0.29837   │    ⋅           ⋅        │    ⋅
 ─────────┼──────────────────────┼─────────────────────────┼─────────────────────────┼────────────
  ⋅       │  0.251841   0.0      │   0.145838    0.0       │  -0.0584861    ⋅        │    ⋅
  ⋅       │   ⋅        -0.29837  │   0.0         0.727953  │   0.0        -0.32959   │    ⋅
 ─────────┼──────────────────────┼─────────────────────────┼─────────────────────────┼────────────
  ⋅       │   ⋅          ⋅       │  -0.0584861   0.0       │   0.131031    0.0       │  -0.0612936  …
  ⋅       │   ⋅          ⋅       │    ⋅         -0.32959   │   0.0         0.701884  │   0.0
 ─────────┼──────────────────────┼─────────────────────────┼─────────────────────────┼────────────
  ⋅       │   ⋅          ⋅       │    ⋅           ⋅        │  -0.0612936   0.0       │   0.128269
 ⋮                                                    ⋮                                 ⋱

julia> weaklaplacian(P) # -diff(P)'diff(P)
(-).((ℵ₀×ℵ₀ adjoint(::BlockedArray{…,::BandedMatrices.BandedMatrix{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcDiffData{Float64}, InfiniteArrays.InfUnitRange{Int64}}}) with eltype Float64 with indices 1:1:ℵ₀×1:1:ℵ₀) * (ℵ₀×ℵ₀ LinearAlgebra.Diagonal{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcGrammatrixData{Float64}} with indices OneToInf()×OneToInf()) * (ℵ₀×ℵ₀-blocked ℵ₀×ℵ₀ BlockedMatrix{Float64, BandedMatrices.BandedMatrix{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcDiffData{Float64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}, BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}}) with indices 1:1:ℵ₀×1:1:ℵ₀) with indices 1:1:ℵ₀×1:1:ℵ₀:
 -0.0  │  -0.0       -0.0      │  -0.0        -0.0      │   ⋅           ⋅        │   ⋅          ⋅        │
 ──────┼───────────────────────┼────────────────────────┼────────────────────────┼───────────────────────┼  …
 -0.0  │  -1.48021   -0.0      │  -0.251841   -0.0      │  -0.0         ⋅        │   ⋅          ⋅        │
 -0.0  │  -0.0       -2.45674  │  -0.0         0.29837  │  -0.0        -0.0      │   ⋅          ⋅        │
 ──────┼───────────────────────┼────────────────────────┼────────────────────────┼───────────────────────┼
 -0.0  │  -0.251841  -0.0      │  -1.33887    -0.0      │   0.233944   -0.0      │  -0.0        ⋅        │     
 -0.0  │  -0.0        0.29837  │  -0.0       -14.3911   │  -0.0         1.31836  │  -0.0       -0.0      │
 ──────┼───────────────────────┼────────────────────────┼────────────────────────┼───────────────────────┼
  ⋅    │  -0.0       -0.0      │   0.233944   -0.0      │  -4.42688    -0.0      │   0.551643  -0.0      │  …  
  ⋅    │   ⋅         -0.0      │  -0.0         1.31836  │  -0.0       -35.9375   │  -0.0        3.02719  │
 ──────┼───────────────────────┼────────────────────────┼────────────────────────┼───────────────────────┼
  ⋅    │   ⋅          ⋅        │  -0.0        -0.0      │   0.551643   -0.0      │  -9.27904   -0.0      │     
  ⋮                                                ⋮                                         ⋱

julia> transform(P, θ -> exp(cos(4θ)))
setindex(ℵ₀-element FillArrays.Zeros{Float64, 1, Tuple{BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}} with indices 1:1:ℵ₀, 51-element Vector{Float64}, 51-element Base.OneTo{Int64}) with indices 1:1:ℵ₀:
  0.6065306597115399
 ───────────────────────
  1.1102230246251565e-16
  0.6936667625493042
 ───────────────────────
  2.220446049250313e-16
 -0.8532350981292589
 ───────────────────────
  1.1102230246251565e-16
  0.3437708747117796
 ───────────────────────
 -1.1102230246251565e-16
  ⋮

julia> Q = SemiclassicalJacobiArc(h, b + 1)
SemiclassicalJacobiArc{Float64} with weight 1 on ArcPolynomials.Arc{Float64}

julia> Q \ P
ℵ₀×ℵ₀-blocked ℵ₀×ℵ₀ BlockedMatrix{Float64, BandedMatrices.BandedMatrix{Float64, ArcPolynomials.SemiclassicalArcPolynomials.SemiclassicalJacobiArcConversionData{Float64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}, BlockedUnitRange{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}}:
 1.0  │  0.0   0.653987  │    ⋅         ⋅        │    ⋅          ⋅        │    ⋅         ⋅        │   ⋅
 ─────┼──────────────────┼───────────────────────┼────────────────────────┼───────────────────────┼─────────  …
  ⋅   │  1.0   0.0       │   0.41004    ⋅        │    ⋅          ⋅        │    ⋅         ⋅        │   ⋅
  ⋅   │   ⋅   -0.302194  │   0.0       0.471423  │    ⋅          ⋅        │    ⋅         ⋅        │   ⋅
 ─────┼──────────────────┼───────────────────────┼────────────────────────┼───────────────────────┼─────────
  ⋅   │   ⋅     ⋅        │  -0.26328   0.0       │   0.361689    ⋅        │    ⋅         ⋅        │   ⋅
  ⋅   │   ⋅     ⋅        │    ⋅       -0.354024  │   0.0        0.444512  │    ⋅         ⋅        │   ⋅
 ─────┼──────────────────┼───────────────────────┼────────────────────────┼───────────────────────┼─────────
  ⋅   │   ⋅     ⋅        │    ⋅         ⋅        │  -0.287268   0.0       │   0.347399   ⋅        │   ⋅       …
  ⋅   │   ⋅     ⋅        │    ⋅         ⋅        │    ⋅        -0.370856  │   0.0       0.433046  │   ⋅
 ─────┼──────────────────┼───────────────────────┼────────────────────────┼───────────────────────┼─────────
  ⋅   │   ⋅     ⋅        │    ⋅         ⋅        │    ⋅          ⋅        │  -0.296915  0.0       │  0.34046
 ⋮                                          ⋮                                                  ⋱

julia> # PiecewiseArcPolynomial

julia> points = [-π, -π/4, π/6, π]
4-element Vector{Float64}:
 -3.141592653589793
 -0.7853981633974483
  0.5235987755982988
  3.141592653589793

julia> P = PiecewiseArcPolynomial{1}(points)
PiecewiseArcPolynomial{1}([-3.141592653589793, -0.7853981633974483, 0.5235987755982988, 3.141592653589793])

julia> P[-0.2, Block(1)] # the hat functions
3-element Vector{Float64}:
 0.0
 0.5567096774790852
 0.44329032252091477

julia> P[-0.2, Block(2)] # first block of bubble functions
3-element Vector{Float64}:
 0.0
 0.9884514158667572
 0.0

julia> P[-0.2, Block(5)] # fourth block of bubble functions
3-element Vector{Float64}:
 0.0
 0.1213246231133819
 0.0

julia> P'P # mass matrix
ℵ₀×ℵ₀-blocked ℵ₀×ℵ₀ CyclicBBBArrowheadMatrix{Float64}:
  1.76881    0.347545   0.370741   │  0.766286  0.0       0.846126  │  -0.184264   0.0        …
  0.347545   1.27352    0.211532   │  0.766286  0.433169  0.0       │   0.184264  -0.0568865
  0.370741   0.211532   1.38122    │  0.0       0.433169  0.846126  │   0.0        0.0568865
 ──────────────────────────────────┼────────────────────────────────┼───────────────────────
  0.766286   0.766286   0.0        │  1.21364    ⋅         ⋅        │   0.0         ⋅
   ⋅         0.433169   0.433169   │   ⋅        0.690935   ⋅        │    ⋅         0.0
  0.846126    ⋅         0.846126   │   ⋅         ⋅        1.3368    │    ⋅          ⋅
 ──────────────────────────────────┼────────────────────────────────┼───────────────────────  …
 -0.184264   0.184264   0.0        │  0.0        ⋅         ⋅        │   0.197877    ⋅
   ⋅        -0.0568865  0.0568865  │   ⋅        0.0        ⋅        │    ⋅         0.0397816
  ⋮                                                    ⋮                                ⋱

julia> transform(P, θ -> sin(2θ) * exp(-cos(3θ) + sin(2θ)))
setindex(ℵ₀-element FillArrays.Zeros{Float64, 1, Tuple{BlockedOneTo{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}} with indices BlockedOneTo(3:3:+∞), 153-element Vector{Float64}, 153-element Base.OneTo{Int64}) with indices BlockedOneTo(3:3:+∞):
 -1.0138567975426973e-13
 -0.7461018060798491
  2.058925752795469
 ───────────────────────
  1.315335813550264
 -0.885194513247942
  0.3376776935489216
 ───────────────────────
 -2.699929596183211
 -2.081690442547321
  ⋮

julia> # PeriodicContinuousPolynomial

julia> points = [-π, -π/4, π/6, π]
4-element Vector{Float64}:
 -3.141592653589793
 -0.7853981633974483
  0.5235987755982988
  3.141592653589793

julia> P = PeriodicContinuousPolynomial{1}(points)
PeriodicContinuousPolynomial{1}([-3.141592653589793, -0.7853981633974483, 0.5235987755982988, 3.141592653589793])

julia> P[-0.2, Block(1)] # the hat functions
3-element Vector{Float64}:
 0.0
 0.5527887453682196
 0.4472112546317806

julia> P[-0.2, Block(2)] # first block of bubble functions
3-element Vector{Float64}:
 0.0
 0.4944266967248986
 0.0

julia> P[-0.2, Block(5)] # fourth block of bubble functions
3-element Vector{Float64}:
 0.0
 0.03813199854959483
 0.0

julia> weaklaplacian(P) # -diff(P)'diff(P)
ℵ₀×ℵ₀-blocked ℵ₀×ℵ₀ CyclicBBBArrowheadMatrix{Float64}:
 -0.806385   0.424413   0.381972  │    ⋅          ⋅         ⋅        │    ⋅          ⋅         ⋅   │  …  
  0.424413  -1.18836    0.763944  │    ⋅          ⋅         ⋅        │    ⋅          ⋅         ⋅   │
  0.381972   0.763944  -1.14592   │    ⋅          ⋅         ⋅        │    ⋅          ⋅         ⋅   │
 ─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────┼
   ⋅          ⋅          ⋅        │  -0.565884    ⋅         ⋅        │    ⋅          ⋅         ⋅   │
   ⋅          ⋅          ⋅        │    ⋅        -1.01859    ⋅        │    ⋅          ⋅         ⋅   │
   ⋅          ⋅          ⋅        │    ⋅          ⋅       -0.509296  │    ⋅          ⋅         ⋅   │
 ─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────┼  …
   ⋅          ⋅          ⋅        │    ⋅          ⋅         ⋅        │  -0.339531    ⋅         ⋅   │
   ⋅          ⋅          ⋅        │    ⋅          ⋅         ⋅        │    ⋅        -0.611155   ⋅   │
  ⋮                                                     ⋮                                    ⋱

julia> transform(P, θ -> sin(2θ) * exp(-cos(3θ) + sin(2θ)))
setindex(ℵ₀-element FillArrays.Zeros{Float64, 1, Tuple{BlockedOneTo{Int64, InfiniteArrays.InfStepRange{Int64, Int64}}}} with indices BlockedOneTo(3:3:+∞), 180-element Vector{Float64}, 180-element Base.OneTo{Int64}) with indices BlockedOneTo(3:3:+∞):
  3.8178239935175855e-15
 -0.7461018060799007
  2.058925752795751
 ───────────────────────
  2.5666569512533077
 -1.7575532146323862
  0.6548171497264742
 ───────────────────────
 -5.733838684506668
 -2.396366684454699
  ⋮

julia> 
```