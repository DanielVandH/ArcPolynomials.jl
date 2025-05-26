This folder contains the code used to generate the results in our paper _A sparse hp-finite element method for piecewise-smooth differential equations with periodic boundary conditions_ (https://arxiv.org/abs/2505.17849) by Daniel VandenHeuvel and Sheehan Olver. 

The code is given in `main.jl`. This script defines all the functions that produce the results and then finally runs a `main(args)` function that executes the code. The `args` argument should be a vector of strings that each specify the parameters for the run, namely:
- `"save"`: Include this argument to save the figures to the `paper/figures/` directory. If not included, the figures will be displayed but not saved.
- `"semiclassical_jacobi_figures"`: Include this argument to generate the figure showing the semiclassical Jacobi polynomials with $b = -1$.
- `"arc_polynomial_figures"`: Include this argument to generate the figure showing the arc polynomials with $b = -1$.
- `"hat_function_figures"`: Include this argument to generate the figure showing the linear trigonometric hat functions.
- `"rate_of_convergence_figures"`: Include this argument to generate the figures showing the rate of convergence examples for the three bases.
- `"screened_poisson"`: Include this argument to execute the results for the screened Poisson example.
- `"heat_equation"`: Include this argument to execute the results for the heat equation example.
- `"linear_schrodinger"`: Include this argument to execute the results for the linear Schrödinger equation example.
- `"convection_diffusion"`: Include this argument to execute the results for the convection-diffusion equation example.

If `args` is empty or `["save"]`, then all the functions will be executed (i.e. all the differential equation examples and the figure examples).

# Running the code

The steps that follow assume you are in the `paper` directory and on Julia 1.11, and are given only for Windows. The first way to run the code is to simply do

```julia
PS C:\Users\User\.julia\dev\ArcPolynomials\paper> julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.11.5 (2025-04-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> include("main.jl")
[ Info: [13:34:58]: Loading C:\Users\User\.julia\dev\ArcPolynomials\paper\main.jl
  Activating project at `C:\Users\User\.julia\dev\ArcPolynomials\paper`
main (generic function with 1 method)

julia> main([])
[ Info: [13:36:24]: Validating arguments
[ Info: [13:36:29]: Running semiclassical jacobi figures
[ Info: [13:36:57]: Running arc polynomial figures
[ Info: [13:36:58]: Running hat function figures
[ Info: [13:37:00]: Running rate of convergence figures
[ Info: [13:37:21]: Running screened poisson
[ Info: [13:38:11]: Running heat equation
[ Info: Loading heat_equation_resolved.jld2
[ Info: Loading heat_equation_underresolved.jld2
[ Info: [13:42:26]: Running linear schrodinger
[ Info: Loading schrodinger.jld2
[ Info: [13:51:21]: Running convection diffusion
[ Info: Loading convection_diffusion.jld2
[ Info: [14:01:17]: Done
0

julia> 
```

Alternatively, you can run the code directly from the command line without first entering the Julia REPL. In this form, the `args` are passed as command line arguments. For example, to run the code with all the figures saved and only with the linear Schrödinger and screened Poisson examples, you would run the following command (the equivalent in the first example above would be `main(["save", "linear_schrodinger", "screened_poisson"])`):

```julia
PS C:\Users\.julia\dev\ArcPolynomials\paper> julia main.jl save linear_schrodinger screened_poisson
[ Info: [13:22:53]: Loading C:\Users\User\.julia\dev\ArcPolynomials\paper\main.jl
  Activating project at `C:\Users\User\.julia\dev\ArcPolynomials\paper`
[ Info: [13:23:18]: Validating arguments
[ Info: [13:23:29]: Running screened poisson
[ Info: [13:24:45]: Running linear schrodinger
[ Info: Loading schrodinger.jld2
[ Info: [13:33:55]: Done
PS C:\Users\User\.julia\dev\ArcPolynomials\paper> 
```

This folder also includes `.jld2` files that save some expensive computations from the examples. If you want to regenerate these files, you can delete them and rerun the code.
