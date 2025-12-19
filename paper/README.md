This folder contains the code used to generate the results in our paper _A sparse $hp$-finite element method for piecewise-smooth differential equations with periodic boundary conditions_ by Daniel VandenHeuvel and Sheehan Olver. 

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
- `"screened_poisson_timings"`: Include this argument to execute the timing results for the screened Poisson example.
- `"eigenvalue_example"`: Include this argument to execute the eigenvalue problem example.

If `args` is empty or `["save"]`, then all the functions will be executed (i.e. all the differential equation examples and the figure examples).

# Running the code

The steps that follow assume you are in the `paper` directory and on Julia 1.12, and are given only for Windows. The first way to run the code is to simply do

```julia
PS \ArcPolynomials\paper> julia main.jl 
[ Info: [23:35:07]: Loading \ArcPolynomials\paper\main.jl
  Activating project at `\ArcPolynomials\paper`
[ Info: [23:37:10]: Validating arguments
[ Info: [23:37:16]: Running semiclassical jacobi figures
[ Info: [23:37:54]: Running arc polynomial figures
[ Info: [23:37:56]: Running hat function figures
[ Info: [23:38:00]: Running rate of convergence figures
[ Info: [23:38:24]: Running screened poisson
[ Info: [23:39:16]: Running heat equation
[ Info: Loading heat_equation_resolved.jld2
[ Info: Loading heat_equation_underresolved.jld2
[ Info: [23:40:13]: Running linear schrodinger
[ Info: Loading schrodinger.jld2
[ Info: [23:40:35]: Running convection diffusion
[ Info: Loading convection_diffusion.jld2
[ Info: [23:41:00]: Running screened poisson timings
[ Info: [23:41:06]: Running eigenvalue example
[ Info: [23:41:11]: Done
```

To run the code with all the figures saved and only with the linear Schrödinger and screened Poisson examples, you would run the following command

```julia
PS \ArcPolynomials\paper> julia main.jl save linear_schrodinger screened_poisson
[ Info: [23:43:55]: Loading \ArcPolynomials\paper\main.jl
  Activating project at `\ArcPolynomials\paper`
[ Info: [23:44:22]: Validating arguments
[ Info: [23:44:36]: Running screened poisson
[ Info: [23:46:11]: Running linear schrodinger
[ Info: Loading schrodinger.jld2
[ Info: [23:46:44]: Done
PS \ArcPolynomials\paper> 
```

This folder also includes `.jld2` files that save some expensive computations from the examples. If you want to regenerate these files, you can delete them and rerun the code.