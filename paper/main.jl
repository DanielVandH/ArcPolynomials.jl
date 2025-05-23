using Dates
ct() = Dates.format(now(), "HH:MM:SS")
@info "[$(ct())]: Loading $(@__FILE__)"

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using CairoMakie, ArcPolynomials, ContinuumArrays, LazyArrays,
    ClassicalOrthogonalPolynomials, BlockArrays, LinearAlgebra,
    InfiniteLinearAlgebra, ExponentialUtilities, Test, Dates,
    SemiclassicalOrthogonalPolynomials, FillArrays, OhMyThreads, JLD2,
    OrderedCollections, RecurrenceRelationshipArrays, BandedMatrices, LibGEOS,
    MatrixFactorizations, BlockBandedMatrices, SparseArrays, ForwardDiff

using LazyArrays: paddeddata
using InfiniteLinearAlgebra: pad
padc(x, ax) = pad(collect(x), ax)

function ContinuumArrays.plan_grid_transform(F::Fourier, szs::Tuple{Block{1,Int}}, dims=ntuple(identity, Val(1)))
    return ContinuumArrays.plan_grid_transform(F, last(axes(F, 2)[only(szs)]), dims)
end
function get_d2(P, n)
    dP = diff(P)
    Q, D = dP.args
    R = Q \ P
    RS = R[Block(1):Block(n), Block(1):Block(n)]
    DS = D[axes(RS)...]
    _d2P = BlockedArray(Float64.(BigFloat.(Matrix(DS)) * (BigFloat.(Matrix(RS)) \ BigFloat.(Matrix(DS)))), axes(RS))
    d2P = pad(_d2P, axes(R, 1), axes(R, 2))
    return Q * d2P
end
function get_d3(P, n)
    dP = diff(P)
    Q, D = dP.args
    R = Q \ P
    RS = R[Block(1):Block(n), Block(1):Block(n)]
    DS = D[axes(RS)...]
    _d3P = BlockedArray(Float64.(BigFloat.(Matrix(DS)) * (BigFloat.(Matrix(RS)) \ BigFloat.(Matrix(DS)))^2), axes(RS))
    d3P = pad(_d3P, axes(R, 1), axes(R, 2))
    return Q * d3P
end
function expm(F::Fourier, c)
    # Computes exp(c * diff(F).args[2]^2)
    return BlockArrays._BlockArray(
        Diagonal(
            Vcat(
                [reshape([1.0], 1, 1)],
                [[exp(-c * n^2) 0.0; 0.0 exp(-c * n^2)] for n in 1:∞]
            )
        ), (axes(F, 2), axes(F, 2))
    )
end
@test expm(Fourier(), 1.0)[1:101, 1:101] ≈ exp((diff(Fourier()).args[2]^2)[1:101, 1:101])
@test expm(Fourier(), 2.3im)[1:101, 1:101] ≈ exp(2.3im * (diff(Fourier()).args[2]^2)[1:101, 1:101])
@test expm(Fourier(), 2.3im + 0.2)[1:101, 1:101] ≈ exp((2.3im + 0.2) * (diff(Fourier()).args[2]^2)[1:101, 1:101])
to_bigfloat(x) = BigFloat(x)
to_bigfloat(x::Complex) = Complex{BigFloat}(x)
function drift(u)
    return abs(only(float.(diff(to_bigfloat.(u[[-π, π]])))))
end
function compute_drifts(u, P, dP, d2P, d3P)
    vals = (P * u.args[2], dP * u.args[2], d2P * u.args[2], d3P * u.args[2])
    return drift.(vals)
end

function semiclassical_jacobi_figures(savefig)
    z = LinRange(0, 1, 1000)
    P = SemiclassicalJacobi(2.0, -1 / 2, -1.0, -1 / 2)
    Q = SemiclassicalJacobi(2.0, 1 / 2, -1.0, 1 / 2)
    colors = [:blue, :red, :green, :purple, :orange]
    fig = Figure(fontsize=32)
    ax1 = Axis(fig[1, 1], width=600, height=300, xlabel=L"z", ylabel=L"P_n(z)", xticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]), yticks=([-2, 0, 2], [L"-2", L"0", L"2"]))
    ax2 = Axis(fig[1, 2], width=600, height=300, xlabel=L"z", ylabel=L"Q_n(z)", xticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]), yticks=([-2, 0, 2], [L"-2", L"0", L"2"]))
    linkaxes!(ax1, ax2)
    series!(ax1, z, P[z, 1:5]', solid_color=colors, linewidth=4)
    series!(ax2, z, Q[z, 1:5]', solid_color=colors, linewidth=4)
    Legend(fig[1, 3],
        [LineElement(color=clr, linewidth=4) for clr in colors],
        [L"n = 0", L"n = 1", L"n = 2", L"n = 3", L"n = 4"]
    )
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "semiclassical_jacobi.pdf"), fig)
end

function arc_polynomial_figures(savefig)
    h, b = 0.2, -1.0
    P = SemiclassicalJacobiArc(h, b)

    ϕ = acos(h) - eps()
    θ = LinRange(-ϕ, ϕ, 1000)

    colors = [:blue, :red, :green, :purple, :orange]
    fig = Figure(fontsize=32)
    ax1 = Axis(fig[1, 1], width=600, height=300, xlabel=L"\theta", ylabel=L"p_n(\theta)", xticks=([-1, 0, 1], [L"-1", L"0", L"1"]), yticks=([-1, -0.5, 0, 0.5, 1], [L"-1", L"-0.5", L"0", L"0.5", L"1"]))
    ax2 = Axis(fig[1, 2], width=600, height=300, xlabel=L"\theta", ylabel=L"q_n(\theta)", xticks=([-1, 0, 1], [L"-1", L"0", L"1"]), yticks=([-1, -0.5, 0, 0.5, 1], [L"-1", L"-0.5", L"0", L"0.5", L"1"]))
    linkaxes!(ax1, ax2)
    series!(ax1, θ, P[θ, 1:2:9]', solid_color=colors, linewidth=4)
    series!(ax2, θ, P[θ, 2:2:10]', solid_color=colors, linewidth=4)
    Legend(fig[1, 3],
        [LineElement(color=clr, linewidth=4) for clr in colors],
        [L"n = 0", L"n = 1", L"n = 2", L"n = 3", L"n = 4"]
    )
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "arc_polynomials.pdf"), fig)
end

function hat_function_figures(savefig)
    r = 3
    S = TrigonometricSpline(LinRange(-π, π, r + 1))
    θ = LinRange(-π, π, 2500)

    colors = [:blue, :red, :green]
    fig = Figure(fontsize=32)
    ax = Axis(fig[1, 1], width=900, height=300, xlabel=L"\theta", ylabel=L"\phi(\theta)", xticks=([-π, -π / 3, 0, π / 3, π], [L"-\pi", L"-\pi/3", L"0", L"\pi/3", L"\pi"]), yticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]))
    xlims!(ax, -π, π)
    ylims!(ax, -0.05, 1.25)
    series!(ax, θ, S[θ, 1:3]', solid_color=colors, linewidth=2)
    vlines!(ax, S.points, color=:black, linestyle=:dash)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "hat_functions.pdf"), fig)
end

function rate_of_convergence_figures(savefig)
    function ellipse_coordinates(ρ, x₀, ℓᵢ, θ)
        α = (ρ + 1 / ρ) / 2
        β = (ρ - 1 / ρ) / 2
        x = @. α * cos(θ)
        y = @. β * sin(θ)
        z = complex.(x, y)
        z′ = ℓᵢ / 2 * z .+ x₀
        x′, y′ = reim(z′)
        return x′, y′
    end
    ellipse_coordinates(ρ, 𝛉, θ, i::Int) = ellipse_coordinates(ρ, (𝛉[i] + 𝛉[i+1]) / 2, 𝛉[i+1] - 𝛉[i], θ)
    function dellipse_coordinates(ρ, x₀, ℓᵢ, θ)
        ϕᵢ = ℓᵢ / 2
        hᵢ = cos(ϕᵢ)
        x₀′ = (1 + hᵢ) / 2
        ℓᵢ′ = 1 - hᵢ
        x, y = ellipse_coordinates(ρ, x₀′, ℓᵢ′, θ)
        z = complex.(x, y)
        d = acos.(z)
        x, y = reim(d)
        @. x = x + x₀
        append!(x, 2x[end] .- reverse(x))
        append!(y, reverse(y))
        append!(x, reverse(x))
        append!(y, 2y[end] .- reverse(y))
        @. y = y - y[1]
        return x, y
    end
    dellipse_coordinates(ρ, 𝛉, i::Int, θ) = dellipse_coordinates(ρ, (𝛉[i] + 𝛉[i+1]) / 2, 𝛉[i+1] - 𝛉[i], θ)
    function expand_and_group(F::Fourier, f::T, ax::BlockRange) where {T}
        v = collect(transform(F[:, ax], f))
        cfs = Vector{Float64}[]
        push!(cfs, [v[1]])
        for i in 2:2:length(v)
            push!(cfs, [v[i], v[i+1]])
        end
        return cfs, 0:(length(cfs)-1)
    end
    function expand_and_group(W::PeriodicContinuousPolynomial{0}, f::T, ax::BlockRange) where {T}
        M = length(W.points) - 1
        v = reshape(collect(transform(W[:, ax], f)), (M, :))
        cfs = collect.(eachcol(v))
        return cfs, 0:(length(cfs)-1)
    end
    function expand_and_group(P::PiecewiseArcPolynomial{0}, f::T, ax::BlockRange) where {T}
        M = length(P.points) - 1
        v = reshape(collect(transform(P[:, ax], f)), (M, :))
        cfs = Vector{Float64}[]
        push!(cfs, v[:, 1])
        for i in 2:2:size(v, 2)
            push!(cfs, [v[:, i]; v[:, i+1]])
        end
        return cfs, 0:(length(cfs)-1)
    end
    function estimate_ρ(F::Fourier, z)
        dist = minimum(abs ∘ imag, z)
        return exp(dist)
    end
    function estimate_ρ(W::PeriodicContinuousPolynomial, z)
        ρ = minimum(1:(length(W.points)-1)) do i
            Eᵢ = (𝛉[i] + 𝛉[i+1]) / 2
            ℓᵢ = 𝛉[i+1] - 𝛉[i]
            ϕ = 2 / ℓᵢ * (z - Eᵢ)
            α = (abs(ϕ + 1) + abs(ϕ - 1)) / 2
            α + sqrt(α^2 - 1)
        end
        return ρ
    end
    function estimate_ρ(P::PiecewiseArcPolynomial, z)
        ρ = minimum(1:(length(P.points)-1)) do i
            Eᵢ = (𝛉[i] + 𝛉[i+1]) / 2
            zᵢ = z - Eᵢ
            yᵢ = cos(zᵢ)
            ℓᵢ = 𝛉[i+1] - 𝛉[i]
            ϕᵢ = ℓᵢ / 2
            hᵢ = cos(ϕᵢ)
            x₀′ = (1 + hᵢ) / 2
            ℓᵢ′ = 1 - hᵢ
            yᵢ′ = 2 / ℓᵢ′ * (yᵢ - x₀′)
            α = (abs(yᵢ′ + 1) + abs(yᵢ′ - 1)) / 2
            α + sqrt(α^2 - 1)
        end
        return ρ
    end

    𝛉 = [-π, -π / 2, 0, 1 / 2, π / 3, π]
    M = length(𝛉) - 1
    θ = LinRange(-π, π, 2500)
    θ′ = LinRange(-π, -eps(), 2500)

    ρ = [1.0, 2.0, 3.0]
    α = log.(ρ)

    P_poly_θ = []
    W_poly_θ = []
    P_poly_θ_union = []
    W_poly_θ_union = []

    for j in eachindex(ρ)
        W_polys = map(1:(length(𝛉)-1)) do i
            x, y = ellipse_coordinates(ρ[j], 𝛉, θ, i)
            poly = [[[x, y] for (x, y) in zip(x[begin:end-1], y[begin:end-1])]]
            push!(poly[1], poly[1][1])
            return LibGEOS.Polygon(poly)
        end
        P_polys = map(1:(length(𝛉)-1)) do i
            if ρ[j] > 1
                x, y = dellipse_coordinates(ρ[j], 𝛉, i, θ′)
            else
                x = collect(LinRange(𝛉[i], 𝛉[i+1], 2000))
                y = zeros(2000)
            end
            poly = [[[x, y] for (x, y) in zip(x[begin:end-1], y[begin:end-1])]]
            push!(poly[1], poly[1][1])
            return LibGEOS.Polygon(poly)
        end
        P_union = foldl(LibGEOS.union, P_polys)
        W_union = foldl(LibGEOS.union, W_polys)
        push!(P_poly_θ, P_polys)
        push!(W_poly_θ, W_polys)
        push!(P_poly_θ_union, P_union)
        push!(W_poly_θ_union, W_union)
    end

    fig = Figure(fontsize=31)
    ax11 = Axis(fig[1, 1], width=600, height=400, title=L"$ $Arc Polynomial", xlabel=L"Re$(\theta)$", ylabel=L"Im$(\theta)$", xticks=([-π, 0.0, π], [L"-\pi", L"0", L"\pi"]), yticks=(-1.5:1.5, [L"%$s" for s in -1.5:1.5]))
    ax12 = Axis(fig[1, 2], width=600, height=400, title=L"$ $Integrated Legendre", xlabel=L"Re$(\theta)$", ylabel=L"Im$(\theta)$", xticks=([-π, 0.0, π], [L"-\pi", L"0", L"\pi"]))
    ax13 = Axis(fig[1, 3], width=600, height=400, title=L"$ $Fourier", xlabel=L"Re$(\theta)$", ylabel=L"Im$(\theta)$", xticks=([-π, 0.0, π], [L"-\pi", L"0", L"\pi"]))
    linkaxes!(ax11, ax12, ax13)
    hideydecorations!.((ax12, ax13))
    colors = [:red, :blue, :black]
    for i in 1:3
        ρ = [1.0, 2.0, 3.0]
        α = log.(ρ)
        lines!(ax11, P_poly_θ_union[i], color=colors[i], linewidth=3)
        lines!(ax12, W_poly_θ_union[i], color=colors[i], linewidth=3)
        hlines!(ax13, [α[i], -α[i]], color=colors[i], linewidth=5)
    end
    ylims!(ax11, -1.6, 1.6)
    ylims!(ax12, -1.6, 1.6)
    ylims!(ax13, -1.6, 1.6)
    Label(fig[1, 1, Top()], L"$ $(a)", halign=:left)
    Label(fig[1, 2, Top()], L"$ $(b)", halign=:left)
    Label(fig[1, 3, Top()], L"$ $(c)", halign=:left)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "rate_of_convergence_regions.pdf"), fig)

    𝛉 = [-π, 0, π]
    M = length(𝛉) - 1
    F = Fourier()
    P = PiecewiseArcPolynomial{0}(𝛉)
    W = PeriodicContinuousPolynomial{0}(𝛉)
    y = 1 / 10
    f = let y = y
        x -> exp(sin(x)) / (cos(2x) - cosh(2y))
    end
    poles = [y * im, -y * im]
    vQax = ClassicalOrthogonalPolynomials.increasingtruncations(axes(P, 2))
    i = 5
    vfourier = collect(transform(F[:, vQax[i]], f))
    vlegendre = collect(transform(W[:, vQax[i]], f))
    varc = collect(transform(P[:, vQax[i]], f))
    nfourier = length(vfourier)
    nlegendre = length(vlegendre)
    narc = length(varc)
    fourier_idx = 0:(nfourier-1)
    legendre_idx = 0:(nlegendre-1)
    arc_idx = 0:(narc-1)
    fourier_deg, legendre_deg, arc_deg = Int[], Int[], Int[]
    fourier_maxes, legendre_maxes, arc_maxes = Float64[], Float64[], Float64[]
    push!(fourier_deg, 0)
    push!(fourier_maxes, abs(vfourier[1]))
    for i in 2:2:nfourier
        push!(fourier_deg, i ÷ 2, i ÷ 2)
        push!(fourier_maxes, max(abs(vfourier[i]), abs(vfourier[i+1])))
    end
    legendre_rsp = reshape(vlegendre, (M, :))
    for i in axes(legendre_rsp, 2)
        append!(legendre_deg, fill(i - 1, M))
        push!(legendre_maxes, maximum(abs.(legendre_rsp[:, i])))
    end
    append!(arc_deg, fill(0, M))
    push!(arc_maxes, maximum(abs.(varc[1:M])))
    arc_rsp = reshape(varc[M+1:end], (2M, :))
    for i in axes(arc_rsp, 2)
        append!(arc_deg, fill(i, 2M))
        push!(arc_maxes, maximum(abs.(arc_rsp[:, i])))
    end
    Fρ = minimum([estimate_ρ(F, y) for y in poles])
    Wρ = minimum([estimate_ρ(W, y) for y in poles])
    Pρ = minimum([estimate_ρ(P, y) for y in poles])
    Fslop = Fρ .^ (-unique(fourier_deg)) .* 3.0
    Wslop = Wρ .^ (-unique(legendre_deg)) .* 250.0
    Pslop = Pρ .^ (-unique(arc_deg)) .* 20.0
    Fslop_flat = Fρ .^ (-fourier_deg) .* 3.0
    Wslop_flat = Wρ .^ (-legendre_deg) .* 250.0
    Pslop_flat = Pρ .^ (-arc_deg) .* 20.0

    fig = Figure(fontsize=31)
    ax1 = Axis(fig[1, 1], width=800, height=400, yscale=log10,
        xlabel=L"$ $Degree", ylabel=L"\max|\hat{f}_n|",
        title=L"$ $(a) Polynomial degree", titlealign=:left)
    scatter!(ax1, unique(fourier_deg), fourier_maxes, color=:red, markersize=12, marker=:circle)
    scatter!(ax1, unique(legendre_deg), legendre_maxes, color=:green, markersize=12, marker=:rect)
    scatter!(ax1, unique(arc_deg), arc_maxes, color=:black, markersize=12, marker=:+)
    lines!(ax1, unique(fourier_deg), Fslop, color=:red, linewidth=4)
    lines!(ax1, unique(legendre_deg), Wslop, color=:green, linewidth=4)
    lines!(ax1, unique(arc_deg), Pslop, color=:black, linewidth=4)
    text!(ax1, [20], [1e-5 / 6], text=[L"\rho = %$(round(Pρ, digits = 4))"], rotation=-0.3)
    text!(ax1, [50], [1e-1], text=[L"\rho = %$(round(Fρ, digits = 4))"])
    text!(ax1, [50], [1e-5], text=[L"\rho = %$(round(Wρ, digits = 4))"], rotation=-0.25)
    xlims!(ax1, 0, 64)
    ylims!(ax1, 1e-16, 1e6)

    ax2 = Axis(fig[1, 2], width=800, height=400, yscale=log10,
        xlabel=L"$ $Degrees of freedom", ylabel=L"|\hat{f}_n|",
        title=L"$ $(b) Degrees of freedom", titlealign=:left)
    scatter!(ax2, fourier_idx, abs.(vfourier), color=:red, markersize=12, marker=:circle)
    scatter!(ax2, legendre_idx, abs.(vlegendre), color=:green, markersize=12, marker=:rect)
    scatter!(ax2, arc_idx, abs.(varc), color=:black, markersize=12, marker=:+)
    figle = lines!(ax2, legendre_idx, Wslop_flat, color=:green, linewidth=4)
    figar = lines!(ax2, arc_idx, Pslop_flat, color=:black)
    figfo = lines!(ax2, fourier_idx, Fslop_flat, color=:red, linewidth=4)
    xlims!(ax2, 0, 360)
    ylims!(ax2, 1e-16, 1e6)

    L = Legend(fig[2, 1:2], [figfo, figle, figar], ["Fourier", "Integrated Legendre", "Arc Polynomial"], fontsize=31, orientation=:horizontal)

    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "rate_of_convergence_slopes.pdf"), fig)
end

function screened_poisson(savefig)
    ω = 3 / 2
    𝛉 = LinRange(-π, π, 10)
    M = length(𝛉) - 1
    f = θ -> 2 + sign(abs(θ) - π / 3)
    P = PiecewiseArcPolynomial{1}(𝛉)
    W = PeriodicContinuousPolynomial{1}(𝛉)
    F = Fourier()
    P0 = PiecewiseArcPolynomial{0}(𝛉)
    W0 = PeriodicContinuousPolynomial{0}(𝛉)

    fP, fW = transform.((P0, W0), f)
    np, nw = length.(paddeddata.((fP, fW))) .+ 15M

    ΔP, ΔW = -diff(P)'diff(P), -diff(W)'diff(W)
    MP, MW = P'P, W'W
    MP0, MW0 = P0'P0, W0'W0
    RP, RW = P0 \ P, W0 \ W

    lhsP, rhsP = -ΔP + ω^2 * MP, RP' * MP0 * fP
    lhsW, rhsW = -ΔW + ω^2 * MW, RW' * MW0 * fW

    lhsPtc = principal_submatrix(lhsP, Block(np ÷ M))
    rhsPtc = BlockedVector(Vector(rhsP[1:np]), blocksizes(lhsPtc, 1))
    lhsWtc = principal_submatrix(lhsW, Block(nw ÷ M))
    rhsWtc = BlockedVector(Vector(rhsW[1:nw]), blocksizes(lhsWtc, 1))
    uPsoltc = ldiv!(reversecholesky!(Symmetric(lhsPtc)), rhsPtc)
    uWsoltc = ldiv!(reversecholesky!(Symmetric(lhsWtc)), rhsWtc)
    uP, uW = P * padc(uPsoltc, axes(P, 2)), W * padc(uWsoltc, axes(W, 2))

    d2P, d2W = get_d2(P, 15), get_d2(W, 15)
    duP, duW = diff(uP), diff(uW)
    d2uP, d2uW = d2P * uP.args[2], d2W * uW.args[2]

    θ = LinRange(-π, π, 100)
    function exact_soln(θ)
        if θ < -π / 3
            return 3 / ω^2 - (2 * cosh(pi * ω + θ * ω)) / (ω^2 * (2 * cosh((2 * pi * ω) / 3) + 1))
        elseif -π / 3 < θ < π / 3
            return 1 / ω^2 + (4 * cosh((pi * ω) / 3) * cosh(θ * ω)) / (ω^2 * (2 * cosh((2 * pi * ω) / 3) + 1))
        elseif θ > π / 3
            return 3 / ω^2 - (2 * cosh(pi * ω - θ * ω)) / (ω^2 * (2 * cosh((2 * pi * ω) / 3) + 1))
        end
    end
    uFθ = exact_soln.(θ)
    duFθ = ForwardDiff.derivative.(exact_soln, θ)
    d2uFθ = ForwardDiff.derivative.(θ -> ForwardDiff.derivative(exact_soln, θ), θ)

    colors = [:red, :blue, :black]
    styles = [:solid, :dash, :dashdot]
    fig = Figure(fontsize=40)
    ax1 = Axis(fig[1, 1], width=600, height=300, xlabel=L"\theta", ylabel=L"|u - \hat{u}|", yscale=log10, yticks=([1e-17, 1e-16, 1e-15, 1e-14], [L"10^{-17}", L"10^{-16}", L"10^{-15}", L"10^{-14}"]), xticks=([-2, 0, 2], [L"-2", L"0", L"2"]))
    ax2 = Axis(fig[1, 2], width=600, height=300, xlabel=L"\theta", ylabel=L"|u' - \hat{u}'|", yscale=log10, yticks=([1e-17, 1e-16, 1e-15, 1e-14], [L"10^{-17}", L"10^{-16}", L"10^{-15}", L"10^{-14}"]), xticks=([-2, 0, 2], [L"-2", L"0", L"2"]))
    ax3 = Axis(fig[1, 3], width=600, height=300, xlabel=L"\theta", ylabel=L"|u'' - \hat{u}''|", yscale=log10, yticks=([1e-16, 1e-15, 1e-14, 1e-13, 1e-12], [L"10^{-16}", L"10^{-15}", L"10^{-14}", L"10^{-13}", L"10^{-12}"]), xticks=([-2, 0, 2], [L"-2", L"0", L"2"]))
    linkxaxes!(ax1, ax2, ax3)

    uPθ, uWθ = uP[θ], uW[θ]
    duPθ, duWθ = duP[θ], duW[θ]
    d2uPθ, d2uWθ = d2uP[θ], d2uW[θ]

    err(u, uf) = abs(u - uf)
    uPerr, uWerr = err.(uPθ, uFθ), err.(uWθ, uFθ)
    duPerr, duWerr = err.(duPθ, duFθ), err.(duWθ, duFθ)
    d2uPerr, d2uWerr = err.(d2uPθ, d2uFθ), err.(d2uWθ, d2uFθ)

    lines!(ax1, θ, uPerr, color=:red, linestyle=:solid, linewidth=4)
    lines!(ax1, θ, uWerr, color=:blue, linestyle=:dash, linewidth=4)
    lines!(ax2, θ, duPerr, color=:red, linestyle=:solid, linewidth=4)
    lines!(ax2, θ, duWerr, color=:blue, linestyle=:dash, linewidth=4)
    lines!(ax3, θ, d2uPerr, color=:red, linestyle=:solid, linewidth=4)
    lines!(ax3, θ, d2uWerr, color=:blue, linestyle=:dash, linewidth=4)

    Label(fig[1, 1, Top()], L"$ $(a)", halign=:left)
    Label(fig[1, 2, Top()], L"$ $(b)", halign=:left)
    Label(fig[1, 3, Top()], L"$ $(c)", halign=:left)

    ylims!(ax1, 1e-17, 1e-14)
    ylims!(ax2, 1e-17, 1e-14)
    ylims!(ax3, 1e-16, 1e-12)

    Legend(fig[1, 4],
        [LineElement(color=clr, linestyle=style, linewidth=4) for (clr, style) in zip(colors[1:2], styles[1:2])],
        [L"P^{(-1), \theta}", L"W^{(-1), \theta}"]
    )
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "screened_poisson.pdf"), fig)
end

function linear_schrodinger(savefig)
    𝛉 = LinRange(-π, π, 4)
    f = θ -> sin(7θ) + exp(-cos(θ))
    P = PiecewiseArcPolynomial{1}(𝛉)
    W = PeriodicContinuousPolynomial{1}(𝛉)
    F = Fourier()

    M = length(𝛉) - 1
    fF, fP, fW = transform.((F, P, W), f)
    np, nw = length.(paddeddata.((fP, fW))) .+ 2M

    ΔP, ΔW = -diff(P)'diff(P), -diff(W)'diff(W)
    MP, MW = P'P, W'W
    D = F \ diff(F)

    MPtc, ΔPtc = principal_submatrix(MP, Block(np ÷ M)), principal_submatrix(ΔP, Block(np ÷ M))
    MPchol = reversecholesky!(Symmetric(MPtc)).factors
    argP = -im * (MPchol' \ (MPchol \ ΔPtc))
    MWtc, ΔWtc = principal_submatrix(MW, Block(nw ÷ M)), principal_submatrix(ΔW, Block(nw ÷ M))
    MWchol = reversecholesky!(Symmetric(MWtc)).factors
    argW = -im * (MWchol' \ (MWchol \ ΔWtc))

    fPn = complex.(collect(fP[1:np]))
    fWn = complex.(collect(fW[1:nw]))

    trange = collect(LinRange(0, 25, 20001))
    if !isfile(joinpath(@__DIR__, "schrodinger.jld2"))
        @info "schrodinger.jld2 not found. Computing the matrix exponentials. This may take a very long time."
        let trange, argP, argW, fPn, fWn, P, W, F
            @time uP, uW = tmap(1:2) do i
                arg = (argP, argW)[i]
                u0 = (fPn, fWn)[i]
                B = (P, W)[i]
                res = expv_timestep(trange, arg, u0, verbose=true, tau=1e-6, adaptive=false)
                [B * pad(u, axes(B, 2)) for u in eachcol(res)]
                # [B * pad(expv(t, arg, u0, tol=1e-13, m = 50), axes(B, 2)) for t in trange]
                # Not using the faster alternative above since, for W, the solution is not accurate for larger t 
            end
            uF = [F * (expm(F, -im * t) * fF) for t in trange]
            @save joinpath(@__DIR__, "schrodinger.jld2") uP uW uF
        end
    end
    @info "Loading schrodinger.jld2"
    @load joinpath(@__DIR__, "schrodinger.jld2") uP uW uF

    nt = length(trange)
    Pd, dPd, d2Pd, d3Pd, = [zeros(nt) for _ in 1:4]
    Wd, dWd, d2Wd, d3Wd = [zeros(nt) for _ in 1:4]
    Fd, dFd, d2Fd, d3Fd = [zeros(nt) for _ in 1:4]
    dP, dW = diff(P), diff(W)
    d2P, d2W = get_d2(P, 31), get_d2(W, 31)
    d3P, d3W = get_d3(P, 31), get_d3(W, 31)
    for j in eachindex(trange)
        Pd[j], dPd[j], d2Pd[j], d3Pd[j] = compute_drifts(uP[j], P, dP, d2P, d3P)
        Wd[j], dWd[j], d2Wd[j], d3Wd[j] = compute_drifts(uW[j], W, dW, d2W, d3W)
        Fd[j], dFd[j], d2Fd[j], d3Fd[j] = compute_drifts(uF[j], F, F * D, F * D^2, F * D^3)
    end

    θ = LinRange(-π, π, 250)
    colors = [:blue, :red, :green, :purple, :orange, :black]
    fig = Figure(fontsize=32)
    ax1 = Axis(fig[1, 1], width=600, height=300, xlabel=L"\theta", ylabel=L"Re $u(t)$", title=L"Re $u(t)$")
    ax2 = Axis(fig[1, 2], width=600, height=300, xlabel=L"\theta", ylabel=L"Re $u(t)$", title=L"Re $u(t)$")
    ax3 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u$ Drift", title=L"$u$ Drift", yscale=log10)
    ax4 = Axis(fig[2, 2], width=600, height=300, xlabel=L"t", ylabel=L"$u'$ Drift", title=L"$u'$ Drift", yscale=log10)
    ax5 = Axis(fig[3, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u''$ Drift", title=L"$u''$ Drift", yscale=log10)
    ax6 = Axis(fig[3, 2], width=600, height=300, xlabel=L"t", ylabel=L"$u'''$ Drift", title=L"$u'''$ Drift", yscale=log10)
    linkxaxes!(ax1, ax2)
    linkaxes!(ax3, ax4, ax5, ax6)
    hideydecorations!(ax2, grid=false, minorgrid=false, minorticks=false)
    hidexdecorations!.((ax3, ax4), grid=false, minorgrid=false, minorticks=false)
    ts = 0:5:25
    tidx = findall(t -> t in ts, trange)
    for i in 1:6
        series!(ax1, θ, real(uP[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:solid)
        series!(ax1, θ, real(uW[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:dash)
        series!(ax1, θ, real(uF[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:dashdot)
        series!(ax2, θ, imag(uP[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:solid)
        series!(ax2, θ, imag(uW[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:dash)
        series!(ax2, θ, imag(uF[tidx[i]][θ])', solid_color=colors[i], linewidth=4, linestyle=:dashdot)
    end
    lines!(ax3, trange, Pd, color=(:red, 0.7), linestyle=:solid, linewidth=4)
    lines!(ax3, trange, Wd, color=(:blue, 0.7), linestyle=:dash, linewidth=4)
    lines!(ax3, trange, Fd, color=(:green, 0.7), linestyle=:dashdot, linewidth=4)
    lines!(ax4, trange, dPd, color=(:red, 0.7), linestyle=:solid, linewidth=4)
    lines!(ax4, trange, dWd, color=(:blue, 0.7), linestyle=:dash, linewidth=4)
    lines!(ax4, trange, dFd, color=(:green, 0.7), linestyle=:dashdot, linewidth=4)
    lines!(ax5, trange, d2Pd, color=(:red, 0.7), linestyle=:solid, linewidth=4)
    lines!(ax5, trange, d2Wd, color=(:blue, 0.7), linestyle=:dash, linewidth=4)
    lines!(ax5, trange, d2Fd, color=(:green, 0.7), linestyle=:dashdot, linewidth=4)
    lines!(ax6, trange, d3Pd, color=(:red, 0.7), linestyle=:solid, linewidth=4)
    lines!(ax6, trange, d3Wd, color=(:blue, 0.7), linestyle=:dash, linewidth=4)
    lines!(ax6, trange, d3Fd, color=(:green, 0.7), linestyle=:dashdot, linewidth=4)
    Legend(fig[1, 3],
        [LineElement(color=clr, linewidth=4) for clr in colors], [L"t = %$(t)" for t in ts]
    )
    Legend(fig[2:3, 3],
        [LineElement(color=clr, linewidth=4) for clr in (:red, :blue, :green)],
        [L"P^{(-1), \theta}", L"W^{(-1), \theta}", L"F"]
    )
    Label(fig[1, 1, Top()], L"$ $(a)", halign=:left)
    Label(fig[1, 2, Top()], L"$ $(b)", halign=:left)
    Label(fig[2, 1, Top()], L"$ $(c)", halign=:left)
    Label(fig[2, 2, Top()], L"$ $(d)", halign=:left)
    Label(fig[3, 1, Top()], L"$ $(e)", halign=:left)
    Label(fig[3, 2, Top()], L"$ $(f)", halign=:left)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "schrodinger.pdf"), fig)
end

function convection_diffusion(savefig)
    𝛉 = [-π, -π / 4, π / 4, π]
    f = θ -> exp(-cos(4θ)) * (abs(θ) ≤ π / 4 ? sin(3θ) : sin(θ))
    v = θ -> -sin(θ) / 1000
    P = PiecewiseArcPolynomial{1}(𝛉)
    W = PeriodicContinuousPolynomial{1}(𝛉)
    F = Fourier()

    M = length(𝛉) - 1
    fF, fP, fW = let f = f
        transform.((F, P, W), (θ -> f(rem2pi(θ, RoundNearest)), f, f))
    end
    np, nw = length.(paddeddata.((fP, fW))) .+ 2M

    QP, QW = PiecewiseArcPolynomial{0}(P), PeriodicContinuousPolynomial{0}(W)
    vP, vW = expand.((QP, QW), v)

    ΔP, ΔW = weaklaplacian(P), weaklaplacian(W)
    DP, DW = diff(P).args[2], diff(W).args[2]
    MP1, MW1 = P'P, W'W
    MP0, MW0 = QP'QP, QW'QW
    JP, JW = (vP.*QP).args[2], (vW.*QW).args[2]
    RP, RW = QP \ P, QW \ W
    Pop = ΔP - RP' * MP0 * JP * DP
    Wop = ΔW - RW' * MW0 * JW * DW

    argP = Matrix(MP1[1:np, 1:np]) \ Matrix(Pop[1:np, 1:np])
    argW = Matrix(MW1[1:nw, 1:nw]) \ Matrix(Wop[1:nw, 1:nw])
    fPn, fWn = collect(fP[1:np]), collect(fW[1:nw])
    trange = collect(LinRange(0, 2.5, 5001))
    if !isfile(joinpath(@__DIR__, "convection_diffusion.jld2"))
        @info "convection_diffusion.jld2 not found. Computing the matrix exponentials. This may take a very long time."
        let trange, argP, argW, fPn, fWn, P, W
            @time uP, uW = tmap(1:2) do i
                arg = (argP, argW)[i]
                u0 = (fPn, fWn)[i]
                B = (P, W)[i]
                res = expv_timestep(trange, arg, u0, verbose=true, tau=1e-6, adaptive=false)
                [B * pad(u, axes(B, 2)) for u in eachcol(res)]
            end
            @save joinpath(@__DIR__, "convection_diffusion.jld2") uP uW
        end
    end
    @info "Loading convection_diffusion.jld2"
    @load joinpath(@__DIR__, "convection_diffusion.jld2") uP uW

    nt = length(trange)
    Pd, dPd, d2Pd, d3Pd = [zeros(nt) for _ in 1:4]
    Wd, dWd, d2Wd, d3Wd = [zeros(nt) for _ in 1:4]
    dP, dW = diff(P), diff(W)
    d2P, d2W = get_d2(P, 29), get_d2(W, 11) # _, 11
    d3P, d3W = get_d3(P, 45), get_d3(W, 57)
    for j in eachindex(trange)
        Pd[j], dPd[j], d2Pd[j], d3Pd[j] = compute_drifts(uP[j], P, dP, d2P, d3P)
        Wd[j], dWd[j], d2Wd[j], d3Wd[j] = compute_drifts(uW[j], W, dW, d2W, d3W)
    end

    θ = LinRange(-π, π, 250)
    colors = [:blue, :red, :green, :purple, :orange, :black]
    fig = Figure(fontsize=32)
    ax1 = Axis(fig[1, 1:2], width=1200, height=300, xlabel=L"\theta", ylabel=L"u(t)", title=L"$ $Solution")
    ax4 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u''$ Drift", title=L"$u''$ Drift", yscale=log10)
    ax5 = Axis(fig[2, 2], width=600, height=300, xlabel=L"t", ylabel=L"$u'''$ Drift", title=L"$u'''$ Drift", yscale=log10)
    ts = LinRange(trange[1], trange[end], 6)
    tidx = [findfirst(trange .== t) for t in ts]
    series!(ax1, θ, stack([uP[i][θ] for i in tidx])', color=colors, linestyle=:solid, linewidth=4)
    series!(ax1, θ, stack([uW[i][θ] for i in tidx])', color=colors, linestyle=:dash, linewidth=4)
    Legend(fig[1, 3], [LineElement(color=clr, linewidth=4) for clr in colors], [L"t = %$(t)" for t in ts])
    Legend(fig[2, 3], [LineElement(color=clr, linewidth=4) for clr in (:red, :blue)], [L"P^{(-1), \theta}", L"W^{(-1), \theta}"])
    lines!(ax4, trange, d2Pd, color=(:red, 0.7))
    lines!(ax4, trange, d2Wd, color=(:blue, 0.7))
    lines!(ax5, trange, d3Pd, color=(:red, 0.7))
    lines!(ax5, trange, d3Wd, color=(:blue, 0.7))
    Label(fig[1, 1:2, Top()], L"$ $(a)", halign=:left)
    Label(fig[2, 1, Top()], L"$ $(b)", halign=:left)
    Label(fig[2, 2, Top()], L"$ $(c)", halign=:left)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "convection_diffusion.pdf"), fig)
end

function heat_equation(savefig)
    h = 0.5 * 1e-2
    𝛉 = [-π, -π / 4 - h, -π / 4 + h, π / 4 - h, π / 4 + h, π]
    spl = ContinuumArrays.LinearSpline(𝛉)
    splc = spl * [0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
    f = let splc = splc
        θ -> splc[θ]
    end
    P = PiecewiseArcPolynomial{1}(𝛉)
    W = PeriodicContinuousPolynomial{1}(𝛉)

    ## Resolved
    M = length(𝛉) - 1
    fP, fW = let f = f
        transform.((P, W), (f, f))
    end
    np, nw = length.(paddeddata.((fP, fW))) .+ 2M

    ΔP, ΔW = weaklaplacian(P), weaklaplacian(W)
    MP, MW = P'P, W'W

    ΔPtc = Symmetric(principal_submatrix(ΔP, Block(np ÷ M)))
    ΔWtc = Symmetric(principal_submatrix(ΔW, Block(nw ÷ M)))
    MPtc = Symmetric(principal_submatrix(MP, Block(np ÷ M)))
    MWtc = Symmetric(principal_submatrix(MW, Block(nw ÷ M)))

    argP = sparse(MPtc) \ ΔPtc
    argW = sparse(MWtc) \ ΔWtc
    fPn, fWn = collect(fP[1:np]), collect(fW[1:nw])
    trange = collect(LinRange(0, 2.5, 5001))
    if !isfile(joinpath(@__DIR__, "heat_equation_resolved.jld2"))
        @info "heat_equation_resolved.jld2 not found. Computing the matrix exponentials."
        let trange = trange, fWn = fWn, fPn = fPn, argW = argW, argP = argP
            @time uP, uW = tmap(1:2) do i
                local arg, u0, B
                arg = (argP, argW)[i]
                u0 = (fPn, fWn)[i]
                B = (P, W)[i]
                res = expv_timestep(trange, arg, u0, verbose=true, tau=1e-6, tol=1e-44, adaptive=false)
                [B * pad(u, axes(B, 2)) for u in eachcol(res)]
            end
            @save joinpath(@__DIR__, "heat_equation_resolved.jld2") uP uW
        end
    end
    @info "Loading heat_equation_resolved.jld2"
    @load joinpath(@__DIR__, "heat_equation_resolved.jld2") uP uW

    nt = length(trange)
    Pd, dPd, d2Pd, d3Pd = [zeros(nt) for _ in 1:4]
    Wd, dWd, d2Wd, d3Wd = [zeros(nt) for _ in 1:4]
    dP, dW = diff(P), diff(W)
    d2P, d2W = get_d2(P, 29), get_d2(W, 11) # _, 11
    d3P, d3W = get_d3(P, 45), get_d3(W, 57)
    for j in eachindex(trange)
        Pd[j], dPd[j], d2Pd[j], d3Pd[j] = compute_drifts(uP[j], P, dP, d2P, d3P)
        Wd[j], dWd[j], d2Wd[j], d3Wd[j] = compute_drifts(uW[j], W, dW, d2W, d3W)
    end

    θ = LinRange(-π, π, 250)
    colors = [:blue, :red, :green, :purple, :orange, :black]
    fig = Figure(fontsize=42)
    ax1 = Axis(fig[1, 1:3], width=1800, height=300, xlabel=L"\theta", ylabel=L"u(t)", title=L"$ $Solution", xticks=([-π, 0, π], [L"-\pi", L"0", L"\pi"]), yticks=([0, 1, 2], [L"0", L"1", L"2"]))
    # ax2 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u$ Drift", title=L"$u$ Drift", yscale=log10)
    ax3 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u'$ Drift", title=L"$u'$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-15, 1e-10, 1e-5, 1e0], [L"10^{-15}", L"10^{-10}", L"10^{-5}", L"1"]))
    ax4 = Axis(fig[2, 2], width=600, height=300, xlabel=L"t", ylabel=L"$u''$ Drift", title=L"$u''$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-15, 1e-10, 1e-5, 1e0], [L"10^{-15}", L"10^{-10}", L"10^{-5}", L"1"]))
    ax5 = Axis(fig[2, 3], width=600, height=300, xlabel=L"t", ylabel=L"$u'''$ Drift", title=L"$u'''$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-10, 1e-5, 1e0, 1e5, 1e10], [L"10^{-10}", L"10^{-5}", L"1", L"10^{5}", L"10^{10}"]))
    ts = [0, 0.5, 1, 1.5, 2, 2.5]
    tidx = [searchsortedfirst(trange, t) for t in ts]
    series!(ax1, θ, stack([uP[i][θ] for i in tidx])', color=colors, linestyle=:solid, linewidth=4)
    series!(ax1, θ, stack([uW[i][θ] for i in tidx])', color=colors, linestyle=:dash, linewidth=4)
    Legend(fig[1, 4], [LineElement(color=clr, linewidth=4) for clr in colors], [L"t = %$(t)" for t in ts])
    Legend(fig[2, 4], [LineElement(color=clr, linewidth=4) for clr in (:red, :blue)], [L"P^{(-1), \theta}", L"W^{(-1), \theta}"])
    #lines!(ax2, trange, Pd, color=(:red, 0.7))
    #lines!(ax2, trange, Wd, color=(:blue, 0.7))
    lines!(ax3, trange, dPd, color=(:red, 0.7))
    lines!(ax3, trange, dWd, color=(:blue, 0.7))
    lines!(ax4, trange, d2Pd, color=(:red, 0.7))
    lines!(ax4, trange, d2Wd, color=(:blue, 0.7))
    lines!(ax5, trange, d3Pd, color=(:red, 0.7))
    lines!(ax5, trange, d3Wd, color=(:blue, 0.7))
    #ylims!(ax4, 1e-16, 1e-7)
    Label(fig[1, 1:3, Top()], L"$ $(a)", halign=:left)
    Label(fig[2, 1, Top()], L"$ $(b)", halign=:left)
    Label(fig[2, 2, Top()], L"$ $(c)", halign=:left)
    Label(fig[2, 3, Top()], L"$ $(d)", halign=:left)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "heat_equation_resolved.pdf"), fig)

    ## Underresolved
    np, nw = 20, 5

    ΔP, ΔW = weaklaplacian(P), weaklaplacian(W)
    MP, MW = P'P, W'W

    ΔPtc = Symmetric(Matrix(ΔP[1:np, 1:np]))
    ΔWtc = Symmetric(Matrix(ΔW[1:nw, 1:nw]))
    MPtc = Symmetric(Matrix(MP[1:np, 1:np]))
    MWtc = Symmetric(Matrix(MW[1:nw, 1:nw]))

    argP = sparse(MPtc) \ ΔPtc
    argW = sparse(MWtc) \ ΔWtc
    fPn, fWn = collect(fP[1:np]), collect(fW[1:nw])
    if !isfile(joinpath(@__DIR__, "heat_equation_underresolved.jld2"))
        @info "heat_equation_underresolved.jld2 not found. Computing the matrix exponentials."
        let trange = trange, fWn = fWn, fPn = fPn, argW = argW, argP = argP
            @time uP, uW = tmap(1:2) do i
                local arg, u0, B
                arg = (argP, argW)[i]
                u0 = (fPn, fWn)[i]
                B = (P, W)[i]
                res = expv_timestep(trange, arg, u0, verbose=true, tau=1e-6, tol=1e-44, adaptive=false)
                [B * pad(u, axes(B, 2)) for u in eachcol(res)]
            end
            @save joinpath(@__DIR__, "heat_equation_underresolved.jld2") uP uW
        end
    end
    @info "Loading heat_equation_underresolved.jld2"
    @load joinpath(@__DIR__, "heat_equation_underresolved.jld2") uP uW

    nt = length(trange)
    Pd, dPd, d2Pd, d3Pd = [zeros(nt) for _ in 1:4]
    Wd, dWd, d2Wd, d3Wd = [zeros(nt) for _ in 1:4]
    dP, dW = diff(P), diff(W)
    d2P, d2W = get_d2(P, 29), get_d2(W, 11) # _, 11
    d3P, d3W = get_d3(P, 45), get_d3(W, 57)
    for j in 1:length(trange)
        Pd[j], dPd[j], d2Pd[j], d3Pd[j] = compute_drifts(uP[j], P, dP, d2P, d3P)
        Wd[j], dWd[j], d2Wd[j], d3Wd[j] = compute_drifts(uW[j], W, dW, d2W, d3W)
    end

    θ = LinRange(-π, π, 250)
    colors = [:blue, :red, :green, :purple, :orange, :black]
    fig = Figure(fontsize=42)
    ax1 = Axis(fig[1, 1:3], width=1800, height=300, xlabel=L"\theta", ylabel=L"u(t)", title=L"$ $Solution", xticks=([-π, 0, π], [L"-\pi", L"0", L"\pi"]), yticks=([0, 1, 2], [L"0", L"1", L"2"]))
    # ax2 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u$ Drift", title=L"$u$ Drift", yscale=log10)
    ax3 = Axis(fig[2, 1], width=600, height=300, xlabel=L"t", ylabel=L"$u'$ Drift", title=L"$u'$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-15, 1e-10, 1e-5, 1e0], [L"10^{-15}", L"10^{-10}", L"10^{-5}", L"1"]))
    ax4 = Axis(fig[2, 2], width=600, height=300, xlabel=L"t", ylabel=L"$u''$ Drift", title=L"$u''$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-15, 1e-10, 1e-5, 1e0], [L"10^{-15}", L"10^{-10}", L"10^{-5}", L"1"]))
    ax5 = Axis(fig[2, 3], width=600, height=300, xlabel=L"t", ylabel=L"$u'''$ Drift", title=L"$u'''$ Drift", yscale=log10, xticks=([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]), yticks=([1e-10, 1e-5, 1e0, 1e5, 1e10], [L"10^{-10}", L"10^{-5}", L"1", L"10^{5}", L"10^{10}"]))
    tidx = [searchsortedfirst(trange[1:end], t) for t in ts]
    series!(ax1, θ, stack([uP[i][θ] for i in tidx])', color=colors, linestyle=:solid, linewidth=4)
    series!(ax1, θ, stack([uW[i][θ] for i in tidx])', color=colors, linestyle=:dash, linewidth=4)
    Legend(fig[1, 4], [LineElement(color=clr, linewidth=4) for clr in colors], [L"t = %$(t)" for t in ts])
    Legend(fig[2, 4], [LineElement(color=clr, linewidth=4) for clr in (:red, :blue)], [L"P^{(-1), \theta}", L"W^{(-1), \theta}"])
    #lines!(ax2, trange, Pd, color=(:red, 0.7))
    #lines!(ax2, trange, Wd, color=(:blue, 0.7))
    lines!(ax3, trange, dPd, color=(:red, 0.7))
    lines!(ax3, trange, dWd, color=(:blue, 0.7))
    lines!(ax4, trange, d2Pd, color=(:red, 0.7))
    lines!(ax4, trange, d2Wd, color=(:blue, 0.7))
    lines!(ax5, trange, d3Pd, color=(:red, 0.7))
    lines!(ax5, trange, d3Wd, color=(:blue, 0.7))
    #ylims!(ax4, 1e-16, 1e-7)
    Label(fig[1, 1:3, Top()], L"$ $(a)", halign=:left)
    Label(fig[2, 1, Top()], L"$ $(b)", halign=:left)
    Label(fig[2, 2, Top()], L"$ $(c)", halign=:left)
    Label(fig[2, 3, Top()], L"$ $(d)", halign=:left)
    resize_to_layout!(fig)
    display(fig)
    savefig && save(joinpath(@__DIR__, "figures", "heat_equation_underresolved.pdf"), fig)
end

const NAME_FUNC_DICT = OrderedDict(
    "semiclassical_jacobi_figures" => semiclassical_jacobi_figures,
    "arc_polynomial_figures" => arc_polynomial_figures,
    "hat_function_figures" => hat_function_figures,
    "rate_of_convergence_figures" => rate_of_convergence_figures,
    "screened_poisson" => screened_poisson,
    "heat_equation" => heat_equation,
    "linear_schrodinger" => linear_schrodinger,
    "convection_diffusion" => convection_diffusion,
    "save" => x -> nothing
)

function validate_args(args)
    valid_keys = keys(NAME_FUNC_DICT)
    invalid_args = setdiff(args, valid_keys)
    isempty(invalid_args) || error("Invalid arguments: $invalid_args. Valid arguments are: $(collect(valid_keys))")
end
info(name) = @info "[$(ct())]: Running $(replace(name, "_" => " "))"

function (@main)(args)
    @info "[$(ct())]: Validating arguments"
    validate_args(args)
    savefig = "save" in args
    runall = isempty(args) || args == ["save"]
    foreach(NAME_FUNC_DICT) do (name, fnc)
        name == "save" && return
        if runall || name in args
            info(name)
            fnc(savefig)
        end
    end
    @info "[$(ct())]: Done"
    return 0
end