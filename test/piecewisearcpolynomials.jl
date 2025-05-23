using ArcPolynomials,
    Test,
    BlockArrays,
    LinearAlgebra,
    IntervalSets,
    ContinuumArrays,
    ClassicalOrthogonalPolynomials,
    QuadGK,
    LazyArrays,
    InfiniteRandomArrays,
    InfiniteLinearAlgebra,
    MatrixFactorizations,
    OrdinaryDiffEq,
    SparseArrays

const AP = ArcPolynomials
const SAP = AP.SemiclassicalArcPolynomials
const PAP = AP.PiecewiseArcPolynomials

@testset "Construction" begin
    P = PiecewiseArcPolynomial{0}(10)
    h = cos(step(P.points) / 2)
    @test P.P == SemiclassicalJacobiArc(h, -1.0)
    @test P.P0 == SemiclassicalJacobiArc(h, 0.0)
    @test P.points == LinRange(-pi, pi, 11)
    P1 = PiecewiseArcPolynomial{1}(10)
    @test P1.P == SemiclassicalJacobiArc(h, -1.0)
    @test P1.P0 == SemiclassicalJacobiArc(h, 0.0)
    @test P1.points == LinRange(-pi, pi, 11)
    P2 = PiecewiseArcPolynomial{1}(P)
    @test P2 == P1 && P1 != P
    @test PAP.has_equal_spacing(P)
    @test PAP.has_equal_spacing(P1)
    @test PAP.has_equal_spacing(P2)
    @test PAP.get_P(P, 1) === P.P
    @test PAP.get_P(P1, 3) === P1.P
    @test PAP.get_P(P2, 2) === P2.P
    @test PAP.get_P0(P, 3) === P.P0
    @test PAP.get_P0(P1, 2) === P1.P0
    @test PAP.get_P0(P2, 1) === P2.P0

    P = PiecewiseArcPolynomial{0}([-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π])
    @test !PAP.has_equal_spacing(P)
    for i in 1:10
        h = cos((P.points[i+1] - P.points[i]) / 2)
        @test P.P[i] == SemiclassicalJacobiArc(h, -1.0)
        @test P.P0[i] == SemiclassicalJacobiArc(h, 0.0)
        @test P.P[i] === PAP.get_P(P, i)
        @test P.P0[i] === PAP.get_P0(P, i)
    end
    @test P.points == [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π]
    P1 = PiecewiseArcPolynomial{1}([-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π])
    @test !PAP.has_equal_spacing(P1)
    for i in 1:10
        h = cos((P1.points[i+1] - P1.points[i]) / 2)
        @test P1.P[i] == SemiclassicalJacobiArc(h, -1.0)
        @test P1.P0[i] == SemiclassicalJacobiArc(h, 0.0)
        @test P1.P[i] === PAP.get_P(P1, i)
        @test P1.P0[i] === PAP.get_P0(P1, i)
    end
    @test P1.points == [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π]

    @test_throws AssertionError PiecewiseArcPolynomial{0}(LinRange(-pi, pi, 2))
    @test_throws AssertionError PiecewiseArcPolynomial{1}(1)
    @test_throws AssertionError PiecewiseArcPolynomial{0}([-π, 2.0, 2.3, 3.5])
    @test_throws AssertionError PiecewiseArcPolynomial{1}([-π, 2.0, 1.0, π])
    @test_throws AssertionError PiecewiseArcPolynomial{1}([-3.5, 2.0, 1.0, π])
end

@testset "find_element" begin
    test_points = LinRange(-pi / 2, 3pi / 2, 250)
    for n in 2:10
        P = PiecewiseArcPolynomial{0}(n)
        for θ in test_points
            i = PAP.find_element(P, θ)
            @test P.points[i] ≤ AP.map_to_std_range(θ) ≤ P.points[i+1]
        end
    end

    P = PiecewiseArcPolynomial{0}([-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π])
    test_points = LinRange(-π, π, 500)
    for θ in test_points
        i = PAP.find_element(P, θ)
        @test P.points[i] ≤ AP.map_to_std_range(θ) ≤ P.points[i+1]
    end

    P = PiecewiseArcPolynomial{0}([-π, -2.5, π])
    test_points = LinRange(-π, π, 500)
    for θ in test_points
        i = PAP.find_element(P, θ)
        @test P.points[i] ≤ AP.map_to_std_range(θ) ≤ P.points[i+1]
    end
end

@testset "Evaluation" begin
    @testset "O = 0" begin
        for r in (10, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PiecewiseArcPolynomial{0}(r)
            ϕ = diff(P.points) ./ 2
            for θ in LinRange(-pi, pi, 50)
                local i
                for outer i in 1:(length(P.points)-1)
                    P.points[i] ≤ θ ≤ P.points[i+1] && break
                end
                blockvals_8 = P[θ, Block.(Base.OneTo(8))]
                blockvals_9 = P[θ, Block.(Base.OneTo(9))] # getindex branches on the parity of the maximum block
                result = zeros(9(length(P.points) - 1))
                for j in 1:9
                    result[i+(j-1)*(length(P.points)-1)] = PAP.get_P0(P, i)[affine(P.points[i] .. P.points[i+1], -ϕ[i] .. ϕ[i])[θ], j]
                end
                @test result[1:8(length(P.points)-1)] ≈ blockvals_8
                @test result ≈ blockvals_9

                ## Now test the specific getindex methods that allow for non-blockoneto use 
                @test blockvals_8 ≈ P[θ, Block(1):Block(8)]
                @test blockvals_9 ≈ P[θ, Block(1):Block(9)]
                for j in 1:9
                    result_block = result[(1+(j-1)*(length(P.points)-1)):(j*(length(P.points)-1))]
                    @test P[θ, Block(j)] ≈ result_block
                    for k in 1:(length(P.points)-1)
                        @test P[θ, BlockIndex(j, k)] ≈ result_block[k]
                    end
                end
                manual_result = [P[θ, k] for k in eachindex(result)]
                @test manual_result ≈ result
            end
        end
    end

    @testset "O = 1" begin
        for r in (9, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PiecewiseArcPolynomial{1}(r)
            rm1 = P.points[begin:end-1]
            nel = length(rm1)
            ϕ = diff(P.points) / 2
            @test P[rm1, Block(1)] ≈ I                                        # Test the hat functios 
            @test P[P.points, Block(2):Block(12)] ≈ zeros(nel + 1, 11nel) atol = 1e-12   # Test the bubble functions
            for θ in LinRange(-pi, pi, 250)
                i = PAP.find_element(P, θ)
                for m in 1:9
                    # Compare blocks
                    idx = Base.OneTo(m)
                    pkg_vals = P[θ, Block.(idx)]
                    spl = TrigonometricSpline(P.points)
                    vals = BlockedArray(zeros(m * nel), fill(nel, m))
                    vals[Block(1)] .= spl[θ, :]
                    a, b = P.points[i], P.points[i+1]
                    θ′ = affine(a .. b, -ϕ[i] .. ϕ[i])[θ]
                    for j in 2:m
                        vals[BlockIndex(j, i)] .= PAP.get_P(P, i)[θ′, j+1]
                    end
                    @test vals ≈ pkg_vals atol = 1e-12

                    # Non-block getindex 
                    @test pkg_vals ≈ P[θ, Block(1):Block(m)]
                    for j in 1:m
                        vals_block = vals[Block(j)]
                        @test P[θ, Block(j)] ≈ vals_block atol = 1e-12
                        for k in 1:nel
                            @test P[θ, BlockIndex(j, k)] ≈ vals_block[k] atol = 1e-12
                        end
                    end
                    manual_result = [P[θ, k] for k in eachindex(vals)]
                    @test manual_result ≈ vals atol = 1e-12
                end
            end
        end
    end
end

@testset "Gram Matrix" begin
    function inpgr(P, j, k)
        local integrand, val
        integrand = θ -> P[θ, j] * P[θ, k]
        val, _ = quadgk(integrand, P.points..., atol=1e-12, rtol=1e-12) # We pass P.points snice sometimes quadgk fails to detect the step in the function
        abs(val) < 1e-5 && (val = 0.0)
        return val
    end
    @testset "O = 0" begin
        for r in (2, 3, collect(LinRange(-pi, pi, 4)),  [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PiecewiseArcPolynomial{0}(r)
            gram = [inpgr(P, j, k) for j in 1:24, k in 1:24]
            X = (P'P)[1:24, 1:24]
            @test gram ≈ X
            @test P'P isa PAP.CyclicBBBArrowheadMatrix
        end
    end
    @testset "O = 1" begin
        for r in (2, 3, collect(LinRange(-pi, pi, 4)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PiecewiseArcPolynomial{1}(r)
            gram = [inpgr(P, j, k) for j in 1:24, k in 1:24]
            X = (P'P)[1:24, 1:24]
            @test gram ≈ X
            @test P'P isa PAP.CyclicBBBArrowheadMatrix
        end
    end
end

@testset "InterlacedVector" begin
    a = InfRandVector()
    b = InfRandVector()
    v = PAP.InterlacedVector(a, b)
    @test length(v) == ∞
    @test v[1:2:100] == a[1:50]
    @test v[2:2:100] == b[1:50]
end

@testset "Conversion" begin
    function conversion_matrix(P::PiecewiseArcPolynomial{1})
        P0 = PiecewiseArcPolynomial{0}(P)
        M = P0'P0
        U = zeros(24, 24)
        points = Tuple(unique!(sort(union(P.points, LinRange(-pi, pi, 20)))))
        for j in axes(U, 2)
            for i in axes(U, 1)
                integrand = θ -> P[θ, j] * P0[θ, i]
                U[i, j] = quadgk(integrand, points..., atol=1e-12, rtol=1e-12)[1]
                abs(U[i, j]) < 1e-5 && (U[i, j] = 0.0)
            end
        end
        U = inv(M[1:24, 1:24]) * U # Infinite loop when we do inv(M) ... At least inv(M)[1:32, 1:32] == inv(M[1:32, 1:32]) since M is diagonal
        return U
    end
    for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        P = PiecewiseArcPolynomial{1}(r)
        P0 = PiecewiseArcPolynomial{0}(P)
        U = conversion_matrix(P)
        C = P0 \ P
        @test eltype(C.C[1]) === Float64
        @test U ≈ C[1:24, 1:24] atol = 1e-4

        for θ in LinRange(-pi, pi, 51)
            pvals = P0[θ, 1:1000]
            for n in 1:50
                lhs = P[θ, n]
                rhs = dot(pvals, C[1:1000, n])
                @test lhs ≈ rhs atol = 1e-5
            end
        end
    end
end

@testset "Differentiation" begin
    function _get_derivative(P, j, θ)
        h = 1e-6
        return (P[θ+h, j] - P[θ-h, j]) / (2h)
    end
    function differentiation_matrix(P::PiecewiseArcPolynomial{1})
        P0 = PiecewiseArcPolynomial{0}(P)
        M = P0'P0
        U = zeros(24, 24)
        points = Tuple(unique!(sort(union(P.points, LinRange(-pi, pi, 10)))))
        for j in axes(U, 2)
            for i in axes(U, 1)
                integrand = θ -> P0[θ, i] * _get_derivative(P, j, θ)
                U[i, j] = quadgk(integrand, points..., atol=1e-9, rtol=1e-9)[1]
                abs(U[i, j]) < 1e-5 && (U[i, j] = 0.0)
            end
        end
        U = inv(M[1:24, 1:24]) * U # Infinite loop when we do inv(M) ... At least inv(M)[1:32, 1:32] == inv(M[1:32, 1:32]) since M is diagonal
        return U
    end
    for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        P = PiecewiseArcPolynomial{1}(r)
        D = diff(P).args[2]
        U = differentiation_matrix(P)
        @test U ≈ D[1:24, 1:24] atol = 1e-6
    end
end

@testset "Weak Laplacian" begin
    for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π], [-π,-π/4,π/4,π])
        P = PiecewiseArcPolynomial{1}(r)
        Δ = -diff(P)'diff(P)
        Δ2 = weaklaplacian(P)
        @test Δ2 isa PAP.CyclicBBBArrowheadMatrix
        @test Matrix(principal_submatrix(Δ, Block(24))) ≈ Matrix(principal_submatrix(Δ2, Block(24))) atol = 1e-6 rtol=1e-6
    end
end

@testset "Transform" begin
    f1 = θ -> cos(θ)
    θ = LinRange(-pi, pi, 250)
    for r in (6, 24, 2, collect(LinRange(-pi, pi, 7)), collect(LinRange(-pi, pi, 25)), collect(LinRange(-pi, pi, 3)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        P = PiecewiseArcPolynomial{0}(r)
        _r = r isa Int ? r : length(r) - 1
        g = expand(P, f1)
        c = g.args[2]
        @test axes(c) === (BlockedOneTo(_r:_r:∞),)
        @test c[Block(2)] == c.args[1][:, 2]
        @test BlockArrays.blockcolsupport(c) == BlockRange(Base.OneTo(size(c.args[1].parent.args[2], 1)))
        fvals = f1.(θ)
        gvals = g[θ]
        @test fvals ≈ gvals

        P = PiecewiseArcPolynomial{1}(r)
        fs = let P = P
            (
                θ -> cos(θ) * sin(θ),
                θ -> P[θ, 1] + P[θ, 3] + P[θ, 15],
                θ -> P[θ, 2] + cos(θ) + P[θ, 3] * exp(sin(θ))
            )
        end
        foreach(fs) do f2
            g = expand(P, f2)
            c = g.args[2]
            @test axes(c) === (BlockedOneTo(_r:_r:∞),)
            @test c[Block(1)] == c.args[2][1:_r]
            if _r != 2
                @test c[BlockIndex(1, 3)] == c.args[2][3]
                @test c[3] == c.args[2][3]
            end
            fvals = f2.(θ)
            gvals = g[θ]
            @test fvals ≈ gvals atol = 1e-4
        end
    end
end

function random_trigpoly(N)
    M = 2N + 1
    coeffs = 5randn(M)
    F = Fourier()
    coeff = InfiniteLinearAlgebra.pad(coeffs, axes(F, 2))
    return let F = F * coeff
        θ -> F[θ]
    end
end
@testset "Fourier expansions" begin
    for M in (2:15..., [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        for O in (0, 1)
            P = PiecewiseArcPolynomial{O}(M)
            for N in Iterators.flatten((0:20, 51:48:200, repeat([0, 1, 2, 3], 5), repeat([0], 10)))
                f = random_trigpoly(N)
                tf = transform(P, f)
                dat = LazyArrays.paddeddata(tf)
                nel = length(P.points) - 1
                if O == 0
                    flag = length(dat) ≤ nel * (2N + 1)
                else
                    flag = length(dat) ≤ nel * max(2N, 1)
                end
                if flag
                    @test flag
                else
                    if O == 0
                        @test dat[(nel*(2N+1)+1):end] ≈ zeros(length(dat) - nel * (2N + 1)) atol = 1e-10
                    else
                        if N == 0
                            @test dat[(nel+1):end] ≈ zeros(length(dat) - nel) atol = 1e-10
                        else
                            @test dat[(nel*max(2N, 1)+1):end] ≈ zeros(length(dat) - nel * max(2N, 1)) atol = 1e-10
                        end
                    end
                end
            end
        end
    end
end

@testset "Multiplication" begin
    for f in (
        θ -> ((y, x) = sincos(θ); x),
        θ -> ((y, x) = sincos(θ); y),
        θ -> ((y, x) = sincos(θ); y^3 + x^13 + x + y - x * y + 100)
    )
        for O in (2, 3, [-π,0.2,0.5,π], 5, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PiecewiseArcPolynomial{0}(O)
            a = expand(P, f)
            aP = a .* P
            @test aP.args[2] isa ArcPolynomials.CyclicBBBArrowheadMatrices.InterlacedMatrix
            for θ in LinRange(-π, π, 15)
                @test aP[θ, Block(1):Block(10)] ≈ f(θ) * P[θ, Block(1):Block(10)] atol = 1e-8 rtol = 1e-8
            end

            Q = PiecewiseArcPolynomial{1}(O)
            aP = a .* Q
            for θ in LinRange(-π, π, 15)
                @test aP[θ, Block(1):Block(10)] ≈ f(θ) * Q[θ, Block(1):Block(10)] atol = 1e-8 rtol = 1e-8
            end
        end
    end
end