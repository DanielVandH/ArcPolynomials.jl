using ArcPolynomials,
    Test,
    PiecewiseOrthogonalPolynomials,
    DomainSets,
    QuasiArrays,
    BlockArrays,
    FillArrays,
    Infinities,
    ContinuumArrays,
    ClassicalOrthogonalPolynomials,
    LinearAlgebra,
    QuadGK

@testset "Definition" begin
    points = LinRange(-π, π, 10)
    P = PeriodicContinuousPolynomial{0}(points)
    @test P.points === points
    @test PeriodicContinuousPolynomial{0}(9) == P
    @test axes(P) == (Inclusion(ℝ), blockedrange(Fill(length(P.points) - 1, ∞)))
end

@testset "Evaluation" begin
    @testset "O = 0" begin
        for r in (10, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PeriodicContinuousPolynomial{0}(r)
            for θ in LinRange(-pi, pi, 50)
                local i
                for outer i in 1:(length(P.points)-1)
                    P.points[i] ≤ θ ≤ P.points[i+1] && break
                end
                blockvals_8 = P[θ, Block.(Base.OneTo(8))]
                blockvals_9 = P[θ, Block.(Base.OneTo(9))] # getindex branches on the parity of the maximum block
                result = zeros(9(length(P.points) - 1))
                for j in 1:9
                    result[i+(j-1)*(length(P.points)-1)] = Legendre()[affine(P.points[i] .. P.points[i+1], -1 .. 1)[θ], j]
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
            P = PeriodicContinuousPolynomial{1}(r)
            Q = ContinuousPolynomial{1}(P.points)
            rm1 = P.points[begin:end-1]
            nel = length(rm1)
            ϕ = diff(P.points) / 2
            @test P[rm1, Block(1)] ≈ I   
            @test P[rm1, collect(1:nel)] ≈ I                                     # Test the hat functios 
            @test P[P.points, Block(2):Block(12)] ≈ zeros(nel + 1, 11nel) atol = 1e-12   # Test the bubble functions
            @test P[P.points, (nel+1):(nel+1000)] ≈ zeros(nel + 1, 1000) atol = 1e-12
            for θ in LinRange(-pi, pi, 250)
                i = ArcPolynomials.PeriodicContinuousPolynomials.find_element(P, θ)
                for m in 1:9
                    # Compare blocks
                    idx = Base.OneTo(m)
                    pkg_vals = P[θ, Block.(idx)]
                    spl = PeriodicLinearSpline(P.points)
                    vals = BlockedArray(zeros(m * nel), fill(nel, m))
                    vals[Block(1)] .= spl[θ, :]
                    a, b = P.points[i], P.points[i+1]
                    θ′ = affine(a .. b, -1 .. 1)[θ]
                    for j in 2:m
                        vals[BlockIndex(j, i)] .= Q[θ, Block(j)][i]
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

@testset "Evaluation (Old)" begin
    p1 = LinRange(-π, π, 10)
    p2 = LinRange(-π, π, 23)
    p3 = LinRange(-π, π, 3)
    p4 = collect(LinRange(-π, π, 7))
    for points in (p1, p2, p3, p4, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        P = PeriodicContinuousPolynomial{0}(points)
        Q = ContinuousPolynomial{0}(points)
        θ = LinRange(-π, π, 500)
        @test P[θ, 1] ≈ Q[θ, 1]
        @test P[θ, Block(2)] ≈ Q[θ, Block(2)]
        @test P[θ, Block(1):Block(6)] ≈ Q[θ, Block(1):Block(6)]
        @test P[θ, Block(1)] ≈ P[θ, collect(1:length(P.points)-1)]

        P = PeriodicContinuousPolynomial{1}(points)
        Q = ContinuousPolynomial{1}(points)
        θ = LinRange(-π, π, 500)
        @test P[θ, Block(1)] ≈ PeriodicLinearSpline(P.points)[θ, :]
        @test P[θ, Block(2)] ≈ Q[θ, Block(2)]
        @test P[θ, Block(2):Block(6)] ≈ Q[θ, Block(2):Block(6)]
        @test P[θ, Block(1)] ≈ P[θ, collect(1:length(P.points)-1)]
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
        for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PeriodicContinuousPolynomial{0}(r)
            gram = [inpgr(P, j, k) for j in 1:52, k in 1:52]
            X = (P'P)[1:52, 1:52]
            @test gram ≈ X
        end
    end
    @testset "O = 1" begin
        for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-pi, -2.0, -0.53, 0.85, 1.52, 1.69, 1.97, pi], [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PeriodicContinuousPolynomial{1}(r)
            gram = [inpgr(P, j, k) for j in 1:52, k in 1:52]
            X = (P'P)[1:52, 1:52]
            @test gram ≈ X
            @test P'P isa ArcPolynomials.CyclicBBBArrowheadMatrices.CyclicBBBArrowheadMatrix
        end
    end
end

@testset "Conversion" begin
    function conversion_matrix(P::PeriodicContinuousPolynomial{1})
        P0 = PeriodicContinuousPolynomial{0}(P)
        M = P0'P0
        U = zeros(48, 48)
        points = Tuple(unique!(sort(union(P.points, LinRange(-pi, pi, 20)))))
        for j in axes(U, 2)
            for i in axes(U, 1)
                integrand = θ -> P[θ, j] * P0[θ, i]
                U[i, j] = quadgk(integrand, points..., atol=1e-12, rtol=1e-12)[1]
                abs(U[i, j]) < 1e-5 && (U[i, j] = 0.0)
            end
        end
        U = inv(M[1:48, 1:48]) * U # Infinite loop when we do inv(M) ... At least inv(M)[1:32, 1:32] == inv(M[1:32, 1:32]) since M is diagonal
        return U
    end
    for r in (2, 3, 5, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-pi, -2.0, -0.53, 0.85, 1.52, 1.69, 1.97, pi], [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
        P = PeriodicContinuousPolynomial{1}(r)
        P0 = PeriodicContinuousPolynomial{0}(P)
        U = conversion_matrix(P)
        C = P0 \ P
        @test U ≈ C[1:48, 1:48] atol = 1e-4

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
    function differentiation_matrix(P::PeriodicContinuousPolynomial{1})
        P0 = PeriodicContinuousPolynomial{0}(P)
        M = P0'P0
        U = zeros(48, 48)
        points = Tuple(unique!(sort(union(P.points, LinRange(-pi, pi, 10)))))
        for j in axes(U, 2)
            for i in axes(U, 1)
                integrand = θ -> P0[θ, i] * _get_derivative(P, j, θ)
                U[i, j] = quadgk(integrand, points..., atol=1e-9, rtol=1e-9)[1]
                abs(U[i, j]) < 1e-5 && (U[i, j] = 0.0)
            end
        end
        U = inv(M[1:48, 1:48]) * U # Infinite loop when we do inv(M) ... At least inv(M)[1:32, 1:32] == inv(M[1:32, 1:32]) since M is diagonal
        return U
    end
    for r in (2, 3, 5, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-pi, -2.0, -0.53, 0.85, 1.52, 1.69, 1.97, pi], [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π], [-π, 2.0, π])
        P = PeriodicContinuousPolynomial{1}(r)
        D = diff(P).args[2]
        U = differentiation_matrix(P)
        @test U ≈ D[1:48, 1:48] atol = 1e-6
    end
end

@testset "Weak Laplacian" begin
    for r in (2, 3, 6, collect(LinRange(-pi, pi, 4)), collect(LinRange(-pi, pi, 7)), [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π], [-π,-π/4,π/4,π])
        P = PeriodicContinuousPolynomial{1}(r)
        Δ = -diff(P)'diff(P)
        Δ2 = weaklaplacian(P)
        @test Δ2 isa ArcPolynomials.CyclicBBBArrowheadMatrices.CyclicBBBArrowheadMatrix
        @test Matrix(principal_submatrix(Δ, Block(24))) ≈ Matrix(principal_submatrix(Δ2, Block(24))) atol = 1e-6 rtol=1e-6
    end
end

@testset "Transform" begin
    for M in (2, 3, 4, 5, 6, 7, 8, 9, 10, [-pi, -2.0, -0.53, 0.85, 1.52, 1.69, 1.97, pi], [-π, 2.0, π])
        P0 = PeriodicContinuousPolynomial{0}(M)
        f = (cos, exp, sin)
        t = LinRange(-pi, pi, 500)
        for fs in f
            g = expand(P0, fs)
            lhs = fs.(t)
            rhs = g[t]
            @test lhs ≈ rhs rtol = 1e-6 atol = 1e-6
        end

        P = PeriodicContinuousPolynomial{1}(M)
        f = (cos, sin, x -> cos(2x) - sin(5x) + exp(-cos(x)))
        t = LinRange(-pi, pi, 500)
        for fs in f
            g = expand(P, fs)
            lhs = fs.(t)
            rhs = g[t]
            @test lhs ≈ rhs rtol = 1e-6 atol = 1e-6
        end
    end
end

@testset "Multiplication" begin
    for f in (
        θ -> ((y, x) = sincos(θ); x),
        θ -> ((y, x) = sincos(θ); y^5),
        θ -> ((y, x) = sincos(θ); y^3 + x^13 + x + y - x * y + 100)
    )
        for O in (2, 3, [-π,0.2,0.3,π], 5, [-π, -2.5, -1.8, -1.2, -0.6, 0.0, 0.3, 1.3, 1.5, 2.3, π], [-π, -1.0, π])
            P = PeriodicContinuousPolynomial{0}(O)
            a = expand(P, f)
            aP = a .* P
            @test aP.args[2] isa ArcPolynomials.CyclicBBBArrowheadMatrices.InterlacedMatrix
            for θ in LinRange(-π, π, 15)
                @test aP[θ, Block(1):Block(10)] ≈ f(θ) * P[θ, Block(1):Block(10)] atol = 1e-8 rtol = 1e-8
            end

            Q = PeriodicContinuousPolynomial{1}(O)
            aP = a .* Q
            for θ in LinRange(-π, π, 16)
                @test aP[θ, Block(1):Block(10)] ≈ f(θ) * Q[θ, Block(1):Block(10)] atol = 1e-8 rtol = 1e-8
            end
        end
    end
end