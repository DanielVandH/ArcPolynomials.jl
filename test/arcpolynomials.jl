using ArcPolynomials,
    Test,
    ForwardDiff,
    StaticArrays,
    ContinuumArrays,
    FastGaussQuadrature,
    ClassicalOrthogonalPolynomials,
    SpecialFunctions,
    LinearAlgebra,
    SemiclassicalOrthogonalPolynomials,
    QuasiArrays,
    FillArrays,
    InfiniteLinearAlgebra,
    LazyArrays,
    BandedMatrices,
    HypergeometricFunctions,
    ArrayLayouts,
    SparseArrays,
    BlockArrays,
    QuadGK,
    RecurrenceRelationshipArrays

import ClassicalOrthogonalPolynomials: orthogonalityweight, golubwelsch, gaussradau, gausslobatto
import FastGaussQuadrature: gausslegendre
import Infinities: ℵ₁, ℵ₀
const AP = ArcPolynomials
const SAP = AP.SemiclassicalArcPolynomials

_Yn1(R::SemiclassicalJacobiArc{T}, n, x, y) where {T} = R.P[SAP._mapx(R, x), n+1]::T
_Yn2(R::SemiclassicalJacobiArc{T}, n, x, y) where {T} = iszero(n) ? zero(T) : y * R.Q[SAP._mapx(R, x), n]::T

function _differentiable_arc(n, h, b, θ)
    T = promote_type(typeof(h), typeof(b), typeof(θ))
    R = SemiclassicalJacobiArc{T}(h, b)
    return Base.unsafe_getindex(R, θ, n)
end
@test_broken _differentiable_arc(3, 0.3, 0.0, ForwardDiff.Dual(0.2, 1.0)) # ForwardDiff doesn't work

@testset "Arc" begin
    @test AP.Arc() == AP.Arc{Float64}(0.0)
    @test AP.Arc(1 / 2) == AP.Arc{Float64}(1 / 2)
    @test AP.Arc(0.2).h == 0.2
    @test 0.2 ∈ AP.Arc(0.0)
    @test 0.5 ∉ AP.Arc(0.9)
    @test 0.5 ∈ AP.Arc(0.8)
    @test all(∈(AP.Arc(0.4)), ContinuumArrays.checkpoints(AP.Arc(0.4)))
    @test QuasiArrays.cardinality(AP.Arc(0.2)) == ℵ₁
end

@testset "ArcInclusion" begin
    a = AP.ArcInclusion(0.3)
    @test checkindex(Bool, a, 0.5)
    @test !checkindex(Bool, a, 1.5)
end

@testset "ArcWeight" begin
    a = SAP.ArcWeight(0.3, 0.5)
    @test a.h == 0.3 && a.b == 0.5
    @test axes(a) == (AP.ArcInclusion(0.3),)
    @test a[0.3] ≈ (cos(0.3) - a.h)^a.b
end

@testset "Definition" begin
    @testset "Evaluation" begin
        for (h, b) in ((0.3, -1.0), (-1.0, -1 / 2))
            R = SemiclassicalJacobiArc(h, b)
            @test size(R) == (ℵ₁, ℵ₀)
            if h != -1.0
                P = SemiclassicalJacobi(2 / (1 - h), -1 / 2, b, -1 / 2)
                Q = SemiclassicalJacobi(2 / (1 - h), 1 / 2, b, 1 / 2)
            else
                P = SemiclassicalJacobi(1.0, -1 / 2, b - 1 / 2, 0.0)
                Q = SemiclassicalJacobi(1.0, 1 / 2, b + 1 / 2, 0.0)
            end
            for n in 1:100
                d = n ÷ 2
                idx = iszero(d) ? (1,) : (2d, 2d + 1)
                for θ in LinRange(-acos(R.h), acos(R.h), 25)
                    x, y = cos(θ), sin(θ)
                    xt = (x - 1) / (R.h - 1)
                    xt = xt < 0 ? 0.0 : xt > 1 ? 1.0 : xt
                    if d == 0
                        @test R[θ, idx[1]] ≈ P[xt, d+1]
                    else
                        @test R[θ, idx[2]] ≈ P[xt, d+1]
                        @test R[θ, idx[1]] ≈ y * Q[xt, d]
                    end
                end
            end
            for θ in LinRange(-acos(R.h) + eps(), acos(R.h) - eps(), 25)
                x, y = cos(θ), sin(θ)
                xt = (x - 1) / (R.h - 1)
                xt = xt < 0 ? 0.0 : xt > 1 ? 1.0 : xt
                @test R[θ, Block(1)] == [P[xt, 1]]
                @test R[θ, Block(2)] == [y * Q[xt, 1], P[xt, 2]]
                @test R[θ, Block(10)] == [y * Q[xt, 9], P[xt, 10]]
                @test R[θ, blockedrange([1, 2, 2, 2, 2, 2])] == vcat(R[θ, Block(1)], R[θ, Block(2)], R[θ, Block(3)], R[θ, Block(4)], R[θ, Block(5)], R[θ, Block(6)])
                @test R[θ, BlockIndex(1, 1)] == R[θ, Block(1)][1]
                @test R[θ, BlockIndex(2, 1)] == R[θ, Block(2)][1]
                @test R[θ, BlockIndex(2, 2)] == R[θ, Block(2)][2]
                @test R[θ, BlockIndex(10, 1)] == R[θ, Block(10)][1]
                @test R[θ, BlockIndex(10, 2)] == R[θ, Block(10)][2]
            end
            @inferred R[0.2, 1]
            @inferred R[0.2, Block(1)]
            @inferred R[0.2, blockedrange([1, 2])]
            @inferred R[0.2, BlockIndex(1, 1)]
        end
    end

    function inp(f, g, w, a, b, xgw, wgw)
        local integrand, val
        integrand = θ -> f(cos(θ), sin(θ)) * g(cos(θ), sin(θ)) * w(cos(θ))
        val = (b - a) / 2 * dot(wgw, integrand.((b - a) / 2 .* xgw .+ (a + b) / 2))
        return val
    end
    function get_xy(R, t)
        local x, y
        t = t .* (R.h .- 1) .+ 1
        x = t
        y = @. sqrt(1 - t^2)
        return x, y
    end
    function dinp(f, g, w, x, y)
        local inp
        inp = 0.0
        for (wⱼ, xⱼ, yⱼ) in zip(w, x, y)
            inp += wⱼ * (f(xⱼ, yⱼ) * g(xⱼ, yⱼ) + f(xⱼ, -yⱼ) * g(xⱼ, -yⱼ))
        end
        return inp
    end

    @testset "Orthogonality" begin
        @testset "Continuous inner product" begin
            # Test that our orthogonal polynmials satisfy the expected orthogonality relation.
            xgw, wgw = gausslegendre(250)
            for (h, b) in ((0.3, 1.0), (-1.0, 0.0))
                R = SemiclassicalJacobiArc(h, b)
                w = x -> orthogonalityweight(R)[acos(x)]
                θ = acos(R.h)
                a, b = -θ, θ
                for m in 0:5
                    for n in 0:5
                        f = (x, y) -> _Yn1(R, m, x, y)
                        g = (x, y) -> _Yn2(R, n, x, y)
                        val = inp(f, g, w, a, b, xgw, wgw)
                        @test val ≈ 0.0 atol = 1e-4
                        if n ≠ m
                            val = inp(f, (x, y) -> _Yn1(R, n, x, y), w, a, b, xgw, wgw)
                            @test val ≈ 0.0 atol = 1e-4
                            val = inp((x, y) -> _Yn2(R, m, x, y), g, w, a, b, xgw, wgw)
                            @test val ≈ 0.0 atol = 1e-4
                        end
                    end
                end
            end
        end

        @testset "Discrete inner product" begin
            for (h, b) in ((0.3, 1 / 2), (-1.0, 0.0))
                R = SemiclassicalJacobiArc(h, b)
                for nn in 4:4:16
                    tgw, wgw = golubwelsch(R.P, nn)
                    tgr1, wgr1 = gaussradau(R.P, nn, 0.0)
                    tgr2, wgr2 = gaussradau(R.P, nn, 1.0)
                    tgl, wgl = gausslobatto(R.P, nn)
                    xgw, ygw = get_xy(R, tgw)
                    xgr1, ygr1 = get_xy(R, tgr1)
                    xgr2, ygr2 = get_xy(R, tgr2)
                    xgr2[end] = h
                    ygr2[end] = sqrt(1 - h^2)
                    xgl, ygl = get_xy(R, tgl)
                    xgl[end] = h
                    ygl[end] = sqrt(1 - h^2)
                    for (x, y, w) in ((xgw, ygw, wgw), (xgr1, ygr1, wgr1), (xgr2, ygr2, wgr2), (xgl, ygl, wgl))
                        for m in 0:nn
                            for n in 0:nn
                                f = (x, y) -> _Yn1(R, m, x, y)
                                g = (x, y) -> _Yn2(R, n, x, y)
                                val = dinp(f, g, w, x, y)
                                @test val ≈ 0.0 atol = 1e-4
                                val1 = dinp(f, (x, y) -> _Yn1(R, n, x, y), w, x, y)
                                val2 = dinp((x, y) -> _Yn2(R, m, x, y), g, w, x, y)
                                if n ≠ m
                                    @test val1 ≈ 0.0 atol = 1e-4
                                    @test val2 ≈ 0.0 atol = 1e-4
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Coefficients" begin
        R = SemiclassicalJacobiArc(0.3, 1.0)
        f = (x, y) -> exp(x) * cos(-x) + x^3 - 5x^5
        fc = zeros(16)
        xgw, wgw = gausslegendre(250)
        w = x -> orthogonalityweight(R)[acos(x)]
        θ = acos(R.h)
        a, b = -θ, θ
        for n in 1:16
            g = (x, y) -> R[atan(y, x), n]
            fc[n] = inp(f, g, w, a, b, xgw, wgw) / inp(g, g, w, a, b, xgw, wgw)
        end
        _fc = Vcat(fc, Zeros(∞))
        Rf = R * _fc
        for θ in LinRange(a, b, 250)
            @test Rf[θ] ≈ f(cos(θ), sin(θ)) rtol = 1e-6
        end
    end
end

@testset "Expansions" begin
    function test_expand(R, f)
        local f1, f2, g, θ, n
        g = expand(R, f)
        θ = acos(R.h + eps(R.h))
        n = 250
        f1, f2 = zeros(n), zeros(n)
        for (i, ϕ) in pairs(LinRange(-θ, θ, n))
            f1[i] = f(ϕ)
            f2[i] = g[ϕ]
        end
        return f1, f2
    end
    R = SemiclassicalJacobiArc(0.3, 0.0)
    f1 = θ -> 1 - cos(θ) + sin(θ)
    f2 = θ -> cos(cos(θ))
    f3 = θ -> cos(cos(θ) + cos(cos(θ) * sin(θ) - sin(sin(θ)))) + cos(θ)^64
    f4 = θ -> 1.0
    f5 = let R = R
        θ -> R[θ, 1] - 3R[θ, 3] + 10R[θ, 5]
    end
    @test all(f -> isapprox(test_expand(R, f)...), (f1, f2, f3, f4, f5))
    @test transform(R, f5)[1:8] ≈ [1.0, 0.0, -3.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    @test transform(R, f4)[1:5] ≈ [1.0, 0.0, 0.0, 0.0, 0.0]
    @test transform(R, f3)[1:20] ≈ R[:, 1:20] \ f3.(axes(R, 1))
    R = SemiclassicalJacobiArc(0.5, -1.0)
    f1 = θ -> 5.0 + (cos(θ) - 0.5) * sin(θ)
    f2 = θ -> 7.0 + exp(cos(θ) * sin(θ)) + 1 - cos(θ)
    f3 = θ -> 5.0 + sin(θ) * (5.0 + 1 - cos(θ)) + cos(θ)
    f4 = let R = R
        θ -> R[θ, 1] - 3R[θ, 3] + 10R[θ, 2] - 0.5R[θ, 7]
    end
    @test all(f -> isapprox(test_expand(R, f)...), (f1, f2, f3, f4))
    @test transform(R, f1)[1:8] ≈ [5.0, 0.0, 0.0, R.h, 0.0, 0.0, 0.0, 0.0]
    @test transform(R, f3)[1:8] ≈ [5 + R.h, 5 + R.h, R.h, -R.h, 0.0, 0.0, 0.0, 0.0]
    @test transform(R, f4)[1:8] ≈ [1.0, 10.0, -3.0, 0.0, 0.0, 0.0, -0.5, 0.0]
    @test transform(R, f2)[1:20] ≈ R[:, 1:20] \ f2.(axes(R, 1))
end

@testset "isassigned" begin
    h, b = 0.3, 0.0
    R = SemiclassicalJacobiArc(h, b)
    @test Base.isassigned(R, 0.2, 1)
    @test Base.isassigned(R, 0.5, 23)
    @test !Base.isassigned(R, 0.5, -2)
    @test !Base.isassigned(R, 1.3, 3)
    @test !Base.isassigned(R, -1.3, 3)
end

@testset "Infinite slice" begin
    h, b = 0.2, 0.3
    R = SemiclassicalJacobiArc(h, b)
    KK = R[0.324, :]
    @test KK[1] == R[0.324, 1]
    @test KK[5] == R[0.324, 5]
    @test KK isa SubArray
    @test axes(KK) == (axes(R, 2),)
end

@testset "Jacobi Matrix" begin
    @testset "JacobiX" begin
        f = θ -> ((y, x) = sincos(θ); x^2 + 5x * y + 17x^3 + exp(-(1 - x) * (1 - y)))
        xf = let f = f
            θ -> cos(θ) * f(θ)
        end
        R = SemiclassicalJacobiArc(0.3, 1.0)
        c1 = transform(R, f)
        c2 = transform(R, xf)
        X = jacobimatrix(Val(1), R)
        @test X * c1 ≈ c2
        R = SemiclassicalJacobiArc(0.3, -1.0)
        c1 = transform(R, f)
        c2 = transform(R, xf)
        X = jacobimatrix(Val(1), R)
        @test X * c1 ≈ c2

        function test_jacobix(R::SemiclassicalJacobiArc)
            local X, θ, n, lhs, rhs
            X = jacobimatrix(Val(1), R)
            θ = acos(R.h + eps(R.h))
            n = 250
            lhs, rhs = zeros(n), zeros(n)
            for (i, ϕ) in pairs(LinRange(-θ, θ, n))
                y, x = sincos(ϕ)
                range = max(1, i - 2):(i+2)
                lhs[i] = x * R[ϕ, i]
                rhs[i] = dot(view(X, range, i), R[ϕ, range])
            end
            return lhs, rhs
        end
        R = SemiclassicalJacobiArc(0.3, 1.0)
        lhs, rhs = test_jacobix(R)
        @test lhs ≈ rhs
        R = SemiclassicalJacobiArc(0.1, -1.0)
        lhs, rhs = test_jacobix(R)
        @test lhs ≈ rhs
    end
    @testset "JacobiY" begin
        f = θ -> ((y, x) = sincos(θ); x^2 + 5x * y + 17x^3 + exp(-(1 - x) * (1 - y)))
        yf = let f = f
            θ -> sin(θ) * f(θ)
        end
        R = SemiclassicalJacobiArc(0.3, 1.0)
        c1 = transform(R, f)
        c2 = transform(R, yf)
        Y = jacobimatrix(Val(2), R)
        @test Y * c1 ≈ c2
        R = SemiclassicalJacobiArc(0.3, -1.0)
        c1 = transform(R, f)
        c2 = transform(R, yf)
        Y = jacobimatrix(Val(2), R)
        @test Y * c1 ≈ c2

        function test_jacobiy(R::SemiclassicalJacobiArc)
            local X, θ, n, lhs, rhs
            X = jacobimatrix(Val(2), R)
            θ = acos(R.h + eps(R.h))
            n = 250
            lhs, rhs = zeros(n), zeros(n)
            for (i, ϕ) in pairs(LinRange(-θ, θ, n))
                y, x = sincos(ϕ)
                range = max(1, i - 3):(i+3)
                lhs[i] = y * R[ϕ, i]
                rhs[i] = dot(view(X, range, i), R[ϕ, range])
            end
            return lhs, rhs
        end
        R = SemiclassicalJacobiArc(0.3, 1.0)
        lhs, rhs = test_jacobiy(R)
        @test lhs ≈ rhs
        R = SemiclassicalJacobiArc(0.1, -1.0)
        lhs, rhs = test_jacobiy(R)
        @test lhs ≈ rhs
    end
end

@testset "Differentiation" begin
    @testset "DiffP" begin
        for (h, b) in ((0.1, -1.0), (0.3, 1.0), (0.0, -1.0))
            R = SemiclassicalJacobiArc(h, b)
            Dp = SAP.semiclassicaljacobiarc_diffp(R)
            R1 = SemiclassicalJacobiArc(R.b + 1, R)
            ϕ = acos(R.h + 1e-3)
            e = 1e-5
            for n in 0:20
                for θ in LinRange(-ϕ, ϕ, 1000)
                    s, c = sincos(θ)
                    su, cu = sincos(θ + e)
                    sl, cl = sincos(θ - e)
                    pu = _Yn1(R, n, cu, su)
                    pl = _Yn1(R, n, cl, sl)
                    lhs = (pu - pl) / (2e)
                    rhs = 0.0
                    for k in colsupport(Dp, n + 1)
                        q = _Yn2(R1, k, c, s)
                        d = Dp[k, n+1]
                        rhs += q * d
                    end
                    @test lhs ≈ rhs rtol = 1e-2
                end
            end
        end
    end
    @testset "DiffQ" begin
        for (h, b) in ((0.1, -1.0), (0.2, 1.0), (0.0, -1.0))
            R = SemiclassicalJacobiArc(h, b)
            Dq = SAP.semiclassicaljacobiarc_diffq(R)
            R1 = SemiclassicalJacobiArc(R.b + 1, R)
            ϕ = acos(R.h + 1e-3)
            e = 1e-4
            for n in 0:20
                for θ in LinRange(-ϕ, ϕ, 100)
                    s, c = sincos(θ)
                    su, cu = sincos(θ + e)
                    sl, cl = sincos(θ - e)
                    qu = _Yn2(R, n + 1, cu, su)
                    ql = _Yn2(R, n + 1, cl, sl)
                    lhs = (qu - ql) / (2e)
                    rhs = 0.0
                    for k in 1:(n+2)
                        p = _Yn1(R1, k - 1, c, s)
                        d = Dq[k, n+1]
                        rhs += p * d
                    end
                    @test lhs ≈ rhs rtol = 1e-3
                end
            end
        end
    end

    @testset "Differentiation Matrix" begin
        function test_diff(R, f, df)
            local R1, D, c, dfc, dc, N
            R1, D = LazyArrays.arguments(diff(R))
            c = transform(R, f)
            dfc = transform(R1, df)
            dc = D * c
            N = colsupport(c) ∪ colsupport(dfc)
            return dc[N], dfc[N]
        end

        for (h, b) in ((0.1, -1.0), (0.2, 1.0), (0.0, -1.0))
            R = SemiclassicalJacobiArc(h, b)
            f = θ -> cos(θ)
            df = θ -> -sin(θ)
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> sin(θ)
            df = θ -> cos(θ)
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> cos(θ) + sin(θ)
            df = θ -> -sin(θ) + cos(θ)
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> cos(θ)^2
            df = θ -> -2cos(θ)sin(θ)
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> cos(θ)^5
            df = θ -> -5cos(θ)^4 * sin(θ)
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> sin(θ)^5
            df = θ -> 5cos(θ) * sin(θ)^4
            @test isapprox(test_diff(R, f, df)...)
            f = θ -> exp(-cos(θ) * sin(θ))
            df = θ -> -exp(-cos(θ) * sin(θ)) * (cos(θ)^2 - sin(θ)^2)
            @test isapprox(test_diff(R, f, df)...)
        end
    end
end

@testset "Grammatrix" begin
    function inp(R, j, k)
        local integrand, t, a, b, val
        integrand = θ -> R[θ, j] * R[θ, k]
        t = acos(R.h)
        a, b = -t, t
        val, _ = quadgk(integrand, a, b, atol=1e-12, rtol=1e-12)
        abs(val) < 1e-5 && (val = 0.0)
        return val
    end
    for h in (-0.3, 0.0, 0.3)
        for b in (-1.0, 0.0)
            R = SemiclassicalJacobiArc(h, b)
            X = R'R
            if b == 0.0
                @test isdiag(X)
            end
            XX = X[1:12, 1:12]
            YY = inp.((R,), 1:12, (1:12)')
            diag = zeros(12)
            @test XX ≈ YY
            if b == 0
                t = 2 / (1 - h)
                diag[1:2:end] .= 4acsc(sqrt(t))
                diag[2:2:end] .= (1 - h)^2 / 2 * (t^2 * acsc(sqrt(t)) + (2 - t) * sqrt(t - 1))
                ZZ = Diagonal(diag)
                @test XX ≈ ZZ
            end
        end
    end
end

function mu_coefficients(n, h)
    P = SemiclassicalJacobiArc(h, 0.0)
    X = jacobimatrix(P.P)
    τ = P.P.t
    α, β = SemiclassicalOrthogonalPolynomials._linear_coefficients(P.P.t, P.P.a, P.P.b, P.P.c)
    μ = zeros(n + 1, n + 1)
    μ[1, 1] = 4acsc(sqrt(τ))
    μ[1, 2] = μ[1, 1] * (1 + (h - 1) * α)
    μ[2, 2] = μ[1, 1] * (h - 1) / β
    for nn in 2:n
        for jj in 0:nn
            m = nn + 1
            j = jj + 1
            c, a, b = (j > 1 ? X[j-1, j] : 0.0), X[j, j], X[j+1, j]
            x = rand()
            @test x * P.P[x, j] ≈ c * (j > 1 ? P.P[x, j-1] : 0.0) + a * P.P[x, j] + b * P.P[x, j+1]
            μ[j, m] = (2 + 2(h - 1) * a) * μ[j, m-1] - μ[j, m-2] + 2(h - 1) * (c * (j > 1 ? μ[j-1, m-1] : 0.0) + b * (j + 1 > m ? 0.0 : μ[j+1, m-1]))
        end
    end
    return UpperTriangular(μ)
end
function test_mu_coefficients(n, h)
    P = SemiclassicalJacobiArc(h, 0.0)
    μ = mu_coefficients(n, h)
    ϕ = acos(h)
    for m in 0:n
        for θ in LinRange(-ϕ, ϕ, 200)
            val = 0.0
            for j in 0:m
                val += μ[j+1, m+1] * P[θ, 2(j+1)-1] / μ[1, 1]
            end
            @test val ≈ cos(m * θ) rtol = 1e-12 atol = 1e-12
        end
        tf = transform(P, let m = m
            θ -> cos(m * θ)
        end)
        @test μ[1:(m+1), m+1] ./ μ[1, 1] ≈ tf[1:2:(2m+1)]
        @test tf[2:2:1000] ≈ zeros(500) atol = 1e-13
    end
end
function eta_coefficients(n, h)
    P = SemiclassicalJacobiArc(h, 0.0)
    X = jacobimatrix(P.Q)
    τ = P.P.t
    α, β = SemiclassicalOrthogonalPolynomials._linear_coefficients(P.Q.t, P.Q.a, P.Q.b, P.Q.c)
    η = zeros(n, n)
    η[1, 1] = (1 - h)^2 * (τ^2 * acsc(sqrt(τ)) + (2 - τ) * sqrt(τ - 1)) / 2
    η[1, 2] = η[1, 1] * 2(1 + (h - 1) * α)
    η[2, 2] = η[1, 1] * 2(h - 1) / (β)
    for m in 3:n
        for j in 1:m
            c, a, b = (j > 1 ? X[j-1, j] : 0.0), X[j, j], X[j+1, j]
            x = rand()
            @test x * P.Q[x, j] ≈ c * (j > 1 ? P.Q[x, j-1] : 0.0) + a * P.Q[x, j] + b * P.Q[x, j+1]
            η[j, m] = (2 + 2(h - 1) * a) * η[j, m-1] - η[j, m-2] + 2(h - 1) * (c * (j > 1 ? η[j-1, m-1] : 0.0) + b * (j + 1 > m ? 0.0 : η[j+1, m-1]))
        end
    end
    return UpperTriangular(η)
end
function test_eta_coefficients(n, h)
    P = SemiclassicalJacobiArc(h, 0.0)
    η = eta_coefficients(n, h)
    ϕ = acos(h)
    for m in 1:n
        for θ in LinRange(-ϕ, ϕ, 200)
            val = 0.0
            for j in 1:m
                val += η[j, m] * P[θ, 2j] / η[1, 1]
            end
            @test val ≈ sin(m * θ) rtol = 1e-12 atol = 1e-12
        end
        tf = transform(P, let m = m
            θ -> sin(m * θ)
        end)
        @test η[1:m, m] ./ η[1, 1] ≈ tf[2:2:(2m)]
        @test tf[1:2:1000] ≈ zeros(500) atol = 1e-13
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
    test_mu_coefficients(10, 0.3)
    test_mu_coefficients(17, 0.2)
    test_mu_coefficients(9, -0.5)
    test_eta_coefficients(10, 0.3)
    test_eta_coefficients(17, 0.2)
    test_eta_coefficients(9, -0.5)

    P = SemiclassicalJacobiArc(0.3, -1.0)
    for N in 0:10
        f = random_trigpoly(N)
        tf = transform(P, f)
        flag = length(LazyArrays.paddeddata(tf)) ≤ 2N + 1
        if flag
            @test flag
        else
            @test LazyArrays.paddeddata(tf)[(2N+2):end] ≈ zeros(length(LazyArrays.paddeddata(tf)) - 2N - 1) atol = 1e-12
        end
    end
    P = SemiclassicalJacobiArc(0.3, 2.0)
    for N in 0:25
        f = random_trigpoly(N)
        tf = transform(P, f)
        flag = length(LazyArrays.paddeddata(tf)) ≤ 2N + 1
        if flag
            @test flag
        else
            @test LazyArrays.paddeddata(tf)[(2N+2):end] ≈ zeros(length(LazyArrays.paddeddata(tf)) - 2N - 1) atol = 1e-12
        end
    end
end

@testset "ArcMultiplicationMatrix" begin
    for f in (
        θ -> ((y, x) = sincos(θ); x),
        θ -> ((y, x) = sincos(θ); y),
        θ -> ((y, x) = sincos(θ); x + y - x * y + x^13),
    )
        for (h, b) in ((0.3, -1.0), (0.0, 0.0), (0.1, 1.0))
            P = SemiclassicalJacobiArc(h, b)
            a = expand(P, f)
            M22, M21, M12, M11 = SAP.multiplication_blocks(a)
            Ja = zeros(1000, 1000)
            for j in 1:500
                Ja[1:2:end, 2j-1] .= M22[1:500, j]
                Ja[2:2:end, 2j-1] .= M12[1:500, j]
                Ja[1:2:end, 2j] .= M11[1:500, j]
                Ja[2:2:end, 2j] .= M21[1:500, j]
            end
            Jam = SAP.ArcMultiplicationMatrix(a)
            @test Jam[1:250, 1:250] ≈ Ja[1:250, 1:250]
            @test copy(Jam) === Jam
            @test size(Jam) == size(M22)
            @test axes(Jam) == axes(M22)
            bw = length(LazyArrays.paddeddata(a.args[2]))
            @test bandwidths(Jam) === (bw, bw)
            @test MemoryLayout(Jam) == SAP.ArcMultiplicationLayout()
            @test BandedMatrices.isbanded(Jam)
            @test MemoryLayout(@view Jam[1:5, 1:5]) == SAP.ArcMultiplicationLayout()
            @test MemoryLayout(@view Jam[1:5, :]) == LazyArrays.LazyBandedLayout()
            @test MemoryLayout(@view Jam[:, 1:5]) == LazyArrays.LazyBandedLayout()
            @test MemoryLayout(@view Jam[6:∞, 6:∞]) == LazyArrays.LazyBandedLayout()
            @test MemoryLayout(@view Jam[6:∞, :]) == LazyArrays.LazyBandedLayout()
            @test MemoryLayout(@view Jam[:, 6:∞]) == LazyArrays.LazyBandedLayout()
            @test Jam[1:100, 1:100] isa BandedMatrix
            @test BandedMatrix(sparse([Jam[i, j] for i in 1:100, j in 1:100])) == Jam[1:100, 1:100]
        end
    end
end

@testset "Multiplication Matrix" begin
    f = θ -> cos(θ)
    h, b = 0.3, -1.0
    P = SemiclassicalJacobiArc(h, b)
    a = expand(P, f)
    Jam = SAP.ArcMultiplicationMatrix(a)
    Jx = jacobimatrix(Val(1), P)
    @test Jam[1:500, 1:500] ≈ Jx[1:500, 1:500]
    f = θ -> sin(θ)
    a = expand(P, f)
    Jam = SAP.ArcMultiplicationMatrix(a)
    Jx = jacobimatrix(Val(2), P)
    @test Jam[1:500, 1:500] ≈ Jx[1:500, 1:500]

    for f in (
        θ -> ((y, x) = sincos(θ); y^5),
        θ -> ((y, x) = sincos(θ); y^3 + x^13 + x + y - x * y + 100)
    )
        for (h, b) in ((0.3, -1.0), (0.0, 0.0), (0.1, 1.0))
            P = SemiclassicalJacobiArc(h, b)
            a = expand(P, f)
            aP = a .* P
            for θ in LinRange(-acos(h) + eps(), acos(h) - eps(), 10)
                lhs = aP[θ, Block(1):Block(20)] # FIXME: Why doesn't this have block structure?
                rhs = f(θ) * P[θ, Block(1):Block(20)]
                @test lhs ≈ rhs atol = 1e-8 rtol = 1e-8
            end
        end
    end
end