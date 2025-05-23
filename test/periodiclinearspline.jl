using ArcPolynomials,
    Test,
    ForwardDiff,
    QuadGK,
    QuasiArrays,
    ContinuumArrays,
    DomainSets

const AP = ArcPolynomials

@testset "Trigonometric Spline" begin
    p1 = LinRange(-pi, pi, 10)
    p2 = LinRange(-pi, pi, 23)
    p3 = [-pi / 2, 0.1, 0.3, 1.0, 2.5, 2.6, 3.0, 4.5, 3pi / 2] .- pi / 2
    p4 = LinRange(-pi, pi, 3)
    for points in (p1, p2, p3, p4)
        inp = let points = points
            (S, i, j) -> begin
                integrand = θ -> S[θ, i] * S[θ, j]
                val, _ = quadgk(integrand, LinRange(-pi, pi, 20)..., atol=1e-12, rtol=1e-12)
                abs(val) < 1e-5 && (val = 0.0)
                return val
            end
        end
        S = PeriodicLinearSpline(points)
        @test axes(S) == (Inclusion(ℝ), Base.OneTo(1:length(S.points)-1))
        for i in 1:(length(points)-1)
            @test S[points[i], i] ≈ 1.0
            for j in eachindex(points)
                i == j && continue
                if j == lastindex(points) && i == 1
                    @test S[points[j], i] ≈ 1.0
                else
                    @test S[points[j], i] ≈ 0.0 atol = 1e-12
                end
            end
        end
        S = PeriodicLinearSpline(points)
        X = inp.(Ref(S), 1:(length(points)-1), (1:(length(points)-1))')
        Y = S'S
        @test X ≈ Y atol = 1e-9
        @test grid(S, 12) == S.points
        @test S == S
    end
end

@testset "ForwardDiff" begin
    function _diff(spl, i, x)
        z = BigFloat(x)
        h = 1e-10
        return (spl[z + h, i] - spl[z - h, i]) / (2h)
    end
    function _fddiff(spl, i, x)
        return ForwardDiff.derivative(θ -> spl[θ, i], x)
    end
    spl = PeriodicLinearSpline([-pi / 2, 0.1, 0.3, 1.0, 2.5, 2.6, 3.0, 4.5, 3pi / 2] .- pi / 2)
    for i in 1:(length(spl.points)-1)
        for θ in LinRange(-4.4012pi, 4.4012pi, 250) # Near integer multiples of 2pi, mod2pi is not differentiable and so ForwardDiff gives us NaNs. Not sure what the correct workaround is. (This only matters for values outside of [-pi/2, 3pi/2].)
            @test _diff(spl, i, θ) ≈ _fddiff(spl, i, θ) rtol = 1e-4 atol = 1e-4
        end
    end
end