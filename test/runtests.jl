using ArcPolynomials, Test, SafeTestsets

@testset verbose = true "ArcPolynomials.jl" begin
    @safetestset "ArcPolynomials" begin
        include("arcpolynomials.jl")
    end
    @safetestset "TrigonometricSpline" begin
        include("trigonometricspline.jl")
    end
    @safetestset "CyclicBandedMatrices" begin
        include("cyclicbandedmatrices.jl")
    end
    @safetestset "CyclicArrowheadMatrices" begin
        include("cyclicarrowhead.jl")
    end
    @safetestset "PiecewiseArcPolynomials" begin
        include("piecewisearcpolynomials.jl")
    end
    @safetestset "PeriodicLinearSpline" begin
        include("periodiclinearspline.jl")
    end 
    @safetestset "PeriodicContinuousPolynomials" begin
        include("periodiccontinuouspolynomial.jl")
    end
end 
