using ArcPolynomials,
    Test,
    BandedMatrices,
    LazyArrays,
    FillArrays,
    Infinities,
    BlockArrays,
    BlockBandedMatrices,
    PiecewiseOrthogonalPolynomials,
    InfiniteRandomArrays,
    LinearAlgebra,
    MatrixFactorizations,
    ArrayLayouts,
    SparseArrays,
    Random

const AP = ArcPolynomials
const PAP = AP.PiecewiseArcPolynomials
const CBMs = AP.CyclicBandedMatrices
const CBM = CBMs.CyclicBandedMatrix
const CyclicBBBs = AP.CyclicBBBArrowheadMatrices
const CyclicBBB = CyclicBBBs.CyclicBBBArrowheadMatrix

@testset "InterlacedMatrix" begin
    D = [BandedMatrix((0 => randn(4) .+ 25, 2 => randn(2), -1 => randn(3)), (4, 4)) for _ in 1:4]
    Df = CyclicBBBs.InterlacedMatrix(D)
    @test MemoryLayout(Df) == BlockBandedMatrices.BandedBlockBandedLayout()
    @test Base.BroadcastStyle(typeof(Df)) == CyclicBBBs.InterlacedMatrixStyle() == CyclicBBBs.InterlacedMatrixStyle(Val(2))
    @test blockbandwidths(Df) == (1, 2)
    @test blocksize(Df) == (4, 4)
    @test subblockbandwidths(Df) == (0, 0)
    Ds = BlockedArray(zeros(16, 16), (BlockedOneTo(4:4:16), BlockedOneTo(4:4:16)))
    @test axes(Df) == axes(Ds)
    for i in 1:4
        for j in 1:4
            Ds[Block(i, j)] .= Diagonal([D[1][i, j], D[2][i, j], D[3][i, j], D[4][i, j]])
        end
    end
    @test Df == Ds
    @test Df[3, 2] == Ds[3, 2]
    @test Df[Block(2, 2)] == Ds[Block(2, 2)]
    @test Df[Block(2, 3)] == Ds[Block(2, 3)]
    @test Df' == Ds'
    @test transpose(Df) == transpose(Ds)
    @test copy(Df) == Ds && !(copy(Df) === Df)
    @test 2Df == 2Ds && 2Df isa CyclicBBBs.InterlacedMatrix
    @test Df / 2 == Ds / 2 && Df / 2 isa CyclicBBBs.InterlacedMatrix
    @test Df ./ 2 == Ds ./ 2 && Df ./ 2 isa CyclicBBBs.InterlacedMatrix
    @test Df + 3Df == Ds + 3Ds && Df + 3Df isa CyclicBBBs.InterlacedMatrix
    @test -Df == -Ds && -Df isa CyclicBBBs.InterlacedMatrix
    Df[3, 7] = 27.5
    @test Df[3, 7] == 27.5
    @test_throws ArgumentError Df[3, 6] = 27.3
    @test Df[Block(1, 1)] isa Diagonal

    Ds = similar(Df)
    @test typeof(Ds) == typeof(Df)
    @test axes(Df) == axes(Ds)
    @test MemoryLayout(Df) == MemoryLayout(Ds)
    @test Ds isa CyclicBBBs.InterlacedMatrix
    @test !(Ds === Df)
    @test Ds ≠ Df
    copyto!(Ds, Df)
    @test Ds == Df

    for _ in 1:100
        D = [BandedMatrix((0 => randn(4) .+ 25, 2 => randn(2), -1 => randn(3)), (4, 4)) for _ in 1:4]
        @test Matrix(Symmetric(CyclicBBBs.InterlacedMatrix(D))) == Matrix(CyclicBBBs.InterlacedMatrix(Symmetric.(D)))
    end

    D = [BandedMatrix((0 => randn(4) .+ 25, 2 => randn(2), -1 => randn(3)), (4, 4)) for _ in 1:4]
    Df = CyclicBBBs.InterlacedMatrix(D)
    Dc = LinearAlgebra.cholcopy(Symmetric(Df))
    @test eltype(Dc) == eltype(Df)
    @test Dc == Symmetric(Df) && !(Dc === Symmetric(Df))
    @test typeof(Dc) == typeof(Symmetric(Df))
    chol = cholesky(Symmetric(Df))
    U = chol.U
    UF = cholesky(Symmetric(Matrix(Df))).U
    @test U' * U ≈ Symmetric(Df)
    @test Symmetric(Df) == Dc # no mutation 
    @test U ≈ UF
    cholesky!(Dc)
    @test UpperTriangular(Dc) ≈ UF

    D = [BandedMatrix((0 => randn(17) .+ 250, 2 => randn(15), -1 => randn(16)), (17, 17)) for _ in 1:4]
    Df = CyclicBBBs.InterlacedMatrix(D)
    Dc = LinearAlgebra.cholcopy(Symmetric(Df))
    chol = reversecholesky(Symmetric(Df))
    U = chol.U
    UF = reversecholesky(Symmetric(Matrix(Df))).U
    @test U * U' ≈ Symmetric(Df)
    @test Symmetric(Df) == Dc # no mutation
    @test U ≈ UF
    reversecholesky!(Dc)
    @test UpperTriangular(Dc) ≈ UF
end

@testset "Constructor" begin
    n = 4
    p = 5
    A = CBM(BandedMatrix(0 => 1:n, 1 => 1:n-1, -1 => 1:n-1), 1 / 2, -1 / 2)
    B = ntuple(_ -> BandedMatrix((0 => randn(n - 1), -1 => randn(n - 1)), (n, n)), 2)
    C = ntuple(_ -> CBM(BandedMatrix((0 => randn(n), 1 => randn(n - 1)), (n, n)), 1.0, 1.0), 3)
    D = fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p - 2), -1 => randn(p - 1)), (p, p)), n - 1)
    K = CyclicBBB{Float64}(A, B, C, D)
    @test K isa CyclicBBB
    @test K.A === A
    @test K.B === B
    @test K.C === C
    @test K.D == CyclicBBBs.InterlacedMatrix(D)
    @test axes(K) == (BlockedOneTo(4:3:19), BlockedOneTo(4:3:19))
    @test size(K) == (19, 19)
    @test blockbandwidths(K) == (3, 2)
    @test blocksize(K) == (6, 6)
    @test MemoryLayout(K) isa CyclicBBBs.CyclicArrowheadLayouts

    data = Hcat(Ones{Float64}(∞), Zeros{Float64}(∞), 2Ones{Float64}(∞), 5Ones{Float64}(∞))
    D = Fill(BandedMatrices._BandedMatrix(data', axes(data, 1), 2, 1), n)
    K = CyclicBBB{Float64}(A, B, C, D)
    @test K isa CyclicBBB
    @test axes(K) == (BlockedOneTo(4:3:∞), BlockedOneTo(4:3:∞))
    @test size(K) == (∞, ∞)
    @test blockbandwidths(K) == (3, 2)
    @test blocksize(K) == (∞, ∞)
    @test MemoryLayout(K) isa CyclicBBBs.CyclicArrowheadLayouts

    @test K' isa CyclicBBB
    @test (K')[1:300, 1:300] == K[1:300, 1:300]'
end

@testset "copy" begin
    n = 4
    p = 5
    A = CBM(BandedMatrix(0 => 1:n, 1 => 1:n-1, -1 => 1:n-1), 1 / 2, -1 / 2)
    B = ntuple(_ -> BandedMatrix((0 => randn(n - 1), -1 => randn(n - 1)), (n, n)), 2)
    C = ntuple(_ -> CBM(BandedMatrix((0 => randn(n), 1 => randn(n - 1)), (n, n)), 1.0, 1.0), 3)
    D = fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p - 2), -1 => randn(p - 1)), (p, p)), n)
    K = CyclicBBB{Float64}(A, B, C, D)
    KK = copy(K)
    @test KK == K
    @test !(KK === K)
    @test KK isa CyclicBBB
end

@testset "Algebra" begin
    n = 4
    A = CBM(
        BandedMatrix(0 => rand(n) .+ 10, 1 => randn(n - 1), -1 => rand(n - 1)),
        1 / 5, -1 / 2
    )
    B = ntuple(_ -> BandedMatrix(0 => randn(n), -1 => randn(n - 1)), 2)
    C = ntuple(_ -> CBM(
            BandedMatrix(0 => randn(n), 1 => randn(n - 1)),
            1.0, 1.0
        ), 3)
    D = fill(BandedMatrix(0 => randn(n) .+ 10, 2 => randn(n - 1)), n)
    K = CyclicBBB{Float64}(A, B, C, D)

    _2K = 2K
    _K2 = K * 2
    _2oK = 2 \ K
    _Ko2 = K / 2
    _KpK = K + K
    _KmK = K - K
    @test all(x -> x isa CyclicBBB, (_2K, _K2, _2oK, _Ko2, _KpK, _KmK))
    @test _2K.A == 2A && _2K.B == map(x -> 2x, B) && _2K.C == map(x -> 2x, C) && _2K.D.D == 2D
    @test _K2.A == 2A && _K2.B == map(x -> 2x, B) && _K2.C == map(x -> 2x, C) && _K2.D.D == 2D
    @test _2oK.A == 2 \ A && _2oK.B == map(x -> 2 \ x, B) && _2oK.C == map(x -> 2 \ x, C) && _2oK.D.D == 2 \ D
    @test _Ko2.A == A / 2 && _Ko2.B == map(x -> x / 2, B) && _Ko2.C == map(x -> x / 2, C) && _Ko2.D.D == D / 2
    @test _KpK.A == A + A && _KpK.B == map(x -> x + x, B) && _KpK.C == map(x -> x + x, C) && _KpK.D.D == D + D
    @test _KmK.A == A - A && _KmK.B == map(x -> x - x, B) && _KmK.C == map(x -> x - x, C) && _KmK.D.D == D - D

    @test 2K == K * 2 == K + K == 2Matrix(K)
    @test all(iszero, K - K)
    @test K / 2 == 2 \ K == Matrix(K) / 2

    data = Hcat(Ones{Float64}(∞), Zeros{Float64}(∞), 2Ones{Float64}(∞), 5Ones{Float64}(∞))
    D = Fill(BandedMatrices._BandedMatrix(data', axes(data, 1), 2, 1), n)
    K = CyclicBBB{Float64}(A, B, C, D)

    _2K = 2K
    _K2 = K * 2
    _2oK = 2 \ K
    _Ko2 = K / 2
    _KpK = K + K
    _KmK = K - K
    @test all(x -> x isa CyclicBBB, (_2K, _K2, _2oK, _Ko2, _KpK, _KmK))
    @test _2K.A == 2A && _2K.B == map(x -> 2x, B) && _2K.C == map(x -> 2x, C) && length(_2K.D.D) == length(D) && all(d -> _2K.D.D[d][1:300, 1:300] == 2D[d][1:300, 1:300], eachindex(D))
    @test _K2.A == 2A && _K2.B == map(x -> 2x, B) && _K2.C == map(x -> 2x, C) && length(_K2.D.D) == length(D) && all(d -> _K2.D.D[d][1:300, 1:300] == 2D[d][1:300, 1:300], eachindex(D))
    @test _2oK.A == 2 \ A && _2oK.B == map(x -> 2 \ x, B) && _2oK.C == map(x -> 2 \ x, C) && length(_2oK.D.D) == length(D) && all(d -> _2oK.D.D[d][1:300, 1:300] == 2 \ D[d][1:300, 1:300], eachindex(D))
    @test _Ko2.A == A / 2 && _Ko2.B == map(x -> x / 2, B) && _Ko2.C == map(x -> x / 2, C) && length(_Ko2.D.D) == length(D) && all(d -> _Ko2.D.D[d][1:300, 1:300] == D[d][1:300, 1:300] / 2, eachindex(D))
    @test _KpK.A == A + A && _KpK.B == map(x -> x + x, B) && _KpK.C == map(x -> x + x, C) && length(_KpK.D.D) == length(D) && all(d -> _KpK.D.D[d][1:300, 1:300] == D[d][1:300, 1:300] + D[d][1:300, 1:300], eachindex(D))
    @test _KmK.A == A - A && _KmK.B == map(x -> x - x, B) && _KmK.C == map(x -> x - x, C) && length(_KmK.D.D) == length(D) && all(d -> _KmK.D.D[d][1:300, 1:300] == D[d][1:300, 1:300] - D[d][1:300, 1:300], eachindex(D))
end

@testset "mul" begin
    n = 4
    A = CBM(
        BandedMatrix(0 => rand(n) .+ 10, 1 => randn(n - 1), -1 => rand(n - 1)),
        1 / 5, -1 / 2
    )
    B = ntuple(_ -> BandedMatrix(0 => randn(n), -1 => randn(n - 1)), 2)
    C = ntuple(_ -> CBM(
            BandedMatrix(0 => randn(n), 1 => randn(n - 1)),
            1.0, 1.0
        ), 3)
    D = fill(BandedMatrix(0 => randn(n) .+ 10, 2 => randn(n - 1)), n)
    K = CyclicBBB{Float64}(A, B, C, D)

    x = randn(size(K, 1))
    X = randn(size(K))
    X̃ = randn(size(K, 1), 5)
    @test K * x ≈ Matrix(K) * x
    @test K' * x ≈ Matrix(K)' * x
    @test K * X ≈ Matrix(K) * X
    @test K * X̃ ≈ Matrix(K) * X̃
    @test X * K ≈ X * Matrix(K)

    data = Hcat(Ones{Float64}(∞), Zeros{Float64}(∞), 2Ones{Float64}(∞), 5Ones{Float64}(∞))
    D = Fill(BandedMatrices._BandedMatrix(data', axes(data, 1), 2, 1), n)
    K = CyclicBBB{Float64}(A, B, C, D)
    x = InfRandVector(; dist=InfiniteRandomArrays.Normal())
    X = InfRandMatrix(; dist=InfiniteRandomArrays.Normal())
    X̃ = InfRandMatrix(∞, 5; dist=InfiniteRandomArrays.Normal())
    @test (K*x)[1:300] ≈ (K[1:1000, 1:1000]*x[1:1000])[1:300]
    @test (K'*x)[1:300] ≈ (K[1:1000, 1:1000]'*x[1:1000])[1:300]
    @test (K*X)[1:300, 1:300] ≈ (K[1:1000, 1:1000]*X[1:1000, 1:300])[1:300, 1:300]
    @test (K*X̃)[1:300, 1:5] ≈ (K[1:1000, 1:1000]*X̃[1:1000, 1:5])[1:300, 1:5]
    @test (X*K)[1:300, 1:300] ≈ (X[1:1000, 1:1000]*K[1:1000, 1:1000])[1:300, 1:300]
end

@testset "Triangular" begin
    n = 4
    A = CBM(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n - 1), -1 => randn(n - 1)), 1 / 10, 1 / 2)
    B = [CBM(BandedMatrix((0 => randn(n - 1), -1 => randn(n - 2)), (n, n)), 0.0, -1.0) for _ in 1:2]
    C = [CBM(BandedMatrix((0 => randn(n - 1), 1 => randn(n - 2)), (n, n)), 1.0, 0.0) for _ in 1:2]
    D = fill(BandedMatrix((0 => randn(n) .+ 10, 2 => randn(n - 2), -1 => randn(n - 1)), (n, n)), n)
    K = CyclicBBB{Float64}(A, B, C, D)
    data = Hcat(Ones{Float64}(∞), Zeros{Float64}(∞), 2Ones{Float64}(∞), 5Ones{Float64}(∞))
    Dinf = Fill(BandedMatrices._BandedMatrix(data', axes(data, 1), 2, 1), n)
    Kinf = CyclicBBB{Float64}(A, B, C, Dinf)

    c = randn(size(K, 1))

    for T in (UpperTriangular(K), UnitUpperTriangular(K), LowerTriangular(K), UnitLowerTriangular(K), UpperTriangular(K)')
        @test T \ c ≈ Matrix(T) \ c
        @test c' / T ≈ c' / Matrix(T)
    end
    for K in (K, Kinf)
        for Typ in (UpperTriangular, UnitUpperTriangular)
            @test Typ(K).A == Typ(K.A)
            @test Typ(K).B == K.B
            @test isempty(Typ(K).C)
            if K === Kinf
                @test all(d -> Typ(K).D.D[d][1:300, 1:300] == Typ(K.D.D[d])[1:300, 1:300], eachindex(K.D.D))
            else
                @test Typ(K).D.D == map(Typ, K.D.D)
            end
        end
        for Typ in (LowerTriangular, UnitLowerTriangular)
            @test Typ(K).A == Typ(K.A)
            @test isempty(Typ(K).B)
            @test Typ(K).C == K.C
            if K === Kinf
                @test all(d -> Typ(K).D.D[d][1:300, 1:300] == Typ(K.D.D[d])[1:300, 1:300], eachindex(K.D.D))
            else
                @test Typ(K).D.D == map(Typ, K.D.D)
            end
        end
    end
end

@testset "Broadcast fixes" begin
    P = PiecewiseArcPolynomial{1}(10)
    M = P'P
    Msub = M[1:100, 1:100]
    @test (2M)[1:100, 1:100] ≈ 2Msub
    @test (M*2)[1:100, 1:100] ≈ 2Msub
    @test (M/2)[1:100, 1:100] ≈ Msub / 2
end

@testset "MemoryLayout fixes" begin
    𝛉 = [-π, -π / 4, 0, π / 4, π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    Q = ContinuousPolynomial{1}(𝛉)
    R = PeriodicContinuousPolynomial{1}(𝛉)

    SA = Symmetric(P'P)
    SB = Symmetric(Q'Q)
    SC = Symmetric(R'R)
    A = parent(SA)
    B = parent(SB)
    C = parent(SC)
    @test MemoryLayout(SA) === MemoryLayout(SC) === SymmetricLayout{CyclicBBBs.LazyCyclicArrowheadLayout}()
    @test MemoryLayout(A) === MemoryLayout(C) === CyclicBBBs.LazyCyclicArrowheadLayout()
    @test colsupport(SA, 1) == colsupport(A, 1) == 1:12
    @test rowsupport(SA, 1) == rowsupport(A, 1) == 1:12
    @test colsupport(SA, 8) == colsupport(A, 8) == 1:16
    @test colsupport(SA, 37) == colsupport(A, 37) == 29:48
    @test colsupport(SC, 1) == colsupport(C, 1) == 1:12
    @test rowsupport(SC, 1) == rowsupport(C, 1) == 1:12
    @test colsupport(SC, 8) == colsupport(C, 8) == 1:16
    @test colsupport(SC, 37) == colsupport(C, 37) == 29:48

    𝛉 = [-π, -π / 4, π / 4, π]
    W = PeriodicContinuousPolynomial{1}(𝛉)
    P = PiecewiseArcPolynomial{1}(𝛉)
    DP, DW = diff(P).args[2], diff(W).args[2]
    @test MemoryLayout(DP) == MemoryLayout(DW) == CyclicBBBs.LazyCyclicArrowheadLayout()
end

@testset "Submatrix" begin
    n = 4
    p = 10
    A = CBM(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n - 1), -1 => randn(n - 1)), 1 / 10, 1 / 2)
    B = [CBM(BandedMatrix((0 => randn(n - 1), -1 => randn(n - 2)), (n, n)), 0.0, -1.0) for _ in 1:2]
    C = [CBM(BandedMatrix((0 => randn(n - 1), 1 => randn(n - 2)), (n, n)), 1.0, 0.0) for _ in 1:2]
    D = [BandedMatrix((0 => randn(p) .+ 25p, 2 => randn(p - 2), -1 => randn(p - 1)), (10, 10)) for _ in 1:n]
    K = CyclicBBB{Float64}(A, B, C, D)
    Dfull = K[Block.(2:p+1), Block.(2:p+1)]
    @test CyclicBBBs.to_interlace(Dfull) == D
    @inferred CyclicBBBs.to_interlace(Dfull)

    𝛉 = [-π; π * (2sort!(rand(8)) .- 1); π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    _M = M[Block.(1:50), Block.(1:50)]
    for k in 1:40
        A = CyclicBBBs.principal_submatrix(_M, Block(k))
        @test A == M[Block.(1:k), Block.(1:k)]
    end
    for k in 1:40
        A = CyclicBBBs.principal_submatrix(M, Block(k))
        @test A == M[Block.(1:k), Block.(1:k)]
        @test all(≤((2, 2)), bandwidths.(A.D.D))
    end
end

@testset "Reverse Cholesky" begin
    𝛉 = [-π, -1.76, -1.58, -1.53, -1.12, -0.8, 0.09, π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 4
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    nb = length(blockaxes(K, 1))
    A = K.A
    B = @view K[Block(1), Block(2):Block(nb)]
    C = @view K[Block(2):Block(nb), Block(1)]
    D = @view K[Block(2):Block(nb), Block(2):Block(nb)]
    L̃ = reversecholesky(D).L
    L₀ = reversecholesky(Matrix(A) - Matrix(B) * Matrix(inv(L̃)) * Matrix(inv(L̃)') * Matrix(B')).L
    L = BlockedArray(zeros(eltype(K), size(K)), axes(K))
    L[Block(1), Block(1)] = L₀
    L[Block(2):Block(nb), Block(1)] = inv(L̃)' * B'
    L[Block(2):Block(nb), Block(2):Block(nb)] = L̃
    L = LowerTriangular(L)
    @test L' * L ≈ K

    𝛉 = [-π; π * (2sort!(rand(6)) .- 1); π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 4
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kcopy = Matrix(K)
    chol = reversecholesky(Symmetric(K))
    @test chol.factors * chol.factors' ≈ K
    chol = reversecholesky!(Symmetric(K))
    @test chol.factors * chol.factors' ≈ Kcopy

    𝛉 = [-π; π * (2sort!(rand(16)) .- 1); π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 8
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kcopy = Matrix(K)
    chol = reversecholesky(Symmetric(K))
    @test chol.factors * chol.factors' ≈ K
    chol = reversecholesky!(Symmetric(K))
    @test chol.factors * chol.factors' ≈ Kcopy

    𝛉 = [-π; π * (2sort!(rand(16)) .- 1); π]
    P = PiecewiseArcPolynomial{0}(𝛉)
    M = P'P
    k = 4
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kcopy = Matrix(K)
    chol = reversecholesky(Symmetric(K))
    @test chol.factors * chol.factors' ≈ K
    chol = reversecholesky!(Symmetric(K))
    @test chol.factors * chol.factors' ≈ Kcopy

    𝛉 = LinRange(-π, π, 3)
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 6
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kcopy = Matrix(K)
    chol = reversecholesky(Symmetric(K))
    @test chol.factors * chol.factors' ≈ K
    chol = reversecholesky!(Symmetric(K))
    @test chol.factors * chol.factors' ≈ Kcopy

    𝛉 = LinRange(-π, π, 13)
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 8
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kcopy = Matrix(K)
    chol = reversecholesky(Symmetric(K))
    @test chol.factors * chol.factors' ≈ K
    chol = reversecholesky!(Symmetric(K))
    @test chol.factors * chol.factors' ≈ Kcopy

    𝛉 = LinRange(-π, π, 17)
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 25
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    b = rand(size(K, 2))
    chol = reversecholesky(Symmetric(K)).factors
    z = chol \ b
    x = chol' \ z
    @test K * x ≈ b

    𝛉 = LinRange(-π, π, 3)
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 8
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    b = rand(size(K, 2))
    chol = reversecholesky(Symmetric(K)).factors
    z = chol \ b
    x = chol' \ z
    @test K * x ≈ b

    𝛉 = [-π; π * (2sort!(rand(16)) .- 1); π]
    P = PiecewiseArcPolynomial{1}(𝛉)
    M = P'P
    k = 8
    K = CyclicBBBs.principal_submatrix(M, Block(k))
    Kc = copy(K)
    b = rand(size(K, 2))
    bc = copy(b)
    chol = reversecholesky!(Symmetric(K))
    ldiv!(chol, b)
    @test Kc * b ≈ bc

    ___A = CBMs._CyclicBandedMatrix(
        [2.4886e-311 1.24396e-311 0.0 0.0
            -0.0475711 -0.0475711 -0.0475711 -0.0475711
            3.62943 3.62943 3.62943 3.62943
            -0.0475711 -0.0475711 -0.0475711 -0.0475711
            0.0 0.0 6.95277e-310 1.24395e-311], 1, 1, (2, 2))
    ___B = (
        CBMs._CyclicBandedMatrix(
            [0.5890486225480862 2.7988533303504e-311 2.798853585069e-311 2.7988713686575e-311; 0.5890486225480862 0.5890486225480862 0.5890486225480862 0.5890486225480862; 0.5890486225480862 0.5890486225480862 0.5890486225480862 0.0],
            1, 0, (1, 1)
        ),
        CBMs._CyclicBandedMatrix(
            [0.11780972450961724 1.564373409690294e-309 4.3196e-319 1.4099784044774426; -0.11780972450961724 -0.11780972450961724 -0.11780972450961724 -0.11780972450961724; 0.11780972450961724 0.11780972450961724 0.11780972450961724 0.0],
            1, 0, (1, 1)
        )
    )
    ___C = (
        CBMs._CyclicBandedMatrix(
            [0.0 0.5890486225480862 0.5890486225480862 0.5890486225480862; 0.5890486225480862 0.5890486225480862 0.5890486225480862 0.5890486225480862; 2.7988533303504e-311 2.798853585069e-311 2.7988713686575e-311 0.5890486225480862],
            0, 1, (1, 1)
        ),
        CBMs._CyclicBandedMatrix(
            [0.0 0.11780972450961724 0.11780972450961724 0.11780972450961724; -0.11780972450961724 -0.11780972450961724 -0.11780972450961724 -0.11780972450961724; 1.564373409690294e-309 4.3196e-319 1.4099784044774426 0.11780972450961724],
            0, 1, (1, 1)
        )
    )
    ___D = [
        BandedMatrices._BandedMatrix(
            [0.0 0.0 -0.0336599
                0.0 0.0 0.0
                1.32007 0.576616 0.386223
                0.0 0.0 0.0
                -0.0336599 -0.01122 -0.00509999], 1:3, 2, 2
        ),
        BandedMatrices._BandedMatrix(
            [0.0 0.0 -0.0336599
                0.0 0.0 0.0
                1.32007 0.576616 0.386223
                0.0 0.0 0.0
                -0.0336599 -0.01122 -0.00509999], 1:3, 2, 2),
        BandedMatrices._BandedMatrix(
            [0.0 0.0 -0.0336599
                0.0 0.0 0.0
                1.32007 0.576616 0.386223
                0.0 0.0 0.0
                -0.0336599 -0.01122 -0.00509999], 1:3, 2, 2),
        BandedMatrices._BandedMatrix(
            [0.0 0.0 -0.0336599
                0.0 0.0 0.0
                1.32007 0.576616 0.386223
                0.0 0.0 0.0
                -0.0336599 -0.01122 -0.00509999], 1:3, 2, 2)
    ]
    S = Symmetric(CyclicBBB(___A, ___B, ___C, ___D))
    chol = reversecholesky(S)
    @test chol.factors * chol.factors' ≈ S
    _S = deepcopy(S)
    chol = reversecholesky!(S)
    @test chol.factors * chol.factors' ≈ _S
    b = rand(size(S, 2))
    bc = copy(b)
    __S = deepcopy(_S)
    chol = reversecholesky!(_S)
    ldiv!(chol, b)
    @test Matrix(__S) * b ≈ bc
end