using ArcPolynomials,
    Test,
    BandedMatrices,
    ArrayLayouts,
    LinearAlgebra,
    SemiseparableMatrices,
    MatrixFactorizations

const AP = ArcPolynomials
const CBMs = AP.CyclicBandedMatrices
const CBM = CBMs.CyclicBandedMatrix

@testset "_maybeextend" begin
    A = brand(20, 20, 2, 3)
    @test CBMs._maybeextend(A) == A && !(CBMs._maybeextend(A) === A)
    A = brand(20, 20, 0, 0)
    @test CBMs._maybeextend(A) == BandedMatrix(-1 => zeros(19), 0 => A[band(0)], 1 => zeros(19)) && bandwidths(CBMs._maybeextend(A)) == (1, 1)
    A = brand(20, 20, 3, 0)
    @test CBMs._maybeextend(A) == BandedMatrix(-3 => A[band(-3)], -2 => A[band(-2)], -1 => A[band(-1)], 0 => A[band(0)], 1 => zeros(19)) && bandwidths(CBMs._maybeextend(A)) == (3, 1)
    A = brand(20, 20, 0, 1)
    @test CBMs._maybeextend(A) == BandedMatrix(-1 => zeros(19), 0 => A[band(0)], 1 => A[band(1)]) && bandwidths(CBMs._maybeextend(A)) == (1, 1)
    A = brand(20, 20, -3, -3)
    @test size(CBMs._maybeextend(A).data) == (3, 20)
    A = brand(20, 20, 3, -6)
    @test size(CBMs._maybeextend(A).data) == (5, 20)
    A = brand(20, 20, 4, -2)
    @test CBMs._maybeextend(A) == BandedMatrix(-4 => A[band(-4)], -3 => A[band(-3)], -2 => A[band(-2)], -1 => A[band(-1)], 0 => A[band(0)], 1 => zeros(19)) && bandwidths(CBMs._maybeextend(A)) == (4, 1)
    A = brand(2, 2, 0, 1)
    @test CBMs._maybeextend(A) == A && bandwidths(CBMs._maybeextend(A)) == (1, 1)
    A = brand(2, 2, 0, 0)
    @test CBMs._maybeextend(A) == A && bandwidths(CBMs._maybeextend(A)) == (1, 1)
    A = brand(2, 2, -1, 0)
    @test bandwidths(CBMs._maybeextend(A)) == (1, 1)
end

@testset "Constructor" begin
    function _to_cbm(A, a, b)
        mat = zeros(size(A))
        copyto!(mat, A)
        mat[1, end] = a
        mat[end, 1] = b
        mat
    end
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == _to_cbm(A, a, b)

    A = brand(10, 10, 1, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B.extended_ℓu == CBMs._extended_bpbandwidths(B) == (1, 1)
    @test B == _to_cbm(A, a, b)

    A = brand(20, 20, -3, 2)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == _to_cbm(A, a, b)

    A = brand(20, 20, 0, 1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == _to_cbm(A, a, b)

    A = brand(2, 2, 0, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == [A[1, 1] a; b A[2, 2]]

    A = brand(2, 2, 1, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == [A[1, 1] a; b A[2, 2]]

    A = brand(2, 2, -1, -1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == [A[1, 1] a; b A[2, 2]]

    A = brand(2, 2, 0, 1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test B == [A[1, 1] a; b A[2, 2]]

    B = BandedMatrix(-2 => [0.0], -1 => [0.23722433407530427, 0.08758173278821277], 0 => [0.32713718227154587, 0.36876596676325474, 0.28473466997327074], 1 => [0.23722433407530427, 0.08758173278821277], 2 => [0.0])
    a = 0.06092010379722579
    C = CBM(B, a, a)
    @test C[1,3]==C[3,1]==a
    @test C ≈ B + [0 0 a; 0 0 0; a 0 0]
end

@testset "bandpart" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._bpbandwidths(B) == (2, 3)
    @test CBMs._bpbandwidths(B, 1) == 2
    @test CBMs._bpbandwidths(B, 2) == 3
    @test CBMs._extended_bpbandwidths(B) == (2, 3)
    @test CBMs._extended_bpbandwidths(B, 1) == 2
    @test CBMs._extended_bpbandwidths(B, 2) == 3
    @test CBMs.bandedpart(B) == A

    A = brand(20, 20, 2, -1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._bpbandwidths(B) == (2, -1)
    @test CBMs._bpbandwidths(B, 1) == 2
    @test CBMs._bpbandwidths(B, 2) == -1
    @test CBMs._extended_bpbandwidths(B) == (2, 1)
    @test CBMs._extended_bpbandwidths(B, 1) == 2
    @test CBMs._extended_bpbandwidths(B, 2) == 1
    @test CBMs.bandedpart(B) == A

    A = brand(20, 20, 0, 1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._bpbandwidths(B) == (0, 1)
    @test CBMs._bpbandwidths(B, 1) == 0
    @test CBMs._bpbandwidths(B, 2) == 1
    @test CBMs._extended_bpbandwidths(B) == (1, 1)
    @test CBMs._extended_bpbandwidths(B, 1) == 1
    @test CBMs._extended_bpbandwidths(B, 2) == 1
    @test CBMs.bandedpart(B) == A

    A = brand(2, 2, 0, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._bpbandwidths(B) == (0, 0)
    @test CBMs._bpbandwidths(B, 1) == 0
    @test CBMs._bpbandwidths(B, 2) == 0
    @test CBMs._extended_bpbandwidths(B) == (1, 1)
    @test CBMs._extended_bpbandwidths(B, 1) == 1
    @test CBMs._extended_bpbandwidths(B, 2) == 1

    A = brand(10, 10, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._bpbandwidths(B') == (3, 2)
    @test CBMs._bpbandwidths(B', 1) == 3
    @test CBMs._bpbandwidths(B', 2) == 2
    @test CBMs._extended_bpbandwidths(B') == (3, 2)
    @test CBMs._extended_bpbandwidths(B', 1) == 3
    @test CBMs._extended_bpbandwidths(B', 2) == 2
    @test CBMs.bandedpart(B') == CBMs.bandedpart(B)'
end

@testset "axes" begin
    A = brand(20, 20, 2, 3)
    @test CBMs._ncols(A) == 20
    a, b = rand(2)
    B = CBM(A, a, b)
    @test CBMs._ncols(B) == 20
    @test CBMs._raxis(B) == Base.OneTo(20)
    @test axes(B) == axes(A)
    @test size(B) == size(A)
end

@testset "layout" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test MemoryLayout(B) == CBMs.CyclicBandedLayout{typeof(MemoryLayout(A))}()
    @test MemoryLayout(Symmetric(B)) == CBMs.SymmetricLayout{typeof(MemoryLayout(B))}()

    @test colsupport(B, 1) == 1:20
    @test colsupport(B, 2) == 1:4
    @test colsupport(B, 5) == 2:7
    @test colsupport(B, 18) == 15:20
    @test colsupport(B, 20) == 1:20
    @test rowsupport(B, 1) == 1:20
    @test rowsupport(B, 2) == 1:5
    @test rowsupport(B, 5) == 3:8
    @test rowsupport(B, 7) == 5:10
    @test rowsupport(B, 18) == 16:20
    @test rowsupport(B, 20) == 1:20
    @test colsupport(B, 1:2) == 1:20
    @test colsupport(B, 2:5) == 1:7
    @test colsupport(B, 19:20) == 1:20
    @test rowsupport(B, 1:2) == 1:20
    @test rowsupport(B, 2:5) == 1:8
    @test rowsupport(B, 19:20) == 1:20

    A = brand(20, 20, 0, 1)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test colsupport(B, 1) == 1:20
    @test colsupport(B, 2) == 1:2
    @test colsupport(B, 7) == 6:7
    @test colsupport(B, 20) == 1:20
    @test rowsupport(B, 1) == 1:20
    @test rowsupport(B, 2) == 2:3
    @test rowsupport(B, 7) == 7:8
    @test rowsupport(B, 20) == 1:20

    A = brand(20, 20, 0, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test colsupport(B, 1) == 1:20
    @test colsupport(B, 2) == 2:2
    @test colsupport(B, 7) == 7:7
    @test colsupport(B, 20) == 1:20
    @test rowsupport(B, 1) == 1:20
    @test rowsupport(B, 2) == 2:2
    @test rowsupport(B, 7) == 7:7
    @test rowsupport(B, 20) == 1:20

    A = brand(10, 10, 1, 0)
    a, b = rand(), 0.0
    B = CBM(A, a, b)
    BB = CBMs.MᵀM(B)
    BBp = CBMs.bandedpart(BB)
    @test colsupport(BB, 1) == 1:10
    @test colsupport(BB, 2) == 1:3
    @test colsupport(BB, 3) == 2:4
    @test colsupport(BB, 4) == 3:5
    @test colsupport(BB, 5) == 4:6
    @test colsupport(BB, 6) == 5:7
    @test colsupport(BB, 7) == 6:8
    @test colsupport(BB, 8) == 7:9
    @test colsupport(BB, 9) == 8:10
    @test colsupport(BB, 10) == 1:10
    @test rowsupport(BB, 1) == 1:10
    @test rowsupport(BB, 2) == 1:3
    @test rowsupport(BB, 3) == 2:4
    @test rowsupport(BB, 4) == 3:5
    @test rowsupport(BB, 5) == 4:6
    @test rowsupport(BB, 6) == 5:7
    @test rowsupport(BB, 7) == 6:8
    @test rowsupport(BB, 8) == 7:9
    @test rowsupport(BB, 9) == 8:10
    @test rowsupport(BB, 10) == 1:10
end

⊙(x, y) = replace(x -> isnan(x) ? -Inf : x, x) == replace(y -> isnan(y) ? -Inf : y, y)

@testset "convert/similar" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    @test convert(CBM{Float64}, B) === B
    B32 = convert(CBM{Float32}, B)
    @test eltype(B32) == Float32
    @test eltype(B) == Float64
    @test typeof(parent(B32)) == Matrix{Float32} && parent(B32) ⊙ convert(Matrix{Float32}, parent(B))
    @test CBMs._bpbandwidths(B32) == (2, 3)

    SB = similar(B)
    @test SB isa CBM
    @test size(SB) == size(B)
    @test eltype(SB) == eltype(B)
    @test axes(SB) == axes(B)

    SB = similar(B, Float32)
    @test SB isa CBM{Float32}
    @test size(SB) == size(B)
    @test eltype(SB) == Float32
    @test axes(SB) == axes(B)

    SB = similar(B, Float32, 5, 5)
    @test SB isa CBM{Float32}
    @test size(SB) == (5, 5)
    @test eltype(SB) == Float32
    @test axes(SB) == (1:5, 1:5)
end

@testset "getindex" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)

    @test B[band(0)] == A[band(0)]
    @test B[band(1)] == A[band(1)]
    @test B[band(2)] == A[band(2)]
    @test B[2, 3] == A[2, 3]
    @test B[1, end] == a
    @test B[end, 1] == b
    @test B[end, end+1] == a
    @test B[end+1, end] == b
    @test CBMs._isinband(2, 3, 2, 2)
    @test !CBMs._isinband(2, 3, 10, 19)
    @test CBMs._isincorner(20, 1, 20)
    @test !CBMs._isincorner(20, 1, 19)
    @test CBMs._isincorner(20, 20, 1)
    @test !CBMs._isincorner(20, 19, 1)
    @test_throws BoundsError B[18, 200]
end

@testset "setindex!" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)

    B[1, 2] = 10
    @test B[1, 2] == 10
    B[1, end] = 20
    @test B[1, end] == 20
    B[end, end+1] = 30
    @test B[end, end+1] == B[1, end] == 30
    B[end+1, end] = 40
    @test B[end+1, end] == B[end, 1] == 40
    B[3, 3] = 50
    @test B[3, 3] == 50
    B[17, 2] = 0
    @test B[17, 2] == 0
    @test_throws ArgumentError B[17, 5] = 2

    A = brand(2, 2, 0, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    B[1, 2] = 10
    @test B[1, 2] == 10
    B[2, 1] = 20
    @test B[2, 1] == 20
    B[1, 1] = 30
    @test B[1, 1] == 30
    B[2, 2] = 40

    A = brand(3, 3, 1, 0)
    a, b = rand(2)
    B = CBM(A, a, b)
    B[1, 2] = 10
    @test B[1, 2] == 10
    B[2, 1] = 20
    @test B[2, 1] == 20
    B[1, 1] = 30
    @test B[1, 1] == 30
    B[2, 2] = 40
    @test B[2, 2] == 40
    B[3, 3] = 50
    @test B[3, 3] == 50
    B[1, 3] = 60
    @test B[1, 3] == 60
    B[3, 1] = 70
    @test B[3, 1] == 70
    B[2, 3] = 80
    @test B[2, 3] == 80
    B[3, 2] = 90
    @test B[3, 2] == 90
end

@testset "copy" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)

    BC = copy(B)
    @test !(BC === B)
    @test !(BC.data === B.data)
    @test BC == B

    B2 = similar(B)
    copyto!(B2, B)
    @test B2 == B
end

@testset "show" begin
    A = brand(20, 20, 2, 3)
    a, b = rand(2)
    B = CBM(A, a, b)
    lay = MemoryLayout(B)
    @test CBMs.layout_replace_in_print_matrix(lay, B, 2, 2, string(B[2, 2])) == string(B[2, 2])
    @test CBMs.layout_replace_in_print_matrix(lay, B, 1, 1, string(B[1, 1])) == string(B[1, 1])
    @test CBMs.layout_replace_in_print_matrix(lay, B, 1, size(B, 2), string(B[1, end])) == string(B[1, end])
    @test CBMs.layout_replace_in_print_matrix(lay, B, size(B, 1), 1, string(B[end, 1])) == string(B[end, 1])
    @test CBMs.layout_replace_in_print_matrix(lay, B, 13, 1, string(B[13, 1])) == " ⋅ "
end

@testset "Diagonal Multiplication" begin
    A = brand(20, 20, 1, 2)
    B = CBM(A, randn(), randn())
    D = Diagonal(randn(20))
    BD = B * D
    @test BD isa CBM
    @test BD ≈ Matrix(B) * D
    DB = D * B
    @test DB isa CBM
    @test DB ≈ D * Matrix(B)

    A = CBM(brand(2, 2, 0, 0), randn(2)...)
    D = Diagonal(randn(2))
    AD = A * D
    @test AD isa CBM
    @test AD ≈ Matrix(A) * D
    DA = D * A
    @test DA isa CBM
    @test DA ≈ D * Matrix(A)

    A = CBM(brand(2, 2, 0, 1), randn(2)...)
    D = Diagonal(randn(2))
    AD = A * D
    @test AD isa CBM
    @test AD ≈ Matrix(A) * D
    DA = D * A
    @test DA isa CBM
    @test DA ≈ D * Matrix(A)

    A = CBM(brand(2, 2, 1, 0), randn(2)...)
    D = Diagonal(randn(2))
    AD = A * D
    @test AD isa CBM
    @test AD ≈ Matrix(A) * D
    DA = D * A
    @test DA isa CBM
    @test DA ≈ D * Matrix(A)

    A = CBM(brand(2, 2, 1, 1), randn(2)...)
    D = Diagonal(randn(2))
    AD = A * D
    @test AD isa CBM
    @test AD ≈ Matrix(A) * D
    DA = D * A
    @test DA isa CBM
    @test DA ≈ D * Matrix(A)
end

function rand_bspd(n, u)
    L = brand(n, n, u, 0)
    B = BandedMatrix(Symmetric(L * L')) + n * I
    return B
end
function rand_spd_cbm(n, u)
    B = rand_bspd(n, u)
    a = randn()
    return CBM(B, a, a)
end

@testset "Cholesky" begin
    function test_chol(n, u)
        A = rand_spd_cbm(n, u)
        AA = copy(A)
        chol = cholesky!(Symmetric(A))
        L, U = chol
        @test L * U ≈ AA
        @test chol.factors isa AlmostBandedMatrix
        chol = cholesky(Symmetric(AA))
        L, U = chol
        @test L * U ≈ AA
        @test chol.factors isa AlmostBandedMatrix
        _chol = cholesky(Matrix(AA))
        _L, _U = _chol
        @test _L * _U ≈ AA
        @test _L ≈ L
        @test _U ≈ U
    end
    test_chol(3, 1)
    test_chol(4, 2)
    test_chol(5, 2)
    test_chol(6, 3)
    test_chol(10, 4)
    test_chol(100, 2)
    test_chol(50, 1)
    test_chol(2000, 5)
end

@testset "Reverse Cholesky" begin
    function test_revchol(n, u)
        A = rand_spd_cbm(n, u)
        AA = copy(A)
        chol = reversecholesky!(Symmetric(A))
        U, L = chol
        @test ApplyMatrix(*, U, L) ≈ AA # U * L is an infinite loop or just extremely slow..
        @test chol.factors isa AlmostBandedMatrix
        chol = reversecholesky(Symmetric(AA))
        U, L = chol
        @test ApplyMatrix(*, U, L) ≈ AA
        @test chol.factors isa AlmostBandedMatrix
        _chol = reversecholesky(Matrix(AA))
        _U, _L = _chol
        @test ApplyMatrix(*, _U, _L) ≈ AA
        @test _L ≈ L
        @test _U ≈ U
    end
    test_revchol(3, 1)
    test_revchol(4, 2)
    test_revchol(5, 2)
    test_revchol(6, 3)
    test_revchol(10, 4)
    test_revchol(100, 2)
    test_revchol(50, 1)
    test_revchol(287, 5)

    n = 3
    A = CBM(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n - 1), -1 => randn(n - 1)), 0.0, 0.0)
    S = Symmetric(A)
    U, L = reversecholesky(S)
    @test U * L ≈ Matrix(S)
end

@testset "Definition" begin
    for _ in 1:1000
        n = 100
        ℓ, u = rand(-3:3), rand(-3:3)
        B = brand(n, ℓ, u)
        a, b = randn(2)
        A = CBM(B, a, b)
        for i in 1:n
            for j in 1:n
                bij = B[i, j]
                if (ℓ < i - j) || (j - i > u)
                    @test iszero(B[i, j])
                end
                if (ℓ < i - j) || (j - i > u)
                    if !(n - i + j ≤ 1) && !(n + i - j ≤ 1)
                        @test iszero(A[i, j])
                    end
                end
            end
        end
    end
end

@testset "Product" begin
    for _ in 1:1000
        n = 100
        ℓ1, ℓ2 = rand(-3:3, 2)
        u1, u2 = rand(-3:3, 2)
        B1 = brand(n, ℓ1, u1)
        B2 = brand(n, ℓ2, u2)
        A = CBM(B1, randn(2)...) * CBM(B2, randn(2)...)
        w1 = max(u1, u2) + 1
        w2 = max(ℓ1, ℓ2) + 1
        ℓ = max(0, ℓ1 + ℓ2)
        u = max(0, u1 + u2)
        for i in 1:n
            for j in 1:n
                if (ℓ < i - j) || (j - i > u)
                    if !(n - i + j ≤ w1) && !(n + i - j ≤ w2)
                        @test iszero(A[i, j])
                    end
                end
            end
        end
    end
end

@testset "Broadcast" begin
    for _ in 1:100
        ℓ, u = rand(-3:3, 2)
        B1 = brand(100, ℓ, u)
        B2 = brand(100, ℓ, u)
        A, B = CBM(B1, randn(2)...), CBM(B2, randn(2)...)
        @test Base.BroadcastStyle(typeof(A)) == CBMs.CyclicBandedMatrixStyle() == CBMs.CyclicBandedMatrixStyle(Val(2))
        @test Base.BroadcastStyle(CBMs.CyclicBandedMatrixStyle(), Base.Broadcast.DefaultArrayStyle(Val(2))) == Base.Broadcast.DefaultArrayStyle(Val(2))
        @test Base.BroadcastStyle(Base.Broadcast.DefaultArrayStyle(Val(2)), CBMs.CyclicBandedMatrixStyle()) == Base.Broadcast.DefaultArrayStyle(Val(2))
        @test -A == -Matrix(A) && -A isa CBM
        @test 2A == 2Matrix(A) && 2A isa CBM
        @test A + B == Matrix(A) + Matrix(B) && A + B isa CBM
        @test A - B == Matrix(A) - Matrix(B) && A - B isa CBM
        @test A * 2 == Matrix(A) * 2 && A * 2 isa CBM
        @test A ./ 2 == Matrix(A) ./ 2 && A ./ 2 isa CBM

        bc = Base.broadcasted(/, A, rand(size(A, 1)))
        dest = similar(bc, Float64)
        @test size(dest) == size(A) && dest isa CBM && dest !== A && CBMs._bpbandwidths(dest) == CBMs._bpbandwidths(A)
        bc = Base.broadcasted((x, y) -> 1.0, A, rand(size(A, 1)))
        dest = similar(bc, Float64)
        @test size(dest) == size(A) && dest isa CBM && dest !== A && CBMs._bpbandwidths(dest) == (100, 100) .- (1, 1)

        x = rand(100)
        @test A .* x == Matrix(A) .* x && A .* x isa CBM
        @test A ./ x == Matrix(A) ./ x && A ./ x isa CBM
    end

    B = brand(10, 10, 0, 0)
    A = CBM(B, rand(2)...)
    x = rand(10)
    C = A .* x
    @test Matrix(C) == Matrix(A) .* x
    @test C isa CBM && CBMs._bpbandwidths(C) == (0, 0)

    M1 = brand(10, 10, 0, 1)
    M2 = brand(10, 10, 0, 1)
    A = brand(10, 10, 1, 1)
    CM1 = CBM(M1, rand(), 0.0)
    CM2 = CBM(M2, rand(), 0.0)
    CA = CBM(A, rand(2)...)
    @test CA - (CM1 + CM2) ≈ Matrix(CA) - (Matrix(CM1) + Matrix(CM2))
    E = CA - (CM1 + CM2)
    @test E isa CBM && CBMs._bpbandwidths(E) == (1, 1)

    A = CBM(brand(10, 10, 0, 1), rand(2)...)
    B = A'
    C = 2B
    @test C isa CBM && C ≈ 2Matrix(A)' && CBMs._bpbandwidths(C) == (1, 0)
end

@testset "Outer" begin
    B = CBM(brand(10, 10, 0, 0), rand(2)...)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)
    B = CBM(brand(2, 2, 0, 0), rand(2)...)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)

    B = CBM(brand(10, 10, 1, 0), rand(2)...)
    @test_throws AssertionError CBMs.MMᵀ(B)
    B = CBM(brand(10, 10, 1, 0), rand(), 0.0)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
    B = CBM(brand(2, 2, 1, 0), rand(2)...)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)

    B = CBM(brand(10, 10, 0, 1), rand(2)...)
    @test_throws AssertionError CBMs.MMᵀ(B)
    B = CBM(brand(10, 10, 0, 1), 0.0, rand())
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
    B = CBM(brand(2, 2, 0, 1), rand(2)...)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)

    B = CBM(brand(10, 10, 1, 1), rand(2)...)
    @test_throws ArgumentError CBMs.MMᵀ(B)

    B = CBM(brand(3, 3, 1, 0), rand(), 0.0)
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)

    B = CBM(brand(3, 3, 0, 1), 0.0, rand())
    BB = CBMs.MMᵀ(B)
    @test BB ≈ Matrix(B) * Matrix(B)'
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
end

@testset "MᵀM" begin
    B = CBM(brand(10, 10, 0, 0), rand(2)...)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)
    B = CBM(brand(2, 2, 0, 0), rand(2)...)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)

    B = CBM(brand(10, 10, 1, 0), rand(2)...)
    @test_throws AssertionError CBMs.MᵀM(B)
    B = CBM(brand(10, 10, 1, 0), rand(), 0.0)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
    B = CBM(brand(2, 2, 1, 0), rand(2)...)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (0, 0)

    B = CBM(brand(10, 10, 0, 1), rand(2)...)
    @test_throws AssertionError CBMs.MᵀM(B)
    B = CBM(brand(10, 10, 0, 1), 0.0, rand())
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
    B = CBM(brand(2, 2, 0, 1), rand(2)...)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)

    B = CBM(brand(10, 10, 1, 1), rand(2)...)
    @test_throws ArgumentError CBMs.MᵀM(B)

    B = CBM(brand(3, 3, 1, 0), rand(), 0.0)
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)

    B = CBM(brand(3, 3, 0, 1), 0.0, rand())
    BB = CBMs.MᵀM(B)
    @test BB ≈ Matrix(B)' * Matrix(B)
    @test BB isa CBM && CBMs._bpbandwidths(BB) == (1, 1)
end

@testset "diagonal_rinv!" begin
    A = CBM(brand(10, 10, 1, 1), rand(2)...)
    B = Diagonal(rand(10))
    C = A * inv(B)
    CBMs.diagonal_rinv!(A, B)
    @test A ≈ C
    expr = @allocations CBMs.diagonal_rinv!(A, B)
    @test iszero(expr)
    @inferred CBMs.diagonal_rinv!(A, B)

    B = Diagonal(rand(9))
    @test_throws DimensionMismatch CBMs.diagonal_rinv!(A, B)

    B = Diagonal(rand(10))
    B[2, 2] = 0
    @test_throws SingularException(2) CBMs.diagonal_rinv!(A, B)

    A = CBM(brand(2, 2, 0, 1), rand(2)...)
    B = Diagonal(rand(2))
    C = A * inv(B)
    CBMs.diagonal_rinv!(A, B)
    @test A ≈ C

    A = CBM(brand(2, 2, 1, 0), rand(2)...)
    B = Diagonal(rand(2))
    C = A * inv(B)
    CBMs.diagonal_rinv!(A, B)
    @test A ≈ C

    A = CBM(brand(2, 2, 0, 0), rand(2)...)
    B = Diagonal(rand(2))
    C = A * inv(B)
    CBMs.diagonal_rinv!(A, B)
    @test A ≈ C
end

@testset "diagonal_subrmul!" begin
    A = CBM(brand(10, 10, 1, 1), rand(2)...)
    D = Diagonal(rand(10))
    M = CBM(brand(10, 10, 1, 1), rand(2)...)
    C = A - M * D
    CBMs.diagonal_subrmul!(A, D, M)
    @test A ≈ C
    expr = @allocations CBMs.diagonal_subrmul!(A, D, M)
    @test iszero(expr)
    @inferred CBMs.diagonal_subrmul!(A, D, M)

    D = Diagonal(rand(9))
    @test_throws DimensionMismatch CBMs.diagonal_subrmul!(A, D, M)
    D = Diagonal(rand(10))
    M = CBM(brand(9, 9, 1, 1), rand(2)...)
    @test_throws DimensionMismatch CBMs.diagonal_subrmul!(A, D, M)

    A = CBM(brand(2, 2, 0, 1), rand(2)...)
    D = Diagonal(rand(2))
    M = CBM(brand(2, 2, 0, 1), rand(2)...)
    C = A - M * D
    CBMs.diagonal_subrmul!(A, D, M)
    @test A ≈ C

    A = CBM(brand(2, 2, 1, 0), rand(2)...)
    D = Diagonal(rand(2))
    M = CBM(brand(2, 2, 1, 0), rand(2)...)
    C = A - M * D
    CBMs.diagonal_subrmul!(A, D, M)
    @test A ≈ C

    A = CBM(brand(2, 2, 0, 0), rand(2)...)
    D = Diagonal(rand(2))
    M = CBM(brand(2, 2, 0, 0), rand(2)...)
    C = A - M * D
    CBMs.diagonal_subrmul!(A, D, M)
    @test A ≈ C
end

@testset "mat_sub_MMᵀ!" begin
    A = CBM(brand(10, 10, 0, 0), randn(2)...)
    M = CBM(brand(10, 10, 0, 0), randn(2)...)
    C = A - M * M'
    CBMs.mat_sub_MMᵀ!(A, M)
    @test A ≈ C
    expr = @allocations CBMs.mat_sub_MMᵀ!(A, M)
    @test iszero(expr)
    @inferred CBMs.mat_sub_MMᵀ!(A, M)

    A = CBM(brand(10, 10, 1, 1), randn(2)...)
    M = CBM(brand(10, 10, 1, 0), randn(2)...)
    @test_throws AssertionError CBMs.mat_sub_MMᵀ!(A, M)
    M = CBM(brand(10, 10, 0, 1), 0.0, randn())
    C = A - M * M'
    CBMs.mat_sub_MMᵀ!(A, M)
    @test A ≈ C

    A = CBM(brand(10, 10, 1, 1), randn(2)...)
    M = CBM(brand(10, 10, 1, 0), rand(), 0.0)
    C = A - M * M'
    CBMs.mat_sub_MMᵀ!(A, M)
    @test A ≈ C

    A = CBM(brand(2, 2, 0, 0), randn(2)...)
    M = CBM(brand(2, 2, 0, 0), randn(2)...)
    C = A - M * M'
    CBMs.mat_sub_MMᵀ!(A, M)
    @test A ≈ C
end