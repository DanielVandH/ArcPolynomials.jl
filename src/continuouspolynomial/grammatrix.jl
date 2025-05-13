function grammatrix_layout(_, P::PeriodicContinuousPolynomial{0,T,G}) where {T,G}
    r = P.points
    nel = length(r) - 1
    hs = diff(r)
    return CyclicBBBArrowheadMatrix{T}(Diagonal(hs), (), (), [Diagonal(hs[i] ./ (3:2:∞)) for i in 1:nel])
end # hs[i] / (3:2:∞) is L * hs/2, where L = 2 / (2n+1) with n = 1:∞ is (Legendre()'Legendre()).diag[2:end]

function grammatrix_layout(_, P::PeriodicContinuousPolynomial{1,T,G}) where {T,G}
    L = Legendre{T}()
    U = Ultraspherical{T}(-one(T) / 2)
    R = L \ U

    r = P.points
    nel = length(r) - 1
    hs = diff(r)
    shs = sqrt.(hs)

    R11 = CyclicBandedMatrix(_BandedMatrix(Hcat(vcat(zero(T), shs[begin:end-1]), shs)', oneto(nel), 0, 1), nel == 2 ? shs[1] : zero(T), shs[end]) / 2
    R21 = CyclicBandedMatrix(_BandedMatrix(Hcat(vcat(zero(T), shs[begin:end-1]), -shs)', oneto(nel), 0, 1), nel == 2 ? shs[1] : zero(T), shs[end]) / (2sqrt(3))

    A = MᵀM(R11) + MᵀM(R21)
    C = (R11 .* (shs ./ 3), R21 .* (shs ./ 5sqrt(3)))
    B = (C[1]', C[2]')

    n = 0:∞
    Ls = 1 ./ (2 .* n .+ 1)
    D = [
        _BandedMatrix(
            Hcat(
                Vcat(zero(T), zero(T), R[band(2)][3:end] .* Ls[2:end] .* R[band(0)][4:end] .* hs[i]),
                Zeros{T}(∞),
                (R[band(2)] .^ 2 .* Ls .+ R[band(0)][3:end] .^ 2 .* Ls[3:end]) .* hs[i],
                Zeros{T}(∞),
                R[band(2)][3:end] .* Ls[2:end] .* R[band(0)][4:end] .* hs[i]
            )',
            axes(R, 1), 2, 2
        ) for i in 1:nel
    ]

    return CyclicBBBArrowheadMatrix(A, B, C, D)
end