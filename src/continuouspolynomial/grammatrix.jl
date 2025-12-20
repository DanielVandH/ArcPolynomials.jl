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

    R11 = CyclicBandedMatrix(_BandedMatrix(hcat(vcat(zero(T), @view(shs[begin:end-1]) / 2), shs / 2)', oneto(nel), 0, 1), nel == 2 ? shs[1] / 2 : zero(T), shs[end] / 2)
    R21 = CyclicBandedMatrix(_BandedMatrix(hcat(vcat(zero(T), @view(shs[begin:end-1]) / (2sqrt(3))), -shs / (2sqrt(3)))', oneto(nel), 0, 1), nel == 2 ? shs[1] / (2sqrt(3)) : zero(T), shs[end] / (2sqrt(3)))

    A = MᵀM(R11) + MᵀM(R21)
    C = (R11 .* (shs ./ 3), R21 .* (shs ./ 5sqrt(3)))
    B = (C[1]', C[2]')

    n = 0:∞
    Ls = 1 ./ (2 .* n .+ 1)
    Rband2 = @view(R.data.args[1][3:end])
    Rband0 = R.data.args[3]'
    _rb = @views Rband2[3:end] .* Ls[2:end] .* Rband0[4:end]
    _rc = (Rband2 .^ 2 .* Ls .+ Rband0[3:end] .^ 2 .* Ls[3:end])
    D = let _rb = _rb, _rc = _rc, R = R, T = T
        map(1:nel) do i
            rb = _rb .* hs[i]
            rc = _rc .* hs[i]
            _BandedMatrix(
                Hcat(
                    Vcat(zero(T), zero(T), rb),
                    Zeros{T}(∞),
                    rc,
                    Zeros{T}(∞),
                    rb
                )',
                axes(R, 1), 2, 2
            )
        end
    end

    return CyclicBBBArrowheadMatrix(A, B, C, D)
end