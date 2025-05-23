function diff(P::PiecewiseArcPolynomial{1,T}; dims=1) where {T}
    nel = length(P.points) - 1
    if has_equal_spacing(P)
        D21 = semiclassicaljacobiarc_diffp(P.P)
        D12 = semiclassicaljacobiarc_diffq(P.P)
    else
        D21 = [semiclassicaljacobiarc_diffp(get_P(P, i)) for i in 1:nel]
        D12 = [semiclassicaljacobiarc_diffq(get_P(P, i)) for i in 1:nel]
    end

    hs = diff(P.points)
    invh = inv.(hs)
    _D00 = BandedMatrix{T}(0 => -invh, 1 => @view(invh[begin:end-1]))
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? one(T) * invh[1] : zero(T), one(T) * invh[end])

    if has_equal_spacing(P)
        h = P.P0.h
        α, β = _linear_coefficients(P.P0.P.t, P.P0.P.a, P.P0.P.b, P.P0.P.c)
        ζ = csc(hs[1]) * h * β * (sqrt(1 - h^2) * (2α - 1) - (sqrt(1 - h^2) - acos(h)) / (1 - h))
        M = 2sum(orthogonalityweight(P.P0.P)) # grammatrix(P0)[nel+1,nel+1]
        _D20_diag = fill(ζ / M, nel)
        _D20_supdiag = -_D20_diag
    else
        _D20_diag = map(1:nel) do i
            _P0 = get_P0(P, i)
            _h = _P0.h
            _α, _β = _linear_coefficients(_P0.P.t, _P0.P.a, _P0.P.b, _P0.P.c)
            _ζ = csc(hs[i]) * _h * _β * (sqrt(1 - _h^2) * (2_α - 1) - (sqrt(1 - _h^2) - acos(_h)) / (1 - _h))
            _M = 2sum(orthogonalityweight(_P0.P)) # grammatrix(P0)[nel+1,nel+1]
            return _ζ / _M
        end
        _D20_supdiag = -_D20_diag
        circshift!(_D20_supdiag, 1) # [1, 2, ..., n] => [2, 3, ..., n, 1]
    end
    _D20 = _BandedMatrix(Vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])

    A = D00
    B = ()
    C = (Zeros{T}(nel, nel), D20)
    if has_equal_spacing(P)
        Ddata = Hcat(
            Vcat(zero(T), Vcat(InterlacedVector(Zeros{T}(∞), D21[band(2)]))),
            Zeros{T}(∞),
            InterlacedVector(D21[band(1)], D12[band(0)][2:end]),
            Zeros{T}(∞),
            InterlacedVector(Zeros{T}(∞), D12[band(-1)][2:end])
        )
        _Ddata = _BandedMatrix(Ddata', axes(D12, 1), 2, 2)
        D = [_Ddata for _ in invh]
    else
        D = [
            _BandedMatrix(
                Hcat(
                    Vcat(zero(T), Vcat(InterlacedVector(Zeros{T}(∞), D21[i][band(2)]))),
                    Zeros{T}(∞),
                    InterlacedVector(D21[i][band(1)], D12[i][band(0)][2:end]),
                    Zeros{T}(∞),
                    InterlacedVector(Zeros{T}(∞), D12[i][band(-1)][2:end])
                )',
                axes(D12[i], 1), 2, 2
            ) for i in 1:nel
        ]
    end
    DQP = CyclicBBBArrowheadMatrix(A, B, C, D)

    P0 = PiecewiseArcPolynomial{0}(P)
    return P0 * DQP
end

function weaklaplacian_layout(_, P::PiecewiseArcPolynomial{1,T}) where {T}
    nel = length(P.points) - 1
    if has_equal_spacing(P)
        D21 = semiclassicaljacobiarc_diffp(P.P)
        D12 = semiclassicaljacobiarc_diffq(P.P)
    else
        D21 = [semiclassicaljacobiarc_diffp(get_P(P, i)) for i in 1:nel]
        D12 = [semiclassicaljacobiarc_diffq(get_P(P, i)) for i in 1:nel]
    end

    ## Compute ||p₀||⁻¹P₀₂ᵀH′. Some repetition here from diff() but it's OK
    hs = diff(P.points)
    invh = (sqrt ∘ inv).(hs)
    _D00 = BandedMatrix{T}(0 => -invh, 1 => @view(invh[begin:end-1]))
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? one(T) * invh[1] : zero(T), one(T) * invh[end])

    ## Compute ||p₀||⁻¹P₁₂ᵀH′
    if has_equal_spacing(P)
        h = P.P0.h
        α, β = _linear_coefficients(P.P0.P.t, P.P0.P.a, P.P0.P.b, P.P0.P.c)
        ζ = csc(hs[1]) * h * β * (sqrt(1 - h^2) * (2α - 1) - (sqrt(1 - h^2) - acos(h)) / (1 - h))
        _D20_diag = fill(ζ, nel)
        _D20_supdiag = -_D20_diag
    else
        _D20_diag = map(1:nel) do i
            _P0 = get_P0(P, i)
            _h = _P0.h
            _α, _β = _linear_coefficients(_P0.P.t, _P0.P.a, _P0.P.b, _P0.P.c)
            _ζ = csc(hs[i]) * _h * _β * (sqrt(1 - _h^2) * (2_α - 1) - (sqrt(1 - _h^2) - acos(_h)) / (1 - _h))
            return _ζ
        end
        _D20_supdiag = -_D20_diag
        circshift!(_D20_supdiag, 1) # [1, 2, ..., n] => [2, 3, ..., n, 1]
    end
    _D20 = _BandedMatrix(Vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])
    D20_sq = D20 .* invh

    ## Compute D
    qs = (hs .- sin.(hs)) ./ 2
    if has_equal_spacing(P)
        Dlace22 = Diagonal(fill(D12[2, 2], nel))
        D = Hcat(
            Vcat(zero(T), zero(T), InterlacedVector(-D21[band(2)] .* qs[1] .* D21[band(1)], -D12[band(-1)][2:end] .* hs[1] .* D12[band(0)][3:end])),
            Zeros{T}(∞),
            Vcat(-D21[band(1)][1]^2 * qs[1], InterlacedVector(-(D12[band(0)][2:end] .^ 2 .+ D12[band(-1)][2:end] .^ 2) .* hs[1], -(D21[band(1)][2:end] .^ 2 .+ D21[band(2)] .^ 2) .* qs[1])),
            Zeros{T}(∞),
            InterlacedVector(-D21[band(2)] .* qs[1] .* D21[band(1)], -D12[band(-1)][2:end] .* hs[1] .* D12[band(0)][3:end]),
        )
        _Ddata = _BandedMatrix(D', axes(D12, 1), 2, 2)
        D = [_Ddata for _ in invh]
    else
        Dlace22 = Diagonal([D12[i][2, 2] for i in 1:nel])
        D = [
            _BandedMatrix(
                Hcat(
                    Vcat(zero(T), zero(T), InterlacedVector(-D21[i][band(2)] .* qs[i] .* D21[i][band(1)], -D12[i][band(-1)][2:end] .* hs[i] .* D12[i][band(0)][3:end])),
                    Zeros{T}(∞),
                    Vcat(-D21[i][band(1)][1]^2 * qs[i], InterlacedVector(-(D12[i][band(0)][2:end] .^ 2 .+ D12[i][band(-1)][2:end] .^ 2) .* hs[i], -(D21[i][band(1)][2:end] .^ 2 .+ D21[i][band(2)] .^ 2) .* qs[i])),
                    Zeros{T}(∞),
                    InterlacedVector(-D21[i][band(2)] .* qs[i] .* D21[i][band(1)], -D12[i][band(-1)][2:end] .* hs[i] .* D12[i][band(0)][3:end]),
                )',
                axes(D12[i], 1), 2, 2
            ) for i in 1:nel
        ]
    end

    ## Now compute A, B
    A = -(MᵀM(D00) + MᵀM(D20_sq))
    B = (Zeros{T}(nel, nel), (-D20 .* Dlace22.diag)')
    C = (B[1], B[2]')
    Δ = CyclicBBBArrowheadMatrix(A, B, C, D)
    return Δ
end

