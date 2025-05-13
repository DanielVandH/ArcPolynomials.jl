function grammatrix_layout(_, P::PiecewiseArcPolynomial{0,T,G}) where {T,G}
    r = P.points
    nel = length(r) - 1
    ps = diff(r)
    qs = (ps .- sin.(ps)) ./ 2
    return CyclicBBBArrowheadMatrix{T}(Diagonal(ps), (), (), [Diagonal(InterlacedVector(Fill(qs[i], ∞), Fill(ps[i], ∞))) for i in 1:nel])
end

function grammatrix_layout(_, P::PiecewiseArcPolynomial{1,T,G}) where {T,G}
    # Some repetition here from P0 \ P but it's OK
    hs = diff(P.points)
    shs = sqrt.(hs)
    qs = (hs .- sin.(hs)) ./ 2
    isqs = sqrt.(qs)
    sqs = (inv).(isqs)
    nel = length(hs)
    if has_equal_spacing(P)
        Ψ11 = P.P0.Q \ P.P.Q
        Ψ22 = P.P0.P \ P.P.P
    else
        Ψ11 = [get_P0(P, i).Q \ get_P(P, i).Q for i in 1:length(P.points)-1]
        Ψ22 = [get_P0(P, i).P \ get_P(P, i).P for i in 1:length(P.points)-1]
    end

    # Compute ||p||⁻¹P₀₂ᵀH
    _R00 = BandedMatrix{T}(0 => shs ./ 2, 1 => @view(shs[begin:end-1]) ./ 2)
    R00 = CyclicBandedMatrix(_R00, nel == 2 ? shs[1] / 2 : zero(T), shs[end] / 2)
    R00_sq = R00 .* shs

    # Compute ||q||⁻¹P₁₁ᵀH 
    h = diff(P.points)
    _R10_diag = @. (sin(h) - h) / (4sin(h / 2)) .* sqs
    _R10_supdiag = -_R10_diag
    circshift!(_R10_supdiag, 1) # [1, 2, ..., n] => [2, 3, ..., n, 1]
    _R10 = _BandedMatrix(vcat(_R10_supdiag', _R10_diag'), oneto(nel), 0, 1)
    R10 = CyclicBandedMatrix(_R10, nel == 2 ? _R10_supdiag[2] : zero(T), _R10_supdiag[1])
    R10_sq = R10 .* isqs

    # Compute A₀₁
    if has_equal_spacing(P)
        A01 = Fill(Ψ22[1, 2], nel)
    else
        A01 = [Ψ22[i][1, 2] for i in 1:nel]
    end

    # Compute B₀₁
    if has_equal_spacing(P)
        B01 = Fill(Ψ11[1, 2], nel)
    else
        B01 = [Ψ11[i][1, 2] for i in 1:nel]
    end

    # Compute A, B, C
    A = MᵀM(R00) + MᵀM(R10)
    C = (R00_sq .* A01, R10_sq .* B01)
    B = (C[1]', C[2]')

    # Compute D 
    if has_equal_spacing(P)
        Ddata = Hcat(
            InterlacedVector(Ψ22[band(1)] .* Ψ22[band(0)] * hs[1], Ψ11[band(1)] .* Ψ11[band(0)] * qs[1]), # FIXME: Convert to F type
            Zeros{T}(∞),
            InterlacedVector((Ψ22[band(1)] .^ 2 .+ Ψ22[band(0)][2:end] .^ 2) .* hs[1], (Ψ11[band(1)] .^ 2 .+ Ψ11[band(0)][2:end] .^ 2) .* qs[1]), # FIXME: Convert to F type
            Zeros{T}(∞),
            InterlacedVector(Ψ22[band(1)][2:end] .* Ψ22[band(0)][2:end] * hs[1], Ψ11[band(1)][2:end] .* Ψ11[band(0)][2:end] * qs[1])
        )
        D = [_BandedMatrix(Ddata', axes(Ψ11, 1), 2, 2) for _ in 1:nel]
    else
        D = [
            _BandedMatrix(
                Hcat(
                    InterlacedVector(Ψ22[i][band(1)] .* Ψ22[i][band(0)] * hs[i], Ψ11[i][band(1)] .* Ψ11[i][band(0)] * qs[i]), # FIXME: Convert to F type
                    Zeros{T}(∞),
                    InterlacedVector((Ψ22[i][band(1)] .^ 2 .+ Ψ22[i][band(0)][2:end] .^ 2) .* hs[i], (Ψ11[i][band(1)] .^ 2 .+ Ψ11[i][band(0)][2:end] .^ 2) .* qs[i]), # FIXME: Convert to F type
                    Zeros{T}(∞),
                    InterlacedVector(Ψ22[i][band(1)][2:end] .* Ψ22[i][band(0)][2:end] * hs[i], Ψ11[i][band(1)][2:end] .* Ψ11[i][band(0)][2:end] * qs[i])
                )',
                axes(Ψ11[i], 1), 2, 2
            ) for i in 1:nel
        ]
    end
    M = CyclicBBBArrowheadMatrix(A, B, C, D)
    return M
end