function _diff_compute_Ddata_equal(_D21::A, _D12::B) where {A,B}
    T = promote_type(eltype(_D21), eltype(_D12))
    _D21_band2 = view(arguments(bandeddata(_D21))[1], 3:∞)
    _D21_band1 = view(arguments(bandeddata(_D21))[2], 2:∞)
    _D12_bandn1 = _D12.ev
    _D12_band0 = _D12.dv
    Ddata = Hcat(
        Vcat(zero(T), Vcat(InterlacedVector(Zeros{T}(∞), _D21_band2))),
        Zeros{T}(∞),
        InterlacedVector(_D21_band1, view(_D12_band0, 2:∞)),
        Zeros{T}(∞),
        InterlacedVector(Zeros{T}(∞), view(_D12_bandn1, 2:∞))
    )
    _Ddata = _BandedMatrix(Ddata', axes(_D12, 1), 2, 2)
    return _Ddata
end

function _diff_compute_D20_diag_equal(P::PP, h, Δh) where {PP}
    α, β = _linear_coefficients(P.t, P.a, P.b, P.c)
    ζ = csc(Δh) * h * β * (sqrt(1 - h^2) * (2α - 1) - (sqrt(1 - h^2) - acos(h)) / (1 - h))
    M = 2sum(orthogonalityweight(P)) # grammatrix(P0)[nel+1,nel+1]
    _D20_diag = ζ / M
    return _D20_diag
end

function _diff_compute_D20_equal(P::PP, nel::Int, Δh, h) where {PP}
    T = eltype(P)
    _D20_diag = Fill(_diff_compute_D20_diag_equal(P, h, Δh), nel)
    _D20_supdiag = -_D20_diag
    _D20 = _BandedMatrix(vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])
    return D20
end

function _diff_compute_D20_from_diag(_D20_diag::Vector{T}, nel) where {T}
    _D20_supdiag = -_D20_diag
    circshift!(_D20_supdiag, 1)
    _D20 = _BandedMatrix(vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])
    return D20
end

function _diff_compute_D00_equal(P::PP, nel) where {PP}
    T = eltype(P)
    Δh = P.points[begin+1] - P.points[begin]
    iΔh = inv(Δh)
    _D00 = _BandedMatrix(hcat(Fill(iΔh, nel), Fill(-iΔh, nel))', oneto(nel), 0, 1)
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? iΔh : zero(T), iΔh)
    return D00, Δh
end

function _diff_compute_D00_unequal(P::PP, nel) where {PP}
    T = eltype(P)
    hs = diff(P.points)
    invh = inv.(hs)
    _D00 = _BandedMatrix([vcat(zero(T), @view(invh[begin:end-1]));; -invh]', oneto(nel), 0, 1)
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? one(T) * invh[1] : zero(T), one(T) * invh[end])
    return D00, hs
end

function _diff_equal(P::PiecewiseArcPolynomial{1,T}) where {T}
    nel = length(P.points) - 1
    D00, Δh = _diff_compute_D00_equal(P, nel)
    
    # Compute D21 and D12 once
    _D21 = semiclassicaljacobiarc_diffp(P.P)
    _D12 = semiclassicaljacobiarc_diffq(P.P)
    
    # Use them for both D20 and Ddata
    D20 = _diff_compute_D20_equal(P.P0.P, nel, Δh, P.P0.h)
    Ddata = _diff_compute_Ddata_equal(_D21, _D12)
    
    A = D00
    B = ()
    C = (Zeros{T}(nel, nel), D20)
    return PiecewiseArcPolynomial{0}(P) * CyclicBBBArrowheadMatrix(A, B, C, Fill(Ddata, nel))
end

function _diff_unequal(P::PiecewiseArcPolynomial{1,T}) where {T}
    nel = length(P.points) - 1
    D00, hs = _diff_compute_D00_unequal(P, nel)
    
    # Compute D20 diagonal values and D data together
    _D20_diag = Vector{T}(undef, nel)
    D = Vector{BandedMatrix{T}}(undef, nel)
    
    for i in 1:nel
        # Compute semiclassical functions once per element
        Pi = get_P(P, i)
        _D21 = semiclassicaljacobiarc_diffp(Pi)
        _D12 = semiclassicaljacobiarc_diffq(Pi)
        
        # Use for D20
        _P0 = get_P0(P, i)
        _D20_diag[i] = _diff_compute_D20_diag_equal(_P0.P, _P0.h, hs[i])
        
        # Use for Ddata
        D[i] = _diff_compute_Ddata_equal(_D21, _D12)
    end
    D = map(identity, D) # concrete type 

    D20 = _diff_compute_D20_from_diag(_D20_diag, nel)
    
    A = D00
    B = ()
    C = (Zeros{T}(nel, nel), D20)
    return PiecewiseArcPolynomial{0}(P) * CyclicBBBArrowheadMatrix(A, B, C, D)
end

function diff(P::PiecewiseArcPolynomial{1,T}; dims=1) where {T}
    has_equal_spacing(P) ? _diff_equal(P) : _diff_unequal(P)
end

function _weaklaplacian_compute_D00_equal(P::PiecewiseArcPolynomial{1,T}, nel) where {T}
    Δh = P.points[begin+1] - P.points[begin]
    iΔh_sqrt = sqrt(inv(Δh))
    _D00 = _BandedMatrix(hcat(Fill(iΔh_sqrt, nel), Fill(-iΔh_sqrt, nel))', oneto(nel), 0, 1)
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? iΔh_sqrt : zero(T), iΔh_sqrt)
    return D00, Δh
end

function _weaklaplacian_compute_D00_unequal(P::PiecewiseArcPolynomial{1,T}, nel) where {T}
    hs = diff(P.points)
    invh = (sqrt ∘ inv).(hs)
    _D00 = _BandedMatrix([vcat(zero(T), @view(invh[begin:end-1]));; -invh]', oneto(nel), 0, 1)
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? one(T) * invh[1] : zero(T), one(T) * invh[end])
    return D00, hs, invh
end

function _weaklaplacian_compute_D20_diag_equal(P::PP, h, Δh) where {PP}
    α, β = _linear_coefficients(P.t, P.a, P.b, P.c)
    ζ = csc(Δh) * h * β * (sqrt(1 - h^2) * (2α - 1) - (sqrt(1 - h^2) - acos(h)) / (1 - h))
    return ζ
end

function _weaklaplacian_compute_D20_equal(P::PP, nel, Δh, h) where {PP}
    T = eltype(P)
    _D20_diag = Fill(_weaklaplacian_compute_D20_diag_equal(P, h, Δh), nel)
    _D20_supdiag = -_D20_diag
    _D20 = _BandedMatrix(vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])
    return D20
end

function _weaklaplacian_compute_D20_from_diag(_D20_diag::Vector{T}, nel) where {T}
    _D20_supdiag = -_D20_diag
    circshift!(_D20_supdiag, 1)
    _D20 = _BandedMatrix(vcat(_D20_supdiag', _D20_diag'), oneto(nel), 0, 1)
    D20 = CyclicBandedMatrix(_D20, nel == 2 ? _D20_supdiag[2] : zero(T), _D20_supdiag[1])
    return D20
end

function _weaklaplacian_compute_Ddata_equal(_D21::A, _D12::B, q, h) where {A,B}
    T = promote_type(eltype(_D21), eltype(_D12))
    D21_band2 = view(arguments(bandeddata(_D21))[1], 3:∞)
    D21_band1 = view(arguments(bandeddata(_D21))[2], 2:∞)
    D12_bandn1 = _D12.ev
    D12_band0 = _D12.dv
    
    D1 = Vcat(zero(T), zero(T), InterlacedVector(-D21_band2 .* q .* D21_band1, -view(D12_bandn1, 2:∞) .* h .* view(D12_band0, 3:∞)))
    D2 = Zeros{T}(∞)
    D3 = Vcat(-_D21[1, 2]^2 * q, InterlacedVector(-(view(D12_band0, 2:∞) .^ 2 .+ view(D12_bandn1, 2:∞) .^ 2) .* h, -(view(D21_band1, 2:∞) .^ 2 .+ D21_band2 .^ 2) .* q))
    D4 = Zeros{T}(∞)
    D5 = InterlacedVector(-D21_band2 .* q .* D21_band1, -view(D12_bandn1, 2:∞) .* h .* view(D12_band0, 3:∞))
    D = Vcat(transpose(D1), transpose(D2), transpose(D3), transpose(D4), transpose(D5))

    _Ddata = _BandedMatrix(D, axes(_D12, 1), 2, 2)
    return _Ddata
end

function _weaklaplacian_equal(P::PiecewiseArcPolynomial{1,T}) where {T}
    nel = length(P.points) - 1
    D00, Δh = _weaklaplacian_compute_D00_equal(P, nel)
    invh = sqrt(inv(Δh))
    
    # Compute D21 and D12 once
    _D21 = semiclassicaljacobiarc_diffp(P.P)
    _D12 = semiclassicaljacobiarc_diffq(P.P)
    
    # Use for D20, Dlace22, and Ddata
    D20 = _weaklaplacian_compute_D20_equal(P.P0.P, nel, Δh, P.P0.h)
    D20_sq = D20 .* invh
    Dlace22 = fill(_D12[2, 2], nel)
    q = (Δh - sin(Δh)) / 2
    Ddata = _weaklaplacian_compute_Ddata_equal(_D21, _D12, q, Δh)
    
    A = -(MᵀM(D00) + MᵀM(D20_sq))
    B = (Zeros{T}(nel, nel), (-D20 .* Dlace22)')
    C = (B[1], B[2]')
    return CyclicBBBArrowheadMatrix(A, B, C, Fill(Ddata, nel))
end

function _weaklaplacian_unequal(P::PiecewiseArcPolynomial{1,T}) where {T}
    nel = length(P.points) - 1
    D00, hs, invh = _weaklaplacian_compute_D00_unequal(P, nel)
    
    # Compute everything per element in one loop
    _D20_diag = Vector{T}(undef, nel)
    Dlace22 = Vector{T}(undef, nel)
    D = Vector{BandedMatrix{T}}(undef, nel)
    
    for i in 1:nel
        # Compute semiclassical functions once per element
        Pi = get_P(P, i)
        _D21 = semiclassicaljacobiarc_diffp(Pi)
        _D12 = semiclassicaljacobiarc_diffq(Pi)
        
        # Use for D20
        _P0 = get_P0(P, i)
        _D20_diag[i] = _weaklaplacian_compute_D20_diag_equal(_P0.P, _P0.h, hs[i])
        
        # Use for Dlace22
        Dlace22[i] = _D12[2, 2]
        
        # Use for Ddata
        _hs = hs[i]
        _qs = (_hs - sin(_hs)) / 2
        D[i] = _weaklaplacian_compute_Ddata_equal(_D21, _D12, _qs, _hs)
    end

    D = map(identity, D) # concrete type
    
    D20 = _weaklaplacian_compute_D20_from_diag(_D20_diag, nel)
    D20_sq = D20 .* invh
    
    A = -(MᵀM(D00) + MᵀM(D20_sq))
    B = (Zeros{T}(nel, nel), (-D20 .* Dlace22)')
    C = (B[1], B[2]')
    return CyclicBBBArrowheadMatrix(A, B, C, D)
end

function weaklaplacian_layout(_, P::PiecewiseArcPolynomial{1,T}) where {T}
    has_equal_spacing(P) ? _weaklaplacian_equal(P) : _weaklaplacian_unequal(P)
end