function diff(P::PeriodicContinuousPolynomial{1,T}; dims=1) where {T}
    U = Ultraspherical{T}(-one(T)/2)
    DU = diff(U).args[2]

    nel = length(P.points) - 1 
    hs = diff(P.points)
    invh = inv.(hs)
    _D00 = BandedMatrix{T}(0 => -invh, 1 => @view(invh[begin:end-1]))
    D00 = CyclicBandedMatrix(_D00, nel == 2 ? one(T) * invh[1] : zero(T), one(T) * invh[end])
    Ddata = _BandedMatrix(-Ones{T}(∞)', axes(DU, 1), 0, 0)
    D = [Ddata * 2h⁻¹ for h⁻¹ in invh]

    P0 = PeriodicContinuousPolynomial{0}(P)
    return P0 * CyclicBBBArrowheadMatrix(D00, (), (), D)
end

function weaklaplacian_layout(_, P::PeriodicContinuousPolynomial{1,T}) where {T}
    U = Ultraspherical{T}(-one(T)/2)
    DU = diff(U).args[2]

    r = P.points
    nel = length(r) - 1
    hs = diff(r)
    ihs = inv.(hs)
    ishs = sqrt.(ihs)

    # Compute A 
    _D0 = BandedMatrix{T}(0 => -ishs, 1 => @view(ishs[begin:end-1]))
    D0 = CyclicBandedMatrix(_D0, nel == 2 ? ishs[1] : zero(T), ishs[end])
    A = -MᵀM(D0)

    # Compute D 
    n = 1:∞
    Ls = -4 ./ (2 .* n .+ 1)
    Ddata = _BandedMatrix(Ls', axes(DU, 1), 0, 0)
    D = [Ddata * h⁻¹ for h⁻¹ in ihs]

    return CyclicBBBArrowheadMatrix(A, (), (), D)
end