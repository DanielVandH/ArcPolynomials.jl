function \(P0::PeriodicContinuousPolynomial{0,T}, P1::PeriodicContinuousPolynomial{1,V}) where {T, V}
    F = float(promote_type(T, V))
    @assert P0.points == P1.points
    L = Legendre{F}()
    U = Ultraspherical{F}(-one(F)/2)
    R = L \ U   

    nel = length(P1.points) - 1
    _R00 = _BandedMatrix(Ones{F}(2, nel) / 2, oneto(nel), 0, 1)
    _R10 = _BandedMatrix(Vcat(Ones{F}(1, nel), -Ones{F}(1, nel)) / 2, oneto(nel), 0, 1)
    R00 = CyclicBandedMatrix(_R00, nel == 2 ? one(F)/ 2 : zero(F), one(F) / 2)
    R10 = CyclicBandedMatrix(_R10, nel == 2 ? one(F) / 2 : zero(F), one(F) / 2)
    R01 = Diagonal(Fill(one(F)/3, nel))

    A = R00 
    B = (R01,)
    C = (R10,)
    Ddata = Vcat(R[band(2)]', R[band(1)]', R[band(0)][3:end]')
    D = Fill(_BandedMatrix(Ddata, axes(R, 1), 1, 1), nel)
    conversion = CyclicBBBArrowheadMatrix(A, B, C, D)

    return conversion
end