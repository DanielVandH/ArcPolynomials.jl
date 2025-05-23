struct InterlacedVector{T,A<:AbstractVector{T},B<:AbstractVector{T}} <: LazyVector{T}
    a::A
    b::B
end
size(::InterlacedVector) = (∞,)
function getindex(v::InterlacedVector, i::Int)
    @boundscheck checkbounds(Bool, v, i) || throw(BoundsError(v, i))
    return isodd(i) ? v.a[i÷2+1] : v.b[i÷2]
end

function \(P0::PiecewiseArcPolynomial{0,T}, P::PiecewiseArcPolynomial{1,V}) where {T,V}
    F = float(promote_type(T, V))
    @assert P0.points == P.points
    if has_equal_spacing(P)
        Ψ11 = P.P0.Q \ P.P.Q
        Ψ22 = P.P0.P \ P.P.P
    else
        Ψ11 = [get_P0(P, i).Q \ get_P(P, i).Q for i in 1:length(P.points)-1]
        Ψ22 = [get_P0(P, i).P \ get_P(P, i).P for i in 1:length(P.points)-1]
    end

    nel = length(P.points) - 1
    _R00 = _BandedMatrix(Ones{F}(2, nel) / 2, oneto(nel), 0, 1)
    h = diff(P.points)
    _R10_diag = @. (sin(h) - h) / (4sin(h / 2))
    _R10_supdiag = -_R10_diag 
    circshift!(_R10_supdiag, 1) # [1, 2, ..., n] => [2, 3, ..., n, 1]
    _R10 = _BandedMatrix(vcat(_R10_supdiag', _R10_diag'), oneto(nel), 0, 1)
    if has_equal_spacing(P)
        _R10_scale = 2(1 - P.P0.h)^2 * sum(orthogonalityweight(P.P0.Q)) # grammatrix(P0)[nel+1,nel+1]
    else
        _R10_scale = [2(1 - get_P0(P, i).h)^2 * sum(orthogonalityweight(get_P0(P, i).Q)) for i in 1:nel]
    end
    R00 = CyclicBandedMatrix(_R00, nel == 2 ? one(F) / 2 : zero(F), one(F) / 2)
    R10 = CyclicBandedMatrix(_R10, nel == 2 ? _R10_supdiag[2] : zero(F), _R10_supdiag[1]) ./ _R10_scale
    if has_equal_spacing(P)
        R01 = Diagonal(Fill(Ψ22[1, 2], nel))
    else
        R01 = Diagonal([Ψ22[i][1, 2] for i in 1:nel])
    end

    A = R00
    B = (R01,)
    C = (R10,)
    if has_equal_spacing(P)
        Ddata = Hcat(
            InterlacedVector(Ψ22[band(1)], Ψ11[band(1)]), # FIXME: Convert to F type
            Zeros{F}(∞),
            InterlacedVector(Ψ22[band(0)][2:end], Ψ11[band(0)][2:end]), # FIXME: Convert to F type
        )
        D = [_BandedMatrix(Ddata', axes(Ψ11, 1), 1, 1) for _ in 1:nel]
    else
        D = [
            _BandedMatrix(
                Hcat(
                    InterlacedVector(Ψ22[i][band(1)], Ψ11[i][band(1)]), # FIXME: Convert to F type
                    Zeros{F}(∞),
                    InterlacedVector(Ψ22[i][band(0)][2:end], Ψ11[i][band(0)][2:end]), # FIXME: Convert to F type
                )',
                axes(Ψ11[i], 1), 1, 1
            ) for i in 1:nel
        ]
    end
    conversion = CyclicBBBArrowheadMatrix(A, B, C, D)

    return conversion
end