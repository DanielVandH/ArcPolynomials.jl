function layout_broadcasted(::Tuple{ExpansionLayout{PiecewisePolynomialLayout{0}}, PiecewisePolynomialLayout{0}}, ::typeof(*), a, P::PeriodicContinuousPolynomial{0})
    @assert basis(a) == P 
    _, c = arguments(a)
    T = eltype(c)
    nel = length(P.points) - 1 
    cdata = paddeddata(c)
    B = Legendre{T}()
    ops = map(1:nel) do i 
        coeffs = pad(cdata[i:nel:end], axes(B, 2)) # If we just do c[i:nel:end], then we don't get an ExpansionLayout for the result
        ela = B * coeffs
        _, Ja = arguments(ela .* B)
        Ja
    end
    return P * InterlacedMatrix(ops)
end

function layout_broadcasted(::Tuple{ExpansionLayout{PiecewisePolynomialLayout{0}}, PiecewisePolynomialLayout{1}}, ::typeof(*), a, C::PeriodicContinuousPolynomial{1})
    P = PeriodicContinuousPolynomial{0}(C)
    return (a .* P) * (P \ C)
end