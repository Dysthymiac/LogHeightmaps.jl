
export SmoothingSpline3D, get_peaks, eval_spline, get_length, predict_points


function simpsons_rule13(f, a, b)
    h = (b - a) / 2
    return (1/3) * h * (f(a) + 4f(a+h) + f(b))
end

function simpsons_rule38(f, a, b)
    h = (b - a) / 3
    return (3/8) * h * (f(a) + 3f(a+h) + 3f(a+2h) + f(b))
end

function quadrature_length5(df, a, b)
       gauss_lengendre_coefficients =
       [
              (0.0, 0.5688889),
              (-0.5384693, 0.47862867),
              (0.5384693, 0.47862867),
              (-0.90617985, 0.23692688),
              (0.90617985, 0.23692688),
       ]
       c1 = (b - a)/2
       c2 = (a + b)/2
       length = c1 * sum(norm(df(c1* abscissa + c2)) * weight for (abscissa, weight) ∈ gauss_lengendre_coefficients)
       return length
end

struct SmoothingSpline3D{T}
    splineXY::SmoothingSpline{T}
    splineXZ::SmoothingSpline{T}
    coeffsXY::AbstractMatrix{T}
    coeffsXZ::AbstractMatrix{T}
    cumlengths::AbstractVector{T}
end

function get_coeffs(spl::SmoothingSpline)
    n = length(spl.Xdesign)
    zeroγ = [zero(eltype(spl.γ))]
    coeffs = map(spl.Xdesign, spl.Xdesign[2:end], spl.g, spl.g[2:end], vcat(zeroγ, spl.γ), vcat(spl.γ, zeroγ)) do xl, xr, gl, gr, yl, yr
        h = xr-xl
        return [
            gl, 
            -gl + gr - (1/3)*(h^2)*yl - (1/6)*(h^2)*yr, 
            (1/2)*(h^2)*yl, 
            -(1/6)*(h^2)*yl + (1/6)*(h^2)*yr
            ]
    end
    xl, xr, gl, gr, yr = spl.Xdesign[1], spl.Xdesign[2], spl.g[1], spl.g[2], spl.γ[1]
    h = xr - xl
    first_coeff = [gl, -gl + gr - (1/6)*(h^2)*yr, 0, 0]
    xl, xr, gl, gr, yl = spl.Xdesign[n-1], spl.Xdesign[n], spl.g[n-1], spl.g[n], spl.γ[n-2]
    h = xr - xl
    # last_coeff = [gl - (1/6)*(h^2)*yl, -gl + gr + (1/6)*(h^2)*yl, 0, 0]
    last_coeff = [gr, -gl + gr + (1/6)*(h^2)*yl, 0, 0]
    
    return hcat(first_coeff, stack(coeffs), last_coeff)
end

function get_peaks(spl::SmoothingSpline{T}, coeffs=get_coeffs(spl)) where T
    peaks = T[]
    for (xl, xr, coeffs) ∈ zip(spl.Xdesign, spl.Xdesign[2:end], eachcol(coeffs[:, 2:end]))
        h = xr - xl
        c0, c1, c2, c3 = coeffs
        D = c2.^2 - 4c3 * c1
        push!(peaks, xl)
        if D > 0
            t1 = (-c2 + √D)/2c3
            if t1 ∈ 0..1
                push!(peaks, xl + h * t1)
            end
            t2 = (-c2 - √D)/2c3
            if t2 ∈ 0..1
                push!(peaks, xl + h * t2)
            end
        elseif D ≈ 0
            push!(peaks, xl + h * (-c2)/2c3)
        end
    end
    push!(peaks, spl.Xdesign[end])
    return peaks
end

get_deriv(t, coeffs) = coeffs[2] + t * coeffs[3] + t^2 * coeffs[4]

function get_deriv(spl::SmoothingSpline3D, x)
    Xdesign = spl.splineXY.Xdesign
    idxl = searchsortedlast(Xdesign, x)
    idxr = idxl + 1
    if idxl < 1
        h = Xdesign[2] - Xdesign[1]
        cXY = spl.coeffsXY[:, 1]
        cXZ = spl.coeffsXZ[:, 1]
        t1 = (Xdesign[1] - x) / h
    elseif idxl ≥ length(Xdesign)
        h = Xdesign[end] - Xdesign[end-1]
        cXY = spl.coeffsXY[:, end]
        cXZ = spl.coeffsXZ[:, end]
        t1 = (x - Xdesign[end]) / h
    else
        h = Xdesign[idxr] - Xdesign[idxl]
        cXY = spl.coeffsXY[:, idxl + 1]
        cXZ = spl.coeffsXZ[:, idxl + 1]
        t1 = (x - Xdesign[idxl]) / h
    end
    return [h, get_deriv(t1, cXY), get_deriv(t1, cXZ)]
end

apply_coeff(t, coeff) = coeff[1] + t * coeff[2] + t^2 * coeff[3] + t^3 * coeff[4]

function eval_spline(spl::SmoothingSpline3D, x)
    Xdesign = spl.splineXY.Xdesign
    idxl = searchsortedlast(Xdesign, x)
    idxr = idxl + 1
    if idxl < 1
        h = Xdesign[2] - Xdesign[1]
        cXY = spl.coeffsXY[:, 1]
        cXZ = spl.coeffsXZ[:, 1]
        t1 = (Xdesign[1] - x) / h
        return [Xdesign[1] - t1 * h, apply_coeff(t1, cXY), apply_coeff(t1, cXZ)]
    elseif idxl ≥ length(Xdesign)
        h = Xdesign[end] - Xdesign[end-1]
        cXY = spl.coeffsXY[:, end]
        cXZ = spl.coeffsXZ[:, end]
        t1 = (x - Xdesign[end]) / h
        return [Xdesign[idxl] + t1 * h, apply_coeff(t1, cXY), apply_coeff(t1, cXZ)]
    else
        h = Xdesign[idxr] - Xdesign[idxl]
        cXY = spl.coeffsXY[:, idxl + 1]
        cXZ = spl.coeffsXZ[:, idxl + 1]
        t1 = (x - Xdesign[idxl]) / h
        return [Xdesign[idxl] + t1 * h, apply_coeff(t1, cXY), apply_coeff(t1, cXZ)]
    end
end

get_segment_length_deriv(t, h, coeffsY, coeffsZ) = √(h^2 + get_deriv(t, coeffsY)^2 + get_deriv(t, coeffsZ)^2)

function get_length(spl::SmoothingSpline3D, x; segment_length=false)
    Xdesign = spl.splineXY.Xdesign
    idxl = searchsortedlast(Xdesign, x)
    idxr = idxl + 1
    if idxl < 1
        h = Xdesign[2] - Xdesign[1]
        cXY = spl.coeffsXY[:, 1]
        cXZ = spl.coeffsXZ[:, 1]
        t = (Xdesign[1] - x) / h
        len =  t * norm((h, cXY[2], cXZ[2]))
        l0 = segment_length ? 0 : spl.cumlengths[1]
        return l0 - len
    elseif idxl ≥ length(Xdesign)
        h = Xdesign[end] - Xdesign[end-1]
        cXY = spl.coeffsXY[1:2, end]
        cXZ = spl.coeffsXZ[1:2, end]
        t = (x - Xdesign[end]) / h
        len = t * norm((h, cXY[2], cXZ[2]))
        l0 = segment_length ? 0 : spl.cumlengths[end]
        return l0 + len
    else
        h = Xdesign[idxr] - Xdesign[idxl]
        cXY = spl.coeffsXY[:, idxl + 1]
        cXZ = spl.coeffsXZ[:, idxl + 1]
        t = (x - Xdesign[idxl]) / h
        # l = quadrature_length5(t1->get_segment_length_deriv(t1, h, cXY, cXZ), 0, t) 
        l = t * (spl.cumlengths[idxr] - spl.cumlengths[idxl])
        l0 = segment_length ? 0 : spl.cumlengths[idxl]
        return l0 + l
    end
end

function SmoothingSpline3D(X, Y, Z; λ=250.0) 
    splineXY = SmoothingSplines.fit(SmoothingSpline, X, Y, λ)
    splineXZ = SmoothingSplines.fit(SmoothingSpline, X, Z, λ)
    coeffsXY = get_coeffs(splineXY)
    coeffsXZ = get_coeffs(splineXZ)
    cumlengths = vcat([0], cumsum(
        [quadrature_length5(t->get_segment_length_deriv(t, h, cXY, cXZ), 0, 1) 
            for (h, cXY, cXZ) ∈ zip(X[2:end] .- X[1:end-1], eachcol(coeffsXY[:, 2:end]), eachcol(coeffsXZ[:, 2:end]))]))
    peaks_Y = get_peaks(splineXY)
    peaks_Z = get_peaks(splineXZ)
    return SmoothingSpline3D(splineXY, splineXZ, coeffsXY, coeffsXZ, cumlengths)
end


(spl::SmoothingSpline3D)(x) = (SmoothingSplines.predict(spl.splineXY, x), SmoothingSplines.predict(spl.splineXZ, x))
predict_points(spl::SmoothingSpline3D, x) = (x, spl(x)...)

normalize(x) = x ./ norm(x)

# function fit_3D_centerline(X, Y, Z; segments=30, λ=250.0)
    
#     minX, maxX = extrema(X)
#     Xs = range(minX, maxX, length=segments+1)

#     splX = SmoothingSplines.fit(SmoothingSpline, X, Y, λ) 
#     splY = SmoothingSplines.fit(SmoothingSpline, X, Z, λ) 

#     return Xs, splX, splY
# end

# function predict_3D_centerline(Xs, splX, splY)
#     Y_pred = SmoothingSplines.predict(splX, Xs) # fitted vector
#     Z_pred = SmoothingSplines.predict(splY, Xs) # fitted vector
#     return Xs, Y_pred, Z_pred
# end

# function fit_predict_3D_centerline(X, Y, Z; segments=30, λ=250.0)
#     return predict_3D_centerline(fit_3D_centerline(X, Y, Z; segments=segments, λ=λ)...)
# end

# function fit_predict_3D_centerline(points; segments=30, λ=250.0)
#     res = fit_predict_3D_centerline(points[1, :], points[2, :], points[3, :]; segments=segments, λ=λ)
#     return stack(res, dims=1)
# end