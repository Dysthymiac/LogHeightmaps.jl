export to_heightmap_coordinates, to_heightmap_coordinates!, from_heightmap_coordinates, from_heightmap_coordinates!
# export straighten_x, straighten_x!, curve_x, curve_x!
struct DistanceOptimParams{T1, T2, T3, T4, T5}
    xl::T1
    h::T2
    cXY::T3
    cXZ::T4
    point::T5
end
function (o::DistanceOptimParams)(t)
    O = SVector(o.xl + t * o.h, apply_coeff(t, o.cXY), apply_coeff(t, o.cXZ))
    n = SVector(o.h, get_deriv(t, o.cXY), get_deriv(t, o.cXZ)) 
    return ((o.point - O) ⋅ n)^2
end

function straighten_x!(points, spline; iterations=2)
    
    baseOY = SVector(0, 1, 0)
    n = size(points, 2)
    filt0 = trues(n)
    filt1 = falses(n)
    filt = falses(n)
    unused = trues(n)
    xs = vcat(spline.splineXY.Xdesign, [spline.splineXY.Xdesign[end]])

    normals =  get_deriv.(Ref(spline), xs) .|> normalize
    origins = predict_points.(Ref(spline), xs)
    
    for (i, normal, origin, (dy, cy, by, ay), (dz, cz, bz, az)) ∈ zip(1:length(normals), normals, origins, eachcol(spline.coeffsXY), eachcol(spline.coeffsXZ))
        cXY, cXZ = (dy, cy, by, ay), (dz, cz, bz, az)
        if i < length(normals)
            filt1 .= (points' * normal) .≤ (origin ⋅ normal)
        else
            filt1 .= true
        end
        
        filt .= filt0 .& filt1
        ps = @view points[:, filt .& unused]
        xp = @view ps[1, :]
        yp = @view ps[2, :]
        zp = @view ps[3, :]

        isempty(xp) && continue

        if i == 1
            xl = spline.splineXY.Xdesign[1]
            h = spline.splineXY.Xdesign[2] - spline.splineXY.Xdesign[1]
            b = @. cy*dy - cy*yp + cz*dz - cz*zp + h*xl - h*xp
            a = cy^2 + cz^2 + h^2
            t = -b ./ a
        elseif i == length(normals)
            xl = spline.splineXY.Xdesign[end]
            h = spline.splineXY.Xdesign[end] - spline.splineXY.Xdesign[end-1]
            b = @. cy*dy - cy*yp + cz*dz - cz*zp + h*xl - h*xp
            a = cy^2 + cz^2 + h^2
            t = -b ./ a
        else
            cXY, cXZ = SVector{4}(spline.coeffsXY[:, i]), SVector{4}(spline.coeffsXZ[:, i])
            xl = spline.splineXY.Xdesign[i-1]
            h = spline.splineXY.Xdesign[i] - spline.splineXY.Xdesign[i-1]
            t = tmap1(point->optimize(DistanceOptimParams(xl, h, cXY, cXZ, point), -0.1, 1.1; iterations=iterations) |> Optim.minimizer, eachcol(ps))
            t = clamp.(t, 0, 1)
        end
        
        x = xl .+ t .* h
        OX = SVector.(h, get_deriv.(t, Ref(cXY)), get_deriv.(t, Ref(cXZ))) .|> normalize
        OZ = normalize.(OX .× Ref(baseOY))
        OY = normalize.(OZ .× OX)
        
        T = hcat.(OX, OY, OZ) .|> transpose
        O = SVector.(x, apply_coeff.(t, Ref(cXY)), apply_coeff.(t, Ref(cXZ)))
        l = get_length.(Ref(spline), x)
        
        zero_x(x) = x .* SVector(0, 1, 1)
        
        ps .= stack(zero_x.(T .* (eachcol(ps) .- O)) .+ SVector.(l, 0, 0))
    
        unused[filt] .= false
        filt0 .= .!filt1
    end

    return points
end

straighten_x(spline, points; iterations=2) = straighten_x!(copy(points), spline; iterations=iterations)

struct LengthOptimParams{T1, T2, T3, T4}
    h::T1
    cXY::T2
    cXZ::T3
    xp::T4
end
get_segment_length_deriv_func(o::LengthOptimParams) = get_segment_length_deriv_inner(t1) = get_segment_length_deriv(t1, o.h, o.cXY, o.cXZ)
(o::LengthOptimParams)(t) = (quadrature_length5(get_segment_length_deriv_func(o), 0, t) - o.xp)^2

function curve_x!(points, spline; iterations=2)
    baseOY = SVector(0, 1, 0)
    n = size(points, 2)
    filt0 = trues(n)
    filt1 = falses(n)
    filt = falses(n)
    unused = trues(n)

    lens =  vcat(spline.cumlengths, [spline.cumlengths[end]])
    
    for (i, len, cXY, cXZ) ∈ zip(1:length(lens), lens, eachcol(spline.coeffsXY), eachcol(spline.coeffsXZ))
        cXY, cXZ = SVector{4}(cXY), SVector{4}(cXZ)
        if i < length(lens)
            filt1 .= points[1, :] .< len
        else
            filt1 .= true
        end
        
        filt .= filt0 .& filt1
        ps = @view points[:, filt .& unused]
        xp = @view ps[1, :]

        isempty(xp) && continue
        
        xp .-= lens[clamp(i-1, 1, end)]
        if i == 1
            xl = spline.splineXY.Xdesign[1]
            h = spline.splineXY.Xdesign[2] - spline.splineXY.Xdesign[1]
            t = xp ./ norm((h, cXY[2], cXZ[2]))
        elseif i == length(lens)
            xl = spline.splineXY.Xdesign[end]
            h = spline.splineXY.Xdesign[end] - spline.splineXY.Xdesign[end-1]
            t = xp ./ norm((h, cXY[2], cXZ[2]))
        else
            xl = spline.splineXY.Xdesign[i-1]
            h = spline.splineXY.Xdesign[i] - spline.splineXY.Xdesign[i-1]
            t = tmap1(xp0->optimize(LengthOptimParams(h, cXY, cXZ, xp0), 0, 1; iterations=iterations) |> Optim.minimizer, xp)
        end
        
        x = xl .+ t .* h
        O = SVector.(x, apply_coeff.(t, Ref(cXY)), apply_coeff.(t, Ref(cXZ)))

        OX = SVector.(h, get_deriv.(t, Ref(cXY)), get_deriv.(t, Ref(cXZ))) .|> normalize
        OZ = normalize.(OX .× Ref(baseOY))
        OY = normalize.(OZ .× OX)
        T = hcat.(OX, OY, OZ)
        l = get_length.(Ref(spline), x, segment_length=true)
        xp .-= l
        
        ps .= stack(T .* eachcol(ps) .+ O)
        
        unused[filt] .= false
        filt0 .= .!filt1
    end

    return points
end

curve_x(spline, points; iterations=2) = curve_x!(copy(points), spline; iterations=iterations)

scale_range(val, oldMin, oldRange, newMin, newRange) = newMin + newRange * (val - oldMin) / oldRange

function rescale_points!(points, width, height, minX=nothing, maxX=nothing)
    minX = @something minX minimum(points[1, :])
    maxX = @something maxX maximum(points[1, :])
    oldLRange = maxX - minX
    points[2, :] .= scale_range.(@view(points[2, :]), -π, 2π, 1, width-1)
    points[1, :] .= scale_range.(@view(points[1, :]), minX, oldLRange, 1, height-1)
    return points, minX, maxX
end
rescale_points(points, width, height, minY, maxY) = rescale_points!(copy(points), width, height, minY, maxY)

function rescale_points_back!(points, width, height, minX=nothing, maxX=nothing)
    minX = @something minX minimum(points[1, :])
    maxX = @something maxX maximum(points[1, :])
    newRange = maxX - minX
    points[2, :] .= scale_range.(@view(points[2, :]), 1, width-1, -π, 2π)
    points[1, :] .= scale_range.(@view(points[1, :]), 1, height-1, minX, newRange)
    return points, minX, maxX
end
rescale_points_back(points, width, height, minY, maxY) = rescale_points_back!(copy(points), width, height, minY, maxY)


function to_cylindrical!(points)
    points[2:3, :] .= @views [atan.(points[3, :], points[2, :])'; hypot.(points[3, :], points[2, :])']
    return points
end

to_cylindrical(points) = to_cylindrical!(copy(points))

function from_cylindrical!(points)
    points[2:3, :] .= @views [(points[3, :] .* cos.(points[2, :]))'; (points[3, :] .* sin.(points[2, :]))']
    return points
end

from_cylindrical(points) = from_cylindrical!(copy(points))


function to_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    straighten_x!(points, centerline)
    to_cylindrical!(points)
    points, minX, maxX = rescale_points!(points, width, height, minX, maxX)
    return points, minX, maxX
end

to_heightmap_coordinates(points, 
    centerline, 
    width, 
    height, 
    minX=nothing, 
    maxX=nothing) = to_heightmap_coordinates!(copy(points), centerline, width, height, minX, maxX)


function from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    points, minX, maxX = rescale_points_back!(points, width, height, minX, maxX)
    from_cylindrical!(points)
    curve_x!(points, centerline)
    return points, minX, maxX
end

from_heightmap_coordinates(points, 
    centerline, 
    width, 
    height, 
    minX=nothing, 
    maxX=nothing) = from_heightmap_coordinates!(copy(points), centerline, width, height, minX, maxX)