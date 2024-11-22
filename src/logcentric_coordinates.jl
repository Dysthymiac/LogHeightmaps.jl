export to_heightmap_coordinates, to_heightmap_coordinates!, from_heightmap_coordinates, from_heightmap_coordinates!
export rescale_points!, rescale_points, rescale_points_back!, rescale_points_back
export from_logcentric_coordinates, from_logcentric_coordinates!, to_logcentric_coordinates, to_logcentric_coordinates!
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


"""
    rescale_points!(points, width, height, min_l=nothing, max_l=nothing)
    
In-place version of [`rescale_points`](@ref) function.
Rescale log-centric points to so that ``l ∈ [1, height], θ ∈ [1, width]``. 
`points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively. 
`min_l` and `max_l` are minimum and maximum values of l. If not provided explicitly, estimated from points. 

"""
function rescale_points!(points, width, height, min_l=nothing, max_l=nothing)
    min_l = @something min_l minimum(points[1, :])
    max_l = @something max_l maximum(points[1, :])
    oldLRange = max_l - min_l
    points[2, :] .= scale_range.(@view(points[2, :]), -π, 2π, 1, width-1)
    points[1, :] .= scale_range.(@view(points[1, :]), min_l, oldLRange, 1, height-1)
    return points, min_l, max_l
end
"""
    rescale_points!(points, width, height, min_l=nothing, max_l=nothing)
    
Rescale log-centric points to so that ``l ∈ [1, height], θ ∈ [1, width]``. 
`points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively. 
`min_l` and `max_l` are minimum and maximum values of l. If not provided explicitly, estimated from points. 

"""
rescale_points(points, width, height, min_l=nothing, max_l=nothing) = rescale_points!(copy(points), width, height, min_l, max_l)
"""
    rescale_points_back!(points, width, height, min_l, max_l)
    
In-place version of [`rescale_points_back`](@ref) function.
Rescale log-centric points from ``l ∈ [1, height], θ ∈ [1, width]`` to ``l ∈ [min_l, max_l], θ ∈ [-π, π]``. 
`points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively. 

"""
function rescale_points_back!(points, width, height, min_l, max_l)
    min_l = @something min_l minimum(points[1, :])
    max_l = @something max_l maximum(points[1, :])
    newRange = max_l - min_l
    points[2, :] .= scale_range.(@view(points[2, :]), 1, width-1, -π, 2π)
    points[1, :] .= scale_range.(@view(points[1, :]), 1, height-1, min_l, newRange)
    return points, min_l, max_l
end
"""
    rescale_points_back(points, width, height, min_l, max_l)
    
Rescale log-centric points from ``l ∈ [1, height], θ ∈ [1, width]`` to ``l ∈ [min_l, max_l], θ ∈ [-π, π]``. 
`points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively. 

"""
rescale_points_back(points, width, height, min_l, max_l) = rescale_points_back!(copy(points), width, height, min_l, max_l)


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

"""
    to_logcentric_coordinates!(points, centerline)
    
In-place version of the [`to_logcentric_coordinates`](@ref) function.
Convert 3 × n matrix of 3D cartesian points to the logcentric coordinates ``l, θ, ρ``.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details. 
Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. 
"""
function to_logcentric_coordinates!(points, centerline)
    straighten_x!(points, centerline)
    to_cylindrical!(points)
    return points
end
"""
    to_logcentric_coordinates(points, centerline)
    
Convert 3 × n matrix of 3D cartesian points to the logcentric coordinates ``l, θ, ρ``.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.
Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. 
"""
to_logcentric_coordinates(points, centerline) = to_logcentric_coordinates!(copy(points), centerline)
"""
    from_logcentric_coordinates!(points, centerline)
    
In-place version of the [`from_logcentric_coordinates`](@ref) function.
Convert 3 × n matrix of logcentric coordinates ``l, θ, ρ`` to 3D cartesian points.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details. 
Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. 
"""
function from_logcentric_coordinates!(points, centerline)
    from_cylindrical!(points)
    curve_x!(points, centerline)
    return points
end
"""
    from_logcentric_coordinates(points, centerline)
    
Convert 3 × n matrix of logcentric coordinates ``l, θ, ρ`` to 3D cartesian points.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details. 
Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. 
"""
from_logcentric_coordinates(points, centerline) = from_logcentric_coordinates!(copy(points), centerline)
"""
    to_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    
In-place version of the [`to_heightmap_coordinates`](@ref) function.
Convert 3 × n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap).
Applies [`to_logcentric_coordinates`](@ref) and [`rescale_points`](@ref).

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

"""
function to_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    to_logcentric_coordinates!(points, centerline)
    points, minX, maxX = rescale_points!(points, width, height, minX, maxX)
    return points, minX, maxX
end

"""
    to_heightmap_coordinates(points, centerline, width, height, minX=nothing, maxX=nothing)
    
Convert 3 × n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap).
Applies [`to_logcentric_coordinates`](@ref) and [`rescale_points`](@ref).

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

"""
to_heightmap_coordinates(points, 
    centerline, 
    width, 
    height, 
    minX=nothing, 
    maxX=nothing) = to_heightmap_coordinates!(copy(points), centerline, width, height, minX, maxX)

"""
    from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    
In-place version of the [`from_heightmap_coordinates`](@ref) function.
Convert ``3 × n`` matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap).
Applies [`rescale_points_back`](@ref) and [`from_logcentric_coordinates`](@ref).

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.
"""
function from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    points, minX, maxX = rescale_points_back!(points, width, height, minX, maxX)
    from_logcentric_coordinates!(points, centerline)
    return points, minX, maxX
end
"""
    from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)
    
Convert ``3 × n`` matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap).
Applies [`rescale_points_back`](@ref) and [`from_logcentric_coordinates`](@ref).

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.
"""
from_heightmap_coordinates(points, 
    centerline, 
    width, 
    height, 
    minX=nothing, 
    maxX=nothing) = from_heightmap_coordinates!(copy(points), centerline, width, height, minX, maxX)