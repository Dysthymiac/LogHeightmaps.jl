using CUDA # It is recommended to import CUDA to make use of GPU for the calculation of the heightmap
using LogHeightmaps
using MAT
using GLMakie
using Images
using Statistics

include("load_log.jl")
# file_path = "path/to/pointcloud.mat"

data = MAT.matread(file_path)

points = data["laser"][:, 2:4]' |> collect

# scatter(points[[3, 1, 2], :], fxaa=true, color=points[3, :])

@time "Filtering" points_filter, circles = filter_point_cloud(data["laser"]', debug_plots=false)
# points_filter, circles = filter_point_cloud(points, debug_plots=false)

points = points[[3, 1, 2], points_filter]

# f = scatter(points, fxaa=true, color=points[1, :])
# lines!(circles[2:4, :])
# display(f)

@time "Centerline" centerline = SmoothingSpline3D(eachrow(circles[2:4, :])...)
# @time "Centerline" centerline = SmoothingSpline3D(eachrow(circles[2:4, :])..., n_segments=nothing)

width = 360
height = size(circles, 2)

@time "Coordinate conversion" heightmap_coords, minX, maxX = to_heightmap_coordinates(points, 
            centerline, 
            width, 
            height)

real_height = maxX - minX
@time "Heightmap" heightmap = calculate_heightmap(heightmap_coords, 0.5, width, height, real_height)
normalize(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))


branches = segment_branches(heightmap, 15, (real_height, 2π * mean(heightmap)))

Gray.([normalize(heightmap) normalize(branches)]) |> display

display(GLMakie.Screen(), scatter(heightmap_coords, fxaa=true, color=heightmap_coords[3, :]))
# segment_branches(I, branch_radius, real_size)

converted_back, _, _ = from_heightmap_coordinates(heightmap_coords, centerline, width, height, minX, maxX)

f = scatter(points, fxaa=true, color=heightmap_coords[3, :])
lines!(predict_points(centerline, range(minX, maxX, 100))..., linewidth=4)
meshscatter!(circles[2:4, :], markersize=10)
display(GLMakie.Screen(), f)


f = scatter(converted_back, fxaa=true, color=heightmap_coords[3, :], colormap=:magma)
display(GLMakie.Screen(), f)


##

# center_points = circles[2:4, :]
# f =  scatter(center_points)

# lines!(predict_points(centerline, center_points[1, :])...)

# peaks = get_3D_peaks(centerline.splineXY, centerline.splineXZ)
# clean_peaks = peaks[[true; abs.(diff(peaks)) .> 5]]

# new_centerline = SmoothingSpline3D(predict_points(centerline, peaks)..., λ=0.0)
# new_new_centerline = SmoothingSpline3D(predict_points(centerline, clean_peaks)..., λ=0.0)

# lines!(predict_points(new_centerline, center_points[1, :])...)
# lines!(predict_points(new_new_centerline, center_points[1, :])...)

# scatter!(predict_points(centerline, peaks)..., marker=:x, markersize=10)
# scatter!(predict_points(centerline, clean_peaks)..., marker=:x, markersize=10)

# display(f)