using LogHeightmaps
using MAT
using GLMakie
using Images

log = 1
run = 3
file_path = 

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

width = 360
height = size(circles, 2)

@time "Coordinate conversion" heightmap_coords, minX, maxX = to_heightmap_coordinates(points, 
            centerline, 
            width, 
            height)

real_height = maxX - minX
@time "Heightmap" heightmap = calculate_heightmap(heightmap_coords, 0.1, width, height, real_height)
Gray.((heightmap .- minimum(heightmap)) ./ (maximum(heightmap) - minimum(heightmap))) |> display
scatter(heightmap_coords, fxaa=true, color=heightmap_coords[3, :])
