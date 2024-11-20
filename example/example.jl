using LogHeightmaps
using MAT
using GLMakie

log = 1
run = 3
file_path = joinpath(raw"/home/ted/OneDrive-LUT", raw"datasets/Honkalahti_20180324/logs/point_clouds_mat_new", "mustola_20180321_log" * string(log, pad=2) * "_run$run.mat")

data = MAT.matread(file_path)

points = data["laser"][:, 2:4]' |> collect

# scatter(points[[3, 1, 2], :], fxaa=true, color=points[3, :])

points_filter, circles = filter_point_cloud(data["laser"]', debug_plots=false)
# points_filter, circles = filter_point_cloud(points, debug_plots=false)

points = points[[3, 1, 2], points_filter]

f = scatter(points, fxaa=true, color=points[1, :])

lines!(circles[2:4, :])

display(f)