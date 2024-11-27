var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = LogHeightmaps","category":"page"},{"location":"#LogHeightmaps","page":"Home","title":"LogHeightmaps","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for LogHeightmaps.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [LogHeightmaps]","category":"page"},{"location":"#LogHeightmaps.ExtremaSegments","page":"Home","title":"LogHeightmaps.ExtremaSegments","text":"struct ExtremaSegments{T}\n\nA class to specify that the spline should be divided into segments between the extrema points of the original spline. min_x_step specifies the minimum difference between control points.\n\n\n\nmin_x_step\n\n\n\n\n\n","category":"type"},{"location":"#LogHeightmaps.SmoothingSpline3D","page":"Home","title":"LogHeightmaps.SmoothingSpline3D","text":"struct SmoothingSpline3D{T}\n\nA class for the 3D smoothing spline, i.e. a class that holds two splines: X ↦ Y and X ↦ Z. Uses splines from SmoothingSplines.jl. \n\n\n\nsplineXY\nsplineXZ\ncoeffsXY\ncoeffsXZ\ncumlengths\n\n\n\n\n\n","category":"type"},{"location":"#LogHeightmaps.SmoothingSpline3D-Tuple{Any, Any, Any}","page":"Home","title":"LogHeightmaps.SmoothingSpline3D","text":"SmoothingSpline3D(X, Y, Z; n_segments=ExtremaSegments(5), λ=250.0)\n\nFunction that creates a SmoothingSpline3D from given points. \n\nKeywords\n\nn_segments=ExtremaSegments(5): specifies how to divide the resulting spline into segments. Can be one of 3 possible values:   if n_segments::Nothing then original X values will be used to divide the spline into segments.   if n_segments::ExtremaSegments then the segments will be chosen according to the extremas of the spline.   if n_segments::Integer then range(minimum(X), maximum(X), n_segments+1) will be used as control points.\nλ=250.0: the smoothing parameter. Please refer to SmoothingSplines.jl package for more information.\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.calculate_heightmap!-NTuple{4, Any}","page":"Home","title":"LogHeightmaps.calculate_heightmap!","text":"calculate_heightmap!(heightmap, rescaled_points, α, width, height, real_length; use_cuda=true)\n\nIn-place version of the calculate_heightmap function. Calculate heightmap from the points in log-centric coordinate system.  rescaled_points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter.  It is assumed that l is already rescaled to the [1, height] range, and θ is rescaled to [1, width].  real_length is the difference between minimum and maximum l values before rescaling. \n\nRefer to [1], [2] for more details.\n\nKeywords\n\nuse_cuda=cuda_is_functional(): it is possible to use CUDA to speed up the computation.\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.calculate_heightmap-NTuple{5, Any}","page":"Home","title":"LogHeightmaps.calculate_heightmap","text":"calculate_heightmap(rescaled_points, α, width, height, real_length; use_cuda=true)\n\nCalculate heightmap from the points in log-centric coordinate system.  rescaled_points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter.  It is assumed that l is already rescaled to the [1, height] range, and θ is rescaled to [1, width].  real_length is the difference between minimum and maximum l values before rescaling. \n\nRefer to [1], [2] for more details.\n\nKeywords\n\nuse_cuda=cuda_is_functional(): it is possible to use CUDA to speed up the computation.\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.filter_point_cloud-Tuple{Any}","page":"Home","title":"LogHeightmaps.filter_point_cloud","text":"filter_point_cloud(data; kws...)\n\nFilter d  n matrix of log point cloud points from noise, where n is the number of points and d is the dimensionality. If d=3, assumes that the points are in cartesian coordinates. If d  4, assumes that the first axis corresponds to a number of layer (for a line scanners) and the next 3 are coordinates. It is assumed that the log is oriented along the z axis.\n\nReturns a tuple of (points_filter, circles), where points_filter is a bitarray specifying which points are left after the filtering, and circles is a 4 × m array containing layer number (cross-section index), center x, center y and radius values, and m is the number of layers (cross-sections).\n\nKeywords\n\nsnake_size=180: number of points to use for active contour \nsnake_kwparams=(α=0.01, β=0.5, γ=10, σ=20, dist_α=1): keyword parameters for the active contour algorithm defined as a NamedTuple\nmin_radius=75: minimum radius of a log cross-section \nmax_radius=200: maximum radius of a log cross-section \nmax_resiudal=4: maximum residual error \npoints_threshold=300: minimum number of points per cross-section \nfiltered_threshold=200: minimum number of points to remain after filtering per cross-section.\nmax_distance=10: maximum distance from snake.\nmax_rad_diff=20: maximum difference in estimated radii.\nsnake_coverage=0.8: the required proportion of active contour points that are near the final points.\nsnake_iterations=10: iterations for the active contour estimation.\nmax_skipped_layers=5: if more layers are skipped, it is assumed that the end of the log has been reached.\ndebug_plots=false: specifies whether to plot each layer with filtering information. Requires GLMakie package to be installed. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.from_heightmap_coordinates","page":"Home","title":"LogHeightmaps.from_heightmap_coordinates","text":"from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)\n\nConvert 3  n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap). Applies rescale_points_back and from_logcentric_coordinates.\n\nRefer to [1], [2] for more details.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.from_heightmap_coordinates!","page":"Home","title":"LogHeightmaps.from_heightmap_coordinates!","text":"from_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)\n\nIn-place version of the from_heightmap_coordinates function. Convert 3  n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap). Applies rescale_points_back and from_logcentric_coordinates.\n\nRefer to [1], [2] for more details.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.from_logcentric_coordinates!-Tuple{Any, Any}","page":"Home","title":"LogHeightmaps.from_logcentric_coordinates!","text":"from_logcentric_coordinates!(points, centerline)\n\nIn-place version of the from_logcentric_coordinates function. Convert 3 × n matrix of logcentric coordinates l θ ρ to 3D cartesian points.\n\nRefer to [1], [2] for more details.  Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.from_logcentric_coordinates-Tuple{Any, Any}","page":"Home","title":"LogHeightmaps.from_logcentric_coordinates","text":"from_logcentric_coordinates(points, centerline)\n\nConvert 3 × n matrix of logcentric coordinates l θ ρ to 3D cartesian points.\n\nRefer to [1], [2] for more details.  Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.get_length-Tuple{SmoothingSpline3D, Any}","page":"Home","title":"LogHeightmaps.get_length","text":"get_length(spl::SmoothingSpline3D, x; segment_length=false)\n\nFunction that returns the length of spline spl at given x.  Uses 5-point Gaussian quadratures to approximate the length.\n\nNote: if given x is less than the first control point of spline,  the returned value will be negative (because this is used for coordinate transformation).\n\nKeywords\n\nsegment_length=false: whether to compute the length from the first control point or only the length of the segment that contains x\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.predict_points-Tuple{SmoothingSpline3D, Any}","page":"Home","title":"LogHeightmaps.predict_points","text":"predict_points(spl::SmoothingSpline3D, x)\n\nEvaluate spline spl at points x. Returns a tuple (X, Y, Z) of evaluated coordinates. If multiple values of x were provided, returns Y and Z as vectors of corresponding sizes.\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.rescale_calculate_heightmap","page":"Home","title":"LogHeightmaps.rescale_calculate_heightmap","text":"rescale_calculate_heightmap(logcentric_points, α, width, height, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional())\n\nCalculate heightmap from the points in log-centric coordinate system.  logcentric_points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter.  min_l and max_l are minimum and maximum values of l. If not provided explicitly, estimated from points.  real_length is the estimated real length of long. If not provided explicitly, calculated as max_l - min_l.\n\nRefer to [1], [2] for more details.\n\nKeywords\n\nuse_cuda=cuda_is_functional(): it is possible to use CUDA to speed up the computation.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.rescale_calculate_heightmap!","page":"Home","title":"LogHeightmaps.rescale_calculate_heightmap!","text":"rescale_calculate_heightmap(logcentric_points, α, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional())\n\nIn-place version of rescale_calculate_heightmap function. Calculate heightmap from the points in log-centric coordinate system.  logcentric_points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter.  min_l and max_l are minimum and maximum values of l. If not provided explicitly, estimated from points.  real_length is the estimated real length of long. If not provided explicitly, calculated as max_l - min_l.\n\nRefer to [1], [2] for more details.\n\nKeywords\n\nuse_cuda=cuda_is_functional(): it is possible to use CUDA to speed up the computation.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.rescale_points","page":"Home","title":"LogHeightmaps.rescale_points","text":"rescale_points!(points, width, height, min_l=nothing, max_l=nothing)\n\nRescale log-centric points to so that l  1 height θ  1 width.  points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively.  min_l and max_l are minimum and maximum values of l. If not provided explicitly, estimated from points. \n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.rescale_points!","page":"Home","title":"LogHeightmaps.rescale_points!","text":"rescale_points!(points, width, height, min_l=nothing, max_l=nothing)\n\nIn-place version of rescale_points function. Rescale log-centric points to so that l  1 height θ  1 width.  points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively.  min_l and max_l are minimum and maximum values of l. If not provided explicitly, estimated from points. \n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.rescale_points_back!-NTuple{5, Any}","page":"Home","title":"LogHeightmaps.rescale_points_back!","text":"rescale_points_back!(points, width, height, min_l, max_l)\n\nIn-place version of rescale_points_back function. Rescale log-centric points from l  1 height θ  1 width to l  min_l max_l θ  -π π.  points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.rescale_points_back-NTuple{5, Any}","page":"Home","title":"LogHeightmaps.rescale_points_back","text":"rescale_points_back(points, width, height, min_l, max_l)\n\nRescale log-centric points from l  1 height θ  1 width to l  min_l max_l θ  -π π.  points must be a 3×n AbstractMatrix, where the rows correspond to l, θ, ρ values respectively. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.segment_branches-Tuple{Any, Any, Any}","page":"Home","title":"LogHeightmaps.segment_branches","text":"segment_branches(I, branch_radius, real_size)\n\nSegment branches from the heightmap image using Difference of Gaussians. branch_radius specifies approximate radius of the branch on the surface in mm. real_size is a tuple containing an estimate of the real size of the heightmap, usually calculated as (real_height, 2π * mean(heightmap)).\n\nRefer to [1], [2] for more details.\n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.to_heightmap_coordinates","page":"Home","title":"LogHeightmaps.to_heightmap_coordinates","text":"to_heightmap_coordinates(points, centerline, width, height, minX=nothing, maxX=nothing)\n\nConvert 3 × n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap). Applies to_logcentric_coordinates and rescale_points.\n\nRefer to [1], [2] for more details.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.to_heightmap_coordinates!","page":"Home","title":"LogHeightmaps.to_heightmap_coordinates!","text":"to_heightmap_coordinates!(points, centerline, width, height, minX=nothing, maxX=nothing)\n\nIn-place version of the to_heightmap_coordinates function. Convert 3 × n matrix of 3D cartesian points to the heightmap coordinates (logcentric coordinates scaled to the size of the heightmap). Applies to_logcentric_coordinates and rescale_points.\n\nRefer to [1], [2] for more details.\n\n\n\n\n\n","category":"function"},{"location":"#LogHeightmaps.to_logcentric_coordinates!-Tuple{Any, Any}","page":"Home","title":"LogHeightmaps.to_logcentric_coordinates!","text":"to_logcentric_coordinates!(points, centerline)\n\nIn-place version of the to_logcentric_coordinates function. Convert 3 × n matrix of 3D cartesian points to the logcentric coordinates l θ ρ.\n\nRefer to [1], [2] for more details.  Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. \n\n\n\n\n\n","category":"method"},{"location":"#LogHeightmaps.to_logcentric_coordinates-Tuple{Any, Any}","page":"Home","title":"LogHeightmaps.to_logcentric_coordinates","text":"to_logcentric_coordinates(points, centerline)\n\nConvert 3 × n matrix of 3D cartesian points to the logcentric coordinates l θ ρ.\n\nRefer to [1], [2] for more details. Note that this conversion differs from the published version and does not approximate centerline as line segments, but works with cubic splines directly. \n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"F. Zolotarev, T. Eerola, L. Lensu, H. Kälviäinen, T. Helin, H. Haario, T. Kauppi and J. Heikkinen. Modelling internal knot distribution using external log features. Computers and Electronics in Agriculture {179} (2020).\n\n\n\nF. Zolotarev. Computer Vision for Virtual Sawing and Timber Tracing. Ph.D. Thesis, Lappeenranta-Lahti University of Technology LUT (2022).\n\n\n\n","category":"page"}]
}
