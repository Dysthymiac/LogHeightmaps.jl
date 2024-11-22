module LogHeightmaps

using StaticArrays, LinearAlgebra
using SmoothingSplines
import IntervalSets...
using Optim
using PolynomialRoots
using ThreadTools
using DocStringExtensions


using SparseArrays, LinearAlgebra, IterativeSolvers, FillArrays, LazyArrays, ImageFiltering, FFTW

include("point_cloud_filtering.jl")
include("centerline_estimation.jl")
include("logcentric_coordinates.jl")
include("heightmap_generation.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("../ext/GLMakieExt.jl")
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDAExt.jl")
end
end

end
