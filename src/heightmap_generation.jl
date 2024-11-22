export calculate_heightmap, calculate_heightmap!, rescale_calculate_heightmap, rescale_calculate_heightmap!, segment_branches

function cuda_is_functional() 
    ext = Base.get_extension(LogHeightmaps, :CUDAExt)
    if isnothing(ext)
        return false
    else
        ext.cuda_is_functional()
    end
end

"""
    rescale_calculate_heightmap(logcentric_points, α, width, height, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional())
    
Calculate heightmap from the points in log-centric coordinate system. 
`logcentric_points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter. 
`min_l` and `max_l` are minimum and maximum values of l. If not provided explicitly, estimated from points. 
`real_length` is the estimated real length of long. If not provided explicitly, calculated as `max_l - min_l`.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

# Keywords
- `use_cuda=cuda_is_functional()`: it is possible to use CUDA to speed up the computation.
"""
rescale_calculate_heightmap(logcentric_points, α, width, height, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional()) = 
    rescale_calculate_heightmap!(zeros(height, width), logcentric_points, α, min_l, max_l, real_length; use_cuda=use_cuda)

"""
    rescale_calculate_heightmap(logcentric_points, α, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional())
    
In-place version of [`rescale_calculate_heightmap`](@ref) function.
Calculate heightmap from the points in log-centric coordinate system. 
`logcentric_points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter. 
`min_l` and `max_l` are minimum and maximum values of l. If not provided explicitly, estimated from points. 
`real_length` is the estimated real length of long. If not provided explicitly, calculated as `max_l - min_l`.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

# Keywords
- `use_cuda=cuda_is_functional()`: it is possible to use CUDA to speed up the computation.
"""
function rescale_calculate_heightmap!(heightmap, logcentric_points, α, min_l=nothing, max_l=nothing, real_length=nothing; use_cuda=cuda_is_functional())
    height, width = size(heightmap)
    rescaled_points, min_l, max_l = rescale_points(logcentric_points, width, height, min_l, max_l)
    calculate_heightmap!(heightmap, rescaled_points, α, real_length; use_cuda=use_cuda)
    return heightmap
end

"""
    calculate_heightmap(rescaled_points, α, width, height, real_length; use_cuda=true)
    
Calculate heightmap from the points in log-centric coordinate system. 
`rescaled_points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter. 
It is assumed that l is already rescaled to the `[1, height]` range, and θ is rescaled to `[1, width]`. 
`real_length` is the difference between minimum and maximum l values before rescaling. 

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

# Keywords
- `use_cuda=cuda_is_functional()`: it is possible to use CUDA to speed up the computation.
"""
calculate_heightmap(rescaled_points, α, width, height, real_length; use_cuda=cuda_is_functional()) = 
    calculate_heightmap!(zeros(height, width), rescaled_points, α, real_length; use_cuda=use_cuda)

function cg_cuda!(::Nothing, ::Nothing) end

function do_cg!(b, R; use_cuda=cuda_is_functional())
    if use_cuda
        cg_cuda!(b, R)
    else
        cg!(b, R, b)
    end
end


"""
    calculate_heightmap!(heightmap, rescaled_points, α, width, height, real_length; use_cuda=true)
    
In-place version of the [`calculate_heightmap`](@ref) function.
Calculate heightmap from the points in log-centric coordinate system. 
`rescaled_points` must be a 3×n `AbstractMatrix`, where the rows correspond to l, θ, ρ values respectively, α is a smoothing parameter. 
It is assumed that l is already rescaled to the `[1, height]` range, and θ is rescaled to `[1, width]`. 
`real_length` is the difference between minimum and maximum l values before rescaling. 

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

# Keywords
- `use_cuda=cuda_is_functional()`: it is possible to use CUDA to speed up the computation.
"""
function calculate_heightmap!(heightmap, rescaled_points, α, real_length; use_cuda=cuda_is_functional())
    height, width = size(heightmap)
    b = vec(heightmap)
    invΔθ = 360 / width
    invΔL = real_length / height
    reg = construct_regularizer(width, height, invΔθ, invΔL) #|> cu

    R = constructAb!(b, rescaled_points, width, height)# |> cu
    R .+= α*reg
    do_cg!(b, R; use_cuda=use_cuda)
    return heightmap
end

make_odd(x) = x + 1 - x % 2

normalize_from_extrema(x, min, max) = ((x - min) / (max - min))
normalize_from_extrema(x) = normalize_from_extrema.(x, extrema(x)...)

function gaussian_second_diff(σ, sz, dim)
    k = Kernel.gaussian((σ, σ), make_odd.(sz))
    x = axes(k)[dim]
    res = (-(-1 .+ x.^2 ./ σ^2) .* k ./ (2π * σ^4)) |> parent
    return res[1:sz[1], 1:sz[2]] 
end
gaussian_second_diff_y(σ, sz) = gaussian_second_diff(σ, sz, 1)
gaussian_second_diff_x(σ, sz) = gaussian_second_diff(σ, sz, 1)

gaussian_second_diff_weighted_sum(σy, σx, sz) = σy .* gaussian_second_diff_y(σy, sz) .+ σx .* gaussian_second_diff_x(σx, sz) 

"""
    segment_branches(I, branch_radius, real_size)
    
Segment branches from the heightmap image using Difference of Gaussians.
`branch_radius` specifies approximate radius of the branch on the surface in mm.
`real_size` is a tuple containing an estimate of the real size of the heightmap, usually calculated as `(real_height, 2π * mean(heightmap))`.

Refer to [zolotarev2020modelling](@cite), [zolotarev2022](@cite) for more details.

"""
function segment_branches(I, branch_radius, real_size)
    unpad(I) = I[size(I, 1)÷2+1:end, :]
    pad(I) = vcat(I[end:-1:1, :], I)
    σ = branch_radius .* size(I) ./ real_size
    Ip = pad(I)
    kernel = gaussian_second_diff_weighted_sum(σ..., size(Ip))
    return fft(Ip) .* fft(kernel) |> ifft |> ifftshift |> real |> unpad |> x->max.(x, 0) |> normalize_from_extrema
end


function construct_regularizer(width, height, invΔθ, invΔL)
    szN = width * height
    szL = szN-width
    szθ = szN-height

    valsL = fill(invΔL, 2szL)
    valsL[1:2:end] .= -invΔL
    
    valsθ = fill(invΔθ, 2szN)
    valsθ[1:2:2height] .= -invΔθ
    valsθ[2height+2:2:end] .= -invΔθ

    rowsL = Vector{Int32}(undef, 2szL)
    rowsL[1:2:end] .= 1:szL
    rowsL[2:2:end] .= 1:szL
    
    rowsθ = Vector{Int32}(undef, 2szN)
    rowsθ[1:2:2height] .= 1:height
    rowsθ[2:2:2height] .= szθ+1:szN
    rowsθ[2height+1:2:2szN] .= 1:szθ
    rowsθ[2height+2:2:2szN] .= height+1:szN
    
    colsL = Vector{Int32}(undef, szN+1)
    colsL[1] = 1
    colsL[2:szL+1] .= 2:2:2szL
    colsL[szL+2:end] .= 2szL+1

    colsθ = Vector{Int32}(undef, szN+1)
    colsθ[1:szN] .= 1:2:2szN
    colsθ[end] = 2szN+1

    @inbounds begin
        DL = SparseMatrixCSC(szN, szN, colsL, rowsL, valsL)
        Dθ = SparseMatrixCSC(szN, szN, colsθ, rowsθ, valsθ)
    end
    return transpose(Dθ) * Dθ .+ transpose(DL) * DL
end


function constructAb!(b, converted_points::AbstractMatrix{T}, width, height) where T
    N = size(converted_points, 2)
    M = height * width
    colptr = Vector{Int32}(undef, N+1)
    rowvec = Vector{Int32}(undef, 4N)
    nzvec = Vector{T}(undef, 4N)

    @inbounds begin
        x1 = @~ floor.(Int32, @view converted_points[2, :])
        y1 = @~ floor.(Int32, @view converted_points[1, :])
        
        x2 = @~ min.(x1 .+ 1, width)
        y2 = @~ min.(y1 .+ 1, height)
        wx = @~ converted_points[2, :] .- x1
        wy = @~ converted_points[1, :] .- y1

        colptr[1:N] .= 1:4:4N
        colptr[end] = 4N+1

        rowvec[1:4:end] .= y1.+(x1.-1).*height
        rowvec[2:4:end] .= y2.+(x1.-1).*height
        rowvec[3:4:end] .= y1.+(x2.-1).*height
        rowvec[4:4:end] .= y2.+(x2.-1).*height

        nzvec[1:4:end] .= wx.*wy
        nzvec[2:4:end] .= wx.*(1 .-wy)
        nzvec[3:4:end] .= (1 .-wx).*wy
        nzvec[4:4:end] .= (1 .-wx).*(1 .-wy)
        Aᵗ = SparseMatrixCSC(M, N, colptr, rowvec, nzvec)
    end
    dropzeros!(Aᵗ)
    AᵗA = Aᵗ * transpose(Aᵗ)
    @views mul!(b, Aᵗ, converted_points[3, :])
    
    return AᵗA
end

