module CUDAExt

# __init__() = println("CUDA extension was loaded!")

# Load main package and triggers
using CUDA, CUDA.CUSPARSE
using SparseArrays

using LogHeightmaps

# Extend functionality in main package with types from the triggers
cuda_is_functional() = CUDA.functional()


sparse_to_cuda(x::SparseMatrixCSC{Tv, Ti}) where {Tv,Ti<:Integer} = CuSparseMatrixCSC{Tv, Ti}(CuArray(x.colptr), CuArray(x.rowval), CuArray(x.nzval), (x.m, x.n))

function LogHeightmaps.cg_cuda!(b, R)
    cub = CuArray(b)
    cuR = sparse_to_cuda(R)
    LogHeightmaps.cg!(cub, cuR, cub)
    copyto!(b, cub)
end

end


