# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray, @allowscalar

import ArraysOfArrays

# Two O(1) scalar reads during construction are acceptable:
ArraysOfArrays._scalar_first_last(x::AbstractGPUArray) = @allowscalar (first(x), last(x))

# Vectorized formulation of the O(n) partition checks, to avoid scalar
# indexing of GPU arrays:
function ArraysOfArrays._partition_sizes_valid(elem_ptr::AbstractGPUArray{<:Integer}, kernel_size::AbstractVector)
    ep_lo = view(elem_ptr, firstindex(elem_ptr):(lastindex(elem_ptr) - 1))
    ep_hi = view(elem_ptr, (firstindex(elem_ptr) + 1):lastindex(elem_ptr))
    len = ep_hi .- ep_lo

    klen = prod.(kernel_size)
    # kernel_size may still live on the host:
    klen_dev = klen isa AbstractGPUArray ? klen : copyto!(similar(elem_ptr, eltype(klen), size(klen)), klen)

    # klen == 0 requires len == 0; the max guard keeps the untaken mod
    # branch free of division by zero:
    valid = (len .>= 0) .& ifelse.(
        klen_dev .== 0,
        len .== 0,
        mod.(len, max.(klen_dev, one(eltype(klen_dev)))) .== 0
    )
    return all(valid)
end

end # module ArraysOfArraysGPUArraysCoreExt
