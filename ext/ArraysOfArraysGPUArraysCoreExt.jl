# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray, AnyGPUArray, @allowscalar

import ArraysOfArrays

# Two O(1) scalar reads during construction are acceptable:
ArraysOfArrays._scalar_first_last(x::AbstractGPUArray) = @allowscalar (first(x), last(x))

# Per-element access on device-resident data costs one device operation
# each, so non-scalar getindex indexes the flat data in a single operation:
ArraysOfArrays._prefers_flat_getindex(::Type{<:AbstractGPUArray}) = true


# Vectorized shape-info comparison, to avoid scalar indexing of GPU
# arrays. The shape vectors may reside on the host and device in any
# combination:
function ArraysOfArrays._shapeinfo_equal(a::AbstractGPUArray, b::AbstractVector)
    axes(a) == axes(b) || return false
    b_dev = copyto!(similar(a, eltype(b)), b)
    mapreduce(==, &, a, b_dev, init = true)
end

function ArraysOfArrays._shapeinfo_equal(a::AbstractVector, b::AbstractGPUArray)
    axes(a) == axes(b) || return false
    a_dev = copyto!(similar(b, eltype(a)), a)
    mapreduce(==, &, a_dev, b, init = true)
end

ArraysOfArrays._shapeinfo_equal(a::AbstractGPUArray, b::AbstractGPUArray) =
    a === b || (axes(a) == axes(b) && mapreduce(==, &, a, b, init = true))


function ArraysOfArrays._elem_lengths_equal(a::AbstractGPUArray{<:Integer}, b::AbstractGPUArray{<:Integer})
    length(a) == length(b) || return false
    length(a) < 2 && return true
    @views all((a[(begin + 1):end] .- a[begin:(end - 1)]) .== (b[(begin + 1):end] .- b[begin:(end - 1)]))
end

function ArraysOfArrays._elem_lengths_equal(a::AbstractGPUArray{<:Integer}, b::AbstractVector{<:Integer})
    length(a) == length(b) || return false
    ArraysOfArrays._elem_lengths_equal(a, copyto!(similar(a, eltype(b)), b))
end

function ArraysOfArrays._elem_lengths_equal(a::AbstractVector{<:Integer}, b::AbstractGPUArray{<:Integer})
    length(a) == length(b) || return false
    ArraysOfArrays._elem_lengths_equal(copyto!(similar(b, eltype(a)), a), b)
end

# Data may be a device array or a view of one:
ArraysOfArrays._vecdata_equal(a::AnyGPUArray{<:Any,1}, b::AnyGPUArray{<:Any,1}) =
    axes(a) == axes(b) && mapreduce(==, &, a, b, init = true)

ArraysOfArrays._vecdata_isequal(a::AnyGPUArray{<:Any,1}, b::AnyGPUArray{<:Any,1}) =
    axes(a) == axes(b) && mapreduce(isequal, &, a, b, init = true)

# Vectorized formulation of the O(n) partition checks, to avoid scalar
# indexing of GPU arrays. The shape vectors may also reside on the host and
# device in any combination:
function ArraysOfArrays._partition_sizes_valid(elem_ptr::AbstractGPUArray{<:Integer}, kernel_size::AbstractVector)
    _gpu_partition_sizes_valid(elem_ptr, kernel_size)
end

function ArraysOfArrays._partition_sizes_valid(elem_ptr::AbstractVector{<:Integer}, kernel_size::AbstractGPUArray)
    ep_dev = copyto!(similar(kernel_size, eltype(elem_ptr), size(elem_ptr)), elem_ptr)
    _gpu_partition_sizes_valid(ep_dev, kernel_size)
end

# Disambiguation:
function ArraysOfArrays._partition_sizes_valid(elem_ptr::AbstractGPUArray{<:Integer}, kernel_size::AbstractGPUArray)
    _gpu_partition_sizes_valid(elem_ptr, kernel_size)
end

function _gpu_partition_sizes_valid(elem_ptr::AbstractGPUArray{<:Integer}, kernel_size::AbstractVector)
    ep_lo = view(elem_ptr, firstindex(elem_ptr):(lastindex(elem_ptr) - 1))
    ep_hi = view(elem_ptr, (firstindex(elem_ptr) + 1):lastindex(elem_ptr))
    len = ep_hi .- ep_lo

    klen = prod.(kernel_size)
    # kernel_size may still live on the host:
    klen_dev = klen isa AbstractGPUArray ? klen : copyto!(similar(elem_ptr, eltype(klen), size(klen)), klen)

    # The pointers are compared, the sign of their difference would wrap for
    # narrow pointer types. klen == 0 requires len == 0; the max guard keeps
    # the untaken mod branch free of division by zero:
    valid = (ep_hi .>= ep_lo) .& ifelse.(
        klen_dev .== 0,
        len .== 0,
        mod.(len, max.(klen_dev, one(eltype(klen_dev)))) .== 0
    )
    return all(valid)
end

end # module ArraysOfArraysGPUArraysCoreExt
