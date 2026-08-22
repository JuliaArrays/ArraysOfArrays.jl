# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysGPUKernelsExt

using GPUArraysCore: AbstractGPUArray

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

using ArraysOfArrays: ArraysOfArrays, VectorOfArrays, innerlengths


@kernel function _segmented_mapreduce_kernel!(out, f, op, @Const(data), @Const(elem_ptr), init)
    i = @index(Global, Linear)
    j0 = elem_ptr[i]
    j1 = elem_ptr[i + 1] - 1
    if init isa ArraysOfArrays._NoInit
        # Empty element arrays have been excluded beforehand:
        acc = f(data[j0])
        for j in (j0 + 1):j1
            acc = op(acc, f(data[j]))
        end
    else
        acc = init
        for j in j0:j1
            acc = op(acc, f(data[j]))
        end
    end
    out[i] = acc
end


# Element pointers of the chunks that long element arrays are split into,
# one work item per chunk:
@kernel function _chunk_ptrs_kernel!(chunk_ptr, @Const(elem_ptr), @Const(offsets), chunklen)
    i = @index(Global, Linear)
    # offsets[k] is the number of chunks of the element arrays before k:
    k = searchsortedlast(offsets, i - 1)
    chunk_ptr[i] = elem_ptr[k] + (i - offsets[k] - 1) * chunklen
end


function _on_backend(backend, x::AbstractArray)
    KA.get_backend(x) == backend && return x
    y = KA.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end


# Each work item reduces one element array, so a single long element array
# would serialize the whole reduction. Long element arrays are therefore
# split into chunks of at most this many elements, which are reduced
# separately:
const _max_chunk_length = 256

# Splitting costs an additional pass over the chunk results, so it only
# pays off if the element arrays are several chunks long:
const _min_length_for_chunking = 4 * _max_chunk_length

function _segmented_pass(f, op, init, data, elem_ptr, ::Type{T_out}, backend) where {T_out}
    n = length(elem_ptr) - 1
    out = KA.allocate(backend, T_out, n)
    if n > 0
        # Stream-ordered like GPU array operations, no synchronization here:
        kernel! = _segmented_mapreduce_kernel!(backend)
        kernel!(out, f, op, data, elem_ptr, init; ndrange = n)
    end
    return out
end


function _segmented_mapreduce(f, op, init, data, elem_ptr, ::Type{T_out}, backend) where {T_out}
    n = length(elem_ptr) - 1
    # Whether to split is decided from the mean element array length, which
    # is known on the host: reading the longest length from device data would
    # force a synchronization on every reduction. So a single very long
    # element array among many short ones is not split:
    if n == 0 || length(data) <= n * _min_length_for_chunking
        return _segmented_pass(f, op, init, data, elem_ptr, T_out, backend)
    end

    # Split the element arrays into chunks of at most _max_chunk_length
    # elements. The chunks of an element array are consecutive, so their
    # results form the element arrays of a shorter partition, which is then
    # reduced in turn.
    #
    # Only a few element arrays can be longer than the chunk length on
    # average, so their chunk counts are accumulated on the host. This also
    # keeps the implementation independent of device-side scan support:
    ep_host = Array(elem_ptr)
    counts = cld.(diff(ep_host), _max_chunk_length)
    offsets_host = pushfirst!(cumsum(counts), zero(eltype(counts)))
    n_chunks = last(offsets_host)
    # The decision above uses the full backing data, which may cover more
    # than the elements do (e.g. for views), so it can turn out that no
    # element array is actually longer than a chunk:
    n_chunks == n && return _segmented_pass(f, op, init, data, elem_ptr, T_out, backend)
    offsets = _on_backend(backend, offsets_host)

    chunk_ptr = similar(elem_ptr, n_chunks + 1)
    _chunk_ptrs_kernel!(backend)(chunk_ptr, elem_ptr, offsets, _max_chunk_length; ndrange = n_chunks)
    copyto!(view(chunk_ptr, (n_chunks + 1):(n_chunks + 1)), view(elem_ptr, (n + 1):(n + 1)))

    # Chunks are never empty, so the initial value must not be applied
    # here, only when the chunk results are combined:
    partials = _segmented_pass(f, op, ArraysOfArrays._NoInit(), data, chunk_ptr, T_out, backend)
    return _segmented_mapreduce(identity, op, init, partials, offsets .+ 1, T_out, backend)
end


# Segmented reduction over the flat data of GPU-resident vectors of arrays:
function ArraysOfArrays._innermapreduce_impl(f, op, init, A::VectorOfArrays{T,N,M,<:AbstractGPUArray}) where {T,N,M}
    data = A.data
    backend = KA.get_backend(data)
    elem_ptr = _on_backend(backend, A.elem_ptr)

    if init isa ArraysOfArrays._NoInit && any(iszero, innerlengths(A))
        throw(ArgumentError("Reducing over empty element arrays requires an init value"))
    end

    T_f = Base.promote_op(f, T)
    T_init = init isa ArraysOfArrays._NoInit ? T_f : typeof(init)
    T_out = promote_type(T_f, Base.promote_op(op, T_init, T_f))

    return _segmented_mapreduce(f, op, init, data, elem_ptr, T_out, backend)
end

end # module ArraysOfArraysGPUKernelsExt
