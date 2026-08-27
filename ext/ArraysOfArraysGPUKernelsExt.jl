# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysGPUKernelsExt

using GPUArraysCore: AbstractGPUArray

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

using ArraysOfArrays: ArraysOfArrays, VectorOfArrays, innerlengths, innerreduce, NestedArrayStyle


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


# Vectors of arrays whose data lives on a device. The shape information may
# be host- or device-resident:
const _GPUVectorOfArrays{T,N,M} = VectorOfArrays{T,N,M,<:AbstractGPUArray}

# The N == 1 (vector-element) case. Only here does the flat segment view
# equal the logical element and does an element index reduce to a plain
# linear Int, so only these use the segment-view kernels for the
# shape-dependent operations (generic map, argmin/argmax):
const _GPUVectorOfVectors{T,M} = VectorOfArrays{T,1,M,<:AbstractGPUArray}


# Apply a whole-element function g to each element array. One work item per
# element constructs a view and calls g on it, so g must be compilable for
# the device (many Base reductions are, but those that throw on empty
# collections, like maximum, are not - those are routed to the segmented
# reduction kernels instead, see below):
@kernel function _segmented_map_kernel!(out, g, @Const(data), @Const(elem_ptr))
    i = @index(Global, Linear)
    j0 = elem_ptr[i]
    j1 = elem_ptr[i + 1] - 1
    @inbounds out[i] = g(view(data, j0:j1))
end

# Defer to the generic map (correct for host-resident shape info; a clean
# scalar-indexing error for device-resident shape info) instead of forcing a
# result through the segment-view kernel:
_segmented_map_fallback(g, A) = invoke(map, Tuple{Any, AbstractArray}, g, A)

function _segmented_map(g, A::_GPUVectorOfArrays)
    # The flat segment view only equals the logical element for vector
    # elements; for N > 1 it would drop the element shape, so defer:
    ndims(eltype(A)) == 1 || return _segmented_map_fallback(g, A)
    T_out = Base.promote_op(g, eltype(A))
    # Array results (or a type that did not infer to something concrete)
    # cannot go through the scalar-result kernel; defer to the generic map,
    # mirroring the scalar-result guard in the broadcast path below:
    (isconcretetype(T_out) && !(T_out <: AbstractArray)) || return _segmented_map_fallback(g, A)
    data = A.data
    backend = KA.get_backend(data)
    elem_ptr = _on_backend(backend, A.elem_ptr)
    n = length(A)
    out = KA.allocate(backend, T_out, n)
    n > 0 && _segmented_map_kernel!(backend)(out, g, data, elem_ptr; ndrange = n)
    return out
end


# Per-element argmin/argmax for vector elements. Base.argmin/argmax do not
# compile for the device (their empty-collection handling throws), so the
# index-tracking loop is written out. The update predicate mirrors Base's
# findmax/findmin reducers (isless for argmax, Base.isgreater for argmin), so
# ties keep the first extremum and NaN/-0.0 order exactly like Base:
@kernel function _seg_argextreme_kernel!(out, better, @Const(data), @Const(elem_ptr))
    i = @index(Global, Linear)
    j0 = elem_ptr[i]
    j1 = elem_ptr[i + 1] - 1
    @inbounds best = data[j0]
    bestidx = 1
    for j in (j0 + 1):j1
        @inbounds v = data[j]
        if better(best, v)
            best = v
            bestidx = j - j0 + 1
        end
    end
    @inbounds out[i] = bestidx
end

function _segmented_argextreme(better, A::_GPUVectorOfVectors)
    any(iszero, innerlengths(A)) && throw(ArgumentError("Cannot take argmin/argmax of empty element arrays"))
    data = A.data
    backend = KA.get_backend(data)
    elem_ptr = _on_backend(backend, A.elem_ptr)
    n = length(A)
    out = KA.allocate(backend, Int, n)
    n > 0 && _seg_argextreme_kernel!(backend)(out, better, data, elem_ptr; ndrange = n)
    return out
end


# map over the element arrays of a device-resident vector of arrays, without
# host-side scalar indexing. The general path uses the segment-view kernel
# only for vector elements with a scalar result and defers everything else to
# the generic map (see _segmented_map); shape-insensitive reductions are
# routed to the segmented reduction kernels and stay valid for any element
# dimensionality:
Base.map(g, A::_GPUVectorOfArrays) = _segmented_map(g, A)
# identity returns the element arrays, not scalars, so it keeps the generic
# outer-copy behavior (disambiguation against map(::typeof(identity), ...)):
Base.map(f::typeof(identity), A::_GPUVectorOfArrays) = invoke(map, Tuple{typeof(identity), VectorOfArrays}, f, A)
# maximum/minimum are shape-insensitive and keep Base's element type, so they
# use the chunked segmented reduction for any element dimensionality. sum is
# deliberately not routed here: Base's sum widens small integers via add_sum
# (e.g. sum(Int8[100, 28]) == 128::Int), which innersum's plain + does not, so
# sum is left to the general segment-view kernel (N == 1, which reproduces
# Base's sum) and the generic map fallback (N > 1):
Base.map(::typeof(maximum), A::_GPUVectorOfArrays) = innerreduce(max, A)
Base.map(::typeof(minimum), A::_GPUVectorOfArrays) = innerreduce(min, A)
# argmin/argmax return an index into the element, which is a plain linear Int
# only for vector elements; higher-dimensional elements would need Cartesian
# indices, so these are restricted to N == 1 and larger element
# dimensionalities fall back to the generic map (see _segmented_map):
Base.map(::typeof(argmin), A::_GPUVectorOfVectors) = _segmented_argextreme(Base.isgreater, A)
Base.map(::typeof(argmax), A::_GPUVectorOfVectors) = _segmented_argextreme(isless, A)


# Single-argument outer broadcast g.(A) with a scalar result over a
# device-resident vector of arrays routes to the same kernel; array
# results and fused/multi-argument broadcasts keep the generic behavior:
function Base.copy(bc::Base.Broadcast.Broadcasted{NestedArrayStyle{1}, <:Any, F, <:Tuple{<:_GPUVectorOfArrays}}) where {F}
    A = bc.args[1]
    ElType = Base.Broadcast.combine_eltypes(bc.f, (A,))
    if isconcretetype(ElType) && !(ElType <: AbstractArray)
        return map(bc.f, A)
    else
        return Base.copy(Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}}(bc.f, bc.args, bc.axes))
    end
end


end # module ArraysOfArraysGPUKernelsExt
