# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysDiskArraysExt

using DiskArrays: DiskArrays, AbstractDiskArray, ChunkStyle

import ArraysOfArrays
using ArraysOfArrays: ArrayOfSimilarArrays, VectorOfArrays, AbstractNestedArrayStyle, innersize

# Every element access on disk-backed data is a separate disk read, so
# non-scalar getindex of nested views of such data reads the whole
# selection from the flat data in a single block read instead:
ArraysOfArrays._prefers_flat_getindex(::Type{<:AbstractDiskArray}) = true

# DiskArrays cannot index empty arrays, which have no chunks:
ArraysOfArrays._flat_getindex(data::AbstractDiskArray, idxs...) =
    isempty(data) ? Array{eltype(data)}(undef, size(view(data, idxs...))) : data[idxs...]

# Likewise, stacking a disk-backed VectorOfArrays reads the covered data in
# a single ranged getindex (disk arrays also do not support the reshape of
# a lazy view that the in-memory implementation uses):
ArraysOfArrays._covered_data(data::AbstractDiskArray{<:Any,1}, r::AbstractUnitRange{Int}) =
    ArraysOfArrays._flat_getindex(data, r)

# Broadcasts over nested arrays with disk-backed data, and over disk arrays
# combined with nested arrays, are evaluated in blocks along the outer axis
# (see NestedArrayStyle). Blocks aim at the default chunk size of DiskArrays
# in memory and, where the flat data is sliced directly, at multiples of
# the chunk length:
Base.Broadcast.BroadcastStyle(s::AbstractNestedArrayStyle{N}, ::ChunkStyle{M}) where {N,M} =
    typeof(s)(Val(max(N, M)))

# DiskArrays cannot chunk empty arrays, so these are read in a single block
# (see _flat_getindex):
function _flat_block_length(data::AbstractDiskArray, bytes_per_elem::Integer)
    isempty(data) && return typemax(Int)
    chunk_len = last(DiskArrays.approx_chunksize(DiskArrays.eachchunk(data)))
    budget = max(1, (DiskArrays.default_chunk_size[] * 10^6) ÷ max(1, bytes_per_elem))
    return chunk_len * max(1, budget ÷ chunk_len)
end

ArraysOfArrays._block_length(a::AbstractDiskArray{<:Any,1}) =
    _flat_block_length(a, DiskArrays.element_size(a))

# Zero-dimensional disk arrays broadcast along the outer axis. They are read
# once per block and do not limit its length:
ArraysOfArrays._block_length(::AbstractDiskArray{<:Any,0}) = typemax(Int)

ArraysOfArrays._block_length(A::ArrayOfSimilarArrays{<:Any,<:Any,1,<:AbstractDiskArray}) =
    _flat_block_length(A.data, prod(innersize(A)) * DiskArrays.element_size(A.data))

# Blocks are nested arrays over in-memory arrays, with their element types:
ArraysOfArrays._block_eltype(::ArrayOfSimilarArrays{T,M,1,<:AbstractDiskArray}) where {T,M} =
    ArraysOfArrays._aosa_eltype(Array{T,M+1}, Val(M), Val(1))
ArraysOfArrays._block_eltype(::VectorOfArrays{T,N,<:Any,<:AbstractDiskArray,VI}) where {T,N,VI} =
    ArraysOfArrays._voa_eltype(Vector{T}, Vector{eltype(VI)}, Val(N))

# The flat block length is divided by the mean element length, so the
# blocks are not aligned to the chunks:
function ArraysOfArrays._block_length(A::VectorOfArrays{<:Any,<:Any,<:Any,<:AbstractDiskArray})
    flat_len = _flat_block_length(A.data, DiskArrays.element_size(A.data))
    ep_first, ep_last = ArraysOfArrays._scalar_first_last(A.elem_ptr)
    mean_len = max(1, (Int(ep_last) - Int(ep_first)) ÷ max(1, length(A)))
    return max(1, flat_len ÷ mean_len)
end

end # module ArraysOfArraysDiskArraysExt
