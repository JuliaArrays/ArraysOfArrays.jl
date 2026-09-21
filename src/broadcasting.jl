
# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

const _RefLike{T} = Union{Tuple{T}, Ref{T}}


"""
    abstract type ArraysOfArrays.AbstractNestedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N}

Supertype of the broadcast styles of nested array types like
[`VectorOfArrays`](@ref) and [`ArrayOfSimilarArrays`](@ref).

Packages that define array types with their own broadcast style can
resolve style combination by specializing
`Base.Broadcast.BroadcastStyle(::AbstractNestedArrayStyle{N}, ::TheirStyle)`.
Dispatch on this supertype, not on its subtypes.
"""
abstract type AbstractNestedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
@compat public AbstractNestedArrayStyle


"""
    ArraysOfArrays.NestedArrayStyle{N}()

Broadcast style of nested array types like [`VectorOfArrays`](@ref) and
[`ArrayOfSimilarArrays`](@ref).

Broadcasts at this level apply `f` to whole element arrays, as in
`(x -> 2 .* x).(A)`. Results whose data is stored in an `Array` are
collected into a nested array that shares no data with `A`; this includes
views of the element arrays, so `(x -> x).(A)` and `identity.(A)` copy
them. Other results go into a plain `Array`, as in Base. Only that the
result is an `AbstractArray` of the values of `f` is guaranteed: the
nested array type (currently a [`VectorOfArrays`](@ref), whose elements
may be ragged even if `A` is an [`ArrayOfSimilarArrays`](@ref)) is an
implementation detail that may change between minor versions.
`convert(VectorOfSimilarArrays, result)` turns results of equal size into
an [`ArrayOfSimilarArrays`](@ref) without copying. Use [`bcastat`](@ref)
to broadcast over the *contents* of the element arrays instead.

# Extended help

A broadcast is packed into a nested array if it runs over a single outer
dimension with `Base.OneTo` axes and its result type is inferred as a
concrete array type with at least one dimension whose data is stored in an
`Array` (or `Memory`): such an array itself, or a view, reshape or
reinterpretation of one. Structured, static, bit, offset and GPU arrays
would have to be densified or transferred to the host, so they, like all
other results, keep the default broadcast behavior. So do the element
views of nested arrays backed by other storage types (e.g. `ElasticArray`),
which are therefore not copied.

Since packed results are copies, in-place operations on the elements like
`fill!.(A, 0)` copy all data into a discarded result; use `foreach` or
`map` for those. Results that a nested array cannot represent (offset
axes, or an empty leading but non-empty trailing dimension) throw an
`ArgumentError`; use `map` for those as well.

With StructArrays loaded, broadcasts over a `StructArray` with nested-array
columns return a `StructArray` for struct-valued results and follow the
rules above otherwise. The original `StructArray` is indexed, so
specialized `getindex` methods of it are used.

With DiskArrays loaded, broadcasts over nested arrays with disk-backed
data, also combined with other disk arrays and in place, are evaluated in
blocks along the outer axis, each block read from disk at once. The
results are in memory, `f` receives in-memory element arrays. `map` and
iteration access the elements one by one, so broadcast over such data.

# Implementation

Packages that define array types with their own broadcast style resolve
style combination via [`ArraysOfArrays.AbstractNestedArrayStyle`](@ref).
"""
struct NestedArrayStyle{N} <: AbstractNestedArrayStyle{N} end
@compat public NestedArrayStyle

NestedArrayStyle{M}(::Val{N}) where {M,N} = NestedArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{<:AbstractArrayOfSimilarArrays{<:Any,<:Any,N}}) where {N} = NestedArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:VectorOfArrays}) = NestedArrayStyle{1}()

function Base.copy(bc::Broadcast.Broadcasted{NestedArrayStyle{N}}) where {N}
    blocked = _copy_in_blocks(bc)
    blocked === nothing || return blocked
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if N == 1 && _packable_result(ElType) && axes(bc, 1) isa Base.OneTo
        return _collect_nested(bc, ElType)
    else
        # Everything else behaves like the default broadcast machinery:
        return copy(convert(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, bc))
    end
end

# Evaluation in blocks along the outer axis, for arguments with expensive
# element access but cheap bulk access, like nested arrays over disk-backed
# data (see the DiskArrays extension). Such arguments request a block length
# and are read block by block via getindex, other arrays are sliced as
# views. Each block is evaluated with the original style, so the packing
# rules apply per block.

_block_length(x) = nothing
_block_length(bc::Broadcast.Broadcasted) = _bcast_blocklength(bc.args)

_bcast_blocklength(args::Tuple) = Base.afoldl((len, x) -> _min_blocklength(len, _block_length(x)), nothing, args...)

_min_blocklength(::Nothing, ::Nothing) = nothing
_min_blocklength(::Nothing, b::Integer) = b
_min_blocklength(a::Integer, ::Nothing) = a
_min_blocklength(a::Integer, b::Integer) = min(a, b)

# The flattened broadcast and its blocks, nothing if it is not to be
# evaluated in blocks. Fused expressions are flattened, so that all
# arguments are leaves. The block length is the smallest one requested by
# the arguments that span the outer axis, arguments that broadcast along
# it are read once per block:
function _bcast_blocks(bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{1}})
    ax = only(axes(bc))
    (isempty(ax) || !(ax isa Base.OneTo) || _bcast_blocklength(bc.args) === nothing) && return nothing
    fbc = Broadcast.flatten(bc)
    blen = _bcast_blocklength(map(a -> _spans(a, ax) ? a : nothing, fbc.args))
    return fbc, Iterators.partition(ax, something(blen, length(ax)))
end
_bcast_blocks(::Broadcast.Broadcasted) = nothing

function _copy_in_blocks(bc::Broadcast.Broadcasted)
    blocked = _bcast_blocks(bc)
    blocked === nothing && return nothing
    fbc, blocks = blocked
    results = map(r -> Broadcast.materialize(_block_bcast(fbc, r)), blocks)
    # A single block is not copied again:
    return length(results) == 1 ? only(results) : reduce(vcat, results)
end

function _copyto_in_blocks!(dest::AbstractArray, bc::Broadcast.Broadcasted)
    blocked = _bcast_blocks(bc)
    blocked === nothing && return nothing
    fbc, blocks = blocked
    foreach(r -> copyto!(view(dest, r), _block_bcast(fbc, r)), blocks)
    return dest
end

function _block_bcast(fbc::Broadcast.Broadcasted{Style}, r::AbstractUnitRange) where {Style}
    ax = only(axes(fbc))
    args = map(a -> _block_arg(a, r, ax), fbc.args)
    _bcast_blocklength(args) === nothing || throw(ArgumentError("Block arguments must not request blocks themselves"))
    return Broadcast.Broadcasted{Style}(fbc.f, args)
end

_spans(x, ax) = false
_spans(a::AbstractArray, ax) = axes(a, 1) == ax
_spans(t::Tuple, ax) = length(t) == length(ax)

# Arguments that request blocks are read in any case, so that they no
# longer do, those that broadcast along the outer axis in full:
_block_arg(x, r, ax) = x
_block_arg(t::Tuple, r, ax) = _spans(t, ax) ? t[r] : t
function _block_arg(a::AbstractArray, r, ax)
    if _block_length(a) === nothing
        return _spans(a, ax) ? view(a, r) : a
    else
        return a[_spans(a, ax) ? UnitRange(r) : UnitRange(axes(a, 1))]
    end
end

function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{NestedArrayStyle{1}})
    blocked = _copyto_in_blocks!(dest, bc)
    blocked === nothing || return blocked
    return invoke(copyto!, Tuple{AbstractArray,Broadcast.Broadcasted}, dest, bc)
end

# The nested styles specialize only copy; StructArrays allocates the columns
# of struct-valued results via similar (see the StructArrays extension):
Base.similar(bc::Broadcast.Broadcasted{<:AbstractNestedArrayStyle{N}}, ::Type{T}, dims) where {N,T} =
    similar(convert(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, bc), T, dims)

# Results that are packed into a nested array: concrete arrays whose data
# is stored in an Array (or Memory), i.e. Arrays and views, reshapes and
# reinterpretations of them. Unknown array types are not packed
# (fail-closed): structured, lazy, static, bit and offset arrays would have
# to be densified, and device arrays, which include DenseArray subtypes
# outside of GPUArraysCore (e.g. Reactant), must not be copied to the host
# element by element. Extensions may opt in dense host array types, but
# must recurse into the type of the underlying memory if that is a
# parameter (see the FixedSizeArrays extension):
function _packable_result(::Type{ET}) where {ET}
    ET <: AbstractArray && isconcretetype(ET) && ndims(ET) >= 1 && _host_storage(ET)
end

_host_storage(::Type) = false
_host_storage(::Type{<:Array}) = true
# Memory is host memory by its address-space parameter (device memory
# would be a different GenericMemory):
@static if isdefined(Base, :Memory)
    _host_storage(::Type{<:Memory}) = true
end
# Wrappers keep their data in their parent:
_host_storage(::Type{<:SubArray{<:Any,<:Any,P}}) where {P} = _host_storage(P)
_host_storage(::Type{<:Base.ReshapedArray{<:Any,<:Any,P}}) where {P} = _host_storage(P)
_host_storage(::Type{<:Base.ReinterpretArray{<:Any,<:Any,<:Any,P}}) where {P} = _host_storage(P)
# Resolves the method ambiguity at the bottom type:
_host_storage(::Type{Union{}}) = false

function _collect_nested(bc::Broadcast.Broadcasted, ::Type{<:AbstractArray{T,M}}) where {T,M}
    n = length(axes(bc, 1))
    dest = VectorOfArrays{T,M}()
    sizehint!(dest.elem_ptr, n + 1)
    sizehint!(dest.kernel_size, n)
    for i in eachindex(bc)
        x = bc[i]
        _require_packable(x)
        # The size of the first result is a good guess for the others:
        isempty(dest) && sizehint!(dest.data, n * length(x))
        push!(dest, x)
    end
    return dest
end

# A VectorOfArrays stores only the leading dimensions of its elements and
# reads them back with one-based axes, so it cannot represent these:
function _require_packable(x::AbstractArray)
    Base.has_offset_axes(x) && throw(ArgumentError(
        "Cannot pack arrays with offset axes into a nested array, use map instead of broadcast"
    ))
    sz = size(x)
    prod(Base.front(sz)) == 0 && last(sz) != 0 && throw(ArgumentError(
        "Cannot pack arrays with an empty leading and a non-empty trailing dimension into a nested array, use map instead of broadcast"
    ))
    return nothing
end


_idx_type(::VectorOfSimilarArrays) = Int
_idx_type(A::VectorOfArrays) = eltype(A.elem_ptr)

_similar_idx_vector(A::VectorOfSimilarArrays, ::Type{T}, n::Integer) where T = similar(A.data, T, n)
_similar_idx_vector(A::VectorOfArrays, ::Type{T}, n::Integer) where T = similar(A.elem_ptr, T, n)

function _new_vector_of_arrays_with_lengths(
    A::Union{VectorOfSimilarArrays, VectorOfArrays}, ::Type{T},
    new_kernel_size::AbstractArray{<:Tuple{Vararg{Integer,M}}},
    new_lengths::AbstractVector{<:Integer}
) where {T,M}
    new_data = similar(A.data, T, sum(new_lengths))

    new_elem_ptr = _similar_idx_vector(A, _idx_type(A), length(new_lengths) + 1)
    _require_elem_ptr_range(new_elem_ptr, firstindex(new_data) + length(new_data))
    _elem_ptr_cumsum!(new_elem_ptr, new_data, new_lengths)

    newA = VectorOfArrays(new_data, new_elem_ptr, new_kernel_size, no_consistency_checks)
    return newA
end


# A view on the result of view, using an array of indices, would allocate due
# to reindexing, so call SubArray directly:
_noreindex_view(A, idxs...) = SubArray(A, idxs)
_noreindex_view(A::AbstractArray{T,N}, ::Vararg{Colon,N}) where {T,N} = A

# SubArray is constructed directly and indexed with `@inbounds`, so indices
# (e.g. logical masks) must be converted and bounds-checked here:
function _elem_view(a, idxs...)
    new_idxs = _to_indices(a, idxs)
    _noreindex_view(a, Base.ensure_indexable(new_idxs)...)
end

_generic_size(A) = size(A)
_generic_size(tpl::Tuple) = (length(tpl),)

Base.@propagate_inbounds function _to_indices(A, idxs)
    new_idxs = Base.to_indices(A, idxs)
    @boundscheck Base.checkbounds(A, new_idxs...)
    return new_idxs
end

# Limited to vectors of vectors for now.
# ToDo: Extend to vectors of arrays.
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::Union{VectorOfSimilarVectors, PartsView},
    Idxs::Union{AbstractVector{<:AbstractVector{<:Integer}},AbstractVector{Colon},_RefLike{<:Union{AbstractVector{<:Integer},Colon}}}...
)
    return _bcast_getindex_impl(A, Idxs...)
end

function _bcast_getindex_impl(A, Idxs...)
    # Checks size compatibility:
    bcsz = Base.Broadcast.broadcast_shape(size(A), map(_generic_size, Idxs)...)

    new_sizes = _similar_idx_vector(A, NTuple{length(Idxs),_idx_type(A)}, prod(bcsz))
    broadcast!(new_sizes, A, Idxs...) do a, idxs...
        map(length, _to_indices(a, idxs))
    end

    new_lengths = prod.(new_sizes)
    new_kernel_size = Base.tail.(new_sizes)

    T = eltype(A.data)
    newA = _new_vector_of_arrays_with_lengths(A, T, new_kernel_size, new_lengths)
    newA .= _elem_view.(A, Idxs...)
    return newA
end


# Fast path: equal-length index vectors select the same number of entries
# from each element, so the result is a VectorOfSimilarVectors again:
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::VectorOfSimilarVectors,
    Idx::VectorOfSimilarVectors{<:Integer}
)
    # Checks size compatibility:
    bcsz = Base.Broadcast.broadcast_shape(size(A), size(Idx))
    checkindex(Bool, axes(A.data, 1), Idx.data) || throw(BoundsError(A, (Idx,)))

    sz_inner = only(innersize(Idx))
    new_data = similar(A.data, (sz_inner, bcsz...))
    newA = VectorOfSimilarVectors(new_data)

    newA .= _noreindex_view.(A, Idx)
    return newA
end

# Logical masks require to_indices semantics, use the general implementation:
Base.Broadcast.broadcasted(::typeof(getindex), A::VectorOfSimilarVectors, Idx::VectorOfSimilarVectors{Bool}) =
    _bcast_getindex_impl(A, Idx)


# Fast path: a single index vector shared by all elements selects a
# rectangular region of the underlying data:
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::VectorOfSimilarVectors,
    Idx::_RefLike{<:Union{AbstractVector{<:Integer},Colon}}
)
    new_data = A.data[only(Idx), :]
    return VectorOfSimilarVectors(new_data)
end



Base.@propagate_inbounds function _findall!(B::AbstractVector, A::AbstractVector{Bool})
    @boundscheck let n::Int = 0
        @inbounds for i in eachindex(A)
            n += A[i] ? 1 : 0
        end
        n == length(B) || throw(ArgumentError("_findall! requires output array of correct size"))
    end

    i_B::Int = firstindex(B)
    #@inbounds
    for i_A in eachindex(A)
        if A[i_A]
            B[i_B] = i_A
            i_B += 1
        end
    end
    return B
end

function Base.Broadcast.broadcasted(
    ::typeof(findall),
    A::Union{VectorOfSimilarVectors{Bool}, PartsView{Bool}}
)
    new_lengths = _similar_idx_vector(A, _idx_type(A), length(A))
    new_lengths .= sum.(A)

    new_kernel_size = map(_ -> (), new_lengths)

    newA = _new_vector_of_arrays_with_lengths(A, Int, new_kernel_size, new_lengths)
    # foreach, not broadcast: the results would be packed into a discarded copy
    foreach(_findall!, newA, A)
    return newA
end
