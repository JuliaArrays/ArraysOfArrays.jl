
# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

const _RefLike{T} = Union{Tuple{T}, Ref{T}}


"""
    ArraysOfArrays.NestedArrayStyle{N}()

Broadcast style of nested array types like [`VectorOfArrays`](@ref) and
[`ArrayOfSimilarArrays`](@ref).

Broadcasts over a single outer dimension that produce array-valued results
(`f` receives whole element arrays and returns arrays, as in
`(x -> 2 .* x).(A)`) return a [`VectorOfArrays`](@ref) instead of a `Vector`
of arrays, where possible. All other broadcasts behave like the default
broadcast machinery. Use [`bcastat`](@ref) to broadcast over the *contents*
of the element arrays instead.
"""
struct NestedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

NestedArrayStyle{M}(::Val{N}) where {M,N} = NestedArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{<:AbstractArrayOfSimilarArrays{<:Any,<:Any,N}}) where {N} = NestedArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:VectorOfArrays}) = NestedArrayStyle{1}()

function Base.copy(bc::Broadcast.Broadcasted{NestedArrayStyle{N}}) where {N}
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if N == 1 && ElType <: Array && isconcretetype(ElType) && axes(bc, 1) isa Base.OneTo
        return _collect_nested(bc, ElType)
    else
        # Everything else behaves like the default broadcast machinery:
        return copy(convert(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, bc))
    end
end

function _collect_nested(bc::Broadcast.Broadcasted, ::Type{Array{T,M}}) where {T,M}
    dest = VectorOfArrays{T,M}()
    sizehint!(dest.elem_ptr, length(axes(bc, 1)) + 1)
    sizehint!(dest.kernel_size, length(axes(bc, 1)))
    for i in eachindex(bc)
        push!(dest, bc[i])
    end
    return dest
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
    fill!(new_elem_ptr, 0)
    cumsum!(view(new_elem_ptr, firstindex(new_elem_ptr) + 1:lastindex(new_elem_ptr)), new_lengths)
    new_elem_ptr .+= firstindex(new_data)

    newA = VectorOfArrays(new_data, new_elem_ptr, new_kernel_size, no_consistency_checks)
    return newA
end


# A view on the result of view, using an array of indices, would allocate due
# to reindexing, so call SubArray directly:
_noreindex_view(A, idxs...) = SubArray(A, idxs)
_noreindex_view(A::AbstractArray{T,N}, ::Vararg{Colon,N}) where {T,N} = A

# SubArray is constructed directly and indexed with `@inbounds`, so indices
# must be converted (e.g. logical masks) and bounds-checked here:
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
    _findall!.(newA, A)
    return newA
end
