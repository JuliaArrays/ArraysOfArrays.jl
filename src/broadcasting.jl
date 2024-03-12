
# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

const _RefLike{T} = Union{Tuple{T}, Ref{T}}


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

_generic_size(A) = size(A)
_generic_size(tpl::Tuple) = (length(tpl),)

Base.Base.@propagate_inbounds function _to_indices(A, idxs)
    new_idxs = Base.to_indices(A, idxs)
    @boundscheck Base.checkbounds(A, new_idxs...)
    return new_idxs
end

# Limited to vectors of vectors for now.
# ToDo: Extend to vectors of arrays.
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::Union{VectorOfSimilarVectors, VectorOfVectors},
    Idxs::Union{AbstractVector{<:AbstractVector{<:Integer}},AbstractVector{Colon},_RefLike{<:Union{AbstractVector{<:Integer},Colon}}}...
)
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
    newA .= _noreindex_view.(A, Idxs...)
    return newA
end


# Limited to vectors of vectors for now.
# ToDo: Extend to vectors of arrays.
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::VectorOfSimilarVectors,
    Idxs::Union{VectorOfSimilarVectors{<:Integer},AbstractVector{Colon},_RefLike{<:Union{AbstractVector{<:Integer},Colon}}}...
)
    # Checks size compatibility:
    Base.Broadcast.broadcast_shape(size(A), map(size, Idxs)...)

    sz_inner = map(only ∘ innersize, Idxs)
    sz_outer = size(A)
    new_data = similar(A.data, (sz_inner..., sz_outer...))
    newA = VectorOfSimilarVectors(new_data)

    newA .= _noreindex_view.(A, Idxs...)
    return newA
end


# Limited to vectors of vectors for now.
# ToDo: Extend to vectors of arrays.
function Base.Broadcast.broadcasted(
    ::typeof(getindex),
    A::VectorOfSimilarVectors,
    Idxs::_RefLike{<:Union{AbstractVector{<:Integer},Colon}}...
)
    data = A.data
    inner_idxs = map(only, Idxs)
    outer_idxs = (:,)
    new_data = data[inner_idxs..., outer_idxs...]
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
    A::Union{VectorOfSimilarVectors{Bool}, VectorOfVectors{Bool}}
)
    new_lengths = _similar_idx_vector(A, _idx_type(A), length(A))
    new_lengths .= sum.(A)

    new_kernel_size = map(_ -> (), new_lengths)

    newA = _new_vector_of_arrays_with_lengths(A, Int, new_kernel_size, new_lengths)
    _findall!.(newA, A)
    return newA
end
