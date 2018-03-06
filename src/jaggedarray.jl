# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


doc"""
    AbstractJaggedArray{T,N,M} <: AbstractArray{T,N}

An `AbstractJaggedArray` represents an `N`-dimensional jagged (ragged) array
of element type `T`. It can be interpreted as a hierarchy of nested Vectors (a
vector of vectors of vectors of ...), but with reverse indexing (the index of
the innermost vector is the first index of the jagged array).

The first `M >= 1` dimensions of `J::AbstractJaggedArray{T,N,M}` are
non-jagged, i.e. `J[:,:,...,i,j,...]` will return a standard `Array` for up to
`M` colons.

The following standard functions take a special meaning for jagged arrays:

* `indices(J, i::Integer, union)`: Union of all valid index ranges for
  dimension `i` (for all possible indices for all other dimensions).
* `indices(J, i::Integer, intersect)`: Intersect of all valid index ranges for
  dimension `i`.
* `indices(J, i::Integer)`: Range of valid indices for dimension `i`. Will
  throw an exception if
  `indices(J, i::Integer, union) != indices(J, i::Integer, intersect)`.
* `indices(J)` and `indices(J, union|intersect)`: Index ranges for all
  dimensions.
* `size(J, ...)`: Analog to behavior of `indices`.
"""
abstract type AbstractJaggedArray{T,N,M} <: AbstractArray{T,N} end
export AbstractJaggedArray

# Possible additional types:
# JaggedArrayImpl (supertype for JaggedArray and DenseJaggedArrayView)
# DenseJaggedArrayView
# ScatteredJaggedArrayView
# Same struct DenseJaggedArray for JaggedArray and DenseJaggedArrayView?


_default_nullvalue(T) = zero(T)
_default_nullvalue(T::Integer) = typemax(T)
_default_nullvalue(T::AbstractFloat) = NaN(T)

mutable struct JaggedArrayImpl <: AbstractArray{T,N}{
    T,
    N,
    DV<:DenseVector{T},
    OV<:AbstractVector{Int}
} <: AbstractJaggedArray{T,N,M}
    # M == N-1
    indices::NTuple{N,OV} # Sparse indices
    offsets::NTuple{N,OV}
    data::DV
    nullvalue::T
end


import Base.==
(==)(A::JaggedArrayImpl, B::JaggedArrayImpl) =
    ndims(A) == ndims(B) && A.indices == B.indices &&
    A.offsets == B.offsets  && A.data == B.data


Base.parent(A::JaggedArrayImpl) = A.data

Base.size(A::JaggedArrayImpl) = ... !!!

@propagate_inbounds Base.getindex(A::JaggedArrayImpl, i::Integer) = getindex(A.data, i)
@propagate_inbounds Base.setindex!(A::JaggedArrayImpl, x, i::Integer) = setindex!(A.data, x, i)

@propagate_inbounds Base.getindex(A::JaggedArrayImpl, Vararg{Integer,N}) =
    A.nullvalue # !!!! dummy implementation


#=
doc"""
    JaggedArray{T,N,M} <: AbstractJaggedArray{T,N,M}

Default implementation of `AbstractJaggedArray`.

Constructors:

    * `JaggedArray{T,N,M}()`
    * `JaggedArray{T,N}()` = `JaggedArray{T,N,1}()`
    * `JaggedArray(::AbstractArray)` (via `convert`).

The typical way of constructing a multi-dimensional `JaggedArray{T,N,M}` is
recursive, via `append!` of arrays `JaggedArray{T,N-1,M}`. The typical way
of constructing a `JaggedArray{T,N,N-1}` is via `append!` of arrays
`Array{T,N-1}`.
"""
const JaggedArray{...} = JaggedArrayImpl{...}
export JaggedArray
=#



#=
function _consistency_asserts(J::JaggedArray)
    array_sizes = V.array_sizes
    offsets = V.offsets
    global_offset = V.global_offset
    data = V.data
    @assert indices(array_sizes) == indices(offsets)
    if !isempty(offsets)
        @assert first(offsets) + global_offset == first(eachindex(data))
    end
end


function _next_offset(J::JaggedArray)
    if isempty(V.offsets)
        0 - V.global_offset
    else
        last(V.offsets) + prod(last(V.array_sizes))
    end::Int
end


function _view_reshape_spec(J::JaggedArray, i::Integer)
    s = V.array_sizes[i]
    o = V.offsets[i]
    l = prod(s)
    from = V.global_offset + o + first(linearindices(V.data))
    r = from:(from + l - 1)
    if checkbounds(Bool, V.offsets, i + 1)
        @assert l == V.offsets[i + 1] - o
    else
        @assert l == last(eachindex(V.data)) - o
    end
    (r, s)
end


import Base.==
(==)(X::JaggedArray, Y::JaggedArray) = X.array_sizes == Y.array_sizes && X.data == Y.data

Base.parent(J::JaggedArray) = V.data
Base.size(J::JaggedArray) = size(V.array_sizes)


function Base.getindex(J::JaggedArray, i::Integer)
    r, s = _view_reshape_spec(V, i)
    reshape(view(V.data, r), s...)
end

@inline function Base.getindex(J::JaggedArray, vecidx::Integer, arridxs::Tuple{Integer})
    @boundscheck checkbounds(V, arridxs, vecidx)
    offs = V.global_offset + V.offsets[vecidx]
    V.data[offs + arridxs[1]]
end

@inline function Base.getindex(J::JaggedArray{T,N}, vecidx::Integer, arridxs::NTuple{N,Integer}) where {T,N}
    # ...
end


function Base.setindex!(J::JaggedArray{T,N}, x::AbstractArray{U,N}, i::Integer) where {T,N,U}
    r, s = _view_reshape_spec(V, i) 
    s == size(x) || throw(DimensionMismatch("Can't assign array to element $i of JaggedArray, array size is incompatible"))
    copy!(V.data, first(r), x, first(linearindices(x)), length(r))
    V
end

Base.length(J::JaggedArray) = length(V.array_sizes)
Base._length(J::JaggedArray) = Base._length(V.array_sizes)
Base.linearindices(J::JaggedArray) = linearindices(V.array_sizes)


function Base.push!(J::JaggedArray{T,N}, x::AbstractArray{U,N}) where {T,N,U}
    s = size(x)
    l = length(linearindices(x))
    o = _next_offset(V)
    @assert l == prod(s)
    push!(V.offsets, o)
    push!(V.array_sizes, s)
    append!(V.data, x)
    V
end

=#
