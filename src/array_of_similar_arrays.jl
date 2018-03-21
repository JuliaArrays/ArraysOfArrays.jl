# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

@doc doc"""
    AbstractArrayOfSimilarArrays{T,M,N} <: AbstractArray{AbstractArray{T,M},N}

An array that contains arrays that have the same size/axes. The array is
internally stored in flattened form as some kind of array of dimension
`M + N`. The flattened form can be accessed via `parent(A)`.

Subtypes must implement (in addition to typical array operations):

    parent(A::SomeArrayOfSimilarArrays)::AbstractArray{T,M+N}

The following type aliases are defined:

* `AbstractVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}`
* `AbstractArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}`
* `AbstractVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}`
"""
abstract type AbstractArrayOfSimilarArrays{T,M,N} <: AbstractArray{AbstractArray{T,M},N} end
export AbstractArrayOfSimilarArrays

const AbstractVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}
export AbstractVectorOfSimilarArrays

const AbstractArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}
export AbstractArrayOfSimilarVectors

const AbstractVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}
export AbstractVectorOfSimilarVectors



@doc doc"""
    ArrayOfSimilarArrays{T,M,N,L,P} <: AbstractArrayOfSimilarArrays{T,M,N}

Represents a view of an array of dimension `L = M + N` as an array of
dimension M with elements that are arrays with dimension N. All element arrays
implicitly have equal size/axes.

Constructors:

    ArrayOfSimilarArrays{N}(parent::AbstractArray)

The following type aliases are defined:

* `VectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}`
* `ArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}`
* `VectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}`

`VectorOfSimilarArrays` supports `push!()`, etc., provided the underlying
array supports resizing of it's last dimension (e.g. an `ElasticArray`).
"""
struct ArrayOfSimilarArrays{
    T, M, N, L,
    P<:AbstractArray{T,L}
} <: AbstractArrayOfSimilarArrays{T,M,N}
    data::P

    function ArrayOfSimilarArrays{M}(parent::AbstractArray{T,L}) where {T,M,L}
        size_inner, size_outer = split_tuple(size(parent), Val{M}())
        N = length(size_outer)
        P = typeof(parent)
        new{T,M,N,L,P}(parent)
    end
end

export ArrayOfSimilarArrays


function _size_inner(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    s = if !isempty(A)
        map(Int, size(A[1]))
    else
        ntuple(_ -> zero(Int), Val(M))
    end

    all(X -> size(X) == s, A) || throw(DimensionMismatch("Shape of element arrays of A is not equal, can't determine common shape"))
    s
end

function _size_inner(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    sz_inner, sz_outer = split_tuple(size(A.data), Val{M}())
    sz_inner
end


function ArrayOfSimilarArrays(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    B = ArrayOfSimilarArrays{M}(Array{T,M+N}(_size_inner(A)..., size(A)...))
    copy!(B, A)
end


import Base.==
(==)(A::ArrayOfSimilarArrays{T,M,N}, B::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} =
    (A.data == B.data)


Base.parent(A::ArrayOfSimilarArrays) = A.data


Base.size(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} = split_tuple(size(A.data), Val{M}())[2]


Base.@propagate_inbounds Base.getindex(A::ArrayOfSimilarArrays{T,M,N}, idxs::Vararg{Integer,N}) where {T,M,N} =
    view(A.data, _ncolons(Val{M}())..., idxs...)


Base.@propagate_inbounds Base.setindex!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}, idxs::Vararg{Integer,N}) where {T,M,N,U} =
    setindex!(A.data, x, _ncolons(Val{M}())..., idxs...)


@static if VERSION < v"0.7.0-DEV.2791"
    Base.repremptyarray(io::IO, X::ArrayOfSimilarArrays{T,M,N,L,P}) where {T,M,N,L,P} = print(io, "ElasticArray{$T,$M,$N,$L,$P}(", join(size(X),','), ')')
end


@inline function Base.resize!(A::ArrayOfSimilarArrays{T,M,N}, dims::Vararg{Integer,N}) where {T,M,N}
    resize!(A.data, _size_inner(A)..., dims...)
    A
end


function Base.similar(A::ArrayOfSimilarArrays{T,M,N}, ::Type{<:AbstractArray{U}}, dims::Dims) where {T,M,N,U}
    data = A.data
    size_inner, size_outer = split_tuple(size(data), Val{M}())
    ArrayOfSimilarArrays{M}(similar(data, U, size_inner..., dims...))
end


function Base.resize!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    _size_inner(dest) != _size_inner(src) && throw(DimensionMismatch("Can't append, shape of element arrays of source and dest are not equal"))
    append!(dest.data, src.data)
    dest
end


function Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    _size_inner(dest) != _size_inner(src) && throw(DimensionMismatch("Can't append, shape of element arrays of source and dest are not equal"))
    append!(dest.data, src.data)
    dest
end

Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    append!(dest, ArrayOfSimilarArrays(src))


function Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    _size_inner(dest) != _size_inner(src) && throw(DimensionMismatch("Can't prepend, shape of element arrays of source and dest are not equal"))
    prepend!(dest.data, src.data)
    dest
end

Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    prepend!(dest, ArrayOfSimilarArrays(src))


UnsafeArrays.unsafe_uview(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} =
    ArrayOfSimilarArrays{M}(uview(A.data))



const VectorOfSimilarArrays{
    T, M, L,
    P<:AbstractArray{T,L}
} = ArrayOfSimilarArrays{T,M,1,L,P}

export VectorOfSimilarArrays

VectorOfSimilarArrays(parent::AbstractArray{T,L}) where {T,L} =
    ArrayOfSimilarArrays{L-1}(parent)


@inline Base.IndexStyle(V::VectorOfSimilarArrays) = IndexLinear()


Base.@propagate_inbounds Base.getindex(A::VectorOfSimilarArrays{T,M}, rng::Union{Colon,UnitRange{<:Integer}}) where {T,M} =
    VectorOfSimilarArrays(view(A.data, _ncolons(Val{M}())..., rng))


function Base.push!(V::VectorOfSimilarArrays{T,M}, x::AbstractArray{U,M}) where {T,M,U}
    size(x) != Base.front(size(V.data)) && throw(DimensionMismatch("Can't push, shape of source and elements of target is incompatible"))
    append!(V.data, x)
    V
end

function Base.pop!(V::VectorOfSimilarArrays)
    isempty(V) && throw(ArgumentError("array must be non-empty"))
    x = V[end]
    resize!(V, size(V, 1) - 1)
    x
end

function Compat.pushfirst!(V::VectorOfSimilarArrays{T,M}, x::AbstractArray{U,M}) where {T,M,U}
    size(x) != Base.front(size(V.data)) && throw(DimensionMismatch("Can't push, shape of source and elements of target is incompatible"))
    prepend!(V.data, x)
    V
end

# Will need equivalent of resize! that resizes in front of data instead of in back:
# Compat.popfirst!(V::ArrayOfSimilarArrays) = ...



const ArrayOfSimilarVectors{
    T, N, L,
    P<:AbstractArray{T,L}
} = ArrayOfSimilarArrays{T,1,N,L,P}

export ArrayOfSimilarVectors

ArrayOfSimilarVectors(parent::AbstractArray{T,L}) where {T,L} =
    ArrayOfSimilarArrays{1}(parent)



const VectorOfSimilarVectors{
    T,
    P<:AbstractArray{T,2}
} = ArrayOfSimilarArrays{T,1,1,2,P}

export VectorOfSimilarVectors

VectorOfSimilarVectors(parent::AbstractArray{T,2}) where {T} =
    ArrayOfSimilarArrays{1}(parent)
