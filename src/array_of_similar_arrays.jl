# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    AbstractArrayOfSimilarArrays{T,M,N} <: AbstractArray{<:AbstractArray{T,M},N}

An array that contains arrays that have the same size/axes. The array is
internally stored in flattened form as some kind of array of dimension
`M + N`. The flattened form can be accessed via `flatview(A)`.

Subtypes must implement (in addition to typical array operations):

    flatview(A::SomeArrayOfSimilarArrays)::AbstractArray{T,M+N}

The following type aliases are defined:

* `AbstractVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}`
* `AbstractArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}`
* `AbstractVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}`
"""
abstract type AbstractArrayOfSimilarArrays{T,M,N} <: AbstractArray{Array{T,M},N} end
export AbstractArrayOfSimilarArrays

const AbstractVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}
export AbstractVectorOfSimilarArrays

const AbstractArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}
export AbstractArrayOfSimilarVectors

const AbstractVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}
export AbstractVectorOfSimilarVectors



"""
    ArrayOfSimilarArrays{T,M,N,L,P} <: AbstractArrayOfSimilarArrays{T,M,N}

Represents a view of an array of dimension `L = M + N` as an array of
dimension M with elements that are arrays with dimension N. All element arrays
implicitly have equal size/axes.

Constructors:

    ArrayOfSimilarArrays{T,M,N}(flat_data::AbstractArray)
    ArrayOfSimilarArrays{T,M}(flat_data::AbstractArray)

The following type aliases are defined:

* `VectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}`
* `ArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}`
* `VectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}`

`VectorOfSimilarArrays` supports `push!()`, etc., provided the underlying
array supports resizing of it's last dimension (e.g. an `ElasticArray`).

The nested array can also be created using the function [`nestedview`](@ref)
and the wrapped flat array can be accessed using [`flatview`](@ref)
afterwards:

```julia
A_flat = rand(2,3,4,5,6)
A_nested = nestedview(A_flat, 2)
A_nested isa AbstractArray{<:AbstractArray{T,2},3} where T
flatview(A_nested) === A_flat
```
"""
struct ArrayOfSimilarArrays{
    T, M, N, L,
    P<:AbstractArray{T,L}
} <: AbstractArrayOfSimilarArrays{T,M,N}
    data::P

    function ArrayOfSimilarArrays{T,M,N}(flat_data::AbstractArray{U,L}) where {T,M,N,L,U}
        size_inner, size_outer = split_tuple(size(flat_data), Val{M}())
        require_ndims(flat_data, _add_vals(Val{M}(), Val{N}()))
        conv_parent = _convert_elype(T, flat_data)
        P = typeof(conv_parent)
        new{T,M,N,L,P}(conv_parent)
    end

    function ArrayOfSimilarArrays{T,M}(flat_data::AbstractArray{U,L}) where {T,M,L,U}
        size_inner, size_outer = split_tuple(size(flat_data), Val{M}())
        N = length(size_outer)
        conv_parent = _convert_elype(T, flat_data)
        P = typeof(conv_parent)
        new{T,M,N,L,P}(conv_parent)
    end
end

export ArrayOfSimilarArrays

function ArrayOfSimilarArrays{T,M,N}(A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U}
    B = ArrayOfSimilarArrays{T,M,N}(Array{T}(undef, innersize(A)..., size(A)...))
    copyto!(B, A)
end

ArrayOfSimilarArrays{T}(A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    ArrayOfSimilarArrays{T,M,N}(A)

ArrayOfSimilarArrays(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} =
    ArrayOfSimilarArrays{T,M,N}(A)


Base.convert(R::Type{ArrayOfSimilarArrays{T,M,N}}, flat_data::AbstractArray{U,L}) where {T,M,N,L,U} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarArrays{T,M}}, flat_data::AbstractArray{U,L}) where {T,M,L,U} = R(flat_data)

Base.convert(R::Type{ArrayOfSimilarArrays{T,M,N}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarArrays{T}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarArrays}, A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} = R(A)


@inline function innersize(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    front_tuple(size(A.data), Val{M}())
end


@inline function _innerlength(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    prod(innersize(A))
end


import Base.==
(==)(A::ArrayOfSimilarArrays{T,M,N}, B::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} =
    (A.data == B.data)


"""
    flatview(A::ArrayOfSimilarArrays{T,M,N,L,P})::P

Returns the array of dimensionality `L = M + N` wrapped by `A`. The shape of
the result may be freely changed without breaking the inner consistency of
`A`.
"""
flatview(A::ArrayOfSimilarArrays) = A.data


Base.size(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} = split_tuple(size(A.data), Val{M}())[2]



Base.@propagate_inbounds Base.getindex(A::ArrayOfSimilarArrays{T,M,N}, idxs::Vararg{Integer,N}) where {T,M,N} =
    view(A.data, _ncolons(Val{M}())..., idxs...)


Base.@propagate_inbounds Base.setindex!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}, idxs::Vararg{Integer,N}) where {T,M,N,U} =
    setindex!(A.data, x, _ncolons(Val{M}())..., idxs...)

Base.@propagate_inbounds function Base.unsafe_view(A::ArrayOfSimilarArrays{T,M,N}, idxs::Vararg{Union{Real, AbstractArray},N}) where {T,M,N}
    dataview = view(A.data, _ncolons(Val{M}())..., idxs...)
    L = length(size(dataview))
    N_view = L - M
    ArrayOfSimilarArrays{T,M,N_view}(dataview)
end


@inline function Base.resize!(A::ArrayOfSimilarArrays{T,M,N}, dims::Vararg{Integer,N}) where {T,M,N}
    resize!(A.data, innersize(A)..., dims...)
    A
end


function Base.similar(A::ArrayOfSimilarArrays{T,M,N}, ::Type{<:AbstractArray{U}}, dims::Dims) where {T,M,N,U}
    data = A.data
    size_inner, size_outer = split_tuple(size(data), Val{M}())
    ArrayOfSimilarArrays{T,M,N}(similar(data, U, size_inner..., dims...))
end


function Base.copyto!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    copyto!(dest.data, src.data)
    dest
end


function Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    innersize(dest) != innersize(src) && throw(DimensionMismatch("Can't append, shape of element arrays of source and dest are not equal"))
    append!(dest.data, src.data)
    dest
end

Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    append!(dest, ArrayOfSimilarArrays(src))


function Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    innersize(dest) != innersize(src) && throw(DimensionMismatch("Can't prepend, shape of element arrays of source and dest are not equal"))
    prepend!(dest.data, src.data)
    dest
end

Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    prepend!(dest, ArrayOfSimilarArrays(src))


UnsafeArrays.unsafe_uview(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} =
    ArrayOfSimilarArrays{T,M,N}(uview(A.data))


function innermap(f::Base.Callable, A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    new_data = map(f, A.data)
    U = eltype(new_data)
    ArrayOfSimilarArrays{U,M,N}(new_data)
end


function deepmap(f::Base.Callable, A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    new_data = deepmap(f, A.data)
    U = eltype(new_data)
    ArrayOfSimilarArrays{U,M,N}(new_data)
end



Base.@pure _result_is_nested(idxs_outer::Tuple, idxs_inner::Tuple) =
    Val{!(Base.index_dimsum(idxs_outer...) isa Tuple{}) && !(Base.index_dimsum(idxs_inner...) isa Tuple{})}()

Base.@pure ndims_after_getindex(idxs::Tuple) = Val{length(Base.index_dimsum(idxs...))}()


Base.@propagate_inbounds function deepgetindex(A::ArrayOfSimilarArrays{T,M,N,L}, idxs::Vararg{Any,L}) where {T,M,N,L}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    nested = _result_is_nested(idxs_outer, idxs_inner)
    _deepgetindex_impl_aosa(A, idxs_outer, idxs_inner, nested)
end

Base.@propagate_inbounds _deepgetindex_impl_aosa(A::ArrayOfSimilarArrays, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{false}) =
    getindex(A.data, idxs_inner..., idxs_outer...)

Base.@propagate_inbounds function _deepgetindex_impl_aosa(A::ArrayOfSimilarArrays, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{true})
    new_data = getindex(A.data, idxs_inner..., idxs_outer...)
    nestedview(new_data, ndims_after_getindex(idxs_inner))
end


Base.@propagate_inbounds function deepsetindex!(A::ArrayOfSimilarArrays{T,M,N,L}, x, idxs::Vararg{Any,L}) where {T,M,N,L}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    _deepsetindex_impl_aosa!(A, x, idxs_outer, idxs_inner, _result_is_nested(idxs_outer, idxs_inner))
    A
end

Base.@propagate_inbounds _deepsetindex_impl_aosa!(A::ArrayOfSimilarArrays, x, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{false}) =
    setindex!(A.data, x, idxs_inner..., idxs_outer...)

Base.@propagate_inbounds _deepsetindex_impl_aosa!(A::ArrayOfSimilarArrays, x, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{true}) =
    _deepsetindex_impl!(A, x, idxs_outer, idxs_inner)

# TODO: Specialized method _deepsetindex_impl_aosa!(A::ArrayOfSimilarArrays, x::ArrayOfSimilarArrays, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{true}) =


Base.@propagate_inbounds function deepview(A::ArrayOfSimilarArrays{T,M,N,L}, idxs::Vararg{Any,L}) where {T,M,N,L}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    nested = _result_is_nested(idxs_outer, idxs_inner)
    _deepview_impl_aosa(A, idxs_outer, idxs_inner, nested)
end

Base.@propagate_inbounds _deepview_impl_aosa(A::ArrayOfSimilarArrays, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{false}) =
    view(A.data, idxs_inner..., idxs_outer...)

Base.@propagate_inbounds function _deepview_impl_aosa(A::ArrayOfSimilarArrays, idxs_outer::Tuple, idxs_inner::Tuple, nested::Val{true})
    new_data = view(A.data, idxs_inner..., idxs_outer...)
    nestedview(new_data, ndims_after_getindex(idxs_inner))
end



const VectorOfSimilarArrays{
    T, M, L,
    P<:AbstractArray{T,L}
} = ArrayOfSimilarArrays{T,M,1,L,P}

export VectorOfSimilarArrays

VectorOfSimilarArrays{T}(flat_data::AbstractArray{U,L}) where {T,U,L} =
    ArrayOfSimilarArrays{T,length(Base.front(size(flat_data))),1}(flat_data)

VectorOfSimilarArrays(flat_data::AbstractArray{T,L}) where {T,L} =
    ArrayOfSimilarArrays{T,length(Base.front(size(flat_data))),1}(flat_data)

VectorOfSimilarArrays{T}(A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} =
    VectorOfSimilarArrays{T,M}(A)

VectorOfSimilarArrays(A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} =
    VectorOfSimilarArrays{T,M}(A)


Base.convert(R::Type{VectorOfSimilarArrays{T}}, flat_data::AbstractArray{U,L}) where {T,U,L} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarArrays}, flat_data::AbstractArray{T,L}) where {T,L} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarArrays{T}}, A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} = R(A)
Base.convert(R::Type{VectorOfSimilarArrays}, A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} = R(A)


@inline Base.IndexStyle(A::VectorOfSimilarArrays) = IndexLinear()


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

function Base.pushfirst!(V::VectorOfSimilarArrays{T,M}, x::AbstractArray{U,M}) where {T,M,U}
    size(x) != Base.front(size(V.data)) && throw(DimensionMismatch("Can't push, shape of source and elements of target is incompatible"))
    prepend!(V.data, x)
    V
end

# Will need equivalent of resize! that resizes in front of data instead of in back:
# popfirst!(V::ArrayOfSimilarArrays) = ...



const ArrayOfSimilarVectors{
    T, N, L,
    P<:AbstractArray{T,L}
} = ArrayOfSimilarArrays{T,1,N,L,P}

export ArrayOfSimilarVectors

ArrayOfSimilarVectors{T}(flat_data::AbstractArray{U,L}) where {T,U,L} =
    ArrayOfSimilarArrays{T,1,length(Base.front(size(flat_data)))}(flat_data)

ArrayOfSimilarVectors(flat_data::AbstractArray{T,L}) where {T,L} =
    ArrayOfSimilarArrays{T,1,length(Base.front(size(flat_data)))}(flat_data)

ArrayOfSimilarVectors{T}(A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} =
    ArrayOfSimilarVectors{T,N}(A)

ArrayOfSimilarVectors(A::AbstractArray{<:AbstractVector{T},N}) where {T,N} =
    ArrayOfSimilarVectors{T,N}(A)


Base.convert(R::Type{ArrayOfSimilarVectors{T}}, flat_data::AbstractArray{U,L}) where {T,U,L} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarVectors}, flat_data::AbstractArray{T,L}) where {T,L} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarVectors{T}}, A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarVectors}, A::AbstractArray{<:AbstractVector{T},N}) where {T,N} = R(A)


const VectorOfSimilarVectors{
    T,
    P<:AbstractArray{T,2}
} = ArrayOfSimilarArrays{T,1,1,2,P}

export VectorOfSimilarVectors

VectorOfSimilarVectors{T}(flat_data::AbstractArray{U,2}) where {T,U} =
    ArrayOfSimilarArrays{T,1,1}(flat_data)

VectorOfSimilarVectors(flat_data::AbstractArray{T,2}) where {T} =
    VectorOfSimilarVectors{T}(flat_data)

VectorOfSimilarVectors{T}(A::AbstractVector{<:AbstractVector{U}}) where {T,U} =
    ArrayOfSimilarArrays{T,1}(A)

VectorOfSimilarVectors(A::AbstractVector{<:AbstractVector{T}}) where {T} =
    VectorOfSimilarVectors{T}(A)

Base.convert(R::Type{VectorOfSimilarVectors{T}}, flat_data::AbstractArray{U,2}) where {T,U} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarVectors}, flat_data::AbstractArray{T,2}) where {T} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarVectors{T}}, A::AbstractVector{<:AbstractVector{U}}) where {T,U} = R(A)
Base.convert(R::Type{VectorOfSimilarVectors}, A::AbstractVector{<:AbstractVector{T}}) where {T} = R(A)


@inline Base.IndexStyle(A::VectorOfSimilarVectors) = IndexLinear()


"""
    sum(X::AbstractVectorOfSimilarArrays)
    sum(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights)

Compute the sum of the elements vectors of `X`. Equivalent to `sum` of
`flatview(X)` along the last dimension.
"""
Base.sum(X::AbstractVectorOfSimilarArrays{T,M}) where {T,M} = sum(flatview(X); dims = M + 1)


"""
    mean(X::AbstractVectorOfSimilarArrays)
    mean(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights)

Compute the mean of the elements vectors of `X`. Equivalent to `mean` of
`flatview(X)` along the last dimension.
"""
Statistics.mean(X::AbstractVectorOfSimilarArrays{T,M}) where {T,M} =
    mean(flatview(X); dims = M + 1)


"""
    var(X::AbstractVectorOfSimilarArrays; corrected::Bool = true)
    var(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights; corrected::Bool = true)

Compute the sample variance of the elements vectors of `X`. Equivalent to
`var` of `flatview(X)` along the last dimension.
"""
Statistics.var(X::AbstractVectorOfSimilarArrays{T,M}; corrected::Bool = true) where {T,M} =
    var(flatview(X); dims = M + 1, corrected = corrected)


"""
    cov(X::AbstractVectorOfSimilarVectors; corrected::Bool = true)
    cov(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true)

Compute the covariance matrix between the elements of the elements of `X`
along `X`. Equivalent to `cov` of `flatview(X)` along dimension 2.
"""
Statistics.cov(X::AbstractVectorOfSimilarVectors; corrected::Bool = true) =
    cov(flatview(X); dims = 2, corrected = corrected)


"""
    cor(X::AbstractVectorOfSimilarVectors)
    cor(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights)

Compute the Pearson correlation matrix between the elements of the elements of
 `X` along `X`. Equivalent to `cor` of `flatview(X)` along dimension 2.
"""
Statistics.cor(X::AbstractVectorOfSimilarVectors) =
    cor(flatview(X); dims = 2)
