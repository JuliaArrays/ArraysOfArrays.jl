# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    AbstractArrayOfSimilarArrays{T,M,N,ET<:AbstractArray{T}} <: AbstractSlices{ET,N}

An array that contains arrays that have the same size/axes. The array is
internally stored in flattened form as some kind of array of dimension
`M + N`. The flattened form can be accessed via `flatview(A)`.

Subtypes must implement (in addition to typical array operations):

    flatview(A::SomeArrayOfSimilarArrays)::AbstractArray{T,M+N}
    getslicemap(A::SomeArrayOfSimilarArrays)

The following type aliases are defined:

* `AbstractVectorOfSimilarArrays{T,M,ET} = AbstractArrayOfSimilarArrays{T,M,1,ET}`
* `AbstractArrayOfSimilarVectors{T,N,ET} = AbstractArrayOfSimilarArrays{T,1,N,ET}`
* `AbstractVectorOfSimilarVectors{T,ET} = AbstractArrayOfSimilarArrays{T,1,1,ET}`
"""
abstract type AbstractArrayOfSimilarArrays{T,M,N,ET<:AbstractArray{T}} <: AbstractSlices{ET,N} end
export AbstractArrayOfSimilarArrays

const AbstractVectorOfSimilarArrays{T,M,ET<:AbstractArray{T}} = AbstractArrayOfSimilarArrays{T,M,1,ET}
export AbstractVectorOfSimilarArrays

const AbstractArrayOfSimilarVectors{T,N,ET<:AbstractVector{T}} = AbstractArrayOfSimilarArrays{T,1,N,ET}
export AbstractArrayOfSimilarVectors

const AbstractVectorOfSimilarVectors{T,ET<:AbstractVector{T}} = AbstractArrayOfSimilarArrays{T,1,1,ET}
export AbstractVectorOfSimilarVectors



"""
    ArrayOfSimilarArrays{T,M,N,P} <: AbstractArrayOfSimilarArrays{T,M,N}

Represents a view of an array of dimension `M + N` as an array of
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
    T, M, N,
    P<:AbstractArray{T},
    ET<:AbstractArray{T,M}
} <: AbstractArrayOfSimilarArrays{T,M,N,ET}
    data::P

    function ArrayOfSimilarArrays{T,M,N}(flat_data::AbstractArray{U}) where {T,M,N,U}
        require_ndims(flat_data, _add_vals(Val{M}(), Val{N}()))
        conv_parent = _convert_elype(T, flat_data)
        P = typeof(conv_parent)
        ET = Base.promote_op(view, P, _nColons(Val{M}())..., _nInts(Val{N}())...)
        new{T,M,N,P,ET}(conv_parent)
    end
end

function ArrayOfSimilarArrays{T,M}(flat_data::AbstractArray{U}) where {T,M,U}
    _, size_outer = split_tuple(size(flat_data), Val{M}())
    N = length(size_outer)
    ArrayOfSimilarArrays{T,M,N}(flat_data)
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


Base.convert(R::Type{ArrayOfSimilarArrays{T,M,N}}, flat_data::AbstractArray{U}) where {T,M,N,U} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarArrays{T,M}}, flat_data::AbstractArray{U}) where {T,M,U} = R(flat_data)

Base.convert(R::Type{ArrayOfSimilarArrays{T,M,N}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarArrays{T}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarArrays}, A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} = R(A)

joinedview(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} = A.data
Base.stack(A::ArrayOfSimilarArrays) = joinedview(A)

function Base.Array(A::ArrayOfSimilarArrays{T,M,N,P,ET}) where {T,M,N,P,ET}
    new_ET = Base.promote_op(similar, ET)
    return Array{new_ET,N}(A)
end

function getslicemap(::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    return (_ncolons(Val{M}())..., _oneto_tpl(Val{N}())...)
end

#!!!!!
@inline function innersize(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    front_tuple(size(A.data), Val{M}())
end


import Base.==
(==)(A::ArrayOfSimilarArrays{T,M,N}, B::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} =
    (A.data == B.data)


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
    # ToDo: Don't use similar if data is an ElasticArray?
    ArrayOfSimilarArrays{T,M,N}(similar(data, U, size_inner..., dims...))
end


function Base.deepcopy(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    ArrayOfSimilarArrays{T,M,N}(deepcopy(A.data))
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


Base.map(::typeof(identity), A::ArrayOfSimilarArrays) = A
Base.Broadcast.broadcasted(::typeof(identity), A::ArrayOfSimilarArrays) = A


Base.@pure _result_is_nested(idxs_outer::Tuple, idxs_inner::Tuple) =
    Val{!(Base.index_dimsum(idxs_outer...) isa Tuple{}) && !(Base.index_dimsum(idxs_inner...) isa Tuple{})}()

Base.@pure ndims_after_getindex(idxs::Tuple) = Val{length(Base.index_dimsum(idxs...))}()


const VectorOfSimilarArrays{
    T, M,
    P<:AbstractArray{T},
    ET<:AbstractArray{T}
} = ArrayOfSimilarArrays{T,M,1,P,ET}

export VectorOfSimilarArrays

VectorOfSimilarArrays{T}(flat_data::AbstractArray{U}) where {T,U} =
    ArrayOfSimilarArrays{T,length(Base.front(size(flat_data))),1}(flat_data)

VectorOfSimilarArrays(flat_data::AbstractArray{T}) where {T} =
    ArrayOfSimilarArrays{T,length(Base.front(size(flat_data))),1}(flat_data)

VectorOfSimilarArrays{T}(A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} =
    VectorOfSimilarArrays{T,M}(A)

VectorOfSimilarArrays(A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} =
    VectorOfSimilarArrays{T,M}(A)


Base.convert(R::Type{VectorOfSimilarArrays{T}}, flat_data::AbstractArray{U}) where {T,U} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarArrays}, flat_data::AbstractArray{T}) where {T} = R(flat_data)
Base.convert(R::Type{VectorOfSimilarArrays{T}}, A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} = R(A)
Base.convert(R::Type{VectorOfSimilarArrays}, A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} = R(A)


@inline Base.IndexStyle(::Type{<:VectorOfSimilarArrays}) = IndexLinear()


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


function _empty_data_size(A::VectorOfSimilarArrays{T,M}) where {T,M}
    size_inner, size_outer = split_tuple(size(A.data), Val{M}())
    empty_size_outer = map(x -> zero(x), size_outer)
    (size_inner..., empty_size_outer...)
end

function Base.empty(A::VectorOfSimilarArrays{T,M}, ::Type{<:AbstractArray{U}}) where {T,M,U}
    new_data_size = _empty_data_size(A)
    # ToDo: Don't use similar if data is an ElasticArray?
    VectorOfSimilarArrays{T,M}(similar(A.data, U, new_data_size...))
end

function Base.empty!(A::VectorOfSimilarArrays{T,M}) where {T,M}
    resize!(A.data, _empty_data_size(A))
    A
end



const ArrayOfSimilarVectors{
    T, N,
    P<:AbstractArray{T},
    ET<:AbstractVector{T}
} = ArrayOfSimilarArrays{T,1,N,P,ET}

export ArrayOfSimilarVectors

ArrayOfSimilarVectors{T}(flat_data::AbstractArray{U}) where {T,U} =
    ArrayOfSimilarArrays{T,1,length(Base.front(size(flat_data)))}(flat_data)

ArrayOfSimilarVectors(flat_data::AbstractArray{T}) where {T} =
    ArrayOfSimilarArrays{T,1,length(Base.front(size(flat_data)))}(flat_data)

ArrayOfSimilarVectors{T}(A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} =
    ArrayOfSimilarVectors{T,N}(A)

ArrayOfSimilarVectors(A::AbstractArray{<:AbstractVector{T},N}) where {T,N} =
    ArrayOfSimilarVectors{T,N}(A)


Base.convert(R::Type{ArrayOfSimilarVectors{T}}, flat_data::AbstractArray{U}) where {T,U} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarVectors}, flat_data::AbstractArray{T}) where {T} = R(flat_data)
Base.convert(R::Type{ArrayOfSimilarVectors{T}}, A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} = R(A)
Base.convert(R::Type{ArrayOfSimilarVectors}, A::AbstractArray{<:AbstractVector{T},N}) where {T,N} = R(A)


const VectorOfSimilarVectors{
    T,
    P<:AbstractArray{T,2},
    ET<:AbstractVector{T}
} = ArrayOfSimilarArrays{T,1,1,P,ET}

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


Base.sum(X::AbstractVectorOfSimilarArrays{T,M}) where {T,M} =
    sum(flatview(X); dims = M + 1)[_ncolons(Val{M}())...]

Statistics.mean(X::AbstractVectorOfSimilarArrays{T,M}) where {T,M} =
    mean(flatview(X); dims = M + 1)[_ncolons(Val{M}())...]

Statistics.var(X::AbstractVectorOfSimilarArrays{T,M}; corrected::Bool = true, mean = nothing) where {T,M} =
    var(flatview(X); dims = M + 1, corrected = corrected, mean = mean)[_ncolons(Val{M}())...]

Statistics.std(X::AbstractVectorOfSimilarArrays{T,M}; corrected::Bool = true, mean = nothing) where {T,M} =
    std(flatview(X); dims = M + 1, corrected = corrected, mean = mean)[_ncolons(Val{M}())...]

Statistics.cov(X::AbstractVectorOfSimilarVectors; corrected::Bool = true) =
    cov(flatview(X); dims = 2, corrected = corrected)

Statistics.cor(X::AbstractVectorOfSimilarVectors) =
    cor(flatview(X); dims = 2)


"""
    nestedview(A::AbstractArray{T,M+N}, M::Integer)
    nestedview(A::AbstractArray{T,2})

AbstractArray{<:AbstractArray{T,M},N}

View array `A` in as an `N`-dimensional array of `M`-dimensional arrays by
wrapping it into an [`ArrayOfSimilarArrays`](@ref).

It's also possible to use a `StaticVector` of length `S` as the type of the
inner arrays via

    nestedview(A::AbstractArray{T}, ::Type{StaticArrays.SVector{S}})
    nestedview(A::AbstractArray{T}, ::Type{StaticArrays.SVector{S,T}})
"""
function nestedview end
export nestedview

@inline nestedview(A::AbstractArray{T,L}, M::Integer) where {T,L} =
    ArrayOfSimilarArrays{T,M}(A)

@inline nestedview(A::AbstractArray{T,L}, ::Val{M}) where {T,L,M} =
    ArrayOfSimilarArrays{T,M}(A)

@inline nestedview(A::AbstractArray{T,2}) where {T} =
    VectorOfSimilarVectors{T}(A)
