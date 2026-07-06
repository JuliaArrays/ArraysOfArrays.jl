# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    AbstractArrayOfSimilarArrays{T,M,N,ET<:AbstractArray{T,M}} <: AbstractSlices{ET,N}

An array that contains arrays that have the same size/axes. The array is
internally stored in flattened form as some kind of array of dimension
`M + N`, in memory order. The flattened form can be accessed via
`flatview(A)`.

Subtypes must implement (in addition to typical array operations)

    ArraysOfArrays.fused(A::SomeArrayOfSimilarArrays)::AbstractArray{T,M+N}

which must return the underlying flat array. All split-mode operations
([`getsplitmode`](@ref), [`stacked`](@ref), [`flatview`](@ref),
[`innersize`](@ref), [`getslicemap`](@ref), etc.) are then provided
automatically.

The following type aliases are defined:

* `AbstractVectorOfSimilarArrays{T,M,ET} = AbstractArrayOfSimilarArrays{T,M,1,ET}`
* `AbstractArrayOfSimilarVectors{T,N,ET} = AbstractArrayOfSimilarArrays{T,1,N,ET}`
* `AbstractVectorOfSimilarVectors{T,ET} = AbstractArrayOfSimilarArrays{T,1,1,ET}`
"""
abstract type AbstractArrayOfSimilarArrays{T,M,N,ET<:AbstractArray{T,M}} <: AbstractSlices{ET,N} end
export AbstractArrayOfSimilarArrays

const AbstractVectorOfSimilarArrays{T,M,ET<:AbstractArray{T,M}} = AbstractArrayOfSimilarArrays{T,M,1,ET}
export AbstractVectorOfSimilarArrays

const AbstractArrayOfSimilarVectors{T,N,ET<:AbstractVector{T}} = AbstractArrayOfSimilarArrays{T,1,N,ET}
export AbstractArrayOfSimilarVectors

const AbstractVectorOfSimilarVectors{T,ET<:AbstractVector{T}} = AbstractArrayOfSimilarArrays{T,1,1,ET}
export AbstractVectorOfSimilarVectors



"""
    struct SplitSlices{M,N} <: AbstractSlicingMode{M,N}

The split mode of [`ArrayOfSimilarArrays`](@ref): memory-ordered slicing
with `M` inner and `N` outer dimensions.

Constructor:

```
SplitSlices{M,N}()
```

See also [`AbstractSlicingMode`](@ref).
"""
struct SplitSlices{M,N} <: AbstractSlicingMode{M,N} end
export SplitSlices

is_memordered_splitmode(::SplitSlices) = true

getinnerdims(obj::Tuple, ::SplitSlices{M,N}) where {M,N} = front_tuple(obj, Val(M))
getouterdims(obj::Tuple, ::SplitSlices{M,N}) where {M,N} = back_tuple(obj, Val(N))

@inline splitup(A::AbstractArray{T}, ::SplitSlices{M,N}) where {T,M,N} = ArrayOfSimilarArrays{T,M,N}(A)


# Subtypes of AbstractArrayOfSimilarArrays only need to implement `fused`
# (plus the typical array operations), all split-mode operations are derived
# from it:

function fused(A::AbstractArrayOfSimilarArrays)
    throw(ArgumentError("Subtypes of AbstractArrayOfSimilarArrays like $(nameof(typeof(A))) must implement ArraysOfArrays.fused"))
end

@inline getsplitmode(::AbstractArrayOfSimilarArrays{T,M,N}) where {T,M,N} = SplitSlices{M,N}()
@inline unstackmode(A::AbstractArrayOfSimilarArrays) = getsplitmode(A)

@inline stacked(A::AbstractArrayOfSimilarArrays) = fused(A)

@inline Base.parent(A::AbstractArrayOfSimilarArrays) = fused(A)

@inline vecflattened(A::AbstractArrayOfSimilarArrays) = vec(fused(A))
# Disambiguation:
@inline vecflattened(A::AbstractVectorOfSimilarArrays) = vec(fused(A))
@inline vecflattened(A::AbstractArrayOfSimilarVectors) = vec(fused(A))
@inline vecflattened(A::AbstractVectorOfSimilarVectors) = vec(fused(A))

function getslicemap(::AbstractArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    return (_ncolons(Val{M}())..., _oneto_tpl(Val{N}())...)
end

# `stack` must return an independent array, unlike `stacked`. Julia does not
# dispatch on keyword arguments, so dispatch on the value of `dims` instead:
Base.stack(A::AbstractArrayOfSimilarArrays; dims::Union{Integer,Colon} = :) = _stack_impl(A, dims)

# Fast path for the default layout: a single bulk copy instead of the
# element-by-element copy that generic `stack` would do:
_stack_impl(A::AbstractArrayOfSimilarArrays, ::Colon) = copy(fused(A))
_stack_impl(A::AbstractArrayOfSimilarArrays, dims::Integer) = stack(collect(A); dims)

@inline Base.:(==)(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}) where {M,N} = (stacked(A) == stacked(B))
@inline Base.isequal(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}) where {M,N} = isequal(stacked(A), stacked(B))
@inline Base.isapprox(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}; kwargs...) where {M,N} = isapprox(stacked(A), stacked(B); kwargs...)


"""
    ArrayOfSimilarArrays{T,M,N,P,ET} <: AbstractArrayOfSimilarArrays{T,M,N,ET}

Represents a view of an array of dimension `M + N` as an `N`-dimensional
array with elements that are `M`-dimensional arrays. All element arrays
implicitly have equal size/axes.

Constructors:

    ArrayOfSimilarArrays{T,M,N}(data::AbstractArray)
    ArrayOfSimilarArrays{T,M}(data::AbstractArray)

The following type aliases are defined:

* `VectorOfSimilarArrays{T,M} = ArrayOfSimilarArrays{T,M,1}`
* `ArrayOfSimilarVectors{T,N} = ArrayOfSimilarArrays{T,1,N}`
* `VectorOfSimilarVectors{T} = ArrayOfSimilarArrays{T,1,1}`

`VectorOfSimilarArrays` supports `push!()`, etc., provided the underlying
array supports resizing of its last dimension (e.g. an `ElasticArray`).

The nested array can also be created using the function [`sliced`](@ref)
and the wrapped flat array can be accessed using [`flatview`](@ref)
afterwards:

```julia
A_flat = rand(2,3,4,5,6)
A_nested = sliced(A_flat, Val(2))
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

    function ArrayOfSimilarArrays{T,M,N}(data::AbstractArray{T,L}) where {T,M,N,L}
        _require_ndims(Val(L), _add_vals(Val(M), Val(N)))
        P = typeof(data)
        ET = Base.promote_op(view, P, _nColons(Val(M))..., _nInts(Val(N))...)
        new{T,M,N,P,ET}(data)
    end
end

export ArrayOfSimilarArrays


function ArrayOfSimilarArrays{T,M,N}(orig_data::AbstractArray{U,L}) where {T,M,N,U,L}
    conv_data = _convert_eltype(T, orig_data)::AbstractArray{T,L}
    return ArrayOfSimilarArrays{T,M,N}(conv_data)
end

function ArrayOfSimilarArrays{T,M}(data::AbstractArray{U,L}) where {T,M,U,L}
    N = _val_value(_subtract_vals(Val(L), Val(M)))
    ArrayOfSimilarArrays{T,M,N}(data)
end

Base.convert(::Type{ArrayOfSimilarArrays{T,M,N}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = ArrayOfSimilarArrays{T,M,N}(stacked(A))
Base.convert(::Type{ArrayOfSimilarArrays{T}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} = ArrayOfSimilarArrays{T,M,N}(stacked(A))
Base.convert(::Type{ArrayOfSimilarArrays}, A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} = ArrayOfSimilarArrays{T,M,N}(stacked(A))


@inline fused(A::ArrayOfSimilarArrays) = A.data

function Base.Array(A::ArrayOfSimilarArrays{T,M,N,P,ET}) where {T,M,N,P,ET}
    new_ET = Base.promote_op(similar, ET)
    return Array{new_ET,N}(A)
end


Base.size(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} = split_tuple(size(A.data), Val{M}())[2]


Base.@propagate_inbounds Base.getindex(A::ArrayOfSimilarArrays{T,M,N}, idxs::Vararg{Integer,N}) where {T,M,N} =
    view(A.data, _ncolons(Val{M}())..., idxs...)


Base.@propagate_inbounds Base.setindex!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}, idxs::Vararg{Integer,N}) where {T,M,N,U} =
    setindex!(A.data, x, _ncolons(Val{M}())..., idxs...)

function Base.fill!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}) where {T,M,N,U}
    size(x) == innersize(A) || throw(DimensionMismatch("Can't fill ArrayOfSimilarArrays with an array of different size than its elements"))
    A.data .= reshape(x, size(x)..., ntuple(_ -> 1, Val(N))...)
    return A
end

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
    size_inner = front_tuple(size(data), Val{M}())
    # ToDo: Don't use similar if data is an ElasticArray?
    ArrayOfSimilarArrays{U,M,length(dims)}(similar(data, U, size_inner..., dims...))
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


const VectorOfSimilarArrays{
    T, M,
    P<:AbstractArray{T},
    ET<:AbstractArray{T,M}
} = ArrayOfSimilarArrays{T,M,1,P,ET}

export VectorOfSimilarArrays

VectorOfSimilarArrays{T}(data::AbstractArray{U}) where {T,U} = ArrayOfSimilarArrays{T,length(Base.front(size(data))),1}(data)
VectorOfSimilarArrays(data::AbstractArray{T}) where {T} = ArrayOfSimilarArrays{T,length(Base.front(size(data))),1}(data)

Base.convert(::Type{VectorOfSimilarArrays{T}}, A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} = ArrayOfSimilarArrays{T,M,1}(stacked(A))
Base.convert(::Type{VectorOfSimilarArrays}, A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} = ArrayOfSimilarArrays{T,M,1}(stacked(A))


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


# Concatenating vectors of similar arrays concatenates their underlying
# data along its last dimension, with a single allocation:

function _vcat_vosas(Vs)
    isempty(Vs) && throw(ArgumentError("reducing over an empty collection is not allowed"))
    new_data = _cat_lastdim(map(fused, Vs))
    return VectorOfSimilarArrays(new_data)
end

Base.vcat(V1::AbstractVectorOfSimilarArrays, Vs::AbstractVectorOfSimilarArrays...) = _vcat_vosas((V1, Vs...))

Base.reduce(::typeof(vcat), Vs::AbstractVector{<:AbstractVectorOfSimilarArrays}) = _vcat_vosas(Vs)


function _empty_data_size(A::VectorOfSimilarArrays{T,M}) where {T,M}
    size_inner, size_outer = split_tuple(size(A.data), Val{M}())
    empty_size_outer = map(x -> zero(x), size_outer)
    (size_inner..., empty_size_outer...)
end

function Base.empty(A::VectorOfSimilarArrays{T,M}, ::Type{<:AbstractArray{U}}) where {T,M,U}
    new_data_size = _empty_data_size(A)
    # ToDo: Don't use similar if data is an ElasticArray?
    ArrayOfSimilarArrays{U,M,1}(similar(A.data, U, new_data_size...))
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

ArrayOfSimilarVectors{T}(data::AbstractArray{U}) where {T,U} = ArrayOfSimilarArrays{T,1,length(Base.front(size(data)))}(data)
ArrayOfSimilarVectors(data::AbstractArray{T}) where {T} = ArrayOfSimilarArrays{T,1,length(Base.front(size(data)))}(data)

Base.convert(R::Type{ArrayOfSimilarVectors{T}}, A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} = R(stacked(A))
Base.convert(R::Type{ArrayOfSimilarVectors}, A::AbstractArray{<:AbstractVector{T},N}) where {T,N} = R(stacked(A))


const VectorOfSimilarVectors{
    T,
    P<:AbstractArray{T,2},
    ET<:AbstractVector{T}
} = ArrayOfSimilarArrays{T,1,1,P,ET}

export VectorOfSimilarVectors

VectorOfSimilarVectors{T}(data::AbstractArray{U,2}) where {T,U} = ArrayOfSimilarArrays{T,1,1}(data)
VectorOfSimilarVectors(data::AbstractArray{T,2}) where {T} = VectorOfSimilarVectors{T}(data)

Base.convert(R::Type{VectorOfSimilarVectors{T}}, A::AbstractVector{<:AbstractVector{U}}) where {T,U} = R(stacked(A))
Base.convert(R::Type{VectorOfSimilarVectors}, A::AbstractVector{<:AbstractVector{T}}) where {T} = R(stacked(A))


# Fast paths over the flat data for the full (dims = :) reductions. Julia
# does not dispatch on keyword arguments, so dispatch on the value of `dims`
# instead and forward non-Colon dims to the generic implementations:

Base.sum(X::AbstractVectorOfSimilarArrays; dims::Union{Colon,Integer} = :) = _aosa_sum(X, dims)

_aosa_sum(X::AbstractVectorOfSimilarArrays{T,M}, ::Colon) where {T,M} =
    sum(flatview(X); dims = M + 1)[_ncolons(Val{M}())...]
_aosa_sum(X::AbstractVectorOfSimilarArrays, dims::Integer) =
    Base.@invoke sum(X::AbstractArray; dims = dims)

Statistics.mean(X::AbstractVectorOfSimilarArrays; dims::Union{Colon,Integer} = :) = _aosa_mean(X, dims)

_aosa_mean(X::AbstractVectorOfSimilarArrays{T,M}, ::Colon) where {T,M} =
    mean(flatview(X); dims = M + 1)[_ncolons(Val{M}())...]
_aosa_mean(X::AbstractVectorOfSimilarArrays, dims::Integer) =
    Base.@invoke mean(X::AbstractArray; dims = dims)

Statistics.var(X::AbstractVectorOfSimilarArrays; corrected::Bool = true, mean = nothing, dims::Union{Colon,Integer} = :) =
    _aosa_var(X, corrected, mean, dims)

_aosa_var(X::AbstractVectorOfSimilarArrays{T,M}, corrected::Bool, mean, ::Colon) where {T,M} =
    var(flatview(X); dims = M + 1, corrected = corrected, mean = mean)[_ncolons(Val{M}())...]
_aosa_var(X::AbstractVectorOfSimilarArrays, corrected::Bool, mean, dims::Integer) =
    Base.@invoke var(X::AbstractArray; corrected = corrected, mean = mean, dims = dims)

Statistics.std(X::AbstractVectorOfSimilarArrays; corrected::Bool = true, mean = nothing, dims::Union{Colon,Integer} = :) =
    _aosa_std(X, corrected, mean, dims)

_aosa_std(X::AbstractVectorOfSimilarArrays{T,M}, corrected::Bool, mean, ::Colon) where {T,M} =
    std(flatview(X); dims = M + 1, corrected = corrected, mean = mean)[_ncolons(Val{M}())...]
_aosa_std(X::AbstractVectorOfSimilarArrays, corrected::Bool, mean, dims::Integer) =
    Base.@invoke std(X::AbstractArray; corrected = corrected, mean = mean, dims = dims)

Statistics.cov(X::AbstractVectorOfSimilarVectors; corrected::Bool = true) =
    cov(flatview(X); dims = 2, corrected = corrected)

Statistics.cor(X::AbstractVectorOfSimilarVectors) =
    cor(flatview(X); dims = 2)



# Deprecations:

@deprecate ArrayOfSimilarArrays{T,M,N}(A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} ArrayOfSimilarArrays{T,M,N}(stacked(A)) false
@deprecate ArrayOfSimilarArrays{T}(A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} ArrayOfSimilarArrays{T,M,N}(stacked(A)) false
@deprecate ArrayOfSimilarArrays(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} ArrayOfSimilarArrays{T,M,N}(stacked(A)) false

@deprecate Base.convert(::Type{ArrayOfSimilarArrays{T,M,N}}, data::AbstractArray{U,L}) where {T,M,N,U<:Number,L} ArrayOfSimilarArrays{T,M,N}(data) false
@deprecate Base.convert(::Type{ArrayOfSimilarArrays{T,M}}, data::AbstractArray{U,L}) where {T,M,U<:Number,L} ArrayOfSimilarArrays{T,M}(data) false


@deprecate VectorOfSimilarArrays{T}(A::AbstractVector{<:AbstractArray{U,M}}) where {T,M,U} ArrayOfSimilarArrays{T,M,1}(stacked(A)) false
@deprecate VectorOfSimilarArrays(A::AbstractVector{<:AbstractArray{T,M}}) where {T,M} ArrayOfSimilarArrays{T,M,1}(stacked(A)) false

@deprecate Base.convert(::Type{VectorOfSimilarArrays{T}}, data::AbstractArray{U}) where {T,U<:Number} VectorOfSimilarArrays{T}(data) false
@deprecate Base.convert(::Type{VectorOfSimilarArrays}, data::AbstractArray{T}) where {T<:Number} VectorOfSimilarArrays(data) false


@deprecate ArrayOfSimilarVectors{T}(A::AbstractArray{<:AbstractVector{U},N}) where {T,N,U} ArrayOfSimilarArrays{T,1,N}(stacked(A)) false
@deprecate ArrayOfSimilarVectors(A::AbstractArray{<:AbstractVector{T},N}) where {T,N} ArrayOfSimilarArrays{T,1,N}(stacked(A)) false

@deprecate Base.convert(::Type{ArrayOfSimilarVectors{T}}, data::AbstractArray{U}) where {T,U<:Number} ArrayOfSimilarVectors{T}(data) false
@deprecate Base.convert(::Type{ArrayOfSimilarVectors}, data::AbstractArray{T}) where {T<:Number} ArrayOfSimilarVectors(data) false


@deprecate VectorOfSimilarVectors{T}(A::AbstractVector{<:AbstractVector{U}}) where {T,U} ArrayOfSimilarArrays{T,1,1}(stacked(A)) false
@deprecate VectorOfSimilarVectors(A::AbstractVector{<:AbstractVector{T}}) where {T} ArrayOfSimilarArrays{T,1,1}(stacked(A)) false

@deprecate Base.convert(::Type{VectorOfSimilarVectors{T}}, data::AbstractArray{U,2}) where {T,U<:Number} VectorOfSimilarVectors{T}(data) false
@deprecate Base.convert(::Type{VectorOfSimilarVectors}, data::AbstractArray{T,2}) where {T<:Number} VectorOfSimilarVectors(data) false


"""
    nestedview(A::AbstractArray{T,M+N}, M::Integer)
    nestedview(A::AbstractArray{T,2})

Deprecated, use [`sliced`](@ref) instead.
"""
function nestedview end
export nestedview

@deprecate nestedview(A::AbstractArray, M::Integer) sliced(A, Val(M))
@deprecate nestedview(A::AbstractArray, ::Val{M}) where {M} sliced(A, Val(M))
@deprecate nestedview(A::AbstractArray{T,2}) where {T} sliced(A)
