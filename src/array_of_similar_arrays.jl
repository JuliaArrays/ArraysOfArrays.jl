# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    AbstractArrayOfSimilarArrays{T,M,N,ET<:AbstractArray{T,M}} <: AbstractSlices{ET,N}

An array that contains arrays that have the same size/axes. The array is
internally stored in flattened form as some kind of array of dimension
`M + N`, in memory order. The flattened form can be accessed via
`flatview(A)`.

# Implementation

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
    struct SplitSlices{M} <: AbstractSlicingMode{M}

The split mode of [`ArrayOfSimilarArrays`](@ref): memory-ordered slicing
with `M` inner dimensions.

Constructor:

```
SplitSlices{M}()
```

See also [`AbstractSlicingMode`](@ref).
"""
struct SplitSlices{M} <: AbstractSlicingMode{M} end
export SplitSlices

is_memordered_splitmode(::SplitSlices) = true

getinnerdims(obj::Tuple, ::SplitSlices{M}) where {M} = _front_tuple(obj, Val(M))
getouterdims(obj::NTuple{L,Any}, ::SplitSlices{M}) where {L,M} = _back_tuple(obj, Val(L - M))

@inline function splitup(A::AbstractArray{T,L}, ::SplitSlices{M}) where {T,L,M}
    M isa Integer && 0 <= M <= L || throw(ArgumentError("Cannot split an array with $L dimensions using $M inner dimensions"))
    return ArrayOfSimilarArrays{T,M,L-M}(A)
end


# Stacking restores SplitSlices splits, and unlike fused it also accepts
# generic nested arrays. Restricting the nesting dimensionalities catches
# mode mismatches at dispatch time, stack rejects ragged elements:
(f::FuseArrays{SplitSlices{M}})(A::AbstractArray{<:AbstractArray{<:Any,M}}) where {M} = stacked(A)

function (f::FuseArrays{SplitSlices{M}})(A::AbstractArray) where {M}
    throw(ArgumentError("Inverse of SplitSlices{$M} requires an array of $M-dimensional arrays"))
end


# All split-mode operations derive from `fused`:

function fused(A::AbstractArrayOfSimilarArrays)
    throw(ArgumentError("Subtypes of AbstractArrayOfSimilarArrays like $(nameof(typeof(A))) must implement ArraysOfArrays.fused"))
end

@inline getsplitmode(::AbstractArrayOfSimilarArrays{T,M}) where {T,M} = SplitSlices{M}()
@inline unstackmode(A::AbstractArrayOfSimilarArrays) = getsplitmode(A)

@inline Base.parent(A::AbstractArrayOfSimilarArrays) = fused(A)

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

# Per-element reductions reduce over the inner dimensions of the flat data:
function _innermapreduce_impl(f, op, init, A::AbstractArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    data = fused(A)
    dims = ntuple(identity, Val(M))
    r = if init isa _NoInit
        mapreduce(f, op, data; dims = dims)
    else
        mapreduce(f, op, data; dims = dims, init = init)
    end
    return reshape(r, size(A))
end


# Comparisons must be equivalent to elementwise comparison: empty arrays
# with equal outer axes are equal even if their element sizes differ.
@inline Base.:(==)(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}) where {M,N} =
    axes(A) == axes(B) && (isempty(A) || stacked(A) == stacked(B))
@inline Base.isequal(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}) where {M,N} =
    axes(A) == axes(B) && (isempty(A) || isequal(stacked(A), stacked(B)))
@inline Base.isapprox(A::AbstractArrayOfSimilarArrays{<:Any,M,N}, B::AbstractArrayOfSimilarArrays{<:Any,M,N}; kwargs...) where {M,N} =
    axes(A) == axes(B) && (isempty(A) || isapprox(stacked(A), stacked(B); kwargs...))


"""
    ArrayOfSimilarArrays{T,M,N,P,ET} <: AbstractArrayOfSimilarArrays{T,M,N,ET}

Represents a view of an array of dimension `M + N` as an `N`-dimensional
array with elements that are `M`-dimensional arrays. All element arrays
implicitly have equal size/axes.

User code should typically not instantiate `ArrayOfSimilarArrays` directly,
but use [`splitup`](@ref) with a [`SplitSlices`](@ref) mode.

# Implementation

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
        _require_one_based_outer(data, Val(N))
        P = typeof(data)
        ET = Base.promote_op(view, P, _nColons(Val(M))..., _nInts(Val(N))...)
        new{T,M,N,P,ET}(data)
    end
end

# The outer dimensions of A are indexed directly into the outer dimensions
# of the underlying data, so their axes must agree:
@inline function _require_one_based_outer(data::AbstractArray, ::Val{N}) where N
    if !all(map(isone ∘ first, _back_tuple(axes(data), Val(N))))
        throw(ArgumentError("ArrayOfSimilarArrays requires the outer axes of the underlying data to be one-based"))
    end
    nothing
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


Base.size(A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N} = _back_tuple(size(A.data), Val{N}())


Base.@propagate_inbounds Base.getindex(A::ArrayOfSimilarArrays{T,M,N}, idxs::Vararg{Integer,N}) where {T,M,N} =
    view(A.data, _ncolons(Val{M}())..., idxs...)


Base.@propagate_inbounds Base.setindex!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}, idxs::Vararg{Integer,N}) where {T,M,N,U} =
    setindex!(A.data, x, _ncolons(Val{M}())..., idxs...)

# For zero-dimensional elements the generic method would degenerate into a
# scalar assignment of the element array itself:
Base.@propagate_inbounds Base.setindex!(A::ArrayOfSimilarArrays{T,0,N}, x::AbstractArray{U,0}, idxs::Vararg{Integer,N}) where {T,N,U} =
    setindex!(A.data, x[], idxs...)

function Base.fill!(A::ArrayOfSimilarArrays{T,M,N}, x::AbstractArray{U,M}) where {T,M,N,U}
    size(x) == innersize(A) || throw(DimensionMismatch("Can't fill ArrayOfSimilarArrays with an array of different size than its elements"))
    # reshape rebases to one-based axes, so element arrays with offset axes
    # can be filled from a value of matching size but different axes:
    reshape(A.data, size(A.data)) .= reshape(x, size(x)..., ntuple(_ -> 1, Val(N))...)
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


# Requests for element types of a different dimensionality than M fall
# through to the generic similar implementations:
# The elements of an ArrayOfSimilarArrays are views, so similar can only
# preserve the structure when the requested element type is the element
# type of A itself, as reached via Base's similar(A) and similar(A, dims).
# All other element types go to the generic implementation:
function Base.similar(A::ArrayOfSimilarArrays{T,M,N,P,ET}, ::Type{ET}, dims::Dims) where {T,M,N,P,ET}
    # ToDo: Don't use similar if data is an ElasticArray?
    ArrayOfSimilarArrays{T,M,length(dims)}(similar(A.data, T, innersize(A)..., dims...))
end


function Base.copyto!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    innersize(dest) != innersize(src) && throw(DimensionMismatch("Can't copy, shape of element arrays of source and dest are not equal"))
    copyto!(dest.data, src.data)
    dest
end


function Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    innersize(dest) != innersize(src) && throw(DimensionMismatch("Can't append, shape of element arrays of source and dest are not equal"))
    append!(dest.data, src.data)
    dest
end

Base.append!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    append!(dest, convert(ArrayOfSimilarArrays, src))


function Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::ArrayOfSimilarArrays{U,M,N}) where {T,M,N,U}
    innersize(dest) != innersize(src) && throw(DimensionMismatch("Can't prepend, shape of element arrays of source and dest are not equal"))
    prepend!(dest.data, src.data)
    dest
end

Base.prepend!(dest::ArrayOfSimilarArrays{T,M,N}, src::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U} =
    prepend!(dest, convert(ArrayOfSimilarArrays, src))

# Like Base's map/broadcast of identity over vectors of views, the result
# shares the element data. ArrayOfSimilarArrays is an immutable wrapper
# with no outer state of its own, so a fresh wrapper of the same data
# would be egal to A anyway:
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
    size(x) != innersize(V) && throw(DimensionMismatch("Can't push, shape of source and elements of target is incompatible"))
    append!(V.data, x)
    V
end

function Base.pop!(V::VectorOfSimilarArrays)
    isempty(V) && throw(ArgumentError("array must be non-empty"))
    # Copy, a view would alias data beyond the new size:
    x = copy(V[end])
    resize!(V, size(V, 1) - 1)
    x
end

function Base.pushfirst!(V::VectorOfSimilarArrays{T,M}, x::AbstractArray{U,M}) where {T,M,U}
    size(x) != innersize(V) && throw(DimensionMismatch("Can't pushfirst, shape of source and elements of target is incompatible"))
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


function Base.empty(A::VectorOfSimilarArrays{T,M}, ::Type{<:AbstractArray{U}}) where {T,M,U}
    # ToDo: Don't use similar if data is an ElasticArray?
    ArrayOfSimilarArrays{U,M,1}(similar(A.data, U, innersize(A)..., 0))
end

Base.empty!(A::VectorOfSimilarArrays) = resize!(A, 0)



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

# Reductions that combine per-element reductions with the same operation
# run over the flat data in a single pass. Empty inputs and empty elements
# go to the generic implementation, to preserve its errors:
for (f, op) in ((:maximum, :max), (:minimum, :min))
    @eval function Base.mapreduce(::typeof($f), ::typeof($op), A::AbstractArrayOfSimilarArrays; kwargs...)
        if isempty(kwargs) && !isempty(A) && prod(innersize(A)) > 0
            $f(fused(A))
        else
            invoke(mapreduce, Tuple{typeof($f),typeof($op),AbstractArray}, $f, $op, A; kwargs...)
        end
    end
end

# The flat sum is Base's (pairwise) sum over the flat data, so for
# floating-point elements it may differ from the element-wise grouping by
# the last bits, like any reassociated sum:
function Base.mapreduce(::typeof(sum), ::typeof(+), A::AbstractArrayOfSimilarArrays; kwargs...)
    if isempty(kwargs) && !isempty(A)
        sum(fused(A))
    else
        invoke(mapreduce, Tuple{typeof(sum),typeof(+),AbstractArray}, sum, +, A; kwargs...)
    end
end


_flatreduce(f, X::AbstractVectorOfSimilarArrays{T,M}, ::Colon; kwargs...) where {T,M} =
    f(flatview(X); dims = M + 1, kwargs...)[_ncolons(Val{M}())...]
_flatreduce(f, X::AbstractVectorOfSimilarArrays, dims::Integer; kwargs...) =
    invoke(f, Tuple{AbstractArray}, X; dims = dims, kwargs...)

Base.sum(X::AbstractVectorOfSimilarArrays; dims::Union{Colon,Integer} = :) =
    _flatreduce(sum, X, dims)

Statistics.mean(X::AbstractVectorOfSimilarArrays; dims::Union{Colon,Integer} = :) =
    _flatreduce(mean, X, dims)

Statistics.var(X::AbstractVectorOfSimilarArrays; corrected::Bool = true, mean = nothing, dims::Union{Colon,Integer} = :) =
    _flatreduce(var, X, dims; corrected = corrected, mean = mean)

Statistics.std(X::AbstractVectorOfSimilarArrays; corrected::Bool = true, mean = nothing, dims::Union{Colon,Integer} = :) =
    _flatreduce(std, X, dims; corrected = corrected, mean = mean)

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
