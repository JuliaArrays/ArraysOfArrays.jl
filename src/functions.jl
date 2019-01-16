# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    innermap(f::Base.Callable, A::AbstractArray{<:AbstractArray})

Nested `map` at depth 2. Equivalent to `map(X -> map(f, X) A)`.
"""
function innermap end
export innermap

innermap(f::Base.Callable, A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} =
    map(X -> map(f, X), A)



"""
    deepmap(f::Base.Callable, x::Any)
    deepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray{<:...}})

Applies `map` at the deepest possible layer of nested arrays. If `A` is not
a nested array, `deepmap` behaves identical to `Base.map`.
"""
function deepmap end
export deepmap

deepmap(f::Base.Callable, x::Any) = map(f, x)

deepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray}) =
    map(X -> deepmap(f, X), A)


"""
    flatview(A::AbstractArray)
    flatview(A::AbstractArray{<:AbstractArray{<:...}})

View array `A` in a suitable flattened form. The shape of the flattened form
will depend on the type of `A`. If the `A` is not a nested array, the return
value is `A` itself. When no type-specific method is available, `flatview`
will fall back to `Base.Iterators.flatten`.
"""
function flatview end
export flatview

@inline flatview(A::AbstractArray) = A

# TODO: Implement flatview on generic nested arrays via new `FlatView`, using
# deepgetindex to implement getindex, etc.
@inline flatview(A::AbstractArray{<:AbstractArray}) = Base.Iterators.flatten(A)


"""
    nestedview(A::AbstractArray{T,M+N}, M::Integer)
    nestedview(A::AbstractArray{T,2})

AbstractArray{<:AbstractArray{T,M},N}

View array `A` in as an `M`-dimensional array of `N`-dimensional arrays by
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


"""
    innersize(A:AbstractArray{<:AbstractArray}, [dim])

Returns the size of the element arrays of `A`. Fails if the element arrays
are not of equal size.
"""
function innersize end
export innersize

function innersize(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    s = if !isempty(A)
        let sz_A = size(first(A))
            ntuple(i -> Int(sz_A[i]), Val(M))
        end
    else
        ntuple(_ -> zero(Int), Val(M))
    end

    let s = s
        if any(X -> size(X) != s, A)
            throw(DimensionMismatch("Shape of element arrays of A is not equal, can't determine common shape"))
        end
    end

    s
end

@inline innersize(A::AbstractArray{<:AbstractArray}, dim::Integer) =
    innersize(A)[dim]


"""
    deepgetindex(A::AbstractArray, idxs...)
    deepgetindex(A::AbstractArray{<:AbstractArray, N}, idxs...) where {N}

Recursive `getindex` on flat or nested arrays. If A is an array of arrays,
uses the first `N` entries in `idxs` on `A`, then the rest on the inner
array(s). If A is not a nested array, `deepgetindex` is equivalent to
`getindex`.

See also [`deepsetindex!`](@ref) and [`deepview`](@ref).
"""
function deepgetindex end
export deepgetindex

Base.@propagate_inbounds deepgetindex(A::AbstractArray{T,N}, idxs::Vararg{Any,N}) where {T,N} = getindex(A, idxs...)
Base.@propagate_inbounds deepgetindex(A::AbstractArray{<:AbstractArray,N}, idxs::Vararg{Any,N}) where {N} = getindex(A, idxs...)

Base.@propagate_inbounds function deepgetindex(A::AbstractArray{<:AbstractArray,N}, idxs...) where {N}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    _deepgetindex_impl(A, idxs_outer, idxs_inner)
end

Base.@propagate_inbounds _deepgetindex_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Real}, idxs_inner::Tuple) where {N} =
    deepgetindex(getindex(A, idxs_outer...), idxs_inner...)

Base.@propagate_inbounds _deepgetindex_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Any}, idxs_inner::Tuple) where {N} =
    _deepgetindex_tupled.(view(A, idxs_outer...), (idxs_inner,))

Base.@propagate_inbounds _deepgetindex_tupled(A::AbstractArray, idxs::Tuple) = deepgetindex(A, idxs...)


"""
    deepsetindex!(A::AbstractArray, x, idxs...)
    deepsetindex!(A::AbstractArray{<:AbstractArray,N}, x, idxs...) where {N}

Recursive `setindex!` on flat or nested arrays. If A is an array of arrays,
uses the first `N` entries in `idxs` on `A`, then the rest on the inner
array(s). If A is not a nested array, `deepsetindex!` is equivalent to
`setindex!`.

See also [`deepgetindex`](@ref) and [`deepview`](@ref).
"""
function deepsetindex! end
export deepsetindex!

Base.@propagate_inbounds deepsetindex!(A::AbstractArray{T,N}, x, idxs::Vararg{Any,N}) where {T,N} = setindex!(A, x, idxs...)
Base.@propagate_inbounds deepsetindex!(A::AbstractArray{<:AbstractArray,N}, x, idxs::Vararg{Any,N}) where {N} = setindex!(A, x, idxs...)

Base.@propagate_inbounds function deepsetindex!(A::AbstractArray{<:AbstractArray,N}, x, idxs...) where {N}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    _deepsetindex_impl!(A, x, idxs_outer, idxs_inner)
    A
end

Base.@propagate_inbounds function _deepsetindex_impl!(A::AbstractArray{<:AbstractArray}, x, idxs_outer::NTuple{N,Real}, idxs_inner::Tuple) where {N}
    B = getindex(A, idxs_outer...)
    deepsetindex!(B, x, idxs_inner...)
end

Base.@propagate_inbounds function _deepsetindex_impl!(A::AbstractArray{<:AbstractArray}, x, idxs_outer::NTuple{N,Any}, idxs_inner::Tuple) where {N}
    B = view(A, idxs_outer...)
    for i in eachindex(B, x)
        deepsetindex!(B[i], x[i], idxs_inner...)
    end
end



"""
    deepview(A::AbstractArray, idxs...)
    deepview(A::AbstractArray{<:AbstractArray, N}, idxs...) where {N}

Recursive `view` on flat or nested arrays. If A is an array of arrays,
uses the first `N` entries in `idxs` on `A`, then the rest on the inner
array(s). If A is not a nested array, `deepview` is equivalent to `view`.

See also [`deepgetindex`](@ref) and [`deepsetindex!`](@ref).
"""
function deepview end
export deepview

Base.@propagate_inbounds deepview(A::AbstractArray{T,N}, idxs::Vararg{Any,N}) where {T,N} = view(A, idxs...)
Base.@propagate_inbounds deepview(A::AbstractArray{<:AbstractArray,N}, idxs::Vararg{Any,N}) where {N} = view(A, idxs...)

Base.@propagate_inbounds function deepview(A::AbstractArray{<:AbstractArray,N}, idxs...) where {N}
    idxs_outer, idxs_inner = split_tuple(idxs, Val{N}())
    _deepview_impl(A, idxs_outer, idxs_inner)
end

Base.@propagate_inbounds _deepview_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Real}, idxs_inner::NTuple{M,Real}) where {N,M} =
    deepview(getindex(A, idxs_outer...), idxs_inner...)

Base.@propagate_inbounds _deepview_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Real}, idxs_inner::NTuple{M,Any}) where {N,M} =
    deepview(getindex(A, idxs_outer...), idxs_inner...)

Base.@propagate_inbounds _deepview_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Any}, idxs_inner::NTuple{M,Real}) where {N,M} =
    throw(ArgumentError("deepview not supported yes with outer indices $idxs_outer and inner indices $idxs_inner"))

Base.@propagate_inbounds _deepview_impl(A::AbstractArray{<:AbstractArray}, idxs_outer::NTuple{N,Any}, idxs_inner::NTuple{M,Any}) where {N,M} =
    _deepview_tupled.(view(A, idxs_outer...), (idxs_inner,))

Base.@propagate_inbounds _deepview_tupled(A::AbstractArray, idxs::Tuple) = deepview(A, idxs...)


"""
    abstract_nestedarray_type(T_inner::Type, ::Val{ndims_tuple})

Return the type of nested `AbstractArray`s. `T_inner` specifies the element
type of the innermost layer of arrays, `ndims_tuple` specifies the
dimensionality of each nesting layer (outer arrays first).

If `ndims_tuple` is empty, the returns is the (typically scalar) type
`T_inner` itself.
"""
function abstract_nestedarray_type end
export abstract_nestedarray_type


Base.@pure function abstract_nestedarray_type(::Type{T_inner}, outer::Val{ndims_tuple}) where {T_inner,ndims_tuple}
    _abstract_nestedarray_type_impl(T_inner, ndims_tuple...)
end

Base.@pure _abstract_nestedarray_type_impl(::Type{T_inner}) where {T_inner} = T_inner

Base.@pure _abstract_nestedarray_type_impl(::Type{T_inner}, N) where {T_inner} = AbstractArray{T_inner, N}

Base.@pure _abstract_nestedarray_type_impl(::Type{T_inner}, N, M, ndims_tuple...) where {T_inner} =
    AbstractArray{<:_abstract_nestedarray_type_impl(T_inner, M, ndims_tuple...), N}
