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
value is `A` itself. Only specific types of nested arrays are supported.
"""
function flatview end
export flatview

@inline flatview(A::AbstractArray) = A
@inline flatview(A::AbstractArray{<:AbstractArray}) = throw(ArgumentError("flatview not implemented nested arrays of type $(nameof(typeof(A)))"))


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

@inline innersize(tpl::Tuple{T}) where T = size(only(tpl))
@inline innersize(ref::Ref) = size(only(ref))


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
