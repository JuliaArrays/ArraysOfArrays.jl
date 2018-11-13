# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    nestedmap2(f::Base.Callable, A::AbstractArray{<:AbstractArray})

Nested `map` at depth 2. Equivalent to `map(X -> map(f, X) A)`.
"""
function nestedmap2 end
export nestedmap2

nestedmap2(f::Base.Callable, A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N} =
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

@inline flatview(A::AbstractArray{<:AbstractArray}) = Base.Iterators.flatten(A)


"""
    nestedview(A::AbstractArray{T,M+N}, M::Integer)

AbstractArray{<:AbstractArray{T,M},N}

View array `A` in as an `M`-dimensional array of `N`-dimensional arrays by
wrapping it into an [`ArrayOfSimilarArrays`](@ref).
"""
function nestedview end
export nestedview

@inline nestedview(A::AbstractArray{T,L}, M::Integer) where {T,L} =
    ArrayOfSimilarArrays{T,M}(A)


"""
    innersize(A:AbstractArray{<:AbstractArray}, [dim])

Returns the size of the element arrays of `A`. Fails if the element arrays
are not of equal size.
"""
function innersize end
export innersize

function innersize(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    s = if !isempty(A)
        sz_A = size(first(A))
        ntuple(i -> Int(sz_A[i]), Val(M))
    else
        ntuple(_ -> zero(Int), Val(M))
    end

    all(X -> size(X) == s, A) || throw(DimensionMismatch("Shape of element arrays of A is not equal, can't determine common shape"))
    s
end

@inline innersize(A::AbstractArray{<:AbstractArray}, dim::Integer) =
    innersize(A)[dim]
