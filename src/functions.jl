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
    deepmap(f::Base.Callable, A::AbstractArray)
    deepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray{<:...}})

Applies `map` at the deepest possible layer of nested arrays.
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

flatview(A::AbstractArray) = A

flatview(A::AbstractArray{<:AbstractArray}) = Base.Iterators.flatten(A)
