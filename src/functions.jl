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
