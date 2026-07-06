# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStaticArraysCoreExt

import StaticArraysCore
using StaticArraysCore: StaticArray, SVector

import ArraysOfArrays
using ArraysOfArrays: sliced


@inline ArraysOfArrays.flatview(A::AbstractArray{SA,N}) where {S,T,M,N,SA<:StaticArray{S,T,M}} =
    reshape(reinterpret(T, A), size(SA)..., size(A)...)


@inline function ArraysOfArrays.sliced(A::AbstractArray{T}, SA::Type{SVector{S,T}}) where {T,S}
    size_A = size(A)
    size_A[1] == S || throw(DimensionMismatch("Length $S of static vector type does not match first dimension of array of size $size_A"))
    reshape(reinterpret(SA, A), ArraysOfArrays._tail(size_A)...)
end

@inline ArraysOfArrays.sliced(A::AbstractArray{T}, ::Type{SVector{S}}) where {T,S} =
    sliced(A, SVector{S,T})


# Deprecated:

Base.@deprecate ArraysOfArrays.nestedview(A::AbstractArray{T}, SA::Type{SVector{S,T}}) where {T,S} ArraysOfArrays.sliced(A, SA) false
Base.@deprecate ArraysOfArrays.nestedview(A::AbstractArray{T}, SA::Type{SVector{S}}) where {T,S} ArraysOfArrays.sliced(A, SA) false


end # module ArraysOfArraysStaticArraysCoreExt
