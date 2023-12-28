# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStaticArraysCoreExt

import StaticArraysCore
using StaticArraysCore: StaticArray, SVector

import ArraysOfArrays
using ArraysOfArrays: nestedview


@inline ArraysOfArrays.flatview(A::AbstractArray{SA,N}) where {S,T,M,N,SA<:StaticArray{S,T,M}} =
    reshape(reinterpret(T, A), size(SA)..., size(A)...)


@inline function ArraysOfArrays.nestedview(A::AbstractArray{T}, SA::Type{SVector{S,T}}) where {T,S}
    size_A = size(A)
    size_A[1] == S || throw(DimensionMismatch("Length $S of static vector type does not match first dimension of array of size $size_A"))
    reshape(reinterpret(SA, A), ArraysOfArrays._tail(size_A)...)
end

@inline ArraysOfArrays.nestedview(A::AbstractArray{T}, ::Type{SVector{S}}) where {T,S} =
    nestedview(A, SVector{S,T})


end # module ArraysOfArraysStaticArraysCoreExt
