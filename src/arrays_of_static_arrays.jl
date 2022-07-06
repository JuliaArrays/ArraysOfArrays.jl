# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


@inline flatview(A::AbstractArray{SA,N}) where {S,T,M,N,SA<:StaticArraysCore.StaticArray{S,T,M}} =
    reshape(reinterpret(T, A), size(SA)..., size(A)...)


@inline function nestedview(A::AbstractArray{T}, SA::Type{StaticArraysCore.SVector{S,T}}) where {T,S}
    size_A = size(A)
    size_A[1] == S || throw(DimensionMismatch("Length $S of static vector type does not match first dimension of array of size $size_A"))
    reshape(reinterpret(SA, A), _tail(size_A)...)
end

@inline nestedview(A::AbstractArray{T}, ::Type{StaticArraysCore.SVector{S}}) where {T,S} =
    nestedview(A, StaticArraysCore.SVector{S,T})
