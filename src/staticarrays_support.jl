# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


@inline flatview(A::AbstractArray{SA,N}) where {S,T,M,N,SA<:StaticArrays.StaticArray{S,T,M}} =
    reshape(reinterpret(T, A), size(SA)..., size(A)...)


@inline function nestedview(A::AbstractArray{T}, SA::Type{StaticArrays.SVector{S,T}}) where {T,S}
    size_A = size(A)
    size_A[1] == S || throw(DimensionMismatch("Length $S of static vector type does not match first dimension of array of size $size_A"))
    reshape(reinterpret(SA, A), _tail(size_A)...)
end

@inline nestedview(A::AbstractArray{T}, ::Type{StaticArrays.SVector{S}}) where {T,S} =
    nestedview(A, StaticArrays.SVector{S,T})
