# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


#=

function _split_dims(dims::NTuple{N,Integer}) where {N}
    int_dims = Int.(dims)
    Base.front(int_dims), int_dims[end]
end

=#

@inline _tail_impl(x, ys...) = (ys...,)
@inline _tail(x) = _tail_impl(x...)


Base.@pure _ncolons(::Val{N}) where N = ntuple(_ -> Colon(), Val{N}())


Base.@propagate_inbounds front_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    Base.ntuple(i -> x[i], Val{M}())

Base.@propagate_inbounds back_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    Base.ntuple(i -> x[i + N - M], Val{M}())

Base.@propagate_inbounds split_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    (front_tuple(x, Val{M}()), back_tuple(x, Val{N - M}()))

Base.@propagate_inbounds swap_front_back_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    (back_tuple(x, Val{N - M}())..., front_tuple(x, Val{M}())...)


_convert_elype(::Type{T}, A::AbstractArray{T}) where {T} = A

_convert_elype(::Type{T}, A::AbstractArray{U}) where {T,U} = broadcast(x -> convert(T, x), A)


Base.@pure _add_vals(Val_M::Val{M}, Val_N::Val{N}) where {M,N} =
    Val{length((ntuple(identity, Val_M)..., ntuple(identity, Val_N)...))}()


Base.@pure require_ndims(A::AbstractArray{T,N}, Val_N::Val{N}) where {T,N} =
    nothing

Base.@pure require_ndims(A::AbstractArray{T,M}, Val_N::Val{N}) where {T,M,N} =
    throw(ArgumentError("Require an array with $N dimensions"))
