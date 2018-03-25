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


@inline function _split_tuple_impl(x::NTuple{N,Any}, y::Tuple, ::Val{N}) where {N}
    x, y
end

@inline function _split_tuple_impl(x::NTuple{M,Any}, y::Tuple, ::Val{N}) where {M, N}
    _split_tuple_impl((x..., y[1]), _tail(y), Val{N}())
end

@inline split_tuple(x, ::Val{N}) where {N} = _split_tuple_impl((), x, Val{N}())
