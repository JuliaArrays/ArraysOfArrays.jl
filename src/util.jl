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
Base.@pure _nColons(::Val{N}) where N = ntuple(_ -> Colon, Val{N}())

@inline _oneto_tpl(::Val{N}) where N = ntuple(identity, Val{N}())
Base.@pure _nInts(::Val{N}) where N = ntuple(_ -> Int, Val{N}())


Base.@propagate_inbounds front_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    Base.ntuple(i -> x[i], Val{M}())

Base.@propagate_inbounds back_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    Base.ntuple(i -> x[i + N - M], Val{M}())

Base.@propagate_inbounds split_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    (front_tuple(x, Val{M}()), back_tuple(x, Val{N - M}()))

Base.@propagate_inbounds swap_front_back_tuple(x::NTuple{N,Any}, ::Val{M}) where {N,M} =
    (back_tuple(x, Val{N - M}())..., front_tuple(x, Val{M}())...)


_convert_elype(::Type{T}, A::AbstractArray{T}) where {T} = A

_convert_elype(::Type{T}, A::AbstractArray{U}) where {T,U} = broadcast(Base.Fix1(convert, T), A)


Base.@pure _add_vals(::Val{A}, ::Val{B}) where {A,B} = Val{A + B}()

Base.@pure require_ndims(A::AbstractArray{T,N}, Val_N::Val{N}) where {T,N} =
    nothing

Base.@pure require_ndims(A::AbstractArray{T,M}, Val_N::Val{N}) where {T,M,N} =
    throw(ArgumentError("Require an array with $N dimensions"))

