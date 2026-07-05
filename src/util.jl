# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


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


_convert_eltype(::Type{T}, A::AbstractArray{T}) where {T} = A

_convert_eltype(::Type{T}, A::AbstractArray{U}) where {T,U} = broadcast(Base.Fix1(convert, T), A)


Base.@pure _add_vals(::Val{A}, ::Val{B}) where {A,B} = Val{A + B}()

Base.@pure _subtract_vals(::Val{A}, ::Val{B}) where {A,B} = Val{A - B}()

@inline _require_ndims(::Val{N}, ::Val{N}) where {N} = nothing

function _require_ndims(::Val{N1}, ::Val{N2}) where {N1,N2}
    throw(ArgumentError("Require an array with $N2 dimensions, but got an array with $N1 dimensions"))
end

Base.@pure _val_value(::Val{x}) where x = x
