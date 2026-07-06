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


# Internal sentinel for "no init value given":
struct _NoInit end


# Concatenate arrays of equal ndims and equal size except in their last
# dimension along their last dimension, with a single allocation:
_cat_lastdim(datas) = _cat_lastdim_impl(datas, Val(ndims(first(datas))))

function _cat_lastdim_impl(datas, ::Val{N}) where {N}
    data1 = first(datas)
    inner_sz = Base.front(size(data1))
    foreach(datas) do d
        ndims(d) == N && Base.front(size(d)) == inner_sz || throw(DimensionMismatch("Can't concatenate arrays with different sizes in their non-last dimensions along their last dimension"))
    end

    T = mapreduce(eltype, promote_type, datas)
    n_lastdim = sum(d -> size(d, N), datas)
    result = similar(data1, T, (inner_sz..., n_lastdim))

    colons = ntuple(_ -> :, Val(N - 1))
    offset = firstindex(result, N)
    for d in datas
        n = size(d, N)
        result[colons..., offset:(offset + n - 1)] = d
        offset += n
    end
    return result
end
