# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStaticArraysCoreExt

using StaticArraysCore: StaticArray, SArray, SVector, SMatrix
using StaticArraysCore: tuple_length, tuple_prod

import ArraysOfArrays
using ArraysOfArrays: StaticSlices, SplitSlices, UnknownSplitMode, splitup, fused
using ArraysOfArrays: _front_tuple, _back_tuple


# StaticSlices is about shape, not type: the element type of the static
# arrays always follows from the element type of the flat array, replacing
# any element type the mode may specify. Specializations cover the SArray
# family, the fallback rebuilds types with a single element-type
# parameter, like subtypes of FieldVector:
_sa_with_eltype(::Type{SArray{S,U,N,L}}, ::Type{T}) where {S,U,N,L,T} = SArray{S,T,N,L}
_sa_with_eltype(::Type{SVector{S}}, ::Type{T}) where {S,T} = SVector{S,T}
_sa_with_eltype(::Type{SMatrix{M,N}}, ::Type{T}) where {M,N,T} = SMatrix{M,N,T,M * N}
_sa_with_eltype(::Type{SArray{S}}, ::Type{T}) where {S<:Tuple,T} = SArray{S,T,tuple_length(S),tuple_prod(S)}

function _sa_with_eltype(::Type{SA}, ::Type{T}) where {SA<:StaticArray,T}
    SAT = try Base.typename(SA).wrapper{T} catch; nothing end
    SAT isa Type && SAT <: StaticArray{<:Tuple,T} && isconcretetype(SAT) ||
        throw(ArgumentError("Cannot determine the static array type of shape $SA with element type $T"))
    return SAT
end

# The reverse: SArray family types become their familiar element-type-free
# aliases, so arrays that differ only in their element type have equal
# split modes. The aliases differ in which type parameters they leave
# free, so each dimensionality needs its own method. Other static array
# types keep their element type:
_sa_without_eltype(::Type{SArray{Tuple{S1},T,1,L}}) where {S1,T,L} = SVector{S1}
_sa_without_eltype(::Type{SArray{Tuple{S1,S2},T,2,L}}) where {S1,S2,T,L} = SMatrix{S1,S2}
_sa_without_eltype(::Type{SArray{S,T,N,L}}) where {S,T,N,L} = SArray{S}
_sa_without_eltype(::Type{SA}) where {SA<:StaticArray} = SA

# ndims of the SArray{S} spelling is not determined by Base.ndims:
ArraysOfArrays.StaticSlices(::Type{SArray{S}}) where {S<:Tuple} = StaticSlices{SArray{S},tuple_length(S)}()


function ArraysOfArrays.splitup(A::AbstractArray{T}, ::StaticSlices{SA0}) where {T,SA0<:StaticArray}
    SA = _sa_with_eltype(SA0, T)
    isbitstype(SA) && sizeof(SA) > 0 || throw(ArgumentError("splitup with a StaticSlices mode requires an isbits element type with a size greater than zero, which $SA is not. For traced or GPU element types, use a SplitSlices mode instead"))
    innsz = size(SA)
    M = ndims(SA)
    ndims(A) >= M && _front_tuple(size(A), Val(M)) == innsz ||
        throw(DimensionMismatch("Size $innsz of static array type does not match the leading dimensions of an array of size $(size(A))"))
    all(r -> isone(first(r)), _back_tuple(axes(A), Val(ndims(A) - M))) ||
        throw(ArgumentError("splitup with a StaticSlices mode requires the outer axes of the array to be one-based"))
    outsz = _back_tuple(size(A), Val(ndims(A) - M))
    return reinterpret(reshape, SA, reshape(A, prod(innsz), outsz...))
end

@inline function ArraysOfArrays._fused_impl(A::AbstractArray{SA}, ::StaticSlices) where {S,T,M,SA<:StaticArray{S,T,M}}
    return reshape(reinterpret(T, A), size(SA)..., size(A)...)
end

# Mutable static arrays are not isbits and cannot be reinterpreted, nested
# arrays of them get the generic nested-array treatment:
@inline function ArraysOfArrays.getsplitmode(A::AbstractArray{SA}) where {S,T,M,SA<:StaticArray{S,T,M}}
    # reinterpret cannot handle mutable or zero-size static arrays:
    if isbitstype(SA) && sizeof(SA) > 0
        StaticSlices(_sa_without_eltype(SA))
    else
        UnknownSplitMode{typeof(A)}()
    end
end

@inline function ArraysOfArrays.unstackmode(A::AbstractArray{SA}) where {S,T,M,SA<:StaticArray{S,T,M}}
    if !all(r -> isone(first(r)), axes(A))
        # splitup of the stacked array cannot reproduce offset outer axes:
        UnknownSplitMode{typeof(A)}()
    elseif isbitstype(SA) && sizeof(SA) > 0
        StaticSlices(_sa_without_eltype(SA))
    else
        SplitSlices{M}()
    end
end

# Memory-ordered, so stacking equals fusing (zero-copy):
@inline ArraysOfArrays._stacked_impl(A::AbstractArray{<:StaticArray}, smode::StaticSlices) = fused(A)

@inline ArraysOfArrays.flatview(A::AbstractArray{SA,N}) where {S,T,M,N,SA<:StaticArray{S,T,M}} =
    reshape(reinterpret(T, A), size(SA)..., size(A)...)

@inline ArraysOfArrays.sliced(A::AbstractArray, SA::Type{<:StaticArray}) = splitup(A, StaticSlices(SA))


# Deprecated:

Base.@deprecate ArraysOfArrays.nestedview(A::AbstractArray{T}, SA::Type{SVector{S,T}}) where {T,S} ArraysOfArrays.sliced(A, SA) false
Base.@deprecate ArraysOfArrays.nestedview(A::AbstractArray{T}, SA::Type{SVector{S}}) where {T,S} ArraysOfArrays.sliced(A, SA) false


end # module ArraysOfArraysStaticArraysCoreExt
