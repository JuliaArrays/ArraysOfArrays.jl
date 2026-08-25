# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysReactantStaticArraysCoreExt

using GPUArraysCore: @allowscalar
using Reactant: TracedRNumber
using StaticArraysCore: StaticArray

import ArraysOfArrays
using ArraysOfArrays: AbstractArrayOfSimilarArrays, StaticSlices
using ArraysOfArrays: fused, getsplitmode
using ArraysOfArrays: _front_tuple, _back_tuple, _ncolons


# Traced arrays are symbolic and have no memory that could be
# reinterpreted, so splitting them with StaticSlices returns a lazy view
# instead: element access constructs the static arrays from traced
# scalars, while bulk operations (fused, stacked, flatview, reductions,
# broadcasts) work on the underlying flat array via the
# AbstractArrayOfSimilarArrays machinery and so lower to tensor
# operations:
struct StaticSlicesView{
    T<:TracedRNumber,M,N,
    SA<:StaticArray{<:Tuple,T},
    P<:AbstractArray{T}
} <: AbstractArrayOfSimilarArrays{T,M,N,SA}
    data::P
end

@inline ArraysOfArrays.fused(B::StaticSlicesView) = B.data

@inline ArraysOfArrays.getsplitmode(B::StaticSlicesView{T,M,N,SA}) where {T,M,N,SA} =
    StaticSlices(ArraysOfArrays._sa_without_eltype(SA))

# Disambiguation against the slices-based methods of the package and the
# eltype-based methods of the StaticArraysCore extension:
@inline ArraysOfArrays.unstackmode(B::StaticSlicesView) = getsplitmode(B)
@inline ArraysOfArrays._stacked_impl(B::StaticSlicesView, ::StaticSlices) = B.data
@inline ArraysOfArrays.flatview(B::StaticSlicesView) = B.data

Base.size(B::StaticSlicesView{T,M,N}) where {T,M,N} = _back_tuple(size(B.data), Val(N))

Base.IndexStyle(::Type{<:StaticSlicesView}) = IndexCartesian()

Base.@propagate_inbounds function Base.getindex(B::StaticSlicesView{T,M,N,SA}, I::Vararg{Int,N}) where {T,M,N,SA}
    elem = view(B.data, _ncolons(Val(M))..., I...)
    # Scalar reads of traced arrays emit IR operations, they never cause
    # a device synchronization, so they are exempt from allowscalar:
    return @allowscalar SA(ntuple(i -> elem[i], Val(length(SA))))
end


function ArraysOfArrays.splitup(A::AbstractArray{<:TracedRNumber}, ::StaticSlices{SA0}) where {SA0<:StaticArray}
    SA = ArraysOfArrays._sa_with_eltype(SA0, eltype(A))
    innsz = size(SA)
    M = ndims(SA)
    ndims(A) >= M && _front_tuple(size(A), Val(M)) == innsz ||
        throw(DimensionMismatch("Size $innsz of static array type does not match the leading dimensions of an array of size $(size(A))"))
    N = ndims(A) - M
    return StaticSlicesView{eltype(A),M,N,SA,typeof(A)}(A)
end


end # module ArraysOfArraysReactantStaticArraysCoreExt
