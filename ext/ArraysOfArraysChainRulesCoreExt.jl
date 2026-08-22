# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, AbstractThunk, AbstractZero, unthunk, @thunk, @non_differentiable

using ArraysOfArrays: getsplitmode, is_memordered_splitmode, splitup, fused, stacked, unstackmode,
    flatview, innersize, vecflattened, partitioned, innersum, _scalar_first_last
using ArraysOfArrays: getslicemap, getinnerdims, getouterdims, consgrouped_ptrs
using ArraysOfArrays: _convert_eltype
using ArraysOfArrays: NonSplitMode, AbstractSplitMode, AbstractSlicingMode
using ArraysOfArrays: AbstractArrayOfSimilarArrays, ArrayOfSimilarArrays, VectorOfArrays


struct _MappedMaybeThunk{F, T} <: AbstractThunk
    f::F
    x::T
end
ChainRulesCore.unthunk(mt::_MappedMaybeThunk) = mt.f(unthunk(mt.x))
mapthunk(f::F, x::T) where {F,T} = _MappedMaybeThunk{F,T}(f, x)
mapthunk(f, x::AbstractZero) = x


# Functions that only query shape/structure information, in sync with the
# zero-derivative rules of the Mooncake extension:
@non_differentiable getsplitmode(::Any)
@non_differentiable unstackmode(::Any)
@non_differentiable is_memordered_splitmode(::Any)
@non_differentiable innersize(::Any)
@non_differentiable innersize(::Any, ::Any)
@non_differentiable getslicemap(::Any)
@non_differentiable getinnerdims(::Any, ::Any)
@non_differentiable getouterdims(::Any, ::Any)
@non_differentiable consgrouped_ptrs(::Any)


function ChainRulesCore.rrule(::typeof(splitup), A::AbstractArray, smode::NonSplitMode)
    return splitup(A, smode), _nosplit_pullback
end
_nosplit_pullback(ΔΩ) = NoTangent(), ΔΩ, NoTangent()

function ChainRulesCore.rrule(::typeof(splitup), A::AbstractArray, smode::AbstractSplitMode)
    function splitup_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ, NoTangent()
        return NoTangent(), @thunk(_splitup_cotangent(unthunk(ΔΩ), A, smode)), NoTangent()
    end
    return splitup(A, smode), splitup_pullback
end

# The cotangent of the nested output may be a nested array of any type
# (e.g. a plain vector of vectors), its backing data need not be aligned
# with the primal data, and the primal data may contain regions not covered
# by any element. So the element cotangents are scattered into a zero array
# of the primal's shape:
function _scatter_cotangent(Δ, A::AbstractArray, smode::AbstractSplitMode)
    ΔA = fill!(similar(A), zero(eltype(A)))
    splitup(ΔA, smode) .= Δ
    return ΔA
end

_splitup_cotangent(Δ, A::AbstractArray, smode::AbstractSplitMode) = _scatter_cotangent(Δ, A, smode)

# Slicings cover their data completely, so equally-sliced cotangents can be
# reused directly:
function _splitup_cotangent(Δ::AbstractArray{<:AbstractArray}, A::AbstractArray, smode::AbstractSlicingMode)
    getsplitmode(Δ) == smode && axes(fused(Δ)) == axes(A) && return _convert_eltype(eltype(A), fused(Δ))
    return _scatter_cotangent(Δ, A, smode)
end



# Fusing and stacking are identity-like on non-nested arrays, and split the
# cotangent again for nested ones:
_passthrough_pullback(ΔΩ) = NoTangent(), ΔΩ
_resplit_pullback(ΔΩ, smode) = NoTangent(), mapthunk(Base.Fix2(splitup, smode), ΔΩ)

ChainRulesCore.rrule(::typeof(fused), A::AbstractArray) = fused(A), _passthrough_pullback

function ChainRulesCore.rrule(::typeof(fused), A::AbstractArray{<:AbstractArray})
    return fused(A), Base.Fix2(_resplit_pullback, getsplitmode(A))
end


ChainRulesCore.rrule(::typeof(stacked), A::AbstractArray) = stacked(A), _passthrough_pullback

function ChainRulesCore.rrule(::typeof(stacked), A::AbstractArray{<:AbstractArray})
    return stacked(A), Base.Fix2(_resplit_pullback, unstackmode(A))
end
function ChainRulesCore.rrule(::typeof(stack), A::AbstractArrayOfSimilarArrays)
    return stack(A), Base.Fix2(_resplit_pullback, unstackmode(A))
end



# reshape rebases to one-based axes, so tangents for flat data with offset
# axes are built on an array that has the axes of the primal data:
function _unflatten_like(Δ::AbstractVector, data::AbstractArray)
    all(isone ∘ first, axes(data)) && return reshape(_convert_eltype(eltype(data), Δ), size(data))
    Δdata = similar(data)
    copyto!(reshape(Δdata, length(Δdata)), Δ)
    return Δdata
end

function ChainRulesCore.rrule(::typeof(vecflattened), A::AbstractArrayOfSimilarArrays)
    smode = getsplitmode(A)
    data = fused(A)
    function aosa_vecflattened_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ
        return NoTangent(), splitup(_unflatten_like(unthunk(ΔΩ), data), smode)
    end
    return vecflattened(A), aosa_vecflattened_pullback
end

# The cotangent of a flattened view is one-based and covers only the data
# that the elements cover, so it must be scattered back into the index space
# of the primal data, which may have offset axes and uncovered regions. Like
# the other tangents that these rules allocate, it takes the element type of
# the primal, independently of how much of the data the elements cover:
function _scatter_flat_cotangent(Δ::AbstractVector, data::AbstractVector, covered_from::Integer)
    axes(Δ) == axes(data) && return _convert_eltype(eltype(data), Δ)
    padded = fill!(similar(data), zero(eltype(data)))
    copyto!(padded, covered_from, Δ, firstindex(Δ), length(Δ))
    return padded
end

function ChainRulesCore.rrule(::typeof(vecflattened), A::VectorOfArrays)
    smode = getsplitmode(A)  # makes a defensive copy, safe to close over
    data = A.data
    covered_from, _ = _scalar_first_last(A.elem_ptr)
    function voa_vecflattened_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ
        Δdata = _scatter_flat_cotangent(unthunk(ΔΩ), data, covered_from)
        return NoTangent(), splitup(Δdata, smode)
    end
    return vecflattened(A), voa_vecflattened_pullback
end


function ChainRulesCore.rrule(::typeof(partitioned), A::AbstractVector, lengths::AbstractVector{<:Integer})
    return partitioned(A, lengths), _partitioned_pullback_for(A)
end

function ChainRulesCore.rrule(::typeof(partitioned), A::AbstractVector, shapes::AbstractVector{<:Dims})
    return partitioned(A, shapes), _partitioned_pullback_for(A)
end

function _partitioned_pullback_for(A::AbstractVector)
    function partitioned_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ, NoTangent()
        Δflat = vecflattened(unthunk(ΔΩ))
        ΔA = _scatter_flat_cotangent(Δflat, A, firstindex(A))
        return NoTangent(), ΔA, NoTangent()
    end
    return partitioned_pullback
end


# Each element of the innersum cotangent expands uniformly over the inner
# dimensions of the tangent:
function ChainRulesCore.rrule(::typeof(innersum), A::AbstractArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    smode = getsplitmode(A)
    data = fused(A)
    function innersum_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ
        Δ = unthunk(ΔΩ)
        Δdata = similar(data)
        # reshape rebases to one-based axes, so flat data with offset axes
        # can be filled from the one-based cotangent:
        reshape(Δdata, size(Δdata)) .= reshape(Δ, ntuple(_ -> 1, Val(M))..., size(Δ)...)
        return NoTangent(), splitup(Δdata, smode)
    end
    return innersum(A), innersum_pullback
end


function _aosa_ctor_fromflat_pullback(ΔΩ)
    ΔΩ isa AbstractZero && return NoTangent(), ΔΩ
    NoTangent(), flatview(convert(ArrayOfSimilarArrays, unthunk(ΔΩ)))
end

function ChainRulesCore.rrule(::Type{ArrayOfSimilarArrays{T,M,N}}, flat_data::AbstractArray{U}) where {T,M,N,U}
    return ArrayOfSimilarArrays{T,M,N}(flat_data), _aosa_ctor_fromflat_pullback
end

function ChainRulesCore.rrule(::Type{ArrayOfSimilarArrays{T,M,N}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U}
    return ArrayOfSimilarArrays{T,M,N}(A), _passthrough_pullback
end


function ChainRulesCore.rrule(::typeof(flatview), A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    function flatview_pullback(ΔΩ)
        ΔΩ isa AbstractZero && return NoTangent(), ΔΩ
        data = unthunk(ΔΩ)
        NoTangent(), ArrayOfSimilarArrays{eltype(data),M,N}(data)
    end
    
    return flatview(A), flatview_pullback
end


end # module ArraysOfArraysChainRulesCoreExt
