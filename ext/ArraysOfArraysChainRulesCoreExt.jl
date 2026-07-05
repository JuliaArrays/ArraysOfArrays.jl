# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, AbstractThunk, Thunk, unthunk, @thunk, @non_differentiable

using ArraysOfArrays: getsplitmode, is_memordered_splitmode, splitup, fused, stacked, unstackmode,
    flatview, innersize
using ArraysOfArrays: NonSplitMode, AbstractSplitMode
using ArraysOfArrays: AbstractArrayOfSimilarArrays, ArrayOfSimilarArrays


struct _MappedMaybeThunk{F, T} <: AbstractThunk
    f::F
    x::T
end
ChainRulesCore.unthunk(mt::_MappedMaybeThunk) = mt.f(unthunk(mt.x))
mapthunk(f::F, x::T) where {F,T} = _MappedMaybeThunk{F,T}(f, x)
mapthunk(::Type{F}, x::T) where {F,T} = _MappedMaybeThunk{Type{F},T}(F, x)


@non_differentiable getsplitmode(::Any)
@non_differentiable unstackmode(::Any)
@non_differentiable innersize(::Any)
@non_differentiable is_memordered_splitmode(::Any)


function ChainRulesCore.rrule(::typeof(splitup), A::AbstractArray, smode::NonSplitMode)
    return splitup(A, smode), _nosplit_pullback
end
_nosplit_pullback(ΔΩ) = NoTangent(), ΔΩ, NoTangent()

function ChainRulesCore.rrule(::typeof(splitup), A::AbstractArray, smode::AbstractSplitMode)
    return splitup(A, smode), _splitview_pullback
end
_splitview_pullback(ΔΩ) = NoTangent(), mapthunk(fused, ΔΩ), NoTangent()



ChainRulesCore.rrule(::typeof(fused), A::AbstractArray) = fused(A), _nofuse_pullback
_nofuse_pullback(ΔΩ) = NoTangent(), ΔΩ

function ChainRulesCore.rrule(::typeof(fused), A::AbstractArray{<:AbstractArray})
    return fused(A), Base.Fix2(_fused_pullback, getsplitmode(A))
end
_fused_pullback(ΔΩ, smode) = NoTangent(), mapthunk(Base.Fix2(splitup, smode), ΔΩ)


ChainRulesCore.rrule(::typeof(stacked), A::AbstractArray) = stacked(A), _nostack_pullback
_nostack_pullback(ΔΩ) = NoTangent(), ΔΩ

function ChainRulesCore.rrule(::typeof(stacked), A::AbstractArray{<:AbstractArray})
    return stacked(A), Base.Fix2(_stacked_pullback, unstackmode(A))
end
function ChainRulesCore.rrule(::typeof(stack), A::AbstractArrayOfSimilarArrays)
    return stack(A), Base.Fix2(_stacked_pullback, unstackmode(A))
end
_stacked_pullback(ΔΩ, smode) = NoTangent(), mapthunk(Base.Fix2(splitup, smode), ΔΩ)



function _aosa_ctor_fromflat_pullback(ΔΩ)
    NoTangent(), flatview(convert(ArrayOfSimilarArrays, unthunk(ΔΩ)))
end

function ChainRulesCore.rrule(::Type{ArrayOfSimilarArrays{T,M,N}}, flat_data::AbstractArray{U}) where {T,M,N,U}
    return ArrayOfSimilarArrays{T,M,N}(flat_data), _aosa_ctor_fromflat_pullback
end

_aosa_ctor_fromnested_pullback(ΔΩ) = NoTangent(), ΔΩ

function ChainRulesCore.rrule(::Type{ArrayOfSimilarArrays{T,M,N}}, A::AbstractArray{<:AbstractArray{U,M},N}) where {T,M,N,U}
    return ArrayOfSimilarArrays{T,M,N}(A), _aosa_ctor_fromnested_pullback
end


function ChainRulesCore.rrule(::typeof(flatview), A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    function flatview_pullback(ΔΩ)
        data = unthunk(ΔΩ)
        NoTangent(), ArrayOfSimilarArrays{eltype(data),M,N}(data)
    end
    
    return flatview(A), flatview_pullback
end


end # module ArraysOfArraysChainRulesCoreExt
