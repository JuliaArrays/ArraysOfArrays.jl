# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, Thunk, unthunk, @thunk, @non_differentiable

using ArraysOfArrays: ArrayOfSimilarArrays
using ArraysOfArrays: flatview


struct _MappedMaybeThunk{F, T} <: AbstractThunk
    f::F
    x::T
end
ChainRulesCore.unthunk(mt::_MappedMaybeThunk) = mt.f(unthunk(mt.x))
mapthunk(f::F, x::T) where {F,T} = _MappedMaybeThunk{F,T}(f, x)
mapthunk(::Type{F}, x::T) where {F,T} = _MappedMaybeThunk{Type{F},T}(F, x)


@non_differentiable getsplitmode(::Any)
@non_differentiable innersize(::Any)
@non_differentiable is_memordered_splitmode(::Any)


function ChainRulesCore.rrule(::typeof(splitview), A::AbstractArray, ::NonSplitMode)
    return flatview(A), _partview_pullback
end
_unpart_partview_pullback(ΔΩ) = NoTangent(), ΔΩ, NoTangent()

function ChainRulesCore.rrule(::typeof(flatview), A::AbstractArray)
    return flatview(A), _unpart_flatview_pullback
end
_unpart_flatview_pullback(ΔΩ) = NoTangent(), ΔΩ


function ChainRulesCore.rrule(::typeof(splitview), A::AbstractArray, partmode::AbstractSlicingMode)
    return splitview(A, partmode), _partview_pullback
end
_partview_pullback(ΔΩ) = NoTangent(), mapthunk(flatview, ΔΩ), NoTangent()

function ChainRulesCore.rrule(::typeof(flatview), A::AbstractArray{<:AbstractArray})
    smode = getsplitmode(A)
    return flatview(A), Base.Fix2(_flatview_pullback, smode)
end
_flatview_pullback(ΔΩ, smode) = NoTangent(), mapthunk(Base.Fix2(splitview, smode), ΔΩ)



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
