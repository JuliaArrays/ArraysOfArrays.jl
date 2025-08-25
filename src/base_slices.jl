# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    struct BaseSlicing{N,TPL<:Tuple{Vararg{Int,N}}} <: AbstractSlicingMode

The split mode of `Slices`.

Constructor:

```
BaseSlicing(slicemap::Tuple{Vararg{Union{Colon,Int}}}})
```

`slicemap` equals the `slicemap` property of `Base.Slice` objects.

See also [`AbstractSlicingMode`](@ref).
"""
struct BaseSlicing{M,N,TPL<:Tuple{Vararg{Union{Colon,Int}}}} <: AbstractSlicingMode{M,N}
    slicemap::TPL
end
export BaseSlicing

function is_memordered_splitmode(smode::BaseSlicing{M,N}) where {M,N}
    dims = _oneto_tpl(Val(M+N))
    issorted((getinnerdims(dims, smode)..., getouterdims(dims, smode)...))
end


@inline @generated function getinnerdims(obj::Tuple, ::BaseSlicing{M,N,SliceMapT}) where {M,N,SliceMapT}
    expr = Expr(:tuple)
    slicepars = SliceMapT.parameters
    for i in 1:length(slicepars)
        if slicepars[i] <: Colon
            push!(expr.args, :(obj[$i]))
        end
    end
    return expr
end


@inline @generated function getouterdims(obj::Tuple, smode::BaseSlicing{M,N,SliceMapT}) where {M,N,SliceMapT}
    slicepars = SliceMapT.parameters
    outdimidxs = Int[]
    for i in 1:length(slicepars)
        if slicepars[i] <: Integer
            push!(outdimidxs, i)
        end
    end

    outdimidxs_expr = Expr(:tuple, outdimidxs...)
    idxorder_expr = Expr(:tuple, [:(slicemap[$i]) for i in outdimidxs]...)
    result_expr = Expr(:tuple, [:(obj[outdimidxs[idxorder[$i]]]) for i in eachindex(outdimidxs)]...)

    quote
        slicemap = smode.slicemap
        outdimidxs = $outdimidxs_expr
        idxorder = $idxorder_expr
        return $result_expr
    end
end


@inline function getsplitmode(A::Slices)
    M = ndims(eltype(A))
    N = ndims(A)
    slicemap = A.slicemap
    BaseSlicing{M,N,typeof(slicemap)}(slicemap)
end

function splitview(A::AbstractArray, smode::BaseSlicing)
    slicemap = smode.slicemap
    axs = getouterdims(axes(A), smode)
    return Slices(A, slicemap, axs)
end

@inline joinedview(A::Slices) = parent(A)


@inline stacked(A::Slices) = _stacked_slices_impl(A, getsplitmode(A))

_stacked_slices_impl(A::Slices, ::BaseSlicing{1,1,Tuple{Colon,Int}}) = joinedview(A)

function _stacked_slices_impl(A::Slices, smode::BaseSlicing{M,N,SliceMapT}) where {M,N,SliceMapT}
    A_joined = joinedview(A)
    if is_memordered_splitmode(smode)
        return A_joined
    else
        dimorder = (getinnerdims(_dimstpl(A_joined), smode)..., getouterdims(_dimstpl(A_joined), smode)...)
        return permutedims(A_joined, dimorder)::typeof(A_joined)
    end
end


@inline innersize(A::AbstractSlices) = getinnerdims(size(joinedview(A)), getsplitmode(A))
