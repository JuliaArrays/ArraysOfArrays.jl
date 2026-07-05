# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

"""
    struct BaseSlicing{M,N,TPL<:Tuple{Vararg{Union{Colon,Int}}}} <: AbstractSlicingMode{M,N}

The split mode of `Base.Slices` (as returned by `eachslice`, `eachcol` and
`eachrow`).

Constructor:

```
BaseSlicing{M,N,TPL}(slicemap::TPL)
```

`slicemap` equals the `slicemap` property of `Base.Slices` objects.

See also [`AbstractSlicingMode`](@ref).
"""
struct BaseSlicing{M,N,TPL<:Tuple{Vararg{Union{Colon,Int}}}} <: AbstractSlicingMode{M,N}
    slicemap::TPL
end
export BaseSlicing

is_memordered_splitmode(::BaseSlicing{1,1,Tuple{Colon,Int}}) = true

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

    # With less than two outer dimensions the result is fully determined by
    # SliceMapT already:
    if length(outdimidxs) < 2
        return Expr(:tuple, [:(obj[$i]) for i in outdimidxs]...)
    end

    # vals[j] is the entry of obj and pos[j] the outer dimension number for
    # the j-th outer dimension of the parent array. The result must be
    # ordered by outer dimension number, so it is vals in inverse-perm order
    # of pos. pos is only known at runtime, its values are not encoded in
    # SliceMapT.
    vals_expr = Expr(:tuple, [:(obj[$i]) for i in outdimidxs]...)
    pos_expr = Expr(:tuple, [:(slicemap[$i]) for i in outdimidxs]...)

    quote
        slicemap = smode.slicemap
        vals = $vals_expr
        pos = $pos_expr
        return _invpermuted(vals, pos)
    end
end

@inline function _invpermuted(vals::NTuple{N,Any}, pos::NTuple{N,Int}) where N
    ntuple(k -> vals[findfirst(==(k), pos)::Int], Val(N))
end


@inline getslicemap(A::Slices) = A.slicemap

@inline function getsplitmode(A::Slices)
    M = ndims(eltype(A))
    N = ndims(A)
    slicemap = getslicemap(A)
    BaseSlicing{M,N,typeof(slicemap)}(slicemap)
end

function splitup(A::AbstractArray, smode::BaseSlicing)
    slicemap = smode.slicemap
    axs = getouterdims(axes(A), smode)
    return Slices(A, slicemap, axs)
end

@inline fused(A::Slices) = parent(A)
