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
struct BaseSlicing{N,TPL<:Tuple{Vararg{Union{Colon,Int},N}}} <: AbstractSlicingMode
    slicemap::TPL
end


@inline getsplitmode(A::Slices) = BaseSlicing(A.slicemap)
@inline unsplitview(A::Slices) = parent(A)


@inline stacked(A::Slices) = reshape(parent(A), (length(A), prod(size(parent(A))) ÷ length(A)))

@inline innersize(A::AbstractSlices) = getinnnerdims(size(flatview(A)), getsplitmode(A))


@inline @generated function getinnnerdims(obj::Tuple, smode::BaseSlicing{N,SliceMapT}) where {N,SliceMapT}
    # slicemap may be something like (Colon(), 2, Colon(), 1, Colon()),
    # extract only the elements of obj where the slicemap is a Colon.
    expr = Expr(:tuple)
    slicepars = SliceMapT.parameters
    for i in 1:length(slicepars)
        if slicepars[i] <: Colon
            push!(expr.args, :(obj[$i]))
        end
    end
    return expr
end


@inline @generated function getouterdims(obj::Tuple, smode::BaseSlicing{N,SliceMapT}) where {N,SliceMapT}
    # slicemap may be something like (Colon(), 2, Colon(), 1, Colon()),
    # extract only the elements of obj where the slicemap is a Colon.
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


function is_memordered_splitmode(smode::BaseSlicing)
    slicemap = smode.slicemap
    dims = _oneto_tpl(Val(length(slicemap)))
    issorted((_extract_innerdims(dims, slicemap)..., _extract_outerdims(dims, slicemap)...))
end
