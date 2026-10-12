# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStructArraysExt

using StructArrays: StructArray, StructArrayStyle, components, staticschema
using Base.Broadcast: Broadcast, Broadcasted

import ArraysOfArrays
using ArraysOfArrays: AbstractNestedArrayStyle

# StructArrays treats parent styles that specialize copy, like the nested
# styles, as non-native: it replaces StructArray arguments by broadcasts
# that rebuild the declared element type from the columns and hands the
# result to the parent style, which loses the struct-of-arrays result and
# bypasses specialized getindex methods of the StructArray. Instead,
# struct-valued results take StructArrays' native path and all others the
# nested-array path, both with the original arguments (so the in-place
# methods of the nested styles must accept StructArray arguments, which
# the Base fallbacks do).

# Like StructArrays.isnonemptystructtype, but only for concrete types
# (non-concrete result types have no definite fields), and arrays (mutable
# structs since Julia 1.11) and types are not records:
_struct_result(::Type{T}) where {T} =
    isconcretetype(T) && isstructtype(T) && fieldcount(T) != 0 && !(T <: AbstractArray) && !(T <: Type)

# A StructArray with columns that request block-wise evaluation (see
# NestedArrayStyle) does so too if its blocks can be rebuilt from the
# column blocks:
ArraysOfArrays._block_length(A::StructArray{T}) where {T} =
    _block_struct_type(T, values(components(A))) === nothing ? nothing :
    ArraysOfArrays._min_block_length(values(components(A))...)

ArraysOfArrays._block_eltype(A::StructArray{T}) where {T} = something(_block_struct_type(T, values(components(A))), T)

ArraysOfArrays._read_block(A::StructArray, idxs...) = _read_structarray_block(A, idxs...)
ArraysOfArrays._read_block(A::StructArray{<:Any,0}) = _read_structarray_block(A)

function _read_structarray_block(A::StructArray{T}, idxs...) where {T}
    cols = values(components(A))
    return StructArray{_block_struct_type(T, cols)}(map(c -> ArraysOfArrays._slice_block(c, idxs...), cols))
end

# The element type of the blocks: the declared element type may name the
# element types of disk-backed columns, which differ from those of their
# in-memory blocks. Such field types are replaced by those of the blocks,
# including in type parameters. Everything else in the declared type is
# preserved. If the resulting type cannot hold the blocks, there is no
# block element type. This folds at compile time:
function _block_struct_type(::Type{T}, cols::Tuple) where {T}
    isconcretetype(T) || return T
    fts = fieldtypes(staticschema(T))
    new_fts = map((ft, c) -> ft === eltype(c) ? ArraysOfArrays._block_eltype(c) : ft, fts, cols)
    new_fts === fts && return T
    R = _reparameterized(T, fts, new_fts)
    return R !== nothing && fieldtypes(staticschema(R)) === new_fts ? R : nothing
end

_reparameterized(::Type{<:Tuple}, fts, new_fts) = Tuple{new_fts...}
_reparameterized(::Type{T}, fts, new_fts) where {T<:NamedTuple} = NamedTuple{fieldnames(T), Tuple{new_fts...}}
function _reparameterized(::Type{T}, fts, new_fts) where {T}
    params = _type_params(T)
    isempty(params) && return nothing
    return Base.typename(T).wrapper{map(p -> _new_param(p, fts, new_fts), params)...}
end

# Julia has no public API for the parameters of a type:
@generated _type_params(::Type{T}) where {T} = Expr(:tuple, T.parameters...)

_new_param(p, ::Tuple{}, ::Tuple{}) = p
_new_param(p, fts::Tuple, new_fts::Tuple) =
    first(fts) === p && first(new_fts) !== p ? first(new_fts) : _new_param(p, Base.tail(fts), Base.tail(new_fts))

function Base.copy(bc::Broadcasted{StructArrayStyle{S,N}}) where {S<:AbstractNestedArrayStyle,N}
    blocked = ArraysOfArrays._copy_in_blocks(bc)
    blocked === nothing || return blocked
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if _struct_result(ElType)
        return invoke(copy, Tuple{Broadcasted}, bc)
    else
        return copy(convert(Broadcasted{S}, bc))
    end
end

Base.copyto!(dest::AbstractArray, bc::Broadcasted{StructArrayStyle{S,N}}) where {S<:AbstractNestedArrayStyle,N} =
    copyto!(dest, convert(Broadcasted{S}, bc))

Broadcast.materialize!(::StructArrayStyle{S}, dest, bc::Broadcasted) where {S<:AbstractNestedArrayStyle} =
    Broadcast.materialize!(S(), dest, bc)

end # module ArraysOfArraysStructArraysExt
