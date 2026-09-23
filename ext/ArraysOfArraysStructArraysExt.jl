# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStructArraysExt

using StructArrays: StructArray, StructArrayStyle, components
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
# NestedArrayStyle) does so as well. A sliced StructArray keeps its declared
# element type, which may not fit the in-memory blocks of disk-backed
# columns, so blocks are rebuilt from the column blocks. The element type
# of a block is that of its elements if the declared one is concrete, so
# that results infer the same way as for in-memory data:
ArraysOfArrays._block_length(A::StructArray) = ArraysOfArrays._bcast_blocklength(values(components(A)))

function ArraysOfArrays._read_block(A::StructArray{T}, idxs...) where {T}
    cols = map(c -> ArraysOfArrays._slice_block(c, idxs...), components(A))
    T <: Union{Tuple,NamedTuple} && return StructArray(cols)
    B = StructArray{Base.typename(T).wrapper}(values(cols))
    return isconcretetype(T) ? StructArray{typeof(first(B))}(values(cols)) : B
end

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
