# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysStructArraysExt

using StructArrays: StructArrayStyle
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

function Base.copy(bc::Broadcasted{StructArrayStyle{S,N}}) where {S<:AbstractNestedArrayStyle,N}
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
