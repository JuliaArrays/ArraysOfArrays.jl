# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysFixedSizeArraysExt

using FixedSizeArrays: FixedSizeArray, FixedSizeVector

import ArraysOfArrays

# FixedSizeVectors cannot be resized, so they can be shared between a
# VectorOfArrays and a split mode without a defensive copy:
ArraysOfArrays._shapeinfo_copy(x::FixedSizeVector) = x

# Fixed-size arrays are packed into a nested array by outer broadcasts like
# Arrays if their underlying memory is host memory. Mem may be any
# DenseVector, including device vectors, so the decision is delegated to
# it (see NestedArrayStyle):
ArraysOfArrays._host_storage(::Type{<:FixedSizeArray{<:Any,<:Any,Mem}}) where {Mem} =
    ArraysOfArrays._host_storage(Mem)

end # module ArraysOfArraysFixedSizeArraysExt
