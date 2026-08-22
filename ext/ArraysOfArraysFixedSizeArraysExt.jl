# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysFixedSizeArraysExt

using FixedSizeArrays: FixedSizeVector

import ArraysOfArrays

# FixedSizeVectors cannot be resized, so they can be shared between a
# VectorOfArrays and a split mode without a defensive copy:
ArraysOfArrays._shapeinfo_copy(x::FixedSizeVector) = x

end # module ArraysOfArraysFixedSizeArraysExt
