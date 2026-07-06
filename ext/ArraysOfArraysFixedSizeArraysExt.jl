# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysFixedSizeArraysExt

using FixedSizeArrays: FixedSizeVector

import ArraysOfArrays

# FixedSizeVectors cannot be resized, so sharing them between a
# VectorOfArrays and a split mode is safe and no defensive copy is
# required:
ArraysOfArrays._shapeinfo_copy(x::FixedSizeVector) = x

end # module ArraysOfArraysFixedSizeArraysExt
