# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysInverseFunctionsExt

import InverseFunctions
using InverseFunctions: NoInverse

using ArraysOfArrays: AbstractSplitMode, NonSplitMode, UnknownSplitMode, FuseArrays

InverseFunctions.inverse(smode::AbstractSplitMode) = FuseArrays(smode)
InverseFunctions.inverse(f::FuseArrays) = f.smode

# NonSplitMode acts as the identity on arrays of its dimensionality:
InverseFunctions.inverse(smode::NonSplitMode) = smode

# UnknownSplitMode does not support fused:
InverseFunctions.inverse(smode::UnknownSplitMode) = NoInverse(smode)

end # module ArraysOfArraysInverseFunctionsExt
