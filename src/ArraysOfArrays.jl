# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ArraysOfArrays

using Compat
using Compat.Markdown
using Compat: axes

include("util.jl")
include("array_of_similar_arrays.jl")
include("jaggedarray.jl")

end # module
