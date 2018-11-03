# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ArraysOfArrays

using UnsafeArrays

include("util.jl")
include("functions.jl")
include("array_of_similar_arrays.jl")
include("vector_of_arrays.jl")

end # module
