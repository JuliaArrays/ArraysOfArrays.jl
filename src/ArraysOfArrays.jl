# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    module ArraysOfArrays

Efficient storage and handling of nested arrays.

ArraysOfArrays provides two different types of nested arrays:
[`ArrayOfSimilarArrays`](@ref section_ArrayOfSimilarArrays) and
[`VectorOfArrays`](@ref section_VectorOfArrays).
"""
module ArraysOfArrays

using Statistics

include("util.jl")
include("functions.jl")
include("array_of_similar_arrays.jl")
include("vector_of_arrays.jl")
include("broadcasting.jl")

@static if !isdefined(Base, :get_extension)
    include("../ext/ArraysOfArraysAdaptExt.jl")
    include("../ext/ArraysOfArraysChainRulesCoreExt.jl")
    include("../ext/ArraysOfArraysStaticArraysCoreExt.jl")
end

end # module
