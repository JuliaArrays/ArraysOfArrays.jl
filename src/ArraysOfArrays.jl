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

using Adapt
using Requires
using Statistics
using UnsafeArrays
using ChainRulesCore

include("util.jl")
include("functions.jl")
include("array_of_similar_arrays.jl")
include("vector_of_arrays.jl")


function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("staticarrays_support.jl")
    @require StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" include("statsbase_support.jl")
end


end # module
