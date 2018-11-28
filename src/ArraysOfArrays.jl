# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

__precompile__(true)

module ArraysOfArrays

using Requires
using Statistics
using UnsafeArrays

include("util.jl")
include("functions.jl")
include("array_of_similar_arrays.jl")
include("vector_of_arrays.jl")


function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("staticarrays_support.jl")
    @require StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" include("statsbase_support.jl")
end


end # module
