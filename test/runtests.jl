# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "Package ArraysOfArrays" begin
    include("functions.jl")
    include("array_of_similar_arrays.jl")
    include("vector_of_arrays.jl")
end
