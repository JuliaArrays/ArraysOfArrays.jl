# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ArraysOfArrays" begin
    include("test_aqua.jl")
    include("functions.jl")
    include("array_of_similar_arrays.jl")
    include("vector_of_arrays.jl")
    include("broadcasting.jl")
    include("test_docs.jl")
end # testset
