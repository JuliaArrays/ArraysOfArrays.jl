# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import ArraysOfArrays

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(ArraysOfArrays))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        ArraysOfArrays,
        ambiguities = true
    )
end # testset
