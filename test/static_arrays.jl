# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays


@testset "StaticArray Extension" begin
    @testset "flatview and sliced" begin
        A = [(@SArray randn(3, 2, 4)) for i in 1:2, j in 1:2]
        @test @inferred(sliced(flatview(A), Val(3))) == A

        B = rand(3, 2, 4)
        @test @inferred(sliced(flatview(B), SVector{3})) == @inferred(sliced(B, Val(1)))
        @test_deprecated nestedview(flatview(B), SVector{3})
    end
end
