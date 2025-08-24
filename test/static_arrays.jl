# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays


@testset "StaticArray Extension" begin
    @testset "flatview and nestedview" begin
        A = [(@SArray randn(3, 2, 4)) for i in 1:2, j in 1:2]
        @test @inferred(nestedview(flatview(A), Val(3))) == A

        B = rand(3, 2, 4)
        @test @inferred(nestedview(flatview(B), SVector{3})) == @inferred(nestedview(B, Val(1)))
    end
end
