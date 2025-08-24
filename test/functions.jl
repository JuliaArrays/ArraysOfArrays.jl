# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays


@testset "functions" begin
    Aes1 = eachslice(rand(5,6,7,8,9); dims = (4,5))
    Aes2 = eachslice(rand(5,6,7,8,9); dims = (4,2))

    function gen_nested()
        A11 = [1 2; 3 4]
        A21 = [4 5 6; 7 8 9]
        A12 = [10 11; 12 13; 14 15]
        A22 = [16 17; 18 19]
        hcat([A11, A21], [A12, A22])
    end


    @testset "flatview and nestedview" begin
        A = [(@SArray randn(3, 2, 4)) for i in 1:2, j in 1:2]
        @test @inferred(nestedview(flatview(A), Val(3))) == A

        B = rand(3, 2, 4)
        @test @inferred(nestedview(flatview(B), SVector{3})) == @inferred(nestedview(B, Val(1)))

        @test @inferred flatview(Aes1) == parent(Aes1)
        @test_throws ArgumentError @inferred flatview(Aes2)
    end


    @testset "getslicemap" begin
        @test @inferred(getslicemap(Aes1)) == (:, :, :, 1, 2)
        @test @inferred(getslicemap(Aes2)) == (:, 2, :, 1, :)
    end


    @testset "innersize" begin
        @test @inferred(innersize(rand(3,4,5))) == ()
        @test @inferred(innersize([[1, 2, 3], [4, 5, 6]])) == (3,)
        @test @inferred(innersize([[]])) == (0,)
        @test @inferred(innersize([2:5])) == (4,)
        @test @inferred(innersize((2:5,))) == (4,)
        @test @inferred(innersize(Ref(2:5))) == (4,)
        @test_throws DimensionMismatch @inferred(innersize([[1, 2, 3], [4, 5]]))

        @test @inferred(innersize(Aes1)) == (5, 6, 7)
        @test @inferred(innersize(Aes2)) == (5, 7, 9)
    end
end
