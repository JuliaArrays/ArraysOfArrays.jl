# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays


@testset "functions" begin
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
    end


    @testset "innersize" begin
        @test @inferred(innersize([[1, 2, 3], [4, 5, 6]])) == (3,)
        @test @inferred(innersize([[]])) == (0,)
        @test @inferred(innersize([2:5])) == (4,)
        @test @inferred(innersize((2:5,))) == (4,)
        @test @inferred(innersize(Ref(2:5))) == (4,)
        @test_throws DimensionMismatch @inferred(innersize([[1, 2, 3], [4, 5]]))
    end


    @testset "abstract_nestedarray_type" begin
        @test @inferred(abstract_nestedarray_type(Int, Val(()))) == Int
        @test @inferred(abstract_nestedarray_type(Int, Val((2,)))) == AbstractArray{Int, 2}
        @test @inferred(abstract_nestedarray_type(Float32, Val((2,3,4)))) ==
            AbstractArray{<:AbstractArray{<:AbstractArray{Float32, 4}, 3}, 2}
    end
end
