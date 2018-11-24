# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test


@testset "functions" begin
    function gen_nested()
        A11 = [1 2; 3 4]
        A21 = [4 5 6; 7 8 9]
        A12 = [10 11; 12 13; 14 15]
        A22 = [16 17; 18 19]
        hcat(Array[A11, A21], Array[A12, A22])
    end


    @testset "deepgetindex" begin
        A = gen_nested()
        @test @inferred(deepgetindex(A, 1, 2)) === A[1, 2]
        @test @inferred(deepgetindex(A, 1, 2, 2, 1)) == A[1, 2][2, 1]
        @test @inferred(deepgetindex(A, 1, 2, 1:2, 1)) == A[1, 2][1:2, 1]
        @test @inferred(deepgetindex(A, 1, 1:2, 2, 1)) == [A[1, 1][2, 1], A[1, 2][2, 1]]
        @test @inferred(deepgetindex(A, 1, 1:2, 1:2, 2)) == [A[1, 1][1:2, 2], A[1, 2][1:2, 2]]
        @test_throws MethodError deepgetindex(A, 1, 2, 2, 1, 2)

        B = rand(3, 4, 5)
        @test @inferred(deepgetindex(B, 2, 3, 4)) === getindex(B, 2, 3, 4)
        @test_throws MethodError deepgetindex(B, 6)
        @test_throws MethodError deepgetindex(B, 2, 3)
    end


    @testset "deepsetindex!" begin
        B12 = [20 21 22; 23 24 25]
        X = [[41, 42], [43, 44]]

        A = gen_nested()
        @test @inferred(deepsetindex!(A, B12, 1, 2)) === A
        @test A[1, 2] == B12

        A = gen_nested()
        @test @inferred(deepsetindex!(A, 42, 1, 2, 2, 1)) === A
        @test A[1, 2][2, 1] == 42

        A = gen_nested()
        @test @inferred(deepsetindex!(A, X, 1, 1:2, 1:2, 2)) === A
        @test deepgetindex(A, 1, 1:2, 1:2, 2) == X
    end


    @testset "deepview" begin
        A = gen_nested()
        @test @inferred(deepview(A, 1, 2)) == view(A, 1, 2)
        @test @inferred(deepview(A, 1, 2, 2, 1)) == view(A[1, 2], 2, 1)
        @test @inferred(deepview(A, 1, 2, 1:2, 1)) == view(A[1, 2], 1:2, 1)
        @test_throws ArgumentError deepview(A, 1, 1:2, 2, 1)
        @test @inferred(deepview(A, 1, 1:2, 1:2, 2)) == [A[1, 1][1:2, 2], A[1, 2][1:2, 2]]
        @test_throws MethodError deepview(A, 1, 2, 2, 1, 2)

        B = rand(3, 4, 5)
        @test @inferred(deepview(B, 2, 3, 4)) === view(B, 2, 3, 4)
        @test_throws MethodError deepview(B, 6)
        @test_throws MethodError deepview(B, 2, 3)
    end
end
