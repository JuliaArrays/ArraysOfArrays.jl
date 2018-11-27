# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using UnsafeArrays


@testset "vector_of_arrays" begin
    @testset "examples" begin
        VA = VectorOfArrays{Float64, 2}()

        push!(VA, rand(2, 3))
        push!(VA, rand(4, 2))

        @test size(VA[1]) == (2,3)
        @test size(VA[2]) == (4,2)

        # -------------------------------------------------------------------


        VA_flat = flatview(VA)
        @test VA_flat isa Vector{Float64}

        # -------------------------------------------------------------------


        VA_flat = flatview(VA)
        @test view(VA_flat, 7:14) == vec(VA[2])

        fill!(view(VA_flat, 7:14), 2.4)
        @test all(x -> x == 2.4, VA[2])

        fill!(view(VA_flat, 7:14), 4.2)
        @test all(x -> x == 4.2, VA[2])

        # -------------------------------------------------------------------


        @test length(@inferred resize!(VA, 1)) == 1
        @test_throws ArgumentError resize!(VA, 4)

        # -------------------------------------------------------------------

        using UnsafeArrays

        A = nestedview(rand(2,3,4,5), 2)

        @test isbits(A[2,2]) == false

        @test @uviews A begin
            isbits(A[2,2]) == true
        end

        # -------------------------------------------------------------------

        A = [1, 1, 2, 3, 3, 2, 2, 2]
        A_grouped_ref = [[1, 1], [2], [3, 3], [2, 2, 2]]
        elem_ptr = consgrouped_ptrs(A)
        @test first.(@inferred(VectorOfVectors(A, elem_ptr))) == [1, 2, 3, 2]

        B = [1, 2, 3, 4, 5, 6, 7, 8]
        B_grouped_ref = [[1, 2], [3], [4, 5], [6, 7, 8]]
        @test @inferred(VectorOfVectors(B, elem_ptr)) == B_grouped_ref

        C = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        C_grouped_ref = [[1.1, 2.2], [3.3], [4.4, 5.5], [6.6, 7.7, 8.8]]

        @test @inferred(consgroupedview(A, B)) isa VectorOfVectors
        @test consgroupedview(A, B) == B_grouped_ref

        @test @inferred(consgroupedview(A, (B, C))) isa NTuple{2, VectorOfVectors}
        @test consgroupedview(A, (B, C)) == (B_grouped_ref, C_grouped_ref)

        data = (a = A, b = B, c = C)
        @test @inferred(consgroupedview(A, data)) isa NamedTuple{(:a, :b, :c),<:NTuple{3,AbstractVector}}
        result = consgroupedview(A, data)
        @test result == (a = A_grouped_ref, b = B_grouped_ref, c = C_grouped_ref)
        @test flatview(result.a) === data.a
        @test flatview(result.b) === data.b
        @test flatview(result.c) === data.c
    end
end
