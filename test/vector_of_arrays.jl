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
    end
end
