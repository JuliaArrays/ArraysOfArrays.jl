# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: getinnerdims, getouterdims

include("testdefs.jl")

@testset "functions" begin
    A_0 = fill(Float32(42))
    A_0_flat = fill(Float32(42))

    A_1 = Float32[3, 8, 4, 6]
    A_1_flat = Float32[3, 8, 4, 6]

    A_1e = Float32[]
    A_1e_flat = Float32[]

    A_2 = [Float32[3 8; 4 6], Float32[1 2; 5 7; 9 0]]
    A_2_flat = Float32[3, 8, 4, 6, 1, 2, 5, 7, 9, 0]

    A_2e = Matrix{Float32}[]
    A_2e_flat = Matrix{Float32}(undef, 0, 0)

    A_2b = [Float32[3 8; 4 6], Float32[1 2; 9 0]]
    A_2b_flat = stack(A_2b)

    A_3 = [[Float32[3, 8, 4], Float32[1, 2, 5]], [Float32[6, 7], Float32[9, 0]]]
    A_3_flat = reduce(vcat, A_3)

    @testset "getsplitmode" begin
        @test @inferred(getsplitmode(A_0)) isa NonSplitMode{0}
        @test @inferred(getsplitmode(A_1)) isa NonSplitMode{1}
        @test @inferred(getsplitmode(A_1e)) isa NonSplitMode{1}
        @test @inferred(getsplitmode(A_2)) isa UnknownSplitMode{typeof(A_2)}
        @test @inferred(getsplitmode(A_2e)) isa UnknownSplitMode{typeof(A_2e)}
        @test @inferred(getsplitmode(A_2b)) isa UnknownSplitMode{typeof(A_2b)}
        @test @inferred(getsplitmode(A_3)) isa UnknownSplitMode{typeof(A_3)}
    end

    @testset "innersize" begin
        @test @inferred(innersize(A_0)) == ()
        @test @inferred(innersize(A_1)) == ()
        @test @inferred(innersize(A_1e)) == ()
        @test_throws DimensionMismatch innersize(A_2)
        @test @inferred(innersize(A_2e)) == (0, 0)
        @test @inferred(innersize(A_2b)) == (2, 2)
        @test @inferred(innersize(A_3)) == (2,)
    end

    @testset "innermap" begin
        f = x -> x^2
        @test @inferred(innermap(f, A_0)) == fill(1764)
        @test @inferred(innermap(f, A_1)) == Float32[9, 64, 16, 36]
        @test @inferred(innermap(f, A_2)) == [Float32[9 64; 16 36], Float32[1 4; 25 49; 81 0]]
        @test_throws MethodError innermap(f, A_3)

        @test @inferred(innermap(length, A_0)) == fill(1)
        @test @inferred(innermap(length, A_1)) == fill(1, 4)
        @test @inferred(innermap(length, A_2)) == [fill(1, 2, 2), fill(1, 3, 2)]
        @test @inferred(innermap(length, A_3)) == [fill(3, 2), fill(2, 2)]
    end

    @testset "deepmap" begin
        f = x -> x^2
        @test @inferred(deepmap(f, A_0)) == fill(1764)
        @test @inferred(deepmap(f, A_1)) == Float32[9, 64, 16, 36]
        @test @inferred(deepmap(f, A_2)) == [Float32[9 64; 16 36], Float32[1 4; 25 49; 81 0]]
        @test @inferred(deepmap(f, A_3)) == [[Float32[9, 64, 16], Float32[1, 4, 25]], [Float32[36, 49], Float32[81, 0]]]

        @test @inferred(deepmap(length, A_0)) == fill(1)
        @test @inferred(deepmap(length, A_1)) == fill(1, 4)
        @test @inferred(deepmap(length, A_2)) == [fill(1, 2, 2), fill(1, 3, 2)]
        @test @inferred(deepmap(length, A_3)) == [[fill(1, 3), fill(1, 3)], [fill(1, 2), fill(1, 2)]]
    end

    @testset "Nested array API" begin
        test_api(map_f, A_0, A_0_flat)
        test_api(map_f, A_1, A_1_flat)
        test_api(map_f, A_1e, A_1e_flat)
        test_api(map_f, A_2, A_2_flat)
        test_api(map_f, A_2e, A_2e_flat)
        test_api(map_f, A_2b, A_2b_flat)
        test_api(map_f, A_3, A_3_flat)
    end
end
