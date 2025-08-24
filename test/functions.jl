# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: getinnerdims, getouterdims


@testset "functions" begin
    A_0 = fill(Float32(42))
    A_1 = Float32[3, 8, 4, 6]
    A_1e = Float32[]
    A_2 = [Float32[3 8; 4 6], Float32[1 2; 5 7; 9 0]]
    A_2e = Matrix{Float32}[]
    A_2b = [Float32[3 8; 4 6], Float32[1 2; 9 0]]
    A_3 = [[Float32[3, 8, 4], Float32[1, 2, 5]], [Float32[6, 7], Float32[9, 0]]]

    @testset "Nested array API" begin
        for A in [A_0, A_1, A_1e, A_2, A_2e, A_2b, A_3]
            non_nested = eltype(A) <: Number

            @test @inferred(getsplitmode(A)) isa if non_nested
                NonSplitMode{ndims(A)}
            else
                UnknownSplitMode{typeof(A)}
            end

            smode = getsplitmode(A)

            dimdixs = ntuple(identity, ndims(A))
            if smode  isa UnknownSplitMode
                @test @inferred(is_memordered_splitmode(smode)) == false
                @test_throws ArgumentError getinnerdims(dimdixs, smode)
                @test_throws ArgumentError getouterdims(dimdixs, smode)
            else
                @test @inferred(is_memordered_splitmode(smode)) == true
                @test getinnerdims(dimdixs, smode) == ()
                @test getouterdims(dimdixs, smode) == dimdixs
            end

            if non_nested
                @test_throws ArgumentError joinedview(A)
                @test_throws ArgumentError flatview(A)
            end

            stacked_A = try stack(A); catch; nothing; end
            if isnothing(stacked_A)
                @test_throws DimensionMismatch stacked(A)
            else
                @inferred(stacked(A)) == stacked_A
                @test_throws ArgumentError splitview(stacked_A, smode)
            end
        end
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
end
