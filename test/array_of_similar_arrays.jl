# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ElasticArrays
using UnsafeArrays


@testset "array_of_similar_arrays" begin
    function rand_flat_array(Val_N::Val{N}) where {N}
        sz_max = (2,3,2,4,5)
        sz = ntuple(i -> sz_max[i], Val_N)
        x = rand(sz...)
    end


    function rand_nested_similar_arrays(Val_M::Val{M}, Val_N::Val{N}) where {M,N}
        sz_max = (2,3,2,4,5)
        sz_inner = ntuple(i -> sz_max[i], Val_M)
        sz_outer = ntuple(i -> sz_max[i + M], Val_N)

        A = Array{Array{Float64, M}, N}(undef, sz_outer...)
        for i in eachindex(A)
            A[i] = rand(sz_inner...)
        end
        A
    end


    @inline function test_from_flat(::Type{TT}, ::Type{RT}, Val_L::Val{L}) where {
        TT<:ArrayOfSimilarArrays,
        RT<:ArrayOfSimilarArrays,
        L
    }
        @testset "$TT from Array{Float64,$L}" begin
            A = rand_flat_array(Val_L)
            @test typeof(@inferred TT(A)) == RT
            @test typeof(@inferred convert(TT, A)) == RT
        end
    end


    @inline function test_from_nested(::Type{TT}, ::Type{RT}, Val_M::Val{M}, Val_N::Val{N}) where {
        TT<:ArrayOfSimilarArrays,
        RT<:ArrayOfSimilarArrays,
        M, N
    }
        @testset "$TT from Array{Array{Float64,$M},$N}" begin
            A = rand_nested_similar_arrays(Val_M, Val_N)

            A2_ctor = @inferred TT(A)
            U = eltype(parent(A2_ctor))

            A_U = Array{Array{U,M},N}(A)

            @test typeof(A2_ctor) == RT
            @test A2_ctor == A_U

            A2_conv = @inferred convert(TT, A)
            @test typeof(A2_conv) == RT
            @test A2_conv == A2_ctor

            U = eltype(parent(A2_ctor))
            A3 = @inferred Array(A2_ctor)
            @test typeof(A3) == Array{Array{U,M},N}
            @test A3 == A_U
        end
    end


    @testset "construct/convert from flat array" begin
        test_from_flat(ArrayOfSimilarArrays{Float64,2,3}, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float64}, ArrayOfSimilarVectors{Float64,2,3,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float64}, VectorOfSimilarArrays{Float64,2,3,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float64}, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))

        test_from_flat(ArrayOfSimilarArrays{Float32,2,3}, ArrayOfSimilarArrays{Float32,2,3,5,Array{Float32,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float32}, ArrayOfSimilarVectors{Float32,2,3,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float32}, VectorOfSimilarArrays{Float32,2,3,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))
    end

    @testset "construct/convert from nested arrays" begin
        test_from_nested(ArrayOfSimilarArrays, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float64,2,3}, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float64}, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float32}, ArrayOfSimilarArrays{Float32,2,3,5,Array{Float32,5}}, Val(2), Val(3))

        test_from_nested(ArrayOfSimilarArrays, VectorOfSimilarArrays{Float64,4,5,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(ArrayOfSimilarArrays, ArrayOfSimilarVectors{Float64,4,5,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarArrays, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))

        test_from_nested(VectorOfSimilarArrays{Float64,4}, VectorOfSimilarArrays{Float64,4,5,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays{Float64}, VectorOfSimilarArrays{Float64,4,5,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays{Float32}, VectorOfSimilarArrays{Float32,4,5,Array{Float32,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays, VectorOfSimilarArrays{Float64,4,5,Array{Float64,5}}, Val(4), Val(1))

        test_from_nested(ArrayOfSimilarVectors{Float64,4}, ArrayOfSimilarVectors{Float64,4,5,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors{Float64}, ArrayOfSimilarVectors{Float64,4,5,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors{Float32}, ArrayOfSimilarVectors{Float32,4,5,Array{Float32,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors, ArrayOfSimilarVectors{Float64,4,5,Array{Float64,5}}, Val(1), Val(4))

        test_from_nested(VectorOfSimilarVectors{Float64}, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))
        test_from_nested(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(1), Val(1))
        test_from_nested(VectorOfSimilarVectors, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))
    end
end
