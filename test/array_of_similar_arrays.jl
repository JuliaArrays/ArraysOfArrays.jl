# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ElasticArrays
using Adapt
using ChainRulesTestUtils

using Statistics
using StatsBase: cov2cor

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
            U = eltype(flatview(A2_ctor))

            A_U = Array{Array{U,M},N}(A)

            @test typeof(A2_ctor) == RT
            @test A2_ctor == A_U

            A2_conv = @inferred convert(TT, A)
            @test typeof(A2_conv) == RT
            @test A2_conv == A2_ctor

            U = eltype(flatview(A2_ctor))
            A3 = @inferred Array(A2_ctor)
            @test typeof(A3) == Array{Array{U,M},N}
            @test A3 == A_U
        end
    end


    @testset "construct/convert from flat array" begin
        test_from_flat(ArrayOfSimilarArrays{Float64,2,3}, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(5))
        test_from_flat(ArrayOfSimilarArrays{Float64,2}, ArrayOfSimilarArrays{Float64,2,3,5,Array{Float64,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float64}, ArrayOfSimilarVectors{Float64,2,3,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float64}, VectorOfSimilarArrays{Float64,2,3,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float64}, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))

        test_from_flat(ArrayOfSimilarArrays{Float32,2,3}, ArrayOfSimilarArrays{Float32,2,3,5,Array{Float32,5}}, Val(5))
        test_from_flat(ArrayOfSimilarArrays{Float32,2}, ArrayOfSimilarArrays{Float32,2,3,5,Array{Float32,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float32}, ArrayOfSimilarVectors{Float32,2,3,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float32}, VectorOfSimilarArrays{Float32,2,3,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))

        test_rrule(ArrayOfSimilarArrays{Float64,2,2}, rand(2,3,4,5))
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

        r = @inferred(rand(5,5))
        @test @inferred(flatview(ArrayOfSimilarVectors(r))) == r

        test_rrule(ArrayOfSimilarArrays{Float64,2,2}, [rand(2,3) for i in 1:5, j in 1:6])
    end

    @testset "add remove" begin
        EA_ref1 = rand_flat_array(Val(3))
        EA_ref2 = rand_flat_array(Val(3))
        EA1 = ElasticArray{Float64, 3}(EA_ref1)
        EA2 = ElasticArray{Float64, 3}(EA_ref2)
        AEA1 = ArrayOfSimilarArrays{Float64, 3}(EA1)
        AEA2 = ArrayOfSimilarArrays{Float64, 3}(EA2)
        AEA1_ref = copy(AEA1)
        AEA2_ref = copy(AEA2)
        append!(AEA1, AEA2)
        prepend!(AEA2, AEA1_ref)
        N = size(AEA1.data)[end]
        for i in 1:N
            @test @inferred(reverse(AEA2.data, dims=3)[:,:,i]) == @inferred(AEA1.data[:,:,N+1-i])
        end

        A1 = ArrayOfSimilarArrays{Float64,1}(rand_flat_array(Val(1)))
        A2 = ArrayOfSimilarArrays{Float64,1}(rand_flat_array(Val(1)))
        A1_data = copy(A1.data)
        A2_data = copy(A2.data)
        append!(A1, A2)
        @test A1.data == vcat(A1_data, A2_data)
        prepend!(A1, A1)
        len = @inferred(length(A1.data))
        @test A1.data[1:Int(len/2)] == A1.data[Int(len/2 + 1):end]
    end

    @testset "similar and copyto!" begin
        A = ArrayOfSimilarArrays{Float64,1}(rand_flat_array(Val(1)))
        @test (@inferred copyto!((@inferred similar(A)), A)) == A

        A = ArrayOfSimilarArrays{Float64,2}(rand_flat_array(Val(5)))
        @test (@inferred copyto!((@inferred similar(A)), A)) == A

        A_data = rand_flat_array(Val(4))
        A = ArrayOfSimilarArrays{Float64, 2}(A_data)
        A_similar = similar(A, Array{Float64, 2}, size(A))
        @test @inferred(size(A)) == @inferred(size(A_similar))
        @test @inferred(size(A.data)) == @inferred(size(A_similar.data))
        @test typeof(A_similar.data) == typeof(A_data)
        @test typeof(A_similar) == typeof(A)
    end


    @testset "adapt" begin
        A_flat = rand(2,3,4,5,6)
        A_nested = nestedview(A_flat, 2)
        @test @inferred(adapt(identity, A_nested)) == A_nested
        @test typeof(adapt(identity, A_nested)) == typeof(A_nested)
    end


    @testset "deepcopy" begin
        A = ArrayOfSimilarArrays{Float64,1}(rand_flat_array(Val(1)))
        @test (@inferred deepcopy(A)) == A
        @test typeof(deepcopy(A)) == typeof(A)
    end


    @testset "flatview" begin
        A = rand_nested_similar_arrays(Val(3), Val(2))
        B = ArrayOfSimilarArrays(A)
        @inferred(flatview(B))[:] == collect(flatview(A))
    end


    @testset "deepgetindex" begin
        A = rand_nested_similar_arrays(Val(3), Val(2))
        B = ArrayOfSimilarArrays(A)

        @test deepgetindex(A, 3, 4, 2, 1, 2) == @inferred deepgetindex(B, 3, 4, 2, 1, 2)
        @test deepgetindex(A, 2:3, 4, 2, 1, 2) == @inferred deepgetindex(B, 2:3, 4, 2, 1, 2)
        @test deepgetindex(A, 2:3, 2:4, 2, 1, 2) == @inferred deepgetindex(B, 2:3, 2:4, 2, 1, 2)
        @test deepgetindex(A, 2, 4, :, 1, 2) == @inferred deepgetindex(B, 2, 4, :, 1, 2)
        @test deepgetindex(A, 2, 4, :, 1, 1:2) == @inferred deepgetindex(B, 2, 4, :, 1, 1:2)
        @test deepgetindex(A, 2:3, 4, :, 1, 2) == @inferred deepgetindex(B, 2:3, 4, :, 1, 2)
        @test deepgetindex(A, 2:3, 4, :, 1, 1:2) == @inferred deepgetindex(B, 2:3, 4, :, 1, 1:2)
    end


    @testset "deepsetindex!" begin
        function testdata()
            A = rand_nested_similar_arrays(Val(3), Val(2))
            B = ArrayOfSimilarArrays(A)
            A, B
        end

        A, B = testdata()
        @test deepsetindex!(A, 42, 3, 4, 2, 1, 2) == @inferred deepsetindex!(B, 42, 3, 4, 2, 1, 2)
        @test deepgetindex(B, 3, 4, 2, 1, 2) == 42

        A, B = testdata()
        X1 = rand(2)
        @test deepsetindex!(A, X1, 2:3, 4, 2, 1, 2) == @inferred deepsetindex!(B, X1, 2:3, 4, 2, 1, 2)
        @test deepgetindex(B, 2:3, 4, 2, 1, 2) == X1

        A, B = testdata()
        X2 = rand(2,2)
        @test deepsetindex!(A, X2, 2, 4, :, 1, 1:2) == @inferred deepsetindex!(B, X2, 2, 4, :, 1, 1:2)
        @test deepgetindex(B, 2, 4, :, 1, 1:2) == X2

        A, B = testdata()
        X3 = [rand(2,2), rand(2,2)]
        @test deepsetindex!(A, X3, 2:3, 4, :, 1, 1:2) == @inferred deepsetindex!(B, X3, 2:3, 4, :, 1, 1:2)
        @test deepgetindex(B, 2:3, 4, :, 1, 1:2) == X3
    end


    @testset "deepview" begin
        A = rand_nested_similar_arrays(Val(3), Val(2))
        B = ArrayOfSimilarArrays(A)

        @test deepview(A, 3, 4, 2, 1, 2) == @inferred deepview(B, 3, 4, 2, 1, 2)
        @test deepgetindex(A, 2:3, 4, 2, 1, 2) == @inferred deepview(B, 2:3, 4, 2, 1, 2)
        @test deepgetindex(A, 2:3, 2:4, 2, 1, 2) == @inferred deepview(B, 2:3, 2:4, 2, 1, 2)
        @test deepview(A, 2, 4, :, 1, 2) == @inferred deepview(B, 2, 4, :, 1, 2)
        @test deepview(A, 2, 4, :, 1, 1:2) == @inferred deepview(B, 2, 4, :, 1, 1:2)
        @test deepview(A, 2:3, 4, :, 1, 2) == @inferred deepview(B, 2:3, 4, :, 1, 2)
        @test deepview(A, 2:3, 4, :, 1, 1:2) == @inferred deepview(B, 2:3, 4, :, 1, 1:2)
    end


    @testset "empty" begin
        A = [rand(2,3), rand(2,3), rand(2,3)]
        B = ArrayOfSimilarArrays(A)
        @test typeof(@inferred empty(B)) == typeof(B)
        @test empty(A) == empty(B)

        C = VectorOfSimilarArrays{Float64,2}(ElasticArray(B.data))
        @test @inferred(empty(C)) == empty(A)
        @test @inferred(empty!(deepcopy(C))) == empty(A)
        @test @inferred(empty(C)) == @inferred(empty!(deepcopy(C)))
        @test append!(empty(C), C) == A
        @test append!(empty!(deepcopy(C)), C) == A
    end

    @testset "stats" begin
        VV = [rand(3) for i in 1:10]
        VV_aosa = ArrayOfSimilarArrays(VV)

        VA = [rand(2,3,3) for i in 1:10]
        VA_aosa = ArrayOfSimilarArrays(VA)

        array_cmp(A, B) = (A ≈ B) && (size(A) == size(B))

        function test_statistics_op(op::Function)
            @testset "$op" begin
                @test array_cmp(@inferred(op(VV_aosa)), op(VV))

                if op in (var, cov)
                    @test array_cmp(@inferred(op(VV_aosa, corrected = false)), op(VV, corrected = false))
                end

                if (op in (sum, mean, var))
                    @test array_cmp(@inferred(op(VA_aosa)), op(VA))
                end
            end
        end

        test_statistics_op(sum)
        test_statistics_op(mean)
        test_statistics_op(var)
        test_statistics_op(std)
        test_statistics_op(cov)

        @testset "cor" begin
            # Statistics.cor currently results in an error for Vector{Vector},
            # this should be considered a bug, though, since Statistics.cov
            # works fine.
            @test array_cmp(@inferred(cor(VV_aosa)), cov2cor(cov(VV), std(VV)))
        end
    end

    @testset "examples" begin
        A_flat = rand(2,3,4,5,6)
        A_nested = nestedview(A_flat, 2)

        @test A_nested isa AbstractArray{<:AbstractArray{T,2},3} where T
        @test flatview(A_nested) === A_flat

        A_flat = rand(4,4)
        A_nested = @inferred(nestedview(A_flat))

        @test A_nested.data == A_flat
        @test @inferred(size(A_nested))[1] == @inferred(size(A_flat))[1]
        @test @inferred(innersize(A_nested,1)) == @inferred(size(A_flat))[1]

        # -------------------------------------------------------------------

        A_flat = rand(2,3,4,5)
        ASA = @inferred(ArrayOfSimilarArrays{Float64,2,2}(A_flat))
        @test ASA.data == A_flat

        # -------------------------------------------------------------------
        A_nested = nestedview(ElasticArray{Float64}(undef, 2, 3, 0), 2)
        A_nested_copy = deepcopy(A_nested)

        for i in 1:4
            push!(A_nested, rand(2, 3))
        end
        @test size(flatview(A_nested)) == (2, 3, 4)

        resize!(A_nested, 6)
        @test size(flatview(A_nested)) == (2, 3, 6)

        for i in 1:4
            pushfirst!(A_nested, rand(2,3))
        end
        @test size(flatview(A_nested)) == (2, 3, 10)

        for i in 1:4
            pop!(A_nested)
        end
        @test size(flatview(A_nested)) == (2, 3, 6)

        for i in 1:size(A_nested)[1]
            pop!(A_nested)
        end
        @test_throws ArgumentError pop!(A_nested)

    end

    @testset "misc" begin
        N = 4
        r1 = rand(1,4); r2 = rand(1,4); r3 = rand(1,4); r4 = rand(1,4)
        r = vcat(r1,r2,r3,r4)
        VSV = VectorOfSimilarVectors(r)
        VSA = VectorOfSimilarArrays(r)
        ASA = ArrayOfSimilarArrays([r1,r2,r3,r4])

        f = x -> x.*2

        @test @inferred(IndexStyle(VSV)) == IndexLinear()
        @test @inferred(IndexStyle(VSA)) == IndexLinear()
        @test @inferred(IndexStyle(ASA)) == IndexLinear()
        @test VSA == VSV
        @test flatview(VSA) == flatview(VSV)

        @test @inferred(deepmap(f, ASA)).data == ASA.data.*2
        @test @inferred(innermap(f, ASA)).data == ASA.data.*2

        @test @inferred(ArraysOfArrays._innerlength(VSV)) == N
    end

    @testset "map and broadcast" begin
        A_flat = rand(2,3,4,5,6)
        A = nestedview(A_flat, 2)

        for do_map in (map, broadcast)
            @test @inferred(do_map(identity, A)) === A
        end
    end
end
