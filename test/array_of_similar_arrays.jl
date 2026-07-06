# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ElasticArrays
using Adapt
using ChainRulesTestUtils

using Statistics
using StatsBase: cov2cor

include("testdefs.jl")

# Minimal third-party subtype of AbstractArrayOfSimilarArrays, only
# implements the standard array API and `fused`:
if !isdefined(Main, :TestSimilarVectors)
    struct TestSimilarVectors{T,ET} <: AbstractArrayOfSimilarArrays{T,1,1,ET}
        data::Matrix{T}

        TestSimilarVectors(data::Matrix{T}) where {T} =
            new{T,typeof(view(data, :, firstindex(data, 2)))}(data)
    end
    Base.size(A::TestSimilarVectors) = (size(A.data, 2),)
    Base.getindex(A::TestSimilarVectors, i::Int) = view(A.data, :, i)
    ArraysOfArrays.fused(A::TestSimilarVectors) = A.data
end

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
            @test typeof(@inferred TT(A)) <: RT
            
            AosA = TT(A)
            @test typeof(AosA) == typeof(TT(A))
            @test @inferred(eltype(AosA)) == typeof(AosA[1])
            @test @inferred(parent(AosA)) === @inferred(flatview(AosA))
            if eltype(eltype(AosA)) == eltype(A)
                @test @inferred(flatview(AosA)) === A
            else
                @test @inferred(flatview(AosA)) ≈ A
            end
            @test @inferred(stacked(AosA)) === flatview(AosA)
            @test @inferred(stack(AosA)) == flatview(AosA)
            @test stack(AosA) !== flatview(AosA)

            @test_deprecated convert(TT, A) == TT(A)
        end
    end


    @inline function test_from_nested(::Type{TT}, ::Type{RT}, Val_M::Val{M}, Val_N::Val{N}) where {
        TT<:ArrayOfSimilarArrays,
        RT<:ArrayOfSimilarArrays,
        M, N
    }
        @testset "$TT from Array{Array{Float64,$M},$N}" begin
            A = rand_nested_similar_arrays(Val_M, Val_N)

            A2_conv = @inferred convert(TT, A)
            U = eltype(flatview(A2_conv))

            A_U = Array{Array{U,M},N}(A)

            @test typeof(A2_conv) <: RT
            @test A2_conv == A_U

            A2_conv = @inferred convert(TT, A)
            @test typeof(A2_conv) <: RT

            U = eltype(flatview(A2_conv))
            A3 = @inferred Array(A2_conv)
            @test typeof(A3) == Array{Array{U,M},N}
            @test A3 == A_U

            @test_deprecated TT(A) == A2_conv
        end
    end


    @testset "construct/convert from flat array" begin
        test_from_flat(ArrayOfSimilarArrays{Float64,2,3}, ArrayOfSimilarArrays{Float64,2,3,Array{Float64,5}}, Val(5))
        test_from_flat(ArrayOfSimilarArrays{Float64,2}, ArrayOfSimilarArrays{Float64,2,3,Array{Float64,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float64}, ArrayOfSimilarVectors{Float64,2,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float64}, VectorOfSimilarArrays{Float64,2,Array{Float64,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float64}, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(2))

        test_from_flat(ArrayOfSimilarArrays{Float32,2,3}, ArrayOfSimilarArrays{Float32,2,3,Array{Float32,5}}, Val(5))
        test_from_flat(ArrayOfSimilarArrays{Float32,2}, ArrayOfSimilarArrays{Float32,2,3,Array{Float32,5}}, Val(5))
        test_from_flat(ArrayOfSimilarVectors{Float32}, ArrayOfSimilarVectors{Float32,2,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarArrays{Float32}, VectorOfSimilarArrays{Float32,2,Array{Float32,3}}, Val(3))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))
        test_from_flat(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(2))

        test_rrule(ArrayOfSimilarArrays{Float64,2,2}, rand(2,3,4,5))
    end


    @testset "construct/convert from nested arrays" begin
        test_from_nested(ArrayOfSimilarArrays, ArrayOfSimilarArrays{Float64,2,3,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float64,2,3}, ArrayOfSimilarArrays{Float64,2,3,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float64}, ArrayOfSimilarArrays{Float64,2,3,Array{Float64,5}}, Val(2), Val(3))
        test_from_nested(ArrayOfSimilarArrays{Float32}, ArrayOfSimilarArrays{Float32,2,3,Array{Float32,5}}, Val(2), Val(3))

        test_from_nested(ArrayOfSimilarArrays, VectorOfSimilarArrays{Float64,4,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(ArrayOfSimilarArrays, ArrayOfSimilarVectors{Float64,4,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarArrays, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))

        test_from_nested(VectorOfSimilarArrays{Float64,4}, VectorOfSimilarArrays{Float64,4,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays{Float64}, VectorOfSimilarArrays{Float64,4,Array{Float64,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays{Float32}, VectorOfSimilarArrays{Float32,4,Array{Float32,5}}, Val(4), Val(1))
        test_from_nested(VectorOfSimilarArrays, VectorOfSimilarArrays{Float64,4,Array{Float64,5}}, Val(4), Val(1))

        test_from_nested(ArrayOfSimilarVectors{Float64,4}, ArrayOfSimilarVectors{Float64,4,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors{Float64}, ArrayOfSimilarVectors{Float64,4,Array{Float64,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors{Float32}, ArrayOfSimilarVectors{Float32,4,Array{Float32,5}}, Val(1), Val(4))
        test_from_nested(ArrayOfSimilarVectors, ArrayOfSimilarVectors{Float64,4,Array{Float64,5}}, Val(1), Val(4))

        test_from_nested(VectorOfSimilarVectors{Float64}, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))
        test_from_nested(VectorOfSimilarVectors{Float32}, VectorOfSimilarVectors{Float32,Array{Float32,2}}, Val(1), Val(1))
        test_from_nested(VectorOfSimilarVectors, VectorOfSimilarVectors{Float64,Array{Float64,2}}, Val(1), Val(1))

        r = @inferred(rand(5,5))
        @test @inferred(flatview(ArrayOfSimilarVectors(r))) == r

        test_rrule(ArrayOfSimilarArrays{Float64,2,2}, [rand(2,3) for i in 1:5, j in 1:6])
    end

    @testset "AbstractSlices interface" begin
        @test @inferred(ArrayOfSimilarArrays{Float32,2,3}(rand(3,4,5,6,7))) isa AbstractSlices
        let A = ArrayOfSimilarArrays{Float32,2,3}(rand(3,4,5,6,7))
            @test @inferred(A[2, 3, 4]) isa AbstractArray{Float32,2}
            ref_ET = typeof(A[2, 3, 4])
            @test A isa AbstractSlices{ref_ET, 3}
            @test @inferred(eltype(A)) == ref_ET
            @test @inferred(innersize(A)) == (3, 4)
            @test @inferred(getslicemap(A)) == (:, :, 1, 2, 3)
            @test @inferred(parent(A)) === A.data
        end
    end

    @testset "split mode API" begin
        A_flat = rand_flat_array(Val(5))
        A = ArrayOfSimilarArrays{Float64,2,3}(A_flat)
        test_api(A, Array(A), A_flat)

        V_flat = rand_flat_array(Val(2))
        V = VectorOfSimilarVectors(V_flat)
        test_api(V, Array(V), V_flat)

        # mapat operates on the flat data and preserves structure:
        @test @inferred(mapat(abs2, Val(2), A)) == innermap(abs2, A)
        @test typeof(mapat(abs2, Val(2), A)) == typeof(A)
        @test fused(@inferred(mapat(+, Val(2), A, A))) == 2 .* A_flat
        @test_throws DimensionMismatch mapat(+, Val(2), A, ArrayOfSimilarArrays{Float64,3,2}(A_flat))

        @test @inferred(innersizes(A)) == fill(innersize(A), size(A))
        @test @inferred(innerlengths(A)) == fill(prod(innersize(A)), size(A))

        # bcastat with outer-aligned, scalar and flat-matching arguments:
        w = rand(size(A)...)
        r_bc = @inferred bcastat(+, Val(2), A, w)
        @test r_bc isa ArrayOfSimilarArrays
        @test collect(r_bc) == [A[i] .+ w[i] for i in eachindex(A, w)]
        @test fused(bcastat(muladd, Val(2), A, 2.0, A_flat)) == muladd.(A_flat, 2.0, A_flat)

        # vecflattened rrule:
        A_rr = ArrayOfSimilarArrays{Float64,1,1}(rand(3, 4))
        y, pb = rrule(vecflattened, A_rr)
        @test y == vec(A_rr.data)
        t = pb(collect(1.0:12.0))
        @test t[1] == NoTangent()
        @test t[2] isa ArrayOfSimilarArrays
        @test fused(t[2]) == reshape(1.0:12.0, 3, 4)
    end

    @testset "custom subtypes" begin
        data = rand(3, 5)
        B = TestSimilarVectors(data)

        @test @inferred(getsplitmode(B)) === SplitSlices{1,1}()
        @test @inferred(unstackmode(B)) === SplitSlices{1,1}()
        @test @inferred(fused(B)) === data
        @test @inferred(stacked(B)) === data
        @test @inferred(flatview(B)) === data
        @test @inferred(parent(B)) === data
        @test @inferred(vecflattened(B)) == vec(data)
        @test @inferred(innersize(B)) == (3,)
        @test @inferred(getslicemap(B)) == (:, 1)
        @test @inferred(stack(B)) == data
        @test stack(B) !== data
        @test B == VectorOfSimilarVectors(data)
        @test isapprox(B, VectorOfSimilarVectors(data .+ 1e-14))
        @test splitup(fused(B), getsplitmode(B)) == B
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

        # similar must respect the requested element type and outer dims:
        A_similar32 = @inferred similar(A, Array{Float32, 2}, size(A))
        @test eltype(eltype(A_similar32)) == Float32
        @test eltype(A_similar32.data) == Float32
        A_similar_1d = @inferred similar(A, Array{Float64, 2}, (3,))
        @test size(A_similar_1d) == (3,)
        @test innersize(A_similar_1d) == innersize(A)
        @test typeof(A_similar) == typeof(A)
    end


    @testset "adapt" begin
        A_flat = rand(2,3,4,5,6)
        A_nested = sliced(A_flat, 2)
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
        B = convert(ArrayOfSimilarArrays, A)
        @test @inferred(flatview(B)) === B.data
        @test flatview(B) == stacked(A)
        @test_throws ArgumentError flatview(A)
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

        # Non-Colon dims must forward to the generic implementations:
        @test sum(VV_aosa; dims = 1) == sum(collect(VV_aosa); dims = 1)
        @test mean(VV_aosa; dims = 1) == mean(collect(VV_aosa); dims = 1)
        @test @inferred(sum(VV_aosa)) ≈ sum(collect(VV_aosa))
        @test @inferred(mean(VV_aosa)) ≈ mean(collect(VV_aosa))

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
        A_nested = sliced(A_flat, 2)

        @test A_nested isa AbstractArray{<:AbstractArray{T,2},3} where T
        @test flatview(A_nested) === A_flat

        A_flat = rand(4,4)
        A_nested = @inferred(sliced(A_flat))

        @test A_nested.data == A_flat
        @test @inferred(size(A_nested))[1] == @inferred(size(A_flat))[1]
        @test @inferred(innersize(A_nested,1)) == @inferred(size(A_flat))[1]

        # -------------------------------------------------------------------

        A_flat = rand(2,3,4,5)
        ASA = @inferred(ArrayOfSimilarArrays{Float64,2,2}(A_flat))
        @test ASA.data == A_flat

        # -------------------------------------------------------------------
        A_nested = sliced(ElasticArray{Float64}(undef, 2, 3, 0), 2)
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

        @test @inferred(prod(ArraysOfArrays.innersize(VSV))) == N
    end

    @testset "map and broadcast" begin
        A_flat = rand(2,3,4,5,6)
        A = sliced(A_flat, 2)

        for do_map in (map, broadcast)
            @test @inferred(do_map(identity, A)) === A
        end
    end
end
