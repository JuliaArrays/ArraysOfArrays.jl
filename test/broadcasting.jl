# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

# A DenseArray subtype that is not an Array, like device array types outside
# of GPUArraysCore:
struct _MockDenseArray{T,N} <: DenseArray{T,N} end

@testset "broadcasting" begin
    ref_VoA1(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 5)][1:0] : [rand(T, rand(1:9)) for i in 1:n]

    ref_AosA1(T::Type, n::Integer) = [rand(T, 7) for i in 1:n]

    @testset "getindex broadcast specializations" begin
        let A = VectorOfArrays(ref_VoA1(Float32, 100))
            refA = Array(A)

            for Idxs in [
                ([rand(eachindex(a), rand(1:length(a))) for a in A],),
                (PartsView([rand(eachindex(a), rand(1:length(a))) for a in A]),),
                (tuple(1:1),), (tuple([1, 1, 1]),), (tuple(:),),
                (Ref(1:1),), (Ref([1, 1, 1]),), (Ref(:),),
            ]
                @test @inferred(broadcast(getindex, A, Idxs...)) isa VectorOfArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end
        end

        let A = convert(ArrayOfSimilarArrays, ref_AosA1(Float32, 100))
            refA = Array(A)

            for Idxs in [
                ([rand(eachindex(a), rand(1:length(a))) for a in A],),
                (PartsView([rand(eachindex(a), rand(1:length(a))) for a in A]),),
                (fill(:, length(A)),),
            ]
                @test @inferred(broadcast(getindex, A, Idxs...)) isa VectorOfArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end

            for Idxs in [
                (convert(VectorOfSimilarVectors, [rand(eachindex(a), 5) for a in A]),),
                (tuple(3:5),), (tuple([2, 5, 6]),), (tuple(:),),
                (Ref(3:5),), (Ref([2, 5, 6]),), (Ref(:),),
            ]
                refA = Array(A)

                @test @inferred(broadcast(getindex, A, Idxs...)) isa ArrayOfSimilarArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end

            # Logical masks per element:
            masks = convert(VectorOfSimilarVectors, [isodd.(eachindex(a)) for a in A])
            @test getindex.(A, masks) == getindex.(refA, Array(masks))

            # Out-of-bounds indices must be caught:
            @test_throws BoundsError getindex.(A, [fill(length(first(A)) + 1, 2) for a in A])
            @test_throws BoundsError getindex.(A, convert(VectorOfSimilarVectors, [fill(length(first(A)) + 1, 2) for a in A]))
        end
    end

    @testset "findall" begin
        for A in [
            VectorOfArrays(ref_VoA1(Bool, 100)),
            convert(ArrayOfSimilarArrays, ref_AosA1(Bool, 100))
        ]
            refA = Array(A)
        
            @test @inferred(broadcast(findall, A)) isa VectorOfArrays{Int}
            @test findall.(A) == findall.(refA)
        end
    end

    @testset "outer broadcast result types" begin
        A = sliced(rand(6, 10))
        V = VectorOfArrays([rand(3), rand(2), rand(4)])

        for X in (A, V)
            X_ref = collect(X)

            # Results stored in Arrays, including views of the elements with
            # any kind of indices, are packed into a nested array:
            for f in (
                x -> x, x -> view(x, 1:2), x -> view(x, [1, 2]), x -> view(x, x .> 0.5),
                x -> reshape(x, 1, :), x -> reinterpret(Float32, x), x -> 2 .* x, x -> x[1:2],
            )
                r = @inferred broadcast(f, X)
                @test r isa VectorOfArrays
                # isequal, not ==: reinterpreted random bytes may contain NaNs
                @test isequal(r, broadcast(f, X_ref))
            end

            # Other array results (lazy wrappers, bit, range and
            # zero-dimensional arrays) and scalars keep the default behavior:
            for f in (
                x -> x[1:2]', x -> x .> 0.5, x -> 1:length(x),
                x -> fill(sum(x)), sum,
            )
                r = @inferred broadcast(f, X)
                @test r isa Vector
                @test r == broadcast(f, X_ref)
            end
        end

        # Deterministic regression for the NaN case above:
        xnan = [reinterpret(Float64, 0x7ff4000000000000), 1.0]
        rnan = (x -> reinterpret(Float32, x)).(VectorOfArrays([xnan]))
        @test rnan isa VectorOfArrays{Float32}
        @test any(isnan, rnan[1])
        @test isequal(rnan[1], reinterpret(Float32, xnan))

        # Packing copies the elements, the result does not alias A, for
        # identity like for any other function (unlike map(identity, A)):
        A1 = sliced(ones(2, 3))
        for f in (x -> x, identity)
            r = @inferred broadcast(f, A1)
            @test r isa VectorOfArrays
            r[1][1] = 0
            @test A1[1] == [1, 1]
        end
        @test map(identity, A1) === A1

        # Unknown array types are not packed, in particular DenseArray
        # subtypes that are not Arrays (device arrays outside of
        # GPUArraysCore). Behind inference barriers, so that the
        # constant-foldable trait methods actually run:
        @test !ArraysOfArrays._packable_result(Base.inferencebarrier(_MockDenseArray{Float64,1}))
        @test ArraysOfArrays._packable_result(Base.inferencebarrier(Vector{Float64}))
        @test !ArraysOfArrays._host_storage(Base.inferencebarrier(Union{}))
        @static if isdefined(Base, :Memory)
            @test ArraysOfArrays._host_storage(Base.inferencebarrier(Memory{Float64}))
        end

        # Matrix elements:
        B = VectorOfArrays([rand(2, 3), rand(3, 2)])
        rB = @inferred broadcast(x -> x, B)
        @test rB isa VectorOfArrays{Float64,2}
        @test rB == collect(B)
        rBv = @inferred broadcast(vec, B)
        @test rBv isa VectorOfArrays{Float64,1}
        @test rBv == vec.(collect(B))

        # Results that a VectorOfArrays cannot represent are rejected instead
        # of being silently reshaped, and are left to map:
        f_offset = x -> view(x, Base.IdentityUnitRange(2:3))
        @test_throws ArgumentError f_offset.(A)
        @test map(f_offset, A) == [f_offset(x) for x in collect(A)]
        f_empty = x -> view(x, 1:0, :)
        @test_throws ArgumentError f_empty.(B)
        @test size.(map(f_empty, B)) == [(0, 3), (0, 2)]

        # Results of equal size can be turned into an ArrayOfSimilarArrays
        # without copying:
        rA = (x -> 2 .* x).(A)
        sA = convert(VectorOfSimilarVectors, rA)
        @test sA isa VectorOfSimilarVectors{Float64}
        @test sA == rA
        @test Base.mightalias(fused(sA), fused(rA))
        @test_throws DimensionMismatch convert(VectorOfSimilarVectors, (x -> x).(V))
    end
end
