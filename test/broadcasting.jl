# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

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

            # Strided host-array results, including views of the elements,
            # are packed into a nested array:
            for f in (
                x -> x, x -> view(x, 1:2), x -> reshape(x, 1, :),
                x -> reinterpret(Float32, x), x -> 2 .* x, x -> x[1:2],
            )
                r = @inferred broadcast(f, X)
                @test r isa VectorOfArrays
                # isequal, not ==: reinterpreted random bytes may contain NaNs
                @test isequal(r, broadcast(f, X_ref))
            end

            # Deterministic regression for the NaN case above:
            xnan = [reinterpret(Float64, 0x7ff4000000000000), 1.0]
            rnan = (x -> reinterpret(Float32, x)).(VectorOfArrays([xnan]))
            @test rnan isa VectorOfArrays{Float32}
            @test any(isnan, rnan[1])
            @test isequal(rnan[1], reinterpret(Float32, xnan))

            # Other array results (non-strided views, bit, lazy and
            # zero-dimensional arrays) and scalars keep the default behavior:
            for f in (
                x -> view(x, [1, 2]), x -> x .> 0.5, x -> 1:length(x),
                x -> fill(sum(x)), sum,
            )
                r = @inferred broadcast(f, X)
                @test r isa Vector
                @test r == broadcast(f, X_ref)
            end
        end

        # Packing copies the elements, the result does not alias A ...
        A1 = sliced(ones(2, 3))
        r = (x -> x).(A1)
        r[1][1] = 0
        @test A1[1] == [1, 1]
        # ... while identity itself returns A:
        @test identity.(A1) === A1

        # Matrix elements:
        B = VectorOfArrays([rand(2, 3), rand(3, 2)])
        rB = @inferred broadcast(x -> x, B)
        @test rB isa VectorOfArrays{Float64,2}
        @test rB == collect(B)
        rBv = @inferred broadcast(vec, B)
        @test rBv isa VectorOfArrays{Float64,1}
        @test rBv == vec.(collect(B))

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
