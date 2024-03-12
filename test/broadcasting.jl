# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

@testset "broadcasting" begin
    ref_flatview(A::AbstractVector{<:AbstractArray}) = vcat(map(vec, Array(A))...)

    ref_VoA1(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 5)][1:0] : [rand(T, rand(1:9)) for i in 1:n]
    ref_VoA2(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 4, 2)][1:0] : [rand(T, rand(1:4), rand(1:4)) for i in 1:n]
    ref_VoA3(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 3, 2, 4)][1:0] : [rand(T, rand(1:3), rand(1:3), rand(1:3)) for i in 1:n]

    ref_AosA1(T::Type, n::Integer) = [rand(T, 7) for i in 1:n]

    @testset "getindex broadcast specializations" begin
        let A = VectorOfArrays(ref_VoA1(Float32, 100))
            refA = Array(A)

            for Idxs in [
                ([rand(eachindex(a), rand(1:length(a))) for a in A],),
                (VectorOfVectors([rand(eachindex(a), rand(1:length(a))) for a in A]),),
                (tuple(1:1),), (tuple([1, 1, 1]),), (tuple(:),),
                (Ref(1:1),), (Ref([1, 1, 1]),), (Ref(:),),
            ]
                @test @inferred(broadcast(getindex, A, Idxs...)) isa VectorOfArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end
        end

        let A = ArrayOfSimilarArrays(ref_AosA1(Float32, 100))
            refA = Array(A)

            for Idxs in [
                ([rand(eachindex(a), rand(1:length(a))) for a in A],),
                (VectorOfVectors([rand(eachindex(a), rand(1:length(a))) for a in A]),),
            ]
                @test @inferred(broadcast(getindex, A, Idxs...)) isa VectorOfArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end

            for Idxs in [
                (VectorOfSimilarVectors([rand(eachindex(a), 5) for a in A]),),
                (tuple(3:5),), (tuple([2, 5, 6]),), (tuple(:),),
                (Ref(3:5),), (Ref([2, 5, 6]),), (Ref(:),),
            ]
                refA = Array(A)
            
                @test @inferred(broadcast(getindex, A, Idxs...)) isa ArrayOfSimilarArrays{eltype(eltype(A))}
                @test getindex.(A, Idxs...) == getindex.(refA, Idxs...)
            end
        end
    end

    @testset "findall" begin
        for A in [
            VectorOfArrays(ref_VoA1(Bool, 100)),
            ArrayOfSimilarArrays(ref_AosA1(Bool, 100))
        ]
            refA = Array(A)
        
            @test @inferred(broadcast(findall, A)) isa VectorOfArrays{Int}
            @test findall.(A) == findall.(refA)
        end
    end
end
