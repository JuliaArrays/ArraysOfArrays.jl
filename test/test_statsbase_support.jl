# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test
using StatsBase
using Statistics

@testset "StatsBase support" begin
        r = rand(1,100)
        V = VectorOfSimilarVectors{Float64}(r)
        w = FrequencyWeights(rand(100))
        @test isapprox(@inferred(sum(V, w))[1], sum(r, w))
        @test isapprox(@inferred(mean(V,w))[1], mean(r, w))
        @test isapprox(@inferred(var(V,w, corrected=true))[1], var(r, w, corrected=true))
        @test isapprox(@inferred(cov(V, w))[1], var(r, w, corrected=true))
        @test isapprox(@inferred(cor(V,w))[1], 1.0)
    end
end
