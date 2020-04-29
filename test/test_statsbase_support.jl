# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test
using StatsBase
using Statistics

@testset "StatsBase support" begin
    VV = [rand(3) for i in 1:10]
    VV_aosa = ArrayOfSimilarArrays(VV)

    VA = [rand(2,3,3) for i in 1:10]
    VA_aosa = ArrayOfSimilarArrays(VA)

    w = FrequencyWeights(rand(10))

    array_cmp(A, B) = (A ≈ B) && (size(A) == size(B))


    # sum and mean for Vector{Vector} with weights currently fail with
    # the implementations in StatsBase. This should be considered a
    # bug in StatsBase, since Base and Statistics support sum and mean
    # for Vector{Vector} without weights. Also, adding products of vectors
    # and weights is perfectly natural, mathematically.

    _sum(A::AbstractVector{<:AbstractArray}, w::AbstractWeights) =
        sum(A .* w)

    _mean(A::AbstractVector{<:AbstractArray}, w::AbstractWeights) =
        _sum(A, w) ./ sum(w)

    @testset "sum and mean" begin

        @test array_cmp(@inferred(sum(VV_aosa, w)), _sum(VV, w))
        @test array_cmp(@inferred(sum(VA_aosa, w)), _sum(VA, w))

        @test array_cmp(@inferred(mean(VV_aosa, w)), _mean(VV, w))
        @test array_cmp(@inferred(mean(VA_aosa, w)), _mean(VA, w))
    end


    # Weighted var and std are currently not supported for Vector{Vector} by
    # StatsBase. This should be considered a bug in StatsBase, since
    # unweighted var and std for Vector{Vector} are supported by Statistics.

    function _var(A::AbstractVector{<:AbstractArray}, w::FrequencyWeights; corrected = true)
        wmean_A = _mean(A, w)
        wsum = sum(w)
        wsum_corr = corrected ? -1 : 0
        sum([(x .- wmean_A).^2 for x in A] .* w) ./ (wsum + wsum_corr)
    end

    _std(A::AbstractVector{<:AbstractArray}, w::AbstractWeights; corrected = true) =
        sqrt.(_var(A, w, corrected = corrected))

    @testset "var and std" begin
        @test array_cmp(@inferred(var(VV_aosa, w)), _var(VV_aosa, w))
        @test array_cmp(@inferred(var(VV_aosa, w, corrected = false)), _var(VV_aosa, w, corrected = false))
        @test array_cmp(@inferred(var(VA_aosa, w)), _var(VA_aosa, w))
        @test array_cmp(@inferred(var(VA_aosa, w, corrected = false)), _var(VA_aosa, w, corrected = false))

        @test array_cmp(@inferred(std(VV_aosa, w)), _std(VV_aosa, w))
        @test array_cmp(@inferred(std(VV_aosa, w, corrected = false)), _std(VV_aosa, w, corrected = false))
        @test array_cmp(@inferred(std(VA_aosa, w)), _std(VA_aosa, w))
        @test array_cmp(@inferred(std(VA_aosa, w, corrected = false)), _std(VA_aosa, w, corrected = false))
    end


    # For weighted cov of Vector{Vector}, StatsBase currently returns a vector
    # instead of a matrix, with `cov(VV, fill(1, 10)) != cov(VV)`.
    # This should be considered a bug in StatsBase.

    function _cov(A::AbstractVector{<:AbstractVector}, w::FrequencyWeights; corrected = true)
        wmean_A = _mean(A, w)
        wsum = sum(w)
        wsum_corr = corrected ? -1 : 0
        sum([[(A[i][j] - wmean_A[j]) * (A[i][k] - wmean_A[k]) * w[i] for j in eachindex(A[i]), k in eachindex(A[i])] for i in eachindex(A)]) ./ (wsum + wsum_corr)
    end

    @testset "cov" begin
        @test array_cmp(@inferred(cov(VV_aosa, w)), _cov(VV_aosa, w))
        @test array_cmp(@inferred(cov(VV_aosa, w, corrected = false)), _cov(VV_aosa, w, corrected = false))
    end


    # Weighted cor is currently not supported for Vector{Vector} by StatsBase.
    # This should be considered a bug in StatsBase, since unweighted cor
    # for Vector{Vector} is supported by Statistics.

    _cor(A::AbstractVector{<:AbstractVector}, w::AbstractWeights) = cov2cor(_cov(A, w), _std(A, w))

    @testset "cor" begin
        @test array_cmp(@inferred(cor(VV_aosa, w)), _cor(VV, w))
    end
end
