# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


Base.sum(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights) where {T,M} = sum(flatview(X), w, M + 1)

Statistics.mean(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights) where {T,M} =
    vec(mean(flatview(X), w, dims = M + 1))

Statistics.var(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights; corrected::Bool = true) where {T,M} =
    vec(var(flatview(X), w, M + 1; corrected = corrected))

Statistics.cov(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true) =
    cov(flatview(X), w, 2; corrected = corrected)

Statistics.cor(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights) =
    cor(flatview(X), w, 2)
