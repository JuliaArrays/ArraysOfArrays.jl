# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


Base.sum(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights) where {T,M} =
    sum(flatview(X), w, dims = M + 1)[_ncolons(Val{M}())...]

Statistics.mean(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights) where {T,M} =
    mean(flatview(X), w, dims = M + 1)[_ncolons(Val{M}())...]

Statistics.var(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights; corrected::Bool = true) where {T,M} =
    var(flatview(X), w, M + 1; corrected = corrected)[_ncolons(Val{M}())...]

Statistics.std(X::AbstractVectorOfSimilarArrays{T,M}, w::StatsBase.AbstractWeights; corrected::Bool = true) where {T,M} =
    std(flatview(X), w, M + 1; corrected = corrected)[_ncolons(Val{M}())...]

Statistics.cov(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true) =
    cov(flatview(X), w, 2; corrected = corrected)

Statistics.cor(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights) =
    cor(flatview(X), w, 2)
