# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    sum(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights)

Compute the sum of the elements vectors of `X` with weights `w`. Equivalent
to `sum` of `flatview(X)` along dimension 2.
"""
Main.sum(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights) = sum(flatview(X), w, 2)


"""
    mean(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights)

Compute the mean of the elements vectors of `X` with weights `w`. Equivalent
to `mean` of `flatview(X)` along dimension 2.
"""
Statistics.mean(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights) = mean(flatview(X), w, 2)


"""
    cov(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true)

Compute the covariance matrix between the elements of the elements of `X`
along `X` with weights `w`. Equivalent to `cov` of `flatview(X)` along
dimension 2.
"""
Statistics.cov(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true) =
    cov(flatview(X), w, 2; corrected = corrected)


"""
    cor(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights)

Compute the Pearson correlation matrix between the elements of the elements of
 `X` along `X` with weights `w`. Equivalent to `cor` of `flatview(X)` along
 dimension 2.
"""
Statistics.cor(X::VectorOfSimilarVectors, w::StatsBase.AbstractWeights) = cor(flatview(X), w, 2)
