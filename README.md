# ArraysOfArrays.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaArrays.github.io/ArraysOfArrays.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaArrays.github.io/ArraysOfArrays.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Travis Build Status](https://travis-ci.com/JuliaArrays/ArraysOfArrays.jl.svg?branch=master)](https://travis-ci.com/JuliaArrays/ArraysOfArrays.jl)
[![Codecov](https://codecov.io/gh/JuliaArrays/ArraysOfArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaArrays/ArraysOfArrays.jl)


A Julia package for efficient storage and handling of nested arrays.

ArraysOfArrays provides two different types of nested arrays: `ArrayOfSimilarArrays` and `VectorOfArrays`.
An `ArrayOfSimilarArrays` offers a duality of view between representing the same data as both a flat multi-dimensional array and as an array of equally-sized arrays. A `VectorOfArrays` represents a vector of arrays of equal dimensionality but different size. Internally, both types store their data in flat arrays that are accessible to the user `flatview()`.

## Documentation

* [Documentation for stable version](https://JuliaArrays.github.io/ArraysOfArrays.jl/stable)
* [Documentation for development version](https://JuliaArrays.github.io/ArraysOfArrays.jl/dev)
