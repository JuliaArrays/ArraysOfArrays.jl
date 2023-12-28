# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using Test
using ArraysOfArrays
import Documenter

Documenter.DocMeta.setdocmeta!(
    ArraysOfArrays,
    :DocTestSetup,
    :(using ArraysOfArrays);
    recursive=true,
)
Documenter.doctest(ArraysOfArrays)
