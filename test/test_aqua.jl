# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import ArraysOfArrays

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(ArraysOfArrays))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        ArraysOfArrays,
        ambiguities = true
    )
end # testset

Test.@testset "Public API" begin
    # Everything that is documented but not exported must be marked public,
    # names that start with an underscore are internal by convention:
    @static if VERSION >= v"1.11"
        exported = Set(filter(n -> Base.isexported(ArraysOfArrays, n), names(ArraysOfArrays)))
        documented = Set(Symbol(string(b.var)) for b in keys(Base.Docs.meta(ArraysOfArrays)))
        for name in setdiff(documented, exported)
            startswith(string(name), "_") && continue
            Test.@test Base.ispublic(ArraysOfArrays, name)
        end
    end
end # testset
