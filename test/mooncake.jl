# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

import Mooncake
import Random

@testset "Mooncake extension" begin
    rng = Random.Xoshiro(0x2b6ee80184a75b47)

    x_flat = rand(rng, 3, 4)
    x_vec = rand(rng, 10)
    lengths = [2, 3, 5]

    @testset "zero-derivative primitives" begin
        A = sliced(x_flat, Val(1))
        p = partitioned(x_vec, lengths)

        Mooncake.TestUtils.test_rule(rng, getsplitmode, A; is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, getsplitmode, p; is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, unstackmode, A; is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, is_memordered_splitmode, getsplitmode(A); is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, innersize, A; is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, getslicemap, A; is_primitive = true, unsafe_perturb = true)
        Mooncake.TestUtils.test_rule(rng, consgrouped_ptrs, [1, 1, 2, 3, 3]; is_primitive = true, unsafe_perturb = true)
    end

    # Mooncake differentiates the ArraysOfArrays types and operations without
    # custom rules, verify gradient correctness end-to-end:
    @testset "derived rules" begin
        for (label, f, x) in [
            ("sliced and fused", x -> sum(abs2, fused(sliced(x, Val(1)))), x_flat),
            ("stacked", x -> sum(stacked(ArrayOfSimilarArrays{Float64,1,1}(x))), x_flat),
            ("vecflattened of ArrayOfSimilarArrays", x -> sum(abs2, vecflattened(ArrayOfSimilarArrays{Float64,1,1}(x))), x_flat),
            ("partitioned", x -> sum(sum, partitioned(x, lengths)), x_vec),
            ("VectorOfArrays round trip", x -> begin
                p = partitioned(x, lengths)
                sum(abs2, fused(splitup(fused(p), getsplitmode(p))))
            end, x_vec),
            ("reduce vcat", x -> sum(sum, reduce(vcat, [partitioned(x, [2, 3]), partitioned(x, [4])])), x_vec),
            ("innermap", x -> sum(fused(innermap(abs2, sliced(x, Val(1))))), x_flat),
        ]
            @testset "$label" begin
                Mooncake.TestUtils.test_rule(rng, f, x; is_primitive = false, unsafe_perturb = true)
            end
        end
    end
end
