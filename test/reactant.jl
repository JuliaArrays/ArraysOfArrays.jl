# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays
using InverseFunctions: inverse
import Reactant
using Reactant: @jit

# CPU is available everywhere and sufficient to test tracing compatibility:
Reactant.set_default_backend("cpu")

@testset "Reactant extension" begin
    x_cpu = rand(3, 5)
    x = Reactant.to_rarray(x_cpu)

    f_rt(x) = fused(splitup(x, StaticSlices(SVector{3})))
    @test Array(@jit f_rt(x)) ≈ x_cpu

    f_stacked(x) = stacked(splitup(x, StaticSlices(SVector{3})))
    @test Array(@jit f_stacked(x)) ≈ x_cpu

    f_elem(x) = sum(splitup(x, StaticSlices(SVector{3}))[2])
    @test Float64(@jit f_elem(x)) ≈ sum(x_cpu[:, 2])

    f_map(x) = sum(map(v -> sum(abs2, v), splitup(x, StaticSlices(SVector{3}))))
    @test Float64(@jit f_map(x)) ≈ sum(abs2, x_cpu)

    f_innersum(x) = innersum(splitup(x, StaticSlices(SVector{3})))
    @test Array(@jit f_innersum(x)) ≈ vec(sum(x_cpu, dims = 1))

    f_inverse(x) = inverse(StaticSlices(SVector{3}))(splitup(x, StaticSlices(SVector{3})))
    @test Array(@jit f_inverse(x)) ≈ x_cpu

    f_flatview(x) = flatview(splitup(x, StaticSlices(SVector{3})))
    @test Array(@jit f_flatview(x)) ≈ x_cpu

    f_props(x) = begin
        B = splitup(x, StaticSlices(SVector{3}))
        (unstackmode(B) === getsplitmode(B), Base.IndexStyle(typeof(B)) isa IndexCartesian)
    end
    @test @jit(f_props(x)) == (true, true)

    # The traced split has the same split mode as the reinterpret-based
    # one, with genuine static-array elements:
    f_mode_and_eltype(x) = begin
        B = splitup(x, StaticSlices(SVector{3}))
        (getsplitmode(B) === StaticSlices(SVector{3}), eltype(B) <: SVector{3})
    end
    @test @jit(f_mode_and_eltype(x)) == (true, true)

    Y_cpu = rand(2, 3, 5)
    Y = Reactant.to_rarray(Y_cpu)
    f_smatrix(x) = fused(splitup(x, StaticSlices(SMatrix{2,3})))
    @test Array(@jit f_smatrix(Y)) ≈ Y_cpu
end
