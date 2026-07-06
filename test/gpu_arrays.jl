# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using Adapt
using GPUArraysCore: AbstractGPUArray
using JLArrays
import KernelAbstractions as KA

include("testdefs.jl")

# JLArray is the reference AbstractGPUArray implementation, tests must not
# perform scalar indexing on it:
JLArrays.allowscalar(false)

@testset "GPU arrays (via JLArrays)" begin
    @testset "ArrayOfSimilarArrays on device data" begin
        x = jl(rand(Float32, 3, 4, 5))
        A = @inferred ArrayOfSimilarArrays{Float32,1,2}(x)

        @test A[2, 3] isa AbstractGPUArray
        @test @inferred(flatview(A)) === x
        @test @inferred(fused(A)) === x
        @test splitup(fused(A), getsplitmode(A)) == A
        @test @inferred(innersize(A)) == (3,)
        @test collect(fused(innermap(abs, A))) == abs.(collect(x))
        @test stack(A) isa AbstractGPUArray
        A[1, 1] = jl(rand(Float32, 3))

        V = VectorOfSimilarVectors(jl(rand(Float32, 3, 5)))
        @test sum(V) isa AbstractGPUArray
        @test ArraysOfArrays.Statistics.mean(V) isa AbstractGPUArray
        @test vcat(V, V) isa VectorOfSimilarVectors
        @test fused(vcat(V, V)) isa AbstractGPUArray

        @test KA.get_backend(A) == KA.get_backend(x)
    end

    @testset "VectorOfArrays, device data and host shape info" begin
        xv = jl(rand(Float32, 10))
        V = VectorOfArrays(xv, [1, 3, 6, 11], [(), (), ()])
        @test V[2] isa AbstractGPUArray
        @test @inferred(vecflattened(V)) isa AbstractGPUArray
        @test reduce(vcat, [V, V]) isa VectorOfArrays
        @test partitioned(xv, [2, 3, 5]) isa VectorOfArrays
        @test KA.get_backend(V) == KA.get_backend(xv)
    end

    @testset "VectorOfArrays, everything on device" begin
        xv = jl(rand(Float32, 10))

        # Consistency checks must work without scalar indexing:
        V = VectorOfArrays(xv, jl([1, 3, 6, 11]), jl([(), (), ()]))
        @test V isa VectorOfArrays
        @test length(V) == 3
        @test fused(V) === xv

        V2 = VectorOfArrays(jl(rand(Float32, 12)), jl([1, 5, 13]), jl([(2,), (2,)]))
        @test length(V2) == 2

        # Inconsistent shape info must still be detected:
        @test_throws ArgumentError VectorOfArrays(xv, jl([1, 3, 12]), jl([(), ()]))
        @test_throws ArgumentError VectorOfArrays(xv, jl([1, 6, 3]), jl([(), ()]))
        @test_throws ArgumentError VectorOfArrays(xv, jl([1, 4, 11]), jl([(2,), (2,)]))

        # Zero-size kernels are valid for empty elements, invalid otherwise:
        @test VectorOfArrays(xv, jl([1, 1, 11]), jl([(0,), (2,)])) isa VectorOfArrays
        @test_throws ArgumentError VectorOfArrays(xv, jl([1, 3, 11]), jl([(0,), (2,)]))

        # mapat and innerlengths only touch device data and shape info:
        @test collect(fused(mapat(abs2, Val(2), V))) == abs2.(collect(xv))
        @test innerlengths(V) isa AbstractGPUArray
        @test collect(innerlengths(V)) == [2, 3, 5]
        @test collect(innersizes(V2)) == [(2, 2), (2, 4)]

        # bcastat compiles to flat device broadcasts:
        v_outer = jl(Float32[10, 20, 30])
        r_bc = bcastat(+, Val(2), V, v_outer)
        @test r_bc isa VectorOfArrays
        @test fused(r_bc) isa AbstractGPUArray
        @test collect(fused(r_bc)) == collect(xv) .+ [10, 10, 20, 20, 20, 30, 30, 30, 30, 30]

        # Segmented reductions run as a single device kernel:
        xv_h = collect(xv)
        parts_h = [xv_h[1:2], xv_h[3:5], xv_h[6:10]]
        @test innersum(V) isa AbstractGPUArray
        @test collect(innersum(V)) ≈ sum.(parts_h)
        @test collect(innermapreduce(abs2, +, V)) ≈ [sum(abs2, p) for p in parts_h]
        @test collect(innerreduce(max, V; init = -Inf32)) ≈ maximum.(parts_h)
        @test_throws ArgumentError innerreduce(max, VectorOfArrays(xv, jl([1, 3, 3, 11]), jl([(), (), ()])))

        # Split mode round trip on device shape info:
        sm = getsplitmode(V)
        @test splitup(fused(V), sm) isa VectorOfArrays
        @test typeof(splitup(fused(V), sm)) == typeof(V)

        # adapt moves data and shape info:
        V_host = VectorOfArrays([Float32[1, 2], Float32[3, 4, 5]])
        V_dev = adapt(JLArray, V_host)
        @test V_dev.data isa AbstractGPUArray
        @test V_dev.elem_ptr isa AbstractGPUArray
        @test adapt(Array, V_dev) == V_host
        @test adapt(Array, getsplitmode(V_dev)).elem_ptr == getsplitmode(V_host).elem_ptr
    end
end
