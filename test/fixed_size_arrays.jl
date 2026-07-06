# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using FixedSizeArrays: FixedSizeVector

include("testdefs.jl")

@testset "FixedSizeArrays extension" begin
    A = VectorOfArrays([[1, 2], [3, 4, 5]])

    # Fixed-size shape information cannot be resized, so getsplitmode can
    # share it without a defensive copy:
    B = VectorOfArrays(A.data, FixedSizeVector(A.elem_ptr), FixedSizeVector(A.kernel_size))
    sm = getsplitmode(B)
    @test sm.elem_ptr === B.elem_ptr
    @test sm.kernel_size === B.kernel_size
    @test splitup(fused(B), sm) == B
    @test typeof(splitup(fused(B), sm)) == typeof(B)
    @test B == A
    test_api(B, Array(B), B.data)

    # Resizable shape information still gets copied:
    sm_A = getsplitmode(A)
    @test sm_A.elem_ptr !== A.elem_ptr
    @test sm_A.kernel_size !== A.kernel_size

    # partitioned with fixed-size lengths results in fixed-size shape
    # information:
    p = partitioned(collect(1:10), FixedSizeVector([2, 3, 5]))
    @test p == [[1, 2], [3, 4, 5], [6, 7, 8, 9, 10]]
    @test p.elem_ptr isa FixedSizeVector
    @test getsplitmode(p).elem_ptr === p.elem_ptr
end
