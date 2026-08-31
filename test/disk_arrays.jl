# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using DiskArrays

# A minimal disk-backed array that counts block reads:
mutable struct CountingDiskArray{T,N} <: DiskArrays.AbstractDiskArray{T,N}
    data::Array{T,N}
    reads::Int
end
CountingDiskArray(a::Array) = CountingDiskArray(a, 0)
Base.size(a::CountingDiskArray) = size(a.data)
DiskArrays.haschunks(::CountingDiskArray) = DiskArrays.Unchunked()
function DiskArrays.readblock!(a::CountingDiskArray, aout, r::AbstractUnitRange...)
    a.reads += 1
    aout .= view(a.data, r...)
end

@testset "disk-backed data" begin
    data = rand(UInt16, 40, 50)
    disk = CountingDiskArray(data)
    V = VectorOfSimilarVectors(disk)
    V_ref = VectorOfSimilarVectors(data)

    # The extension enables the flat indexing path for disk arrays, the
    # default is the generic path. Behind an inference barrier, so that the
    # constant-foldable trait methods actually run and count as covered:
    @test ArraysOfArrays._prefers_flat_getindex(Base.inferencebarrier(typeof(disk)))
    @test !ArraysOfArrays._prefers_flat_getindex(Base.inferencebarrier(typeof(data)))

    # Scalar element access stays lazy, no reads:
    disk.reads = 0
    el = V[5]
    @test disk.reads == 0
    @test collect(el) == V_ref[5]

    # Non-scalar indexing reads the selection from the flat data in a
    # bounded number of block reads instead of one read per scalar:
    gather = [7, 3, 49, 20]
    mask = falses(50)
    mask[gather] .= true
    for (idxs, max_reads) in [(2:26, 1), ((:), 1), (gather, length(gather)), (mask, length(gather))]
        disk.reads = 0
        r = V[idxs]
        @test disk.reads <= max_reads
        @test r isa VectorOfSimilarVectors{UInt16}
        # The result is materialized in memory:
        @test fused(r) isa Array
        @test r == V_ref[idxs]
    end

    # Bounds violations, including mask axes, are caught before any read:
    disk.reads = 0
    @test_throws BoundsError V[0:3]
    @test_throws BoundsError V[[true, false]]
    @test disk.reads == 0

    # Higher-dimensional inner and outer structure:
    disk4 = CountingDiskArray(rand(2, 3, 4, 5))
    A = ArrayOfSimilarArrays{Float64,2,2}(disk4)
    A_ref = ArrayOfSimilarArrays{Float64,2,2}(disk4.data)
    disk4.reads = 0
    @test A[2:3, 1:2] == A_ref[2:3, 1:2]
    @test disk4.reads == 1
    @test A[:, 2] == A_ref[:, 2]
    # Scalar access stays a lazy view of the disk data:
    disk4.reads = 0
    @test size(A[2, 3]) == (2, 3)
    @test disk4.reads == 0
end
