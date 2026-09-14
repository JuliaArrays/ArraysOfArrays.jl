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

@testset "stacking disk-backed vectors of arrays" begin
    # Stacking and conversion of a disk-backed VectorOfArrays read the
    # covered data in a single block read and return in-memory arrays:
    vdata = rand(UInt16, 12)
    vdisk = CountingDiskArray(vdata)
    Vd = VectorOfArrays(vdisk, [1, 4, 7, 10, 13], fill((), 4))
    Vd_ref = VectorOfArrays(vdata, [1, 4, 7, 10, 13], fill((), 4))

    # The shape information alone determines the inner size, no reads:
    vdisk.reads = 0
    @test innersize(Vd) == (3,)
    @test vdisk.reads == 0

    S = stacked(Vd)
    @test S isa Matrix{UInt16}
    @test vdisk.reads == 1
    @test S == stacked(Vd_ref)

    vdisk.reads = 0
    C = convert(VectorOfSimilarVectors, Vd)
    @test fused(C) isa Matrix{UInt16}
    @test vdisk.reads == 1
    @test C == Vd_ref

    # Only the covered range of a partially covered data vector is read:
    Vp = VectorOfArrays(vdisk, [4, 7, 10], fill((), 2))
    vdisk.reads = 0
    @test stacked(Vp) == stacked(VectorOfArrays(vdata, [4, 7, 10], fill((), 2)))
    @test vdisk.reads == 1

    vdisk.reads = 0
    @test_throws DimensionMismatch stacked(VectorOfArrays(vdisk, [1, 3, 7, 13], fill((), 3)))
    @test vdisk.reads == 0
end
