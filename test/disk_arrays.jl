# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using DiskArrays
using StructArrays
using ArraysOfArrays: NestedArrayStyle
using Base: Broadcast

include("waveform_defs.jl")

# A struct type with a type parameter that its fields do not determine:
struct _Tagged{N,X}
    x::X
end

# A broadcastable that is not an array:
struct _Positions
    n::Int
end
Base.axes(p::_Positions) = (Base.OneTo(p.n),)
Base.ndims(::Type{_Positions}) = 1
Base.getindex(::_Positions, i::Int) = i
Base.broadcastable(p::_Positions) = p
Base.Broadcast.BroadcastStyle(::Type{_Positions}) = Base.Broadcast.DefaultArrayStyle{1}()

# A minimal disk-backed array that counts block reads, optionally chunked:
mutable struct CountingDiskArray{T,N} <: DiskArrays.AbstractDiskArray{T,N}
    data::Array{T,N}
    reads::Int
    chunks::Union{Nothing,Dims{N}}
end
CountingDiskArray(a::Array; chunks = nothing) = CountingDiskArray(a, 0, chunks)
Base.size(a::CountingDiskArray) = size(a.data)
DiskArrays.haschunks(a::CountingDiskArray) = a.chunks === nothing ? DiskArrays.Unchunked() : DiskArrays.Chunked()
DiskArrays.eachchunk(a::CountingDiskArray) =
    a.chunks === nothing ? DiskArrays.estimate_chunksize(a) : DiskArrays.GridChunks(a, a.chunks)
function DiskArrays.readblock!(a::CountingDiskArray, aout, r::AbstractUnitRange...)
    a.reads += 1
    aout .= view(a.data, r...)
end
DiskArrays.writeblock!(a::CountingDiskArray, ain, r::AbstractUnitRange...) = (view(a.data, r...) .= ain; nothing)

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

@testset "block-wise broadcasts over disk-backed data" begin
    n = 2000
    data = rand(UInt16, 1000, n)   # 2 kB per element
    disk = CountingDiskArray(data; chunks = (1000, 32))
    A = ArrayOfSimilarArrays{UInt16,1,1}(disk)
    A_ref = collect(ArrayOfSimilarArrays{UInt16,1,1}(data))
    e = CountingDiskArray(rand(n))
    v = rand(n)
    sa = StructArray((wf = A, e = e, v = v))
    reads() = (r = (disk.reads, e.reads); disk.reads = 0; e.reads = 0; r)

    # A plain disk array combined with a nested array broadcasts with the
    # nested style, instead of conflicting:
    @test Broadcast.combine_styles(A, e) isa NestedArrayStyle{1}
    @test Base.BroadcastStyle(typeof(sa)) isa StructArrays.StructArrayStyle{NestedArrayStyle{1},1}

    # Blocks consist of whole storage chunks, up to the default chunk size of
    # DiskArrays: the data is read in one block with the default of 100 MB,
    # and in blocks of 15 chunks with a budget of 1 MB:
    default_budget = DiskArrays.default_chunk_size[]
    for (budget, nblocks) in ((default_budget, 1), (1, cld(n, 32 * fld(10^6 ÷ 2000, 32))))
        DiskArrays.default_chunk_size[] = budget
        try
            reads()
            @test sum.(A) == sum.(A_ref)
            @test reads() == (nblocks, 0)
            @test map(sum, A) == sum.(A_ref)
            reads()
            dest = zeros(UInt64, n)
            dest .= sum.(A)
            @test dest == sum.(A_ref)
            @test reads() == (nblocks, 0)
            r = (x -> 2 .* x).(A)
            @test r isa VectorOfArrays
            @test r == [2 .* x for x in A_ref]
            @test reads() == (nblocks, 0)
            @test (x -> x .> 0x8000).(A) == [x .> 0x8000 for x in A_ref]
            @test ((x, y) -> sum(x) + y).(A, v) == [sum(x) + y for (x, y) in zip(A_ref, v)]
            @test ((x, y) -> sum(x) + y).(A, e) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
            @test reads() == (3nblocks, nblocks)
            @test ((y, x) -> sum(x) + y).(e, A) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
            @test sum.(A) .+ 1 == sum.(A_ref) .+ 1
            @test sum.(view(A, 11:60)) == sum.(A_ref[11:60])
            @test reads() == (2nblocks + 1, nblocks)
            # Tuple, 0-dim and singleton arguments broadcast along the axis;
            # a disk-backed singleton is read once per block and does not
            # shorten the blocks:
            @test ((x, y) -> sum(x) + y).(view(A, 1:4), (1, 2, 3, 4)) == sum.(A_ref[1:4]) .+ (1:4)
            @test ((x, y) -> sum(x) + y).(A, (7,)) == sum.(A_ref) .+ 7
            z0 = CountingDiskArray(fill(2.0))
            @test ((x, y) -> sum(x) + y).(A, z0) == sum.(A_ref) .+ 2.0
            @test reads() == (2nblocks + 1, 0)
            @test z0.reads == nblocks
            big1 = ArrayOfSimilarArrays{Float64,1,1}(CountingDiskArray(rand(250_000, 1)))
            @test ((x, y) -> sum(x) + sum(y)).(A, big1) == sum.(A_ref) .+ sum(big1.data.data)
            @test reads() == (nblocks, 0)
            @test big1.data.reads == nblocks

            # Arguments aliasing the destination are copied first, as in Base:
            dest2 = collect(1.0:n)
            expected = sum.(A_ref) .+ reverse(dest2)
            dest2 .= ((x, y) -> sum(x) + y).(A, view(dest2, n:-1:1))
            @test dest2 == expected
            @test reads() == (nblocks, 0)
            # ... but a destination that is an argument itself is not, which
            # disk arrays could not do:
            x = collect(1.0:n)
            x .= x .+ sum.(A)
            @test x == (1:n) .+ sum.(A_ref)
            d = CountingDiskArray(zeros(n))
            d .= d .+ sum.(A)
            @test d.data == sum.(A_ref)
            @test reads() == (2nblocks, 0)

            # Blocks widening to different element types are joined like a
            # single broadcast widens (blocks of 480 elements with a 1 MB budget):
            f_wide = (x, i) -> i <= 480 ? 1 : 1.5
            r_wide = f_wide.(A, 1:n)
            @test eltype(r_wide) == eltype(f_wide.(A_ref, 1:n))
            @test r_wide == f_wide.(A_ref, 1:n)
            @test reads() == (nblocks, 0)

            # StructArrays with disk-backed columns:
            @test (x -> sum(x.wf) + x.e).(sa) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
            S = (x -> (s = sum(x.wf), e = x.e)).(sa)
            @test S isa StructArray
            @test S.s == sum.(A_ref) && S.e == e.data
            @test (x -> 2 .* x.wf).(sa) isa VectorOfArrays
            @test reads() == (3nblocks, 3nblocks)
        finally
            DiskArrays.default_chunk_size[] = default_budget
        end
    end

    # Results of struct-valued functions are inferred from the declared
    # element type, for disk-backed like for in-memory columns:
    timedata = fill(0.0:0.5:3.0, n)
    for W in (Waveform, Waveform{eltype(timedata),eltype(A)})
        sw = StructArray{W}((timedata, A))
        sw_mem = StructArray{W === Waveform ? W : Waveform{eltype(timedata),eltype(A_ref)}}((timedata, VectorOfSimilarVectors(data)))
        f = w -> Waveform(w.time, 2 .* w.signal)
        r, r_mem = f.(sw), f.(sw_mem)
        @test typeof(r).name == typeof(r_mem).name
        @test (r isa StructArray ? r.signal : getfield.(r, :signal)) == [2 .* x for x in A_ref]
    end

    # Arguments that broadcast along the outer axis are read once, disk-backed
    # ones included:
    A5 = VectorOfSimilarVectors(data[:, 1:5])
    e1 = CountingDiskArray(rand(1))
    @test ((x, y) -> sum(x) + y).(A5, e1) == [sum(x) + only(e1.data) for x in A5]
    @test e1.reads == 1
    A1 = ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(data[:, 1:1]))
    @test ((x, y) -> sum(x) + y).(A1, v[1:5]) == [sum(A_ref[1]) + y for y in v[1:5]]
    @test A1.data.reads == 1
    sa1 = StructArray((e = CountingDiskArray(rand(1)),))
    @test ((x, y) -> sum(x) + y.e).(A5, sa1) == [sum(x) + only(sa1.e.data) for x in A5]
    dest5 = zeros(5)
    dest5 .= sum.(A1)
    @test dest5 == fill(sum(A_ref[1]), 5)

    # In-memory StructArrays keep their element type, broadcastables that
    # are not arrays prevent evaluation in blocks:
    tagged = StructArray{_Tagged{3,Float64}}((v,))
    @test ((t, x) -> t.x + sum(x)).(tagged, A) == v .+ sum.(A_ref)
    @test ((x, i) -> sum(x) + i).(A, _Positions(n)) == sum.(A_ref) .+ (1:n)

    # A disk-backed destination that is also an argument:
    D = ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(copy(data)))
    D .= (x -> x .÷ 0x2).(D)
    @test D.data.data == data .÷ 0x2

    # Vectors of arrays over disk data, and empty input, which DiskArrays
    # cannot even index:
    V = VectorOfArrays(CountingDiskArray(rand(30)), [1, 4, 9, 20, 31], fill((), 4))
    V_ref = [sum(x) for x in V]
    V.data.reads = 0
    @test sum.(V) == V_ref
    @test V.data.reads == 1
    A0 = ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(rand(UInt16, 1000, 0)))
    @test sum.(A0) == UInt64[]
    @test A0.data.reads == 0

    # Only vectors of arrays are evaluated in blocks:
    A2 = ArrayOfSimilarArrays{UInt16,1,2}(CountingDiskArray(rand(UInt16, 5, 4, 3)))
    A2_ref = sum.(collect(A2))
    A2.data.reads = 0
    @test sum.(A2) == A2_ref
    @test A2.data.reads == 12

    @testset "no method ambiguities in the DiskArrays extension" begin
        ext = Base.get_extension(ArraysOfArrays, :ArraysOfArraysDiskArraysExt)
        @test ext !== nothing
        @test isempty(detect_ambiguities(ext))
    end
end
