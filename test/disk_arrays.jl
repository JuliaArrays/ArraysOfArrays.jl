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

# A struct type whose parameters constrain each other:
struct _Constrained{T,V<:AbstractVector{T}}
    x::T
    v::V
end

# A broadcastable that is not an array:
struct _Positions
    n::Int
end
Base.axes(p::_Positions) = (Base.OneTo(p.n),)
Base.ndims(::Type{_Positions}) = 1
Base.getindex(::_Positions, i::Int) = i
Base.getindex(p::_Positions, I::CartesianIndex{1}) = p[only(Tuple(I))]
Base.broadcastable(p::_Positions) = p
Base.Broadcast.BroadcastStyle(::Type{_Positions}) = Base.Broadcast.DefaultArrayStyle{1}()

# A minimal disk-backed array with optional chunking that counts block reads:
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

# The element type of disk-backed element arrays, used as a field type
# that in-memory blocks cannot be converted to:
const _DiskElem = eltype(ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(zeros(UInt16, 1, 1))))
struct _FixedDisk
    x::_DiskElem
end

# A struct type with a type parameter composed into a field type:
struct _Nest{T}
    x::Tuple{T}
end

# A struct type whose StructArray columns follow a different order than
# its fields:
struct _Swapped{A,B}
    a::A
    b::B
end
StructArrays.staticschema(::Type{_Swapped{A,B}}) where {A,B} = NamedTuple{(:b, :a), Tuple{B, A}}
StructArrays.createinstance(::Type{T}, b, a) where {T<:_Swapped} = T(a, b)

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

    # Likewise, the selected elements of a vector of arrays, which may be
    # empty, unsorted or repeated, are gathered from the flat data at once:
    vdisk = CountingDiskArray(rand(40))
    Vd = VectorOfArrays(vdisk, [1, 3, 3, 10, 21, 41], fill((), 5))
    Vd_ref = VectorOfArrays(vdisk.data, Vd.elem_ptr, Vd.kernel_size)
    for idxs in ([2, 5, 1, 1, 4], [true, false, true, true, false], Int[])
        vdisk.reads = 0
        r = Vd[idxs]
        @test vdisk.reads <= length(idxs)
        @test fused(r) isa Vector{Float64}
        @test r == Vd_ref[idxs]
    end
    # Elements far apart in the flat data, selected out of order (which
    # DiskArrays before 0.4.25 gathered wrongly), and matrix elements:
    Vf = VectorOfArrays(CountingDiskArray(Float32.(1:200)), [1, 6, 150, 155, 201], fill((), 4))
    @test Vf[[3, 1]] == VectorOfArrays(Vf.data.data, Vf.elem_ptr, Vf.kernel_size)[[3, 1]]
    Vm = VectorOfArrays(CountingDiskArray(rand(12)), [1, 5, 5, 13], [(2,), (0,), (4,)])
    @test Vm[[3, 1, 2]] == VectorOfArrays(Vm.data.data, Vm.elem_ptr, Vm.kernel_size)[[3, 1, 2]]
    # Narrow element pointer types are kept, and checked against the size of
    # the selection before anything is read:
    Vn = VectorOfArrays(CountingDiskArray(rand(200)), UInt8[1, 101, 201], fill((), 2))
    @test Vn[[2, 1]].elem_ptr isa Vector{UInt8}
    Vn.data.reads = 0
    @test_throws ArgumentError Vn[[2, 2, 2]]
    @test Vn.data.reads == 0

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
    # nested style, without a style conflict:
    @test Broadcast.combine_styles(A, e) isa NestedArrayStyle{1}
    @test Base.BroadcastStyle(typeof(sa)) isa StructArrays.StructArrayStyle{NestedArrayStyle{1},1}

    # Block lengths are multiples of the chunk length, up to the default chunk
    # size of DiskArrays: the data is read in one block with the default of
    # 100 MB, and in blocks of 15 chunks with a budget of 1 MB:
    default_budget = DiskArrays.default_chunk_size[]
    for (budget, nblocks) in ((default_budget, 1), (1, cld(n, 32 * fld(10^6 ÷ 2000, 32))))
        DiskArrays.default_chunk_size[] = budget
        try
            reads()
            @test sum.(A) == sum.(A_ref)
            @test reads() == (nblocks, 0)
            dest = zeros(UInt64, n)
            dest .= sum.(A)
            @test dest == sum.(A_ref)
            @test reads() == (nblocks, 0)
            r = (x -> 2 .* x).(A)
            @test r isa VectorOfArrays
            @test r == [2 .* x for x in A_ref]
            @test reads() == (nblocks, 0)
            @test ((x, y) -> sum(x) + y).(A, e) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
            @test reads() == (nblocks, nblocks)
            # Repeated arguments are read once per block:
            @test sum.(A) ./ length.(A) == sum.(A_ref) ./ length.(A_ref)
            @test reads() == (nblocks, 0)
            # Singleton arguments are read once per block and do not shorten
            # the blocks:
            z0 = CountingDiskArray(fill(2.0))
            @test ((x, y) -> sum(x) + y).(A, z0) == sum.(A_ref) .+ 2.0
            @test z0.reads == nblocks
            big1 = ArrayOfSimilarArrays{Float64,1,1}(CountingDiskArray(rand(250_000, 1)))
            @test ((x, y) -> sum(x) + sum(y)).(A, big1) == sum.(A_ref) .+ sum(big1.data.data)
            @test big1.data.reads == nblocks
            @test reads() == (2nblocks, 0)
        finally
            DiskArrays.default_chunk_size[] = default_budget
        end
    end

    DiskArrays.default_chunk_size[] = 1
    try
        # Inference with a function whose result type can be inferred for
        # disk-backed element arrays (unlike sum on Julia 1.10):
        f3 = (x, y, z) -> length(x) / length(y) + z
        @test @inferred(broadcast(f3, A, A, e)) == f3.(A_ref, A_ref, e.data)
        dest = zeros(UInt64, n)
        @test @inferred(Broadcast.materialize!(dest, Broadcast.broadcasted(sum, A))) == sum.(A_ref)
        @test (x -> x .> 0x8000).(A) == [x .> 0x8000 for x in A_ref]
        @test ((x, y) -> sum(x) + y).(A, v) == [sum(x) + y for (x, y) in zip(A_ref, v)]
        @test ((y, x) -> sum(x) + y).(e, A) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
        @test sum.(A) .+ 1 == sum.(A_ref) .+ 1
        @test sum.(view(A, 11:60)) == sum.(A_ref[11:60])
        e.reads = 0
        @test ((x, y) -> sum(x) + y).(VectorOfSimilarVectors(data), e) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
        @test e.reads == 1
        # Tuple and singleton arguments broadcast along the axis:
        @test ((x, y) -> sum(x) + y).(view(A, 1:4), (1, 2, 3, 4)) == sum.(A_ref[1:4]) .+ (1:4)
        @test ((x, y) -> sum(x) + y).(A, (7,)) == sum.(A_ref) .+ 7

        # Arguments aliasing the destination are copied first, as in Base:
        dest2 = collect(1.0:n)
        expected = sum.(A_ref) .+ reverse(dest2)
        dest2 .= ((x, y) -> sum(x) + y).(A, view(dest2, n:-1:1))
        @test dest2 == expected
        # A destination that is itself an argument is not copied. Disk arrays
        # do not support that copy:
        x = collect(1.0:n)
        x .= x .+ sum.(A)
        @test x == (1:n) .+ sum.(A_ref)
        d = CountingDiskArray(zeros(n))
        d .= d .+ sum.(A)
        @test d.data == sum.(A_ref)

        # Joining blocks of 480 elements with different element types widens
        # the result type as a single broadcast would:
        f_wide = (x, i) -> i <= 480 ? 1 : 1.5
        r_wide = f_wide.(A, 1:n)
        @test eltype(r_wide) == eltype(f_wide.(A_ref, 1:n))
        @test r_wide == f_wide.(A_ref, 1:n)

        # StructArrays with disk-backed columns:
        @test (x -> sum(x.wf) + x.e).(sa) == [sum(x) + y for (x, y) in zip(A_ref, e.data)]
        S = (x -> (s = sum(x.wf), e = x.e)).(sa)
        @test S isa StructArray
        @test S.s == sum.(A_ref) && S.e == e.data
        @test (x -> 2 .* x.wf).(sa) isa VectorOfArrays
        # Struct-valued results whose type is not inferred, here differing
        # between blocks, are stored in a StructArray as well:
        f_wide_struct = (x, i) -> Base.inferencebarrier((s = sum(x.wf), k = i <= 480 ? i : float(i)))
        S_wide = f_wide_struct.(sa, 1:n)
        @test S_wide isa StructArray
        @test S_wide == f_wide_struct.(StructArray((wf = A_ref, e = e.data, v = v)), 1:n)
        # ... unless StructArrays cannot rebuild them from their fields:
        tagged_wide = (x, i) -> i <= 480 ? _Tagged{:a,Int}(i) : _Tagged{:b,Int}(i)
        @test tagged_wide.(sa, 1:n) == [tagged_wide(nothing, i) for i in 1:n]
    finally
        DiskArrays.default_chunk_size[] = default_budget
    end

    # Results of struct-valued functions are inferred from the declared
    # element type for both disk-backed and in-memory columns:
    timedata = fill(0.0:0.5:3.0, n)
    for W in (Waveform, Waveform{eltype(timedata),eltype(A)})
        sw = StructArray{W}((timedata, A))
        sw_mem = StructArray{W === Waveform ? W : Waveform{eltype(timedata),eltype(A_ref)}}((timedata, VectorOfSimilarVectors(data)))
        f = w -> Waveform(w.time, 2 .* w.signal)
        reads()
        r, r_mem = f.(sw), f.(sw_mem)
        @test reads() == (1, 0)
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

    # In-memory StructArrays keep their element type, disk-backed ones keep
    # type parameters that their fields do not determine. Tuples and scalars
    # allow evaluation in blocks, broadcastables with axes that are neither
    # arrays nor tuples (like _Positions) prevent it:
    tagged = StructArray{_Tagged{3,Float64}}((v,))
    @test ((t, x) -> t.x + sum(x)).(tagged, A) == v .+ sum.(A_ref)
    tagged_disk = StructArray{_Tagged{3,eltype(A)}}((A,))
    @test (t -> t isa _Tagged{3}).(tagged_disk) == trues(n)
    @test (t -> sum(t.x)).(tagged_disk) == sum.(A_ref)

    # Only the declared types of disk-backed columns change in blocks. Other
    # declared types still convert the column elements:
    x32 = Float32.(v)
    Af = ArrayOfSimilarArrays{Float64,1,1}(CountingDiskArray(Float64.(data)))
    cs = StructArray{_Constrained{Float64,eltype(Af)}}((x32, Af))
    @test (c -> c.x + sum(c.v)).(cs) == Float64.(x32) .+ sum.(A_ref)
    ns = StructArray{@NamedTuple{x::Float64, w::eltype(A)}}((x32, A))
    @test (s -> s.x).(ns) == Float64.(x32)
    @test eltype((s -> s.x).(ns)) == Float64

    # Zero-dimensional StructArrays with disk-backed columns:
    s0 = StructArray((e = CountingDiskArray(fill(2.0)),))
    @test ((x, s) -> sum(x) + s.e).(A, s0) == sum.(A_ref) .+ 2.0

    # Columns follow StructArrays.staticschema. StructArrays are accessed
    # element by element if their declared element type has fields that cannot
    # hold in-memory blocks (a fixed field type, a type parameter composed into
    # a field type):
    swapped = StructArray{_Swapped{Float64,eltype(A)}}((b = A, a = e))
    reads()
    @test (s -> s.a + sum(s.b)).(swapped) == e.data .+ sum.(A_ref)
    @test reads() == (1, 1)
    fixed = StructArray{_FixedDisk}((A,))
    @test (t -> sum(t.x)).(fixed) == sum.(A_ref)
    nested = StructArray{_Nest{eltype(A)}}((StructArray((A,)),))
    @test (t -> sum(only(t.x))).(nested) == sum.(A_ref)

    # Blocks of tuples that change element type along the axis are evaluated
    # with the same argument types, so the result is too:
    A4 = ArrayOfSimilarArrays{Float64,1,1}(CountingDiskArray(rand(3, 4); chunks = (3, 2)))
    DiskArrays.default_chunk_size[] = 0
    try
        f_fill = (x, y) -> fill(y, 2)
        t = (1, 2, 3.0, 4.0)
        r4 = f_fill.(A4, t)
        @test A4.data.reads == 2
        @test typeof(r4) == typeof(f_fill.(VectorOfSimilarVectors(A4.data.data), t))
    finally
        DiskArrays.default_chunk_size[] = default_budget
    end
    @test ((x, i) -> sum(x) + i).(A, _Positions(n)) == sum.(A_ref) .+ (1:n)

    # A disk-backed destination that is also an argument, directly or as a
    # StructArray column, must not be copied first, which disk arrays could not
    # do:
    D = ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(copy(data)))
    D .= (x -> x .÷ 0x2).(D)
    @test D.data.data == data .÷ 0x2
    dsa = StructArray((s = CountingDiskArray(zeros(n)),))
    dsa .= ((s, x) -> (s = s + sum(x),)).(dsa.s, A)
    @test dsa.s.data == sum.(A_ref)

    # Vectors of arrays over disk data, read in blocks, and empty input,
    # which DiskArrays cannot index:
    lens = rand(1:50, 300)
    vdata = rand(sum(lens))
    V = VectorOfArrays(CountingDiskArray(vdata; chunks = (1000,)), cumsum([1; lens]), fill((), 300))
    V_ref = VectorOfArrays(vdata, cumsum([1; lens]), fill((), 300))
    DiskArrays.default_chunk_size[] = 0
    try
        @test sum.(V) == sum.(V_ref)
        @test 1 < V.data.reads < 30
        @test (x -> 2 .* x).(V) == [2 .* x for x in V_ref]
    finally
        DiskArrays.default_chunk_size[] = default_budget
    end
    A0 = ArrayOfSimilarArrays{UInt16,1,1}(CountingDiskArray(rand(UInt16, 1000, 0)))
    @test sum.(A0) == UInt64[]
    @test A0.data.reads == 0
    Z = VectorOfSimilarVectors(CountingDiskArray(zeros(UInt16, 0, 4)))
    @test length.(Z) == zeros(Int, 4)
    @test Z[2:3] == [UInt16[], UInt16[]]
    VZ = VectorOfArrays(CountingDiskArray(Float64[]), fill(1, 5), fill((), 4))
    @test length.(VZ) == zeros(Int, 4)
    @test stacked(VZ) == zeros(0, 4)

    # Only vectors of arrays are evaluated in blocks. Other nested arrays
    # still support broadcast:
    A2 = ArrayOfSimilarArrays{UInt16,1,2}(CountingDiskArray(rand(UInt16, 5, 4, 3)))
    @test sum.(A2) == sum.(collect(A2))

    @testset "no method ambiguities in the DiskArrays extension" begin
        ext = Base.get_extension(ArraysOfArrays, :ArraysOfArraysDiskArraysExt)
        @test ext !== nothing
        @test isempty(detect_ambiguities(ext))
    end
end
