# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Statistics
using Test

using Adapt

using ArraysOfArrays: full_consistency_checks, append_elemptr!, element_ptr

include("testdefs.jl")


@testset "vector_of_arrays" begin
    ref_flatview(A::AbstractVector{<:AbstractArray}) = vcat(map(vec, Array(A))...)

    ref_AoA1(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 5)][1:0] : [rand(T, rand(1:9)) for i in 1:n]
    ref_AoA2(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 4, 2)][1:0] : [rand(T, rand(1:4), rand(1:4)) for i in 1:n]
    ref_AoA3(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 3, 2, 4)][1:0] : [rand(T, rand(1:3), rand(1:3), rand(1:3)) for i in 1:n]

    @testset "element pointer handling" begin
        A = [2, 4, 5, 9]; B = [3, 6, 8]
        @test @inferred(append_elemptr!(deepcopy(A), B)) == [2, 4, 5, 9, 12, 14]
        @test @inferred(append_elemptr!([2], [5])) == [2]
    end


    @testset "consistency checks" begin
        # Elements with zero-size kernel dimensions are valid:
        V = VectorOfArrays([zeros(0, 3), ones(2, 3)])
        @test VectorOfArrays(V.data, V.elem_ptr, V.kernel_size) == V

        # Data length per element must be a multiple of the kernel length:
        @test_throws ArgumentError VectorOfArrays(collect(1.0:3.0), [1, 4], [(2,)])
        # A zero-length kernel requires an empty element:
        @test_throws ArgumentError VectorOfArrays(collect(1.0:3.0), [1, 3], [(0,)])
    end


    @testset "ctors" begin
        A1 = ref_AoA1(Float32, 5)
        @test @inferred(VectorOfArrays(deepcopy(A1))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays(deepcopy(A1)) == A1
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64}(deepcopy(A1)) == A1
        @test @inferred(VectorOfArrays{Float64,1}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64,1}(deepcopy(A1)) == A1

        @test @inferred(PartsView(deepcopy(A1))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test PartsView(deepcopy(A1)) == A1
        @test @inferred(PartsView{Float64}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test PartsView{Float64}(deepcopy(A1)) == A1

        A1_empty = ref_AoA1(Float32, 0)
        @test @inferred(VectorOfArrays(deepcopy(A1_empty))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays(deepcopy(A1_empty)) == A1_empty
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64}(deepcopy(A1_empty)) == A1_empty
        @test @inferred(VectorOfArrays{Float64,1}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64,1}(deepcopy(A1_empty)) == A1_empty

        @test @inferred(PartsView(deepcopy(A1_empty))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test PartsView(deepcopy(A1_empty)) == A1_empty
        @test @inferred(PartsView{Float64}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test PartsView{Float64}(deepcopy(A1_empty)) == A1_empty

        A2 = ref_AoA2(Float32, 4)
        @test @inferred(VectorOfArrays(deepcopy(A2))) isa VectorOfArrays{Float32,2,1,Array{Float32,1},Array{Int,1},Array{Tuple{Int},1}}
        @test VectorOfArrays(deepcopy(A2)) == A2
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A2))) isa VectorOfArrays{Float64,2,1,Array{Float64,1},Array{Int,1},Array{Tuple{Int},1}}
        @test VectorOfArrays{Float64}(deepcopy(A2)) == A2
        @test @inferred(VectorOfArrays{Float64,2}(deepcopy(A2))) isa VectorOfArrays{Float64,2,1,Array{Float64,1},Array{Int,1},Array{Tuple{Int},1}}
        @test VectorOfArrays{Float64,2}(deepcopy(A2)) == A2

        A3 = ref_AoA3(Float32, 3)
        @test @inferred(VectorOfArrays(deepcopy(A3))) isa VectorOfArrays{Float32,3,2,Array{Float32,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays(deepcopy(A3)) == A3
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A3))) isa VectorOfArrays{Float64,3,2,Array{Float64,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays{Float64}(deepcopy(A3)) == A3
        @test @inferred(VectorOfArrays{Float64,3}(deepcopy(A3))) isa VectorOfArrays{Float64,3,2,Array{Float64,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays{Float64,3}(deepcopy(A3)) == A3

        A3_empty = ref_AoA3(Float32, 0)
        @test @inferred(VectorOfArrays(deepcopy(A3_empty))) isa VectorOfArrays{Float32,3,2,Array{Float32,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays(deepcopy(A3_empty)) == A3_empty
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A3_empty))) isa VectorOfArrays{Float64,3,2,Array{Float64,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays{Float64}(deepcopy(A3_empty)) == A3_empty
        @test @inferred(VectorOfArrays{Float64,3}(deepcopy(A3_empty))) isa VectorOfArrays{Float64,3,2,Array{Float64,1},Array{Int,1},Array{Tuple{Int,Int},1}}
        @test VectorOfArrays{Float64,3}(deepcopy(A3_empty)) == A3_empty
    end


    @testset "append! and vcat" begin
        A1 = ref_AoA3(Float32, 3); A2 = ref_AoA3(Float32, 0)
        A3 = ref_AoA3(Float32, 4); A4 = ref_AoA3(Float64, 2)

        B1 = VectorOfArrays(A1); B2 = VectorOfArrays(A2);
        B3 = VectorOfArrays(A3); B4 = VectorOfArrays(A4);

        # vcat must return an independent array:
        @test @inferred(vcat(B1)) == B1
        @test vcat(B1) !== B1
        @test vcat(B1).data !== B1.data
        full_consistency_checks(vcat(B1))

        @test @inferred(vcat(B1, B2)) isa VectorOfArrays
        @test vcat(B1, B2) == vcat(A1, A2)
        @test eltype(vcat(B1, B2)) <: AbstractArray{Float32,3}
        @test eltype(vcat(B1, B2)) == eltype(B1)
        full_consistency_checks(vcat(B1, B2))

        @test @inferred(vcat(B1, B3)) isa VectorOfArrays
        @test vcat(B1, B3) == vcat(A1, A3)
        @test eltype(vcat(B1, B3)) <: AbstractArray{Float32,3}
        full_consistency_checks(vcat(B1, B3))

        @test @inferred(vcat(B1, B2, B3, B4)) isa VectorOfArrays
        @test vcat(B1, B2, B3, B4) == vcat(A1, A2, A3, A4)
        @test eltype(vcat(B1, B2, B3, B4)) <: AbstractArray{Float64,3}
        full_consistency_checks(vcat(B1, B2, B3, B4))

        B1_copy = @inferred(copy(B1)); B3_copy = @inferred(copy(B3))
        append!(B1_copy, B3_copy)
        @test B1_copy.data == vcat(B1.data, B3.data)

        # reduce(vcat, ...) uses a single-pass implementation:
        @test @inferred(reduce(vcat, [B1, B2, B3])) isa VectorOfArrays
        @test reduce(vcat, [B1, B2, B3]) == vcat(A1, A2, A3)
        @test reduce(vcat, [B1, B2, B3]) == vcat(B1, B2, B3)
        full_consistency_checks(reduce(vcat, [B1, B2, B3]))

        # vcat and append! of vectors of arrays with unused data regions
        # must only use the data covered by their elements:
        p_partial = partitioned(collect(1:10), [2, 3])
        q = VectorOfArrays([[6], [7, 8]])
        @test vcat(p_partial, q) == vcat(collect(p_partial), collect(q))
        @test reduce(vcat, [p_partial, q]) == vcat(collect(p_partial), collect(q))
        full_consistency_checks(vcat(p_partial, q))

        q_copy = copy(q)
        append!(q_copy, p_partial)
        @test q_copy == vcat(collect(q), collect(p_partial))
        full_consistency_checks(q_copy)

        # Appending to a VectorOfArrays with unused trailing data would
        # corrupt it:
        @test_throws ArgumentError append!(partitioned(collect(1:10), [2, 3]), q)
    end


    @testset "vcat of VectorOfSimilarArrays" begin
        VsoA = [VectorOfSimilarArrays(rand(Float32, 2, 3, n)) for n in (2, 0, 4)]
        Vs_ref = vcat(map(collect, VsoA)...)

        @test @inferred(vcat(VsoA[1])) isa VectorOfSimilarArrays
        @test vcat(VsoA[1]) == VsoA[1]
        @test vcat(VsoA[1]).data !== VsoA[1].data

        @test @inferred(vcat(VsoA...)) isa VectorOfSimilarArrays
        @test vcat(VsoA...) == Vs_ref
        @test @inferred(reduce(vcat, VsoA)) isa VectorOfSimilarArrays
        @test reduce(vcat, VsoA) == Vs_ref

        # Element type promotion like Base.vcat:
        V64 = VectorOfSimilarArrays(rand(Float64, 2, 3, 2))
        @test eltype(eltype(vcat(VsoA[1], V64))) == Float64
        @test vcat(VsoA[1], V64) == vcat(collect(VsoA[1]), collect(V64))

        # Inner size mismatch:
        V_bad = VectorOfSimilarArrays(rand(Float32, 3, 2, 2))
        @test_throws DimensionMismatch vcat(VsoA[1], V_bad)

        # Vectors of similar vectors:
        VsoV = [VectorOfSimilarVectors(rand(3, n)) for n in (2, 4)]
        @test @inferred(reduce(vcat, VsoV)) isa VectorOfSimilarVectors
        @test reduce(vcat, VsoV) == vcat(map(collect, VsoV)...)
    end


    @testset "equality" begin
        # == and isequal must be equivalent to elementwise comparison,
        # independent of the underlying data layout:
        p = partitioned(collect(1:10), [2, 3])   # unused data after covered range
        q = VectorOfArrays([[1, 2], [3, 4, 5]])
        @test collect(p) == collect(q)
        @test p == q
        @test q == p
        @test isequal(p, q)

        @test p != VectorOfArrays([[1, 2], [3, 4, 6]])
        @test p != VectorOfArrays([[1, 2, 3], [4, 5]])
        @test p != VectorOfArrays([[1, 2], [3, 4, 5], [6]])

        # Equal-length parts with different shapes are not equal:
        r1 = VectorOfArrays([[1 2; 3 4]])
        r2 = VectorOfArrays([[1 3; 2 4]])
        r3 = partitioned(collect([1, 3, 2, 4]), [(2, 2)])
        @test r1 != r2
        @test r1 == permutedims.(r2)
        @test r1 == r3 && r2 != r3

        # missing propagates through ==, but not isequal:
        m1 = VectorOfArrays([[1, missing], [3]])
        m2 = VectorOfArrays([[1, missing], [3]])
        @test ismissing(m1 == m2)
        @test isequal(m1, m2)
    end


    @testset "split mode API" begin
        B1 = VectorOfArrays(ref_AoA1(Float32, 5))
        B1e = VectorOfArrays(ref_AoA1(Float32, 0))
        B3 = VectorOfArrays(ref_AoA3(Float32, 3))
        Bu = VectorOfArrays([rand(Float32, 2, 3) for i in 1:4])

        for B in [B1, B1e, B3, Bu]
            @test @inferred(getsplitmode(B)) isa AbstractPartMode{ndims(eltype(B)),1}
            @test @inferred(fused(B)) === B.data
            @test @inferred(splitup(fused(B), getsplitmode(B))) == B
            @test typeof(splitup(fused(B), getsplitmode(B))) == typeof(B)
            test_api(B, Array(B), B.data)
        end

        # mapat operates on the flat data and preserves structure:
        @test @inferred(mapat(abs2, Val(2), B3)) == innermap(abs2, B3)
        @test typeof(mapat(abs2, Val(2), B3)) == typeof(B3)
        @test mapat(+, Val(2), B1, B1) == [x .+ x for x in B1]
        @test_throws DimensionMismatch mapat(+, Val(2), B1, B3)

        @test @inferred(innerlengths(B1)) == length.(collect(B1))
        @test @inferred(innersizes(B3)) == size.(collect(B3))

        # getsplitmode must not be affected by later resizing:
        B_grow = VectorOfArrays([[1, 2], [3, 4, 5]])
        sm_grow = getsplitmode(B_grow)
        push!(B_grow, [6])
        @test splitup(collect(1:5), sm_grow) == [[1, 2], [3, 4, 5]]

        # Uniform element size, so stackable:
        @test @inferred(stacked(Bu)) == stack(Array(Bu))
        @test @inferred(splitup(stacked(Bu), unstackmode(Bu))) == Bu

        # flatview on view-backed VectorOfArrays returns a view of only the
        # covered data range, in a type-stable fashion:
        B3_view = view(B3, 2:3)
        @test @inferred(flatview(B3_view)) isa SubArray
        @test flatview(B3_view) == B3.data[B3.elem_ptr[2]:(B3.elem_ptr[4] - 1)]
        @test @inferred(vecflattened(B3_view)) == flatview(B3_view)
        @test @inferred(flatview(B3)) === B3.data
    end


    @testset "partitioned" begin
        x = collect(1:10)

        p = @inferred(partitioned(x, [2, 3, 5]))
        @test p isa PartsView{Int}
        @test p == [[1, 2], [3, 4, 5], [6, 7, 8, 9, 10]]
        @test @inferred(fused(p)) === x
        @test flatview(p) === x
        test_api(p, Array(p), x)

        # vecflattened may share memory, reduce/mapreduce must not:
        @test @inferred(vecflattened(p)) == x
        @test parent(vecflattened(p)) === x
        @test @inferred(reduce(vcat, p)) == x
        @test reduce(vcat, p) !== x
        @test typeof(reduce(vcat, p)) == typeof(x)
        @test @inferred(mapreduce(vec, vcat, p)) == x
        @test mapreduce(vec, vcat, p) !== x

        # Partial partitions are allowed:
        p_partial = @inferred(partitioned(x, [2, 3]))
        @test p_partial == [[1, 2], [3, 4, 5]]

        @test_throws ArgumentError partitioned(x, [2, 3, 6])
        @test_throws ArgumentError partitioned(x, [2, -1, 5])

        p2 = @inferred(partitioned(x, [(1, 2), (2, 2)]))
        @test p2 isa VectorOfArrays{Int,2}
        @test p2[1] == [1 2]
        @test p2[2] == [3 5; 4 6]
        @test @inferred(fused(p2)) === x
    end


    @testset "bcastat" begin
        x = collect(Float32, 1:10)
        p = partitioned(x, [2, 3, 5])
        v = Float32[10, 20, 30]

        # One value per element, broadcast over the element contents:
        r = @inferred bcastat(+, Val(2), p, v)
        @test r isa PartsView
        @test r == [xi .+ vi for (xi, vi) in zip(p, v)]

        # Scalars broadcast over everything:
        @test bcastat(+, Val(2), p, 1) == [xi .+ 1 for xi in p]

        # Aligned nested arguments and flat-matching arguments:
        @test bcastat(+, Val(2), p, p) == [xi .+ xi for xi in p]
        @test bcastat(+, Val(2), p, x) == [xi .+ xi for xi in p]

        # Mixed argument kinds in one call:
        @test bcastat(muladd, Val(2), p, v, 2) == [muladd.(xi, vi, 2) for (xi, vi) in zip(p, v)]

        # Data not covered by any part gets no contribution:
        p_part = partitioned(x, [2, 3])
        @test bcastat(+, Val(2), p_part, Float32[10, 20]) == [Float32[11, 12], Float32[23, 24, 25]]

        # Depth exceeding the nesting depth applies at the innermost level:
        @test bcastat(+, Val(3), p, v) == bcastat(+, Val(2), p, v)

        # Integer depth relies on constant propagation for type stability:
        bcastat_intdepth(g, A, y) = bcastat(g, 2, A, y)
        @test @inferred(bcastat_intdepth(+, p, v)) == bcastat(+, Val(2), p, v)

        # Two nesting levels over a single flat buffer:
        VV = VectorOfArrays(partitioned(collect(1.0:10.0), [2, 3, 5]), [1, 3, 4], [(), ()])
        w = [100.0, 200.0]
        r2 = bcastat(+, Val(3), VV, w)
        @test r2 isa VectorOfArrays
        @test fused(fused(r2)) == [101, 102, 103, 104, 105, 206, 207, 208, 209, 210]

        @test_throws DimensionMismatch bcastat(+, Val(2), p, Float32[1, 2])
        @test_throws DimensionMismatch bcastat(+, Val(2), p, partitioned(x, [4, 6]))
        @test_throws ArgumentError bcastat(+, Val(2), [[1, 2], [3]], 1)
    end

    @testset "outer broadcast" begin
        A = VectorOfArrays([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
        A_ref = collect(A)

        # Array-valued results at the outer level yield a VectorOfArrays:
        r = (x -> 2 .* x).(A)
        @test r isa VectorOfArrays{Float64,1}
        @test r == [2 .* x for x in A_ref]
        full_consistency_checks(r)

        # Results may be ragged in new ways:
        r2 = (x -> vcat(x, sum(x))).(A)
        @test r2 isa VectorOfArrays{Float64,1}
        @test r2 == [vcat(x, sum(x)) for x in A_ref]

        # Multiple and mixed arguments:
        @test broadcast((x, y) -> x .+ y, A, A) == [x .+ x for x in A_ref]
        @test broadcast((x, y) -> x .+ y, A, A) isa VectorOfArrays
        @test broadcast((x, s) -> x .* s, A, 2.0) isa VectorOfArrays
        @test broadcast((x, s) -> x .+ s, A, [10.0, 20.0, 30.0]) == [x .+ s for (x, s) in zip(A_ref, [10.0, 20.0, 30.0])]

        # Scalar-valued results stay plain arrays:
        @test sum.(A) isa Vector{Float64}
        @test sum.(A) == sum.(A_ref)

        # Matrix elements:
        B = VectorOfArrays([rand(2, 2), rand(3, 2)])
        rB = (x -> 2 .* x).(B)
        @test rB isa VectorOfArrays{Float64,2}
        @test rB == [2 .* x for x in collect(B)]

        # VectorOfSimilarArrays broadcasts to VectorOfArrays too:
        C = VectorOfSimilarVectors(rand(3, 4))
        rC = (x -> x .+ 1).(C)
        @test rC isa VectorOfArrays{Float64,1}
        @test rC == [x .+ 1 for x in collect(C)]

        # Multi-dimensional outer structure falls back to the default
        # behavior:
        D = ArrayOfSimilarArrays{Float64,1,2}(rand(2, 3, 4))
        rD = (x -> 2 .* x).(D)
        @test rD isa Matrix{Vector{Float64}}
        @test rD == [2 .* D[i, j] for i in axes(D, 1), j in axes(D, 2)]
    end

    @testset "inner reductions" begin
        x = collect(Float32, 1:10)
        p = partitioned(x, [2, 3, 5])

        @test @inferred(innermapreduce(abs2, +, p)) == [sum(abs2, xi) for xi in p]
        @test @inferred(innerreduce(max, p)) == [maximum(xi) for xi in p]
        @test @inferred(innersum(p)) == [sum(xi) for xi in p]

        # Empty element arrays require an init value, except for innersum:
        pe = partitioned(collect(Float32, 1:5), [2, 0, 3])
        @test innersum(pe) == Float32[3, 0, 12]
        @test_throws ArgumentError innerreduce(max, pe)
        @test innerreduce(max, pe; init = -Inf32) == Float32[2, -Inf32, 5]
    end

    @testset "rrules" begin
        x = collect(Float64, 1:10)

        for lengths in ([2, 3, 5], [2, 3])
            Y, pb = rrule(partitioned, x, lengths)
            @test Y == partitioned(x, lengths)
            ΔY = [fill(1.0, l) for l in lengths]
            ΔA = pb(ΔY)[2]
            want = zero(x)
            want[1:sum(lengths)] .= 1.0
            @test pb(ΔY)[1] == NoTangent() && pb(ΔY)[3] == NoTangent()
            @test ΔA == want
        end

        Y2, pb2 = rrule(partitioned, x, [(1, 2), (2, 2)])
        @test Y2 == partitioned(x, [(1, 2), (2, 2)])
        ΔA2 = pb2([fill(1.0, 1, 2), fill(1.0, 2, 2)])[2]
        @test ΔA2 == [ones(6); zeros(4)]

        p = partitioned(x, [2, 3])
        y, pb3 = rrule(vecflattened, p)
        @test y == vecflattened(p)
        t = pb3(collect(1.0:5.0))
        @test t[1] == NoTangent()
        @test t[2] isa VectorOfArrays
        @test t[2] == [[1.0, 2.0], [3.0, 4.0, 5.0]]
        @test fused(t[2])[6:10] == zeros(5)
    end


    @testset "indexing" begin
        V1 = @inferred(VectorOfArrays(ref_AoA3(Float32, 3)))
        V2 = @inferred(VectorOfArrays(ref_AoA3(Float32, 3)))
        V12 = vcat(V1, V2)
        ind_style = @inferred(IndexStyle(V12))

        @test ind_style == IndexLinear()

        for i in 1:length(V12)
            @test getindex(V12, i) == V12[i]
        end

        @test getindex(V12, 1:length(V12)) == V12

        @test @inferred(element_ptr(V12)) == V12.elem_ptr

        getindex_of_UR = @inferred(Base._getindex(ind_style, V12, 1:length(V12)))
        getindex_of_vector = @inferred(Base._getindex(ind_style, V12, collect(1:length(V12))))
        @test getindex_of_UR == V12
        @test getindex_of_vector == getindex_of_UR

        VV = @inferred(PartsView{Float64}())
        data = @inferred(rand(5))
        @inferred(push!(VV, data))
        @test @inferred(getindex(VV, 1)) == data
        @test @inferred(size(getindex(VV, 1))) == (5,)


## _view_reshape_spec not yet implemented ##
#       V1_copy = copy(V1)
#       V2_copy = copy(V2)
#       @test setindex!(V1_copy, V1, 1) == V1
#       setindex!(V2_copy, V2, 1)
#       @test V2_copy[1] == V2_copy[2]

## function mul(s) not yet implemented ##
#       sizehint!(v12_copy, 2, (2,2,3))

        zeroed_out = deepmap(x -> 0.0, V12)
        for i in zeroed_out
            @test @inferred(zeros(size(i))) == i
        end

        # Not a good test of variable depth?
        @test innermap(x -> 2*x, V12) == deepmap(x -> 2*x, V12)

    end

    @testset "indexing" begin
        V1 = @inferred(VectorOfArrays(ref_AoA3(Float32, 3)))
        V2 = @inferred(VectorOfArrays(ref_AoA3(Float32, 3)))
        V12 = vcat(V1, V2)
        ind_style = @inferred(IndexStyle(V12))
        @test ind_style == IndexLinear()
        for i in 1:length(V12)
            @test getindex(V12, i) == V12[i]
        end
        @test getindex(V12, 1:length(V12)) == V12

        @test @inferred(element_ptr(V12)) == V12.elem_ptr

        boolidxs = rand(Bool, length(V1))
        @test @inferred(V1[boolidxs]) == V1[eachindex(V1)[boolidxs]]
        

## _view_reshape_spec not yet implemented ##
#       V1_copy = copy(V1)
#       V2_copy = copy(V2)
#       @test setindex!(V1_copy, V1, 1) == V1
#       setindex!(V2_copy, V2, 1)
#       @test V2_copy[1] == V2_copy[2]

## function mul(s) not yet implemented ##
#       sizehint!(v12_copy, 2, (2,2,3))

        # -------------------------------------------------------------------

        zeroed_out = deepmap(x -> 0.0, V12)
        for i in zeroed_out
            @test @inferred(zeros(size(i))) == i
        end

        # Not a good test of variable depth?
        @test innermap(x -> 2*x, V12) == deepmap(x -> 2*x, V12)

        f = x -> 0
        A = @inferred(AbstractArray{AbstractArray{Float64, 3},1}([rand(3,3,3), rand(3,3,3), rand(3,3,3), rand(3,3,3)]))
        A_inner = innermap(f,A)
        A_deep = deepmap(f, A)

        @test A_inner == A_deep

        @test @inferred(size(A_inner)) == @inferred(size(A_deep))
        @test innersize(A,1) == innersize(A,2)
        @test innersize(A,1) == innersize(A,3)
        for i in 1:length(A_deep)
            @test A_deep[i] == zeros(3,3,3)
        end
    end


    @testset "copy" begin
        A = ref_AoA3(Float32, 3);
        B = VectorOfArrays(A);

        @test typeof(@inferred copy(B)) == typeof(B)
        @test copy(A) == A
        @test copy(A) == B
    end


    @testset "empty" begin
        A = ref_AoA3(Float32, 3);
        B = VectorOfArrays(A);

        @test typeof(@inferred empty(B)) == typeof(B)
        @test @inferred(empty(B)) == empty(A)
        @test @inferred(empty!(deepcopy(B))) == empty(A)
        @test @inferred(empty(B)) == @inferred(empty!(deepcopy(B)))
        @test append!(empty(B), B) == A
        @test append!(empty!(deepcopy(B)), B) == A
    end


    @testset "adapt" begin
        A1 = VectorOfArrays(ref_AoA1(Float32, 3))
        @test @inferred(adapt(identity, A1)) == A1
        @test typeof(adapt(identity, A1)) == typeof(A1)

        A3 = VectorOfArrays(ref_AoA3(Float32, 3))
        @test @inferred(adapt(identity, A3)) == A3
        @test typeof(adapt(identity, A3)) == typeof(A3)

        sm = getsplitmode(A3)
        @test @inferred(adapt(identity, sm)) == sm
        @test typeof(adapt(identity, sm)) == typeof(sm)
    end


    @testset "examples" begin
        VA = @inferred(VectorOfArrays{Float64, 2}())

        @inferred(push!(VA, rand(2, 3)))
        @inferred(push!(VA, rand(4, 2)))

        @test @inferred(size(VA[1]) == (2,3))
        @test @inferred(size(VA[2]) == (4,2))

        # -------------------------------------------------------------------

        VV = @inferred(PartsView{Float64}())
        d1 = @inferred(rand(5))
        d2 = @inferred(rand(4))

        @inferred(push!(VV, d1))
        @inferred(push!(VV, d2))

        @test VV[1] == d1
        @test VV[2] == d2

        @test @inferred(size(VV[1])) == (5,)
        @test @inferred(size(VV[2])) == (4,)

        # -------------------------------------------------------------------


        VA_flat = flatview(VA)
        @test view(VA_flat, 7:14) == vec(VA[2])

        fill!(view(VA_flat, 7:14), 2.4)
        @test all(x -> x == 2.4, VA[2])

        fill!(view(VA_flat, 7:14), 4.2)
        @test all(x -> x == 4.2, VA[2])

        # -------------------------------------------------------------------


        @test length(@inferred resize!(VA, 1)) == 1
        @test_throws ArgumentError resize!(VA, 4)

        # -------------------------------------------------------------------

        A = [1, 1, 2, 3, 3, 2, 2, 2]
        A_grouped_ref = [[1, 1], [2], [3, 3], [2, 2, 2]]
        elem_ptr = consgrouped_ptrs(A)
        elem_ptr32 = Int32.(consgrouped_ptrs(A))
        @test first.(@inferred(PartsView(A, elem_ptr))) == [1, 2, 3, 2]
        @test first.(@inferred(PartsView(A, elem_ptr32))) == [1, 2, 3, 2]

        B = [1, 2, 3, 4, 5, 6, 7, 8]
        B_grouped_ref = [[1, 2], [3], [4, 5], [6, 7, 8]]
        @test @inferred(PartsView(B, elem_ptr)) == B_grouped_ref
        @test @inferred(PartsView(B, elem_ptr32)) == B_grouped_ref

        C = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        C_grouped_ref = [[1.1, 2.2], [3.3], [4.4, 5.5], [6.6, 7.7, 8.8]]

        @test @inferred(consgroupedview(A, B)) isa PartsView
        @test consgroupedview(A, B) == B_grouped_ref

        @test @inferred(consgroupedview(A, (B, C))) isa NTuple{2, PartsView}
        @test consgroupedview(A, (B, C)) == (B_grouped_ref, C_grouped_ref)

        data = (a = A, b = B, c = C)
        @test @inferred(consgroupedview(A, data)) isa NamedTuple{(:a, :b, :c),<:NTuple{3,AbstractVector}}
        result = consgroupedview(A, data)
        @test result == (a = A_grouped_ref, b = B_grouped_ref, c = C_grouped_ref)
        @test flatview(result.a) === data.a
        @test flatview(result.b) === data.b
        @test flatview(result.c) === data.c

        nestedV = @inferred(AbstractVector{AbstractArray{Float64, 4}}([rand(4,2,3,1), rand(5,3,1,3), rand(6,4,3,1), rand(9,2,1,2)]))
        VoA1 = @inferred(convert(VectorOfArrays, nestedV))
        @test @inferred(flatview(VoA1)) === VoA1.data == ref_flatview(VoA1)
        @test @inferred(flatview(view(VoA1, 2:3))) == ref_flatview(view(VoA1, 2:3))
        VoA2 = @inferred(convert(VectorOfArrays{Float32, 4}, nestedV))
        @test @inferred(map(Float32, VoA1.data)) == VoA2.data

    end

    @testset "map and broadcast" begin
        A = VectorOfArrays(ref_AoA2(Float32, 4))

        for do_map in (map, broadcast)
            @test @inferred(do_map(identity, A)) === A
        end
    end

    @testset "resize" begin
        A1 = VectorOfArrays{Float64, 1}(ref_AoA1(Float64, 3))
        sizehint!(A1, 5, (10000,))
        @test ccall(:jl_array_size, Int, (Any, UInt), A1.data, 1) == 5*10000
    end
end
