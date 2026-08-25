# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: getinnerdims, getouterdims
using OffsetArrays: OffsetArray

struct _TestSplitMode <: AbstractSplitMode end
struct _TestPartMode <: ArraysOfArrays.AbstractPartMode{1,1} end

include("testdefs.jl")

@testset "functions" begin
    A_0 = fill(Float32(42))
    A_0_flat = fill(Float32(42))

    A_1 = Float32[3, 8, 4, 6]
    A_1_flat = Float32[3, 8, 4, 6]

    A_1e = Float32[]
    A_1e_flat = Float32[]

    A_2 = [Float32[3 8; 4 6], Float32[1 2; 5 7; 9 0]]
    A_2_flat = Float32[3, 8, 4, 6, 1, 2, 5, 7, 9, 0]

    A_2e = Matrix{Float32}[]
    A_2e_flat = Matrix{Float32}(undef, 0, 0)

    A_2b = [Float32[3 8; 4 6], Float32[1 2; 9 0]]
    A_2b_flat = stack(A_2b)

    A_3 = [[Float32[3, 8, 4], Float32[1, 2, 5]], [Float32[6, 7], Float32[9, 0]]]
    A_3_flat = reduce(vcat, A_3)

    @testset "splitup" begin
        # NonSplitMode is validated against the array's dimensionality:
        @test @inferred(splitup(A_1, NonSplitMode{1}())) === A_1
        @test_throws ArgumentError splitup(A_1, NonSplitMode{2}())
    end

    @testset "getsplitmode" begin
        @test @inferred(getsplitmode(A_0)) isa NonSplitMode{0}
        @test @inferred(getsplitmode(A_1)) isa NonSplitMode{1}
        @test @inferred(getsplitmode(A_1e)) isa NonSplitMode{1}
        @test @inferred(getsplitmode(A_2)) isa UnknownSplitMode{typeof(A_2)}
        @test @inferred(getsplitmode(A_2e)) isa UnknownSplitMode{typeof(A_2e)}
        @test @inferred(getsplitmode(A_2b)) isa UnknownSplitMode{typeof(A_2b)}
        @test @inferred(getsplitmode(A_3)) isa UnknownSplitMode{typeof(A_3)}
    end

    @testset "innersize" begin
        @test @inferred(innersize(A_0)) == ()
        @test @inferred(innersize(A_1)) == ()
        @test @inferred(innersize(A_1e)) == ()
        @test_throws DimensionMismatch innersize(A_2)
        @test @inferred(innersize(A_2e)) == (0, 0)
        @test @inferred(innersize(A_2b)) == (2, 2)
        @test @inferred(innersize(A_3)) == (2,)
    end

    @testset "innermap" begin
        f = x -> x^2
        @test @inferred(innermap(f, A_0)) == fill(1764)
        @test @inferred(innermap(f, A_1)) == Float32[9, 64, 16, 36]
        @test @inferred(innermap(f, A_2)) == [Float32[9 64; 16 36], Float32[1 4; 25 49; 81 0]]
        @test_throws MethodError innermap(f, A_3)

        @test @inferred(innermap(length, A_0)) == fill(1)
        @test @inferred(innermap(length, A_1)) == fill(1, 4)
        @test @inferred(innermap(length, A_2)) == [fill(1, 2, 2), fill(1, 3, 2)]
        @test @inferred(innermap(length, A_3)) == [fill(3, 2), fill(2, 2)]
    end

    @testset "deepmap" begin
        f = x -> x^2
        @test @inferred(deepmap(f, A_0)) == fill(1764)
        @test @inferred(deepmap(f, A_1)) == Float32[9, 64, 16, 36]
        @test @inferred(deepmap(f, A_2)) == [Float32[9 64; 16 36], Float32[1 4; 25 49; 81 0]]
        @test @inferred(deepmap(f, A_3)) == [[Float32[9, 64, 16], Float32[1, 4, 25]], [Float32[36, 49], Float32[81, 0]]]

        @test @inferred(deepmap(length, A_0)) == fill(1)
        @test @inferred(deepmap(length, A_1)) == fill(1, 4)
        @test @inferred(deepmap(length, A_2)) == [fill(1, 2, 2), fill(1, 3, 2)]
        @test @inferred(deepmap(length, A_3)) == [[fill(1, 3), fill(1, 3)], [fill(1, 2), fill(1, 2)]]
    end

    @testset "mapat" begin
        f = x -> x^2
        @test @inferred(mapat(f, Val(1), A_1)) == map(f, A_1)
        @test @inferred(mapat(f, Val(2), A_2)) == innermap(f, A_2)
        @test @inferred(mapat(f, Val(3), A_3)) == deepmap(f, A_3)
        @test mapat(length, Val(1), A_2) == length.(A_2)

        # Depth exceeding the nesting depth applies at the innermost level:
        @test mapat(f, Val(5), A_1) == map(f, A_1)

        # Generic nested arrays combine elementwise and may differ in type:
        @test mapat(+, Val(2), [[1, 2], [3]], [[10.0, 20.0], [30.0]]) == [[11.0, 22.0], [33.0]]

        # Multiple inputs, zipped like map:
        @test @inferred(mapat(+, Val(1), A_1, A_1)) == 2 .* A_1
        @test mapat(+, Val(2), A_2, A_2) == innermap(x -> 2 * x, A_2)

        # Integer depth relies on constant propagation for type stability:
        mapat_intdepth(g, A) = mapat(g, 2, A)
        @test @inferred(mapat_intdepth(f, A_2)) == innermap(f, A_2)
    end

    @testset "sliced" begin
        m = rand(2, 3)
        @test sliced(m) == sliced(m, Val(1)) == eachcol(m)
        # Zero inner dimensions produce zero-dimensional element views, like
        # Base's eachslice over all dimensions:
        @test sliced(m, Val(0)) == eachslice(m, dims = (1, 2))
        @test sliced(m, Val(2)) == fill(m)
        @test_throws ArgumentError sliced(m, Val(3))
        @test_throws ArgumentError sliced(m, Val(-1))
    end

    @testset "vecflattened" begin
        # Non-nested arrays are returned resp. reshaped, nested arrays are
        # concatenated:
        v = Float32[3, 8, 4]
        @test @inferred(vecflattened(v)) === v
        m = Float32[1 2; 3 4]
        @test @inferred(vecflattened(m)) == vec(m)
        @test vecflattened([[1, 2], [3]]) == [1, 2, 3]
        @test vecflattened([[1 2; 3 4], [5 6]]) == [1, 3, 2, 4, 5, 6]
        @test vecflattened(reshape([[1, 2], [3, 4]], 1, 2)) == [1, 2, 3, 4]
        @test vecflattened(reshape([[1 2], [3 4]], 1, 2)) == [1, 2, 3, 4]

        # Slicings with several outer dimensions resp. with
        # multi-dimensional elements:
        A = reshape(collect(1:24), 2, 3, 4)
        @test vecflattened(eachslice(A, dims = (2, 3))) == vec(A)
        @test vecflattened(eachslice(A, dims = 3)) == vec(A)
    end

    @testset "innersum" begin
        @test innersum(A_2) == [sum(x) for x in A_2]
        # Element arrays that don't contain numbers are summed elementwise:
        @test innersum([[[1, 2], [3, 4]], [[5, 6]]]) == [[4, 6], [5, 6]]
    end

    @testset "innersizes and innerlengths" begin
        @test @inferred(innersizes(A_2)) == size.(A_2)
        @test @inferred(innerlengths(A_2)) == length.(A_2)
        @test innersizes(A_3) == size.(A_3)
        @test innerlengths(A_2e) == Int[]
    end

    @testset "Nested array API" begin
        test_api(A_0, A_0_flat)
        test_api(A_1, A_1_flat)
        test_api(A_1e, A_1e_flat)
        test_api(A_2, A_2_flat)
        test_api(A_2e, A_2e_flat)
        test_api(A_2b, A_2b_flat)
        test_api(A_3, A_3_flat)
    end

    @testset "generic fallbacks" begin
        @test unstackmode(rand(2, 3)) === NonSplitMode{2}()
        @test unstackmode(AbstractArray[[1], [2, 3]]) isa UnknownSplitMode

        # splitup of a stacked array cannot reproduce offset outer axes, so
        # unstackmode refuses instead of promising a broken round trip:
        Mo = OffsetArray(reshape(collect(1.0:12.0), 3, 4), 1:3, 0:3)
        @test unstackmode(eachcol(Mo)) isa UnknownSplitMode
        @test unstackmode(OffsetArray([[1, 2], [3, 4]], 0:1)) isa UnknownSplitMode
        @test unstackmode(eachcol(reshape(collect(1.0:12.0), 3, 4))) === SplitSlices{1,1}()
        @test ArraysOfArrays._fused_impl(fill(1.0, 2), NonSplitMode{1}()) == fill(1.0, 2)

        # Third-party split modes fall back to fused in FuseArrays and must
        # specialize _bcast_expand for outer-value bcastat arguments:
        @test ArraysOfArrays.FuseArrays(_TestSplitMode())([1, 2, 3]) == [1, 2, 3]
        @test_throws ArgumentError ArraysOfArrays._bcast_expand([1, 2], _TestPartMode(), [1, 2, 3])
    end
end
