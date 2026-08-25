# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: FuseArrays
using InverseFunctions: inverse, NoInverse, test_inverse

@testset "InverseFunctions extension" begin
    test_inverse(NonSplitMode{2}(), rand(2, 3), compare = ==)
    test_inverse(SplitSlices{1,2}(), rand(2, 3, 4), compare = ==)
    test_inverse(SplitParts([1, 4, 8, 11], fill((), 3)), collect(1:10), compare = ==)
    test_inverse(getsplitmode(eachslice(rand(2, 3, 4), dims = 2)), rand(2, 3, 4), compare = ==)

    # The parts need not cover the data, fused still returns the full
    # data vector:
    x = collect(1:10)
    smode = SplitParts([3, 5, 8], fill((), 2))
    @test @inferred(inverse(smode)(smode(x))) === x

    @test @inferred(inverse(SplitSlices{1,2}())) === FuseArrays(SplitSlices{1,2}())
    @test @inferred(inverse(inverse(SplitSlices{1,2}()))) === SplitSlices{1,2}()
    @test @inferred(inverse(NonSplitMode{2}())) === NonSplitMode{2}()
    @test inverse(UnknownSplitMode{Vector{Vector{Int}}}()) isa NoInverse

    A = VectorOfArrays([[1, 2], [3, 4, 5]])
    smode_A = getsplitmode(A)
    @test inverse(inverse(smode_A)) === smode_A
    # getsplitmode copies the shape information, so equal modes and their
    # fusing functions need not be egal:
    @test inverse(smode_A) == FuseArrays(getsplitmode(A))
    @test hash(inverse(smode_A)) == hash(FuseArrays(getsplitmode(A)))

    # Mode mismatches are detected where possible in O(1):
    @test_throws ArgumentError inverse(SplitSlices{1,2}())(rand(2, 3))
    @test_throws ArgumentError inverse(SplitSlices{1,1}())(sliced(rand(2, 3, 4), 2))
    X = rand(2, 3, 4)
    bs = getsplitmode(eachslice(X, dims = 2))
    @test_throws ArgumentError inverse(bs)(eachslice(X, dims = 3))
    @test_throws ArgumentError inverse(bs)(X)
    @test_throws ArgumentError inverse(smode_A)(VectorOfArrays([[1, 2]]))
    @test_throws ArgumentError inverse(smode_A)(VectorOfArrays([[1, 2, 3], [4, 5]]))
    @test_throws ArgumentError inverse(smode_A)(sliced(rand(2, 3), 1))
    @test inverse(smode_A)(A) === fused(A)
    @test FuseArrays(NonSplitMode{2}())(rand(2, 3)) isa Matrix
    @test_throws ArgumentError FuseArrays(NonSplitMode{2}())(rand(3))

    # The inverse of a SplitSlices mode applies stacked, so it also
    # stacks generic nested arrays and non-memory-ordered slicings, which
    # fused does not support resp. does not reorder:
    C = [[1, 2], [3, 4], [5, 6]]
    test_inverse(inverse(unstackmode(C)), C, compare = ==)
    @test inverse(unstackmode(C))(C) == [1 3 5; 2 4 6]
    D = eachrow(rand(2, 3))
    test_inverse(inverse(unstackmode(D)), D, compare = ==)
    @test inverse(unstackmode(D))(D) == stacked(D)
end
