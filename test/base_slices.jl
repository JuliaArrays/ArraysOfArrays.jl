# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

import OffsetArrays  # enables similar/broadcast with offset axes

using ArraysOfArrays: getinnerdims, getouterdims

include("testdefs.jl")

@testset "base_slices" begin
    A_orig = rand(5,6,7,8,9)
    A_orig_mat = rand(5,6)

    Aes1 = eachslice(A_orig; dims = (4,5))
    @test @inferred(getsplitmode(Aes1)) isa BaseSlicing{3,2,Tuple{Colon,Colon,Colon,Int,Int}}
    @test @inferred(getinnerdims((1,2,3,4,5), getsplitmode(Aes1))) == (1,2,3)
    @test @inferred(getouterdims((1,2,3,4,5), getsplitmode(Aes1))) == (4,5)

    Aes2 = eachslice(A_orig; dims = (3,1,5))
    @test @inferred(getsplitmode(Aes2)) isa BaseSlicing{2,3,Tuple{Int,Colon,Int,Colon,Int}}
    @test @inferred(getinnerdims((1,2,3,4,5), getsplitmode(Aes2))) == (2,4)
    @test @inferred(getouterdims((1,2,3,4,5), getsplitmode(Aes2))) == (3,1,5)

    # Non-involutive outer dimension order (a 3-cycle), regression test for
    # getouterdims applying the inverse of the slicemap permutation:
    Aes3 = eachslice(A_orig; dims = (2,3,1))
    @test @inferred(getsplitmode(Aes3)) isa BaseSlicing{2,3,Tuple{Int,Int,Int,Colon,Colon}}
    @test @inferred(getinnerdims((1,2,3,4,5), getsplitmode(Aes3))) == (4,5)
    @test @inferred(getouterdims((1,2,3,4,5), getsplitmode(Aes3))) == (2,3,1)

    Aec = eachcol(A_orig_mat)
    @test @inferred(getsplitmode(Aec)) isa BaseSlicing{1,1,Tuple{Colon,Int}}
    @test @inferred(getinnerdims((1,2), getsplitmode(Aec))) == (1,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aec))) == (2,)

    Aer = eachrow(A_orig_mat)
    @test @inferred(getsplitmode(Aer)) isa BaseSlicing{1,1,Tuple{Int,Colon}}

    # splitup validates the array rank against the slicemap:
    let smode = getsplitmode(eachcol(rand(3, 4)))
        @test_throws ArgumentError splitup(rand(3, 4, 5), smode)
        @test_throws ArgumentError splitup(rand(3), smode)
    end
    @test @inferred(getinnerdims((1,2), getsplitmode(Aer))) == (2,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aer))) == (1,)

    test_api(Aes1, A_orig)
    test_api(Aes2, A_orig)
    test_api(Aes3, A_orig)
    test_api(Aec, A_orig_mat)
    test_api(Aer, A_orig_mat)

    # vecflattened of memory-ordered slicings is a zero-copy view of the
    # underlying data, otherwise the elements are concatenated:
    vf_col = @inferred(vecflattened(Aec))
    @test vf_col == vec(A_orig_mat)
    vf_col[1] = 42
    @test A_orig_mat[1, 1] == 42
    @test vecflattened(Aes1) == vec(A_orig)
    @test vecflattened(Aer) == reduce(vcat, collect(Aer))
    @test vecflattened(Aes2) == mapreduce(vec, vcat, collect(Aes2))

    @testset "offset outer axes" begin
        # Slices of data with offset axes have offset outer axes, which
        # per-element operations must preserve:
        m = reshape(collect(1.0:12.0), 3, 4)
        om = view(m, :, Base.IdentityUnitRange(2:3))
        S = eachslice(om; dims = 2)
        @test axes(innersizes(S)) == axes(S)
        @test all(isequal((3,)), innersizes(S))
        @test axes(innerlengths(S)) == axes(S)
        @test all(isequal(3), innerlengths(S))
        @test splitup(fused(S), getsplitmode(S)) == S
    end

    @testset "eachslice with drop = false" begin
        # Singleton outer dimensions do not appear in the slicemap, so such
        # slicings have no representable split mode and must behave like
        # generic nested arrays instead of silently dropping dimensions:
        B = rand(2, 3, 4)
        S = eachslice(B; dims = 3, drop = false)
        @test @inferred(getsplitmode(S)) isa UnknownSplitMode
        @test_throws ArgumentError fused(S)
        @test_throws ArgumentError flatview(S)
        @test size(stacked(S)) == size(stack(S))
        @test splitup(stacked(S), unstackmode(S)) == S
        @test innersize(S) == (2, 3)
        @test innermap(x -> 2x, S) == map(x -> 2 .* x, S)
        @test deepmap(x -> 2x, S) == map(x -> 2 .* x, S)
        @test vecflattened(S) == mapreduce(vec, vcat, collect(S))
    end
end
