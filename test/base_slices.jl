# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

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
    @test @inferred(getinnerdims((1,2), getsplitmode(Aer))) == (2,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aer))) == (1,)

    test_api(Aes1, Array(Aes1), A_orig)
    test_api(Aes2, Array(Aes2), A_orig)
    test_api(Aes3, Array(Aes3), A_orig)
    test_api(Aec, Array(Aec), A_orig_mat)
    test_api(Aer, Array(Aer), A_orig_mat)

    # vecflattened of memory-ordered slicings is a zero-copy view of the
    # underlying data, otherwise the elements are concatenated:
    vf_col = @inferred(vecflattened(Aec))
    @test vf_col == vec(A_orig_mat)
    vf_col[1] = 42
    @test A_orig_mat[1, 1] == 42
    @test vecflattened(Aes1) == vec(A_orig)
    @test vecflattened(Aer) == reduce(vcat, collect(Aer))
    @test vecflattened(Aes2) == mapreduce(vec, vcat, collect(Aes2))
end
