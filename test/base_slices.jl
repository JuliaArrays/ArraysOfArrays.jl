# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: getinnerdims, getouterdims

@testset "base_slices" begin
    A_orig = rand(5,6,7,8,9)
    A_orig_mat = rand(5,6)

    Aes1 = eachslice(A_orig; dims = (4,5))
    @test getsplitmode(Aes1) isa BaseSlicing{3,2,Tuple{Colon,Colon,Colon,Int,Int}}
    @test @inferred(getinnerdims((1,2,3,4,5), getsplitmode(Aes1))) == (1,2,3)
    @test @inferred(getouterdims((1,2,3,4,5), getsplitmode(Aes1))) == (4,5)

    Aes2 = eachslice(A_orig; dims = (3,1,5))
    @test @inferred(getinnerdims((1,2,3,4,5), getsplitmode(Aes2))) == (2,4)
    @test @inferred(getouterdims((1,2,3,4,5), getsplitmode(Aes2))) == (3,1,5)

    Aec = eachcol(A_cols)
    @test @inferred(getinnerdims((1,2), getsplitmode(Aec))) == (1,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aec))) == (2,)

    Aer = eachrow(A_cols)
    @test @inferred(getinnerdims((1,2), getsplitmode(Aer))) == (2,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aer))) == (1,)

    A = Aes2
    A_unsplit_ref = A_orig
    smode = getsplitmode(A)
    @test @inferred(joinedview(A)) === A_unsplit_ref
    A_unsplit = joinedview(A)
    @test typeof(splitview(A_unsplit_ref, smode)) == typeof(A_unsplit)
    @test splitview(A_unsplit_ref, smode) == A_unsplit
end
