# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using ArraysOfArrays: getinnerdims, getouterdims

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

    Aec = eachcol(A_orig_mat)
    @test @inferred(getsplitmode(Aec)) isa BaseSlicing{1,1,Tuple{Colon,Int}}
    @test @inferred(getinnerdims((1,2), getsplitmode(Aec))) == (1,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aec))) == (2,)

    Aer = eachrow(A_orig_mat)
    @test @inferred(getsplitmode(Aer)) isa BaseSlicing{1,1,Tuple{Int,Colon}}
    @test @inferred(getinnerdims((1,2), getsplitmode(Aer))) == (2,)
    @test @inferred(getouterdims((1,2), getsplitmode(Aer))) == (1,)

    A = Aes2
    A_unsplit_ref = A_orig
    f = x -> x^2

    @test Array(A) isa Array{<:Any,ndims(A)}
    A_array = Array(A)
    @test A == A_array
    @test isequal(A, A_array)

    @test @infered(getsplitmode(A)) isa AbstractSplitMode
    smode = getsplitmode(A)
    if A isa AbstractSlices
        let M = ndims(eltype(A)), N = ndims(A)
            @test smode isa AbstractSlicingMode{M,N}
        end
    end

    @test @inferred(eltype(A)) isa AbstractArray
    T_elem = eltype(A)
    if !isemtpy(A)
        @test @inferred(A[begin]) isa AbstractArray
        A_1 == A[begin]
        @test typeof(@inferred(A[begin])) == T_elem
        @test @inferred(innersize(A)) == size(A_1)
    end

    @inferred(innermap(f, A)) == innermap(f, Array(A))
    @inferred(deepmap(f, A)) == deepmap(f, Array(A))

    _smode_M(::AbstractSlicingMode{M,N}) where {M,N} = M
    _smode_N(::AbstractSlicingMode{M,N}) where {M,N} = N

    if smodes isa AbstractSlicingMode
        M, N = _smode_M(smode), _smode_N(smode)
        A_array_stacked = stack(A_array)
        @test M == ndims(eltype(A))
        @test N == ndims(A)

        @test Array(stack(A)) == A_array_stacked

        if is_memordered_splitmode(smode)
            if A isa Slices
                # stack(A) never returns parent for Slices, even if possible:
                @test @inferred(stack(A)) == A_unsplit_ref
            else
                @test @inferred(stack(A)) === A_unsplit_ref
            end
            @test @inferred(stacked(A)) === A_unsplit_ref
            @test @inferred(flatview(A)) === A_unsplit_ref
        else
            @test Array(@inferred(stack(A))) == A_array_stacked
            @test Array(@inferred(stacked(A))) == A_array_stacked
            @test_throws ArgumentError flatview(A)
        end

        let dimstpl = ntuple(identity, Val(ndims(A_unsplit_ref)))
            @test @infered(getinnerdims(dimstpl, smode)) isa NTuple{M,Int}
            @test @infered(getouterdims(dimstpl, smode)) isa NTuple{N,Int}
            innerdims = getinnerdims(dimstpl, smode)
            outerdims = getouterdims(dimstpl, smode)
            @test Array(permutedims(A_unsplit_ref, (outerdims..., innerdims...))) == A_array_stacked
        end
    end

    if smode isa UnknownSplitMode
        @test_throws ArgumentError joinedview(A)
        @test_throws ArgumentError flatview(A)
        @test_throws ArgumentError splitview(A_unsplit_ref, smode)
    else
        if A isa Slices
            @test joinedview(A) === parent(A)
        end
        @test @inferred(joinedview(A)) === A_unsplit_ref
        A_unsplit = joinedview(A)
        @test typeof(splitview(A_unsplit_ref, smode)) == typeof(A)
        @test splitview(A_unsplit_ref, smode) == A
    end
end
