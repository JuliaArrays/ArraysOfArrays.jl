# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using StaticArrays
using OffsetArrays: OffsetArray
using ArraysOfArrays: getinnerdims, getouterdims

struct _TestPoint2{T} <: FieldVector{2,T}
    x::T
    y::T
end
using InverseFunctions: inverse


@testset "StaticArray Extension" begin
    @testset "flatview and sliced" begin
        A = [(@SArray randn(3, 2, 4)) for i in 1:2, j in 1:2]
        @test @inferred(sliced(flatview(A), Val(3))) == A

        B = rand(3, 2, 4)
        @test @inferred(sliced(flatview(B), SVector{3})) == @inferred(sliced(B, Val(1)))
        @test_deprecated nestedview(flatview(B), SVector{3})
    end

    @testset "StaticSlices" begin
        X = rand(3, 5)
        # StaticSlices is about shape, an element type in the mode is
        # ignored, the element type of the flat array wins:
        for SA in (SVector{3}, SVector{3,Float64}, SVector{3,Float32})
            B = @inferred(splitup(X, StaticSlices(SA)))
            @test eltype(B) === SVector{3,Float64}
            @test B == sliced(X, Val(1))
            @test @inferred(fused(B)) == X
            @test @inferred(getsplitmode(B)) === StaticSlices(SVector{3})
            @test is_memordered_splitmode(getsplitmode(B))
            @test splitup(fused(B), getsplitmode(B)) == B
        end

        Y = rand(2, 3, 5)
        C = @inferred(splitup(Y, StaticSlices(SMatrix{2,3})))
        @test eltype(C) === SMatrix{2,3,Float64,6}
        @test C == sliced(Y, Val(2))
        @test @inferred(fused(C)) == Y
        @test splitup(fused(C), getsplitmode(C)) == C

        @test sliced(Y, SMatrix{2,3}) == sliced(Y, Val(2))
        @test eltype(sliced(Y, SMatrix{2,3})) === SMatrix{2,3,Float64,6}
        @test_throws DimensionMismatch sliced(rand(4, 5), SVector{3})

        A = [SVector{3}(rand(3)) for _ in 1:5]
        @test @inferred(getsplitmode(A)) === StaticSlices(SVector{3})
        @test @inferred(fused(A)) == flatview(A)
        @test splitup(fused(A), getsplitmode(A)) == A
        @test @inferred(unstackmode(A)) === getsplitmode(A)
        @test @inferred(stacked(A)) == flatview(A)
        @test splitup(stacked(A), unstackmode(A)) == A

        M = [SVector{2}(rand(2)) for i in 1:3, j in 1:4]
        @test getsplitmode(M) === StaticSlices(SVector{2})
        @test size(fused(M)) == (2, 3, 4)
        @test splitup(fused(M), getsplitmode(M)) == M

        # Element-type-free modes can be reused across arrays that differ
        # only in their element type:
        A32 = [SVector{3}(rand(Float32, 3)) for _ in 1:5]
        @test getsplitmode(A32) === getsplitmode(A)
        @test eltype(splitup(rand(Float32, 3, 2), getsplitmode(A))) === SVector{3,Float32}
        @test getsplitmode([@SMatrix rand(2, 2)]) === StaticSlices(SMatrix{2,2})

        # Higher-dimensional inner arrays via the SArray{S} spelling:
        Z = rand(2, 2, 2, 4)
        B3 = @inferred(splitup(Z, StaticSlices(SArray{Tuple{2,2,2}})))
        @test eltype(B3) === SArray{Tuple{2,2,2},Float64,3,8}
        @test fused(B3) == Z
        @test getsplitmode(collect(B3)) === StaticSlices(SArray{Tuple{2,2,2}})

        # Static array types outside the SArray family, like FieldVector
        # subtypes, work via the generic element-type fallback and keep
        # their type:
        @test eltype(splitup(rand(2, 5), StaticSlices(_TestPoint2))) === _TestPoint2{Float64}
        @test getsplitmode([_TestPoint2(1.0, 2.0)]) === StaticSlices(_TestPoint2{Float64})
        @test splitup(rand(2, 5), StaticSlices(_TestPoint2)) isa AbstractVector{_TestPoint2{Float64}}

        # The inner/outer dims API of StaticSlices:
        @test getinnerdims((1, 2, 3), StaticSlices(SVector{3})) == (1,)
        @test getouterdims((1, 2, 3), StaticSlices(SVector{3})) == (2, 3)

        # Mutable static arrays cannot be reinterpreted, nested arrays of
        # them get the generic nested-array treatment:
        @test getsplitmode([MVector{2}(rand(2))]) isa UnknownSplitMode
        @test unstackmode([MVector{2}(rand(2))]) === SplitSlices{1}()
        @test_throws ArgumentError splitup(rand(2, 3), StaticSlices(MVector{2}))

        # The same holds for zero-size static arrays, which reinterpret
        # rejects as singleton types:
        Z0 = [SVector{0,Float64}(), SVector{0,Float64}()]
        @test getsplitmode(Z0) isa UnknownSplitMode
        @test unstackmode(Z0) === SplitSlices{1}()
        @test splitup(stacked(Z0), unstackmode(Z0)) == Z0
        @test_throws ArgumentError splitup(zeros(0, 3), StaticSlices(SVector{0}))

        # Offset outer axes are rejected, offset inner axes are allowed:
        Bo = OffsetArray(reshape(collect(1.0:6.0), 2, 3), 1:2, 0:2)
        @test_throws ArgumentError splitup(Bo, StaticSlices(SVector{2}))
        Bi = OffsetArray(reshape(collect(1.0:6.0), 2, 3), 0:1, 1:3)
        @test collect(splitup(Bi, StaticSlices(SVector{2}))) == [SVector{2}(Bi[:, j]) for j in 1:3]
    end

    @testset "StaticSlices inverse" begin
        smode = StaticSlices(SVector{3})
        x = rand(3, 5)
        f = inverse(smode)
        @test f(smode(x)) == x
        @test inverse(f) === smode
        @test_throws ArgumentError f(x)
        @test_throws ArgumentError f([SVector{2}(rand(2)) for _ in 1:5])
    end

    @testset "outer broadcasts with static array results" begin
        # Static array results are not packed into a VectorOfArrays:
        A = sliced(rand(3, 4))
        r = (x -> SVector{3}(x)).(A)
        @test r isa Vector{<:SVector{3}}
        @test r == [SVector{3}(x) for x in collect(A)]
    end
end
