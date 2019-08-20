# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

using UnsafeArrays

using ArraysOfArrays: full_consistency_checks, append_elemptr!


@testset "vector_of_arrays" begin
    ref_AoA1(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 5)][1:0] : [rand(T, rand(1:9)) for i in 1:n]
    ref_AoA2(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 4, 2)][1:0] : [rand(T, rand(1:4), rand(1:4)) for i in 1:n]
    ref_AoA3(T::Type, n::Integer) = n == 0 ? [Array{T}(undef, 3, 2, 4)][1:0] : [rand(T, rand(1:3), rand(1:3), rand(1:3)) for i in 1:n]

    @testset "element pointer handling" begin
        A = [2, 4, 5, 9]; B = [3, 6, 8]
        @test @inferred(append_elemptr!(deepcopy(A), B)) == [2, 4, 5, 9, 12, 14]
        @test @inferred(append_elemptr!([2], [5])) == [2]
    end


    @testset "ctors" begin
        A1 = ref_AoA1(Float32, 5)
        @test @inferred(VectorOfArrays(deepcopy(A1))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays(deepcopy(A1)) == A1
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64}(deepcopy(A1)) == A1
        @test @inferred(VectorOfArrays{Float64,1}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64,1}(deepcopy(A1)) == A1

        @test @inferred(VectorOfVectors(deepcopy(A1))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfVectors(deepcopy(A1)) == A1
        @test @inferred(VectorOfVectors{Float64}(deepcopy(A1))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfVectors{Float64}(deepcopy(A1)) == A1

        A1_empty = ref_AoA1(Float32, 0)
        @test @inferred(VectorOfArrays(deepcopy(A1_empty))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays(deepcopy(A1_empty)) == A1_empty
        @test @inferred(VectorOfArrays{Float64}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64}(deepcopy(A1_empty)) == A1_empty
        @test @inferred(VectorOfArrays{Float64,1}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfArrays{Float64,1}(deepcopy(A1_empty)) == A1_empty

        @test @inferred(VectorOfVectors(deepcopy(A1_empty))) isa VectorOfArrays{Float32,1,0,Array{Float32,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfVectors(deepcopy(A1_empty)) == A1_empty
        @test @inferred(VectorOfVectors{Float64}(deepcopy(A1_empty))) isa VectorOfArrays{Float64,1,0,Array{Float64,1},Array{Int,1},Array{Tuple{},1}}
        @test VectorOfVectors{Float64}(deepcopy(A1_empty)) == A1_empty

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

        @test @inferred(vcat(B1)) === B1
        full_consistency_checks(vcat(B1))

        @test @inferred(vcat(B1, B2)) isa VectorOfArrays
        @test vcat(B1, B2) == vcat(A1, A2)
        @test eltype(vcat(B1, B2)) == Array{Float32,3}
        full_consistency_checks(vcat(B1, B2))

        @test @inferred(vcat(B1, B3)) isa VectorOfArrays
        @test vcat(B1, B3) == vcat(A1, A3)
        @test eltype(vcat(B1, B3)) == Array{Float32,3}
        full_consistency_checks(vcat(B1, B3))

        @test @inferred(vcat(B1, B2, B3, B4)) isa VectorOfArrays
        @test vcat(B1, B2, B3, B4) == vcat(A1, A2, A3, A4)
        @test eltype(vcat(B1, B2, B3, B4)) == Array{Float64,3}
        full_consistency_checks(vcat(B1, B2, B3, B4))
    end


    @testset "copy" begin
        A = ref_AoA3(Float32, 3);
        B = VectorOfArrays(A);

        @test typeof(@inferred copy(A)) == typeof(A)
        @test copy(A) == A
        @test copy(A) == B
    end


    @testset "examples" begin
        VA = VectorOfArrays{Float64, 2}()

        push!(VA, rand(2, 3))
        push!(VA, rand(4, 2))

        @test size(VA[1]) == (2,3)
        @test size(VA[2]) == (4,2)

        # -------------------------------------------------------------------


        VA_flat = flatview(VA)
        @test VA_flat isa Vector{Float64}

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

        using UnsafeArrays

        A = nestedview(rand(2,3,4,5), 2)

        @test isbits(A[2,2]) == false

        @test @uviews A begin
            isbits(A[2,2]) == true
        end

        # -------------------------------------------------------------------

        A = [1, 1, 2, 3, 3, 2, 2, 2]
        A_grouped_ref = [[1, 1], [2], [3, 3], [2, 2, 2]]
        elem_ptr = consgrouped_ptrs(A)
        @test first.(@inferred(VectorOfVectors(A, elem_ptr))) == [1, 2, 3, 2]

        B = [1, 2, 3, 4, 5, 6, 7, 8]
        B_grouped_ref = [[1, 2], [3], [4, 5], [6, 7, 8]]
        @test @inferred(VectorOfVectors(B, elem_ptr)) == B_grouped_ref

        C = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        C_grouped_ref = [[1.1, 2.2], [3.3], [4.4, 5.5], [6.6, 7.7, 8.8]]

        @test @inferred(consgroupedview(A, B)) isa VectorOfVectors
        @test consgroupedview(A, B) == B_grouped_ref

        @test @inferred(consgroupedview(A, (B, C))) isa NTuple{2, VectorOfVectors}
        @test consgroupedview(A, (B, C)) == (B_grouped_ref, C_grouped_ref)

        data = (a = A, b = B, c = C)
        @test @inferred(consgroupedview(A, data)) isa NamedTuple{(:a, :b, :c),<:NTuple{3,AbstractVector}}
        result = consgroupedview(A, data)
        @test result == (a = A_grouped_ref, b = B_grouped_ref, c = C_grouped_ref)
        @test flatview(result.a) === data.a
        @test flatview(result.b) === data.b
        @test flatview(result.c) === data.c
    end
end
