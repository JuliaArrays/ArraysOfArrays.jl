# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Statistics
using Test

using Adapt

using ArraysOfArrays: full_consistency_checks, append_elemptr!, element_ptr


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


    @testset "mapreduce maximum/minimum shortcut" begin
        A1 = ref_AoA3(Float32, 3); A2 = ref_AoA3(Float32, 0)
        A3 = ref_AoA3(Float32, 4); A4 = ref_AoA3(Float64, 2)

        B1 = VectorOfArrays(A1); B2 = VectorOfArrays(A2);
        B3 = VectorOfArrays(A3); B4 = VectorOfArrays(A4);

        @testset "maximum - correctness" begin
            @test mapreduce(maximum, max, B1) == mapreduce(maximum, max, Array(B1))
            @test mapreduce(maximum, max, B2; init=Float32(0.)) == mapreduce(maximum, max, Array(B2); init=Float32(0.))
            @test mapreduce(maximum, max, B3) == mapreduce(maximum, max, Array(B3))
            @test mapreduce(maximum, max, B4) == mapreduce(maximum, max, Array(B4))
        end

        @testset "maximum - performance" begin
            B1_naive = Array(B1)
            mapreduce(maximum, max, B1_naive)
            @test (@allocated mapreduce(maximum, max, B1)) <= (@allocated mapreduce(maximum, max, B1_naive))
        end

        @testset "minimum - correctness" begin
            @test mapreduce(minimum, min, B1) == mapreduce(minimum, min, Array(B1))
            @test mapreduce(minimum, min, B2; init=Float32(0.)) == mapreduce(minimum, min, Array(B2); init=Float32(0.))
            @test mapreduce(minimum, min, B3) == mapreduce(minimum, min, Array(B3))
            @test mapreduce(minimum, min, B4) == mapreduce(minimum, min, Array(B4))
        end

        @testset "minimum - performance" begin
            B1_naive = Array(B1)
            mapreduce(minimum, min, B1_naive)
            @test (@allocated mapreduce(minimum, min, B1)) <= (@allocated mapreduce(minimum, min, B1_naive))
        end
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

        B1_copy = @inferred(copy(B1)); B3_copy = @inferred(copy(B3))
        append!(B1_copy, B3_copy)
        @test B1_copy.data == vcat(B1.data, B3.data)
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

        VV = @inferred(VectorOfVectors{Float64}())
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
    end


    @testset "examples" begin
        VA = @inferred(VectorOfArrays{Float64, 2}())

        @inferred(push!(VA, rand(2, 3)))
        @inferred(push!(VA, rand(4, 2)))

        @test @inferred(size(VA[1]) == (2,3))
        @test @inferred(size(VA[2]) == (4,2))

        # -------------------------------------------------------------------

        VV = @inferred(VectorOfVectors{Float64}())
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
        @test first.(@inferred(VectorOfVectors(A, elem_ptr))) == [1, 2, 3, 2]
        @test first.(@inferred(VectorOfVectors(A, elem_ptr32))) == [1, 2, 3, 2]

        B = [1, 2, 3, 4, 5, 6, 7, 8]
        B_grouped_ref = [[1, 2], [3], [4, 5], [6, 7, 8]]
        @test @inferred(VectorOfVectors(B, elem_ptr)) == B_grouped_ref
        @test @inferred(VectorOfVectors(B, elem_ptr32)) == B_grouped_ref

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
