# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using ArraysOfArrays
using Test

import Aqua
using StructArrays
using ArraysOfArrays: NestedArrayStyle
using Base: Broadcast

include("waveform_defs.jl")

@testset "StructArrays extension" begin
    n = 5
    timedata = fill(0.0:0.1:12.7, n)
    stats(wf) = (mean = sum(wf.signal) / length(wf.signal), t_last = last(wf.time))

    for signal in (VectorOfSimilarVectors(rand(128, n)), VectorOfVectors([rand(128) for _ in 1:n]))
        A = StructArray{Waveform{eltype(timedata),eltype(signal)}}((timedata, signal))
        A_ref = collect(A)
        @test Base.BroadcastStyle(typeof(A)) isa StructArrays.StructArrayStyle{NestedArrayStyle{1},1}

        # Struct-valued results become a StructArray:
        S = @inferred broadcast(stats, A)
        @test S isa StructArray
        @test S.mean == [stats(wf).mean for wf in A_ref]
        @test S.t_last == [stats(wf).t_last for wf in A_ref]
        W = (wf -> Waveform(wf.time, 2 .* wf.signal)).(A)
        @test W isa StructArray{<:Waveform}
        @test W.signal == [2 .* wf.signal for wf in A_ref]

        # Array-valued results are packed into a nested array, other results
        # stay in a plain Vector, like for broadcasts over nested arrays:
        R = @inferred broadcast(wf -> 2 .* wf.signal, A)
        @test R isa VectorOfArrays
        @test R == [2 .* wf.signal for wf in A_ref]
        R2 = @inferred broadcast((wf, x) -> wf.signal .+ x, A, signal)
        @test R2 isa VectorOfArrays
        @test R2 == [wf.signal .+ x for (wf, x) in zip(A_ref, signal)]
        L = @inferred broadcast(wf -> length(wf.signal), A)
        @test L isa Vector{Int}
        @test L == fill(128, n)
        M = (wf -> wf.signal .> 0.5).(A)
        @test M isa Vector{BitVector}
        @test M == [wf.signal .> 0.5 for wf in A_ref]
        @test (wf -> typeof(wf.signal)).(A) == [typeof(wf.signal) for wf in A_ref]

        # In-place broadcasts:
        D = similar(S)
        D .= stats.(A)
        @test D.mean == S.mean && D.t_last == S.t_last
        V = zeros(Int, n)
        V .= (wf -> length(wf.signal)).(A)
        @test V == fill(128, n)

        # Empty input and non-concrete results:
        @test stats.(A[1:0]) isa StructArray
        @test isempty(stats.(A[1:0]))
        f_branchy(wf) = length(wf.signal) > 0 ? (a = 1,) : (b = 2.0,)
        @test f_branchy.(A) == [f_branchy(wf) for wf in A_ref]
    end

    # The specialized getindex is used, the elements are not rebuilt from
    # the columns as the less specific declared element type:
    signal = VectorOfSimilarVectors(rand(128, n))
    B = StructArray{Waveform{eltype(timedata),Vector{Float64}}}((timedata, signal))
    @test !(typeof(B[1]) <: eltype(B))
    @test stats.(B) isa StructArray
    @test stats.(B).mean == [stats(B[i]).mean for i in eachindex(B)]
    @test all((wf -> wf.signal isa SubArray).(B))
    @test (wf -> 2 .* wf.signal).(B) isa VectorOfArrays
    # Struct-valued results are stored according to the element type inferred
    # from the declared element type of B, like for any broadcast, so they
    # must be convertible to it. Results with fresh arrays are:
    W2 = (wf -> Waveform(wf.time, 2 .* wf.signal)).(B)
    @test W2 isa StructArray && eltype(W2) == eltype(B)
    @test W2.signal == [2 .* B[i].signal for i in eachindex(B)]
    # identity.(B) copies the columns (a shortcut of Base for identity):
    I = identity.(B)
    @test I isa StructArray && eltype(I) == eltype(B)
    @test I.signal == collect(signal)
    # The specialized elements themselves are not convertible to the
    # declared element type, so returning them fails, as collect(B) does:
    @test_throws MethodError (wf -> wf).(B)
    @test_throws MethodError collect(B)

    # Broadcasts with the StructArray style over column arguments instead of
    # a StructArray, as constructed by PropertyFunctions:
    a = VectorOfVectors([rand(3), rand(2), rand(4)])
    b = [1.0, 2.0, 3.0]
    style = StructArrays.StructArrayStyle{NestedArrayStyle{1},1}()
    r_struct = Broadcast.materialize(Broadcast.broadcasted(style, (x, y) -> (s = sum(x), y = y), a, b))
    @test r_struct isa StructArray
    @test r_struct.s == sum.(a)
    @test r_struct.y == b
    r_array = Broadcast.materialize(Broadcast.broadcasted(style, (x, y) -> x .+ y, a, b))
    @test r_array isa VectorOfArrays
    @test r_array == [x .+ y for (x, y) in zip(a, b)]

    @testset "no method ambiguities or piracy in the StructArrays extension" begin
        ext = Base.get_extension(ArraysOfArrays, :ArraysOfArraysStructArraysExt)
        @test ext !== nothing
        @test isempty(detect_ambiguities(ext))
        Aqua.test_piracies(ext, treat_as_own = [ArraysOfArrays.AbstractNestedArrayStyle, ArraysOfArrays._block_length, ArraysOfArrays._read_block])
    end
end
