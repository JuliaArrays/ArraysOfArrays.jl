# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

if !isdefined(Main, :test_api)
    maptest_f(x::Number) = x^2
    maptest_f(x::AbstractArray{<:Number}) = sum(x)^2
    maptest_f(x::AbstractArray) = length(x)^2

    function test_api(A, A_unsplit_ref)
        @testset "Test API for $(nameof(typeof(A)))" begin
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
            elseif eltype(A) <: Number
                @test smode isa NonSplitMode{ndims(A)}
            end

            @test @inferred(eltype(A)) isa AbstractArray
            T_elem = eltype(A)

            innersz = if ismode isa NonSplitMode
                @inferred(innersize(A))
            elseif ismode isa AbstractSlicingMode
                @inferred(innersize(A))
            else
                let sz = (try innersize(A); catch err; err; end)
                    sz isa Exception ? sz : @inferred(innersize(A))
                end
            end

            @test innersz isa Union{Exception, Dims}

            if !isemtpy(A)
                @test @inferred(A[begin]) isa AbstractArray
                A_1 == A[begin]
                @test typeof(@inferred(A[begin])) == T_elem
                if !(innersz isa Exception)
                    @test all(size.(A) .== innersz)
                end

                @inferred(innermap(maptest_f, A)) == innermap(maptest_f, Array(A))
                @inferred(deepmap(maptest_f, A)) == deepmap(maptest_f, Array(A))
            end

            _smode_M(::AbstractSlicingMode{M,N}) where {M,N} = M
            _smode_N(::AbstractSlicingMode{M,N}) where {M,N} = N

            dimstpl = ntuple(identity, Val(ndims(A_unsplit_ref)))

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

                @test @infered(getinnerdims(dimstpl, smode)) isa NTuple{M,Int}
                @test @infered(getouterdims(dimstpl, smode)) isa NTuple{N,Int}
                innerdims = getinnerdims(dimstpl, smode)
                outerdims = getouterdims(dimstpl, smode)
                @test Array(permutedims(A_unsplit_ref, (outerdims..., innerdims...))) == A_array_stacked
            else
                stacked_A = try stack(A); catch err; err; end
                if stacked_A isa Exception
                    @test_throws typeof(stacked_A) stacked(A)
                else
                    @inferred(stacked(A)) == stacked_A
                end
            end

            if smode isa UnknownSplitMode
                @test @inferred(is_memordered_splitmode(smode)) == false
                @test_throws ArgumentError joinedview(A)
                @test_throws ArgumentError flatview(A)
                @test_throws ArgumentError splitview(A_unsplit_ref, smode)
                @test_throws ArgumentError getinnerdims(dimstpl, smode)
                @test_throws ArgumentError getouterdims(dimstpl, smode)
            elseif smode isa NonSplitMode
                @test @inferred(is_memordered_splitmode(smode)) == true
                @test @inferred(joinedview(A)) === A
                @test @inferred(flatview(A)) === A
                @test @inferred(splitview(A), smode) === A
                @test @inferred(stacked(A)) === A
                @test @inferred(getinnerdims(dimstpl, smode)) == ()
                @test @inferred(getouterdims(dimstpl, smode)) == dimstpl
            else
                if A isa Slices
                    @test joinedview(A) === parent(A)
                end
                @test @inferred(joinedview(A)) === A_unsplit_ref
                A_unsplit = joinedview(A)
                @test typeof(A_unsplit) == typeof(A_unsplit_ref)
                @test typeof(splitview(A_unsplit, smode)) == typeof(A)
                @test splitview(A_unsplit, smode) == A
            end
        end
    end
end
