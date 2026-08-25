# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ArraysOfArrays" begin
    include("test_aqua.jl")
    include("functions.jl")
    include("base_slices.jl")
    include("array_of_similar_arrays.jl")
    include("vector_of_arrays.jl")
    include("broadcasting.jl")
    include("static_arrays.jl")
    include("fixed_size_arrays.jl")
    include("gpu_arrays.jl")
    include("mooncake.jl")
    # Reactant only supports 64-bit Linux and macOS, and some of its
    # dependencies break already during precompilation on other platforms,
    # so it can't be a static test dependency:
    if Sys.WORD_SIZE == 64 && (Sys.islinux() || Sys.isapple()) && isempty(VERSION.prerelease)
        import Pkg
        Base.identify_package("Reactant") === nothing && Pkg.add("Reactant")
        include("reactant.jl")
    end
    include("inverse_functions.jl")
    include("test_docs.jl")
end # testset
