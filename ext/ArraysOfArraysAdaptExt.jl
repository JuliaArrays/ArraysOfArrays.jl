# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysAdaptExt

import Adapt
using Adapt: adapt

using ArraysOfArrays: ArrayOfSimilarArrays, VectorOfArrays, SplitParts, FuseArrays
using ArraysOfArrays: no_consistency_checks


function Adapt.adapt_structure(to, A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    adapted_data = adapt(to, A.data)
    ArrayOfSimilarArrays{eltype(adapted_data),M,N}(adapted_data)
end

# Declares ArrayOfSimilarArrays a wrapped array, so that packages like
# Reactant can convert it by rebuilding it around a converted parent.
# Old Adapt versions do not provide parent_type:
@static if isdefined(Adapt, :parent_type)
    Adapt.parent_type(::Type{ArrayOfSimilarArrays{T,M,N,P,ET}}) where {T,M,N,P,ET} = P
end


function Adapt.adapt_structure(to, A::VectorOfArrays)
    VectorOfArrays(
        adapt(to, A.data),
        adapt(to, A.elem_ptr),
        adapt(to, A.kernel_size),
        no_consistency_checks
    )
end


function Adapt.adapt_structure(to, smode::SplitParts)
    SplitParts(
        adapt(to, smode.elem_ptr),
        adapt(to, smode.kernel_size)
    )
end


Adapt.adapt_structure(to, f::FuseArrays) = FuseArrays(adapt(to, f.smode))


end # module ArraysOfArraysAdaptExt
