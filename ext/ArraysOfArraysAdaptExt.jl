# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysAdaptExt

import Adapt
using Adapt: adapt

using ArraysOfArrays: ArrayOfSimilarArrays, VectorOfArrays
using ArraysOfArrays: no_consistency_checks


function Adapt.adapt_structure(to, A::ArrayOfSimilarArrays{T,M,N}) where {T,M,N}
    adapted_data = adapt(to, A.data)
    ArrayOfSimilarArrays{eltype(adapted_data),M,N}(adapted_data)
end


function Adapt.adapt_structure(to, A::VectorOfArrays)
    VectorOfArrays(
        adapt(to, A.data),
        adapt(to, A.elem_ptr),
        adapt(to, A.kernel_size),
        no_consistency_checks
    )
end


end # module ArraysOfArraysAdaptExt
