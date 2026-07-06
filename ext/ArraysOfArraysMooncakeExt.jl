# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysMooncakeExt

using Mooncake: DefaultCtx, @zero_derivative

using ArraysOfArrays: ArraysOfArrays, AbstractSplitMode,
    getsplitmode, unstackmode, is_memordered_splitmode, innersize, getslicemap,
    consgrouped_ptrs

# Mooncake can differentiate through the ArraysOfArrays types and operations
# without custom rules, so only the functions that query shape/structure
# information (and so are not differentiable with respect to array contents)
# get explicit zero-derivative rules:

@zero_derivative DefaultCtx Tuple{typeof(getsplitmode), AbstractArray}
@zero_derivative DefaultCtx Tuple{typeof(unstackmode), AbstractArray}
@zero_derivative DefaultCtx Tuple{typeof(is_memordered_splitmode), AbstractSplitMode}
@zero_derivative DefaultCtx Tuple{typeof(innersize), AbstractArray}
@zero_derivative DefaultCtx Tuple{typeof(innersize), AbstractArray, Integer}
@zero_derivative DefaultCtx Tuple{typeof(getslicemap), AbstractArray}
@zero_derivative DefaultCtx Tuple{typeof(ArraysOfArrays.getinnerdims), Tuple{Vararg{Integer}}, AbstractSplitMode}
@zero_derivative DefaultCtx Tuple{typeof(ArraysOfArrays.getouterdims), Tuple{Vararg{Integer}}, AbstractSplitMode}
@zero_derivative DefaultCtx Tuple{typeof(consgrouped_ptrs), AbstractVector}

end # module ArraysOfArraysMooncakeExt
