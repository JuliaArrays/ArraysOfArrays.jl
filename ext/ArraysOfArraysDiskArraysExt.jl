# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysDiskArraysExt

using DiskArrays: AbstractDiskArray

import ArraysOfArrays

# Every element access on disk-backed data is a separate disk read, so
# non-scalar getindex of nested views of such data reads the whole
# selection from the flat data in a single block read instead:
ArraysOfArrays._prefers_flat_getindex(::Type{<:AbstractDiskArray}) = true

# Likewise, stacking a disk-backed VectorOfArrays reads the covered data in
# a single ranged getindex (disk arrays also do not support the reshape of
# a lazy view that the in-memory implementation uses):
ArraysOfArrays._covered_data(data::AbstractDiskArray{<:Any,1}, r::AbstractUnitRange{Int}) = data[r]

end # module ArraysOfArraysDiskArraysExt
