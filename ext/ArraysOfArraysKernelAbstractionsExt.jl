# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysKernelAbstractionsExt

import KernelAbstractions as KA

using ArraysOfArrays: ArrayOfSimilarArrays, VectorOfArrays

# Kernels over nested arrays run on the backend of the underlying data. The
# shape information of a VectorOfArrays may deliberately live on the host
# while its data lives on a device, so no cross-field consistency is
# required here:
KA.get_backend(A::ArrayOfSimilarArrays) = KA.get_backend(A.data)
KA.get_backend(A::VectorOfArrays) = KA.get_backend(A.data)

end # module ArraysOfArraysKernelAbstractionsExt
