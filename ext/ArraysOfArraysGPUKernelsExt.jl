# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

module ArraysOfArraysGPUKernelsExt

using GPUArraysCore: AbstractGPUArray

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

using ArraysOfArrays: ArraysOfArrays, VectorOfArrays, innerlengths


@kernel function _segmented_mapreduce_kernel!(out, f, op, @Const(data), @Const(elem_ptr), init)
    i = @index(Global, Linear)
    j0 = elem_ptr[i]
    j1 = elem_ptr[i + 1] - 1
    if init isa ArraysOfArrays._NoInit
        # Empty element arrays have been excluded beforehand:
        acc = f(data[j0])
        for j in (j0 + 1):j1
            acc = op(acc, f(data[j]))
        end
    else
        acc = init
        for j in j0:j1
            acc = op(acc, f(data[j]))
        end
    end
    out[i] = acc
end


function _on_backend(backend, x::AbstractArray)
    KA.get_backend(x) == backend && return x
    y = KA.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end


# Single-pass segmented reduction for GPU-resident data:
function ArraysOfArrays._innermapreduce_impl(f, op, init, A::VectorOfArrays{T,N,M,<:AbstractGPUArray}) where {T,N,M}
    data = A.data
    backend = KA.get_backend(data)
    elem_ptr = _on_backend(backend, A.elem_ptr)
    n = length(A)

    if init isa ArraysOfArrays._NoInit && any(iszero, innerlengths(A))
        throw(ArgumentError("Reducing over empty element arrays requires an init value"))
    end

    T_f = Base.promote_op(f, T)
    T_init = init isa ArraysOfArrays._NoInit ? T_f : typeof(init)
    T_out = promote_type(T_f, Base.promote_op(op, T_init, T_f))
    out = KA.allocate(backend, T_out, n)

    if n > 0
        # Stream-ordered, like GPU array operations no synchronization here:
        kernel! = _segmented_mapreduce_kernel!(backend)
        kernel!(out, f, op, data, elem_ptr, init; ndrange = n)
    end
    return out
end

end # module ArraysOfArraysGPUKernelsExt
