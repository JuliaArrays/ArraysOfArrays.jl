# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    VectorOfArrays{T,N,M} <: AbstractVector{<:AbstractArray{T,N}}

An `VectorOfArrays` represents a vector of `N`-dimensional arrays (that may
differ in size). Internally, `VectorOfArrays` stores all elements of all
arrays in a single flat vector. `M` must equal `N - 1`

The `VectorOfArrays` itself supports `push!`, `unshift!`, etc., but the size
of each individual array in the vector is fixed. `resize!` is not supported,
as the size of all arrays in the vector must be defined. However, memory space
for up to `n` arrays with a maximum size `s` can be reserved via
`sizehint!(A::VectorOfArrays, n, s::Dims{N})`

Constructors:

```
VectorOfArrays{T, N}()

VectorOfArrays(
    data::AbstractVector,
    elem_ptr::AbstractVector{Int},
    kernel_size::AbstractVector{<:Dims}
    checks::Function = ArraysOfArrays.full_consistency_checks
)
```

Other suitable values for `checks` are `ArraysOfArrays.simple_consistency_checks`
and `ArraysOfArrays.no_consistency_checks`.

The following type aliases are defined:

* `VectorOfVectors{T,VT,VI,VD} = VectorOfArrays{T,1,VT,VI,VD}`
"""
struct VectorOfArrays{
    T, N, M,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    VD<:AbstractVector{Dims{M}}
} <: AbstractVector{Array{T,N}}
    data::VT
    elem_ptr::VI
    kernel_size::VD

    function VectorOfArrays{T,N}() where {T,N}
        M = length(Base.front(ntuple(_ -> 0, Val{N}())))
        data = Vector{T}()
        elem_ptr = [firstindex(data)]
        kernel_size = Vector{Dims{M}}()

        new{
            T, N, M,
            typeof(data),
            typeof(elem_ptr),
            typeof(kernel_size)
        }(data, elem_ptr, kernel_size)
    end

    function VectorOfArrays(
        data::VT,
        elem_ptr::VI,
        kernel_size::VD,
        checks::Function = full_consistency_checks
    ) where {
        T, M,
        VT<:AbstractVector{T},
        VI<:AbstractVector{Int},
        VD<:AbstractVector{Dims{M}}
    }
        N = length((ntuple(_ -> 0, Val{M}())..., 0))
        A = new{T,N,M,VT,VI,VD}(data, elem_ptr, kernel_size)
        checks(A)
        A
    end
end

export VectorOfArrays

function VectorOfArrays{T,N}(A::AbstractVector{AbstractArray{U,N}}) where {T,N,U}
    R = VectorOfArrays{T,N}()
    append!(R, A)
end

VectorOfArrays(A::AbstractVector{AbstractArray{T,N}}) where {T,N} = VectorOfArrays{T,N}(A)


@static if VERSION < v"0.7.0-DEV.3138"
    Base.convert(VA::Type{VectorOfArrays{T,N}}, A::AbstractVector{AbstractArray{U,N}}) where {T,N,U} = VA(A)
    Base.convert(VA::Type{VectorOfArrays}, A::AbstractVector{AbstractArray{T,N}}) where {T,N} = VA(A)
end


function full_consistency_checks(A::VectorOfArrays)
    simple_consistency_checks(A)
    all(eachindex(A.kernel_size)) do i
        len = A.elem_ptr[i+1] - A.elem_ptr[i]
        klen = prod(A.kernel_size[i])
        len > 0 && (klen == 1 || mod(len, klen) == 0)
    end || throw(ArgumentError("VectorOfArrays inconsistent: Content of elem_ptr and kernel_size is inconsistent"))
    nothing
end


function simple_consistency_checks(A::VectorOfArrays{T,N,M}) where {T,N,M}
    M == N - 1 || throw(ArgumentError("VectorOfArrays{T,N,M} inconsistent: M must equal N - 1"))
    firstindex(A.elem_ptr) == firstindex(A.kernel_size) || throw(ArgumentError("VectorOfArrays inconsistent: elem_ptr and kernel_size have incompatible indexing"))
    size(A.elem_ptr, 1) == size(A.kernel_size, 1) + 1 || throw(ArgumentError("VectorOfArrays inconsistent: elem_ptr and kernel_size have incompatible size"))
    first(A.elem_ptr) >= firstindex(A.data) || throw(ArgumentError("VectorOfArrays inconsistent: First elem_ptr inconsistent with data indices"))
    last(A.elem_ptr) - 1 <= lastindex(A.data) || throw(ArgumentError("VectorOfArrays inconsistent: Last elem_ptr inconsistent with data indices"))
    nothing
end


function no_consistency_checks(A::VectorOfArrays)
    nothing
end


Base.@propagate_inbounds function _elem_range_size(A::VectorOfArrays, i::Integer)
    elem_ptr = A.elem_ptr

    from = elem_ptr[i]
    until = elem_ptr[i+1]
    to = until - 1
    len = until - from

    ksize = A.kernel_size[i]
    klen = prod(ksize)
    sz_lastdim = div(len, klen)
    sz = (ksize..., sz_lastdim)

    (from:to, sz)
end


import Base.==
(==)(A::VectorOfArrays, B::VectorOfArrays) =
    A.data == B.data && A.elem_ptr == B.elem_ptr && A.kernel_size == B.kernel_size


Base.parent(A::VectorOfArrays) = A.data

Base.size(A::VectorOfArrays) = size(A.kernel_size)

Base.IndexStyle(A::ArrayOfSimilarArrays) = IndexLinear()


Base.@propagate_inbounds function Base.getindex(A::VectorOfArrays, i::Integer)
    @boundscheck checkbounds(A, i)
    r, s = _elem_range_size(A, i)
    dataview = view(A.data, r)
    Base.__reshape((dataview, IndexStyle(dataview)), s)
end


Base.@propagate_inbounds function Base._getindex(l::IndexStyle, A::VectorOfArrays, idxs::AbstractUnitRange{<:Integer})
    from = first(idxs)
    to = last(idxs)
    elem_ptr = A.elem_ptr[from:(to+1)]
    kernel_size = A.kernel_size[from:to]
    data = A.data[first(elem_ptr):(last(elem_ptr) - 1)]
    broadcast!(+, elem_ptr, elem_ptr, firstindex(data) - first(elem_ptr))
    VectorOfArrays(data, elem_ptr, kernel_size, no_consistency_checks)
end


Base.@propagate_inbounds function Base._getindex(l::IndexStyle, A::VectorOfArrays, idxs::AbstractVector{<:Integer})
    @boundscheck checkbounds(A, idxs)

    A_ep = A.elem_ptr
    A_data = A.data

    elem_ptr = similar(A_ep, length(eachindex(idxs)) + 1)
    delta_i = firstindex(elem_ptr) - firstindex(idxs)

    elem_ptr[firstindex(elem_ptr)] = firstindex(A_data)
    for i in eachindex(idxs)
        idx = idxs[i]
        l = A_ep[idx + 1] - A_ep[idx]
        elem_ptr[i + 1 + delta_i] = elem_ptr[i + delta_i] + l
    end

    data = similar(A_data, last(elem_ptr) - first(elem_ptr))
    if firstindex(data) != firstindex(A_data)
        @assert firstindex(data) != first(elem_ptr)
        broadcast!(+, elem_ptr, elem_ptr, firstindex(data) - first(elem_ptr))
    end

    for i in eachindex(idxs)
        idx = idxs[i]
        l = A_ep[idx + 1] - A_ep[idx]

        # Sanity check:
        @assert l == elem_ptr[i + 1 + delta_i] - elem_ptr[i + delta_i]

        copyto!(data, elem_ptr[i + delta_i], A_data, A_ep[idx], l)
    end

    kernel_size = A.kernel_size[idxs]

    VectorOfArrays(data, elem_ptr, kernel_size, no_consistency_checks)
end


Base.@propagate_inbounds function Base.setindex!(A::VectorOfArrays{T,N}, x::AbstractArray{U,N}, i::Integer) where {T,N,U}
    r, s = _view_reshape_spec(A, i)
    @boundscheck s == size(x) || throw(DimensionMismatch("Can't assign array to element $i of VectorOfArrays, array size is incompatible"))
    A.data[rng] = x
    A
end

Base.length(A::VectorOfArrays) = length(A.kernel_size)

@static if VERSION < v"0.7.0-beta.250"
    Base._length(A::VectorOfArrays) = Base._length(A.kernel_size)
end


function Base.append!(A::VectorOfArrays{T,N}, B::VectorOfArrays{U,N}) where {T,N,U}
    if !isempty(B)
        # Implementation supports A === B

        A_ep = A.elem_ptr
        B_ep = B.elem_ptr
        idxs_B = firstindex(B_ep):(lastindex(B_ep) - 1)
        delta_ep_idx = lastindex(A_ep) + 1 - firstindex(B_ep)
        delta_ep = last(A_ep) - first(B_ep)
        resize!(A_ep, length(eachindex(A_ep)) + length(idxs_B))
        @assert checkbounds(Bool, B_ep, idxs_B)
        @assert checkbounds(Bool, A_ep, broadcast(+, idxs_B, delta_ep_idx))
        @inbounds @simd for i_B in idxs_B
            A_ep[i_B + delta_ep_idx] = B_ep[i_B + 1] + delta_ep
        end

        append!(A.data, B.data)
        append!(A.kernel_size, B.kernel_size)

        simple_consistency_checks(A)
    end
    A
end


function Base.append!(A::VectorOfArrays{T,N}, B::AbstractVector{AbstractArray{U,N}}) where {T,N,U}
    if !isempty(B)
        n_A = length(eachindex(A))
        n_B = length(eachindex(B))
        datalen_A = length(eachindex(A.data))
        datalen_B = zero(Int)
        for i in eachindex(B)
            datalen_B += Int(length(eachindex(B[i])))
        end

        sizehint!(A.data, datalen_A + datalen_B)
        sizehint!(A.elem_ptr, n_A + n_B + 1)
        sizehint!(A.kernel_size, n_A + n_B)

        for i in eachindex(B)
            push!(A, B[i])
        end
    end
    A
end


Base.@propagate_inbounds function Base.unsafe_view(A::VectorOfArrays, idxs::AbstractUnitRange{<:Integer})
    from = first(idxs)
    to = last(idxs)
    VectorOfArrays(
        A.data,
        view(A.elem_ptr, from:(to+1)),
        view(A.kernel_size, from:to),
        no_consistency_checks
    )
end


function Base.sizehint!(A::VectorOfArrays{T,N}, n, s::Dims{N}) where {T,N}
    sizehint!(A.data, n * mul(s))
    sizehint!(A.elem_ptr, n + 1)
    sizehint!(A.kernel_size, n)
    A
end


function Base.push!(A::VectorOfArrays{T,N}, x::AbstractArray{U,N}) where {T,N,U}
    @assert last(A.elem_ptr) == lastindex(A.data) + 1
    append!(A.data, x)
    push!(A.elem_ptr, lastindex(A.data) + 1)
    push!(A.kernel_size, Base.front(size(x)))
    A
end


function UnsafeArrays.uview(A::VectorOfArrays)
    VectorOfArrays(
        uview(A.data),
        uview(A.elem_ptr),
        uview(A.kernel_size),
        no_consistency_checks
    )
end


function nestedmap2(f::Base.Callable, A::VectorOfArrays)
    new_data = map(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end


function deepmap(f::Base.Callable, A::VectorOfArrays)
    new_data = deepmap(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end



const VectorOfVectors{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    VD<:AbstractVector{Dims{0}}
} = VectorOfArrays{T,1,0,VT,VI,VD}

export VectorOfVectors

VectorOfVectors{T}() where {T} = VectorOfArrays{T,1}()

VectorOfVectors(
    data::AbstractVector,
    elem_ptr::AbstractVector{Int},
    checks::Function = consistency_checks
) = VectorOfArrays(
    data,
    elem_ptr,
    similar(A.elem_ptr, Dims{0}, size(elem_ptr, 1) - 1),
    checks
)
