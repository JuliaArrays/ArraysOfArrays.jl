# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    VectorOfArrays{T,N,M} <: AbstractVector{<:AbstractArray{T,N}}

An `VectorOfArrays` represents a vector of `N`-dimensional arrays (that may
differ in size). Internally, `VectorOfArrays` stores all elements of all
arrays in a single flat vector. `M` must equal `N - 1`

The `VectorOfArrays` itself supports `push!`, `unshift!`, etc., but the size
of each individual array in the vector is fixed. `resize!` can be used to
shrink, but not to grow, as the size of the additional element arrays in the
vector would be unknown. However, memory space for up to `n` arrays with a
maximum size `s` can be reserved via
`sizehint!(A::VectorOfArrays, n, s::Dims{N})`.

Constructors:

```julia
VectorOfArrays{T,N}()

VectorOfArrays(A::AbstractVector{<:AbstractArray})
VectorOfArrays{T}(A::AbstractVector{<:AbstractArray})
VectorOfArrays{T,N}(A::AbstractVector{<:AbstractArray})

VectorOfArrays(
    data::AbstractVector,
    elem_ptr::AbstractVector{Int},
    kernel_size::AbstractVector{<:Dims}
    checks::Function = ArraysOfArrays.full_consistency_checks
)
```

Other suitable values for `checks` are `ArraysOfArrays.simple_consistency_checks`
and `ArraysOfArrays.no_consistency_checks`.

`VectorOfVectors` is defined as an type alias:

```julia
`VectorOfVectors{T,VT,VI,VD} = VectorOfArrays{T,1,VT,VI,VD}`
```
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

function VectorOfArrays{T,N}(A::AbstractVector{<:AbstractArray{U,N}}) where {T,N,U}
    R = VectorOfArrays{T,N}()
    append!(R, A)
end

VectorOfArrays{T}(A::AbstractVector{<:AbstractArray{U,N}}) where {T,U,N} = VectorOfArrays{T,N}(A)

VectorOfArrays(A::AbstractVector{<:AbstractArray{T,N}}) where {T,N} = VectorOfArrays{T,N}(A)


Base.convert(VA::Type{VectorOfArrays{T,N}}, A::AbstractVector{AbstractArray{U,N}}) where {T,N,U} = VA(A)
Base.convert(VA::Type{VectorOfArrays}, A::AbstractVector{AbstractArray{T,N}}) where {T,N} = VA(A)


"""
    internal_element_ptr(A::VectorOfArrays)

Returns the internal element pointer vector of `A`.

Do *not* change modify the returned vector in any way, as this would break the
inner consistency of `A`.

Use with care, see [`element_ptr`](@ref) for a safe version of this function.
"""
internal_element_ptr(A::VectorOfArrays) = A.elem_ptr


"""
    element_ptr(A::VectorOfArrays)

Returns a copy of the internal element pointer vector of `A`.
"""
element_ptr(A::VectorOfArrays) = deepcopy(internal_element_ptr(A))



function full_consistency_checks(A::VectorOfArrays)
    simple_consistency_checks(A)
    all(eachindex(A.kernel_size)) do i
        len = A.elem_ptr[i+1] - A.elem_ptr[i]
        klen = prod(A.kernel_size[i])
        len >= 0 && (klen == 1 || mod(len, klen) == 0)
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
    len_p, klen_p = promote(len, klen)
    sz_lastdim = len == 0 ? len_p : div(len_p, klen_p)
    sz = (ksize..., sz_lastdim)

    (from:to, sz)
end


import Base.==
(==)(A::VectorOfArrays, B::VectorOfArrays) =
    A.data == B.data && A.elem_ptr == B.elem_ptr && A.kernel_size == B.kernel_size

"""
    flatview(A::VectorOfArrays{T})::Vector{T}

Returns the internal serialized representation of all element arrays of `A` as
a single vector. Do *not* change the length of the returned vector, as it
would break the inner consistency of `A`.
"""
flatview(A::VectorOfArrays) = A.data

Base.size(A::VectorOfArrays) = size(A.kernel_size)

Base.IndexStyle(A::VectorOfArrays) = IndexLinear()


Base.@propagate_inbounds _reshape_dataview(dataview::AbstractArray, s::NTuple{1,Integer}) = dataview

Base.@propagate_inbounds _reshape_dataview(dataview::AbstractArray, s::NTuple{N,Integer}) where {N} =
    Base.__reshape((dataview, IndexStyle(dataview)), s)


Base.@propagate_inbounds function Base.getindex(A::VectorOfArrays, i::Integer)
    @boundscheck checkbounds(A, i)
    r, s = _elem_range_size(A, i)
    dataview = view(A.data, r)
    _reshape_dataview(dataview, s)
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


@inline function Base.resize!(A::VectorOfArrays{T,N,M}, n::Integer) where {T,M,N}
    old_n = length(A)
    if n > old_n
        throw(ArgumentError("Cannot resize VectorOfArrays from length $old_n to $n, can only shrink, not grow"))
    elseif n < old_n
        resize!(A.data, A.elem_ptr[n+1] - 1)
        resize!(A.elem_ptr, n + 1)
        resize!(A.kernel_size, n)
    end
    A
end


function append_elemptr!(A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer})
    idxs_A = LinearIndices(A)
    idxs_B = LinearIndices(B)
    length_B = length(idxs_B)

    A_from = last(idxs_A) + 1
    A_to = A_from - 1 + length_B - 1
    resize!(A, length(first(idxs_A):A_to))

    B_from = first(idxs_B) + 1
    B_to = last(idxs_B)

    A_idxs_offs = A_from - B_from

    checkindex(Bool, eachindex(B), B_from:B_to)
    checkindex(Bool, eachindex(A), (B_from:B_to) .+ A_idxs_offs)
    @inbounds begin
        value_offset = A[A_from - 1] - B[B_from - 1]
        @simd for i in B_from:B_to
            A[i + A_idxs_offs] = B[i] + value_offset
        end
    end

    A
end


function Base.append!(A::VectorOfArrays{T,N}, B::VectorOfArrays{U,N}) where {T,N,U}
    if !isempty(B)
        append!(A.data, B.data)
        append_elemptr!(A.elem_ptr, B.elem_ptr)
        append!(A.kernel_size, B.kernel_size)
    end
    A
end


function Base.append!(A::VectorOfArrays{T,N}, B::AbstractVector{<:AbstractArray{U,N}}) where {T,N,U}
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


Base.vcat(V::VectorOfArrays) = V

function Base.vcat(Vs::(VectorOfArrays{U,N} where U)...) where {N}
    data = vcat(map(x -> x.data, Vs)...)

    elem_ptr = similar(Vs[1].elem_ptr, 1)
    elem_ptr[1] = firstindex(data)
    @inbounds for i in eachindex(Vs)
        append_elemptr!(elem_ptr, Vs[i].elem_ptr)
    end

    kernel_size = vcat(map(x -> x.kernel_size, Vs)...)

    VectorOfArrays(data, elem_ptr, kernel_size, no_consistency_checks)
end


function Base.copy(V::VectorOfArrays)
    VectorOfArrays(copy(V.data), copy(V.elem_ptr), copy(V.kernel_size), no_consistency_checks)
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


function innermap(f::Base.Callable, A::VectorOfArrays)
    new_data = map(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end


function deepmap(f::Base.Callable, A::VectorOfArrays)
    new_data = deepmap(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end


"""
    VectorOfVectors{T,...} = VectorOfArrays{T,1,...}

Constructors:

```julia
VectorOfVectors(A::AbstractVector{<:AbstractVector})
VectorOfVectors{T}(A::AbstractVector{<:AbstractVector}) where {T}

VectorOfVectors(
    data::AbstractVector, elem_ptr::AbstractVector{Int},
    checks::Function = full_consistency_checks
)

See also [VectorOfArrays](@ref).
```
"""
const VectorOfVectors{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    VD<:AbstractVector{Dims{0}}
} = VectorOfArrays{T,1,0,VT,VI,VD}

export VectorOfVectors

VectorOfVectors{T}() where {T} = VectorOfArrays{T,1}()

VectorOfVectors{T}(A::AbstractVector{<:AbstractVector}) where {T} = VectorOfArrays{T,1}(A)
VectorOfVectors(A::AbstractVector{<:AbstractVector}) = VectorOfArrays(A)

VectorOfVectors(
    data::AbstractVector,
    elem_ptr::AbstractVector{Int},
    checks::Function = full_consistency_checks
) = VectorOfArrays(
    data,
    elem_ptr,
    similar(elem_ptr, Dims{0}, size(elem_ptr, 1) - 1),
    checks
)



"""
    consgrouped_ptrs(A::AbstractVector)

Compute an element pointer vector, suitable for creation of a
`VectorOfVectors` that implies grouping equal consecutive entries of
`A`.

Example:

```julia
    A = [1, 1, 2, 3, 3, 2, 2, 2]
    elem_ptr = consgrouped_ptrs(A)
    first.(VectorOfVectors(A, elem_ptr)) == [1, 2, 3, 2]
```
consgrouped_ptrs
Typically, `elem_ptr` will be used to apply the computed grouping to other
data:

```julia
    B = [1, 2, 3, 4, 5, 6, 7, 8]
    VectorOfVectors(B, elem_ptr) == [[1, 2], [3], [4, 5], [6, 7, 8]]
```
"""
function consgrouped_ptrs end
export consgrouped_ptrs

function consgrouped_ptrs(A::AbstractVector)
    elem_ptr = Vector{Int}()
    idxs = eachindex(A)
    push!(elem_ptr, first(idxs))
    if !isempty(A)
        prev_0 = A[first(idxs)]
        prev::typeof(prev_0) = prev_0
        @inbounds for i in (first(idxs) + 1):last(idxs)
            curr = A[i]
            if (curr != prev)
                push!(elem_ptr, i)
                prev = curr
            end
        end
        push!(elem_ptr, last(idxs) + 1)
    end
    elem_ptr
end


"""
    consgroupedview(source::AbstractVector, target)

Compute a grouping of equal consecutive elements on `source` via
[`consgrouped_ptrs`](@ref) and apply the grouping to target, resp. each
element of `target`. `target` may be an vector or a named or unnamed tuple of
vectors. The result is a `VectorOfVectors`, resp. a tuple of such.

Example:

    A = [1, 1, 2, 3, 3, 2, 2, 2]
    B = [1, 2, 3, 4, 5, 6, 7, 8]
    consgroupedview(A, B) == [[1, 2], [3], [4, 5], [6, 7, 8]]

`consgroupedview` plays well with columnar tables, too:

```julia
    using Tables, TypedTables
    data = Table(
        a = [1, 1, 2, 3, 3, 2, 2, 2],
        b = [1, 2, 3, 4, 5, 6, 7, 8],
        c = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
    )

    result = Table(consgroupedview(data.a, Tables.columns(data)))
```

will return

```
     a          b          c
   ┌──────────────────────────────────────
 1 │ [1, 1]     [1, 2]     [1.1, 2.2]
 2 │ [2]        [3]        [3.3]
 3 │ [3, 3]     [4, 5]     [4.4, 5.5]
 4 │ [2, 2, 2]  [6, 7, 8]  [6.6, 7.7, 8.8]
```

without copying any data:

```
    flatview(result.a) === data.a
    flatview(result.b) === data.b
    flatview(result.c) === data.c
```
"""
function consgroupedview end
export consgroupedview

function consgroupedview(source::AbstractVector, target::AbstractVector)
    elem_ptr = consgrouped_ptrs(source)
    VectorOfVectors(target, elem_ptr)
end

function consgroupedview(source::AbstractVector, target::NTuple{N,AbstractVector}) where {N}
    elem_ptr = consgrouped_ptrs(source)
    map(X -> VectorOfVectors(X, elem_ptr), target)
end

function consgroupedview(source::AbstractVector, target::NamedTuple{syms,<:NTuple{N,AbstractVector}}) where {syms,N}
    elem_ptr = consgrouped_ptrs(source)
    map(X -> VectorOfVectors(X, elem_ptr), target)
end
