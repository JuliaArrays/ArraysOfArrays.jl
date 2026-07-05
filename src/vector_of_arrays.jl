# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


Base.@propagate_inbounds _voa_elem_range(elem_ptr::AbstractVector{<:Integer}, i::Integer) =
    elem_ptr[i]:(elem_ptr[i+1] - 1)

# Element type of a `VectorOfArrays{T,N}` with data of type `VT` and element
# pointers of type `VI`, must match the return type of `getindex` exactly:
@inline function _voa_eltype(::Type{VT}, ::Type{VI}, ::Val{N}) where {VT<:AbstractVector,VI<:AbstractVector{<:Integer},N}
    R = Base.promote_op(_voa_elem_range, VI, Int)
    SV = Base.promote_op(view, VT, R)
    return Base.promote_op(_reshape_dataview, SV, NTuple{N,Int})
end


"""
    VectorOfArrays{T,N,M,VT,VI,VD,ET<:AbstractArray{T,N}} <: AbstractVector{ET}

A `VectorOfArrays` represents a vector of `N`-dimensional arrays (that may
differ in size). Internally, `VectorOfArrays` stores all elements of all
arrays in a single flat vector. `M` must equal `N - 1`.

The `VectorOfArrays` itself supports `push!`, `append!`, etc., but the size
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
    elem_ptr::AbstractVector{<:Integer},
    kernel_size::AbstractVector{<:Dims}
    checks::Function = ArraysOfArrays.full_consistency_checks
)
```

Other suitable values for `checks` are `ArraysOfArrays.simple_consistency_checks`
and `ArraysOfArrays.no_consistency_checks`.

`PartsView` is defined as an type alias:

```julia
`PartsView{T,VT,VI,VD,ET} = VectorOfArrays{T,1,0,VT,VI,VD,ET}`
```
"""
struct VectorOfArrays{
    T, N, M,
    VT<:AbstractVector{T},
    VI<:AbstractVector{<:Integer},
    VD<:AbstractVector{Dims{M}},
    ET<:AbstractArray{T,N}
} <: AbstractVector{ET}
    data::VT
    elem_ptr::VI
    kernel_size::VD

    function VectorOfArrays{T,N}() where {T,N}
        M = length(Base.front(ntuple(_ -> 0, Val{N}())))
        data = Vector{T}()
        elem_ptr = [firstindex(data)]
        kernel_size = Vector{Dims{M}}()

        ET = _voa_eltype(typeof(data), typeof(elem_ptr), Val(N))
        new{
            T, N, M,
            typeof(data),
            typeof(elem_ptr),
            typeof(kernel_size),
            ET
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
        VI<:AbstractVector{<:Integer},
        VD<:AbstractVector{Dims{M}}
    }
        N = length((ntuple(_ -> 0, Val{M}())..., 0))
        ET = _voa_eltype(VT, VI, Val(N))
        A = new{T,N,M,VT,VI,VD,ET}(data, elem_ptr, kernel_size)
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

Do *not* modify the returned vector in any way, as this would break the
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
    r = _voa_elem_range(A.elem_ptr, i)
    len = length(r)

    ksize = A.kernel_size[i]
    klen = prod(ksize)
    len_p, klen_p = promote(len, klen)
    sz_lastdim = len == 0 ? len_p : div(len_p, klen_p)
    sz = (ksize..., Int(sz_lastdim))

    (r, sz)
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
flatview(A::VectorOfArrays{<:Any,N,M,<:Any}) where {N,M} = A.data

function flatview(A::VectorOfArrays{<:Any,N,M,<:Any,<:SubArray}) where {N,M}
    view(A.data, A.elem_ptr[begin]:A.elem_ptr[end]-1)
end


"""
    struct SplitParts{M,VI,VD} <: AbstractPartMode{M,1}

The split mode of [`VectorOfArrays`](@ref): a partition of a vector into
consecutive parts of possibly different size, viewed as a vector of
`M`-dimensional arrays.

Constructor:

```
SplitParts(
    elem_ptr::AbstractVector{<:Integer},
    kernel_size::AbstractVector{Dims{M-1}}
)
```

`elem_ptr` and `kernel_size` equal the equivalent properties of
`VectorOfArrays`.

See also [`AbstractPartMode`](@ref).
"""
struct SplitParts{
    M,
    VI<:AbstractVector{<:Integer},
    VD<:AbstractVector{<:Dims}
} <: AbstractPartMode{M,1}
    elem_ptr::VI
    kernel_size::VD

    function SplitParts(
        elem_ptr::VI,
        kernel_size::VD
    ) where {
        Mk,
        VI<:AbstractVector{<:Integer},
        VD<:AbstractVector{Dims{Mk}}
    }
        M = _val_value(_add_vals(Val(Mk), Val(1)))
        new{M,VI,VD}(elem_ptr, kernel_size)
    end
end
export SplitParts

is_memordered_splitmode(::SplitParts) = true

@inline getsplitmode(A::VectorOfArrays) = SplitParts(A.elem_ptr, A.kernel_size)

@inline fused(A::VectorOfArrays) = A.data

function splitup(A::AbstractVector, smode::SplitParts)
    VectorOfArrays(A, smode.elem_ptr, smode.kernel_size, simple_consistency_checks)
end


function partitioned(A::AbstractVector, lengths::AbstractVector{<:Integer})
    elem_ptr = _elem_ptr_from_lengths(A, lengths)
    kernel_size = similar(lengths, Dims{0})
    fill!(kernel_size, ())
    VectorOfArrays(A, elem_ptr, kernel_size, no_consistency_checks)
end

function partitioned(A::AbstractVector, shapes::AbstractVector{Dims{N}}) where {N}
    elem_ptr = _elem_ptr_from_lengths(A, prod.(shapes))
    kernel_size = Base.front.(shapes)
    VectorOfArrays(A, elem_ptr, kernel_size, no_consistency_checks)
end

function _elem_ptr_from_lengths(A::AbstractVector, lengths::AbstractVector{<:Integer})
    elem_ptr = similar(lengths, Int, length(lengths) + 1)
    i = firstindex(elem_ptr)
    elem_ptr[i] = firstindex(A)
    for l in lengths
        l >= 0 || throw(ArgumentError("Part lengths must not be negative"))
        elem_ptr[i + 1] = elem_ptr[i] + l
        i += 1
    end
    last(elem_ptr) - 1 <= lastindex(A) || throw(ArgumentError("Sum of part lengths exceeds length of data vector"))
    return elem_ptr
end


function vecflattened(A::VectorOfArrays)
    ep = A.elem_ptr
    view(A.data, first(ep):(last(ep) - 1))
end

# Fast paths, must return independent arrays, unlike `vecflattened`:
Base.mapreduce(::typeof(vec), ::typeof(vcat), A::VectorOfArrays) = copy(vecflattened(A))
Base.reduce(::typeof(vcat), A::VectorOfArrays{T,1}) where {T} = copy(vecflattened(A))


Base.size(A::VectorOfArrays) = size(A.kernel_size)

Base.IndexStyle(::Type{<:VectorOfArrays}) = IndexLinear()


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


@inline _explicit_idxs(::AbstractVector, idxs::AbstractVector{<:Integer}) = idxs
@inline _explicit_idxs(eachidx::AbstractVector, idxs::Base.LogicalIndex) = eachidx[idxs]

Base.@propagate_inbounds function Base._getindex(l::IndexStyle, A::VectorOfArrays, raw_idxs::AbstractVector{<:Integer})
    @boundscheck checkbounds(A, raw_idxs)

    idxs = _explicit_idxs(eachindex(A), raw_idxs)

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
    a = A[i]
    # @boundscheck size(a) == size(x) || throw(DimensionMismatch("Can't assign array to element $i of VectorOfArrays, array size is incompatible"))
    a[:] = x
    return A
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
    sizehint!(A.data, n * prod(s))
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


function Base.empty(A::VectorOfArrays{T,N}, ::Type{<:AbstractArray{U,N}}) where {T,N,U}
    empty_data = empty(A.data, U)
    empty_elem_ptr = push!(empty(A.elem_ptr), firstindex(empty_data))
    empty_kernel_size = empty(A.kernel_size)
    VectorOfArrays(empty_data, empty_elem_ptr, empty_kernel_size, no_consistency_checks)
end

function Base.empty!(A::VectorOfArrays)
    empty!(A.data)
    resize!(A.elem_ptr, 1)
    empty!(A.kernel_size)
    A
end


function innermap(f::Base.Callable, A::VectorOfArrays)
    new_data = map(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end


function deepmap(f::Base.Callable, A::VectorOfArrays)
    new_data = deepmap(f, A.data)
    VectorOfArrays(new_data, A.elem_ptr, A.kernel_size, simple_consistency_checks)
end


Base.map(::typeof(identity), A::VectorOfArrays) = A
Base.Broadcast.broadcasted(::typeof(identity), A::VectorOfArrays) = A



"""
    PartsView{T,...} = VectorOfArrays{T,1,0,...}

A vector of vectors (that may differ in length), stored in contiguous,
partitioned form. See [`VectorOfArrays`](@ref) for details.

Constructors:

```julia
PartsView(A::AbstractVector{<:AbstractVector})
PartsView{T}(A::AbstractVector{<:AbstractVector}) where {T}

PartsView(
    data::AbstractVector, elem_ptr::AbstractVector{<:Integer},
    checks::Function = full_consistency_checks
)
```

See also [`VectorOfArrays`](@ref).
"""
const PartsView{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{<:Integer},
    VD<:AbstractVector{Dims{0}},
    ET<:AbstractVector{T}
} = VectorOfArrays{T,1,0,VT,VI,VD,ET}

export PartsView

PartsView{T}() where {T} = VectorOfArrays{T,1}()

PartsView{T}(A::AbstractVector{<:AbstractVector}) where {T} = VectorOfArrays{T,1}(A)
PartsView(A::AbstractVector{<:AbstractVector}) = VectorOfArrays(A)

PartsView(
    data::AbstractVector,
    elem_ptr::AbstractVector{I},
    checks::Function = full_consistency_checks
) where I <: Integer= VectorOfArrays(
    data,
    elem_ptr,
    similar(elem_ptr, Dims{0}, size(elem_ptr, 1) - 1),
    checks
)



"""
    consgrouped_ptrs(A::AbstractVector)

Compute an element pointer vector, suitable for creation of a
`PartsView` that implies grouping equal consecutive entries of
`A`.

Example:

```julia
    A = [1, 1, 2, 3, 3, 2, 2, 2]
    elem_ptr = consgrouped_ptrs(A)
    first.(PartsView(A, elem_ptr)) == [1, 2, 3, 2]
```

Typically, `elem_ptr` will be used to apply the computed grouping to other
data:

```julia
    B = [1, 2, 3, 4, 5, 6, 7, 8]
    PartsView(B, elem_ptr) == [[1, 2], [3], [4, 5], [6, 7, 8]]
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
element of `target`. `target` may be a vector or a named or unnamed tuple of
vectors. The result is a `PartsView`, resp. a tuple of such.

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
    PartsView(target, elem_ptr)
end

function consgroupedview(source::AbstractVector, target::NTuple{N,AbstractVector}) where {N}
    elem_ptr = consgrouped_ptrs(source)
    map(X -> PartsView(X, elem_ptr), target)
end

function consgroupedview(source::AbstractVector, target::NamedTuple{syms,<:NTuple{N,AbstractVector}}) where {syms,N}
    elem_ptr = consgrouped_ptrs(source)
    map(X -> PartsView(X, elem_ptr), target)
end


# Deprecated:

Base.@deprecate_binding VectorOfVectors PartsView
