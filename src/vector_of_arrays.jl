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

User code should typically not instantiate `VectorOfArrays` directly, but use
[`splitup`](@ref) with a [`SplitParts`](@ref) mode or use
[`consgroupedview`](@ref) instead.

# Implementation

Elements are stored via their length and the size of their leading (kernel)
dimensions. If the kernel has zero length the size of the last dimension
cannot be reconstructed, so e.g. an element of size `(0, 3)` reads back
with size `(0, 0)`.

A `VectorOfArrays` created via [`splitup`](@ref) or `view` shares its data
and structural vectors with the split mode resp. the parent array. Mutating
operations like `push!`, `resize!` and `empty!` write through to everything
that shares them. They are rejected before any mutation unless the data and
structural vectors are resizable.

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

`PartsView` is defined as a type alias:

```julia
PartsView{T,VT,VI,VD,ET} = VectorOfArrays{T,1,0,VT,VI,VD,ET}
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


Base.convert(VA::Type{VectorOfArrays{T,N}}, A::AbstractVector{<:AbstractArray{U,N}}) where {T,N,U} = VA(A)
Base.convert(VA::Type{VectorOfArrays}, A::AbstractVector{<:AbstractArray{T,N}}) where {T,N} = VA(A)
Base.convert(::Type{VectorOfArrays{T,N}}, A::VectorOfArrays{T,N}) where {T,N} = A
Base.convert(::Type{VectorOfArrays}, A::VectorOfArrays) = A


"""
    ArraysOfArrays.internal_element_ptr(A::VectorOfArrays)

Returns the internal element pointer vector of `A`.

Do *not* modify the returned vector: this would break the internal
consistency of `A`. See [`element_ptr`](@ref) for a safe alternative.
"""
internal_element_ptr(A::VectorOfArrays) = A.elem_ptr
@compat public internal_element_ptr


"""
    ArraysOfArrays.element_ptr(A::VectorOfArrays)

Returns a copy of the internal element pointer vector of `A`. The pointers
are absolute indices into `fused(A)` and need not start at one, e.g. for
views and partial [`partitioned`](@ref) results.

See also [`getsplitmode`](@ref), which returns the element pointers
together with the kernel sizes.
"""
element_ptr(A::VectorOfArrays) = copy(internal_element_ptr(A))
@compat public element_ptr


"""
    ArraysOfArrays.full_consistency_checks(A::VectorOfArrays)

Check the internal consistency of `A` completely, including whether the
length of each element is compatible with its kernel size. Takes O(length(A))
time, and synchronizes with the device for device-resident structural
vectors.

This is the default value of the `checks` argument of the
[`VectorOfArrays`](@ref) constructor.
"""
function full_consistency_checks(A::VectorOfArrays)
    simple_consistency_checks(A)
    _partition_sizes_valid(A.elem_ptr, A.kernel_size) || throw(ArgumentError("VectorOfArrays inconsistent: Content of elem_ptr and kernel_size is inconsistent"))
    nothing
end
@compat public full_consistency_checks


# Package extensions specialize this for GPU arrays, to avoid scalar indexing:
function _partition_sizes_valid(elem_ptr::AbstractVector{<:Integer}, kernel_size::AbstractVector)
    all(eachindex(kernel_size)) do i
        ep_i, ep_next = elem_ptr[i], elem_ptr[i+1]
        # Compare the pointers, the sign of their difference would wrap for
        # narrow pointer types:
        ep_next < ep_i && return false
        len = ep_next - ep_i
        klen = prod(kernel_size[i])
        klen == 0 ? len == 0 : mod(len, klen) == 0
    end
end


"""
    ArraysOfArrays.simple_consistency_checks(A::VectorOfArrays)

Check the internal consistency of `A` without scanning its structural
vectors: only the first and last element pointer are read, so the check
takes O(1) time. On device-resident structural vectors those two reads
require a device synchronization each.

Suitable as the `checks` argument of the [`VectorOfArrays`](@ref)
constructor.
"""
function simple_consistency_checks(A::VectorOfArrays{T,N,M}) where {T,N,M}
    M == N - 1 || throw(ArgumentError("VectorOfArrays{T,N,M} inconsistent: M must equal N - 1"))
    # Structural vectors must be one-based, only the flat data may have
    # offset axes (it is addressed absolutely via elem_ptr):
    isone(firstindex(A.elem_ptr)) && isone(firstindex(A.kernel_size)) || throw(ArgumentError("VectorOfArrays inconsistent: elem_ptr and kernel_size must be one-based"))
    size(A.elem_ptr, 1) == size(A.kernel_size, 1) + 1 || throw(ArgumentError("VectorOfArrays inconsistent: elem_ptr and kernel_size have incompatible size"))
    ep_first, ep_last = _scalar_first_last(A.elem_ptr)
    ep_first >= firstindex(A.data) || throw(ArgumentError("VectorOfArrays inconsistent: First elem_ptr inconsistent with data indices"))
    ep_last - 1 <= lastindex(A.data) || throw(ArgumentError("VectorOfArrays inconsistent: Last elem_ptr inconsistent with data indices"))
    nothing
end
@compat public simple_consistency_checks

# Package extensions specialize this for GPU arrays, to allow the two scalar
# reads explicitly:
_scalar_first_last(x::AbstractVector) = (first(x), last(x))


# Element pointers are indices into the data, so operations that create new
# pointers require the pointer type to represent them. Checked before any
# data is modified, an unrepresentable index would otherwise wrap silently
# or leave the vector of arrays behind in a modified state:
function _require_elem_ptr_range(elem_ptr::AbstractVector{T}, i::Integer) where {T<:Integer}
    i == i % T ||
        throw(ArgumentError("Element pointers of type $T cannot represent the data index $i"))
    nothing
end


"""
    ArraysOfArrays.no_consistency_checks(A::VectorOfArrays)

Don't check the internal consistency of `A` at all.

Suitable as the `checks` argument of the [`VectorOfArrays`](@ref)
constructor for data and structural vectors that are known to be
consistent with each other.

!!! warning
    Like `@inbounds`, this shifts responsibility to the caller:
    inconsistent structural vectors go undetected and yield silently
    truncated or empty elements, or out-of-bounds access when elements are
    indexed with `@inbounds`.
"""
function no_consistency_checks(A::VectorOfArrays)
    nothing
end
@compat public no_consistency_checks


@inline function _elem_size(ksize::Dims, len::Integer)
    klen = prod(ksize)
    len_p, klen_p = promote(len, klen)
    sz_lastdim = len == 0 ? zero(len_p) : div(len_p, klen_p)
    return (ksize..., Int(sz_lastdim))
end

Base.@propagate_inbounds function _elem_range_size(A::VectorOfArrays, i::Integer)
    r = _voa_elem_range(A.elem_ptr, i)
    sz = _elem_size(A.kernel_size[i], length(r))
    (r, sz)
end


function innerlengths(A::VectorOfArrays)
    ep = A.elem_ptr
    return view(ep, firstindex(ep)+1:lastindex(ep)) .- view(ep, firstindex(ep):lastindex(ep)-1)
end

innersizes(A::VectorOfArrays) = _elem_size.(A.kernel_size, innerlengths(A))


# Equality must be equivalent to elementwise comparison, but can be checked
# much more efficiently via the flat data. The underlying data of A and B may
# be offset relative to their element pointers, so only element sizes and the
# data regions covered by the elements may be compared:

# Device-resident data needs vectorized comparisons, the GPUArraysCore
# extension specializes these:
_vecdata_equal(a::AbstractVector, b::AbstractVector) = a == b
_vecdata_isequal(a::AbstractVector, b::AbstractVector) = isequal(a, b)

function _elem_lengths_equal(elem_ptr_A::AbstractVector{<:Integer}, elem_ptr_B::AbstractVector{<:Integer})
    length(elem_ptr_A) == length(elem_ptr_B) || return false
    iA, iB = firstindex(elem_ptr_A), firstindex(elem_ptr_B)
    @inbounds for k in 0:(length(elem_ptr_A) - 2)
        elem_ptr_A[iA+k+1] - elem_ptr_A[iA+k] == elem_ptr_B[iB+k+1] - elem_ptr_B[iB+k] || return false
    end
    return true
end

import Base.==
function ==(A::VectorOfArrays{<:Any,N}, B::VectorOfArrays{<:Any,N}) where {N}
    axes(A) == axes(B) || return false
    _shapeinfo_equal(A.kernel_size, B.kernel_size) || return false
    _elem_lengths_equal(A.elem_ptr, B.elem_ptr) || return false
    return _vecdata_equal(vecflattened(A), vecflattened(B))
end

function Base.isequal(A::VectorOfArrays{<:Any,N}, B::VectorOfArrays{<:Any,N}) where {N}
    axes(A) == axes(B) && _shapeinfo_equal(A.kernel_size, B.kernel_size) &&
        _elem_lengths_equal(A.elem_ptr, B.elem_ptr) &&
        _vecdata_isequal(vecflattened(A), vecflattened(B))
end

"""
    flatview(A::VectorOfArrays{T})::AbstractVector{T}

Returns the data of all element arrays of `A` as a single vector, without
copying. If the elements of `A` cover its internal storage completely, this
is the internal storage vector itself, otherwise a view of the covered
region (e.g. for views of vectors of arrays and partial
[`partitioned`](@ref) views). Do *not* change the length of the returned
vector, as this would break the internal consistency of `A`.
"""
function flatview(A::VectorOfArrays)
    ep_first, ep_last = _scalar_first_last(A.elem_ptr)
    if ep_first == firstindex(A.data) && ep_last == lastindex(A.data) + 1
        A.data
    else
        view(A.data, ep_first:(ep_last - 1))
    end
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

# Implementation

[`getsplitmode(A::VectorOfArrays)`](@ref) copies the shape information of
`A` where required (typically an O(length) operation), so the resulting
mode is not affected if `A` is resized afterwards. Shape vectors that
cannot be resized may be shared instead. A `VectorOfArrays` created via
[`splitup`](@ref), on the other hand, shares the vectors of the mode it was
created from, like the `VectorOfArrays` inner constructor.
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

Base.:(==)(a::SplitParts, b::SplitParts) =
    a.elem_ptr == b.elem_ptr && a.kernel_size == b.kernel_size

Base.hash(x::SplitParts, h::UInt) =
    hash(x.kernel_size, hash(x.elem_ptr, hash(:SplitParts, h)))

# Defensive copy: the split mode must not be affected if the vector of
# arrays is resized later on. Package extensions may specialize this for
# vector types that cannot be resized:
_shapeinfo_copy(x::AbstractVector) = copy(x)

getsplitmode(A::VectorOfArrays) = SplitParts(_shapeinfo_copy(A.elem_ptr), _shapeinfo_copy(A.kernel_size))

@inline fused(A::VectorOfArrays) = A.data


# Comparing the part boundaries is O(n), vectorized on GPUs. splitup
# shares the shape vectors of the mode it was created from, so round
# trips take the egal fast path:
_shapeinfo_equal(a::AbstractVector, b::AbstractVector) =
    a === b || (axes(a) == axes(b) && a == b)

function (f::FuseArrays{<:SplitParts{N}})(A::VectorOfArrays{T,N}) where {T,N}
    _shapeinfo_equal(A.elem_ptr, f.smode.elem_ptr) &&
        _shapeinfo_equal(A.kernel_size, f.smode.kernel_size) ||
        throw(ArgumentError("Partitioning of array does not match the split mode of the inverse"))
    return fused(A)
end

function (f::FuseArrays{<:SplitParts})(A::AbstractArray)
    throw(ArgumentError("Inverse of a SplitParts mode requires a VectorOfArrays with matching element dimensionality"))
end

function splitup(A::AbstractVector, smode::SplitParts)
    VectorOfArrays(A, smode.elem_ptr, smode.kernel_size, full_consistency_checks)
end

_splitup_trusted(A::AbstractVector, smode::SplitParts) =
    VectorOfArrays(A, smode.elem_ptr, smode.kernel_size, simple_consistency_checks)

function _bcast_expand(x::AbstractArray, smode::SplitParts, ref_flat::AbstractArray)
    ep = smode.elem_ptr
    if x isa AbstractVector && axes(x) == axes(smode.kernel_size)
        # One value per part, broadcast over the contents of that part. The
        # value of x used for data not covered by any part is arbitrary,
        # such data does not contribute to the result:
        idx = similar(ref_flat, Int)
        idx .= firstindex(ref_flat):lastindex(ref_flat)
        seg = clamp.(searchsortedlast.(Ref(ep), idx), firstindex(x), lastindex(x))
        return x[seg]
    elseif x isa AbstractVector && axes(x) == axes(ref_flat)
        return x
    else
        throw(DimensionMismatch("bcastat argument shape matches neither the outer structure nor the flat data of the nested arguments"))
    end
end


function partitioned(A::AbstractVector, lengths::AbstractVector{<:Integer})
    elem_ptr = _elem_ptr_from_lengths(A, lengths)
    kernel_size = fill!(similar(lengths, Dims{0}, length(lengths)), ())
    _splitup_trusted(A, SplitParts(elem_ptr, kernel_size))
end

function partitioned(A::AbstractVector, shapes::AbstractVector{Dims{N}}) where {N}
    shapes_1b = _one_based(shapes)
    elem_ptr = _elem_ptr_from_lengths(A, prod.(shapes_1b))
    kernel_size = Base.front.(shapes_1b)
    _splitup_trusted(A, SplitParts(elem_ptr, kernel_size))
end

# Element pointers of consecutive parts of the given lengths, based at
# firstindex(data). cumsum! requires matching indices, so lengths with
# offset axes are rebased:
function _elem_ptr_cumsum!(elem_ptr::AbstractVector{<:Integer}, data::AbstractVector, lengths::AbstractVector{<:Integer})
    lengths_1b = _one_based(lengths)
    fill!(elem_ptr, zero(eltype(elem_ptr)))
    cumsum!(view(elem_ptr, firstindex(elem_ptr) + 1:lastindex(elem_ptr)), lengths_1b)
    elem_ptr .+= firstindex(data)
    return elem_ptr
end

function _elem_ptr_from_lengths(A::AbstractVector, lengths::AbstractVector{<:Integer})
    all(>=(0), lengths) || throw(ArgumentError("Part lengths must not be negative"))
    elem_ptr = _elem_ptr_cumsum!(similar(lengths, Int, length(lengths) + 1), A, lengths)
    _scalar_first_last(elem_ptr)[2] - 1 <= lastindex(A) || throw(ArgumentError("Sum of part lengths exceeds length of data vector"))
    return elem_ptr
end


function vecflattened(A::VectorOfArrays)
    ep_first, ep_last = _scalar_first_last(A.elem_ptr)
    view(A.data, ep_first:(ep_last - 1))
end

_flatdata(A::VectorOfArrays) = vecflattened(A)

# vecflattened(A) is one-based, so the split mode is rebased accordingly:
function _flatdatamode(A::VectorOfArrays)
    ep = A.elem_ptr
    ep_first, _ = _scalar_first_last(ep)
    delta = ep_first - 1
    rebased_ep = delta == 0 ? _shapeinfo_copy(ep) : ep .- delta
    SplitParts(rebased_ep, _shapeinfo_copy(A.kernel_size))
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
    @boundscheck checkbounds(A, idxs)
    # Empty ranges may lie outside the axes of A:
    from, to = isempty(idxs) ? (firstindex(A), firstindex(A) - 1) : (Int(first(idxs)), Int(last(idxs)))
    elem_ptr = A.elem_ptr[from:(to+1)]
    kernel_size = A.kernel_size[from:to]
    ep_first, ep_last = _scalar_first_last(elem_ptr)
    data = A.data[ep_first:(ep_last - 1)]
    broadcast!(+, elem_ptr, elem_ptr, firstindex(data) - ep_first)
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
    size(a) == size(x) || throw(DimensionMismatch("Can't assign array to element $i of VectorOfArrays, array size is incompatible"))
    a[:] = x
    return A
end

function Base.fill!(A::VectorOfArrays{T,N}, x::AbstractArray{U,N}) where {T,N,U}
    all(isequal(size(x)), innersizes(A)) || throw(DimensionMismatch("Can't fill VectorOfArrays with an array of different size than its elements"))
    reshape(vecflattened(A), size(x)..., length(A)) .= x
    return A
end

Base.length(A::VectorOfArrays) = length(A.kernel_size)


# Structural mutation requires resizable data and structural vectors, so
# operations that grow or shrink a VectorOfArrays check all three before
# modifying any of them. Storage types that support resize! but throw for
# other reasons can still fail with fields already modified:
_is_resizable(::Vector) = true
_is_resizable(x::AbstractVector) = applicable(resize!, x, 0)

function _require_structurally_mutable(A::VectorOfArrays)
    _is_resizable(A.data) && _is_resizable(A.elem_ptr) && _is_resizable(A.kernel_size) ||
        throw(ArgumentError("Operation requires a VectorOfArrays with resizable data and structural vectors"))
    nothing
end

# Elements are appended directly after the last element, so growing requires
# that no unused trailing data is in the way:
function _require_growable(A::VectorOfArrays, what::AbstractString)
    _require_structurally_mutable(A)
    _, ep_last = _scalar_first_last(A.elem_ptr)
    ep_last == lastindex(A.data) + 1 ||
        throw(ArgumentError("Cannot $what a VectorOfArrays that has unused trailing data"))
    nothing
end

@inline function Base.resize!(A::VectorOfArrays{T,N,M}, n::Integer) where {T,M,N}
    _require_structurally_mutable(A)
    old_n = length(A)
    if n > old_n
        throw(ArgumentError("Cannot resize VectorOfArrays from length $old_n to $n, can only shrink, not grow"))
    elseif n < old_n
        resize!(A.data, A.elem_ptr[n+1] - firstindex(A.data))
        resize!(A.elem_ptr, n + 1)
        resize!(A.kernel_size, n)
    end
    A
end


function _append_elemptr!(A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer})
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
        # Only the data covered by the elements of B may be appended:
        _require_growable(A, "append to")
        B_flat = vecflattened(B)
        _require_elem_ptr_range(A.elem_ptr, lastindex(A.data) + length(B_flat) + 1)
        append!(A.data, B_flat)
        _append_elemptr!(A.elem_ptr, B.elem_ptr)
        append!(A.kernel_size, B.kernel_size)
    end
    A
end


function Base.append!(A::VectorOfArrays{T,N}, B::AbstractVector{<:AbstractArray{U,N}}) where {T,N,U}
    if !isempty(B)
        _require_structurally_mutable(A)
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


# Concatenating vectors of arrays concatenates the data covered by their
# elements, in a single pass:

function _vcat_voas(Vs)
    isempty(Vs) && throw(ArgumentError("reducing over an empty collection is not allowed"))
    V1 = first(Vs)

    T = mapreduce(V -> eltype(V.data), promote_type, Vs)
    n_parts = sum(length, Vs)
    ep_bounds = map(V -> _scalar_first_last(V.elem_ptr), Vs)
    n_data = sum(((ep_first, ep_last),) -> Int(ep_last - ep_first), ep_bounds)

    data = similar(V1.data, T, n_data)
    elem_ptr = similar(V1.elem_ptr, n_parts + 1)
    _require_elem_ptr_range(elem_ptr, firstindex(data) + n_data)
    kernel_size = similar(V1.kernel_size, n_parts)

    # The pointers are constructed with vectorized operations, the shape
    # information may reside on a device:
    fill!(view(elem_ptr, firstindex(elem_ptr):firstindex(elem_ptr)), firstindex(data))
    i_ptr = firstindex(elem_ptr)
    i_ksz = firstindex(kernel_size)
    i_data = firstindex(data)
    base = Int(firstindex(data))

    for (V, (ep_first, ep_last)) in zip(Vs, ep_bounds)
        ep = V.elem_ptr
        covered_len = Int(ep_last - ep_first)
        copyto!(data, i_data, V.data, ep_first, covered_len)

        m = length(ep) - 1
        @views elem_ptr[(i_ptr + 1):(i_ptr + m)] .= ep[(begin + 1):end] .- (Int(ep_first) - base)
        i_ptr += m
        i_data += covered_len
        base += covered_len

        ksz = V.kernel_size
        copyto!(kernel_size, i_ksz, ksz, firstindex(ksz), length(ksz))
        i_ksz += length(ksz)
    end

    return VectorOfArrays(data, elem_ptr, kernel_size, no_consistency_checks)
end

Base.vcat(V1::VectorOfArrays{<:Any,N}, Vs::VectorOfArrays{<:Any,N}...) where {N} = _vcat_voas((V1, Vs...))

Base.reduce(::typeof(vcat), Vs::AbstractVector{<:VectorOfArrays}) = _vcat_voas(Vs)
# Disambiguation:
Base.reduce(::typeof(vcat), Vs::AbstractVector{<:VectorOfArrays{<:Any,1}}) = _vcat_voas(Vs)
Base.reduce(::typeof(vcat), Vs::AbstractVector{<:VectorOfArrays{<:Any,2}}) = _vcat_voas(Vs)


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
    _require_growable(A, "push to")
    _require_elem_ptr_range(A.elem_ptr, lastindex(A.data) + length(x) + 1)
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
    _require_structurally_mutable(A)
    empty!(A.data)
    resize!(A.elem_ptr, 1)
    # The data may have had leading regions not covered by any element:
    A.elem_ptr[begin] = firstindex(A.data)
    empty!(A.kernel_size)
    A
end


# Map over the covered data only, the result does not share shape
# information with A:
deepmap(f, A::VectorOfArrays) = _generic_deepmap_impl(f, A)


# Like Base's map/broadcast of identity over vectors of views, the result
# shares the element data, but the outer container (here: the structural
# vectors) is independent:
_outer_copy(A::VectorOfArrays) =
    VectorOfArrays(A.data, _shapeinfo_copy(A.elem_ptr), _shapeinfo_copy(A.kernel_size), no_consistency_checks)

Base.map(::typeof(identity), A::VectorOfArrays) = _outer_copy(A)
Base.Broadcast.broadcasted(::typeof(identity), A::VectorOfArrays) = _outer_copy(A)



"""
    PartsView{T,...} = VectorOfArrays{T,1,0,...}

A vector of vectors (that may differ in length), stored in contiguous,
partitioned form. See [`VectorOfArrays`](@ref) for details.

User code should typically not construct a `PartsView` directly, but use
[`splitup`](@ref) with a [`SplitParts`](@ref) mode or use
[`consgroupedview`](@ref) instead.

# Implementation

Constructors:

```julia
PartsView(A::AbstractVector{<:AbstractVector})
PartsView{T}(A::AbstractVector{<:AbstractVector}) where {T}

PartsView(
    data::AbstractVector, elem_ptr::AbstractVector{<:Integer},
    checks::Function = full_consistency_checks
)
```
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


"""
    VectorOfVectors{T,...} = PartsView{T,...}

Alias for [`PartsView`](@ref).
"""
const VectorOfVectors = PartsView
export VectorOfVectors
