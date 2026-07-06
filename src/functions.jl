# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractSplitMode <: Function

Abstract supertype for array split modes.

Use [`getsplitmode`](@ref) to get the split mode of an array.

Use [`splitup`](@ref) or call `smode::AbstractSplitMode` as a function to
split an array:

```julia
splitup(A, smode) === smode(A)
```

See also [`splitup`](@ref) and [`fused`](@ref).

# Implementation

Subtypes of `AbstractSplitMode` should specialize

* `splitup(A, smode::SomeSplitMode)` for arrays `A`
* `is_memordered_splitmode(smode::SomeSplitMode)`

`(smode::SomeSplitMode)(A)` calls `splitup(A, smode)` by default and should
not be specialized.
"""
abstract type AbstractSplitMode <: Function end
export AbstractSplitMode

(smode::AbstractSplitMode)(A::AbstractArray) = splitup(A, smode)


"""
    is_memordered_splitmode(smode::AbstractSplitMode)::Bool

Check whether `smode` splits in memory order.

If true, inner arrays are stored contiguously in memory in Julia-native
dimension order, and the same is true for the outer dimensions (no dimension
reordering).

If true, `flatview` and `fused` are equivalent.
"""
function is_memordered_splitmode end
export is_memordered_splitmode


"""
    struct NonSplitMode{N} <: AbstractSplitMode

The split mode of unsplit collections that have `N` dimensions.

Constructor: `NonSplitMode{N}()`
"""
struct NonSplitMode{N} <: AbstractSplitMode end
export NonSplitMode

is_memordered_splitmode(::NonSplitMode) = true


"""
    struct UnknownSplitMode{AT} <: AbstractSplitMode

Split mode of generic split objects of type `AT` that have been split in an
unknown way, e.g. nested arrays of type `Array{<:Array}`.

Since the split parts may be (typically are) non-contiguous in memory, this
split mode does not allow for `fused` or `flatview`. It is also not
inferable whether the split object should be interpreted as a sliced array,
a ragged array, or something else.

Constructor: `UnknownSplitMode{T}()`
"""
struct UnknownSplitMode{AT} <: AbstractSplitMode end
export UnknownSplitMode

is_memordered_splitmode(::UnknownSplitMode) = false


"""
    getsplitmode(A::AbstractArray)::NonSplitMode
    getsplitmode(A::AbstractArray{<:AbstractArray})::AbstractSplitMode

Get the split mode of `A`.

`splitup(fused(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible, except if `getsplitmode(A)` is an
`UnknownSplitMode`.

`getsplitmode` should be a zero-copy O(1) operation, if at all possible.
"""
function getsplitmode end
export getsplitmode

@inline getsplitmode(::AbstractArray{<:Any,N}) where N = NonSplitMode{N}()

@inline getsplitmode(A::AbstractArray{<:AbstractArray}) = UnknownSplitMode{typeof(A)}()


"""
    splitup(A::AbstractArray, smode::AbstractSplitMode)

View array `A` in split form, as an array of arrays.

`splitup` should be a zero-copy O(1) operation, if at all possible.

See also [`fused`](@ref) and [`getsplitmode`](@ref).
"""
function splitup end
export splitup

@inline splitup(A::AbstractArray, ::NonSplitMode) = A

function splitup(::AbstractArray, ::UnknownSplitMode)
    throw(ArgumentError("splitup cannot be used with UnknownSplitMode"))
end


"""
    fused(A::AbstractArray)
    fused(A::AbstractArray{<:AbstractArray})

View array `A` in unsplit form.

`splitup(fused(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible, except if `getsplitmode(A)` is an
`UnknownSplitMode`.

If `A` is not a nested array, return `A` itself. If `A` is a split array,
return the original unsplit array.

If `is_memordered_splitmode(getsplitmode(A))` is true, `fused(A)` is
equivalent to [`flatview(A)`](@ref).

`fused` should be a zero-copy O(1) operation, if at all possible.
"""
function fused end
export fused

@inline fused(A::AbstractArray) = A

@inline fused(A::AbstractArray{<:AbstractArray}) = _fused_impl(A, getsplitmode(A))

@inline _fused_impl(obj, ::NonSplitMode) = obj

function _fused_impl(@nospecialize(obj), ::UnknownSplitMode)
    throw(ArgumentError("fused not implemented for objects of type $(nameof(typeof(obj))) with unknown split mode"))
end


"""
    flatview(A::AbstractArray)
    flatview(A::AbstractArray{<:AbstractArray})

View array `A` in a flattened form, with inner dimensions first. The shape of
the flattened form will depend on the type of `A`. If `A` is not a
nested array, the return value is `A` itself. Only specific types of nested
arrays are supported.

`flatview` is a zero-copy O(1) operation.

If `is_memordered_splitmode(getsplitmode(A))` is true, `flatview(A)` is
equivalent to [`fused(A)`](@ref).

For sliced arrays the result of `flatview(A)` will equal [`stacked(A)`](@ref).
For partitioned vectors it will equal [`vecflattened(A)`](@ref), provided
that the parts cover the underlying data completely.
"""
function flatview end
export flatview

@inline flatview(A::AbstractArray) = A
function flatview(A::AbstractArray{<:AbstractArray})
    throw(ArgumentError("flatview not implemented for nested arrays of type $(nameof(typeof(A)))"))
end

function flatview(A::AbstractSlices{<:AbstractArray})
    smode = getsplitmode(A)
    if is_memordered_splitmode(smode)
        return fused(A)
    else
        throw(ArgumentError("flatview requires memory-ordered split/slicing, but array has split mode $smode"))
    end
end



"""
    abstract type AbstractSlicingMode{M,N} <: AbstractSplitMode

Abstract supertype for array slicing modes with `M` inner dimensions and `N`
outer dimensions.

Use `getsplitmode` to get the split mode of a split array.
"""
abstract type AbstractSlicingMode{M,N} <: AbstractSplitMode end
export AbstractSlicingMode


"""
    getslicemap(A::AbstractSlices)

Return the slicemap of `A` with respect to `B = fused(A)`: a tuple with one
entry per dimension of `B`, `Colon()` for sliced (inner) dimensions and `k`
for dimensions indexed by dimension `k` of `A`, so that

```julia
A[i...] == view(B, map(s -> s isa Colon ? (:) : i[s], getslicemap(A))...)
```

E.g. `A = eachslice(B, dims = (3,1,5))` of a five-dimensional `B` has the
slicemap `(2, :, 1, :, 3)`, since `A[i1, i2, i3] == view(B, i2, :, i1, :, i3)`.

Equals the `slicemap` field of `Base.Slices` objects.
"""
function getslicemap end
export getslicemap


"""
    ArraysOfArrays.getinnerdims(tpl::Tuple, smode::AbstractSplitMode)

Get the entries of `tpl` corresponding to the inner dimensions of split
mode `smode`, in the order specified by `smode`.
"""
function getinnerdims end

@inline getinnerdims(::Tuple, ::NonSplitMode) = ()

function getinnerdims(::Tuple, ::UnknownSplitMode)
    throw(ArgumentError("getinnerdims cannot be used with UnknownSplitMode"))
end


"""
    ArraysOfArrays.getouterdims(tpl::Tuple, smode::AbstractSplitMode)

Get the entries of `tpl` corresponding to the outer dimensions of split
mode `smode`, in the order specified by `smode`.
"""
function getouterdims end

@inline getouterdims(x::Tuple, ::NonSplitMode) = x

function getouterdims(::Tuple, ::UnknownSplitMode)
    throw(ArgumentError("getouterdims cannot be used with UnknownSplitMode"))
end


"""
    unstackmode(A::AbstractArray)
    unstackmode(A::AbstractArray{<:AbstractArray})

Get the split mode required to restore `stacked(A)` so that
`splitup(stacked(A), unstackmode(A)) == A`.

The result of `splitup(stacked(A), unstackmode(A))` may have a different
type and underlying memory layout than `A`.
"""
function unstackmode end
export unstackmode

@inline unstackmode(::AbstractArray{<:Any,N}) where N = NonSplitMode{N}()

@inline unstackmode(A::AbstractArray{<:AbstractArray}) = UnknownSplitMode{typeof(A)}()

function unstackmode(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    innersize(A)  # Ensure element arrays have equal size
    return SplitSlices{M,N}()
end


"""
    stacked(A::AbstractArray{T,N})::AbstractArray{T,N}
    stacked(A::AbstractArray{<:AbstractArray{T,M},N})::AbstractArray{T,M+N}

Join stacked arrays of a nested array into a single array along one or more
new dimensions, return non-nested arrays unchanged.

Similar to `Base.stack`, but can return the original underlying array of
sliced arrays in more cases.
"""
function stacked end
export stacked

@inline stacked(A::AbstractArray) = A
@inline stacked(A::AbstractArray{<:AbstractArray}) = _stacked_impl(A, getsplitmode(A))

_stacked_impl(A::AbstractArray{<:AbstractArray}, ::AbstractSplitMode) = stack(A)

function _stacked_impl(A::AbstractSlices{<:AbstractArray}, smode::AbstractSlicingMode)
    A_joined = fused(A)
    is_memordered_splitmode(smode) ? A_joined : _stacked_permutedims(A_joined, smode)
end

function _stacked_permutedims(A_joined::AbstractArray{T,N}, smode::AbstractSlicingMode) where {T,N}
    dimnumbers = _oneto_tpl(Val(N))
    dimorder = (getinnerdims(dimnumbers, smode)..., getouterdims(dimnumbers, smode)...)
    return permutedims(A_joined, dimorder)
end


"""
    sliced(A::AbstractArray{T,2})
    sliced(A::AbstractArray{T,M+N}, Val(M))
    sliced(A::AbstractArray{T,M+N}, M::Integer)

Return a sliced view of `A`, using the columns or the first `M`
dimensions as inner dimensions.

With StaticArrays loaded, `sliced(A, SVector{S})` returns a reinterpreted
array with `SVector{S}` elements instead of an array of views.
"""
function sliced end
export sliced

@inline sliced(A::AbstractArray, M::Integer) = sliced(A, Val(M))
@inline sliced(A::AbstractArray{T,L}, ::Val{M}) where {T,L,M} = splitup(A, SplitSlices{M,L-M}())
@inline sliced(A::AbstractArray{T,2}) where {T} = sliced(A, Val(1))


"""
    innersize(A::AbstractArray{<:AbstractArray}, [dim])

Returns the size of the element arrays of `A`. Fails if the element arrays
are not of equal size.
"""
function innersize end
export innersize

@inline innersize(A::AbstractArray{<:AbstractArray}, dim::Integer) = innersize(A)[dim]

innersize(::AbstractArray{<:Number}) = ()

function innersize(A::AbstractArray{<:AbstractArray{T,M},N}) where {T,M,N}
    s = if !isempty(A)
        let sz_A = size(first(A))
            ntuple(i -> Int(sz_A[i]), Val(M))
        end
    else
        ntuple(_ -> zero(Int), Val(M))
    end

    let s = s
        if any(X -> size(X) != s, A)
            throw(DimensionMismatch("Shape of element arrays of A is not equal, can't determine common shape"))
        end
    end

    return s
end

@inline innersize(A::AbstractSlices{<:AbstractArray{T,M},N}) where {T,M,N} = getinnerdims(size(fused(A)), getsplitmode(A))


"""
    innersizes(A::AbstractArray{<:AbstractArray{T,M}})

Returns the sizes of the element arrays of `A`, as an array of `Dims{M}`
shaped like `A`. In contrast to [`innersize`](@ref), the element arrays do
not need to be of equal size.
"""
function innersizes end
export innersizes

innersizes(A::AbstractArray{<:AbstractArray}) = map(size, A)

innersizes(A::AbstractSlices{<:AbstractArray{T,M},N}) where {T,M,N} = fill(innersize(A), size(A))


"""
    innerlengths(A::AbstractArray{<:AbstractArray})

Returns the lengths of the element arrays of `A`, as an array of `Int`
shaped like `A`.
"""
function innerlengths end
export innerlengths

innerlengths(A::AbstractArray{<:AbstractArray}) = map(length, A)

innerlengths(A::AbstractSlices{<:AbstractArray{T,M},N}) where {T,M,N} = fill(prod(innersize(A)), size(A))


"""
    innermap(f, A::AbstractArray)
    innermap(f, A::AbstractArray{<:AbstractArray})

Nested `map` at depth 2. Equivalent to `map(X -> map(f, X), A)` for arrays
of arrays, otherwise equivalent to `Base.map`.
"""
function innermap end
export innermap

innermap(f, A::AbstractArray) = map(f, A)
innermap(f, A::AbstractArray{<:AbstractArray}) = map(Base.Fix1(map, f), A)
innermap(f, A::AbstractSlices{<:AbstractArray}) = _generic_innermap_impl(f, A)

function _generic_innermap_impl(f, A::AbstractArray)
    joined_A = fused(A)
    mapped_joined_A = map(f, joined_A)
    mapped_A = splitup(mapped_joined_A, getsplitmode(A))
    return mapped_A
end


"""
    abstract type AbstractPartMode{M,N} <: AbstractSplitMode

Abstract supertype for array partition modes with `M` inner dimensions and `N`
outer dimensions.

The mode need not represent a true partition, a partition that discards part
of the original array is allowed. The elements of the partition may also
be reshaped, depending on the mode.

Use `getsplitmode` to get the split mode of a split array.
"""
abstract type AbstractPartMode{M,N} <: AbstractSplitMode end
export AbstractPartMode


"""
    vecflattened(A::AbstractArray{T})::AbstractVector{T}
    vecflattened(A::AbstractArray{<:AbstractArray})::AbstractVector{T}

Concatenate nested arrays into a single vector, return non-nested vectors
unchanged.

If `A` is a nested view of a vector, `vecflattened(A)` should return the
underlying vector in a zero-copy O(1) fashion. So in contrast to
`reduce(vcat, A)` and `mapreduce(vec, vcat, A)`, the result may share
memory with `A`.

# Implementation

The default implementations are

```julia
vecflattened(A::AbstractVector) = A
vecflattened(A::AbstractArray) = vec(A)
vecflattened(A::AbstractVector{<:AbstractVector}) = reduce(vcat, A)
vecflattened(A::AbstractArray{<:AbstractArray}) = mapreduce(vec, vcat, A)
```

Specialize `vecflattened` for custom nested array types that can provide a
zero-copy implementation.
"""
function vecflattened end
export vecflattened

@inline vecflattened(A::AbstractVector) = A
@inline vecflattened(A::AbstractArray) = vec(A)
@inline vecflattened(A::AbstractVector{<:AbstractVector}) = reduce(vcat, A)
@inline vecflattened(A::AbstractVector{<:AbstractArray}) = mapreduce(vec, vcat, A)
@inline vecflattened(A::AbstractArray{<:AbstractVector}) = mapreduce(vec, vcat, A)
@inline vecflattened(A::AbstractArray{<:AbstractArray}) = mapreduce(vec, vcat, A)


"""
    deepmap(f, A::AbstractArray)
    deepmap(f, A::AbstractArray{<:AbstractArray{<:...}})

Applies `map` at the deepest layer of nested arrays. If `A` is not
a nested array, `deepmap` behaves identically to `Base.map`.
"""
function deepmap end
export deepmap

deepmap(f, A::AbstractArray) = map(f, A)
deepmap(f, A::AbstractArray{<:AbstractArray}) = map(Base.Fix1(deepmap, f), A)
deepmap(f, A::AbstractSlices{<:AbstractArray}) = _generic_deepmap_impl(f, A)

function _generic_deepmap_impl(f, A::AbstractArray)
    joined_A = fused(A)
    mapped_joined_A = deepmap(f, joined_A)
    mapped_A = splitup(mapped_joined_A, getsplitmode(A))
    return mapped_A
end


"""
    mapat(f, ::Val{depth}, As::AbstractArray...)

Nested `map` at nesting depth `depth`: apply `f` elementwise to the objects
at depth `depth` of the nested arrays `As`. Depth 1 refers to the elements
of the arrays themselves, so `mapat(f, Val(1), As...)` is equivalent to
`map(f, As...)` and `mapat(f, Val(2), A)` is equivalent to
[`innermap`](@ref)`(f, A)`. If `depth` exceeds the nesting depth of the
arrays, `f` is applied to the innermost elements, like [`deepmap`](@ref).

All of `As` must have the same nesting structure down to `depth`, their
split modes (see [`getsplitmode`](@ref)) must be equal on each nesting
level.

For split arrays (like [`ArrayOfSimilarArrays`](@ref) and
[`VectorOfArrays`](@ref)) `mapat` operates on the underlying flat data
directly, in a single (GPU-compatible) operation per nesting level.

See also [`bcastat`](@ref) for broadcast semantics with arguments of
different nesting depth.
"""
function mapat end
export mapat

@inline mapat(f, ::Val{1}, As::Vararg{AbstractArray,NA}) where {NA} = map(f, As...)

function mapat(f, ::Val{depth}, As::Vararg{AbstractArray,NA}) where {depth,NA}
    depth isa Integer && depth >= 1 || throw(ArgumentError("mapat depth must be a positive integer"))
    smodes = map(getsplitmode, As)
    smode = first(smodes)
    if smode isa UnknownSplitMode
        all(m -> m isa UnknownSplitMode, smodes) || throw(DimensionMismatch("mapat requires arrays with equal nesting structure, but split modes differ"))
        return map((xs...) -> mapat(f, Val(depth - 1), xs...), As...)
    else
        all(isequal(smode), smodes) || throw(DimensionMismatch("mapat requires arrays with equal nesting structure, but split modes differ"))
        return splitup(mapat(f, Val(depth - 1), map(fused, As)...), smode)
    end
end


"""
    bcastat(f, ::Val{depth}, args...)

Broadcast `f` over the contents of nested arrays at nesting depth `depth`,
with AwkwardArrays-like alignment, but with array-of-arrays nesting
semantics:

* Nested (split) arrays with equal split modes are aligned at depth `depth`.
* Arrays that match the outer structure of a shallower nesting level
  contribute one value per element of that level, broadcast over everything
  below it.
* Scalars and `Ref`s broadcast over everything.

`bcastat(f, Val(1), args...)` is equivalent to `broadcast(f, args...)`, and
like [`mapat`](@ref), a `depth` that exceeds the nesting depth of the
arguments applies `f` at the innermost level.

Nested array arguments must be split arrays (like
[`ArrayOfSimilarArrays`](@ref) and [`VectorOfArrays`](@ref)): for these,
`bcastat` compiles to a single flat (GPU-compatible) broadcast per nesting
level.
"""
function bcastat end
export bcastat

@inline bcastat(f, ::Val{1}, args...) = broadcast(f, args...)

function bcastat(f, ::Val{depth}, args...) where {depth}
    depth isa Integer && depth >= 1 || throw(ArgumentError("bcastat depth must be a positive integer"))
    r = _bcast_ref(args...)
    if r === nothing
        # No nested arguments left, apply at the innermost level:
        foreach(_require_known_mode, args)
        return broadcast(f, args...)
    else
        smode, ref_flat = r
        descended = map(arg -> _bcast_descend(arg, smode, ref_flat), args)
        return splitup(bcastat(f, Val(depth - 1), descended...), smode)
    end
end

function _bcast_ref(args...)
    for a in args
        if a isa AbstractArray
            m = getsplitmode(a)
            (m isa AbstractSlicingMode || m isa AbstractPartMode) && return (m, fused(a))
        end
    end
    return nothing
end

_require_known_mode(@nospecialize(x)) = nothing

function _require_known_mode(x::AbstractArray)
    getsplitmode(x) isa UnknownSplitMode && throw(ArgumentError("bcastat requires nested array arguments to be split arrays with a known split mode, like VectorOfArrays or ArrayOfSimilarArrays"))
    nothing
end

function _bcast_descend(x, smode::AbstractSplitMode, ref_flat::AbstractArray)
    x isa AbstractArray || return x
    ndims(x) == 0 && return x
    m = getsplitmode(x)
    if m isa UnknownSplitMode
        throw(ArgumentError("bcastat requires nested array arguments to be split arrays with a known split mode, like VectorOfArrays or ArrayOfSimilarArrays"))
    elseif m isa NonSplitMode
        return _bcast_expand(x, smode, ref_flat)
    else
        isequal(m, smode) || throw(DimensionMismatch("bcastat requires nested array arguments with equal split modes"))
        return fused(x)
    end
end

function _bcast_expand(x::AbstractArray, smode::AbstractSlicingMode{M,N}, ref_flat::AbstractArray) where {M,N}
    if axes(x) == getouterdims(axes(ref_flat), smode)
        is_memordered_splitmode(smode) || throw(ArgumentError("bcastat does not support outer-value broadcasting for slicings that are not in memory order"))
        return reshape(x, ntuple(_ -> 1, Val(M))..., size(x)...)
    elseif axes(x) == axes(ref_flat)
        return x
    else
        throw(DimensionMismatch("bcastat argument shape matches neither the outer structure nor the flat data of the nested arguments"))
    end
end

function _bcast_expand(x::AbstractArray, smode::AbstractPartMode, ref_flat::AbstractArray)
    throw(ArgumentError("bcastat does not support outer-value broadcasting for split mode $(nameof(typeof(smode)))"))
end


"""
    partitioned(A::AbstractVector, lengths::AbstractVector{<:Integer})
    partitioned(A::AbstractVector, shapes::AbstractVector{<:Dims})

Return a partitioned view of `A`, as a vector of arrays.

The parts are consecutive, non-overlapping views of `A`, with sizes given by
`lengths` (resulting in a vector of vectors) or `shapes` (resulting in a
vector of arrays).
"""
function partitioned end
export partitioned

# partitioned methods are defined in vector_of_arrays.jl.

# ToDo: Add partitioned(A::AbstractVector, n::Integer) and partitioned(A::AbstractVector, shape::Dims) ?
