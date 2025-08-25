# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractSplitMode

Abstract supertype for array split modes.

Use [`getsplitmode`](@ref) to get the split mode of an array.

See also [`splitview`](@ref) and [`joinedview`](@ref).
"""
abstract type AbstractSplitMode end
export AbstractSplitMode


"""
    struct NonSplitMode{N} <: AbstractSplitMode

The split mode of unsplit collections that have `N` dimensions.

Constructor: `NonSplitMode{N}()`
"""
struct NonSplitMode{N} <: AbstractSplitMode end
export NonSplitMode


"""
    struct UnknownSplitMode{AT} <: AbstractSplitMode

Split mode of generic split objects of type{T} that have been split in an
unknown way, e.g. nested arrays of type `Array{<:Array}`.

Since the split parts may be (typically are) non-contiguous in memory, this
split mode not allow for `unsplitview` or `flatview`. It is also not
inferrable of the split object should be interpeted as a sliced array,
a ragged array, or something else.

Constructor: `UnknownSplitMode{T}()`
"""
struct UnknownSplitMode{AT} <: AbstractSplitMode end
export UnknownSplitMode


"""
    abstract type AbstractSlicingMode{M,N} <: AbstractSplitMode

Abstract supertype for array slicing modes with `M` inner dimensions and `N`
outer dimensions.

Use `getsplitmode` to get the split mode of an split array.
"""
abstract type AbstractSlicingMode{M,N} <: AbstractSplitMode end
export AbstractSlicingMode



"""
    getsplitmode(A::AbstractArray)::NonSplitMode
    getsplitmode(A::AbstractArray{<:AbstractArray})::AbstractSplitMode

Get the split mode of `A`.

`splitview(joinedview(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible, except if `getsplitmode(A)` is an
`UnknownSplitMode`.

`getsplitmode` should be a zero-copy O(1) operation, if at all possible.
"""
function getsplitmode end
export getsplitmode

@inline getsplitmode(::T) where T = UnknownSplitMode{T}()

@inline getsplitmode(::AbstractArray{<:Any,N}) where N = NonSplitMode{N}()

@inline getsplitmode(A::AbstractArray{<:AbstractArray}) = UnknownSplitMode{typeof(A)}()


"""
    is_memordered_splitmode(smode::AbstractSplitMode)::Bool

Check if `smode` splits in memory-order.
    
If true, inner arrays are stored contiguously in memory in Julia-native
dimension order, and the same is true for the outer dimensions (no dimention
reordering).

If true, `flatview` and `joinedview` are equivalent.
"""
function is_memordered_splitmode end
export is_memordered_splitmode

is_memordered_splitmode(::NonSplitMode) = true
is_memordered_splitmode(::UnknownSplitMode) = false


"""
    ArraysOfArrays.getinnerdims(tpl::Tuple, smode::AbstractSlicing)

Get the entries of `tpl` corresponding to the inner dimensions of slicing
mode `smode`, in the order specified by `smode`.
"""
function getinnerdims end

@inline getinnerdims(::Tuple, ::NonSplitMode) = ()

function getinnerdims(::Tuple, ::UnknownSplitMode)
    throw(ArgumentError("getinnerdims cannot be used with UnknownSplitMode"))
end


"""
    ArraysOfArrays.getouterdims(tpl::Tuple, smode::AbstractSlicing)

Get the entries of `tpl` corresponding to the outer dimensions of slicing
mode `smode`, in the order specified by `smode`.
"""
function getouterdims end

@inline getouterdims(x::Tuple, ::NonSplitMode) = x

function getouterdims(::Tuple, ::UnknownSplitMode)
    throw(ArgumentError("getouterdims cannot be used with UnknownSplitMode"))
end


"""
    splitview(A::AbstractArray, smode::AbstractSplitMode)

View array `A` in split form, as an array of arrays.

`splitview` should be a zero-copy O(1) operation, if at all possible.

See also [`joinedview`](@ref) and [`getsplitmode`](@ref).
"""
function splitview end
export splitview

@inline splitview(obj::Any, ::NonSplitMode) = obj

function splitview(::Any, ::UnknownSplitMode)
    throw(ArgumentError("splitview cannot be used with UnknownSplitMode"))
end


"""
    joinedview(A::AbstractArray)
    joinedview(A::AbstractArray{<:AbstractArray})

View array `A` in unsplit form.

`splitview(joinedview(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible, except if `getsplitmode(A)` is an
`UnknownSplitMode`.

If `A` is not a nested array return `A` itself. If `A` is a split array,
return the original unsplit array.

If `is_memordered_splitmode(getsplitmode(A))` is true, `joinedview(A)` is
equivalent to [`flatview(A)`](@ref).

`joinedview` should be a zero-copy O(1) operation, if at all possible.
"""
function joinedview end
export joinedview

@inline joinedview(obj) = _joinedview_impl(obj, getsplitmode(obj))

@inline joinedview(A::AbstractArray) = A

@inline joinedview(A::AbstractArray{<:AbstractArray}) = _joinedview_impl(A, getsplitmode(A))

@inline _joinedview_impl(obj, ::NonSplitMode) = obj

function _joinedview_impl(@nospecialize(obj), ::UnknownSplitMode)
    throw(ArgumentError("joinedview not implemented for objects of type $(nameof(typeof(obj))) with unknown split mode"))
end


"""
    flatview(A::AbstractArray)
    flatview(A::AbstractArray{<:AbstractArray})

View array `A` in a flattened form, with inner dimensions first. The shape of
the flattened form will depend on the type of `A`. If the `A` is not a
nested array, the return value is `A` itself. Only specific types of nested
arrays are supported.

If `is_memordered_splitmode(getsplitmode(A))` is true, `flatview(A)` is
equivalent to [`joinedview(A)`](@ref).

The result of `flatview(A)` will equal either `stack(A)`
(resp. [`stacked(A)`](@ref)) or `reduce(vcat, A)`, depending on the type of
`A` (sliced-array-like or ragged-array-like).

`flatview` should be a zero-copy O(1) operation, if at all possible.
"""
function flatview end
export flatview

@inline flatview(A::AbstractArray) = A
function flatview(A::AbstractArray{<:AbstractArray})
    throw(ArgumentError("flatview not implemented for nested arrays of type $(nameof(typeof(A)))"))
end

function flatview(A::AbstractSlices)
    smode = getsplitmode(A)
    if is_memordered_splitmode(smode)
        return joinedview(A)
    else
        throw(ArgumentError("flatview required memory-ordered split/slicing, but array has split mode $smode"))
    end
end

"""
    stacked(A::AbstractArray)
    stacked(A::AbstractArray{<:AbstractArray})

Join stacked arrays of a nested array into a single array along one or more
new dimensions, return non-nested arrays unchanged.

Similar to `Base.stack`, but can return the original underlying array of
sliced arrays in more cases.
"""
function stacked end
export stacked

@inline stacked(A::AbstractArray) = A
@inline stacked(A::AbstractArray{<:AbstractArray}) = stack(A)


"""
    innersize(A::AbstractArray{<:AbstractArray}, [dim])

Returns the size of the element arrays of `A`. Fails if the element arrays
are not of equal size.
"""
function innersize end
export innersize

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


@inline innersize(A::AbstractArray{<:AbstractArray}, dim::Integer) =
    innersize(A)[dim]


"""
    innermap(f, A::AbstractArray)
    innermap(f, A::AbstractArray{<:AbstractArray})

Nested `map` at depth 2. Equivalent to `map(X -> map(f, X) A)` for arrays
of arrays, otherwise equivalent to `Base.map`.
"""
function innermap end
export innermap

innermap(f, obj) = _generic_innermap_impl(f, obj)
innermap(f, A::AbstractArray) = map(f, A)
innermap(f, A::AbstractArray{<:AbstractArray}) = map(Base.Fix1(map, f), A)
innermap(f, A::AbstractSlices) = _generic_innermap_impl(f, A)

function _generic_innermap_impl(f, obj)
    joined_obj = joinedview(obj)
    mapped_joined_obj = map(f, joined_obj)
    mapped_obj = splitview(mapped_joined_obj, getsplitmode(obj))
    return mapped_obj
end


"""
    deepmap(f, A::AbstractArray)
    deepmap(f, A::AbstractArray{<:AbstractArray{<:...}})

Applies `map` at the deepest layer of nested arrays. If `A` is not
a nested array, `deepmap` behaves identical to `Base.map`.
"""
function deepmap end
export deepmap

deepmap(f, obj) = _generic_deepmap_impl(f, obj)
deepmap(f, A::AbstractArray) = map(f, A)
deepmap(f, A::AbstractArray{<:AbstractArray}) = map(Base.Fix1(deepmap, f), A)
deepmap(f, A::AbstractSlices) = _generic_deepmap_impl(f, A)

function _generic_deepmap_impl(f, obj)
    joined_obj = joinedview(obj)
    mapped_joined_obj = deepmap(f, joined_obj)
    mapped_obj = splitview(mapped_joined_obj, getsplitmode(obj))
    return mapped_obj
end
