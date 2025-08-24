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
    struct NonSplitMode <: AbstractSplitMode

The split mode of unsplit arrays.

Constructor: `NonSplitMode()`
"""
struct NonSplitMode <: AbstractSplitMode end
export NonSplitMode


"""
    abstract type AbstractSlicingMode <: AbstractSplitMode

Abstract supertype for array split modes.

Use `getsplitmode` to get the split mode of an split array.
"""
abstract type AbstractSlicingMode <: AbstractSplitMode end
export AbstractSlicingMode



"""
    getsplitmode(A::AbstractArray)::NonSplitMode
    getsplitmode(A::AbstractArray{<:AbstractArray})::AbstractSplitMode

Get the split mode of `A`.

`splitview(joinedview(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible.

`getsplitmode` should be a zero-copy O(1) operation, if at all possible.
"""
function getsplitmode end

@inline getsplitmode(::AbstractArray) = NonSplitMode()

function getsplitmode(A::AbstractArray{<:AbstractArray})
    throw(ArgumentError("getsplitmode not implemented for nested arrays of type $(nameof(typeof(A)))"))
end


"""
    is_memordered_splitmode(smode::AbstractSplitMode)::Bool

Check if `smode` splits in memory-order.
    
If true, inner arrays are stored contiguously in memory in Julia-native
dimension order, and the same is true for the outer dimensions (no dimention
reordering).

If true, `flatview` and `joinedview` are equivalent.
"""
function is_memordered_splitmode end

is_memordered_splitmode(::NonSplitMode) = true


"""
    splitview(A::AbstractArray, smode::AbstractSplitMode)

View array `A` in split form, as an array of arrays.

`splitview` should be a zero-copy O(1) operation, if at all possible.

See also [`joinedview`](@ref) and [`getsplitmode`](@ref).
"""
function splitview end
export splitview

@inline splitview(A::AbstractArray, ::NonSplitMode) = A


"""
    joinedview(A::AbstractArray)
    joinedview(A::AbstractArray{<:AbstractArray{<:...}})

View array `A` in unsplit form.

`splitview(joinedview(A), getsplitmode(A))` must equal `A`, and should have
the same type as `A` if at all possible.

If `A` is not a nested array return `A` itself. If `A` is a split array,
return the original unsplit array.

If `is_memordered_splitmode(getsplitmode(A))` is true, `joinedview(A)` is
equivalent to [`flatview(A)`](@ref).

`joinedview` should be a zero-copy O(1) operation, if at all possible.
"""
function joinedview end
export joinedview

@inline joinedview(A::AbstractArray) = A

function joinedview(A::AbstractArray{<:AbstractArray})
    throw(ArgumentError("joinedview not implemented for nested arrays of type $(nameof(typeof(A)))"))
end


"""
    flatview(A::AbstractArray)
    flatview(A::AbstractArray{<:AbstractArray{<:...}})

View array `A` in a flattened form, with inner dimensions first. The shape of
the flattened form will depend on the type of `A`. If the `A` is not a
nested array, the return value is `A` itself. Only specific types of nested
arrays are supported.

If `is_memordered_splitmode(getsplitmode(A))` is true, `flatview(A)` is
equivalent to [`joinedview(A)`](@ref).

The result of `flatview(A)` will equal either `stack(A)`
(resp. [`stacked(A)`](@ref)) or `reduce(vcat, A)`, depending on the type of
`A` (sliced-array-like or ragged-array-like).

`joinedview` should be a zero-copy O(1) operation, if at all possible.
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
    nestedview(A::AbstractArray{T,M+N}, M::Integer)
    nestedview(A::AbstractArray{T,2})

AbstractArray{<:AbstractArray{T,M},N}

View array `A` in as an `N`-dimensional array of `M`-dimensional arrays by
wrapping it into an [`ArrayOfSimilarArrays`](@ref).

It's also possible to use a `StaticVector` of length `S` as the type of the
inner arrays via

    nestedview(A::AbstractArray{T}, ::Type{StaticArrays.SVector{S}})
    nestedview(A::AbstractArray{T}, ::Type{StaticArrays.SVector{S,T}})
"""
function nestedview end
export nestedview

@inline nestedview(A::AbstractArray{T,L}, M::Integer) where {T,L} =
    ArrayOfSimilarArrays{T,M}(A)

@inline nestedview(A::AbstractArray{T,L}, ::Val{M}) where {T,L,M} =
    ArrayOfSimilarArrays{T,M}(A)

@inline nestedview(A::AbstractArray{T,2}) where {T} =
    VectorOfSimilarVectors{T}(A)


"""
    innersize(A:AbstractArray{<:AbstractArray}, [dim])

Returns the size of the element arrays of `A`. Fails if the element arrays
are not of equal size.
"""
function innersize end
export innersize

innersize(::AbstractArray) = ()

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
    deepmap(f::Base.Callable, x::Any)
    deepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray{<:...}})

Applies `map` at the deepest possible layer of nested arrays. If `A` is not
a nested array, `deepmap` behaves identical to `Base.map`.
"""
function deepmap end
export deepmap

deepmap(f::Base.Callable, x::Any) = map(f, x)

deepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray}) = map(X -> deepmap(f, X), A)
