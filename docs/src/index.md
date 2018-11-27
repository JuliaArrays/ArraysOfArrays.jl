# ArraysOfArrays.jl

A Julia package for efficient storage and handling of nested arrays. ArraysOfArrays provides two different types of nested arrays: [`ArrayOfSimilarArrays`](@ref section_ArrayOfSimilarArrays) and [`VectorOfArrays`](@ref section_VectorOfArrays).

This package also defines and exports the following new functions applicable to nested arrays in general:

* [`nestedview`](@ref) and [`flatview`](@ref) switch between a flat and a nested view of the same data.
* [`innersize`](@ref) returns the size of the elements of an array, provided they all have equal size.
* [`deepgetindex`](@ref), [`deepsetindex!`](@ref) and [`deepview`](@ref) provide index-based access across multiple layers of nested arrays
* [`innermap`](@ref) and [`deepmap`](@ref) apply a function to the elements of the inner (resp. innermost) arrays.
* [`abstract_nestedarray_type`](@ref) returns the type of nested `AbstractArray`s for a given innermost element type with multiple layers of nesting.
* [`consgroupedview`](@ref) computes a grouping of equal consecutive elements on a vector and applies it to another vector or (named or unnamed) tuple of vectors.


## [ArrayOfSimilarArrays](@id section_ArrayOfSimilarArrays)

An `ArrayOfSimilarArrays` offers a duality of view between representing the same data as both a flat multi-dimensional array and as an array of equally-sized arrays:

```julia
A_flat = rand(2,3,4,5,6)
A_nested = nestedview(A_flat, 2)
```

creates a view of `A_flat` as an array of arrays:

```julia
A_nested isa AbstractArray{<:AbstractArray{T,2},3} where T
```

`A_flat` is always available via [`flatview`](@ref). `A_flat` and `A_nested` are backed by the same data, no data is copied:

```julia
flatview(A_nested) === A_flat
```

Calling `getindex` on `A_nested` returns a view into `A_flat`:

```julia
fill!(A_nested[2, 4, 3], 4.2)
all(x -> x == 4.2, A_flat[:, :, 2, 4, 3])
```

### Type aliases

The following type aliases are defined:

* `VectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}`
* `ArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}`
* `VectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}`

For each of the types there is also an abstract type (`AbstractArrayOfSimilarArrays`, etc.).

If a `VectorOfSimilarArrays` is backed by an `ElasticArrays.ElasticArray`, additional element arrays can be pushed into it and `resize!` is available too:

### Appending data and resizing

```julia
using ElasticArrays

A_nested = nestedview(ElasticArray{Float64}(undef, 2, 3, 0), 2)

for i in 1:4
    push!(A_nested, rand(2, 3))
end
size(flatview(A_nested)) == (2, 3, 4)

resize!(A_nested, 6)
size(flatview(A_nested)) == (2, 3, 6)
```

There is a full duality between the nested and the flat view of the data. `A_flat` may be resized freely without breaking the inner consistency of `A_nested`: Changes in the shape of one will result in changes in the shape of the other.

### Statistics functions

`AbstractVectorOfSimilarArrays` supports the functions `sum`, `mean` and `var`, `AbstractVectorOfSimilarVectors` additionally support `cov` and `cor`.

Methods for these function are defined both without and with weights (via `StatsBase.AbstractWeights`). Because of this, `ArraysOfArrays` currently requires `StatsBase`. It's possible that this requirement can be dropped in the future, though (see
[Julia issue #29974](https://github.com/JuliaLang/julia/issues/29974)).

## [VectorOfArrays](@id section_VectorOfArrays)

A `VectorOfArrays` represents a vector of arrays of equal dimensionality but different size. It is a nested interpretation of the concept of a "ragged array".

```julia
VA = VectorOfArrays{Float64, 2}()

push!(VA, rand(2, 3))
push!(VA, rand(4, 2))

size(VA[1]) == (2,3)
size(VA[2]) == (4,2)
```

Internally, all data is stored efficiently in a single, flat and memory-contiguous vector, accessible via `flatview`:

```julia
VA_flat = flatview(VA)
VA_flat isa Vector{Float64}
```

Calling `getindex` on `A_nested` returns a view into `A_flat`:

```julia
VA_flat = flatview(VA)
view(VA_flat, 7:14) == vec(VA[2])

fill!(view(VA_flat, 7:14), 2.4)
all(x -> x == 2.4, VA[2])

fill!(view(VA_flat, 7:14), 4.2)
all(x -> x == 4.2, VA[2])
```

### Type aliases
The following type aliases are defined:

* `VectorOfVectors{T,VT,VI,VD} = VectorOfArrays{T,1,VT,VI,VD}`

### Appending data and resizing

A `VectorOfArrays` is grown by appending data to it. `resize!` can be used to shrink it, but not to grow it (the size of the additional element arrays would be unknown):

```julia
length(resize!(VA, 1)) == 1
```

but

```julia
resize!(VA, 4)
```

will fail.

Note: The vector returned by `flatview(VA)` *must not* be resized directly, doing so would break the internal consistency of `VA`.


## Allocation free element access

Element access via `getindex` returns (possibly reshaped) instances of `SubArray` for both `ArrayOfSimilarArrays` and `VectorOfArrays`. Usually this is not a problem, but frequent allocation of a large number of views can become a limiting factor in multi-threaded applications.

Both types support `UnsafeArrays.@uviews` for allocation-free getindex:

```julia
using UnsafeArrays

A = nestedview(rand(2,3,4,5), 2)

isbits(A[2,2]) == false

@uviews A begin
    isbits(A[2,2]) == true
end
```

As always, `UnsafeArray`s should be used with great care: The pointer-based bitstype
views *must not* be allowed to escape the `@uviews` scope, and internal data of `A` *must not* be reallocated (e.g. due to a `push!` or `append!` on `A`) while the `@uviews` scope is active.
