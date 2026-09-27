var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#ArraysOfArrays.jl-1",
    "page": "Home",
    "title": "ArraysOfArrays.jl",
    "category": "section",
    "text": "A Julia package for efficient storage and handling of nested arrays. ArraysOfArrays provides two different types of nested arrays: ArrayOfSimilarArrays and VectorOfArrays."
},

{
    "location": "#section_ArrayOfSimilarArrays-1",
    "page": "Home",
    "title": "ArrayOfSimilarArrays",
    "category": "section",
    "text": "An ArrayOfSimilarArrays offers a duality of view between representing the same data as both a flat multi-dimensional array and as an array of equally-sized arrays:A_flat = rand(2,3,4,5,6)\nA_nested = nestedview(A_flat, 2)creates a view of A_flat as an array of arrays:A_nested isa AbstractArray{<:AbstractArray{T,2},3} where TA_flat is always available via flatview. A_flat and A_nested are backed by the same data, no data is copied:flatview(A_nested) === A_flatCalling getindex on A_nested returns a view into A_flat:fill!(A_nested[2, 4, 3], 4.2)\nall(x -> x == 4.2, A_flat[:, :, 2, 4, 3])"
},

{
    "location": "#Type-aliases-1",
    "page": "Home",
    "title": "Type aliases",
    "category": "section",
    "text": "The following type aliases are defined:VectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}\nArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}\nVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}For each of the types there is also an abstract type (AbstractArrayOfSimilarArrays, etc.).If a VectorOfSimilarArrays is backed by an ElasticArrays.ElasticArray, additional element arrays can be pushed into it and resize! is available too:"
},

{
    "location": "#Appending-data-and-resizing-1",
    "page": "Home",
    "title": "Appending data and resizing",
    "category": "section",
    "text": "using ElasticArrays\n\nA_nested = nestedview(ElasticArray{Float64}(undef, 2, 3, 0), 2)\n\nfor i in 1:4\n    push!(A_nested, rand(2, 3))\nend\nsize(flatview(A_nested)) == (2, 3, 4)\n\nresize!(A_nested, 6)\nsize(flatview(A_nested)) == (2, 3, 6)There is a full duality between the nested and the flat view of the data. A_flat may be resized freely without breaking the inner consistency of A_nested: Changes in the shape of one will result in changes in the shape of the other."
},

{
    "location": "#Statistics-functions-1",
    "page": "Home",
    "title": "Statistics functions",
    "category": "section",
    "text": "AbstractVectorOfSimilarArrays supports the functions sum, mean and var, AbstractVectorOfSimilarVectors additionally support cov and cor.Methods for these function are defined both without and with weights (via StatsBase.AbstractWeights). Because of this, ArraysOfArrays currently requires StatsBase. It\'s possible that this requirement can be dropped in the future, though (see Julia issue #29974)."
},

{
    "location": "#section_VectorOfArrays-1",
    "page": "Home",
    "title": "VectorOfArrays",
    "category": "section",
    "text": "A VectorOfArrays represents a vector of arrays of equal dimensionality but different size. It is a nested interpretation of the concept of a \"ragged array\".VA = VectorOfArrays{Float64, 2}()\n\npush!(VA, rand(2, 3))\npush!(VA, rand(4, 2))\n\nsize(VA[1]) == (2,3)\nsize(VA[2]) == (4,2)Internally, all data is stored efficiently in a single, flat and memory-contiguous vector, accessible via flatview:VA_flat = flatview(VA)\nVA_flat isa Vector{Float64}Calling getindex on A_nested returns a view into A_flat:VA_flat = flatview(VA)\nview(VA_flat, 7:14) == vec(VA[2])\n\nfill!(view(VA_flat, 7:14), 2.4)\nall(x -> x == 2.4, VA[2])\n\nfill!(view(VA_flat, 7:14), 4.2)\nall(x -> x == 4.2, VA[2])"
},

{
    "location": "#Type-aliases-2",
    "page": "Home",
    "title": "Type aliases",
    "category": "section",
    "text": "The following type aliases are defined:VectorOfVectors{T,VT,VI,VD} = VectorOfArrays{T,1,VT,VI,VD}"
},

{
    "location": "#Appending-data-and-resizing-2",
    "page": "Home",
    "title": "Appending data and resizing",
    "category": "section",
    "text": "A VectorOfArrays is grown by appending data to it. resize! can be used to shrink it, but not to grow it (the size of the additional element arrays would be unknown):length(resize!(VA, 1)) == 1butresize!(VA, 4)will fail.Note: The vector returned by flatview(VA) must not be resized directly, doing so would break the internal consistency of VA."
},

{
    "location": "#Allocation-free-element-access-1",
    "page": "Home",
    "title": "Allocation free element access",
    "category": "section",
    "text": "Element access via getindex returns (possibly reshaped) instances of SubArray for both ArrayOfSimilarArrays and VectorOfArrays. Usually this is not a problem, but frequent allocation of a large number of views can become a limiting factor in multi-threaded applications.Both types support UnsafeArrays.@uviews for allocation-free getindex:using UnsafeArrays\n\nA = nestedview(rand(2,3,4,5), 2)\n\nisbits(A[2,2]) == false\n\n@uviews A begin\n    isbits(A[2,2]) == true\nendAs always, UnsafeArrays should be used with great care: The pointer-based bitstype views must not be allowed to escape the @uviews scope, and internal data of A must not be reallocated (e.g. due to a push! or append! on A) while the @uviews scope is active."
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "CurrentModule = ArraysOfArrays\nDocTestSetup  = quote\n    using ArraysOfArrays\nend"
},

{
    "location": "api/#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": "Order = [:type]"
},

{
    "location": "api/#Functions-1",
    "page": "API",
    "title": "Functions",
    "category": "section",
    "text": "Order = [:function]"
},

{
    "location": "api/#ArraysOfArrays.AbstractArrayOfSimilarArrays",
    "page": "API",
    "title": "ArraysOfArrays.AbstractArrayOfSimilarArrays",
    "category": "type",
    "text": "AbstractArrayOfSimilarArrays{T,M,N} <: AbstractArray{<:AbstractArray{T,M},N}\n\nAn array that contains arrays that have the same size/axes. The array is internally stored in flattened form as some kind of array of dimension M + N. The flattened form can be accessed via flatview(A).\n\nSubtypes must implement (in addition to typical array operations):\n\nflatview(A::SomeArrayOfSimilarArrays)::AbstractArray{T,M+N}\n\nThe following type aliases are defined:\n\nAbstractVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}\nAbstractArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}\nAbstractVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.ArrayOfSimilarArrays",
    "page": "API",
    "title": "ArraysOfArrays.ArrayOfSimilarArrays",
    "category": "type",
    "text": "ArrayOfSimilarArrays{T,M,N,L,P} <: AbstractArrayOfSimilarArrays{T,M,N}\n\nRepresents a view of an array of dimension L = M + N as an array of dimension M with elements that are arrays with dimension N. All element arrays implicitly have equal size/axes.\n\nConstructors:\n\nArrayOfSimilarArrays{T,M,N}(flat_data::AbstractArray)\nArrayOfSimilarArrays{T,M}(flat_data::AbstractArray)\n\nThe following type aliases are defined:\n\nVectorOfSimilarArrays{T,M} = AbstractArrayOfSimilarArrays{T,M,1}\nArrayOfSimilarVectors{T,N} = AbstractArrayOfSimilarArrays{T,1,N}\nVectorOfSimilarVectors{T} = AbstractArrayOfSimilarArrays{T,1,1}\n\nVectorOfSimilarArrays supports push!(), etc., provided the underlying array supports resizing of it\'s last dimension (e.g. an ElasticArray).\n\nThe nested array can also be created using the function nestedview and the wrapped flat array can be accessed using flatview afterwards:\n\nA_flat = rand(2,3,4,5,6)\nA_nested = nestedview(A_flat, 2)\nA_nested isa AbstractArray{<:AbstractArray{T,2},3} where T\nflatview(A_nested) === A_flat\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.VectorOfArrays",
    "page": "API",
    "title": "ArraysOfArrays.VectorOfArrays",
    "category": "type",
    "text": "VectorOfArrays{T,N,M} <: AbstractVector{<:AbstractArray{T,N}}\n\nAn VectorOfArrays represents a vector of N-dimensional arrays (that may differ in size). Internally, VectorOfArrays stores all elements of all arrays in a single flat vector. M must equal N - 1\n\nThe VectorOfArrays itself supports push!, unshift!, etc., but the size of each individual array in the vector is fixed. resize! can be used to shrink, but not to grow, as the size of the additional element arrays in the vector would be unknown. However, memory space for up to n arrays with a maximum size s can be reserved via sizehint!(A::VectorOfArrays, n, s::Dims{N}).\n\nConstructors:\n\nVectorOfArrays{T, N}()\n\nVectorOfArrays(\n    data::AbstractVector,\n    elem_ptr::AbstractVector{Int},\n    kernel_size::AbstractVector{<:Dims}\n    checks::Function = ArraysOfArrays.full_consistency_checks\n)\n\nOther suitable values for checks are ArraysOfArrays.simple_consistency_checks and ArraysOfArrays.no_consistency_checks.\n\nThe following type aliases are defined:\n\nVectorOfVectors{T,VT,VI,VD} = VectorOfArrays{T,1,VT,VI,VD}\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.deepmap",
    "page": "API",
    "title": "ArraysOfArrays.deepmap",
    "category": "function",
    "text": "deepmap(f::Base.Callable, x::Any)\ndeepmap(f::Base.Callable, A::AbstractArray{<:AbstractArray{<:...}})\n\nApplies map at the deepest possible layer of nested arrays. If A is not a nested array, deepmap behaves identical to Base.map.\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.flatview",
    "page": "API",
    "title": "ArraysOfArrays.flatview",
    "category": "function",
    "text": "flatview(A::AbstractArray)\nflatview(A::AbstractArray{<:AbstractArray{<:...}})\n\nView array A in a suitable flattened form. The shape of the flattened form will depend on the type of A. If the A is not a nested array, the return value is A itself. When no type-specific method is available, flatview will fall back to Base.Iterators.flatten.\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.flatview-Tuple{ArrayOfSimilarArrays}",
    "page": "API",
    "title": "ArraysOfArrays.flatview",
    "category": "method",
    "text": "flatview(A::ArrayOfSimilarArrays{T,M,N,L,P})::P\n\nReturns the array of dimensionality L = M + N wrapped by A. The shape of the result may be freely changed without breaking the inner consistency of A.\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.flatview-Tuple{VectorOfArrays}",
    "page": "API",
    "title": "ArraysOfArrays.flatview",
    "category": "method",
    "text": "flatview(A::VectorOfArrays{T})::Vector{T}\n\nReturns the internal serialized representation of all element arrays of A as a single vector. Do not change the length of the returned vector, as it would break the inner consistency of A.\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.innersize",
    "page": "API",
    "title": "ArraysOfArrays.innersize",
    "category": "function",
    "text": "innersize(A:AbstractArray{<:AbstractArray}, [dim])\n\nReturns the size of the element arrays of A. Fails if the element arrays are not of equal size.\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.nestedmap2",
    "page": "API",
    "title": "ArraysOfArrays.nestedmap2",
    "category": "function",
    "text": "nestedmap2(f::Base.Callable, A::AbstractArray{<:AbstractArray})\n\nNested map at depth 2. Equivalent to map(X -> map(f, X) A).\n\n\n\n\n\n"
},

{
    "location": "api/#ArraysOfArrays.nestedview",
    "page": "API",
    "title": "ArraysOfArrays.nestedview",
    "category": "function",
    "text": "nestedview(A::AbstractArray{T,M+N}, M::Integer)\nnestedview(A::AbstractArray{T,2})\n\nAbstractArray{<:AbstractArray{T,M},N}\n\nView array A in as an M-dimensional array of N-dimensional arrays by wrapping it into an ArrayOfSimilarArrays.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.sum-Union{Tuple{AbstractArrayOfSimilarArrays{T,M,1}}, Tuple{M}, Tuple{T}} where M where T",
    "page": "API",
    "title": "Base.sum",
    "category": "method",
    "text": "sum(X::AbstractVectorOfSimilarArrays)\nsum(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights)\n\nCompute the sum of the elements vectors of X. Equivalent to sum of flatview(X) along the last dimension.\n\n\n\n\n\n"
},

{
    "location": "api/#Statistics.cor-Tuple{AbstractArrayOfSimilarArrays{T,1,1} where T}",
    "page": "API",
    "title": "Statistics.cor",
    "category": "method",
    "text": "cor(X::AbstractVectorOfSimilarVectors)\ncor(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights)\n\nCompute the Pearson correlation matrix between the elements of the elements of  X along X. Equivalent to cor of flatview(X) along dimension 2.\n\n\n\n\n\n"
},

{
    "location": "api/#Statistics.cov-Tuple{AbstractArrayOfSimilarArrays{T,1,1} where T}",
    "page": "API",
    "title": "Statistics.cov",
    "category": "method",
    "text": "cov(X::AbstractVectorOfSimilarVectors; corrected::Bool = true)\ncov(X::AbstractVectorOfSimilarVectors, w::StatsBase.AbstractWeights; corrected::Bool = true)\n\nCompute the covariance matrix between the elements of the elements of X along X. Equivalent to cov of flatview(X) along dimension 2.\n\n\n\n\n\n"
},

{
    "location": "api/#Statistics.mean-Union{Tuple{AbstractArrayOfSimilarArrays{T,M,1}}, Tuple{M}, Tuple{T}} where M where T",
    "page": "API",
    "title": "Statistics.mean",
    "category": "method",
    "text": "mean(X::AbstractVectorOfSimilarArrays)\nmean(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights)\n\nCompute the mean of the elements vectors of X. Equivalent to mean of flatview(X) along the last dimension.\n\n\n\n\n\n"
},

{
    "location": "api/#Statistics.var-Union{Tuple{AbstractArrayOfSimilarArrays{T,M,1}}, Tuple{M}, Tuple{T}} where M where T",
    "page": "API",
    "title": "Statistics.var",
    "category": "method",
    "text": "var(X::AbstractVectorOfSimilarArrays; corrected::Bool = true)\nvar(X::AbstractVectorOfSimilarArrays, w::StatsBase.AbstractWeights; corrected::Bool = true)\n\nCompute the sample variance of the elements vectors of X. Equivalent to var of flatview(X) along the last dimension.\n\n\n\n\n\n"
},

{
    "location": "api/#Documentation-1",
    "page": "API",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [ArraysOfArrays]\nOrder = [:type, :function]"
},

{
    "location": "LICENSE/#",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "page",
    "text": ""
},

{
    "location": "LICENSE/#LICENSE-1",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "section",
    "text": "using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))"
},

]}
