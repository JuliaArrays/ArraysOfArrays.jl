# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using ArraysOfArrays

# Doctest setup
DocMeta.setdocmeta!(
    ArraysOfArrays,
    :DocTestSetup,
    :(using ArraysOfArrays);
    recursive=true,
)

makedocs(
    sitename = "ArraysOfArrays",
    modules = [ArraysOfArrays],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://JuliaArrays.github.io/ArraysOfArrays.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/JuliaArrays/ArraysOfArrays.jl.git",
    forcepush = true,
    push_preview = true,
)
