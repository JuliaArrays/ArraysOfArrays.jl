# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using ArraysOfArrays

makedocs(
    sitename = "ArraysOfArrays",
    modules = [ArraysOfArrays],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/ArraysOfArrays.jl/stable/"
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/oschulz/ArraysOfArrays.jl.git",
    forcepush = true
)
