# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [fixdoctests]
#
# for local builds.

using Documenter
using ArraysOfArrays

makedocs(
    sitename = "ArraysOfArrays",
    modules = [ArraysOfArrays],
    format = :html,
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://oschulz.github.io/ArraysOfArrays.jl/stable/",
)

deploydocs(
    repo = "github.com/oschulz/ArraysOfArrays.jl.git"
)
