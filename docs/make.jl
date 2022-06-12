using Documenter
using DynamicFactorModeling

makedocs(
    sitename = "DynamicFactorModeling",
    format = Documenter.HTML(),
    modules = [DynamicFactorModeling]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/gionikola/DynamicFactorModeling.jl.git"
)
