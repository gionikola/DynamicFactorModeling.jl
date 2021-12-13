using Documenter
using DynamicFactorModeling

push!(LOAD_PATH, "../src/")
makedocs(
    sitename = "DynamicFactorModeling.jl Documentation",
    pages = ["Index" => "index.md", "Another page" => "anotherPage.md"],
    format = Documenter.HTML(prettyurls = false),
    modules = [DynamicFactorModeling]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/gionikola/DynamicFactorModeling.jl.git", 
    devbranch = "main" 
)
