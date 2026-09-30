using Documenter
using DynamicFactorModeling

makedocs(
    sitename = "DynamicFactorModeling.jl",
    modules = [DynamicFactorModeling],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true",
                             edit_link = "main"),
    pages = [
        "Getting started" => "index.md",
        "Model and estimation" => "model.md",
        "Method audit" => "method_audit.md",
        "Checks and limitations" => "validation.md",
        "State-space tools" => "state_space.md",
        "API reference" => "api.md",
        "Upgrading" => "migration.md",
    ],
    checkdocs = :exports,
    doctest = true,
)

# A local documentation build never deploys. CI opts in explicitly.
if "--deploy" in ARGS
    deploydocs(repo = "github.com/gionikola/DynamicFactorModeling.jl.git",
               devbranch = "main", push_preview = false)
end
