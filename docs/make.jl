using Policies
using Documenter

makedocs(;
    modules=[Policies],
    authors="Scott Jordan",
    repo="https://github.com/DecisionMakingAI/Policies.jl/blob/{commit}{path}#L{line}",
    sitename="Policies.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DecisionMakingAI.github.io/Policies.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DecisionMakingAI/Policies.jl",
)
