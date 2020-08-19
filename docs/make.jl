using DecisionMakingPolicies
using Documenter

makedocs(;
    modules=[DecisionMakingPolicies],
    authors="Scott Jordan",
    repo="https://github.com/DecisionMakingAI/DecisionMakingPolicies.jl/blob/{commit}{path}#L{line}",
    sitename="DecisionMakingPolicies.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DecisionMakingAI.github.io/DecisionMakingPolicies.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DecisionMakingAI/DecisionMakingPolicies.jl",
)
