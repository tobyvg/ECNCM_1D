using ECNCM_1D
using Documenter

DocMeta.setdocmeta!(ECNCM_1D, :DocTestSetup, :(using ECNCM_1D); recursive=true)

makedocs(;
    modules=[ECNCM_1D],
    authors="tobyvg <tobyvangastelen@gmail.com> and contributors",
    repo="https://github.com/tobyvg/ECNCM_1D.jl/blob/{commit}{path}#{line}",
    sitename="ECNCM_1D.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tobyvg.github.io/ECNCM_1D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tobyvg/ECNCM_1D.jl",
    devbranch="main",
)
