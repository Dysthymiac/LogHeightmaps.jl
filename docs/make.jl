using LogHeightmaps
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(LogHeightmaps, :DocTestSetup, :(using LogHeightmaps); recursive=true)

makedocs(;
    modules=[LogHeightmaps],
    authors="Ted <tedzolotarev@gmail.com> and contributors",
    sitename="LogHeightmaps.jl",
    plugins=[bib],
    format=Documenter.HTML(;
        canonical="https://Dysthymiac.github.io/LogHeightmaps.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dysthymiac/LogHeightmaps.jl",
    devbranch="master",
)
