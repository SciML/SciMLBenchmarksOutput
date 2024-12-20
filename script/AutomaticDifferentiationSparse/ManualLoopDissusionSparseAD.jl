
using DifferentiationInterface
using DifferentiationInterfaceTest
using LinearAlgebra
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings
import Enzyme,ForwardDiff,Mooncake
import Markdown, PrettyTables, Printf


bcks = [
    AutoEnzyme(mode=Enzyme.Reverse),
    AutoEnzyme(mode=Enzyme.Forward),
    AutoMooncake(config=nothing),
    AutoForwardDiff(),
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm()
    ),
    AutoSparse(
        AutoEnzyme(mode=Enzyme.Forward);
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm()
    )
    ]


uin() = 0.0
uout() = 0.0
function Diffusion(u)
    du = zero(u)
    for i in eachindex(du,u)
        if i == 1
            ug = uin()
            ud = u[i+1]
        elseif i == length(u)
            ug = u[i-1]
            ud = uout()
        else
            ug = u[i-1]
            ud = u[i+1] 
        end
        du[i] = ug + ud -2*u[i]
    end
    return du
end;


function DDiffusion(u)
    A = diagm(
        -1 => ones(length(u)-1),
        0=>-2 .*ones(length(u)),
        1 => ones(length(u)-1))
    return A
end;


u = rand(1000)
scenarios = [ Scenario{:jacobian,:out}(Diffusion,u,res1=DDiffusion(u))];


df = benchmark_differentiation(bcks, scenarios)
table = PrettyTables.pretty_table(
    String,
    df;
    backend=Val(:markdown),
    header=names(df),
    formatters=PrettyTables.ft_printf("%.1e"),
)

Markdown.parse(table)

