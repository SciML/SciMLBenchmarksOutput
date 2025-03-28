
using DifferentiationInterface, DifferentiationInterfaceTest, DataFrames, DataFramesMeta
import Enzyme, Zygote, Tapir
import Markdown, PrettyTables, Printf

function paritytrig(x::AbstractVector{T}) where {T}
    y = zero(T)
    for i in eachindex(x)
        if iseven(i)
            y += sin(x[i])
        else
            y += cos(x[i])
        end
    end
    return y
end

backends = [
    AutoEnzyme(mode=Enzyme.Reverse),
    AutoTapir(safe_mode=false),
    AutoZygote(),
];

scenarios = [
    GradientScenario(paritytrig; x=rand(100), y=0.0, nb_args=1, place=:inplace),
    GradientScenario(paritytrig; x=rand(10_000), y=0.0, nb_args=1, place=:inplace)
];

data = benchmark_differentiation(backends, scenarios, logging=true);

table = PrettyTables.pretty_table(
    String,
    data;
    backend=Val(:markdown),
    header=names(data),
    formatters=PrettyTables.ft_printf("%.1e"),
)

Markdown.parse(table)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

