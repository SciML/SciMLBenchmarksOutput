
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
    AutoEnzyme(Enzyme.Reverse),
    AutoTapir(),
    AutoZygote(),
];

scenarios = [
    GradientScenario(paritytrig, x=rand(100); operator=:inplace),
    GradientScenario(paritytrig, x=rand(10_000); operator=:inplace)
];

result = benchmark_differentiation(backends, scenarios, logging=false);

data = DataFrame(result);

filtered_data = @chain data begin
    @select(:backend, :operator, :func, :input_type, :input_size, :time, :bytes, :allocs, :compile_fraction, :gc_fraction)
    @rsubset(string(:operator) in ["gradient!"])
end

table = PrettyTables.pretty_table(
    String,
    filtered_data;
    backend=Val(:markdown),
    header=names(filtered_data),
    formatters=PrettyTables.ft_printf("%.1e"),
)

Markdown.parse(table)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

