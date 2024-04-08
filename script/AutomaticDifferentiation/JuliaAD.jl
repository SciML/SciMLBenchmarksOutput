
using DifferentiationInterface, DifferentiationInterfaceTest, DataFrames
import Enzyme, Zygote, Tapir
import Markdown, PrettyTables

function f(x::AbstractVector{T}) where {T}
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

backends = [AutoEnzyme(Enzyme.Reverse), AutoZygote(), AutoTapir()];
scenarios = [GradientScenario(f, x=rand(100)), GradientScenario(f, x=rand(10_000))];
result = benchmark_differentiation(backends, scenarios, logging=true)
data = DataFrame(result)
Markdown.parse(PrettyTables.pretty_table(String, data; backend=Val(:markdown), header=names(data)))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

