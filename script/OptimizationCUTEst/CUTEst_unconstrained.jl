
ENV["GKSwstype"] = "100"

using CUTEst
using DataFrames
using Plots
using StatsPlots
using StatsBase: countmap
using Statistics
using Printf

include(joinpath(isdefined(Main, :WEAVE_ARGS) ? WEAVE_ARGS[:folder] : @__DIR__,
    "cutest_benchmark_utils.jl"))


println("MAX_PROBLEMS_PER_CATEGORY = ", MAX_PROBLEMS_PER_CATEGORY)
println("MAX_NVAR = ", MAX_NVAR)
println("SOLVE_MAXITERS = ", SOLVE_MAXITERS)
println("SOLVE_TIMEOUT_SECONDS = ", SOLVE_TIMEOUT_SECONDS)
println("KNOWN_BAD_PROBLEMS = ", join(sort(collect(KNOWN_BAD_PROBLEMS)), ", "))

unconstrained_problems = select_safe_problems(
    collect(CUTEst.select_sif_problems(contype = "unc"));
    max_var = MAX_NVAR,
    max_con = 0,
)

println("Selected unconstrained problems: ", length(unconstrained_problems))
println(join(unconstrained_problems, ", "))


unc_results = run_benchmarks(
    "unconstrained",
    unconstrained_problems,
    UNCONSTRAINED_SOLVERS,
)

display(unc_results)


unc_summary = summarize_results(unc_results)

plot_solve_times(unc_results, "CUTEst unconstrained Optimization.jl solve time")
plot_success_rates(unc_summary, "CUTEst unconstrained Optimization.jl success rate")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

