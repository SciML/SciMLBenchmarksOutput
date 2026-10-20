
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
println("KNOWN_BAD_PROBLEMS = ", join(sort(collect(keys(KNOWN_BAD_PROBLEMS))), ", "))

unconstrained_candidates = collect(
    CUTEst.select_sif_problems(contype = "unc", max_var = MAX_NVAR, max_con = 0),
)
unconstrained_problems = select_safe_problems(
    unconstrained_candidates;
    max_var = MAX_NVAR,
    max_con = 0,
)

println("Selected unconstrained problems: ", length(unconstrained_problems))
println(join(unconstrained_problems, ", "))

_, unconstrained_size_buckets = classification_buckets(
    MAX_NVAR, 0, NVAR_BUCKET_EDGES,
)
selected_size_buckets = zeros(Int, length(NVAR_BUCKET_EDGES) + 1)
for name in unconstrained_problems
    bucket = get(unconstrained_size_buckets, uppercase(name), 0)
    @assert bucket > 0 "selected problem $name is absent from CUTEst variable-count filters"
    selected_size_buckets[bucket] += 1
end
@assert all(>(0), selected_size_buckets) "unconstrained selection must cover every variable-size bucket"
@assert select_safe_problems(reverse(unconstrained_candidates); max_var = MAX_NVAR, max_con = 0) ==
    unconstrained_problems "selection must not depend on CUTEst candidate order"


# Worst case: 50 problems x 11 solvers x 30 s = 16,500 s, below the 22,500 s bound the
# previous 5-solver / 90 s configuration had.
unconstrained_timeout_seconds = 30.0

unc_results = run_benchmarks(
    "unconstrained",
    unconstrained_problems,
    UNCONSTRAINED_SOLVERS;
    maxtime = unconstrained_timeout_seconds,
)

display(unc_results)


unc_summary = summarize_results(unc_results)

plot_solve_times(unc_results, "CUTEst unconstrained Optimization.jl solve time")
plot_success_rates(unc_summary, "CUTEst unconstrained Optimization.jl success rate")
plot_compile_times(unc_results, "CUTEst unconstrained Optimization.jl compile time")
plot_performance_profile(unc_results,
    "CUTEst unconstrained Optimization.jl performance profile")
plot_work_precision(unc_results, "CUTEst unconstrained Optimization.jl work-precision";
    metric = :grad_norm)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

