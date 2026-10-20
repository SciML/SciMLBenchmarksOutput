
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
println("MAX_NCON = ", MAX_NCON)
println("SOLVE_MAXITERS = ", SOLVE_MAXITERS)
println("SOLVE_TIMEOUT_SECONDS = ", SOLVE_TIMEOUT_SECONDS)
println("KNOWN_BAD_PROBLEMS = ", join(sort(collect(keys(KNOWN_BAD_PROBLEMS))), ", "))

all_equality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_equ_con = true, max_var = MAX_NVAR, max_con = MAX_NCON))
bounded_equality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_equ_con = true, max_var = MAX_NVAR, max_con = MAX_NCON,
    custom_filter = p -> p["variables"]["free"] < p["variables"]["number"]))
free_equality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_equ_con = true, only_free_var = true, max_var = MAX_NVAR, max_con = MAX_NCON))
assert_bounded_pool_coverage(all_equality_candidates, bounded_equality_candidates,
    free_equality_candidates, "Equality")

all_inequality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_ineq_con = true, max_var = MAX_NVAR, max_con = MAX_NCON))
bounded_inequality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_ineq_con = true, max_var = MAX_NVAR, max_con = MAX_NCON,
    custom_filter = p -> p["variables"]["free"] < p["variables"]["number"]))
free_inequality_candidates = collect(CUTEst.select_sif_problems(min_con = 1,
    only_ineq_con = true, only_free_var = true, max_var = MAX_NVAR, max_con = MAX_NCON))
assert_bounded_pool_coverage(all_inequality_candidates, bounded_inequality_candidates,
    free_inequality_candidates, "Inequality")

bounded_equality_problems = select_safe_problems(bounded_equality_candidates)
bounded_inequality_problems = select_safe_problems(bounded_inequality_candidates)

println("Selected bounded equality-constrained problems: ", length(bounded_equality_problems))
println(join(bounded_equality_problems, ", "))
println("Selected bounded inequality-constrained problems: ", length(bounded_inequality_problems))
println(join(bounded_inequality_problems, ", "))


bounded_results = vcat(
    run_benchmarks("bounded equality constrained", bounded_equality_problems,
        CONSTRAINED_SOLVERS),
    run_benchmarks("bounded inequality constrained", bounded_inequality_problems,
        CONSTRAINED_SOLVERS),
)

display(bounded_results)


bounded_summary = summarize_results(bounded_results)

plot_solve_times(bounded_results, "CUTEst bounded constrained Optimization.jl solve time")
plot_success_rates(bounded_summary, "CUTEst bounded constrained Optimization.jl success rate")
plot_compile_times(bounded_results, "CUTEst bounded constrained Optimization.jl compile time")
plot_performance_profile(bounded_results,
    "CUTEst bounded constrained Optimization.jl performance profile")
plot_work_precision(bounded_results,
    "CUTEst bounded constrained Optimization.jl work-precision")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

