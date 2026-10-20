
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

quadratic_linear_problems = select_safe_problems(
    collect(CUTEst.select_sif_problems(objtype = "quadratic", contype = "linear",
        max_var = MAX_NVAR, max_con = MAX_NCON))
)

println("Selected quadratic problems with linear constraints: ",
    length(quadratic_linear_problems))
println(join(quadratic_linear_problems, ", "))


quadratic_results = run_benchmarks(
    "quadratic linear",
    quadratic_linear_problems,
    CONSTRAINED_SOLVERS,
)

display(quadratic_results)


quadratic_summary = summarize_results(quadratic_results)

plot_solve_times(quadratic_results, "CUTEst quadratic Optimization.jl solve time")
plot_success_rates(quadratic_summary, "CUTEst quadratic Optimization.jl success rate")
plot_compile_times(quadratic_results, "CUTEst quadratic Optimization.jl compile time")
plot_performance_profile(quadratic_results,
    "CUTEst quadratic Optimization.jl performance profile")
plot_work_precision(quadratic_results, "CUTEst quadratic Optimization.jl work-precision")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

