---
author: "Torkel Loman"
title: "Nonlinear Solver 23 Test Problems"
---


These benchmarks compares the runtime and error for a range of nonlinear solvers. The problems are a standard set of problems as described [here](https://people.sc.fsu.edu/~jburkardt/m_src/test_nonlin/test_nonlin.html). The solvers are implemented in [NonlinearProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/master/lib/NonlinearProblemLibrary/src/NonlinearProblemLibrary.jl), where you can find the problem function declarations. For each problem we test the following solvers:
- NonlinearSolve.jl's [Newton Raphson](https://docs.sciml.ai/NonlinearSolve/stable/api/nonlinearsolve/#NonlinearSolve.NewtonRaphson) method (`NewtonRaphson()`).
- NonlinearSolve.jl's [Newton trust region](https://docs.sciml.ai/NonlinearSolve/stable/api/nonlinearsolve/#NonlinearSolve.TrustRegion) method (`TrustRegion()`).
- NonlinearSolve.jl's Levenberg-Marquardt method (`LevenbergMarquardt()`).
- MINPACK's [Modified Powell](https://docs.sciml.ai/NonlinearSolve/stable/api/minpack/#NonlinearSolveMINPACK.CMINPACK) method (`CMINPACK(method=:hybr)`).
- MINPACK's [Levenberg-Marquardt](https://docs.sciml.ai/NonlinearSolve/stable/api/minpack/#NonlinearSolveMINPACK.CMINPACK) method (`CMINPACK(method=:lm)`).
- NLSolveJL's [Newton Raphson](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLSolveJL(method=:newton)`).
- NLSolveJL's [Newton trust region](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLSolveJL()`).
- NLSolveJL's [Anderson acceleration](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLSolveJL(method=:anderson)`).
- Sundials's [Newton-Krylov](https://docs.sciml.ai/NonlinearSolve/stable/api/sundials/#Solver-API) method (`KINSOL()`).

Furthermore, for NonlinearSolve.jl's Newton Raphson method we try the following line search options (in addition to the default):
- `HagerZhang`
- `MoreThuente`
- `BackTracking`

and for NonlinearSolve.jl's Newton trust region we try the following radius update schemes (in addition to the default):
- `NLsolve` 
- `NocedalWright` 
- `Hei` 
- `Yuan` 
- `Bastin` 
- `Fan` 
and finally for NonlinearSolve.jl's Levenberg-Marquardt method why try using both the default `α_geodesic` value (`0.75`) and a modified value (`0.5`), and also with and without setting the `NormalCholeskyFactorization` linear solver.

For each benchmarked problem, the second, third, and fourth plots compares the performance of NonlinearSolve's Newton Raphson, Newton trust region, and Levenberg-Marquardt methods, respectively. The first plot compares the best methods from each of these categories to the various methods available from other packages. At the end of the benchmarks, we print a summary table of which solvers suceeded for which problems.

# Setup
Fetch required packages.
```julia
using NonlinearSolve, NonlinearSolveMINPACK, SciMLNLSolve, SimpleNonlinearSolve, LinearSolve, StaticArrays, Sundials
using BenchmarkTools, LinearAlgebra, DiffEqDevTools, NonlinearProblemLibrary, Plots
RUS = RadiusUpdateSchemes;
```

```
Error: Failed to precompile Sundials [c3572dad-4567-51f8-b174-8c6c989267f4]
 to "/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/compiled/v1.9/Sundials/jl_ZDe1Fo".
```




Declare the benchmarked solvers (and their names and plotting options).
```julia
solvers_all = [ 
    (type = :NR,      name = "Newton Raphson (No line search)",                    solver = Dict(:alg=>NewtonRaphson()),                                       color = :salmon1,         markershape = :star4),
    (type = :NR,      name = "Newton Raphson (Hager & Zhang line search)",         solver = Dict(:alg=>NewtonRaphson(linesearch=HagerZhang())),                color = :tomato1,         markershape = :pentagon),
    (type = :NR,      name = "Newton Raphson (More & Thuente line search)",        solver = Dict(:alg=>NewtonRaphson(linesearch=MoreThuente())),               color = :red3,            markershape = :star6),
    (type = :NR,      name = "Newton Raphson (Nocedal & Wright line search)",      solver = Dict(:alg=>NewtonRaphson(linesearch=BackTracking())),              color = :firebrick,       markershape = :heptagon),
    (type = :TR,      name = "Newton Trust Region",                                solver = Dict(:alg=>TrustRegion()),                                         color = :darkslategray1,  markershape = :utriangle),
    (type = :TR,      name = "Newton Trust Region (NLsolve radius update)",        solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.NLsolve)),       color = :deepskyblue1,    markershape = :rect),
    (type = :TR,      name = "Newton Trust Region (Nocedal Wright radius update)", solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.NocedalWright)), color = :cadetblue,       markershape = :diamond),
    (type = :TR,      name = "Newton Trust Region (Hei radius update)",            solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Hei)),           color = :lightslateblue,  markershape = :star5),
    (type = :TR,      name = "Newton Trust Region (Yuan radius update)",           solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Yuan)),          color = :royalblue2,      markershape = :hexagon),
    (type = :TR,      name = "Newton Trust Region (Bastin radius update)",         solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Bastin)),        color = :blue1,           markershape = :star7),
    (type = :TR,      name = "Newton Trust Region (Fan radius update)",            solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Fan)),           color = :navy,            markershape = :octagon),
    (type = :LM,      name = "Levenberg-Marquardt (α_geodesic=0.75)",                   solver = Dict(:alg=>LevenbergMarquardt()),                                                            color = :fuchsia,         markershape = :circle),
    (type = :LM,      name = "Levenberg-Marquardt (α_geodesic, with CCholesky)",        solver = Dict(:alg=>LevenbergMarquardt(linsolve = NormalCholeskyFactorization())),                    color = :orchid4,         markershape = :rtriangle),
    (type = :LM,      name = "Levenberg-Marquardt (α_geodesic=0.5)",                    solver = Dict(:alg=>LevenbergMarquardt(α_geodesic=0.5)),                                              color = :darkorchid1,     markershape = :ltriangle),
    (type = :LM,      name = "Levenberg-Marquardt (α_geodesic=0.5, with CCholesky)",    solver = Dict(:alg=>LevenbergMarquardt(linsolve = NormalCholeskyFactorization(), α_geodesic=0.5)),    color = :purple4,         markershape = :star8),
    (type = :general, name = "Modified Powell (CMINPACK)",                         solver = Dict(:alg=>CMINPACK(method=:hybr)),                                color = :lightgoldenrod2, markershape = :+),
    (type = :general, name = "Levenberg-Marquardt (CMINPACK)",                     solver = Dict(:alg=>CMINPACK(method=:lm)),                                  color = :gold1,           markershape = :x),
    (type = :general, name = "Newton Raphson (NLSolveJL)",                         solver = Dict(:alg=>NLSolveJL(method=:newton)),                             color = :olivedrab1,      markershape = :dtriangle),
    (type = :general, name = "Newton Trust Region (NLSolveJL)",                    solver = Dict(:alg=>NLSolveJL()),                                           color = :green2,          markershape = :rtriangle),
    (type = :general, name = "Newton-Krylov (Sundials)",                           solver = Dict(:alg=>KINSOL()),                                              color = :darkorange,      markershape = :circle)
]
solver_tracker = [];
```

```
Error: UndefVarError: `RUS` not defined
```




Sets tolerances.
```julia
abstols = 1.0 ./ 10.0 .^ (4:12)
reltols = 1.0 ./ 10.0 .^ (4:12);
```



Set plotting defaults.
```julia
mm = Plots.Measures.mm
default(framestyle=:box,legend=:topleft,gridwidth=2, guidefontsize=25, tickfontsize=18, legendfontsize=16, la=0.7, ms=12, ma=0.8)
```

```
Error: UndefVarError: `Plots` not defined
```




Prepares various helper functions for benchmarking a specific problem.
```julia
# Benchmarks a specific problem, checks which solvers can solve it and their performance
function benchmark_problem!(prob_name; solver_tracker=solver_tracker, selected_NR=nothing, selected_TR=nothing, selected_LM=nothing)
    # Finds the problem and the true solution.
    prob = nlprob_23_testcases[prob_name]

    # Finds the solvers that can solve the problem
    successful_solvers = filter(solver -> check_solver(prob, solver), solvers_all)
    push!(solver_tracker, prob_name => successful_solvers)

    # Handles the non-general cases.
    solvers_NR = filter(s -> s.type==:NR, successful_solvers)
    solvers_TR = filter(s -> s.type==:TR, successful_solvers)
    solvers_LM = filter(s -> s.type==:LM, successful_solvers)
    wp_NR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_NR, :solver); names=getfield.(solvers_NR, :name), numruns=100, error_estimate=:l2, maxiters=10000000)
    wp_TR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_TR, :solver); names=getfield.(solvers_TR, :name), numruns=100, error_estimate=:l2, maxiters=10000000)
    wp_LM = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_LM, :solver); names=getfield.(solvers_LM, :name), numruns=100, error_estimate=:l2, maxiters=10000000)

    # Handles the general case
    solvers_general = filter(s -> s.type==:general, successful_solvers)
    add_solver!(solvers_general, selected_TR, solvers_TR, wp_TR)
    add_solver!(solvers_general, selected_LM, solvers_LM, wp_LM)
    add_solver!(solvers_general, selected_NR, solvers_NR, wp_NR)
    wp_general = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_general, :solver); names=getfield.(solvers_general, :name), numruns=100, error_estimate=:l2, maxiters=10000000)
    
    xlimit, ylimit, xticks, yticks = get_limits_and_ticks(wp_general, wp_NR, wp_TR, wp_LM)
    wp_plot_general = plot_wp(wp_general, solvers_general, xguide="", xlimit, ylimit, linewidth=7, true, xticks=(xticks, fill("", length(xticks))),yticks=yticks)
    wp_plot_NR = plot_wp(wp_NR, solvers_NR, xlimit, ylimit, linewidth=7, true; xguide="", yguide="", xticks=(xticks, fill("", length(xticks))), yticks=(yticks, fill("", length(yticks))), right_margin=7mm)
    wp_plot_TR = plot_wp(wp_TR, solvers_TR, xlimit, ylimit, linewidth=7, false; xticks=xticks, yticks=yticks)
    wp_plot_LM = plot_wp(wp_LM, solvers_LM, xlimit, ylimit, linewidth=7, false; yguide="", xticks=xticks, yticks=(yticks, fill("", length(yticks))), right_margin=7mm)
    plot(wp_plot_general, wp_plot_NR, wp_plot_TR, wp_plot_LM, layout=(2,2), size=(1600,2100), left_margin=12mm)
end

# Checks if a solver can sucessfully solve a given problem.
function check_solver(prob, solver)
    try
        sol = solve(prob.prob, solver.solver[:alg]; abstol=1e-8, reltol=1e-8, maxiters=10000000)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("\n[Warn] Solver $(solver.name) returned retcode $(sol.retcode) with an residual norm = $(norm(sol.resid)).\n"; color=:red)
            return false
        elseif norm(sol.resid) > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm = $(norm(sol.resid))).\n"; color=:red)
            return false
        end
        WorkPrecisionSet(prob.prob, [1e-4, 1e-12], [1e-4, 1e-12], [solver.solver]; names=[solver.name], numruns=100, error_estimate=:l2, maxiters=10000000)
    catch e
        Base.printstyled("[Warn] Solver $(solver.name) threw an error: $e.\n"; color=:red)    
        return false
    end
    return true
end

# Adds an additional, selected, solver to the general solver set.
# Adds an additional, selected, solver to the general solver set.
function add_solver!(solvers_general, selected_solver_name, additional_solver_set, wp)
    if isnothing(selected_solver_name)
        isempty(wp.wps) && return
        selected_idx = argmin(mean.(getfield.(wp.wps, :times))) 
    else
        selected_idx = findfirst(s -> s.name==selected_solver_name, additional_solver_set)
        isnothing(selected_solver) && error("The $(selected_solver_name) was designated to be added to the general solver set, however, it seemed to fail on this problem.")
    end
    isnothing(selected_idx) || pushfirst!(solvers_general, additional_solver_set[selected_idx])   
end;
```



Plotting related helper functions.
```julia
# Plots a work-precision diagram.
function plot_wp(wp, selected_solvers, xlimit, ylimit, top; kwargs...)
    color = reshape(getfield.(selected_solvers, :color),1,length(selected_solvers))
    markershape = reshape(getfield.(selected_solvers, :markershape),1,length(selected_solvers))

    if isempty(wp.wps)
        (:xguide in keys(kwargs)) || (kwargs = (; xguide="Error", kwargs...))
        (:yguide in keys(kwargs)) || (kwargs = (; yguide="Time (s)", kwargs...))
        plt = plot(;xlimit=xlimit, ylimit=ylimit, legend=:none, xaxis=:log10, yaxis=:log10, kwargs...)
        if top
            plt_legend = plot(;xlimit=(1e6,1e6+1), ylimit=(0.01,0.011), legend=:outerbottom, axis=false, grid=false, framestyle=:none, margin=0mm, kwargs...)
            return plot(plt_legend, plt, layout = grid(2, 1, heights=[0.25, 0.75]), top_margin=0mm, bottom_margin=0mm)
        else
            plt_legend = plot(;xlimit=(1e6,1e6+1), ylimit=(0.01,0.011), legend=:outertop, axis=false, grid=false, framestyle=:none, margin=0mm, kwargs...)
            return plot(plt, plt_legend, layout = grid(2, 1, heights=[0.75, 0.25]), top_margin=0mm, bottom_margin=0mm)
        end
    end
    plt = plot(wp; color=color, markershape=markershape, xlimit=xlimit, ylimit=ylimit, legend=:none, kwargs...)
    if top
        plt_legend = plot(wp; color=color, markershape=markershape, xlimit=(1e6,1e6+1), ylimit=(0.01,0.011), legend=:outerbottom, axis=false, grid=false, framestyle=:none, margin=0mm, kwargs...)
        return plot(plt_legend, plt, layout = grid(2, 1, heights=[0.25, 0.75]), top_margin=0mm, bottom_margin=0mm)
    else
        plt_legend = plot(wp; color=color, markershape=markershape, xlimit=(1e6,1e6+1), ylimit=(0.01,0.011), legend=:outertop, axis=false, grid=false, framestyle=:none, margin=0mm, kwargs...)
        return plot(plt, plt_legend, layout = grid(2, 1, heights=[0.75, 0.25]), top_margin=0mm, bottom_margin=0mm)
    end
end

# For a set of wp diaggras, get ticks and limits.
function get_limits_and_ticks(args...)
    xlimit = [Inf, -Inf]
    ylimit = [Inf, -Inf]
    for wp in args
        isempty(wp.wps) && continue
        xlim, ylim = xy_limits(wp)
        (xlimit[1] > xlim[1]) && (xlimit[1] = xlim[1])
        (xlimit[2] < xlim[2]) && (xlimit[2] = xlim[2])
        (ylimit[1] > ylim[1]) && (ylimit[1] = ylim[1])
        (ylimit[2] < ylim[2]) && (ylimit[2] = ylim[2])
    end
    xlimit = Tuple(xlimit)
    ylimit = Tuple(ylimit)
    xticks = get_ticks(xlimit)
    yticks = get_ticks(ylimit)
    return xlimit, ylimit, xticks, yticks 
end
# Finds good x and y limits for a work precision diagram.
function xy_limits(wp)
    times = vcat(map(wp -> wp.times, wp.wps)...)
    errors = vcat(map(wp -> wp.errors, wp.wps)...)
    xlimit = 10 .^ (floor(log10(minimum(errors))), ceil(log10(maximum(errors))))
    ylimit = 10 .^ (floor(log10(minimum(times))), ceil(log10(maximum(times))))
    return xlimit, ylimit
end

# Finds good x and y ticks for a work precision diagram.
function arithmetic_sequences(v1, v2)
    sequences = []
    for n in 2:(v2-v1+1)
        d = (v2 - v1) / (n - 1)
        if d == floor(d)  
            sequence = [v1 + (j-1)*d for j in 1:n]
            push!(sequences, sequence)
        end
    end
    return sequences
end
function get_ticks(limit)
    (limit == (Inf, -Inf)) && (return [])
    limit = log10.(limit)
    (limit[1]==-Inf) && return 10.0 .^[limit[1], limit[2]]
    sequences = arithmetic_sequences(limit...)
    selected_seq = findlast(length.(sequences) .< 5)
    if length(sequences[selected_seq]) < 4
        step = (limit[2] - limit[1]) / 6.0
        ticks = [round(Int, limit[1] + i*step) for i in 1:5]
        return 10 .^[limit[1];ticks;limit[2]]
    end
    return 10 .^sequences[selected_seq]
end;
```



# Benchmarks
We here run benchmarks for each of the 23 models. 

### Problem 1 (Generalized Rosenbrock function)
```julia
benchmark_problem!("Generalized Rosenbrock function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 2 (Powell singular function)
```julia
benchmark_problem!("Powell singular function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 3 (Powell badly scaled function)
```julia
benchmark_problem!("Powell badly scaled function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 4 (Wood function)
```julia
benchmark_problem!("Wood function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 5 (Helical valley function)
```julia
benchmark_problem!("Helical valley function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 6 (Watson function)
```julia
benchmark_problem!("Watson function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 7 (Chebyquad function)
```julia
benchmark_problem!("Chebyquad function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 8 (Brown almost linear function)
```julia
benchmark_problem!("Brown almost linear function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 9 (Discrete boundary value function)
```julia
benchmark_problem!("Discrete boundary value function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 10 (Discrete integral equation function)
```julia
benchmark_problem!("Discrete integral equation function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 11 (Trigonometric function)
```julia
benchmark_problem!("Trigonometric function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 12 (Variably dimensioned function)
```julia
benchmark_problem!("Variably dimensioned function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 13 (Broyden tridiagonal function)
```julia
benchmark_problem!("Broyden tridiagonal function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 14 (Broyden banded function)
```julia
benchmark_problem!("Broyden banded function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 15 (Hammarling 2 by 2 matrix square root problem)
```julia
benchmark_problem!("Hammarling 2 by 2 matrix square root problem")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 16 (Hammarling 3 by 3 matrix square root problem)
```julia
benchmark_problem!("Hammarling 3 by 3 matrix square root problem")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 17 (Dennis and Schnabel 2 by 2 example)
```julia
benchmark_problem!("Dennis and Schnabel 2 by 2 example")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 18 (Sample problem 18)
```julia
benchmark_problem!("Sample problem 18")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 19 (Sample problem 19)
```julia
benchmark_problem!("Sample problem 19")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 20 (Scalar problem f(x) = x(x - 5)^2)
```julia
benchmark_problem!("Scalar problem f(x) = x(x - 5)^2")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 21 (Freudenstein-Roth function)
```julia
benchmark_problem!("Freudenstein-Roth function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 22 (Boggs function)
```julia
benchmark_problem!("Boggs function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





### Problem 23 (Chandrasekhar function)
```julia
benchmark_problem!("Chandrasekhar function")
```

```
Error: UndefVarError: `solver_tracker` not defined
```





## Summary of sucessful solvers
Finally, we print a summar of which solvers sucesfully solved which problems.

```julia
solver_sucesses = [(solver in prob[2]) ? "O" : "X" for prob in solver_tracker, solver in solvers_all];
total_sucesses = [sum(solver_sucesses[:,i] .== "O") for i in 1:length(solvers_all)]
solver_outcomes = vcat(total_sucesses', solver_sucesses);
```

```
Error: UndefVarError: `solver_tracker` not defined
```



```julia
using PrettyTables
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, solver_outcomes; backend = Val(:html), header = getfield.(solvers_all, :name), row_names = ["Total successes:"; first.(solver_tracker)], alignment=:c)
println(io, "```")
Text(String(take!(io)))
```

Error: UndefVarError: `solvers_all` not defined




# Appendix

## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonlinearProblem","nonlinear_solver_23_tests.jmd")
```

Computer Information:

```
Julia Version 1.9.3
Commit bed2cd540a1 (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver2)
  Threads: 128 on 128 virtual cores
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
  JULIA_IMAGE_THREADS = 1

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.3.2
  [f3b72e0c] DiffEqDevTools v2.39.0
⌃ [7ed4a6bd] LinearSolve v2.9.2
  [b7050fa9] NonlinearProblemLibrary v0.1.1
⌃ [8913a72c] NonlinearSolve v2.1.0
⌃ [c100e077] NonlinearSolveMINPACK v0.1.3
  [91a5bcdd] Plots v1.39.0
⌃ [08abe8d2] PrettyTables v2.2.7
  [31c91b34] SciMLBenchmarks v0.1.3
  [e9a6253c] SciMLNLSolve v0.1.9
⌃ [727e6d20] SimpleNonlinearSolve v0.1.20
  [90137ffa] StaticArrays v1.6.5
  [c3572dad] Sundials v4.20.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Manifest.toml`
  [47edcb42] ADTypes v0.2.4
  [79e6a3ab] Adapt v3.6.2
  [ec485272] ArnoldiMethod v0.2.0
  [4fba245c] ArrayInterface v7.4.11
  [30b0a656] ArrayInterfaceCore v0.1.29
  [6e4b80f9] BenchmarkTools v1.3.2
  [d1d4a3ce] BitFlags v0.1.7
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
⌅ [fa961155] CEnum v0.4.2
  [2a0fbf3d] CPUSummary v0.2.4
  [49dc2e85] Calculus v0.5.1
⌃ [d360d2e6] ChainRulesCore v1.17.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
⌃ [944b1d66] CodecZlib v0.7.2
  [35d6a980] ColorSchemes v3.24.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.10.0
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.2.1
  [8f4d0f93] Conda v1.9.1
  [187b0558] ConstructionBase v1.5.4
  [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.15.0
  [864edb3b] DataStructures v0.18.15
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v6.130.0
  [f3b72e0c] DiffEqDevTools v2.39.0
  [77a26b50] DiffEqNoiseProcess v5.19.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌅ [b4f34e82] Distances v0.9.2
  [31c24e10] Distributions v0.25.102
  [ffbed154] DocStringExtensions v0.9.3
  [fa6b7ba4] DualNumbers v0.6.8
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.6.2
  [460bff9d] ExceptionUnwrapping v0.1.9
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [7034ab61] FastBroadcast v0.2.7
  [29a986be] FastLapackInterface v2.0.0
⌃ [1a297f60] FillArrays v1.6.1
  [6a86dc24] FiniteDiff v2.21.1
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.36
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.1.5
  [28b8d3ca] GR v0.72.10
  [d7ba0133] Git v1.3.0
  [86223c79] Graphs v1.9.0
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.10.0
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.4
  [92d709cd] IrrationalConstants v0.2.2
  [82899510] IteratorInterfaceExtensions v1.0.0
⌃ [1019f520] JLFzf v0.1.5
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [ef3ab10e] KLU v0.4.1
  [ba0b0d4f] Krylov v0.9.4
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.16.1
  [10f19ff3] LayoutPointers v0.1.14
  [50d2b5c4] Lazy v0.15.1
  [d3d80556] LineSearches v7.2.0
⌃ [7ed4a6bd] LinearSolve v2.9.2
  [2ab3a3ac] LogExpFunctions v0.3.26
  [e6f89c97] LoggingExtras v1.0.3
  [bdcacae8] LoopVectorization v0.12.165
  [4854310b] MINPACK v1.1.1
  [1914dd2f] MacroTools v0.5.11
  [d125e4d3] ManualMemory v0.1.8
  [739be429] MbedTLS v1.1.7
  [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.1.0
  [46d2c3a1] MuladdMacro v0.2.4
  [ffc61752] Mustache v1.0.17
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.2
  [b7050fa9] NonlinearProblemLibrary v0.1.1
⌃ [8913a72c] NonlinearSolve v2.1.0
⌃ [c100e077] NonlinearSolveMINPACK v0.1.3
  [6fe1bfb0] OffsetArrays v1.12.10
  [4d8831e6] OpenSSL v1.4.1
  [429524aa] Optim v1.7.8
  [bac558e1] OrderedCollections v1.6.2
⌃ [90014a1f] PDMats v0.11.26
  [65ce6f38] PackageExtensionCompat v1.0.2
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.7.2
  [b98c9c47] Pipe v1.3.0
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.3.5
  [91a5bcdd] Plots v1.39.0
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.8
  [1d0040c9] PolyesterWeave v0.2.1
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.12
  [aea7be01] PrecompileTools v1.2.0
  [21216c6a] Preferences v1.4.1
⌃ [08abe8d2] PrettyTables v2.2.7
  [1fd47b50] QuadGK v2.9.1
  [74087812] Random123 v1.6.1
  [e6cf234a] RandomNumbers v1.5.3
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v2.39.0
  [f2c3362d] RecursiveFactorization v0.2.20
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.7.1
  [47965b36] RootedTrees v2.19.2
  [7e49a35a] RuntimeGeneratedFunctions v0.5.12
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.39
⌅ [0bca4576] SciMLBase v1.98.1
  [31c91b34] SciMLBenchmarks v0.1.3
  [e9a6253c] SciMLNLSolve v0.1.9
  [c0aeaf25] SciMLOperators v0.3.6
  [6c6a2e73] Scratch v1.2.0
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
⌃ [727e6d20] SimpleNonlinearSolve v0.1.20
  [699a6c99] SimpleTraits v0.9.4
  [66db9d55] SnoopPrecompile v1.0.3
  [b85f4697] SoftGlobalScope v1.1.0
⌃ [a2af1166] SortingAlgorithms v1.1.1
  [47a9eef4] SparseDiffTools v2.8.0
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.1
  [aedffcd0] Static v0.8.8
  [0d7ed370] StaticArrayInterface v1.4.1
  [90137ffa] StaticArrays v1.6.5
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.2
  [4c63d2b9] StatsFuns v1.3.0
⌅ [7792a7ef] StrideArraysCore v0.4.17
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [c3572dad] Sundials v4.20.0
  [2efcf032] SymbolicIndexingInterface v0.2.2
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.11.0
  [62fd8b95] TensorCore v0.1.1
  [8290d209] ThreadingUtilities v0.5.2
⌅ [3bb67fe8] TranscodingStreams v0.9.13
  [d5829a12] TriangularSolve v0.1.19
  [410a4b4d] Tricks v0.1.8
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [5c2747f8] URIs v1.5.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.17.0
  [45397f5d] UnitfulLatexify v1.6.3
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.64
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [700de1a5] ZygoteRules v0.2.3
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2702e6a9] EpollShim_jll v0.0.20230411+0
  [2e619515] Expat_jll v2.5.0+0
⌃ [b22a6f82] FFMPEG_jll v4.4.2+2
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.8+0
  [d2c73de3] GR_jll v0.72.10+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.36.1+2
  [7746bdde] Glib_jll v2.76.5+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2023.2.0+0
  [aacddb02] JpegTurbo_jll v2.1.91+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [1d63c593] LLVMOpenMP_jll v15.0.4+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2023.2.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [458c3c95] OpenSSL_jll v1.1.23+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.5.2+2
  [f50d1b31] Rmath_jll v0.4.0+0
⌅ [fb77eaff] Sundials_jll v5.2.2+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.21.0+1
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.11.5+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.4.4+0
  [f67eecfb] Xorg_libICE_jll v1.0.10+1
  [c834827a] Xorg_libSM_jll v1.2.3+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.4+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [3161d3a3] Zstd_jll v1.5.5+0
  [35ca27e7] eudev_jll v3.2.9+0
⌅ [214eeab7] fzf_jll v0.29.0+0
  [1a1c6b14] gperf_jll v3.1.1+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [2db6ffa8] libevdev_jll v1.11.0+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [36db933b] libinput_jll v1.18.0+0
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [009596ad] mtdev_jll v1.1.6+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+1
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.10.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.5+1
  [deac9b47] LibCURL_jll v8.0.1+1
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+2
  [05823500] OpenLibm_jll v0.8.1+2
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.0+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.8.0+1
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

