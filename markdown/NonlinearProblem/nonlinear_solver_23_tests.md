---
author: "Torkel Loman & Avik Pal"
title: "Nonlinear Solver 23 Test Problems"
---


These benchmarks compares the runtime and error for a range of nonlinear solvers. The problems are a standard set of problems as described [here](https://people.sc.fsu.edu/~jburkardt/m_src/test_nonlin/test_nonlin.html). The solvers are implemented in [NonlinearProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/master/lib/NonlinearProblemLibrary/src/NonlinearProblemLibrary.jl), where you can find the problem function declarations. For each problem we test the following solvers:
- NonlinearSolve.jl's [Newton Raphson](https://docs.sciml.ai/NonlinearSolve/stable/api/nonlinearsolve/#NonlinearSolve.NewtonRaphson) method (`NewtonRaphson()`).
- NonlinearSolve.jl's [Trust Region](https://docs.sciml.ai/NonlinearSolve/stable/api/nonlinearsolve/#NonlinearSolve.TrustRegion) method (`TrustRegion()`).
- NonlinearSolve.jl's Levenberg-Marquardt method (`LevenbergMarquardt()`).
- MINPACK's [Modified Powell](https://docs.sciml.ai/NonlinearSolve/stable/api/minpack/#NonlinearSolveMINPACK.CMINPACK) method (`CMINPACK(method=:hybr)`).
- MINPACK's [Levenberg-Marquardt](https://docs.sciml.ai/NonlinearSolve/stable/api/minpack/#NonlinearSolveMINPACK.CMINPACK) method (`CMINPACK(method=:lm)`).
- NLsolveJL's [Newton Raphson](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLsolveJL(method=:newton)`).
- NLsolveJL's [Trust Region](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLsolveJL()`).
- NLsolveJL's [Anderson acceleration](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/#Solver-API) (`NLsolveJL(method=:anderson)`).
- Sundials's [Newton-Krylov](https://docs.sciml.ai/NonlinearSolve/stable/api/sundials/#Solver-API) method (`KINSOL()`).

Furthermore, for NonlinearSolve.jl's Newton Raphson method we try the following Line Search options (in addition to the default):
- `HagerZhang`
- `MoreThuente`
- `BackTracking`

and for NonlinearSolve.jl's Trust Region we try the following Radius Update schemes (in addition to the default):
- `NLsolve` 
- `NocedalWright` 
- `Hei` 
- `Yuan` 
- `Bastin` 
- `Fan` 
and finally for NonlinearSolve.jl's Levenberg-Marquardt method why try using both the default `α_geodesic` value (`0.75`) and a modified value (`0.5`), and also with and without setting the `CholeskyFactorization` linear solver.

For each benchmarked problem, the second, third, and fourth plots compares the performance of NonlinearSolve's Newton Raphson, Trust Region, and Levenberg-Marquardt methods, respectively. The first plot compares the best methods from each of these categories to the various methods available from other packages. At the end of the benchmarks, we print a summary table of which solvers succeeded for which problems.

# Setup

Fetch required packages.

```julia
using NonlinearSolve, LinearSolve, StaticArrays, Sundials, Setfield,
    BenchmarkTools, LinearAlgebra, DiffEqDevTools, NonlinearProblemLibrary, CairoMakie,
    RecursiveFactorization
import PolyesterForwardDiff, MINPACK, NLsolve

const RUS = RadiusUpdateSchemes;
```




Declare the benchmarked solvers (and their names and plotting options).

```julia
# XXX: Add PETSc
solvers_all = [
    (; pkg = :nonlinearsolve, type = :general, name = "Default PolyAlg.",          solver = Dict(:alg => FastShortcutNonlinearPolyalg(; u0_len = 10))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "Newton Raphson",            solver = Dict(:alg => NewtonRaphson())),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (HagerZhang)",           solver = Dict(:alg => NewtonRaphson(; linesearch = HagerZhang()))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (MoreThuente)",          solver = Dict(:alg => NewtonRaphson(; linesearch = MoreThuente()))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (BackTracking)",         solver = Dict(:alg => NewtonRaphson(; linesearch = BackTracking()))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "Trust Region",              solver = Dict(:alg => TrustRegion())),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (NLsolve Update)",       solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.NLsolve))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Nocedal Wright)",       solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.NocedalWright))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Hei)",                  solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.Hei))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Yuan)",                 solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.Yuan))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Bastin)",               solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.Bastin))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Fan)",                  solver = Dict(:alg => TrustRegion(; radius_update_scheme = RUS.Fan))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "Levenberg-Marquardt",       solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization()))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM with Cholesky",          solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization()))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (α_geodesic=0.5)",       solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization(), α_geodesic=0.5))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (α_geodesic=0.5) Chol.", solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization(), α_geodesic=0.5))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (no Accln.)",            solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization(), disable_geodesic = Val(true)))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (no Accln.) Chol.",      solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization(), disable_geodesic = Val(true)))),
    (; pkg = :nonlinearsolve, type = :general, name = "Pseudo Transient",          solver = Dict(:alg => PseudoTransient(; alpha_initial=10.0))),
    (; pkg = :wrapper,        type = :general, name = "Powell [MINPACK]",          solver = Dict(:alg => CMINPACK(; method=:hybr))),
    (; pkg = :wrapper,        type = :general, name = "LM [MINPACK]",              solver = Dict(:alg => CMINPACK(; method=:lm))),
    (; pkg = :wrapper,        type = :general, name = "NR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL(; method=:newton))),
    (; pkg = :wrapper,        type = :general, name = "TR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL())),
    (; pkg = :wrapper,        type = :general, name = "NR [Sundials]",             solver = Dict(:alg => KINSOL())),
    (; pkg = :wrapper,        type = :general, name = "NR LineSearch [Sundials]",  solver = Dict(:alg => KINSOL(; globalization_strategy=:LineSearch)))
];

solver_tracker = [];
wp_general_tracker = [];
```

```
Error: UndefVarError: `HagerZhang` not defined
```





Sets tolerances.

```julia
abstols = 1.0 ./ 10.0 .^ (4:12)
reltols = 1.0 ./ 10.0 .^ (4:12);
```




Prepares various helper functions for benchmarking a specific problem.

```julia
# Benchmarks a specific problem, checks which solvers can solve it and their performance
function benchmark_problem!(prob_name; solver_tracker=solver_tracker)
    # Finds the problem and the true solution.
    prob = nlprob_23_testcases[prob_name]

    # Finds the solvers that can solve the problem
    successful_solvers = filter(Base.Fix1(check_solver, prob), solvers_all);
    push!(solver_tracker, prob_name => successful_solvers);

    # Handles the non-general cases.
    solvers_NR = filter(s -> s.type==:NR, successful_solvers)
    solvers_TR = filter(s -> s.type==:TR, successful_solvers)
    solvers_LM = filter(s -> s.type==:LM, successful_solvers)
    wp_NR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_NR, :solver);
        names=getfield.(solvers_NR, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = NonlinearSolve.AbsNormTerminationMode(Base.Fix1(maximum, abs)))
    wp_TR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_TR, :solver);
        names=getfield.(solvers_TR, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = NonlinearSolve.AbsNormTerminationMode(Base.Fix1(maximum, abs)))
    wp_LM = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_LM, :solver);
        names=getfield.(solvers_LM, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = NonlinearSolve.AbsNormTerminationMode(Base.Fix1(maximum, abs)))

    # Handles the general case
    solvers_general = filter(s -> s.type==:general, successful_solvers)
    add_solver!(solvers_general, nothing, solvers_TR, wp_TR)
    add_solver!(solvers_general, nothing, solvers_LM, wp_LM)
    add_solver!(solvers_general, nothing, solvers_NR, wp_NR)

    wp_general = WorkPrecisionSet(prob.prob, abstols, reltols,
        getfield.(solvers_general, :solver); names=getfield.(solvers_general, :name),
        numruns=100, error_estimate=:l∞, maxiters=1000)

    push!(wp_general_tracker, prob_name => wp_general)

    fig = plot_collective_benchmark(prob_name, wp_general, wp_NR, wp_TR, wp_LM)

    save(replace(lowercase(prob_name), " " => "_") * "_wpd.svg", fig)

    return fig
end

# Checks if a solver can successfully solve a given problem.
function check_solver(prob, solver)
    try
        sol = solve(prob.prob, solver.solver[:alg]; abstol=1e-8, reltol=1e-8,
            maxiters=1000000,
            termination_condition=NonlinearSolve.AbsNormTerminationMode(Base.Fix1(maximum, abs)))
        if norm(sol.resid, Inf) < 1e-6
            Base.printstyled("[Info] Solver $(solver.name) returned retcode $(sol.retcode) \
                with an residual norm = $(norm(sol.resid, Inf)).\n"; color=:green)
            return true
        else
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual \
                (norm = $(norm(sol.resid, Inf))).\n"; color=:red)
            return false
        end
        WorkPrecisionSet(prob.prob, [1e-4, 1e-12], [1e-4, 1e-12], [solver.solver];
            names=[solver.name], numruns=5, error_estimate=:l∞, maxiters=1000)
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
        selected_idx = argmin(median.(getfield.(wp.wps, :times))) 
    else
        selected_idx = findfirst(s -> s.name==selected_solver_name, additional_solver_set)
        isnothing(selected_solver) && error("The $(selected_solver_name) was designated to \
            be added to the general solver set, however, it seemed to fail on this \
            problem.")
    end
    isnothing(selected_idx) ||
        pushfirst!(solvers_general, additional_solver_set[selected_idx])
end;
```




Plotting related helper functions.

```julia
__log10_zero(x) = ifelse(iszero(x), -100, log10(x))
Makie.inverse_transform(::typeof(__log10_zero)) = exp10
Makie.defaultlimits(::typeof(__log10_zero)) = Makie.defaultlimits(log10)
Makie.defined_interval(::typeof(__log10_zero)) = 0.0..Inf

# Skip minor ticks for __log10_zero scale
function Makie.get_minor_tickvalues(i::IntervalsBetween, scale::typeof(__log10_zero),
        tickvalues, vmin, vmax)
    return []
end

tickformatter(values) = map(values) do v
    e = log10(v)
    if isinteger(e) && e == -100
        return rich("10", superscript("-∞"))
    end
    sup = isinteger(e) ? Int(e) : round(e; digits=2)
    return rich("10", superscript(string(sup)))
end

function __filter_nearzero((ticks, ticklabels))
    if first(ticks) ≈ 1e-100
        idxs = findall(x -> x ≈ 1e-100 || x ≥ 10^-40, ticks)
        return ticks[idxs], ticklabels[idxs]
    end
    return ticks, ticklabels
end

# Plots a work-precision diagram.
function plot_collective_benchmark(prob_name, wp_general, wp_NR, wp_TR, wp_LM)
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1400
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:seaborn_bright, length(solvers_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    fig = with_theme(plot_theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))
        axs = []

        xmin, xmax, ymin, ymax = Inf, -Inf, Inf, -Inf

        for i in 1:2, j in 1:2
            wp = (wp_general, wp_NR, wp_TR, wp_LM)[2 * (i - 1) + j]

            ax = Axis(fig[i + 1, j], ylabel = j == 1 ? L"Time $\mathbf{(s)}$" : "",
                xlabelsize = 22, ylabelsize = 22,
                xlabel = i == 2 ? L"Error: $\mathbf{||f(u^\ast)||_\infty}$" : "",
                xscale = __log10_zero, yscale = __log10_zero,
                xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                xticklabelsize = 20, yticklabelsize = 20,
                xticklabelsvisible = i == 2, yticklabelsvisible = j == 1,
                xticksvisible = i == 2, yticksvisible = j == 1,)
            push!(axs, ax)

            ls = []
            scs = []

            for wpᵢ in wp.wps
                idx = findfirst(s -> s.name == wpᵢ.name, solvers_all)
                errs = getindex.(wpᵢ.errors, :l∞)
                times = wpᵢ.times

                emin, emax = extrema(errs)
                tmin, tmax = extrema(times)
                emin < xmin && (xmin = emin)
                emax > xmax && (xmax = emax)
                tmin < ymin && (ymin = tmin)
                tmax > ymax && (ymax = tmax)

                l = lines!(ax, errs, times; color = colors[idx], linewidth = 5,
                    linestyle = LINESTYLES[solvers_all[idx].pkg], alpha = 0.8,
                    label = wpᵢ.name)
                sc = scatter!(ax, errs, times; color = colors[idx], markersize = 16,
                    strokewidth = 2, marker = Cycled(idx), alpha = 0.8, label = wpᵢ.name)
                push!(ls, l)
                push!(scs, sc)
            end

            legend_title = ("", "Newton Raphson", "Trust Region", "Levenberg-Marquardt")[2 * (i - 1) + j]

            Legend(fig[ifelse(i == 1, 1, 4), j], [[l, sc] for (l, sc) in zip(ls, scs)],
                [wpᵢ.name for wpᵢ in wp.wps], legend_title;
                framevisible=true, framewidth = STROKEWIDTH,
                nbanks = 3, labelsize = 16, titlesize = 16,
                tellheight = true, tellwidth = false, patchsize = (40.0f0, 20.0f0))
        end

        linkaxes!(axs...)

        xmin = max(xmin, 10^-100)

        xticks = __filter_nearzero(Makie.get_ticks(LogTicks(WilkinsonTicks(10; k_min = 5)),
            __log10_zero, tickformatter, xmin, xmax))
        yticks = __filter_nearzero(Makie.get_ticks(LogTicks(WilkinsonTicks(10; k_min = 5)),
            __log10_zero, tickformatter, ymin, ymax))

        foreach(axs) do ax
            ax.xticks = xticks
            ax.yticks = yticks
        end

        fig[0, :] = Label(fig, "Work-Precision Diagram for $(prob_name)",
            fontsize = 24, tellwidth = false, font = :bold)

        fig
    end

    return fig
end
```

```
plot_collective_benchmark (generic function with 1 method)
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





## Summary of successful solvers

Finally, we print a summary of which solvers successfully solved which problems.

```julia
solver_successes = [(solver.name in getfield.(prob[2], :name)) ? "O" : "X" for prob in solver_tracker, solver in solvers_all]
total_successes = [sum(solver_successes[:,i] .== "O") for i in 1:length(solvers_all)]
solver_outcomes = vcat(total_successes', solver_successes);
```

```
Error: UndefVarError: `solver_tracker` not defined
```



```julia
using PrettyTables
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, solver_outcomes; backend = Val(:html), header = getfield.(solvers_all, :name), alignment=:c)
println(io, "```")
Base.Text(String(take!(io)))
```

Error: UndefVarError: `solvers_all` not defined




## Summary of General Solver Performance on All Problems

```julia
fig = begin
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 1
    WIDTH = 1800
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:seaborn_bright, length(solvers_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    with_theme(plot_theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        axs = Matrix{Any}(undef, 5, 5)

        ls = []
        scs = []
        labels = []
        solver_times = []

        for i in 1:5, j in 1:5
            idx = 5 * (i - 1) + j

            idx > length(wp_general_tracker) && break

            prob_name, wp = wp_general_tracker[idx]

            ax = Axis(fig[i, j],
                xscale = __log10_zero, yscale = __log10_zero,
                xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                title = prob_name, titlegap = 10,
                xticklabelsize = 16, yticklabelsize = 16)

            xmin, xmax, ymin, ymax = Inf, -Inf, Inf, -Inf

            for wpᵢ in wp.wps
                idx = findfirst(s -> s.name == wpᵢ.name, solvers_all)
                errs = getindex.(wpᵢ.errors, :l∞)
                times = wpᵢ.times

                emin, emax = extrema(errs)
                tmin, tmax = extrema(times)
                emin < xmin && (xmin = emin)
                emax > xmax && (xmax = emax)
                tmin < ymin && (ymin = tmin)
                tmax > ymax && (ymax = tmax)

                l = lines!(ax, errs, times; color = colors[idx], linewidth = 5,
                    linestyle = LINESTYLES[solvers_all[idx].pkg], alpha = 0.8,
                    label = wpᵢ.name)
                sc = scatter!(ax, errs, times; color = colors[idx], markersize = 16,
                    strokewidth = 2, marker = Cycled(idx), alpha = 0.8, label = wpᵢ.name)

                if wpᵢ.name ∉ labels
                    push!(ls, l)
                    push!(scs, sc)
                    push!(labels, wpᵢ.name)
                end

                if wpᵢ.name ∈ first.(solver_times)
                    idxi = findfirst(x -> first(x) == wpᵢ.name, solver_times)
                    push!(solver_times[idxi][2], median(times) / length(wp.prob.u0))
                else
                    push!(solver_times, wpᵢ.name => [median(times) / length(wp.prob.u0)])
                end
            end

            xmin = max(xmin, 10^-100)

            xticks = __filter_nearzero(Makie.get_ticks(LogTicks(WilkinsonTicks(5; k_min = 3)),
                __log10_zero, tickformatter, xmin, xmax))
            yticks = __filter_nearzero(Makie.get_ticks(LogTicks(WilkinsonTicks(5; k_min = 3)),
                __log10_zero, tickformatter, ymin, ymax))

            ax.xticks = xticks
            ax.yticks = yticks
        end

        ordering = sortperm(median.(last.(solver_times)))

        fig[0, :] = Label(fig, "Work-Precision Diagram for 23 Test Problems",
            fontsize = 24, tellwidth = false, font = :bold)

        fig[:, 0] = Label(fig, "Time (s)", fontsize = 20, tellheight = false, font = :bold,
            rotation = π / 2)
        fig[end + 1, :] = Label(fig,
            L"Error: $\mathbf{||f(u^\ast)||_\infty}$",
            fontsize = 20, tellwidth = false, font = :bold)

        Legend(fig[5, 4:5], [[l, sc] for (l, sc) in zip(ls[ordering], scs[ordering])],
            labels[ordering], "Successful Solvers";
            framevisible=true, framewidth = STROKEWIDTH, orientation = :horizontal,
            titlesize = 20, nbanks = 9, labelsize = 20, halign = :center,
            tellheight = false, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        return fig
    end
end
```

```
Error: UndefVarError: `solvers_all` not defined
```



```julia
save("summary_wp_23test_problems.svg", fig)
```

```
Error: UndefVarError: `fig` not defined
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonlinearProblem","nonlinear_solver_23_tests.jmd")
```

Computer Information:

```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Project.toml`
⌃ [2169fc97] AlgebraicMultigrid v0.6.0
⌃ [6e4b80f9] BenchmarkTools v1.5.0
⌃ [13f3f980] CairoMakie v0.12.16
⌃ [2b5f629d] DiffEqBase v6.158.3
⌃ [f3b72e0c] DiffEqDevTools v2.45.1
⌃ [a0c0ee7d] DifferentiationInterface v0.6.22
⌃ [7da242da] Enzyme v0.13.14
  [40713840] IncompleteLU v0.2.1
  [b964fa9f] LaTeXStrings v1.4.0
  [d3d80556] LineSearches v7.3.0
⌅ [7ed4a6bd] LinearSolve v2.36.2
  [4854310b] MINPACK v1.3.0
  [2774e3e8] NLsolve v4.5.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
⌃ [8913a72c] NonlinearSolve v4.1.0
  [ace2c81b] PETSc v0.3.1
  [98d1487c] PolyesterForwardDiff v0.1.2
  [08abe8d2] PrettyTables v2.4.0
  [f2c3362d] RecursiveFactorization v0.2.23
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [efcf1570] Setfield v1.1.1
⌃ [727e6d20] SimpleNonlinearSolve v2.0.0
⌃ [9f842d2f] SparseConnectivityTracer v0.6.8
⌃ [47a9eef4] SparseDiffTools v2.23.0
  [f1835b91] SpeedMapping v0.3.0
  [860ef19b] StableRNGs v1.0.2
⌃ [90137ffa] StaticArrays v1.9.8
⌃ [c3572dad] Sundials v4.26.1
⌃ [0c5d862f] Symbolics v6.18.3
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Manifest.toml`
⌃ [47edcb42] ADTypes v1.9.0
  [a4c015fc] ANSIColoredPrinters v0.0.1
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
⌃ [7d9f7c33] Accessors v0.1.38
  [22286c92] AccurateArithmetic v0.3.8
⌃ [79e6a3ab] Adapt v4.1.1
  [35492f91] AdaptivePredicates v1.2.0
⌃ [2169fc97] AlgebraicMultigrid v0.6.0
  [66dad0bd] AliasTables v1.1.3
⌃ [27a7e980] Animations v0.4.1
⌃ [4c88cf16] Aqua v0.8.9
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.17.0
⌃ [4c555306] ArrayLayouts v1.10.4
  [67c07d97] Automa v1.1.0
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.7
⌃ [6e4b80f9] BenchmarkTools v1.5.0
  [e2ed5e7c] Bijections v0.1.9
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [8e7c35d0] BlockArrays v1.1.1
⌃ [70df07ce] BracketingNonlinearSolve v1.1.0
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.6
⌃ [159f3aea] Cairo v1.1.0
⌃ [13f3f980] CairoMakie v0.12.16
  [7057c7e9] Cassette v0.3.14
⌃ [d360d2e6] ChainRulesCore v1.25.0
  [fb6a15b2] CloseOpenIntervals v0.1.13
⌃ [944b1d66] CodecZlib v0.7.6
⌃ [a2cac450] ColorBrewer v0.4.0
⌃ [35d6a980] ColorSchemes v3.27.1
⌅ [3da002f7] ColorTypes v0.11.5
⌅ [c3611d14] ColorVectorSpace v0.10.0
⌅ [5ae59095] Colors v0.12.11
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
  [34da2185] Compat v4.16.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
⌃ [f0e56b4a] ConcurrentUtilities v2.4.2
  [8f4d0f93] Conda v1.10.2
  [187b0558] ConstructionBase v1.5.8
  [d38c429a] Contour v0.6.3
  [a2441757] Coverage v1.6.1
  [c36e975a] CoverageTools v1.3.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
⌃ [864edb3b] DataStructures v0.18.20
  [e2d170a0] DataValueInterfaces v1.0.0
⌃ [927a84f5] DelaunayTriangulation v1.6.1
⌃ [2b5f629d] DiffEqBase v6.158.3
⌃ [f3b72e0c] DiffEqDevTools v2.45.1
⌃ [77a26b50] DiffEqNoiseProcess v5.23.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌃ [a0c0ee7d] DifferentiationInterface v0.6.22
  [b4f34e82] Distances v0.10.12
⌃ [31c24e10] Distributions v0.25.113
  [ffbed154] DocStringExtensions v0.9.3
⌃ [e30172f5] Documenter v1.7.0
  [35a29f4d] DocumenterTools v0.1.20
⌃ [5b8099bc] DomainSets v0.7.14
⌃ [7c1d4256] DynamicPolynomials v0.6.0
  [4e289a0a] EnumX v1.0.4
⌃ [7da242da] Enzyme v0.13.14
⌃ [f151be2c] EnzymeCore v0.8.5
  [429591f6] ExactPredicates v2.2.8
⌃ [460bff9d] ExceptionUnwrapping v0.1.10
  [e2ba6199] ExprTools v0.1.10
⌅ [6b7a57c9] Expronicon v0.8.5
⌃ [411431e0] Extents v0.1.4
⌃ [7a1cc6ca] FFTW v1.8.0
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.4
⌃ [5789e2e9] FileIO v1.16.4
⌅ [8fc22ac5] FilePaths v0.8.3
⌃ [48062228] FilePathsBase v0.9.22
  [1a297f60] FillArrays v1.13.0
⌃ [6a86dc24] FiniteDiff v2.26.0
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.38
  [b38be410] FreeType v4.1.1
⌃ [663a7486] FreeTypeAbstraction v0.10.4
  [f62d2435] FunctionProperties v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
⌅ [46192b85] GPUArraysCore v0.1.6
⌃ [61eb1bfa] GPUCompiler v1.0.1
⌃ [68eda718] GeoFormatTypes v0.4.2
⌃ [cf35fbd7] GeoInterface v1.3.8
⌅ [5c1252a2] GeometryBasics v0.4.11
  [d7ba0133] Git v1.3.1
  [a2bd30eb] Graphics v1.1.3
  [86223c79] Graphs v1.12.0
⌃ [3955a311] GridLayoutBase v0.11.0
  [42e2da0e] Grisu v1.0.2
⌃ [708ec375] Gumbo v0.8.2
⌃ [cd3eb016] HTTP v1.10.10
  [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.17
⌃ [34004b35] HypergeometricFunctions v0.3.25
  [7073ff75] IJulia v1.26.0
  [b5f81e59] IOCapture v0.2.5
  [615f187c] IfElse v0.1.1
  [2803e5a7] ImageAxes v0.6.12
  [c817782e] ImageBase v0.1.7
⌃ [a09fc81d] ImageCore v0.10.4
  [82e4d734] ImageIO v0.6.9
  [bc367c6b] ImageMetadata v0.9.10
  [40713840] IncompleteLU v0.2.1
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.5
  [18e54dd8] IntegerMathUtils v0.1.2
  [a98d9a8b] Interpolations v0.15.1
⌃ [d1acc4aa] IntervalArithmetic v0.22.19
  [8197267c] IntervalSets v0.7.10
  [3587e190] InverseFunctions v0.1.17
⌃ [92d709cd] IrrationalConstants v0.2.2
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
⌃ [692b3bcd] JLLWrappers v1.6.1
  [682c06a0] JSON v0.21.4
⌃ [b835a17e] JpegTurbo v0.1.5
  [ef3ab10e] KLU v0.6.0
  [5ab0869b] KernelDensity v0.6.9
⌃ [ba0b0d4f] Krylov v0.9.8
⌃ [929cbde3] LLVM v9.1.3
  [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.5
  [10f19ff3] LayoutPointers v0.1.17
  [0e77f7df] LazilyInitializedFields v1.3.0
⌃ [5078a376] LazyArrays v2.2.1
  [8cdb02fc] LazyModules v0.3.1
  [87fe0de2] LineSearch v0.1.4
  [d3d80556] LineSearches v7.3.0
⌅ [7ed4a6bd] LinearSolve v2.36.2
⌃ [2ab3a3ac] LogExpFunctions v0.3.28
  [e6f89c97] LoggingExtras v1.1.0
  [bdcacae8] LoopVectorization v0.12.171
  [4854310b] MINPACK v1.3.0
  [d8e11817] MLStyle v0.4.17
  [da04e1cc] MPI v0.20.22
  [3da0fdf6] MPIPreferences v0.1.11
⌃ [1914dd2f] MacroTools v0.5.13
⌅ [ee78f7c6] Makie v0.21.16
⌅ [20f20a25] MakieCore v0.8.10
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
  [d0879d2d] MarkdownAST v0.1.2
  [0a4f8689] MathTeXEngine v0.6.2
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.7
  [ffc61752] Mustache v1.0.20
⌃ [d8a4904e] MutableArithmetics v1.5.2
⌃ [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
⌃ [77ba4419] NaNMath v1.0.2
  [f09324ee] Netpbm v1.1.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
⌃ [8913a72c] NonlinearSolve v4.1.0
⌃ [be0214bd] NonlinearSolveBase v1.3.1
⌃ [5959db7a] NonlinearSolveFirstOrder v1.0.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.0.0
⌃ [26075421] NonlinearSolveSpectralMethods v1.0.0
⌃ [d8793406] ObjectFile v0.4.2
  [510215fc] Observables v0.5.5
⌃ [6fe1bfb0] OffsetArrays v1.14.1
  [52e1d378] OpenEXR v0.3.3
  [4d8831e6] OpenSSL v1.4.3
⌃ [429524aa] Optim v1.9.4
⌃ [bac558e1] OrderedCollections v1.6.3
⌃ [90014a1f] PDMats v0.11.31
  [ace2c81b] PETSc v0.3.1
⌃ [f57f5aa1] PNGFiles v0.4.3
  [65ce6f38] PackageExtensionCompat v1.0.2
⌃ [19eb6ba3] Packing v0.5.0
  [5432bcbf] PaddedViews v0.5.12
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [eebad327] PkgVersion v0.3.3
  [995b91a9] PlotUtils v1.4.3
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.16
  [98d1487c] PolyesterForwardDiff v0.1.2
  [1d0040c9] PolyesterWeave v0.2.2
  [647866c9] PolygonOps v0.1.2
  [85a6dd25] PositiveFactorizations v0.2.4
⌃ [d236fae5] PreallocationTools v0.4.24
  [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.4.0
⌃ [27ebfcd6] Primes v0.5.6
  [92933f4c] ProgressMeter v1.10.2
⌃ [43287f4e] PtrArrays v1.2.1
  [4b34888f] QOI v1.0.1
⌃ [1fd47b50] QuadGK v2.11.1
  [74087812] Random123 v1.7.0
  [e6cf234a] RandomNumbers v1.6.0
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
⌃ [731186ca] RecursiveArrayTools v3.27.3
  [f2c3362d] RecursiveFactorization v0.2.23
  [189a3867] Reexport v1.2.2
  [2792f1a3] RegistryInstances v0.1.0
  [05181044] RelocatableFolders v1.0.1
⌃ [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.8.0
  [47965b36] RootedTrees v2.23.1
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.13
⌃ [fdea26ae] SIMD v3.7.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.43
  [322a6be2] Sass v0.2.0
⌃ [0bca4576] SciMLBase v2.59.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.1
⌃ [c0aeaf25] SciMLOperators v0.3.12
⌃ [53ae85a6] SciMLStructures v1.5.0
  [6c6a2e73] Scratch v1.2.1
⌃ [efcf1570] Setfield v1.1.1
⌅ [65257c39] ShaderAbstractions v0.4.1
  [992d4aef] Showoff v1.0.3
  [73760f76] SignedDistanceFields v0.4.0
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.0.0
  [699a6c99] SimpleTraits v0.9.4
  [45858cf5] Sixel v0.1.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
⌃ [9f842d2f] SparseConnectivityTracer v0.6.8
⌃ [47a9eef4] SparseDiffTools v2.23.0
⌃ [0a514795] SparseMatrixColorings v0.4.9
  [e56a9233] Sparspak v0.3.9
⌃ [276daf66] SpecialFunctions v2.4.0
  [f1835b91] SpeedMapping v0.3.0
  [860ef19b] StableRNGs v1.0.2
  [cae243ae] StackViews v0.1.1
⌃ [aedffcd0] Static v1.1.1
  [0d7ed370] StaticArrayInterface v1.8.0
⌃ [90137ffa] StaticArrays v1.9.8
  [1e83bf80] StaticArraysCore v1.4.3
  [82ae8749] StatsAPI v1.7.0
⌃ [2913bbd2] StatsBase v0.34.3
  [4c63d2b9] StatsFuns v1.3.2
  [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
⌃ [892a3eda] StringManipulation v0.4.0
⌅ [09ab397b] StructArrays v0.6.18
  [53d494c1] StructIO v0.3.1
⌃ [c3572dad] Sundials v4.26.1
⌃ [2efcf032] SymbolicIndexingInterface v0.3.34
  [19f23fe9] SymbolicLimits v0.2.2
⌃ [d1185830] SymbolicUtils v3.7.2
⌃ [0c5d862f] Symbolics v6.18.3
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.0
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [8290d209] ThreadingUtilities v0.5.2
⌃ [731e570b] TiffImages v0.11.1
⌃ [a759f4b9] TimerOutputs v0.5.25
  [3bb67fe8] TranscodingStreams v0.11.3
  [d5829a12] TriangularSolve v0.2.1
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌃ [1986cc42] Unitful v1.21.0
  [a7c27f48] Unityper v0.1.6
  [3d5dd08c] VectorizationBase v0.21.71
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [e3aaa7dc] WebP v0.1.3
  [efce3f68] WoodburyMatrices v1.0.0
⌃ [ddb6d928] YAML v0.4.12
⌃ [c2297ded] ZMQ v1.3.0
⌃ [6e34b625] Bzip2_jll v1.0.8+2
  [4e9b3aee] CRlibm_jll v1.0.1+0
⌃ [83423d85] Cairo_jll v1.18.2+1
  [5ae413db] EarCut_jll v2.2.4+0
⌅ [7cc45869] Enzyme_jll v0.0.163+0
⌃ [2e619515] Expat_jll v2.6.2+0
⌅ [b22a6f82] FFMPEG_jll v6.1.2+0
⌃ [f5851436] FFTW_jll v3.3.10+1
⌃ [a3f928ae] Fontconfig_jll v2.13.96+0
⌃ [d7e528f0] FreeType2_jll v2.13.2+0
⌃ [559328eb] FriBidi_jll v1.0.14+0
  [78b55507] Gettext_jll v0.21.0+0
⌃ [59f7168a] Giflib_jll v5.2.2+0
⌃ [f8c6e375] Git_jll v2.46.2+0
⌃ [7746bdde] Glib_jll v2.80.5+0
⌃ [3b182d85] Graphite2_jll v1.3.14+0
  [528830af] Gumbo_jll v0.10.2+0
⌃ [2e76f6c2] HarfBuzz_jll v8.3.1+0
⌃ [e33a78d0] Hwloc_jll v2.11.2+1
  [905a6f67] Imath_jll v3.1.11+0
⌅ [1d5cc7b8] IntelOpenMP_jll v2024.2.1+0
⌃ [aacddb02] JpegTurbo_jll v3.0.4+0
  [c1c5ebd0] LAME_jll v3.100.2+0
⌃ [88015f11] LERC_jll v4.0.0+0
⌅ [dad2f222] LLVMExtra_jll v0.0.34+0
  [1d63c593] LLVMOpenMP_jll v18.1.7+0
⌃ [dd4b983a] LZO_jll v2.10.2+1
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.11.0+0
⌃ [7e76a0d4] Libglvnd_jll v1.6.0+0
⌃ [7add5ba3] Libgpg_error_jll v1.50.0+0
⌃ [94ce4f54] Libiconv_jll v1.17.0+1
⌃ [4b2f31a3] Libmount_jll v2.40.1+0
⌃ [89763e89] Libtiff_jll v4.7.0+0
⌃ [38a345b3] Libuuid_jll v2.40.1+0
⌅ [856f044c] MKL_jll v2024.2.0+0
⌃ [7cb0a576] MPICH_jll v4.2.3+0
⌃ [f1f71cc9] MPItrampoline_jll v5.5.1+0
⌃ [9237b28f] MicrosoftMPI_jll v10.1.4+2
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [656ef2d0] OpenBLAS32_jll v0.3.24+0
  [18a262bb] OpenEXR_jll v3.2.4+0
⌃ [fe0851c0] OpenMPI_jll v5.0.5+0
⌃ [9bd350c2] OpenSSH_jll v9.9.1+1
⌃ [458c3c95] OpenSSL_jll v3.0.15+1
⌃ [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.3+0
  [8fa3689e] PETSc_jll v3.22.0+0
⌃ [36c8627f] Pango_jll v1.54.1+0
⌅ [30392449] Pixman_jll v0.43.4+0
  [f50d1b31] Rmath_jll v0.5.1+0
  [aabda75e] SCALAPACK32_jll v2.2.1+1
⌅ [fb77eaff] Sundials_jll v5.2.2+0
⌃ [02c8fc9c] XML2_jll v2.13.4+0
⌃ [aed1982a] XSLT_jll v1.1.41+0
⌃ [ffd25f8a] XZ_jll v5.6.3+0
⌃ [4f6342f7] Xorg_libX11_jll v1.8.6+0
⌃ [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
⌃ [a3789734] Xorg_libXdmcp_jll v1.1.4+0
⌃ [1082639a] Xorg_libXext_jll v1.3.6+0
⌃ [ea2f1a96] Xorg_libXrender_jll v0.9.11+0
⌃ [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
⌃ [c7cfdc94] Xorg_libxcb_jll v1.17.0+0
⌃ [c5fb5394] Xorg_xtrans_jll v1.5.0+0
⌃ [8f1865be] ZeroMQ_jll v4.3.5+1
⌃ [3161d3a3] Zstd_jll v1.5.6+1
  [b792d7bf] cminpack_jll v1.3.8+0
  [9a68df92] isoband_jll v0.2.3+0
⌃ [a4ae2306] libaom_jll v3.9.0+0
  [0ac62f75] libass_jll v0.15.2+0
  [f638f0a6] libfdk_aac_jll v2.0.3+0
⌃ [b53b4c65] libpng_jll v1.6.44+0
  [47bcb7c8] libsass_jll v3.6.6+0
⌃ [075b6546] libsixel_jll v1.10.3+1
⌃ [a9144af2] libsodium_jll v1.0.20+1
  [f27f6e37] libvorbis_jll v1.3.7+2
⌃ [c5f90fcd] libwebp_jll v1.4.0+0
⌃ [1317d2d5] oneTBB_jll v2021.12.0+0
⌃ [1270edf5] x264_jll v10164.0.0+0
⌅ [dfaa095f] x265_jll v3.6.0+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [8bf52ea8] CRC32c
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
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.1+2
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

