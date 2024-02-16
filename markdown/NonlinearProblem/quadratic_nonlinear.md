---
author: "Avik Pal"
title: "Quadratic Nonlinear Rootfinding Benchmark"
---


This is a test of a simple small nonlinear rootfinding problem.

```julia
using NonlinearSolve, StaticArrays, DiffEqDevTools, Sundials, CairoMakie, LinearSolve,
    LinearAlgebra, StableRNGs, BenchmarkTools, Setfield
import PolyesterForwardDiff, MINPACK, NLsolve, SpeedMapping
RUS = RadiusUpdateSchemes;

f_oop(u, p) = u .* u .- p
f_iip(du, u, p) = @. du = u * u - p

function generate_prob(::Val{N}, ::Val{static}) where {N, static}
    u0_ = ones(N) .+ randn(StableRNG(0), N) * 0.01
    u0 = static ? SVector{N, Float64}(u0_) : u0_
    prob = static ? NonlinearProblem{false}(f_oop, u0, 2.0) :
           NonlinearProblem{true}(f_iip, u0, 2.0)
    return prob
end
```

```
generate_prob (generic function with 1 method)
```





List of Solvers to Benchmark:

```julia
solvers = [
    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "Default PolyAlgorithm",           solver = Dict(:alg => nothing)),

    (; pkg = :simplenonlinearsolve, full_pkgname = "SimpleNonlinearSolve.jl", name = "Simple Newton Raphson",           solver = Dict(:alg => SimpleNewtonRaphson())),
    (; pkg = :simplenonlinearsolve, full_pkgname = "SimpleNonlinearSolve.jl", name = "Simple DFSane",                   solver = Dict(:alg => SimpleDFSane())),
    (; pkg = :simplenonlinearsolve, full_pkgname = "SimpleNonlinearSolve.jl", name = "Simple Trust Region",             solver = Dict(:alg => SimpleTrustRegion(; nlsolve_update_rule = Val(true)))),
    (; pkg = :simplenonlinearsolve, full_pkgname = "SimpleNonlinearSolve.jl", name = "Simple Broyden",                  solver = Dict(:alg => SimpleBroyden())),
    (; pkg = :simplenonlinearsolve, full_pkgname = "SimpleNonlinearSolve.jl", name = "Simple Klement",                  solver = Dict(:alg => SimpleKlement())),

    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "Newton Raphson",                  solver = Dict(:alg => NewtonRaphson(; linsolve = \))),
    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "DFSane",                          solver = Dict(:alg => DFSane())),
    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "Trust Region",                    solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.NLsolve))),
    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "Broyden",                         solver = Dict(:alg => Broyden())),
    (; pkg = :nonlinearsolve,       full_pkgname = "NonlinearSolve.jl",       name = "Klement",                         solver = Dict(:alg => Klement(; linsolve = \))),

    (; pkg = :wrapper,              full_pkgname = "NLsolve.jl",              name = "Newton Raphson (NLsolve.jl)",     solver = Dict(:alg => NLsolveJL(; method = :newton, autodiff = :forward))),
    (; pkg = :wrapper,              full_pkgname = "NLsolve.jl",              name = "Trust Region (NLsolve.jl)",       solver = Dict(:alg => NLsolveJL(; autodiff = :forward))),
    (; pkg = :wrapper,              full_pkgname = "Sundials",                name = "Newton Raphson (Sundials)",       solver = Dict(:alg => KINSOL())),
    (; pkg = :wrapper,              full_pkgname = "SpeedMapping.jl",         name = "Speed Mapping (SpeedMapping.jl)", solver = Dict(:alg => SpeedMappingJL())),
];
```




Configurations:

```julia
abstols = 1.0 ./ 10.0 .^ (3:2:12)
reltols = 1.0 ./ 10.0 .^ (3:2:12)
```

```
5-element Vector{Float64}:
 0.001
 1.0e-5
 1.0e-7
 1.0e-9
 1.0e-11
```





Helper Functions:

```julia
function check_solver(prob, solver)
    prob.u0 isa StaticArray && solver.pkg === :wrapper && return false
    try
        sol = solve(prob, solver.solver[:alg]; abstol = 1e-5, reltol = 1e-5,
            maxiters = 10000)
        err = norm(sol.resid)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("[Warn] Solver $(solver.name) returned retcode $(sol.retcode) \
                              with an residual norm = $(norm(sol.resid)).\n"; color = :red)
            return false
        elseif err > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm \
                              = $(norm(sol.resid))).\n"; color = :red)
            return false
        elseif isinf(err) || isnan(err)
            Base.printstyled("[Warn] Solver $(solver.name) had a residual of $(err).\n";
                color = :red)
            return false
        end
        Base.printstyled("[Info] Solver $(solver.name) successfully solved the problem \
                          (norm = $(norm(sol.resid))).\n"; color = :green)
    catch e
        Base.printstyled("[Warn] Solver $(solver.name) threw an error: $e.\n"; color = :red)
        return false
    end
    return true
end

function generate_wpset(prob, solvers)
    successful_solvers = filter(solver -> check_solver(prob, solver), solvers)
    return WorkPrecisionSet(prob, abstols, reltols, getfield.(successful_solvers, :solver);
        numruns = 50, error_estimate = :l∞, names = getfield.(successful_solvers, :name),
        maxiters = 10000), successful_solvers
end
```

```
generate_wpset (generic function with 1 method)
```





Plotting Functions:

```julia
# This is hardcoded for 4 input length
function plot_all_wpsets(wpset_list, solver_all, titles, suptitle)
    @assert length(wpset_list) == length(titles) == 4
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1400
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:berlin, length(solver_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle,), Scatter = (; cycle,))

    fig = with_theme(plot_theme) do 
        fig = Figure(; size = (WIDTH, HEIGHT))
        # `textbf` doesn't work
        axs = Matrix{Any}(undef, 2, 2)

        for i in 1:2, j in 1:2
            ylabel = j == 1 ? L"$\mathrm{\mathbf{Time}}$ $\mathbf{(s)}$" : ""
            xlabel = i == 2 ? L"$\mathrm{\mathbf{Error}}$: $\mathbf{f(u^\ast)}$ $\mathbf{L_{\infty}}-\mathrm{\mathbf{norm}}$" : ""
            ax = Axis(fig[i, j]; ylabel = ylabel, title = titles[2 * (i - 1) + j],
                titlegap = 10, titlesize = 22, xlabelsize = 22, ylabelsize = 22,
                xticklabelsize = 20, yticklabelsize = 20,
                xlabel = xlabel, xticksvisible =  i == 2, yticksvisible = j == 1,
                xticklabelsvisible = i == 2, yticklabelsvisible = j == 1,
                xscale = log10, yscale = log10, xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH)
            axs[i, j] = ax
        end

        ls, scs = [], []
        label_list = []

        for (idx_solver, solver) in enumerate(reverse(solver_all))
            first_success = true
            for i in 1:2, j in 1:2
                wpset, successful_solvers = wpset_list[2 * (i - 1) + j]

                idx = findfirst(==(solver.name), [solver.name for solver in successful_solvers])
                idx === nothing && continue

                (; name, times, errors) = wpset.wps[idx]
                errors = [err.l∞ for err in errors]
                l = lines!(axs[i, j], errors, times; linestyle = LINESTYLES[solver.pkg],
                    label = name, linewidth = 3, color = colors[idx_solver])
                sc = scatter!(axs[i, j], errors, times; label = name, markersize = 16,
                    strokewidth = 1, marker = Cycled(idx_solver),
                    color = colors[idx_solver])

                if first_success
                    push!(ls, l)
                    push!(scs, sc)
                    push!(label_list, solver.name)
                    first_success = false
                end
            end
        end

        linkaxes!(axs...)

        Legend(fig[3, :], collect(reverse([[l, sc] for (l, sc) in zip(ls, scs)])),
            collect(reverse(label_list)),
            "Successful Solvers"; framevisible=true, framewidth = STROKEWIDTH,
            orientation = :horizontal,
            titlesize = 16, nbanks = 3, labelsize = 16,
            tellheight = true, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        fig[0, :] = Label(fig, suptitle, fontsize = 24, tellwidth = false, font = :bold)

        fig
    end

    return fig
end
```

```
plot_all_wpsets (generic function with 1 method)
```





## Static Array Versions

```julia
probs = [generate_prob(Val(N), Val(true)) for N in [2, 4, 8, 12]];
wpsets = [generate_wpset(prob, solvers) for prob in probs];
titles = ["N = 2", "N = 4", "N = 8", "N = 12"];

fig = plot_all_wpsets(wpsets, solvers, titles, "Static Arrays")
```

```
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 7.743550661734637e-6).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 8.903708005997684e-6).
[Info] Solver Simple DFSane successfully solved the problem (norm = 2.21391
17217408527e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 8
.903708005997684e-6).
[Info] Solver Simple Broyden successfully solved the problem (norm = 7.7435
50661734637e-6).
[Info] Solver Simple Klement successfully solved the problem (norm = 8.7705
08340460738e-6).
[Info] Solver Newton Raphson successfully solved the problem (norm = 8.9037
08005997684e-6).
[Info] Solver DFSane successfully solved the problem (norm = 8.925226244970
55e-6).
[Info] Solver Trust Region successfully solved the problem (norm = 8.903708
005997684e-6).
[Info] Solver Broyden successfully solved the problem (norm = 7.74355066173
4637e-6).
[Info] Solver Klement successfully solved the problem (norm = 7.74279203948
5169e-6).
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 1.0524944462494045e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 1.3175515250018797e-5).
[Info] Solver Simple DFSane successfully solved the problem (norm = 3.21318
9867783094e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 1
.3175515250018797e-5).
[Info] Solver Simple Broyden successfully solved the problem (norm = 1.0524
944462494045e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 1.2719
355158575668e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 1.3175
515250018797e-5).
[Info] Solver DFSane successfully solved the problem (norm = 1.367124759680
58e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 1.317551
5250018797e-5).
[Info] Solver Broyden successfully solved the problem (norm = 1.05249444624
94045e-5).
[Info] Solver Klement successfully solved the problem (norm = 1.04709031908
2685e-5).
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 1.4246490327013533e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 1.9046552208413456e-5).
[Info] Solver Simple DFSane successfully solved the problem (norm = 4.64939
18801382363e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 1
.9046552208413456e-5).
[Info] Solver Simple Broyden successfully solved the problem (norm = 1.4246
490327013533e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 1.8259
373704793516e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 1.9046
552208413456e-5).
[Info] Solver DFSane successfully solved the problem (norm = 1.973829931575
864e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 1.904655
2208413456e-5).
[Info] Solver Broyden successfully solved the problem (norm = 1.42464903270
13533e-5).
[Info] Solver Klement successfully solved the problem (norm = 1.41180265390
8467e-5).
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 1.928785498427112e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 2.2187014239665903e-5).
[Info] Solver Simple DFSane successfully solved the problem (norm = 5.40467
1405346579e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 2
.2187014239665903e-5).
[Info] Solver Simple Broyden successfully solved the problem (norm = 1.9287
85498427112e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 2.1607
111134434963e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 2.2187
014239665903e-5).
[Info] Solver DFSane successfully solved the problem (norm = 2.295755366158
96e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 2.218701
4239665903e-5).
[Info] Solver Broyden successfully solved the problem (norm = 1.92878549841
49875e-5).
[Info] Solver Klement successfully solved the problem (norm = 1.92365300994
5377e-5).
```


![](figures/quadratic_nonlinear_6_1.png)

```julia
save("static_arrays_quadratic.svg", fig)
```

```
CairoMakie.Screen{SVG}
```





## Regular Julia Arrays

```julia
probs = [generate_prob(Val(N), Val(false)) for N in [4, 16, 128, 1024]]
wpsets = [(@show length(prob.u0); generate_wpset(prob, solvers)) for prob in probs]
titles = ["N = 4", "N = 16", "N = 128", "N = 1024"]

fig = plot_all_wpsets(wpsets, solvers, titles, "Regular Julia Arrays")
```

```
length(prob.u0) = 4
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 1.0524944462889887e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 1.3175515250018797e-5).
[Info] Solver Simple DFSane successfully solved the problem (norm = 3.21318
9867783094e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 1
.3175515250018797e-5).
[Info] Solver Simple Broyden successfully solved the problem (norm = 1.0524
944462889887e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 1.2719
355158575668e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 1.3175
515250018797e-5).
[Info] Solver DFSane successfully solved the problem (norm = 1.367124759680
58e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 1.317551
5250018797e-5).
[Info] Solver Broyden successfully solved the problem (norm = 1.05249444628
89887e-5).
[Info] Solver Klement successfully solved the problem (norm = 1.04709031908
2685e-5).
[Info] Solver Newton Raphson (NLsolve.jl) successfully solved the problem (
norm = 1.3175515250018797e-5).
[Info] Solver Trust Region (NLsolve.jl) successfully solved the problem (no
rm = 1.3175515250018797e-5).
[Info] Solver Newton Raphson (Sundials) successfully solved the problem (no
rm = 3.507854466145342e-9).
[Info] Solver Speed Mapping (SpeedMapping.jl) successfully solved the probl
em (norm = 1.9601492832512963e-5).
length(prob.u0) = 16
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 2.3141389150698733e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 2.5044494662558595e-5).
[Info] Solver Simple DFSane successfully solved the problem (norm = 6.11269
711700824e-7).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 2
.5044494662558595e-5).
[Info] Solver Simple Broyden successfully solved the problem (norm = 2.3141
389150698733e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 2.4583
675020896853e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 2.5044
494662558595e-5).
[Info] Solver DFSane successfully solved the problem (norm = 2.578212922891
9075e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 2.504449
4662558595e-5).
[Info] Solver Broyden successfully solved the problem (norm = 2.31413891506
98733e-5).
[Info] Solver Klement successfully solved the problem (norm = 2.31146141902
84395e-5).
[Info] Solver Newton Raphson (NLsolve.jl) successfully solved the problem (
norm = 2.5044494662558595e-5).
[Info] Solver Trust Region (NLsolve.jl) successfully solved the problem (no
rm = 2.5044494662558595e-5).
[Info] Solver Newton Raphson (Sundials) successfully solved the problem (no
rm = 6.033406020887388e-9).
[Info] Solver Speed Mapping (SpeedMapping.jl) successfully solved the probl
em (norm = 5.21219858681873e-5).
length(prob.u0) = 128
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 6.545361092122421e-5).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 6.283767496044401e-11).
[Info] Solver Simple DFSane successfully solved the problem (norm = 1.73578
10548215615e-6).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 6
.283767496044401e-11).
[Info] Solver Simple Broyden successfully solved the problem (norm = 6.5453
61092122703e-5).
[Info] Solver Simple Klement successfully solved the problem (norm = 7.0177
89716845796e-5).
[Info] Solver Newton Raphson successfully solved the problem (norm = 6.2837
67496044401e-11).
[Info] Solver DFSane successfully solved the problem (norm = 7.685980582924
527e-5).
[Info] Solver Trust Region successfully solved the problem (norm = 6.283767
496044401e-11).
[Info] Solver Broyden successfully solved the problem (norm = 6.54536109212
2421e-5).
[Info] Solver Klement successfully solved the problem (norm = 6.53197265209
5864e-5).
[Info] Solver Newton Raphson (NLsolve.jl) successfully solved the problem (
norm = 6.283767496044401e-11).
[Info] Solver Trust Region (NLsolve.jl) successfully solved the problem (no
rm = 6.283767496044401e-11).
[Info] Solver Newton Raphson (Sundials) successfully solved the problem (no
rm = 2.0068313703523284e-8).
[Info] Solver Speed Mapping (SpeedMapping.jl) successfully solved the probl
em (norm = 5.551313074230362e-10).
length(prob.u0) = 1024
[Info] Solver Default PolyAlgorithm successfully solved the problem (norm =
 0.0001905299353528414).
[Info] Solver Simple Newton Raphson successfully solved the problem (norm =
 1.7311553941286643e-10).
[Info] Solver Simple DFSane successfully solved the problem (norm = 4.84115
0649457674e-6).
[Info] Solver Simple Trust Region successfully solved the problem (norm = 1
.7311553941286643e-10).
[Info] Solver Simple Broyden successfully solved the problem (norm = 0.0001
905299353529018).
[Info] Solver Simple Klement successfully solved the problem (norm = 0.0001
967925659287559).
[Info] Solver Newton Raphson successfully solved the problem (norm = 1.7311
553941286643e-10).
[Info] Solver DFSane successfully solved the problem (norm = 0.000216303516
23678866).
[Info] Solver Trust Region successfully solved the problem (norm = 1.731155
3941286643e-10).
[Info] Solver Broyden successfully solved the problem (norm = 0.00019052993
53528414).
[Info] Solver Klement successfully solved the problem (norm = 0.00019007289
321538545).
[Info] Solver Newton Raphson (NLsolve.jl) successfully solved the problem (
norm = 1.7311553941286643e-10).
[Info] Solver Trust Region (NLsolve.jl) successfully solved the problem (no
rm = 1.7311553941286643e-10).
[Info] Solver Newton Raphson (Sundials) successfully solved the problem (no
rm = 5.408987237602836e-8).
[Info] Solver Speed Mapping (SpeedMapping.jl) successfully solved the probl
em (norm = 2.009517968000003e-9).
```


![](figures/quadratic_nonlinear_8_1.png)

```julia
save("regular_arrays_quadratic.svg", fig)
```

```
CairoMakie.Screen{SVG}
```





## Comparison against SimpleNonlinearSolve

```julia
function __set_ad_chunksize(solver::GeneralizedFirstOrderAlgorithm{CJ, N}, ck) where {CJ, N}
    ad = AutoForwardDiff(; chunksize = ck)
    return GeneralizedFirstOrderAlgorithm{CJ, N}(; solver.descent, solver.linesearch,
        solver.trustregion, jacobian_ad = ad, solver.max_shrink_times, solver.forward_ad,
        solver.reverse_ad)
end
function __set_ad_chunksize(solver::SimpleNewtonRaphson, ck)
    solver.autodiff === nothing || return solver
    autodiff = AutoForwardDiff(; chunksize = ck)
    return SimpleNewtonRaphson(; autodiff)
end
function __set_ad_chunksize(solver::SimpleTrustRegion, ck)
    solver.autodiff === nothing || return solver
    autodiff = AutoForwardDiff(; chunksize = ck)
    return SimpleTrustRegion(; autodiff)
end
__set_ad_chunksize(solver, _) = solver

function benchmark_combinations(solvers, probs)
    return map(Iterators.product(solvers, probs)) do (solver, prob)
        try
            solver_concrete = prob.u0 isa SArray ? solver.solver[:alg] :
                __set_ad_chunksize(solver.solver[:alg],
                    min(NonlinearSolve.pickchunksize(prob.u0), 6))

            sol = solve(prob, solver_concrete; abstol = 1e-10, reltol = 1e-10,
                maxiters = 1000, termination_condition = AbsNormTerminationMode())
            @info "Solver $(solver.name) successfully solved the problem with norm = \
                $(norm(sol.resid, Inf))."
            
            if norm(sol.resid, Inf) ≤ 1e-10
                tt = @belapsed solve($prob, $solver_concrete; abstol = 1e-10,
                    reltol = 1e-10, maxiters = 1000)
                @info "Solver $(solver.name) took $(tt) seconds."
                return tt
            else
                return NaN
            end
        catch e
            @error "Solver $(solver.name) threw an error."
            return NaN
        end
    end
end

probs_sa = [generate_prob(Val(N), Val(true)) for N in [2, 4, 8, 12]];
solve_timings_sa = benchmark_combinations(solvers, probs_sa)

probs_reg = [generate_prob(Val(N), Val(false)) for N in [4, 16, 128, 1024]];
solve_timings_reg = benchmark_combinations(solvers, probs_reg)

function create_structure_data(solve_timings)
    df = []
    for (i, solver_setup) in enumerate(solvers)
        if solver_setup.pkg === :simplenonlinearsolve
            alg_name = string(solver_setup.name[8:end])
            timings = solve_timings[i, :]
            list = []
            for (j, other_solver) in enumerate(solvers)
                if other_solver.pkg !== :simplenonlinearsolve &&
                    contains(other_solver.name, alg_name)
                    this_timings = solve_timings[j, :]
                    push!(list, other_solver.name => this_timings ./ timings)
                end
            end
            push!(df, alg_name => list)
        end
    end
    return df
end

df_sa = create_structure_data(solve_timings_sa)
df_reg = create_structure_data(solve_timings_reg)

fig = begin
    ASPECT_RATIO = 0.7
    WIDTH = 1400
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    fig = Figure(; size = (WIDTH, HEIGHT))

    axs = Matrix{Any}(undef, 1, 4)

    xs = reduce(vcat, [fill(i, length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_sa)])
    dodge = reduce(vcat, [collect(1:length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_sa)])

    for i in 1:4
        ys = reduce(vcat, [[xx.second[i] for xx in dfᵢ.second] for dfᵢ in df_sa])

        ax = Axis(fig[1, i];
            ylabel = i == 1 ? L"Relative to $\mathbf{\mathrm{SimpleNonlinearSolve}}$" : "",
            title = L"N = %$(length(probs_sa[i].u0))$",
            titlegap = 10, xticksvisible = false, yticksvisible = true,
            xticklabelsvisible = false, yticklabelsvisible = true, titlesize = 22,
            spinewidth = STROKEWIDTH, xlabelsize = 22, ylabelsize = 22,
            xticklabelrotation = π / 4, xticklabelsize = 20, yticklabelsize = 20)
        axs[1, i] = ax

        barplot!(ax, xs, ys; dogde = dodge, color = dodge, colormap = :berlin,
            strokewidth = 2)

        hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 2)
    end

    linkaxes!(axs...)

    axs = Matrix{Any}(undef, 1, 4)

    xs = reduce(vcat, [fill(i, length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_reg)])
    dodge = reduce(vcat, [collect(1:length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_reg)])

    for i in 1:4
        ys = reduce(vcat, [[xx.second[i] for xx in dfᵢ.second] for dfᵢ in df_reg])

        ax = Axis(fig[2, i];
            ylabel = i == 1 ? L"Relative to $\mathbf{\mathrm{SimpleNonlinearSolve}}$" : "",
            title = L"N = %$(length(probs_reg[i].u0))$",
            titlegap = 10, xticksvisible = true, yticksvisible = true,
            xticklabelsvisible = true, yticklabelsvisible = true, titlesize = 22,
            spinewidth = STROKEWIDTH, xlabelsize = 22, ylabelsize = 22,
            xticks = (1:length(df_sa), [d.first for d in df_sa]),
            xticklabelrotation = π / 4, xticklabelsize = 20, yticklabelsize = 20)
        axs[1, i] = ax

        barplot!(ax, xs, ys; color = dodge, dodge = dodge, colormap = :berlin,
            strokewidth = 2)

        hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 2)
    end

    linkaxes!(axs...)

    fig[0, :] = Label(fig, "Simple Algorithms on Quadratic Root-Finding Problem",
        fontsize = 24, font = :bold)
    fig[1, 0] = Label(fig, "Static Arrays", fontsize = 24, rotation = π / 2,
        tellheight = false)
    fig[2, 0] = Label(fig, "Regular Julia Arrays", fontsize = 24, rotation = π / 2,
        tellheight = false)

    labels = ["NonlinearSolve.jl", "NLsolve.jl", "Sundials"]
    colors = cgrad(:berlin, length(labels); categorical = true)
    elements = [PolyElement(polycolor = colors[i]) for i in 1:3]
    axislegend(axs[1, 4], elements, labels, "Package",
        labelsize = 16, titlesize = 16, framewidth = STROKEWIDTH)

    fig
end
```

![](figures/quadratic_nonlinear_10_1.png)

```julia
save("summary_plot_simplenonlinearsolve.svg", fig)
```

```
CairoMakie.Screen{SVG}
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonlinearProblem","quadratic_nonlinear.jmd")
```

Computer Information:

```
Julia Version 1.10.1
Commit 7790d6f0641 (2024-02-13 20:41 UTC)
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
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.4.0
  [13f3f980] CairoMakie v0.11.8
  [2b5f629d] DiffEqBase v6.147.0
  [f3b72e0c] DiffEqDevTools v2.44.2
  [b964fa9f] LaTeXStrings v1.3.1
  [7ed4a6bd] LinearSolve v2.23.4
  [4854310b] MINPACK v1.2.0
  [2774e3e8] NLsolve v4.5.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
  [8913a72c] NonlinearSolve v3.5.6
  [98d1487c] PolyesterForwardDiff v0.1.1
  [08abe8d2] PrettyTables v2.3.1
  [31c91b34] SciMLBenchmarks v0.1.3
  [efcf1570] Setfield v1.1.1
  [727e6d20] SimpleNonlinearSolve v1.4.1
  [47a9eef4] SparseDiffTools v2.17.0
  [f1835b91] SpeedMapping v0.3.0
  [860ef19b] StableRNGs v1.0.1
  [90137ffa] StaticArrays v1.9.2
  [c3572dad] Sundials v4.24.0
  [0c5d862f] Symbolics v5.18.0
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Manifest.toml`
  [47edcb42] ADTypes v0.2.6
  [a4c015fc] ANSIColoredPrinters v0.0.1
  [621f4979] AbstractFFTs v1.5.0
  [398f06c4] AbstractLattices v0.3.0
  [1520ce14] AbstractTrees v0.4.4
  [22286c92] AccurateArithmetic v0.3.8
  [79e6a3ab] Adapt v4.0.1
  [27a7e980] Animations v0.4.1
⌅ [ec485272] ArnoldiMethod v0.2.0
  [4fba245c] ArrayInterface v7.7.1
  [4c555306] ArrayLayouts v1.6.0
  [67c07d97] Automa v1.0.3
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.7
  [6e4b80f9] BenchmarkTools v1.4.0
  [e2ed5e7c] Bijections v0.1.6
  [d1d4a3ce] BitFlags v0.1.8
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.4
  [96374032] CRlibm v1.0.1
  [159f3aea] Cairo v1.0.5
  [13f3f980] CairoMakie v0.11.8
  [49dc2e85] Calculus v0.5.1
  [d360d2e6] ChainRulesCore v1.21.1
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [523fee87] CodecBzip2 v0.8.2
  [944b1d66] CodecZlib v0.7.4
  [a2cac450] ColorBrewer v0.4.0
  [35d6a980] ColorSchemes v3.24.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.12.0
  [b152e2b5] CompositeTypes v0.1.3
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.3.1
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.4
  [d38c429a] Contour v0.6.2
  [a2441757] Coverage v1.6.0
  [c36e975a] CoverageTools v1.3.0
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [864edb3b] DataStructures v0.18.16
  [e2d170a0] DataValueInterfaces v1.0.0
  [927a84f5] DelaunayTriangulation v0.8.12
  [2b5f629d] DiffEqBase v6.147.0
  [f3b72e0c] DiffEqDevTools v2.44.2
  [77a26b50] DiffEqNoiseProcess v5.20.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.11
  [31c24e10] Distributions v0.25.107
  [ffbed154] DocStringExtensions v0.9.3
  [e30172f5] Documenter v1.2.1
  [35a29f4d] DocumenterTools v0.1.19
  [5b8099bc] DomainSets v0.7.3
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.5.5
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.6.5
  [429591f6] ExactPredicates v2.2.8
  [460bff9d] ExceptionUnwrapping v0.1.10
  [e2ba6199] ExprTools v0.1.10
  [411431e0] Extents v0.1.2
  [7a1cc6ca] FFTW v1.8.0
  [7034ab61] FastBroadcast v0.2.8
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.1
  [5789e2e9] FileIO v1.16.2
  [8fc22ac5] FilePaths v0.8.3
  [48062228] FilePathsBase v0.9.21
  [1a297f60] FillArrays v1.9.3
  [6a86dc24] FiniteDiff v2.22.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.36
  [b38be410] FreeType v4.1.1
  [663a7486] FreeTypeAbstraction v0.10.1
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.1.6
  [cf35fbd7] GeoInterface v1.3.3
  [5c1252a2] GeometryBasics v0.4.10
  [d7ba0133] Git v1.3.0
  [a2bd30eb] Graphics v1.1.2
  [86223c79] Graphs v1.9.0
  [3955a311] GridLayoutBase v0.10.0
  [42e2da0e] Grisu v1.0.2
  [708ec375] Gumbo v0.8.2
  [cd3eb016] HTTP v1.10.2
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [b5f81e59] IOCapture v0.2.4
  [615f187c] IfElse v0.1.1
  [2803e5a7] ImageAxes v0.6.11
  [c817782e] ImageBase v0.1.7
  [a09fc81d] ImageCore v0.10.2
  [82e4d734] ImageIO v0.6.7
  [bc367c6b] ImageMetadata v0.9.9
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.4
  [18e54dd8] IntegerMathUtils v0.1.2
  [a98d9a8b] Interpolations v0.15.1
⌅ [d1acc4aa] IntervalArithmetic v0.22.5
  [8197267c] IntervalSets v0.7.10
  [92d709cd] IrrationalConstants v0.2.2
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [b835a17e] JpegTurbo v0.1.5
  [ef3ab10e] KLU v0.5.0
  [5ab0869b] KernelDensity v0.6.8
  [ba0b0d4f] Krylov v0.9.5
  [b964fa9f] LaTeXStrings v1.3.1
  [2ee39098] LabelledArrays v1.15.1
  [984bce1d] LambertW v0.4.6
  [23fbe1c1] Latexify v0.16.1
  [10f19ff3] LayoutPointers v0.1.15
  [0e77f7df] LazilyInitializedFields v1.2.2
  [50d2b5c4] Lazy v0.15.1
  [5078a376] LazyArrays v1.8.3
  [8cdb02fc] LazyModules v0.3.1
  [9c8b4983] LightXML v0.9.1
  [d3d80556] LineSearches v7.2.0
  [9b3f67b0] LinearAlgebraX v0.2.7
  [7ed4a6bd] LinearSolve v2.23.4
  [2ab3a3ac] LogExpFunctions v0.3.27
  [e6f89c97] LoggingExtras v1.0.3
  [bdcacae8] LoopVectorization v0.12.166
  [4854310b] MINPACK v1.2.0
  [1914dd2f] MacroTools v0.5.13
  [ee78f7c6] Makie v0.20.7
  [20f20a25] MakieCore v0.7.3
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
  [d0879d2d] MarkdownAST v0.1.2
  [b8f27783] MathOptInterface v1.25.3
  [0a4f8689] MathTeXEngine v0.5.7
  [a3b82374] MatrixFactorizations v2.1.0
  [bb5d69b7] MaybeInplace v0.1.1
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.1.0
  [7475f97c] Mods v2.2.4
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.4
  [3b2b4ff1] Multisets v0.4.4
  [102ac46a] MultivariatePolynomials v0.5.4
  [ffc61752] Mustache v1.0.19
  [d8a4904e] MutableArithmetics v1.4.1
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.2
  [f09324ee] Netpbm v1.1.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
  [8913a72c] NonlinearSolve v3.5.6
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.13.0
  [52e1d378] OpenEXR v0.3.2
  [4d8831e6] OpenSSL v1.4.1
  [429524aa] Optim v1.9.2
  [bac558e1] OrderedCollections v1.6.3
  [90014a1f] PDMats v0.11.31
  [f57f5aa1] PNGFiles v0.4.3
  [65ce6f38] PackageExtensionCompat v1.0.2
  [19eb6ba3] Packing v0.5.0
  [5432bcbf] PaddedViews v0.5.12
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [2ae35dd2] Permutations v0.4.20
  [3bbf5609] PikaParser v0.6.1
  [eebad327] PkgVersion v0.3.3
  [995b91a9] PlotUtils v1.4.0
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.9
  [98d1487c] PolyesterForwardDiff v0.1.1
  [1d0040c9] PolyesterWeave v0.2.1
  [647866c9] PolygonOps v0.1.2
  [f27b6e38] Polynomials v4.0.6
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.20
  [aea7be01] PrecompileTools v1.2.0
  [21216c6a] Preferences v1.4.1
  [08abe8d2] PrettyTables v2.3.1
  [27ebfcd6] Primes v0.5.5
  [92933f4c] ProgressMeter v1.9.0
  [4b34888f] QOI v1.0.0
  [1fd47b50] QuadGK v2.9.4
  [74087812] Random123 v1.6.2
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
  [731186ca] RecursiveArrayTools v3.8.1
  [f2c3362d] RecursiveFactorization v0.2.21
  [189a3867] Reexport v1.2.2
  [2792f1a3] RegistryInstances v0.1.0
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [286e9d63] RingLists v0.2.8
  [79098fc4] Rmath v0.7.1
  [47965b36] RootedTrees v2.21.0
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.12
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
  [322a6be2] Sass v0.2.0
  [0bca4576] SciMLBase v2.26.0
  [31c91b34] SciMLBenchmarks v0.1.3
  [c0aeaf25] SciMLOperators v0.3.7
  [6c6a2e73] Scratch v1.2.1
  [efcf1570] Setfield v1.1.1
  [65257c39] ShaderAbstractions v0.4.1
  [992d4aef] Showoff v1.0.3
  [73760f76] SignedDistanceFields v0.4.0
  [777ac1f9] SimpleBufferStream v1.1.0
  [55797a34] SimpleGraphs v0.8.6
  [727e6d20] SimpleNonlinearSolve v1.4.1
  [ec83eff0] SimplePartitions v0.3.1
  [cc47b68c] SimplePolynomials v0.2.17
  [a6525b86] SimpleRandom v0.3.1
  [699a6c99] SimpleTraits v0.9.4
  [45858cf5] Sixel v0.1.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.17.0
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.1
  [f1835b91] SpeedMapping v0.3.0
  [c5dd0088] StableHashTraits v1.1.8
  [860ef19b] StableRNGs v1.0.1
  [cae243ae] StackViews v0.1.1
  [aedffcd0] Static v0.8.10
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.2
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.2
  [4c63d2b9] StatsFuns v1.3.1
  [7792a7ef] StrideArraysCore v0.5.2
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [09ab397b] StructArrays v0.6.17
  [c3572dad] Sundials v4.24.0
  [2efcf032] SymbolicIndexingInterface v0.3.6
  [d1185830] SymbolicUtils v1.5.0
  [0c5d862f] Symbolics v5.18.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [62fd8b95] TensorCore v0.1.1
  [8290d209] ThreadingUtilities v0.5.2
⌅ [731e570b] TiffImages v0.6.8
  [a759f4b9] TimerOutputs v0.5.23
  [3bb67fe8] TranscodingStreams v0.10.3
  [d5829a12] TriangularSolve v0.1.20
  [410a4b4d] Tricks v0.1.8
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [9d95972d] TupleTools v1.5.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [a7c27f48] Unityper v0.1.6
  [3d5dd08c] VectorizationBase v0.21.65
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [efce3f68] WoodburyMatrices v1.0.0
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [6e34b625] Bzip2_jll v1.0.8+1
  [4e9b3aee] CRlibm_jll v1.0.1+0
  [83423d85] Cairo_jll v1.16.1+1
  [5ae413db] EarCut_jll v2.2.4+0
  [2e619515] Expat_jll v2.5.0+0
⌃ [b22a6f82] FFMPEG_jll v4.4.2+2
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [78b55507] Gettext_jll v0.21.0+0
⌃ [f8c6e375] Git_jll v2.36.1+2
  [7746bdde] Glib_jll v2.76.5+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [528830af] Gumbo_jll v0.10.2+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [905a6f67] Imath_jll v3.1.7+0
  [1d5cc7b8] IntelOpenMP_jll v2024.0.2+0
  [aacddb02] JpegTurbo_jll v3.0.1+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.35.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2024.0.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [18a262bb] OpenEXR_jll v3.1.4+0
⌅ [9bd350c2] OpenSSH_jll v8.9.0+1
⌅ [458c3c95] OpenSSL_jll v1.1.23+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [36c8627f] Pango_jll v1.50.14+0
  [30392449] Pixman_jll v0.42.2+0
  [f50d1b31] Rmath_jll v0.4.0+0
⌅ [fb77eaff] Sundials_jll v5.2.2+0
  [02c8fc9c] XML2_jll v2.12.2+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [9a68df92] isoband_jll v0.2.3+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.42+0
  [47bcb7c8] libsass_jll v3.6.4+0
  [075b6546] libsixel_jll v1.10.3+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
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
  [e66e0078] CompilerSupportLibraries_jll v1.1.0+0
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
  [8e850b90] libblastrampoline_jll v5.8.0+1
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

