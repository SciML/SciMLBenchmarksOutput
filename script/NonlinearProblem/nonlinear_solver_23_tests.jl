
using NonlinearSolve, LinearSolve, StaticArrays, Sundials, Setfield,
    BenchmarkTools, LinearAlgebra, DiffEqDevTools, NonlinearProblemLibrary, CairoMakie,
    RecursiveFactorization, Enzyme
import PolyesterForwardDiff, MINPACK, NLsolve, LineSearches

const RUS = RadiusUpdateSchemes;


HagerZhang() = LineSearchesJL(; method = LineSearches.HagerZhang())
MoreThuente() = LineSearchesJL(; method = LineSearches.MoreThuente())

solvers_all = [
    (; pkg = :nonlinearsolve, type = :general, name = "Default PolyAlg.",          solver = Dict(:alg => FastShortcutNonlinearPolyalg(; u0_len = 10, linsolve = \))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "Newton Raphson",            solver = Dict(:alg => NewtonRaphson(; linsolve = \))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (HagerZhang)",           solver = Dict(:alg => NewtonRaphson(; linsolve = \, linesearch = HagerZhang()))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (MoreThuente)",          solver = Dict(:alg => NewtonRaphson(; linsolve = \, linesearch = MoreThuente()))),
    (; pkg = :nonlinearsolve, type = :NR,      name = "NR (BackTracking)",         solver = Dict(:alg => NewtonRaphson(; linsolve = \, linesearch = BackTracking()))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "Trust Region",              solver = Dict(:alg => TrustRegion(; linsolve = \))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (NLsolve Update)",       solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.NLsolve))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Nocedal Wright)",       solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.NocedalWright))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Hei)",                  solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.Hei))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Yuan)",                 solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.Yuan))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Bastin)",               solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.Bastin))),
    (; pkg = :nonlinearsolve, type = :TR,      name = "TR (Fan)",                  solver = Dict(:alg => TrustRegion(; linsolve = \, radius_update_scheme = RUS.Fan))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "Levenberg-Marquardt",       solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization()))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM with Cholesky",          solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization()))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (α_geodesic=0.5)",       solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization(), α_geodesic=0.5))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (α_geodesic=0.5) Chol.", solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization(), α_geodesic=0.5))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (no Accln.)",            solver = Dict(:alg => LevenbergMarquardt(; linsolve = QRFactorization(), disable_geodesic = Val(true)))),
    (; pkg = :nonlinearsolve, type = :LM,      name = "LM (no Accln.) Chol.",      solver = Dict(:alg => LevenbergMarquardt(; linsolve = CholeskyFactorization(), disable_geodesic = Val(true)))),
    (; pkg = :nonlinearsolve, type = :general, name = "Pseudo Transient",          solver = Dict(:alg => PseudoTransient(; linsolve = \, alpha_initial=10.0))),
    (; pkg = :wrapper,        type = :general, name = "Powell [MINPACK]",          solver = Dict(:alg => CMINPACK(; method=:hybr))),
    (; pkg = :wrapper,        type = :LM,      name = "LM [MINPACK]",              solver = Dict(:alg => CMINPACK(; method=:lm))),
    (; pkg = :wrapper,        type = :NR,      name = "NR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL(; method=:newton))),
    (; pkg = :wrapper,        type = :TR,      name = "TR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL())),
    (; pkg = :wrapper,        type = :NR,      name = "NR [Sundials]",             solver = Dict(:alg => KINSOL(; linear_solver = :LapackDense, maxsetupcalls=1))),
    (; pkg = :wrapper,        type = :NR,      name = "NR LineSearch [Sundials]",  solver = Dict(:alg => KINSOL(; linear_solver = :LapackDense, globalization_strategy=:LineSearch, maxsetupcalls=1)))
];

solver_tracker = [];
wp_general_tracker = [];


abstols = 1.0 ./ 10.0 .^ (4:12)
reltols = 1.0 ./ 10.0 .^ (4:12);


function set_ad_chunksize(solvers, u0)
    ck = NonlinearSolve.pickchunksize(u0)
    for i in eachindex(solvers)
        @set! solvers[i].solver[:alg] = __set_ad_chunksize(solvers[i].solver[:alg], ck, length(u0))
    end
    return solvers
end

function __set_ad_chunksize(solver::GeneralizedFirstOrderAlgorithm, ck, N)
    if N > ck
        ad = AutoPolyesterForwardDiff(; chunksize = ck)
    else
        ad = AutoForwardDiff(; chunksize = ck)
    end
    return GeneralizedFirstOrderAlgorithm(; solver.descent, solver.linesearch,
        solver.trustregion, jvp_autodiff = ad, solver.max_shrink_times, solver.vjp_autodiff,
        concrete_jac = solver.concrete_jac, name = solver.name, autodiff = ad)
end
function __set_ad_chunksize(solver::NonlinearSolvePolyAlgorithm, ck, N)
    algs = [__set_ad_chunksize(alg, ck, N) for alg in solver.algs]
    return NonlinearSolvePolyAlgorithm(algs; solver.start_index)
end
__set_ad_chunksize(solver, ck, N) = solver

# Benchmarks a specific problem, checks which solvers can solve it and their performance
function benchmark_problem!(prob_name; solver_tracker=solver_tracker)
    # Finds the problem and the true solution.
    prob = nlprob_23_testcases[prob_name]

    # Finds the solvers that can solve the problem
    solvers_concrete = set_ad_chunksize(solvers_all, prob.prob.u0);
    successful_solvers = filter(solver -> check_solver(prob, solver), solvers_concrete);
    push!(solver_tracker, prob_name => successful_solvers);

    # Handles the non-general cases.
    solvers_NR = filter(s -> s.type==:NR, successful_solvers)
    solvers_TR = filter(s -> s.type==:TR, successful_solvers)
    solvers_LM = filter(s -> s.type==:LM, successful_solvers)
    wp_NR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_NR, :solver);
        names=getfield.(solvers_NR, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))
    wp_TR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_TR, :solver);
        names=getfield.(solvers_TR, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))
    wp_LM = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_LM, :solver);
        names=getfield.(solvers_LM, :name), numruns=100, error_estimate=:l∞,
        maxiters=1000,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))

    # Handles the general case
    solvers_general = filter(s -> s.type==:general, successful_solvers)
    add_solver!(solvers_general, nothing, solvers_TR, wp_TR)
    add_solver!(solvers_general, nothing, solvers_LM, wp_LM)
    add_solver!(solvers_general, nothing, solvers_NR, wp_NR)

    wp_general = WorkPrecisionSet(prob.prob, abstols, reltols,
        getfield.(solvers_general, :solver); names=getfield.(solvers_general, :name),
        numruns=100, error_estimate=:l∞, maxiters=1000,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))

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
            termination_condition=AbsNormTerminationMode(Base.Fix1(maximum, abs)))
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

    colors = cgrad(:tableau_20, length(solvers_all); categorical = true)
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


benchmark_problem!("Generalized Rosenbrock function")


benchmark_problem!("Powell singular function")


benchmark_problem!("Powell badly scaled function")


benchmark_problem!("Wood function")


benchmark_problem!("Helical valley function")


benchmark_problem!("Watson function")


benchmark_problem!("Chebyquad function")


benchmark_problem!("Brown almost linear function")


benchmark_problem!("Discrete boundary value function")


benchmark_problem!("Discrete integral equation function")


benchmark_problem!("Trigonometric function")


benchmark_problem!("Variably dimensioned function")


benchmark_problem!("Broyden tridiagonal function")


benchmark_problem!("Broyden banded function")


benchmark_problem!("Hammarling 2 by 2 matrix square root problem")


benchmark_problem!("Hammarling 3 by 3 matrix square root problem")


benchmark_problem!("Dennis and Schnabel 2 by 2 example")


benchmark_problem!("Sample problem 18")


benchmark_problem!("Sample problem 19")


benchmark_problem!("Scalar problem f(x) = x(x - 5)^2")


benchmark_problem!("Freudenstein-Roth function")


benchmark_problem!("Boggs function")


benchmark_problem!("Chandrasekhar function")


solver_successes = [(solver.name in getfield.(prob[2], :name)) ? "O" : "X" for prob in solver_tracker, solver in solvers_all]
total_successes = [sum(solver_successes[:,i] .== "O") for i in 1:length(solvers_all)]
solver_outcomes = vcat(total_successes', solver_successes);


using PrettyTables
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, solver_outcomes; backend = Val(:html), header = getfield.(solvers_all, :name), alignment=:c)
println(io, "```")
Base.Text(String(take!(io)))


fig = begin
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 1
    WIDTH = 1800
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:tableau_20, length(solvers_all); categorical = true)
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
            labels[ordering], "Solvers";
            framevisible=true, framewidth = STROKEWIDTH, orientation = :horizontal,
            titlesize = 20, nbanks = 9, labelsize = 20, halign = :center,
            tellheight = false, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        return fig
    end
end


save("summary_wp_23test_problems.svg", fig)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

