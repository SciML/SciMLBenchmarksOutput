
using NonlinearSolve, StaticArrays, DiffEqDevTools, Sundials, CairoMakie, LinearSolve,
    LinearAlgebra, StableRNGs, BenchmarkTools, Setfield, RecursiveFactorization
import PolyesterForwardDiff, MINPACK, NLsolve, SpeedMapping

const RUS = RadiusUpdateSchemes;
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5;

f_oop(u, p) = u .* u .- p
f_iip(du, u, p) = @. du = u * u - p

function generate_prob(::Val{N}, ::Val{static}) where {N, static}
    u0_ = ones(N) .+ randn(StableRNG(0), N) * 0.01
    u0 = static ? SVector{N, Float64}(u0_) : u0_
    prob = static ?
           NonlinearProblem{false}(f_oop, u0, 2.0) :
           NonlinearProblem{true}(f_iip, u0, 2.0)
    return prob
end


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


abstols = 1.0 ./ 10.0 .^ (3:2:12)
reltols = 1.0 ./ 10.0 .^ (3:2:12)


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


# This is hardcoded for 4 input length
function plot_all_wpsets(wpset_list, solver_all, titles, suptitle)
    @assert length(wpset_list) == length(titles) == 4
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1400
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:tableau_20, length(solver_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle,), Scatter = (; cycle,))

    fig = with_theme(plot_theme) do 
        fig = Figure(; size = (WIDTH, HEIGHT))
        # `textbf` doesn't work
        axs = Matrix{Any}(undef, 2, 2)

        for i in 1:2, j in 1:2
            ylabel = j == 1 ? L"$\mathrm{\mathbf{Time}}$ $\mathbf{(s)}$" : ""
            xlabel = i == 2 ? L"Error: $\mathbf{||f(u^\ast)||_\infty}$" : ""
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

        fig[0, :] = Label(fig, "Quadratic Problem with $(suptitle): Work Precision Diagram", fontsize = 24, tellwidth = false, font = :bold)

        fig
    end

    return fig
end


probs = [generate_prob(Val(N), Val(true)) for N in [2, 4, 8, 12]];
wpsets = [generate_wpset(prob, solvers) for prob in probs];
titles = ["N = 2", "N = 4", "N = 8", "N = 12"];

fig = plot_all_wpsets(wpsets, solvers, titles, "Static Arrays")


save("static_arrays_quadratic.svg", fig)


probs = [generate_prob(Val(N), Val(false)) for N in [4, 16, 128, 1024]]
wpsets = [(@show length(prob.u0); generate_wpset(prob, solvers)) for prob in probs]
titles = ["N = 4", "N = 16", "N = 128", "N = 1024"]

fig = plot_all_wpsets(wpsets, solvers, titles, "Regular Julia Arrays")


save("regular_arrays_quadratic.svg", fig)


function benchmark_combinations(solvers, probs)
    return map(Iterators.product(solvers, probs)) do (solver, prob)
        try
            solver_concrete = solver.solver[:alg]
            termination_condition = NonlinearSolveBase.AbsNormTerminationMode(
                Base.Fix1(maximum, abs))
            sol = solve(prob, solver_concrete; abstol = 1e-10, reltol = 1e-10,
                maxiters = 1000, termination_condition)
            @info "Solver $(solver.name) successfully solved the problem with norm = \
                $(norm(sol.resid, Inf))."
            
            if norm(sol.resid, Inf) ≤ 1e-10
                tt = @belapsed solve($prob, $solver_concrete; abstol = 1e-10,
                    reltol = 1e-10, maxiters = 1000,
                    termination_condition = $termination_condition)
                @info "Solver $(solver.name) took $(tt) seconds."
                return tt
            else
                return NaN
            end
        catch e
            @error "Solver $(solver.name) threw an error $(e)."
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
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    fig = Figure(; size = (WIDTH, HEIGHT))

    axs = Matrix{Any}(undef, 1, 4)

    xs = reduce(vcat, [fill(i, length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_sa)])
    dodge = reduce(vcat, [collect(1:length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_sa)])

    for i in 1:4
        ys = reduce(vcat, [[xx.second[i] for xx in dfᵢ.second] for dfᵢ in df_sa])

        ax = Axis(fig[1, i];
            ylabel = "",
            title = L"$N = %$(length(probs_sa[i].u0))$",
            titlegap = 10, xticksvisible = false, yticksvisible = true,
            xticklabelsvisible = false, yticklabelsvisible = true, titlesize = 22,
            spinewidth = STROKEWIDTH, xlabelsize = 22, ylabelsize = 22,
            xticklabelrotation = π / 4, xticklabelsize = 20, yticklabelsize = 20)
        axs[1, i] = ax

        barplot!(ax, xs, ys; color = dodge, colormap = :tableau_20, strokewidth = 2)

        hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 2)
    end

    linkaxes!(axs...)

    axs = Matrix{Any}(undef, 1, 4)

    xs = reduce(vcat, [fill(i, length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_reg)])
    dodge = reduce(vcat, [collect(1:length(dfᵢ.second)) for (i, dfᵢ) in enumerate(df_reg)])

    for i in 1:4
        ys = reduce(vcat, [[xx.second[i] for xx in dfᵢ.second] for dfᵢ in df_reg])

        ax = Axis(fig[2, i];
            ylabel = "",
            title = L"$N = %$(length(probs_reg[i].u0))$",
            titlegap = 10, xticksvisible = true, yticksvisible = true,
            xticklabelsvisible = true, yticklabelsvisible = true, titlesize = 22,
            spinewidth = STROKEWIDTH, xlabelsize = 22, ylabelsize = 22,
            xticks = (1:length(df_sa), [d.first for d in df_sa]),
            xticklabelrotation = π / 4, xticklabelsize = 20, yticklabelsize = 20)
        axs[1, i] = ax

        barplot!(ax, xs, ys; color = dodge, dodge = dodge, colormap = :tableau_20,
            strokewidth = 3)

        hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 3)
    end

    linkaxes!(axs...)

    fig[0, :] = Label(fig, "Simple Algorithms on Quadratic Root-Finding Problem",
        fontsize = 24, font = :bold)
    fig[1, 0] = Label(fig, "Static Arrays",
        fontsize = 24, rotation = π / 2, tellheight = false)
    fig[2, 0] = Label(fig, "Regular Julia Arrays",
        fontsize = 24, rotation = π / 2, tellheight = false)
    fig[1:2, end + 1] = Label(fig, "Relative to SimpleNonlinearSolve.jl (Higher is Better)",
        fontsize = 22, tellheight = false, rotation = π / 2)

    labels = ["NonlinearSolve.jl", "NLsolve.jl", "Sundials"]
    colors = cgrad(:tableau_20, length(labels); categorical = true)
    elements = [PolyElement(; polycolor = colors[i], strokewidth = 3) for i in 1:3]
    axislegend(axs[1, 4], elements, labels, "Package", patchsize = (20, 20),
        labelsize = 16, titlesize = 20, framewidth = STROKEWIDTH, rowgap = 5)

    fig
end


save("summary_plot_simplenonlinearsolve.svg", fig)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

