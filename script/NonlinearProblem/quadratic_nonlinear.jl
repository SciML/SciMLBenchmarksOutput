
using NonlinearSolve, StaticArrays, DiffEqDevTools, Sundials, CairoMakie, LinearSolve,
    LinearAlgebra
import MINPACK, NLsolve, SpeedMapping

f_oop(u, p) = u .* u .- p
f_iip(du, u, p) = (du .= u .* u .- p)

function generate_prob(::Val{N}, ::Val{static}) where {N, static}
    u0 = static ? ones(SVector{N, Float64}) : ones(N)
    prob = static ? NonlinearProblem{false}(f_oop, u0, 2.0) :
           NonlinearProblem{true}(f_iip, u0, 2.0)
    return prob
end


solvers = [
    (; type = :simplenonlinearsolve, solver = Dict(:alg => SimpleNewtonRaphson()), name = "SimpleNewtonRaphson"),
    (; type = :simplenonlinearsolve, solver = Dict(:alg => SimpleTrustRegion()), name = "SimpleTrustRegion"),
    (; type = :simplenonlinearsolve, solver = Dict(:alg => SimpleKlement()), name = "SimpleKlement"),
    (; type = :simplenonlinearsolve, solver = Dict(:alg => SimpleDFSane()), name = "SimpleDFSane"),
    (; type = :simplenonlinearsolve, solver = Dict(:alg => SimpleBroyden()), name = "SimpleBroyden"),
    (; type = :nonlinearsolve, solver = Dict(:alg => NewtonRaphson()), name = "NewtonRaphson"),
    (; type = :nonlinearsolve, solver = Dict(:alg => TrustRegion()), name = "TrustRegion"),
    (; type = :nonlinearsolve, solver = Dict(:alg => Broyden()), name = "Broyden"),
    (; type = :nonlinearsolve, solver = Dict(:alg => Klement()), name = "Klement"),
    (; type = :nonlinearsolve, solver = Dict(:alg => DFSane()), name = "DFSane"),
    (; type = :others, solver = Dict(:alg => KINSOL()), name = "Newton's Method (Sundials)"),
    (; type = :others, solver = Dict(:alg => CMINPACK(; method = :lm)), name = "Levenberg-Marquardt (CMINPACK)"),
    (; type = :others, solver = Dict(:alg => CMINPACK(; method = :hybr)), name = "Hybrid Powell (CMINPACK)"),
    (; type = :others, solver = Dict(:alg => SpeedMappingJL()), name = "SpeedMapping"),
    (; type = :others, solver = Dict(:alg => NLsolveJL()), name = "NLsolve"),
]


abstols = 1.0 ./ 10.0 .^ (3:2:12)
reltols = 1.0 ./ 10.0 .^ (3:2:12)


function check_solver(prob, solver)
    try
        sol = solve(prob, solver.solver[:alg]; abstol = 1e-5, reltol = 1e-5,
            maxiters = 10000)
        err = norm(sol.resid)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("[Warn] Solver $(solver.name) returned retcode $(sol.retcode) with an residual norm = $(norm(sol.resid)).\n";
                color = :red)
            return false
        elseif err > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm = $(norm(sol.resid))).\n";
                color = :red)
            return false
        elseif isinf(err) || isnan(err)
            Base.printstyled("[Warn] Solver $(solver.name) had a residual of $(err).\n";
                color = :red)
            return false
        end
        Base.printstyled("[Info] Solver $(solver.name) successfully solved the problem (norm = $(norm(sol.resid))).\n";
            color = :green)
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


function plot_wpset(wpset, successful_solvers, title)
    cycle = Cycle([:color, :marker], covary = true)
    theme = merge(theme_latexfonts(), Theme(Lines = (cycle = cycle,),
        Scatter = (cycle = cycle,)))

    linestyle = Dict(:simplenonlinearsolve => :solid, :nonlinearsolve => :dash,
        :others => :dot)

    fig = with_theme(theme; fontsize = 32) do 
        fig = Figure(; size = (1400, 1200))
        ax = Axis(fig[1, 1]; ylabel = L"Time ($s$)", title, xscale = log10, yscale = log10,
            xlabel = L"Error: $f(x^\ast)$ $L_{\infty}$-norm")

        ls, scs = [], []
        
        for (wp, solver) in zip(wpset.wps, successful_solvers)
            (; name, times, errors) = wp
            errors = [err.l∞ for err in errors]
            l = lines!(ax, errors, times; label = name, linewidth = 3, linestyle = linestyle[solver.type])
            sc = scatter!(ax, errors, times; label = name, markersize = 16, strokewidth = 3)
            push!(ls, l)
            push!(scs, sc)
        end

        Legend(fig[2, 1], [[l, sc] for (l, sc) in zip(ls, scs)],
            [solver.name for solver in successful_solvers], position = :ct, color = :white,
            framevisible=false, label = "Solvers", orientation = :horizontal,
            tellwidth = false, tellheight = true, nbanks = 3, labelsize = 20)

        fig
    end

    return fig
end


prob = generate_prob(Val(2), Val(true))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 2")


prob = generate_prob(Val(4), Val(true))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 4")


prob = generate_prob(Val(8), Val(true))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 8")


prob = generate_prob(Val(10), Val(true))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 10")


prob = generate_prob(Val(2), Val(false))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 2")


prob = generate_prob(Val(8), Val(false))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 8")


prob = generate_prob(Val(32), Val(false))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 32")


prob = generate_prob(Val(128), Val(false))
wpset, successful_solvers = generate_wpset(prob, solvers);
fig = plot_wpset(wpset, successful_solvers, "N = 128")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

