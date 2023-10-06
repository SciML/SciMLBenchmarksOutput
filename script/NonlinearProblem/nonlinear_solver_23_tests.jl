
using NonlinearSolve, NonlinearSolveMINPACK, SciMLNLSolve, SimpleNonlinearSolve, StaticArrays, Sundials
using BenchmarkTools, LinearAlgebra, DiffEqDevTools, NonlinearProblemLibrary, Plots
RUS = RadiusUpdateSchemes;


solvers_all = [ 
    (type = :NR,      name = "Newton Raphson (No line search)",                    solver = Dict(:alg=>NewtonRaphson()),                                       color = :salmon1,         markershape = :star4),
    (type = :NR,      name = "Newton Raphson (Hager & Zhang line search)",         solver = Dict(:alg=>NewtonRaphson(linesearch=HagerZhang())),                color = :tomato1,         markershape = :star5),
    (type = :NR,      name = "Newton Raphson (More & Thuente line search)",        solver = Dict(:alg=>NewtonRaphson(linesearch=MoreThuente())),               color = :red3,            markershape = :star6),
    (type = :NR,      name = "Newton Raphson (Nocedal & Wright line search)",      solver = Dict(:alg=>NewtonRaphson(linesearch=BackTracking())),              color = :firebrick,       markershape = :star7),
    (type = :TR,      name = "Newton Trust Region",                                solver = Dict(:alg=>TrustRegion()),                                         color = :darkslategray1,  markershape = :circle),
    (type = :TR,      name = "Newton Trust Region (NLsolve radius update)",        solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.NLsolve)),       color = :deepskyblue1,    markershape = :utriangle),
    (type = :TR,      name = "Newton Trust Region (Nocedal Wright radius update)", solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.NocedalWright)), color = :cadetblue,       markershape = :rect),
    (type = :TR,      name = "Newton Trust Region (Hei radius update)",            solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Hei)),           color = :lightslateblue,  markershape = :pentagon),
    (type = :TR,      name = "Newton Trust Region (Yuan radius update)",           solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Yuan)),          color = :royalblue2,      markershape = :hexagon),
    (type = :TR,      name = "Newton Trust Region (Bastin radius update)",         solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Bastin)),        color = :blue1,           markershape = :heptagon),
    (type = :TR,      name = "Newton Trust Region (Fan radius update)",            solver = Dict(:alg=>TrustRegion(radius_update_scheme = RUS.Fan)),           color = :navy,            markershape = :octagon),
    (type = :LM,      name = "Levenberg-Marquardt (default α_geodesic)",           solver = Dict(:alg=>LevenbergMarquardt()),                                  color = :orchid4,         markershape = :rect),
    (type = :LM,      name = "Levenberg-Marquardt (α_geodesic=0.5)",               solver = Dict(:alg=>LevenbergMarquardt(α_geodesic=0.5)),                    color = :purple4,         markershape = :diamond),
    (type = :general, name = "Modified Powell (CMINPACK)",                         solver = Dict(:alg=>CMINPACK(method=:hybr)),                                color = :lightgoldenrod2, markershape = :+),
    (type = :general, name = "Levenberg-Marquardt (CMINPACK)",                     solver = Dict(:alg=>CMINPACK(method=:lm)),                                  color = :gold1,           markershape = :x),
    (type = :general, name = "Newton Raphson (NLSolveJL)",                         solver = Dict(:alg=>NLSolveJL(method=:newton)),                             color = :olivedrab1,      markershape = :dtriangle),
    (type = :general, name = "Newton Trust Region (NLSolveJL)",                    solver = Dict(:alg=>NLSolveJL()),                                           color = :green2,          markershape = :rtriangle),
    (type = :general, name = "Anderson acceleration (NLSolveJL)",                  solver = Dict(:alg=>NLSolveJL(method=:anderson)),                           color = :forestgreen,     markershape = :ltriangle),
    (type = :general, name = "Newton-Krylov (Sundials)",                           solver = Dict(:alg=>KINSOL()),                                              color = :darkorange,      markershape = :diamond)
]
solver_tracker = [];


abstols = 1.0 ./ 10.0 .^ (4:12)
reltols = 1.0 ./ 10.0 .^ (4:12);


mm = Plots.Measures.mm
default(framestyle=:box,legend=:topleft,gridwidth=2, guidefontsize=25, legendfontsize=16, lw=3, la=0.8, ms=10, ma=0.8, left_margin=20mm, bottom_margin=15mm, right_margin=0mm)


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
    wp_NR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_NR, :solver); names=getfield.(solvers_NR, :name), numruns=100, error_estimate=:l2, maxiters=1000000)
    wp_TR = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_TR, :solver); names=getfield.(solvers_TR, :name), numruns=100, error_estimate=:l2, maxiters=1000000)
    wp_LM = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_LM, :solver); names=getfield.(solvers_LM, :name), numruns=100, error_estimate=:l2, maxiters=1000000)

    # Handles the general case
    solvers_general = filter(s -> s.type==:general, successful_solvers)
    add_solver!(solvers_general, selected_TR, solvers_TR, wp_TR)
    add_solver!(solvers_general, selected_LM, solvers_LM, wp_LM)
    add_solver!(solvers_general, selected_NR, solvers_NR, wp_NR)
    wp_general = WorkPrecisionSet(prob.prob, abstols, reltols, getfield.(solvers_general, :solver); names=getfield.(solvers_general, :name), numruns=100, error_estimate=:l2, maxiters=1000000)
    
    xlimit, ylimit, xticks, yticks = get_limits_and_ticks(wp_general, wp_NR, wp_TR, wp_LM)
    wp_plot_general = plot_wp(wp_general, solvers_general)
    wp_plot_NR = plot_wp(wp_NR, solvers_NR; yguide="", xlimit=xlimit, ylimit=ylimit, xticks=xticks, yticks=yticks)
    wp_plot_TR = plot_wp(wp_TR, solvers_TR; yguide="", xlimit=xlimit, ylimit=ylimit, xticks=xticks, yticks=yticks)
    wp_plot_LM = plot_wp(wp_LM, solvers_LM; yguide="", xlimit=xlimit, ylimit=ylimit, xticks=xticks, yticks=yticks)
    plot(wp_plot_general, wp_plot_NR, wp_plot_TR, wp_plot_LM, layout=(1,4), size=(3000,800), xlimit=xlimit, ylimit=ylimit, xticks=xticks, yticks=yticks)
end

# Checks if a solver can sucessfully solve a given problem.
function check_solver(prob, solver)
    try
        sol = solve(prob.prob, solver.solver[:alg]; abstol=1e-8, reltol=1e-8, maxiters=1000000)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("\n[Warn] Solver $(solver.name) returned retcode $(sol.retcode) with an residual norm = $(norm(sol.resid)).\n"; color=:red)
            return false
        elseif norm(sol.resid) > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm = $(norm(sol.resid))).\n"; color=:red)
            return false
        end
        WorkPrecisionSet(prob.prob, [1e-4, 1e-12], [1e-4, 1e-12], [solver.solver]; names=[solver.name], numruns=100, error_estimate=:l2, maxiters=1000000)
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


# Plots a work-precision diagram.
function plot_wp(wp, selected_solvers; kwargs...)
    isempty(wp.wps) && (return plot(xaxis=:log10, yaxis=:log10 ;kwargs...))
    color = reshape(getfield.(selected_solvers, :color),1,length(selected_solvers))
    markershape = reshape(getfield.(selected_solvers, :markershape),1,length(selected_solvers))
    plot(wp; color=color, markershape=markershape, legend=:outertop, kwargs...)
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


using PrettyTables
solver_sucesses = [(solver in prob[2]) ? "O" : "X" for prob in solver_tracker, solver in solvers_all];


using PrettyTables
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, solver_sucesses; backend = Val(:html), header = getfield.(solvers_all, :name), row_names = first.(solver_tracker), alignment=:c)
println(io, "```")
Text(String(take!(io)))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

