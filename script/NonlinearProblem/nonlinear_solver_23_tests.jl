
using NonlinearSolve, NonlinearSolveMINPACK, SciMLNLSolve, SimpleNonlinearSolve, StaticArrays, Sundials
using BenchmarkTools, DiffEqDevTools, NonlinearProblemLibrary, Plots


solvers = [ Dict(:alg=>NewtonRaphson()),
            Dict(:alg=>NewtonRaphson(linesearch=HagerZhang())),
            Dict(:alg=>NewtonRaphson(linesearch=MoreThuente())),
            Dict(:alg=>NewtonRaphson(linesearch=BackTracking())),
            Dict(:alg=>NewtonRaphson(linesearch=Static())),
            Dict(:alg=>TrustRegion()),
            Dict(:alg=>LevenbergMarquardt()),
            Dict(:alg=>CMINPACK(method=:hybr)),
            Dict(:alg=>CMINPACK(method=:lm)),
            Dict(:alg=>NLSolveJL(method=:newton)),
            Dict(:alg=>NLSolveJL()),
            Dict(:alg=>NLSolveJL(method=:anderson)),
            Dict(:alg=>KINSOL())]
solvernames =  ["Newton Raphson (No line search)";
                "Newton Raphson (Hager & Zhang line search)";
                "Newton Raphson (More & Thuente line search)";
                "Newton Raphson (Nocedal & Wright line search)";
                "Newton Raphson (Static line search)"; 
                "Newton Trust Region"; 
                "Levenberg-Marquardt"; 
                "Modified Powell (CMINPACK)"; 
                "Levenberg-Marquardt (CMINPACK)"; 
                "Newton Raphson (NLSolveJL)"; 
                "Newton Trust Region (NLSolveJL)"; 
                "Anderson acceleration (NLSolveJL)"; 
                "Newton-Krylov (Sundials)"]
nothing # hide


abstols = 1.0 ./ 10.0 .^ (4:12)
reltols = 1.0 ./ 10.0 .^ (4:12)
nothing # hide


mm = Plots.Measures.mm
default(framestyle=:box,legend=:topleft,gridwidth=2, guidefontsize=12, legendfontsize=9, lw=2, ms=6, left_margin=6mm, bottom_margin=6mm, right_margin=2mm)
markershapes = [:utriangle, :rect, :pentagon, :heptagon, :octagon]
colors = [:violet, :magenta1, :orchid4, :darkorchid2, :blueviolet]
markershapes = [:hexagon, :hexagon, :hexagon, :hexagon, :diamond, :rect, :utriangle, :dtriangle, :star4, :star5, :star7, :circle]
colors = [:lightslateblue, :lightslateblue, :lightslateblue, :lightslateblue, :dodgerblue3, :blue3, :coral2, :red3, :olivedrab1, :green2, :forestgreen, :aqua]
nothing # hide


# Selects the solvers to be benchmakred on a given problem.
function select_solvers(prob; selected_NR=1, solvers=solvers, solvernames=solvernames)
    selected_solvers_all = filter(s_idx -> check_solver(prob, solvers[s_idx], solvernames[s_idx]), 1:length(solvers))
    selected_NR_solvers = filter(ss -> ss<=5, selected_solvers_all)
    selected_solvers = filter(ss -> ss>5, selected_solvers_all)
    (selected_NR in selected_NR_solvers) && (selected_solvers = [selected_NR; selected_solvers])
    return selected_NR_solvers, selected_solvers
end
# Checks if a solver can sucessfully solve a given problem.
function check_solver(prob, solver, solvername)
    try
        true_sol = solve(prob.prob, solver[:alg]; abstol=1e-18, reltol=1e-18, maxiters=10000000)
        if !SciMLBase.successful_retcode(true_sol.retcode)
            Base.printstyled("[Warn] Solver $solvername returned retcode $(true_sol.retcode).\n"; color=:red)
            return false
        end
        WorkPrecisionSet(prob.prob, [1e-4, 1e-12], [1e-4, 1e-12], [solver]; names=[solvername], numruns=20, appxsol=true_sol, error_estimate=:l2, maxiters=10000000)
    catch e
        Base.printstyled("[Warn] Solver $solvername threw an error: $e.\n"; color=:red)    
        return false
    end
    return true
end
nothing # hide


# Finds good x and y limits.
function xy_limits(wp)
    times = vcat(map(wp -> wp.times, wp.wps)...)
    errors = vcat(map(wp -> wp.errors, wp.wps)...)
    xlimit = 10 .^ (floor(log10(minimum(errors))), ceil(log10(maximum(errors))))
    ylimit = 10 .^ (floor(log10(minimum(times))), ceil(log10(maximum(times))))
    return xlimit, ylimit
end

# Find good x and y ticks.
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
    (limit[1]==-Inf) && return 10.0 .^[limit[1], limit[2]]
    sequences = arithmetic_sequences(limit...)
    selected_seq = findlast(length.(sequences) .< 5)
    if length(sequences[selected_seq]) < 4
        step = (limit[2] - limit[1]) / 6.0
        ticks = [round(Int, limit[1] + i*step) for i in 1:5]
        return 10 .^[limit[1];ticks;limit[2]]
    end
    return 10 .^sequences[selected_seq]
end

# Plots a work-precision diagram.
function plot_wp(wp, selected_solvers; colors=permutedims(getindex(colors,selected_solvers)[:,:]), markershapes=permutedims(getindex(markershapes,selected_solvers)[:,:]), kwargs...)
    xlimit, ylimit = xy_limits(wp)
    xticks = get_ticks(log10.(xlimit))
    yticks = get_ticks(log10.(ylimit))
    plot(wp; xlimit=xlimit, ylimit=ylimit, xticks=xticks, yticks=yticks, color=colors, markershape=markershapes, kwargs...)
end
# Plots work precision diagrams for NonlinearSolve's NewtonRaphson solvers, and for a combiantion of all solvers.
function plot_wps(wp_NR, wp, selected_NR_solvers, selected_solvers; kwargs...)
    wp_dia_NR = plot_wp(wp_NR, selected_NR_solvers;  colors=permutedims(getindex(colors_NR,selected_NR_solvers)[:,:]), markershapes=permutedims(getindex(markershapes_NR,selected_NR_solvers)[:,:]), kwargs...)
    wp_dia = plot_wp(wp, selected_solvers; yguide="", kwargs...)
    plot(wp_dia_NR, wp_dia, size=(1100,400))
end
nothing # hide


prob_1 = nlprob_23_testcases["Generalized Rosenbrock function"]
selected_NR_solvers_1, selected_solvers_1 = select_solvers(prob_1)
wp_1_NR = WorkPrecisionSet(prob_1.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_1); names=getindex(solvernames,selected_NR_solvers_1), numruns=100, appxsol=prob_1.true_sol, error_estimate=:l2, maxiters=10000000)
wp_1 = WorkPrecisionSet(prob_1.prob, abstols, reltols, getindex(solvers,selected_solvers_1); names=getindex(solvernames,selected_solvers_1), numruns=100, appxsol=prob_1.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_1_NR, wp_1, selected_NR_solvers_1, selected_solvers_1)


prob_2 = nlprob_23_testcases["Powell singular function"]
selected_NR_solvers_2, selected_solvers_2 = select_solvers(prob_2)
wp_2_NR = WorkPrecisionSet(prob_2.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_2); names=getindex(solvernames,selected_NR_solvers_2), numruns=100, appxsol=prob_2.true_sol, error_estimate=:l2, maxiters=10000000)
wp_2 = WorkPrecisionSet(prob_2.prob, abstols, reltols, getindex(solvers,selected_solvers_2); names=getindex(solvernames,selected_solvers_2), numruns=100, appxsol=prob_2.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_2_NR, wp_2, selected_NR_solvers_2, selected_solvers_2)


prob_3 = nlprob_23_testcases["Powell badly scaled function"]
selected_NR_solvers_3, selected_solvers_3 = select_solvers(prob_3)
wp_3_NR = WorkPrecisionSet(prob_3.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_3); names=getindex(solvernames,selected_NR_solvers_3), numruns=100, appxsol=prob_3.true_sol, error_estimate=:l2, maxiters=10000000)
wp_3 = WorkPrecisionSet(prob_3.prob, abstols, reltols, getindex(solvers,selected_solvers_3); names=getindex(solvernames,selected_solvers_3), numruns=100, appxsol=prob_3.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_3_NR, wp_3, selected_NR_solvers_3, selected_solvers_3)


prob_4 = nlprob_23_testcases["Wood function"]
selected_NR_solvers_4, selected_solvers_4 = select_solvers(prob_4)
wp_4_NR = WorkPrecisionSet(prob_4.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_4); names=getindex(solvernames,selected_NR_solvers_4), numruns=100, appxsol=prob_4.true_sol, error_estimate=:l2, maxiters=10000000)
wp_4 = WorkPrecisionSet(prob_4.prob, abstols, reltols, getindex(solvers,selected_solvers_4); names=getindex(solvernames,selected_solvers_4), numruns=100, appxsol=prob_4.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_4_NR, wp_4, selected_NR_solvers_4, selected_solvers_4)


prob_5 = nlprob_23_testcases["Helical valley function"]
selected_NR_solvers_5, selected_solvers_5 = select_solvers(prob_5)
wp_5_NR = WorkPrecisionSet(prob_5.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_5); names=getindex(solvernames,selected_NR_solvers_5), numruns=100, appxsol=prob_5.true_sol, error_estimate=:l2, maxiters=10000000)
wp_5 = WorkPrecisionSet(prob_5.prob, abstols, reltols, getindex(solvers,selected_solvers_5); names=getindex(solvernames,selected_solvers_5), numruns=100, appxsol=prob_5.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_5_NR, wp_5, selected_NR_solvers_5, selected_solvers_5)


prob_6 = nlprob_23_testcases["Watson function"]
selected_NR_solvers_6, selected_solvers_6 = select_solvers(prob_6)
true_sol_6 = solve(prob_6.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_6_NR = WorkPrecisionSet(prob_6.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_6); names=getindex(solvernames,selected_NR_solvers_6), numruns=100, appxsol=prob_6.true_sol, error_estimate=:l2, maxiters=10000000)
wp_6 = WorkPrecisionSet(prob_6.prob, abstols, reltols, getindex(solvers,selected_solvers_6); names=getindex(solvernames,selected_solvers_6), numruns=100, appxsol=true_sol_6, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_6_NR, wp_6, selected_NR_solvers_6, selected_solvers_6)


prob_7 = nlprob_23_testcases["Chebyquad function"]
selected_NR_solvers_7, selected_solvers_7 = select_solvers(prob_7)
wp_7_NR = WorkPrecisionSet(prob_7.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_7); names=getindex(solvernames,selected_NR_solvers_7), numruns=100, appxsol=prob_7.true_sol, error_estimate=:l2, maxiters=10000000)
wp_7 = WorkPrecisionSet(prob_7.prob, abstols, reltols, getindex(solvers,selected_solvers_7); names=getindex(solvernames,selected_solvers_7), numruns=100, appxsol=prob_7.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_7_NR, wp_7, selected_NR_solvers_7, selected_solvers_7)


prob_8 = nlprob_23_testcases["Brown almost linear function"]
selected_NR_solvers_8, selected_solvers_8 = select_solvers(prob_8)
wp_8_NR = WorkPrecisionSet(prob_.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_8); names=getindex(solvernames,selected_NR_solvers_8), numruns=100, appxsol=prob_8.true_sol, error_estimate=:l2, maxiters=10000000)
wp_8 = WorkPrecisionSet(prob_8.prob, abstols, reltols, getindex(solvers,selected_solvers_8); names=getindex(solvernames,selected_solvers_8), numruns=100, appxsol=prob_8.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_8_NR, wp_8, selected_NR_solvers_8, selected_solvers_8)


prob_9 = nlprob_23_testcases["Discrete boundary value function"]
selected_NR_solvers_9, selected_solvers_9 = select_solvers(prob_9)
true_sol_9 = solve(prob_9.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_9_NR = WorkPrecisionSet(prob_9.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_9); names=getindex(solvernames,selected_NR_solvers_9), numruns=100, appxsol=prob_9.true_sol, error_estimate=:l2, maxiters=10000000)
wp_9 = WorkPrecisionSet(prob_9.prob, abstols, reltols, getindex(solvers,selected_solvers_9); names=getindex(solvernames,selected_solvers_9), numruns=100, appxsol=true_sol_9, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_9_NR, wp_9, selected_NR_solvers_9, selected_solvers_9)


prob_10 = nlprob_23_testcases["Discrete integral equation function"]
selected_NR_solvers_10, selected_solvers_10 = select_solvers(prob_10)
true_sol_10 = solve(prob_10.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_10_NR = WorkPrecisionSet(prob_10.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_10); names=getindex(solvernames,selected_NR_solvers_10), numruns=100, appxsol=prob_10.true_sol, error_estimate=:l2, maxiters=10000000)
wp_10 = WorkPrecisionSet(prob_10.prob, abstols, reltols, getindex(solvers,selected_solvers_10); names=getindex(solvernames,selected_solvers_10), numruns=100, appxsol=true_sol_10, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_10_NR, wp_10, selected_NR_solvers_10, selected_solvers_10)


prob_11 = nlprob_23_testcases["Trigonometric function"]
selected_NR_solvers_11, selected_solvers_11 = select_solvers(prob_11)
true_sol_11 = solve(prob_11.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_11_NR = WorkPrecisionSet(prob_11.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_11); names=getindex(solvernames,selected_NR_solvers_11), numruns=100, appxsol=prob_11.true_sol, error_estimate=:l2, maxiters=10000000)
wp_11 = WorkPrecisionSet(prob_11.prob, abstols, reltols, getindex(solvers,selected_solvers_11); names=getindex(solvernames,selected_solvers_11), numruns=100, appxsol=true_sol_11, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_11_NR, wp_11, selected_NR_solvers_11, selected_solvers_11)


prob_12 = nlprob_23_testcases["Variably dimensioned function"]
selected_NR_solvers_12, selected_solvers_12 = select_solvers(prob_12)
wp_12_NR = WorkPrecisionSet(prob_12.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_12); names=getindex(solvernames,selected_NR_solvers_12), numruns=100, appxsol=prob_12.true_sol, error_estimate=:l2, maxiters=10000000)
wp_12 = WorkPrecisionSet(prob_12.prob, abstols, reltols, getindex(solvers,selected_solvers_12); names=getindex(solvernames,selected_solvers_12), numruns=100, appxsol=prob_12.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_12_NR, wp_12, selected_NR_solvers_12, selected_solvers_12)


prob_13 = nlprob_23_testcases["Broyden tridiagonal function"]
selected_NR_solvers_13, selected_solvers_13 = select_solvers(prob_13)
true_sol_13 = solve(prob_13.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_13_NR = WorkPrecisionSet(prob_13.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_13); names=getindex(solvernames,selected_NR_solvers_13), numruns=100, appxsol=prob_13.true_sol, error_estimate=:l2, maxiters=10000000)
wp_13 = WorkPrecisionSet(prob_13.prob, abstols, reltols, getindex(solvers,selected_solvers_13); names=getindex(solvernames,selected_solvers_13), numruns=100, appxsol=true_sol_13, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_13_NR, wp_13, selected_NR_solvers_13, selected_solvers_13)


prob_14 = nlprob_23_testcases["Broyden banded function"]
selected_NR_solvers_14, selected_solvers_14 = select_solvers(prob_14)
true_sol_14 = solve(prob_14.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_14_NR = WorkPrecisionSet(prob_14.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_14); names=getindex(solvernames,selected_NR_solvers_14), numruns=100, appxsol=prob_14.true_sol, error_estimate=:l2, maxiters=10000000)
wp_14 = WorkPrecisionSet(prob_14.prob, abstols, reltols, getindex(solvers,selected_solvers_14); names=getindex(solvernames,selected_solvers_14), numruns=100, appxsol=true_sol_14, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_14_NR, wp_14, selected_NR_solvers_14, selected_solvers_14)


prob_15 = nlprob_23_testcases["Hammarling 2 by 2 matrix square root problem"]
selected_NR_solvers_15, selected_solvers_15 = select_solvers(prob_15)
wp_15_NR = WorkPrecisionSet(prob_15.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_15); names=getindex(solvernames,selected_NR_solvers_15), numruns=100, appxsol=prob_15.true_sol, error_estimate=:l2, maxiters=10000000)
wp_15 = WorkPrecisionSet(prob_15.prob, abstols, reltols, getindex(solvers,selected_solvers_15); names=getindex(solvernames,selected_solvers_15), numruns=100, appxsol=prob_15.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_15_NR, wp_15, selected_NR_solvers_15, selected_solvers_15)


prob_16 = nlprob_23_testcases["Hammarling 3 by 3 matrix square root problem"]
selected_NR_solvers_16, selected_solvers_16 = select_solvers(prob_16)
wp_16_NR = WorkPrecisionSet(prob_16.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_16); names=getindex(solvernames,selected_NR_solvers_16), numruns=100, appxsol=prob_16.true_sol, error_estimate=:l2, maxiters=10000000)
wp_16 = WorkPrecisionSet(prob_16.prob, abstols, reltols, getindex(solvers,selected_solvers_16); names=getindex(solvernames,selected_solvers_16), numruns=100, appxsol=prob_16.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_16_NR, wp_16, selected_NR_solvers_16, selected_solvers_16)


prob_17 = nlprob_23_testcases["Dennis and Schnabel 2 by 2 example"]
selected_NR_solvers_17, selected_solvers_17 = select_solvers(prob_17)
wp_17_NR = WorkPrecisionSet(prob_17.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_17); names=getindex(solvernames,selected_NR_solvers_17), numruns=100, appxsol=prob_17.true_sol, error_estimate=:l2, maxiters=10000000)
wp_17 = WorkPrecisionSet(prob_17.prob, abstols, reltols, getindex(solvers,selected_solvers_17); names=getindex(solvernames,selected_solvers_17), numruns=100, appxsol=prob_17.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_17_NR, wp_17, selected_NR_solvers_17, selected_solvers_17)


prob_18 = nlprob_23_testcases["Sample problem 18"]
selected_NR_solvers_18, selected_solvers_18 = select_solvers(prob_18)
wp_18_NR = WorkPrecisionSet(prob_18.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_18); names=getindex(solvernames,selected_NR_solvers_18), numruns=100, appxsol=prob_18.true_sol, error_estimate=:l2, maxiters=10000000)
wp_18 = WorkPrecisionSet(prob_18.prob, abstols, reltols, getindex(solvers,selected_solvers_18); names=getindex(solvernames,selected_solvers_18), numruns=100, appxsol=prob_18.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_18_NR, wp_18, selected_NR_solvers_18, selected_solvers_18)


prob_19 = nlprob_23_testcases["Sample problem 19"]
selected_NR_solvers_19, selected_solvers_19 = select_solvers(prob_19)
wp_19_NR = WorkPrecisionSet(prob_19.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_19); names=getindex(solvernames,selected_NR_solvers_19), numruns=100, appxsol=prob_19.true_sol, error_estimate=:l2, maxiters=10000000)
wp_19 = WorkPrecisionSet(prob_19.prob, abstols, reltols, getindex(solvers,selected_solvers_19); names=getindex(solvernames,selected_solvers_19), numruns=100, appxsol=prob_19.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_19_NR, wp_19, selected_NR_solvers_19, selected_solvers_19)


prob_20 = nlprob_23_testcases["Scalar problem f(x) = x(x - 5)^2"]
selected_NR_solvers_20, selected_solvers_20 = select_solvers(prob_20)
wp_20_NR = WorkPrecisionSet(prob_20.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_20); names=getindex(solvernames,selected_NR_solvers_20), numruns=100, appxsol=prob_20.true_sol, error_estimate=:l2, maxiters=10000000)
wp_20 = WorkPrecisionSet(prob_20.prob, abstols, reltols, getindex(solvers,selected_solvers_20); names=getindex(solvernames,selected_solvers_20), numruns=100, appxsol=prob_20.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_20_NR, wp_20, selected_NR_solvers_20, selected_solvers_20)


prob_21 = nlprob_23_testcases["Freudenstein-Roth function"]
selected_NR_solvers_21, selected_solvers_21 = select_solvers(prob_21)
wp_21_NR = WorkPrecisionSet(prob_21.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_21); names=getindex(solvernames,selected_NR_solvers_21), numruns=100, appxsol=prob_21.true_sol, error_estimate=:l2, maxiters=10000000)
wp_21 = WorkPrecisionSet(prob_21.prob, abstols, reltols, getindex(solvers,selected_solvers_21); names=getindex(solvernames,selected_solvers_21), numruns=100, appxsol=prob_21.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_21_NR, wp_21, selected_NR_solvers_21, selected_solvers_21)


prob_22 = nlprob_23_testcases["Boggs function"]
selected_NR_solvers_22, selected_solvers_22 = select_solvers(prob_22)
wp_22_NR = WorkPrecisionSet(prob_22.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_22); names=getindex(solvernames,selected_NR_solvers_22), numruns=100, appxsol=prob_22.true_sol, error_estimate=:l2, maxiters=10000000)
wp_22 = WorkPrecisionSet(prob_22.prob, abstols, reltols, getindex(solvers,selected_solvers_22); names=getindex(solvernames,selected_solvers_22), numruns=100, appxsol=prob_22.true_sol, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_22_NR, wp_22, selected_NR_solvers_22, selected_solvers_22)


prob_23 = nlprob_23_testcases["Chandrasekhar function"]
selected_NR_solvers_23, selected_solvers_23 = select_solvers(prob_23)
true_sol_23 = solve(prob_23.prob, CMINPACK(method=:lm); abstol=1e-18, reltol=1e-18)
wp_23_NR = WorkPrecisionSet(prob_23.prob, abstols, reltols, getindex(solvers,selected_NR_solvers_23); names=getindex(solvernames,selected_NR_solvers_23), numruns=100, appxsol=prob_23.true_sol, error_estimate=:l2, maxiters=10000000)
wp_23 = WorkPrecisionSet(prob_23.prob, abstols, reltols, getindex(solvers,selected_solvers_23); names=getindex(solvernames,selected_solvers_23), numruns=100, appxsol=true_sol_23, error_estimate=:l2, maxiters=10000000)
plot_wps(wp_23_NR, wp_23, selected_NR_solvers_23, selected_solvers_23)


function assembly_row(solvers1,solvers2; n=length(solvers))
    all_solvers = union(solvers1,solvers2)
    return [(i in all_solvers) ? "O" : "X" for j in 1:1, i in 1:n]
end
nothing # hide


sucessful_solvers = [
assembly_row(selected_NR_solvers_1, selected_solvers_1; n=length(solvers));
assembly_row(selected_NR_solvers_2, selected_solvers_2; n=length(solvers));
assembly_row(selected_NR_solvers_3, selected_solvers_3; n=length(solvers));
assembly_row(selected_NR_solvers_4, selected_solvers_4; n=length(solvers));
assembly_row(selected_NR_solvers_5, selected_solvers_5; n=length(solvers));
assembly_row(selected_NR_solvers_6, selected_solvers_6; n=length(solvers));
assembly_row(selected_NR_solvers_7, selected_solvers_7; n=length(solvers));
assembly_row(selected_NR_solvers_8, selected_solvers_8; n=length(solvers));
assembly_row(selected_NR_solvers_9, selected_solvers_9; n=length(solvers));
assembly_row(selected_NR_solvers_10, selected_solvers_10; n=length(solvers));
assembly_row(selected_NR_solvers_12, selected_solvers_12; n=length(solvers));
assembly_row(selected_NR_solvers_13, selected_solvers_13; n=length(solvers));
assembly_row(selected_NR_solvers_14, selected_solvers_14; n=length(solvers));
assembly_row(selected_NR_solvers_15, selected_solvers_15; n=length(solvers));
assembly_row(selected_NR_solvers_16, selected_solvers_16; n=length(solvers));
assembly_row(selected_NR_solvers_17, selected_solvers_17; n=length(solvers));
assembly_row(selected_NR_solvers_18, selected_solvers_18; n=length(solvers));
assembly_row(selected_NR_solvers_19, selected_solvers_19; n=length(solvers));
assembly_row(selected_NR_solvers_20, selected_solvers_20; n=length(solvers));
assembly_row(selected_NR_solvers_21, selected_solvers_21; n=length(solvers));
assembly_row(selected_NR_solvers_22, selected_solvers_22; n=length(solvers));
assembly_row(selected_NR_solvers_23, selected_solvers_23; n=length(solvers))
]
nothing #


using PrettyTables
pretty_table(sucessful_solvers; header=solvernames, alignment=:c)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

