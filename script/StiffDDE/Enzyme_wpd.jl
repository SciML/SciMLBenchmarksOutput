
using DelayDiffEq, DiffEqDevTools, Plots
gr()


function f_enzyme!(du, u, h, p, t)
    τ, K1 = p
    y4_delayed = h(p, t - τ; idxs = 4)
    U = 1.0 / (1.0 + K1 * y4_delayed^3)
    du[1] = 10.5 - u[1] * U
    du[2] = u[1] * U - u[2]
    du[3] = u[2] - u[3]
    du[4] = u[3] - 0.5 * u[4]
    nothing
end

function h_enzyme(p, t; idxs::Union{Nothing, Int} = nothing)
    if idxs === nothing
        [60.0, 10.0, 10.0, 20.0]
    elseif idxs == 1
        60.0
    elseif idxs == 2
        10.0
    elseif idxs == 3
        10.0
    elseif idxs == 4
        20.0
    else
        error("index must be between 1 and 4")
    end
end

τ = 4.0
K1 = 0.0005
prob = DDEProblem(f_enzyme!, h_enzyme, (0.0, 160.0), (τ, K1); constant_lags = [τ])


sol = solve(prob, MethodOfSteps(Rodas5P()); reltol = 1e-14, abstol = 1e-14)
test_sol = TestSolution(sol)
plot(sol; title = "Enzyme Kinetics Solution")


abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas4())),
    Dict(:alg => MethodOfSteps(Rodas5())),
    Dict(:alg => MethodOfSteps(Rodas5P()))]
names = ["Rosenbrock23", "Rodas4", "Rodas5", "Rodas5P"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Enzyme: Rosenbrock Methods (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Enzyme: Rosenbrock Methods (L2 error)")


setups = [Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(SDIRK2())),
    Dict(:alg => MethodOfSteps(KenCarp4())),
    Dict(:alg => MethodOfSteps(Kvaerno4())),
    Dict(:alg => MethodOfSteps(Kvaerno5()))]
names = ["TRBDF2", "SDIRK2", "KenCarp4", "Kvaerno4", "Kvaerno5"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Enzyme: SDIRK Methods (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Enzyme: SDIRK Methods (L2 error)")


setups = [Dict(:alg => MethodOfSteps(Rodas5P())),
    Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(KenCarp4())),
    Dict(:alg => MethodOfSteps(Tsit5())),
    Dict(:alg => MethodOfSteps(DP5())),
    Dict(:alg => MethodOfSteps(OwrenZen5()))]
names = ["Rodas5P", "TRBDF2", "KenCarp4", "Tsit5", "DP5", "OwrenZen5"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e6), error_estimate = :final)
plot(wp; title = "Enzyme: Stiff vs Non-Stiff (final error)")


abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas4())),
    Dict(:alg => MethodOfSteps(Rodas5())),
    Dict(:alg => MethodOfSteps(Rodas5P())),
    Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(KenCarp4()))]
names = ["Rosenbrock23", "Rodas4", "Rodas5", "Rodas5P", "TRBDF2", "KenCarp4"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Enzyme: Low Tolerances (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Enzyme: Low Tolerances (L2 error)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

