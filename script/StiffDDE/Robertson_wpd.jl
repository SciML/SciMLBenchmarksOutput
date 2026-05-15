
using DelayDiffEq, DiffEqDevTools, DDEProblemLibrary, Plots
import DDEProblemLibrary: prob_dde_RADAR5_robertson
gr()


prob = remake(prob_dde_RADAR5_robertson; tspan = (0.0, 1.0))


sol = solve(prob, MethodOfSteps(Rodas5P());
    reltol = 1e-14, abstol = [1e-14, 1e-20, 1e-14], dt = 1e-6)
test_sol = TestSolution(sol)
plot(sol; title = "Robertson DDE Solution (t ∈ [0, 1])")


abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas4())),
    Dict(:alg => MethodOfSteps(Rodas5())),
    Dict(:alg => MethodOfSteps(Rodas5P()))]
names = ["Rosenbrock23", "Rodas4", "Rodas5", "Rodas5P"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final,
    dt = 1e-6)
plot(wp; title = "Robertson: Rosenbrock Methods (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2,
    dt = 1e-6)
plot(wp; title = "Robertson: Rosenbrock Methods (L2 error)")


setups = [Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(SDIRK2())),
    Dict(:alg => MethodOfSteps(KenCarp4()))]
names = ["TRBDF2", "SDIRK2", "KenCarp4"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final,
    dt = 1e-6)
plot(wp; title = "Robertson: SDIRK Methods (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2,
    dt = 1e-6)
plot(wp; title = "Robertson: SDIRK Methods (L2 error)")


setups = [Dict(:alg => MethodOfSteps(Rodas5P())),
    Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(KenCarp4())),
    Dict(:alg => MethodOfSteps(Tsit5())),
    Dict(:alg => MethodOfSteps(DP5()))]
names = ["Rodas5P", "TRBDF2", "KenCarp4", "Tsit5", "DP5"]
wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e6), error_estimate = :final,
    dt = 1e-6)
plot(wp; title = "Robertson: Stiff vs Non-Stiff (final error)")


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
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final,
    dt = 1e-6)
plot(wp; title = "Robertson: Low Tolerances (final error)")


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2,
    dt = 1e-6)
plot(wp; title = "Robertson: Low Tolerances (L2 error)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

