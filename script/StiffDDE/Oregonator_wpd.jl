
using DelayDiffEq, DiffEqDevTools, DDEProblemLibrary, Plots
import DDEProblemLibrary: prob_dde_RADAR5_oregonator
gr()


sol = solve(prob_dde_RADAR5_oregonator, MethodOfSteps(Rodas5P());
    reltol = 1e-14, abstol = 1e-14)
test_sol = TestSolution(sol)
plot(sol; title = "RADAR5 Oregonator Solution")


abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas4())),
    Dict(:alg => MethodOfSteps(Rodas5())),
    Dict(:alg => MethodOfSteps(Rodas5P()))]
names = ["Rosenbrock23", "Rodas4", "Rodas5", "Rodas5P"]
wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Oregonator: Rosenbrock Methods (final error)")


wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Oregonator: Rosenbrock Methods (L2 error)")


wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L∞)
plot(wp; title = "Oregonator: Rosenbrock Methods (L∞ error)")


setups = [Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(SDIRK2())),
    Dict(:alg => MethodOfSteps(KenCarp4())),
    Dict(:alg => MethodOfSteps(Kvaerno4())),
    Dict(:alg => MethodOfSteps(Kvaerno5()))]
names = ["TRBDF2", "SDIRK2", "KenCarp4", "Kvaerno4", "Kvaerno5"]
wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Oregonator: SDIRK Methods (final error)")


wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Oregonator: SDIRK Methods (L2 error)")


wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L∞)
plot(wp; title = "Oregonator: SDIRK Methods (L∞ error)")


setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas5P())),
    Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(KenCarp4())),
    Dict(:alg => MethodOfSteps(Tsit5())),
    Dict(:alg => MethodOfSteps(BS3())),
    Dict(:alg => MethodOfSteps(DP5()))]
names = ["Rosenbrock23", "Rodas5P", "TRBDF2", "KenCarp4", "Tsit5", "BS3", "DP5"]
wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e6), error_estimate = :final)
plot(wp; title = "Oregonator: Stiff vs Non-Stiff (final error)")


abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg => MethodOfSteps(Rosenbrock23())),
    Dict(:alg => MethodOfSteps(Rodas4())),
    Dict(:alg => MethodOfSteps(Rodas5())),
    Dict(:alg => MethodOfSteps(Rodas5P())),
    Dict(:alg => MethodOfSteps(TRBDF2())),
    Dict(:alg => MethodOfSteps(KenCarp4()))]
names = ["Rosenbrock23", "Rodas4", "Rodas5", "Rodas5P", "TRBDF2", "KenCarp4"]
wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :final)
plot(wp; title = "Oregonator: Low Tolerances (final error)")


wp = WorkPrecisionSet(prob_dde_RADAR5_oregonator, abstols, reltols, setups;
    names = names, appxsol = test_sol, maxiters = Int(1e5), error_estimate = :L2)
plot(wp; title = "Oregonator: Low Tolerances (L2 error)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

