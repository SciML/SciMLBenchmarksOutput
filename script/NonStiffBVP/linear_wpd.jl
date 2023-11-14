
using BoundaryValueDiffEq, OrdinaryDiffEq, ODEInterface, DiffEqDevTools, BenchmarkTools, BVProblemLibrary, Plots


setups = [ Dict(:alg=>MIRK2(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK3(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK4(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK5(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK6(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>BVPM2(), :dts=>1.0 ./ 7.0 .^ (1:4)),
            Dict(:alg=>Shooting(Tsit5())),
            Dict(:alg=>Shooting(Vern7())),
            Dict(:alg=>Shooting(DP5())),
            Dict(:alg=>MultipleShooting(10, Tsit5())),
            Dict(:alg=>MultipleShooting(10, Vern7())),
            Dict(:alg=>MultipleShooting(10, DP5())),]
labels = ["MIRK2";
               "MIRK3";
               "MIRK4";
               "MIRK5";
               "MIRK6";
               "BVPM2";
               "Shooting (Tsit5)";
               "Shooting (Vern7)";
               "Shooting (DP5)";
               "MultipleShooting (Tsit5)";
               "MultipleShooting (Vern7)";
               "MultipleShooting (DP5)"];


abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3);


prob_1 = BVProblemLibrary.prob_bvp_linear_1
sol_1 = solve(prob_1, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_1 = TestSolution(sol_1)
wp_1 = WorkPrecisionSet(prob_1, abstols, reltols, setups; names = labels, appxsol = testsol_1, maxiters=Int(1e5))
plot(wp_1)


prob_2 = BVProblemLibrary.prob_bvp_linear_2
sol_2 = solve(prob_2, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_2 = TestSolution(sol_2)
wp_2 = WorkPrecisionSet(prob_2, abstols, reltols, setups; names = labels, appxsol = testsol_2, maxiters=Int(1e5))
plot(wp_2)


prob_3 = BVProblemLibrary.prob_bvp_linear_3
sol_3 = solve(prob_3, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_3 = TestSolution(sol_3)
wp_3 = WorkPrecisionSet(prob_3, abstols, reltols, setups; names = labels, appxsol = testsol_3, maxiters=Int(1e4))
plot(wp_3)


prob_4 = BVProblemLibrary.prob_bvp_linear_4
sol_4 = solve(prob_4, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_4 = TestSolution(sol_4)
wp_4 = WorkPrecisionSet(prob_4, abstols, reltols, setups; names = labels, appxsol = testsol_4, maxiters=Int(1e4))
plot(wp_4)


prob_5 = BVProblemLibrary.prob_bvp_linear_5
sol_5 = solve(prob_5, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_5 = TestSolution(sol_5)
wp_5 = WorkPrecisionSet(prob_5, abstols, reltols, setups; names = labels, appxsol = testsol_5, maxiters=Int(1e4))
plot(wp_5)


prob_6 = BVProblemLibrary.prob_bvp_linear_6
sol_6 = solve(prob_6, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_6 = TestSolution(sol_6.t, sol_6.u)
wp_6 = WorkPrecisionSet(prob_6, abstols, reltols, setups; names = labels, appxsol = testsol_6, maxiters=Int(1e4))
plot(wp_6)


prob_7 = BVProblemLibrary.prob_bvp_linear_7
sol_7 = solve(prob_7, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_7 = TestSolution(sol_7)
wp_7 = WorkPrecisionSet(prob_7, abstols, reltols, setups; names = labels, appxsol = testsol_7, maxiters=Int(1e4))
plot(wp_7)


prob_8 = BVProblemLibrary.prob_bvp_linear_8
sol_8 = solve(prob_8, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_8 = TestSolution(sol_8)
wp_8 = WorkPrecisionSet(prob_8, abstols, reltols, setups; names = labels, appxsol = testsol_8, maxiters=Int(1e4))
plot(wp_8)


prob_9 = BVProblemLibrary.prob_bvp_linear_9
sol_9 = solve(prob_9, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_9 = TestSolution(sol_9)
wp_9 = WorkPrecisionSet(prob_9, abstols, reltols, setups; names = labels, appxsol = testsol_9, maxiters=Int(1e4))
plot(wp_9)


prob_10 = BVProblemLibrary.prob_bvp_linear_10
sol_10 = solve(prob_10, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_10 = TestSolution(sol_10)
wp_10 = WorkPrecisionSet(prob_10, abstols, reltols, setups; names = labels, appxsol = testsol_10, maxiters=Int(1e4))
plot(wp_10)


prob_11 = BVProblemLibrary.prob_bvp_linear_11
sol_11 = solve(prob_11, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_11 = TestSolution(sol_11)
wp_11 = WorkPrecisionSet(prob_11, abstols, reltols, setups; names = labels, appxsol = testsol_11, maxiters=Int(1e4))
plot(wp_11)


prob_12 = BVProblemLibrary.prob_bvp_linear_12
sol_12 = solve(prob_12, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_12 = TestSolution(sol_12)
wp_12 = WorkPrecisionSet(prob_12, abstols, reltols, setups; names = labels, appxsol = testsol_12, maxiters=Int(1e4))
plot(wp_12)


prob_13 = BVProblemLibrary.prob_bvp_linear_13
sol_13 = solve(prob_13, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_13 = TestSolution(sol_13)
wp_13 = WorkPrecisionSet(prob_13, abstols, reltols, setups; names = labels, appxsol = testsol_13, maxiters=Int(1e4))
plot(wp_13)


prob_14 = BVProblemLibrary.prob_bvp_linear_14
sol_14 = solve(prob_14, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_14 = TestSolution(sol_14)
wp_14 = WorkPrecisionSet(prob_14, abstols, reltols, setups; names = labels, appxsol = testsol_14, maxiters=Int(1e4))
plot(wp_14)


prob_15 = BVProblemLibrary.prob_bvp_linear_15
sol_15 = solve(prob_15, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_15 = TestSolution(sol_15)
wp_15 = WorkPrecisionSet(prob_15, abstols, reltols, setups; names = labels, appxsol = testsol_15, maxiters=Int(1e4))
plot(wp_15)


prob_16 = BVProblemLibrary.prob_bvp_linear_16
sol_16 = solve(prob_16, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_16 = TestSolution(sol_16)
wp_16 = WorkPrecisionSet(prob_16, abstols, reltols, setups; names = labels, appxsol = testsol_16, maxiters=Int(1e4))
plot(wp_16)


prob_17 = BVProblemLibrary.prob_bvp_linear_17
sol_17 = solve(prob_17, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_17 = TestSolution(sol_17)
wp_17 = WorkPrecisionSet(prob_17, abstols, reltols, setups; names = labels, appxsol = testsol_17, maxiters=Int(1e4))
plot(wp_17)


prob_18 = BVProblemLibrary.prob_bvp_linear_18
sol_18 = solve(prob_18, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
testsol_18 = TestSolution(sol_18)
wp_18 = WorkPrecisionSet(prob_18, abstols, reltols, setups; names = labels, appxsol = testsol_18, maxiters=Int(1e4))
plot(wp_18)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

