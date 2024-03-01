
using BoundaryValueDiffEq, OrdinaryDiffEq, ODEInterface, DiffEqDevTools, BenchmarkTools, BVProblemLibrary, Plots


setups = [ Dict(:alg=>MIRK2(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK3(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK4(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK5(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>MIRK6(), :dts=>1.0 ./ 10.0 .^ (1:4)),
            Dict(:alg=>BVPM2(), :dts=>1.0 ./ 7.0 .^ (1:4)),
            Dict(:alg=>COLNEW(), :dts=>1.0 ./ 10.0 .^ (1:4)),
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
               "COLNEW";
               "Shooting (Tsit5)";
               "Shooting (Vern7)";
               "Shooting (DP5)";
               "MultipleShooting (Tsit5)";
               "MultipleShooting (Vern7)";
               "MultipleShooting (DP5)"];


abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3);


function benchmark!(prob)
    sol = solve(prob, Shooting(Vern7()), abstol=1e-14, reltol=1e-14)
    testsol = TestSolution(sol)
    wp = WorkPrecisionSet(prob, abstols, reltols, setups; names = labels, appxsol = testsol, maxiters=Int(1e4))
    plot(wp, legend=:outertopright)
end


prob_1 = BVProblemLibrary.prob_bvp_linear_1
benchmark!(prob_1)


prob_2 = BVProblemLibrary.prob_bvp_linear_2
benchmark!(prob_2)


prob_3 = BVProblemLibrary.prob_bvp_linear_3
benchmark!(prob_3)


prob_4 = BVProblemLibrary.prob_bvp_linear_4
benchmark!(prob_4)


prob_5 = BVProblemLibrary.prob_bvp_linear_5
benchmark!(prob_5)


prob_6 = BVProblemLibrary.prob_bvp_linear_6
benchmark!(prob_6)


prob_7 = BVProblemLibrary.prob_bvp_linear_7
benchmark!(prob_7)


prob_8 = BVProblemLibrary.prob_bvp_linear_8
benchmark!(prob_8)


prob_9 = BVProblemLibrary.prob_bvp_linear_9
benchmark!(prob_9)


prob_10 = BVProblemLibrary.prob_bvp_linear_10
benchmark!(prob_10)


prob_11 = BVProblemLibrary.prob_bvp_linear_11
benchmark!(prob_11)


prob_12 = BVProblemLibrary.prob_bvp_linear_12
benchmark!(prob_12)


prob_13 = BVProblemLibrary.prob_bvp_linear_13
benchmark!(prob_13)


prob_14 = BVProblemLibrary.prob_bvp_linear_14
benchmark!(prob_14)


prob_15 = BVProblemLibrary.prob_bvp_linear_15
benchmark!(prob_15)


prob_16 = BVProblemLibrary.prob_bvp_linear_16
benchmark!(prob_16)


prob_17 = BVProblemLibrary.prob_bvp_linear_17
benchmark!(prob_17)


prob_18 = BVProblemLibrary.prob_bvp_linear_18
benchmark!(prob_18)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

