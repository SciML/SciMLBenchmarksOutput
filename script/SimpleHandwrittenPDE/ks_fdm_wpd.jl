
using OrdinaryDiffEq
using DiffEqDevTools
using SciMLOperators
using LinearSolve
using LinearAlgebra
using SparseArrays
using Sundials
using SummationByPartsOperators
const SBP = SummationByPartsOperators
using Plots
gr();


nonlinear_convection!(du, u, p, t) = du .= (-p.alpha / 3) * (u .* (p.D1 * u) + p.D1 * (u .^ 2))

# Construct the problem
function kuramoto_sivashinsky(N, L, alpha)
    D1 = periodic_derivative_operator(derivative_order = 1, accuracy_order = 4,
                                      xmin = -L, xmax = L, N = N)
    D2 = periodic_derivative_operator(derivative_order = 2, accuracy_order = 4,
                                      xmin = -L, xmax = L, N = N)
    D4 = periodic_derivative_operator(derivative_order = 4, accuracy_order = 4,
                                      xmin = -L, xmax = L, N = N)

    x = SBP.grid(D1)
    u0 = @. cos(2π * x / L) # Initial condition
    p = (; D1, alpha)

    tspan = (0.0, 1.0)
    prob = SplitODEProblem(MatrixOperator(-p.alpha / 2 * (sparse(D2) + 1/4 * sparse(D4))),
                           nonlinear_convection!, 
                           u0, tspan, p);

    return x, prob
end;


N = 128  # Number of grid points
L = 16.0  # Domain length
alpha = 30.0 # Time scaling factor
xs, prob = kuramoto_sivashinsky(N, L, alpha)
@time sol = solve(prob, RadauIIA5(autodiff=false); dt = 1e-4, abstol=1e-14, reltol=1e-14, adaptive=true)

test_sol = TestSolution(sol);

tslices = LinRange(prob.tspan..., 50)
ys = mapreduce(sol, hcat, tslices)
plt = heatmap(xs, tslices, ys', xlabel="x", ylabel="t")


abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => IMEXEuler(), :dts => 1e-4 * multipliers),
    Dict(:alg => CNAB2(), :dts => 1e-4 * multipliers),
    Dict(:alg => CNLF2(), :dts => 1e-4 * multipliers),
    Dict(:alg => SBDF2(), :dts => 1e-4 * multipliers),
]
labels = hcat(
    "IMEXEuler",
    "CNAB2",
    "CNLF2",
    "SBDF2",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="Work-Precision Diagram, High Tolerance")


abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => NorsettEuler(), :dts => 1e-4 * multipliers),
    Dict(:alg => NorsettEuler(krylov=true, m=5), :dts => 1e-4 * multipliers), # Maxiters
    Dict(:alg => NorsettEuler(krylov=true, m=20), :dts => 1e-4 * multipliers), # Maxiters
    Dict(:alg => ETDRK2(), :dts => 1e-4 * multipliers),
    Dict(:alg => ETDRK2(krylov=true, m=5), :dts => 1e-4 * multipliers),
    Dict(:alg => ETDRK2(krylov=true, m=20), :dts => 1e-4 * multipliers)
]
labels = hcat(
    "NorsettEuler (caching)", 
    "NorsettEuler (m=5)",
    "NorsettEuler (m=20)",
    "ETDRK2 (caching)", 
    "ETDRK2 (m=5)", 
    "ETDRK2 (m=20)"
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="ExpRK Methods, High Tolerance")


abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => CNAB2(), :dts => 1e-4 * multipliers),
    Dict(:alg => CNAB2(linsolve=KrylovJL_GMRES()), :dts => 1e-4 * multipliers),
    Dict(:alg => ETDRK2(), :dts => 1e-4 * multipliers),
]
labels = hcat(
    "CNAB2 (dense)",
    "CNAB2 (Krylov)", 
    "ETDRK2 (caching)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="Between Families, High Tolerances")


abstols = 0.1 .^ (8:12)
reltols = 0.1 .^ (5:9)
setups = [
    Dict(:alg => KenCarp3()),
    Dict(:alg => KenCarp4()),
    Dict(:alg => KenCarp5()),
    Dict(:alg => KenCarp3(linsolve=KrylovJL_GMRES())),
    Dict(:alg => KenCarp4(linsolve=KrylovJL_GMRES())),
    Dict(:alg => KenCarp5(linsolve=KrylovJL_GMRES())),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=3, linear_solver=:GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=4, linear_solver=:GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:GMRES)),
]
labels = hcat(
    "KenCarp3 (dense)",
    "KenCarp4 (dense)",
    "KenCarp5 (dense)",
    "KenCarp3 (Krylov)",
    "KenCarp4 (Krylov)",
    "KenCarp5 (Krylov)",
    "ARKODE3 (Krylov)",
    "ARKODE4 (Krylov)",
    "ARKODE5 (Krylov)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="IMEX Methods, Krylov Linsolve, Low Tolerances")


abstols = 0.1 .^ (7:11) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => ETDRK3(), :dts => 1e-2 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1e-2 * multipliers),
    Dict(:alg => HochOst4(), :dts => 1e-2 * multipliers),
]
labels = hcat(
    "ETDRK3 (caching)",
    "ETDRK4 (caching)",
    "HochOst4 (caching)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="ExpRK Methods, Low Tolerances")


abstols = 0.1 .^ (7:11)
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:GMRES)),
    Dict(:alg => ETDRK3(), :dts => 1e-2 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1e-2 * multipliers),
]
labels = hcat(
    "ARKODE5 (Krylov)",
    "ETDRK3 (caching)",
    "ETDRK4 (caching)"
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="Between Families, Low Tolerances")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

