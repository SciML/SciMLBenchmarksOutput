using OrdinaryDiffEq
using OrdinaryDiffEqBDF, OrdinaryDiffEqExponentialRK, OrdinaryDiffEqFIRK, OrdinaryDiffEqIMEXMultistep, OrdinaryDiffEqMultirate, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSDIRK, OrdinaryDiffEqStabilizedRK
using ADTypes: AutoFiniteDiff
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
    D1 = periodic_derivative_operator(
        derivative_order = 1, accuracy_order = 4,
        xmin = -L, xmax = L, N = N
    )
    D2 = periodic_derivative_operator(
        derivative_order = 2, accuracy_order = 4,
        xmin = -L, xmax = L, N = N
    )
    D4 = periodic_derivative_operator(
        derivative_order = 4, accuracy_order = 4,
        xmin = -L, xmax = L, N = N
    )

    x = SBP.grid(D1)
    u0 = @. cos(2π * x / L) # Initial condition
    p = (; D1, alpha)

    tspan = (0.0, 1.0)
    prob = SplitODEProblem(
        MatrixOperator(-p.alpha / 2 * (sparse(D2) + 1 / 4 * sparse(D4))),
        nonlinear_convection!,
        u0, tspan, p
    )

    return x, prob
end;


N = 128  # Number of grid points
L = 16.0  # Domain length
alpha = 30.0 # Time scaling factor
xs, prob = kuramoto_sivashinsky(N, L, alpha)
@time sol = solve(prob, RadauIIA5(autodiff = AutoFiniteDiff()); dt = 1.0e-4, abstol = 1.0e-14, reltol = 1.0e-14, adaptive = true)

test_sol = TestSolution(sol);

tslices = LinRange(prob.tspan..., 50)
ys = mapreduce(sol, hcat, tslices)
plt = heatmap(xs, tslices, ys', xlabel = "x", ylabel = "t")


abstols = 0.1 .^ (3:6) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (3:6)
multipliers = 0.3 .^ (0:3)
setups = [
    Dict(:alg => IMEXEuler(), :dts => 1.0e-4 * multipliers),
    Dict(:alg => CNAB2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => CNLF2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => SBDF2(), :dts => 1.0e-3 * multipliers),
]
labels = hcat(
    "IMEXEuler",
    "CNAB2",
    "CNLF2",
    "SBDF2",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "IMEX Methods, High Tolerance")


abstols = 0.1 .^ (3:6) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (3:6)
multipliers = 0.3 .^ (0:3)
setups = [
    Dict(:alg => NorsettEuler(), :dts => 1.0e-4 * multipliers),
    Dict(:alg => NorsettEuler(krylov = true, m = 5), :dts => 1.0e-4 * multipliers),
    Dict(:alg => NorsettEuler(krylov = true, m = 20), :dts => 1.0e-4 * multipliers),
    Dict(:alg => ETDRK2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ETDRK2(krylov = true, m = 5), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ETDRK2(krylov = true, m = 20), :dts => 1.0e-3 * multipliers),
]
labels = hcat(
    "NorsettEuler (caching)",
    "NorsettEuler (m=5)",
    "NorsettEuler (m=20)",
    "ETDRK2 (caching)",
    "ETDRK2 (m=5)",
    "ETDRK2 (m=20)"
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "ExpRK Methods, High Tolerance")


abstols = 0.1 .^ (3:6)
reltols = 0.1 .^ (3:6)

setups = [
    Dict(:alg => ROCK4()),
    Dict(:alg => TSRKC3()),
    Dict(:alg => RKC()),
    Dict(:alg => ROCK2()),
]
labels = hcat(
    "ROCK4",
    "TSRKC3",
    "RKC",
    "ROCK2",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e5)
);

plot(wp, label = labels, markershape = :auto, title = "Stabilized Methods, High Tolerance")


abstols = 0.1 .^ (4:7) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (4:7)
multipliers = 0.5 .^ (0:3)
# The joined problem calls f1 and f2 through its own buffer: a direct call of the
# SplitFunction uses an internal scratch cache that aliases prob.u0, so it would
# silently overwrite the initial condition of every later solve in the sweep.
joined_rhs! = let stiff_cache = zero(prob.u0)
    (du, u, p, t) -> begin
        prob.f.f1(stiff_cache, u, p, t)
        prob.f.f2(du, u, p, t)
        du .+= stiff_cache
    end
end
prob_joined = ODEProblem(joined_rhs!, copy(prob.u0), prob.tspan, prob.p)
setups = [
    Dict(:alg => MREEF(m = 32, order = 4), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => MRAB(k = 2, m = 32), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => MIS(m = 32), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => MRIGARKERK22a(m = 32), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => MRIGARKERK33a(m = 32), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => MRIGARKERK45a(m = 32), :adaptive => false, :dts => 2.0e-3 * multipliers),
    Dict(:alg => CNAB2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ETDRK2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => KenCarp4(linsolve = KrylovJL_GMRES())),
    Dict(:alg => Tsit5(), :prob_choice => 2),
    Dict(:alg => Rodas5P(autodiff = AutoFiniteDiff()), :prob_choice => 2),
    Dict(:alg => FBDF(autodiff = AutoFiniteDiff()), :prob_choice => 2),
    Dict(:alg => NordsieckBDF(autodiff = AutoFiniteDiff()), :prob_choice => 2),
]
labels = hcat(
    "MREEF",
    "MRAB",
    "MIS",
    "MRIGARKERK22a",
    "MRIGARKERK33a",
    "MRIGARKERK45a",
    "CNAB2",
    "ETDRK2",
    "ETDRK4",
    "KenCarp4 (Krylov)",
    "Tsit5",
    "Rodas5P",
    "FBDF", "NordsieckBDF",
)
@time wp = WorkPrecisionSet(
    [prob, prob_joined], abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = [test_sol, test_sol], maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "Multirate Methods, High Tolerance")


Astiff = convert(AbstractMatrix, prob.f.f1.f)
p_stiff = (; D1 = prob.p.D1, alpha = prob.p.alpha, A = Astiff)
stiff_linear!(du, u, p, t) = mul!(du, p.A, u)
prob_swapped = SplitODEProblem(
    nonlinear_convection!, stiff_linear!,
    prob.u0, prob.tspan, p_stiff
)

abstols = 0.1 .^ (4:8)
reltols = 0.1 .^ (4:8)
setups = [
    Dict(:alg => MRIGARKIRK21a(m = 2, autodiff = AutoFiniteDiff())),
    Dict(:alg => MRIGARKESDIRK34a(m = 2, autodiff = AutoFiniteDiff())),
    Dict(:alg => MRIGARKESDIRK46a(m = 2, autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(linsolve = KrylovJL_GMRES()), :prob_choice => 2),
    Dict(:alg => KenCarp4(linsolve = KrylovJL_GMRES()), :prob_choice => 2),
    Dict(:alg => ETDRK4(), :prob_choice => 2, :dts => 1.0e-3 * 0.5 .^ (0:4)),
    Dict(:alg => Tsit5(), :prob_choice => 3),
    Dict(:alg => Rodas5P(autodiff = AutoFiniteDiff()), :prob_choice => 3),
    Dict(:alg => FBDF(autodiff = AutoFiniteDiff()), :prob_choice => 3),
    Dict(:alg => NordsieckBDF(autodiff = AutoFiniteDiff()), :prob_choice => 3),
]
labels = hcat(
    "MRIGARKIRK21a",
    "MRIGARKESDIRK34a",
    "MRIGARKESDIRK46a",
    "KenCarp3 (Krylov)",
    "KenCarp4 (Krylov)",
    "ETDRK4",
    "Tsit5",
    "Rodas5P",
    "FBDF", "NordsieckBDF",
)
@time wp = WorkPrecisionSet(
    [prob_swapped, prob, prob_joined], abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = [test_sol, test_sol, test_sol], maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "Implicit Multirate Methods")


abstols = 0.1 .^ (3:6)
reltols = 0.1 .^ (3:6)
multipliers = 0.3 .^ (0:3)
setups = [
    Dict(:alg => CNAB2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => CNAB2(linsolve = KrylovJL_GMRES()), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ETDRK2(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ROCK4()),
    Dict(:alg => TSRKC3()),
]
labels = hcat(
    "CNAB2 (dense)",
    "CNAB2 (Krylov)",
    "ETDRK2 (caching)",
    "ROCK4",
    "TSRKC3",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e5)
);

plot(wp, label = labels, markershape = :auto, title = "Between Families, High Tolerances")


abstols = 0.1 .^ (7:11)
reltols = 0.1 .^ (7:11)
setups = [
    Dict(:alg => KenCarp3()),
    Dict(:alg => KenCarp4()),
    Dict(:alg => KenCarp5()),
    Dict(:alg => KenCarp3(linsolve = KrylovJL_GMRES())),
    Dict(:alg => KenCarp4(linsolve = KrylovJL_GMRES())),
    Dict(:alg => KenCarp5(linsolve = KrylovJL_GMRES())),
    Dict(:alg => ARKODE(Sundials.Implicit(), order = 3, linear_solver = :GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order = 4, linear_solver = :GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order = 5, linear_solver = :GMRES)),
]
labels = hcat(
    "KenCarp3 (default)",
    "KenCarp4 (default)",
    "KenCarp5 (default)",
    "KenCarp3 (Krylov)",
    "KenCarp4 (Krylov)",
    "KenCarp5 (Krylov)",
    "ARKODE3 (Krylov)",
    "ARKODE4 (Krylov)",
    "ARKODE5 (Krylov)",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e5)
);

plot(wp, label = labels, markershape = :auto, title = "IMEX Methods, Low Tolerances")


abstols = 0.1 .^ (7:11) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => ETDRK3(), :dts => 1.0e-4 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => HochOst4(), :dts => 1.0e-3 * multipliers),
]
labels = hcat(
    "ETDRK3 (caching)",
    "ETDRK4 (caching)",
    "HochOst4 (caching)",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "ExpRK Methods, Low Tolerances")


abstols = 0.1 .^ (7:11)
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => ARKODE(Sundials.Implicit(), order = 5, linear_solver = :GMRES)),
    Dict(:alg => ETDRK3(), :dts => 1.0e-4 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1.0e-3 * multipliers),
    Dict(:alg => ROCK4()),
    Dict(:alg => TSRKC3()),
]
labels = hcat(
    "ARKODE5 (Krylov)",
    "ETDRK3 (caching)",
    "ETDRK4 (caching)",
    "ROCK4",
    "TSRKC3",
)
@time wp = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    print_names = true, names = labels, numruns = 5, error_estimate = :l2,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1.0e6)
);

plot(wp, label = labels, markershape = :auto, title = "Between Families, Low Tolerances")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
