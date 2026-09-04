
using OrdinaryDiffEq
using OrdinaryDiffEqBDF, OrdinaryDiffEqExponentialRK, OrdinaryDiffEqFIRK, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqIMEXMultistep, OrdinaryDiffEqRosenbrock
using DiffEqDevTools
using SciMLOperators
using LinearSolve
using LinearAlgebra
using SparseArrays
using Sundials
using SummationByPartsOperators
const SBP = SummationByPartsOperators
using Plots
gr()


function nls_nonlinear!(du, w, p, t)
    N = p.N
    u = @view w[1:N]
    v = @view w[N+1:2*N]
    du_u = @view du[1:N]
    du_v = @view du[N+1:2*N]
    @. du_u = -2 * (u^2 + v^2) * v
    @. du_v =  2 * (u^2 + v^2) * u
end

function nonlinear_schrodinger(N, L, alpha)
    D1 = fourier_derivative_operator(xmin = -L, xmax = L, N = N)
    D2 = D1^2  # Second derivative via squaring first derivative
    x = SBP.grid(D1)

    D2_mat = Matrix(D2)
    Z = zeros(N, N)
    A = alpha * [Z  -D2_mat;
                 D2_mat  Z]

    u0 = @. cos(π * x / L)
    v0 = zeros(N)
    w0 = [u0; v0]

    p = (; N)
    tspan = (0.0, 1.0)
    prob = SplitODEProblem(MatrixOperator(A), nls_nonlinear!, w0, tspan, p)

    return x, prob
end;


L = 16.0 # Domain half-length
n = 256 # Number of grid points
alpha = 5.0 # Dispersive coefficient
xs, prob = nonlinear_schrodinger(n, L, alpha)

@time sol = solve(prob, AutoVern7(Rodas5P(autodiff=AutoFiniteDiff()));
                  dt = 1e-4, reltol = 1e-12, abstol = 1e-12);

test_sol = TestSolution(sol) # Reference solution for error estimation

tslices = LinRange(prob.tspan..., 50)
ys_u = mapreduce(t -> sol(t)[1:n], hcat, tslices)
plt = heatmap(xs, tslices, ys_u', xlabel = "x", ylabel = "t", title="Re(ψ)")


abstols = 0.1 .^ (5:8)
reltols = 0.1 .^ (2:5)
setups = [
    Dict(:alg => Tsit5()),
    Dict(:alg => Vern7()),
    Dict(:alg => Vern9()),
    Dict(:alg => DP8()),
]
labels = hcat(
    "Tsit5",
    "Vern7",
    "Vern9",
    "DP8",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e7));

plot(wp, label=labels, markershape=:auto, title="Explicit Methods")


abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => IMEXEuler(), :dts => 1e-3 * multipliers),
    Dict(:alg => CNAB2(), :dts => 1e-3 * multipliers),
    Dict(:alg => CNLF2(), :dts => 1e-3 * multipliers),
    Dict(:alg => SBDF2(), :dts => 1e-3 * multipliers),
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

plot(wp, label=labels, markershape=:auto, title="IMEX Methods, High Tolerance")


abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => NorsettEuler(krylov=true, m=5), :dts => 1e-3 * multipliers),
    Dict(:alg => NorsettEuler(krylov=true, m=20), :dts => 1e-3 * multipliers),
    Dict(:alg => ETDRK2(krylov=true, m=5), :dts => 1e-3 * multipliers),
    Dict(:alg => ETDRK2(krylov=true, m=20), :dts => 1e-3 * multipliers),
]
labels = hcat(
    "NorsettEuler (m=5)",
    "NorsettEuler (m=20)",
    "ETDRK2 (m=5)",
    "ETDRK2 (m=20)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="ExpRK Methods, High Tolerance")


abstols = 0.1 .^ (5:8)
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [
    Dict(:alg => Vern7()),
    Dict(:alg => CNAB2(), :dts => 1e-3 * multipliers),
    Dict(:alg => CNAB2(linsolve=KrylovJL_GMRES()), :dts => 1e-3 * multipliers),
    Dict(:alg => ETDRK2(krylov=true, m=20), :dts => 1e-3 * multipliers),
]
labels = hcat(
    "Vern7",
    "CNAB2 (dense linsolve)",
    "CNAB2 (Krylov linsolve)",
    "ETDRK2 (Krylov, m=20)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e7));

plot(wp, label=labels, markershape=:auto, title="Between Families, High Tolerances")


abstols = 0.1 .^ (8:11)
reltols = 0.1 .^ (5:8)
setups = [
    Dict(:alg => Vern7()),
    Dict(:alg => Vern9()),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=3, linear_solver=:Dense)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=4, linear_solver=:Dense)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Dense)),
]
labels = hcat(
    "Vern7",
    "Vern9",
    "ARKODE3",
    "ARKODE4",
    "ARKODE5",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e7));

plot(wp, label=labels, markershape=:auto, title="Non-Stiff vs IMEX, Low Tolerances")


abstols = 0.1 .^ (7:11) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => ETDRK3(krylov=true, m=20), :dts => 1e-3 * multipliers),
    Dict(:alg => ETDRK4(krylov=true, m=20), :dts => 1e-3 * multipliers),
    Dict(:alg => HochOst4(krylov=true, m=20), :dts => 1e-3 * multipliers),
]
labels = hcat(
    "ETDRK3 (m=20)",
    "ETDRK4 (m=20)",
    "HochOst4 (m=20)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="ExpRK Methods, Low Tolerances")


abstols = 0.1 .^ (7:11)
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [
    Dict(:alg => Vern7()),
    Dict(:alg => Vern9()),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Dense)),
    Dict(:alg => ETDRK3(krylov=true, m=20), :dts => 1e-3 * multipliers),
    Dict(:alg => ETDRK4(krylov=true, m=20), :dts => 1e-3 * multipliers),
]
labels = hcat(
    "Vern7",
    "Vern9",
    "ARKODE5",
    "ETDRK3 (m=20)",
    "ETDRK4 (m=20)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e7));

plot(wp, label=labels, markershape=:auto, title="Between Families, Low Tolerances")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

