
using OrdinaryDiffEq
using DiffEqDevTools
using SciMLOperators
using LinearSolve
using LinearAlgebra
using SparseArrays
using Sundials
using ClassicalOrthogonalPolynomials
const COP = ClassicalOrthogonalPolynomials
using Plots
gr();


# Nonlinear component of vector field
function f_nonlinear!(du, u, p, t)
    M, D0, D1, tmp = p
    tmp[2:end-1] .= -(D0 * u) .* (D1 * u) # Nonlinear convection
    ldiv!(du, M, tmp) # Solve the linear system M * du = tmp
end

# Constructor
function burgers(n)
    T = ChebyshevT() # Chebyshev basis
    x1 = reverse(ChebyshevGrid{1}(n - 2)) # 1st kind points, sorted
    x2 = reverse(ChebyshevGrid{2}(n)) # 2nd kind points, sorted
    V = T[x2, 1:n] # Vandermonde matrix, its inverse is transform from values to coefficients
    D0 = diff(T, 0)[x1, 1:n] / V # 0th derivative from x2 to x1 (identity)
    D1 = diff(T, 1)[x1, 1:n] / V # 1st derivative from x2 to x1
    D2 = diff(T, 2)[x1, 1:n] / V # 2nd derivative from x2 to x1


    u0 = @. exp(-x2^2 / (2 * 0.1^2)) # Initial condition
    B_l = [1 zeros(1, n-1)] # Left Dirichlet conditions
    B_r = reverse(B_l) # Right Dirichlet
    M = lu([B_l; D0; B_r]) # Mass matrix for the problem
    nu = 1e-2 # Diffusion coefficient
    D2_bc = [zeros(1, n); D2; zeros(1, n)] # 2nd derivative with boundary conditions
    prob = SplitODEProblem(
                MatrixOperator(nu * (M \ D2_bc)),
                f_nonlinear!, 
                u0, (0.0, 1.0),
                (M, D0, D1, similar(u0))
            )

    return x2, prob
end;


n = 128 # Number of Chebyshev points
xs, prob = burgers(n)
@time sol = solve(prob, Rodas5(autodiff=false), reltol = 1e-12, abstol = 1e-12)

test_sol = TestSolution(sol) # Reference solution for error estimation

tslices = LinRange(prob.tspan..., 50)
ys = mapreduce(sol, hcat, tslices)
heatmap(xs, tslices, ys')


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
    Dict(:alg => NorsettEuler(krylov=true, m=5), :dts => 1e-4 * multipliers),
    Dict(:alg => NorsettEuler(krylov=true, m=20), :dts => 1e-4 * multipliers),
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
    "CNAB2 (dense linsolve)",
    "CNAB2 (Krylov linsolve)", 
    "ETDRK2 (m=5)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="Between Families, High Tolerances")


abstols = 0.1 .^ (7:13)
reltols = 0.1 .^ (4:10)
setups = [
    Dict(:alg => KenCarp3()),
    Dict(:alg => KenCarp4()),
    Dict(:alg => KenCarp5()),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=3, linear_solver=:Band, jac_upper=1, jac_lower=1)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=4, linear_solver=:Band, jac_upper=1, jac_lower=1)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Band, jac_upper=1, jac_lower=1))
]
labels = hcat(
    "KenCarp3",
    "KenCarp4",
    "KenCarp5",
    "ARKODE3",
    "ARKODE4",
    "ARKODE5",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="IMEX Methods, Band Linsolve, Low Tolerances")


abstols = 0.1 .^ (7:13)
reltols = 0.1 .^ (4:10)
setups = [
    Dict(:alg => KenCarp3(linsolve=KrylovJL_GMRES())),
    Dict(:alg => KenCarp4(linsolve=KrylovJL_GMRES())),
    Dict(:alg => KenCarp5(linsolve=KrylovJL_GMRES())),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=3, linear_solver=:GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=4, linear_solver=:GMRES)),
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:GMRES)),
]
labels = hcat(
    "KenCarp3",
    "KenCarp4",
    "KenCarp5",
    "ARKODE3",
    "ARKODE4",
    "ARKODE5",
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
    Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Band, jac_upper=1, jac_lower=1)),
    Dict(:alg => ETDRK3(), :dts => 1e-2 * multipliers),
    Dict(:alg => ETDRK4(), :dts => 1e-2 * multipliers),
]
labels = hcat(
    "ARKODE (Band linsolve)",
    "ETDRK3 (caching)",
    "ETDRK4 (caching)",
)
@time wp = WorkPrecisionSet(prob, abstols, reltols, setups;
    print_names=true, names=labels, numruns=5, error_estimate=:l2,
    save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="Between Families, Low Tolerances")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

