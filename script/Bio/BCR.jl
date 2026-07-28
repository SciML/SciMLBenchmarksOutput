
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools,
      LinearSolve, RecursiveFactorization
using OrdinaryDiffEqBDF, OrdinaryDiffEqSDIRK

gr()
datadir = joinpath(dirname(pathof(ReactionNetworkImporters)), "../data/bcr")
const to = TimerOutput()
tf = 100000.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(datadir, "bcr.net"))
show(to)
rn = complete(prnbng)
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = complete(Catalyst.ode_model(rn))
show(to)

tspan = (0.0, tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[])
show(to)
oprob_sparse = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[]; sparse = true);


@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[], jac = true, sparse = true)
show(to)


@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Approx. number of terms in the ODE
@show length(parameters(rn)); # Number of Parameters


u = oprob.u0
du = copy(u)
p = oprob.p
@timeit to "ODE rhs Eval1" oprob.f(du, u, p, 0.0)
@timeit to "ODE rhs spjac Eval1" sparsejacprob.f(du, u, p, 0.0)
show(to)


@btime oprob.f($du, $u, $p, 0.0)


Js = similar(sparsejacprob.f.jac_prototype)
@timeit to "SparseJac Eval1" sparsejacprob.f.jac(Js, u, p, 0.0)
@timeit to "SparseJac Eval2" sparsejacprob.f.jac(Js, u, p, 0.0)
show(to)


sol = solve(oprob, CVODE_BDF(), saveat = tf/1000.0, reltol = 1e-5, abstol = 1e-5)
plot(sol; idxs = obs, legend = false, fmt = :png)


@time sol = solve(oprob, CVODE_BDF(), abstol = 1/10^12, reltol = 1/10^12)
test_sol = TestSolution(sol);


default(legendfontsize = 7, framestyle = :box, gridalpha = 0.3, gridlinewidth = 2.5)


using IncompleteLU, LinearAlgebra
const τ = 1e2
const τ2 = 1e2

jaccache = sparsejacprob.f.jac(oprob.u0, oprob.p, 0.0)
W = I - 1.0*jaccache
prectmp = ilu(W, τ = τ)

preccache = Ref(prectmp)

function psetupilu(p, t, u, du, jok, jcurPtr, gamma)
    if !jok
        sparsejacprob.f.jac(jaccache, u, p, t)
        jcurPtr[] = true

        # W = I - gamma*J
        @. W = -gamma*jaccache
        idxs = diagind(W)
        @. @view(W[idxs]) = @view(W[idxs]) + 1

        # Build preconditioner on W
        preccache[] = ilu(W, τ = τ)
    end
end
function precilu(z, r, p, t, y, fy, gamma, delta, lr)
    ldiv!(z, preccache[], r)
end

function incompletelu(A, p)
    Pl = ilu(convert(AbstractMatrix, A); τ = τ2)
    return Pl, I
end;


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (5:8);


try
    solve(sparsejacprob, CVODE_BDF(linear_solver = :KLU), abstol = 1e-8, reltol = 1e-8);
catch e
    println("CVODE_BDF with KLU failed: $e")
end


setups = [
    Dict(:alg=>lsoda(), :prob_choice => 1),
    Dict(:alg=>CVODE_BDF(), :prob_choice => 1),
    Dict(
        :alg=>CVODE_BDF(linear_solver = :GMRES, prec = precilu, psetup = psetupilu, prec_side = 1),
        :prob_choice => 2)
];


wp = WorkPrecisionSet(
    [oprob, oprob_sparse, sparsejacprob], abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = [test_sol, test_sol, test_sol], maxiters = Int(1e6), numruns = 1)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (GMRES, iLU)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(
        linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true)),
    Dict(:alg=>QNDF(linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true)),
    Dict(:alg=>FBDF(linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true)),
    Dict(:alg=>KenCarp4(
        linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true))
];


wp = WorkPrecisionSet(sparsejacprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e6), numruns = 1)

names = ["TRBDF2 (GMRES, iLU)" "QNDF (GMRES, iLU)" "FBDF (GMRES, iLU)" "KenCarp4 (GMRES, iLU)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff())),
    Dict(:alg=>QNDF(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff())),
    Dict(:alg=>FBDF(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff())),
    Dict(:alg=>KenCarp4(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff()))
];


wp = WorkPrecisionSet(sparsejacprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e6), numruns = 1)

names = ["TRBDF2 (KLU, sparse jac)" "QNDF (KLU, sparse jac)" "FBDF (KLU, sparse jac)" "KenCarp4 (KLU, sparse jac)"]
plot(wp; label = names)


const _loser_tol = 1e-6
const _loser_maxiters = Int(1e6)
_solve_kwargs = (; abstol = _loser_tol, reltol = _loser_tol, maxiters = _loser_maxiters,
    save_everystep = false)

loser_labels = String[]
loser_elapsed = Float64[]

function _time_loser!(label, prob, alg)
    println("--- $label ---")
    t = @elapsed sol = solve(prob, alg; _solve_kwargs...)
    @show sol.retcode
    println("elapsed = ", t, " s")
    push!(loser_labels, label)
    push!(loser_elapsed, t)
    return sol
end

# Competitive reference (sparse KLU)
_time_loser!("FBDF + KLU (reference)", sparsejacprob,
    FBDF(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff()))

# Dense CVODE Lapack
_time_loser!("CVODE_BDF LapackDense", oprob, CVODE_BDF(linear_solver = :LapackDense))

# Bare CVODE GMRES (no preconditioner)
_time_loser!("CVODE_BDF GMRES (no prec)", oprob, CVODE_BDF(linear_solver = :GMRES))

# Default dense Julia factorizations on the non-sparse problem
_time_loser!("TRBDF2 (default dense)", oprob, TRBDF2(autodiff = AutoFiniteDiff()))
_time_loser!("QNDF (default dense)", oprob, QNDF(autodiff = AutoFiniteDiff()))
_time_loser!("FBDF (default dense)", oprob, FBDF(autodiff = AutoFiniteDiff()))
_time_loser!("KenCarp4 (default dense)", oprob, KenCarp4(autodiff = AutoFiniteDiff()))

# Unpreconditioned GMRES on the dense residual problem
_time_loser!("TRBDF2 GMRES (no prec)", oprob,
    TRBDF2(linsolve = KrylovJL_GMRES(), autodiff = AutoFiniteDiff()))
_time_loser!("QNDF GMRES (no prec)", oprob,
    QNDF(linsolve = KrylovJL_GMRES(), autodiff = AutoFiniteDiff()))
_time_loser!("FBDF GMRES (no prec)", oprob,
    FBDF(linsolve = KrylovJL_GMRES(), autodiff = AutoFiniteDiff()))
_time_loser!("KenCarp4 GMRES (no prec)", oprob,
    KenCarp4(linsolve = KrylovJL_GMRES(), autodiff = AutoFiniteDiff()))


# Relative cost vs the sparse KLU reference (first entry)
ref_t = loser_elapsed[1]
bar(loser_labels, loser_elapsed ./ ref_t; xrotation = 45, legend = false,
    ylabel = "wall time / (FBDF+KLU reference)",
    title = "BCR loser isolation (tol=$_loser_tol, one solve each)",
    size = (900, 500), left_margin = 5Plots.mm, bottom_margin = 15Plots.mm)


setups = [
    Dict(
        :alg=>CVODE_BDF(linear_solver = :GMRES, prec = precilu, psetup = psetupilu, prec_side = 1),
        :prob_choice => 2),
    Dict(
        :alg=>QNDF(linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true),
        :prob_choice => 3),
    Dict(
        :alg=>FBDF(linsolve = KrylovJL_GMRES(; precs = incompletelu), autodiff = AutoFiniteDiff(), concrete_jac = true),
        :prob_choice => 3),
    Dict(:alg=>QNDF(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff()), :prob_choice => 3),
    Dict(:alg=>FBDF(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff()), :prob_choice => 3),
    Dict(:alg=>KenCarp4(linsolve = KLUFactorization(), autodiff = AutoFiniteDiff()), :prob_choice => 3)
];


wp = WorkPrecisionSet(
    [oprob, oprob_sparse, sparsejacprob], abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = [test_sol, test_sol, test_sol], maxiters = Int(1e9), numruns = 200)

names = ["CVODE_BDF (GMRES, iLU)" "QNDF (GMRES, iLU)" "FBDF (GMRES, iLU)" "QNDF (KLU, sparse jac)" "FBDF (KLU, sparse jac)" "KenCarp4 (KLU, sparse jac)"]
colors = [:green :deepskyblue1 :dodgerblue2 :royalblue2 :slateblue3 :lightskyblue]
markershapes = [:octagon :hexagon :rtriangle :pentagon :ltriangle :star5]
plot(wp; label = names, left_margin = 10Plots.mm, right_margin = 10Plots.mm,
    xticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], yticks = [1e0, 1e1, 1e2, 1e3],
    color = colors, markershape = markershapes, legendfontsize = 15,
    tickfontsize = 15, guidefontsize = 15, legend = :topright, lw = 20,
    la = 0.8, markersize = 20, markerstrokealpha = 1.0, markerstrokewidth = 1.5,
    gridalpha = 0.3, gridlinewidth = 7.5, size = (1100, 1000))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

