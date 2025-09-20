
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools,
      LinearSolve, RecursiveFactorization

gr()
const to = TimerOutput()
tf = 150.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Models/fceri_gamma2.net"))
show(to)
rn = complete(prnbng.rn)
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = complete(convert(ODESystem, rn))
show(to)

tspan = (0.0, tf)
@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[], jac = true, sparse = true)
show(to)
@timeit to "ODEProb No Jac" oprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[])
show(to)
oprob_sparse = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[]; sparse = true);


@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Apprx. number of terms in the ODE
@show length(parameters(rn)); # Number of Parameters


u = oprob.u0
du = copy(u)
p = oprob.p
@timeit to "ODE rhs Eval1" sparsejacprob.f(du, u, p, 0.0)
sparsejacprob.f(du, u, p, 0.0)


@btime sparsejacprob.f($du, $u, $p, 0.0)


sol = solve(oprob, CVODE_BDF(), saveat = tf/1000.0, reltol = 1e-5, abstol = 1e-5)
plot(sol; idxs = obs, legend = false, fmt = :png)


@time sol = solve(sparsejacprob, CVODE_BDF(linear_solver = :GMRES), reltol = 1e-15, abstol = 1e-15)
test_sol = TestSolution(sol);


default(legendfontsize = 7, framestyle = :box, gridalpha = 0.3, gridlinewidth = 2.5)


using IncompleteLU, LinearAlgebra

jaccache = sparsejacprob.f.jac(oprob.u0, oprob.p, 0.0)
W = I - 1.0*jaccache
prectmp = ilu(W, τ = 50.0)
preccache = Ref(prectmp)

const τ1 = 5
function psetupilu(p, t, u, du, jok, jcurPtr, gamma)
    if !jok
        sparsejacprob.f.jac(jaccache, u, p, t)
        jcurPtr[] = true

        # W = I - gamma*J
        @. W = -gamma*jaccache
        idxs = diagind(W)
        @. @view(W[idxs]) = @view(W[idxs]) + 1

        # Build preconditioner on W
        preccache[] = ilu(W, τ = τ1)
    end
end
function precilu(z, r, p, t, y, fy, gamma, delta, lr)
    ldiv!(z, preccache[], r)
end

const τ2 = 5
function incompletelu(W, du, u, p, t, newW, Plprev, Prprev, solverdata)
    if newW === nothing || newW
        Pl = ilu(convert(AbstractMatrix, W), τ = τ2)
    else
        Pl = Plprev
    end
    Pl, nothing
end;


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (5:8);


setups = [
    Dict(:alg=>lsoda(), :prob_choice => 1),
    Dict(:alg=>CVODE_BDF(), :prob_choice => 1),
    Dict(:alg=>CVODE_BDF(linear_solver = :LapackDense), :prob_choice => 1),
    Dict(:alg=>CVODE_BDF(linear_solver = :GMRES), :prob_choice => 1),
    Dict(
        :alg=>CVODE_BDF(linear_solver = :GMRES, prec = precilu, psetup = psetupilu, prec_side = 1),
        :prob_choice => 2),
    Dict(:alg=>CVODE_BDF(linear_solver = :KLU), :prob_choice => 3)
];


wp = WorkPrecisionSet(
    [oprob, oprob_sparse, sparsejacprob], abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = [test_sol, test_sol, test_sol], maxiters = Int(1e9), numruns = 10)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (LapackDense)" "CVODE_BDF (GMRES)" "CVODE_BDF (GMRES, iLU)" "CVODE_BDF (KLU, sparse jac)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(autodiff = false)),
    Dict(:alg=>QNDF(autodiff = false)),
    Dict(:alg=>FBDF(autodiff = false)),
    Dict(:alg=>KenCarp4(autodiff = false))
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e12), dtmin = 1e-18, numruns = 10)

names = ["TRBDF2" "QNDF" "FBDF" "KenCarp4"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(linsolve = KrylovJL_GMRES(), autodiff = false)),
    Dict(:alg=>QNDF(linsolve = KrylovJL_GMRES(), autodiff = false)),
    Dict(:alg=>FBDF(linsolve = KrylovJL_GMRES(), autodiff = false)),
    Dict(:alg=>KenCarp4(linsolve = KrylovJL_GMRES(), autodiff = false))
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e12), dtmin = 1e-18, numruns = 10)

names = ["TRBDF2 (GMRES)" "QNDF (GMRES)" "FBDF (GMRES)" "KenCarp4 (GMRES)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(
        linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true)),
    Dict(:alg=>QNDF(linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true)),
    Dict(:alg=>FBDF(linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true)),
    Dict(:alg=>KenCarp4(
        linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true))
];


wp = WorkPrecisionSet(sparsejacprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e12), dtmin = 1e-18, numruns = 10)

names = ["TRBDF2 (GMRES, iLU)" "QNDF (GMRES, iLU)" "FBDF (GMRES, iLU)" "KenCarp4 (GMRES, iLU)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2(linsolve = KLUFactorization(), autodiff = false)),
    Dict(:alg=>QNDF(linsolve = KLUFactorization(), autodiff = false)),
    Dict(:alg=>FBDF(linsolve = KLUFactorization(), autodiff = false)),
    Dict(:alg=>KenCarp4(linsolve = KLUFactorization(), autodiff = false))
];


wp = WorkPrecisionSet(sparsejacprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e12), dtmin = 1e-18, numruns = 10)

names = ["TRBDF2 (KLU, sparse jac)" "QNDF (KLU, sparse jac)" "FBDF (KLU, sparse jac)" "KenCarp4 (KLU, sparse jac)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>CVODE_Adams()),
    Dict(:alg=>Tsit5()),
    Dict(:alg=>BS5()),
    Dict(:alg=>VCABM()),
    Dict(:alg=>Vern6()),
    Dict(:alg=>Vern7()),
    Dict(:alg=>Vern8()),
    Dict(:alg=>Vern9())
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e9), numruns = 10)

names = ["CVODE_Adams" "Tsit5" "BS5" "Vern6" "Vern7" "Vern8" "Vern9"]
plot(wp; label = names)


setups = [
    Dict(:alg=>CVODE_BDF(linear_solver = :GMRES), :prob_choice => 1),
    Dict(
        :alg=>CVODE_BDF(linear_solver = :GMRES, prec = precilu, psetup = psetupilu, prec_side = 1),
        :prob_choice => 2),
    Dict(
        :alg=>QNDF(linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true),
        :prob_choice => 3),
    Dict(
        :alg=>FBDF(linsolve = KrylovJL_GMRES(), autodiff = false, precs = incompletelu, concrete_jac = true),
        :prob_choice => 3),
    Dict(:alg=>Tsit5())
];


wp = WorkPrecisionSet(
    [oprob, oprob_sparse, sparsejacprob], abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = [test_sol, test_sol, test_sol], maxiters = Int(1e9), numruns = 200)

names = ["CVODE_BDF (GMRES)" "CVODE_BDF (GMRES, iLU)" "QNDF (GMRES, iLU)" "FBDF (GMRES, iLU)" "Tsit5"]
colors = [:darkgreen :green :deepskyblue1 :dodgerblue2 :orchid2]
markershapes = [:rect :octagon :hexagon :rtriangle :ltriangle]
plot(wp; label = names, left_margin = 10Plots.mm, right_margin = 10Plots.mm,
    xticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    yticks = [1e-1, 1e0, 1e1, 1e2], color = colors, markershape = markershapes,
    legendfontsize = 15, tickfontsize = 15, guidefontsize = 15, legend = :topright,
    lw = 20, la = 0.8, markersize = 20, markerstrokealpha = 1.0,
    markerstrokewidth = 1.5, gridalpha = 0.3, gridlinewidth = 7.5, size = (1100, 1000))


echo = false
using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

