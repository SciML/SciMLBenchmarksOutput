
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools,
      RecursiveFactorization

gr()
const to = TimerOutput()
tf = 20.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Models/multistate.net"))
show(to)
rn = complete(prnbng.rn)
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = complete(convert(ODESystem, rn))
show(to)

tspan = (0.0, tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[])
show(to);


@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    osys, Float64[], tspan, Float64[], jac = true, sparse = true)
show(to)


@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Apprx. number of terms in the ODE
@show length(parameters(rn)); # Number of Parameters


u = oprob.u0
du = copy(u)
p = oprob.p
@timeit to "ODE rhs Eval1" oprob.f(du, u, p, 0.0)
@timeit to "ODE rhs Eval2" oprob.f(du, u, p, 0.0)
sparsejacprob.f(du, u, p, 0.0)


@btime oprob.f($du, $u, $p, 0.0)


sol = solve(oprob, CVODE_BDF(), saveat = tf/1000.0, reltol = 1e-5, abstol = 1e-5)
plot(sol; idxs = obs, legend = false, fmt = :png)


@time sol = solve(oprob, CVODE_BDF(), reltol = 1e-15, abstol = 1e-15)
test_sol = TestSolution(sol);


default(legendfontsize = 7, framestyle = :box, gridalpha = 0.3, gridlinewidth = 2.5)


abstols = 1.0 ./ 10.0 .^ (6:10)
reltols = 1.0 ./ 10.0 .^ (6:10);


setups = [
    Dict(:alg=>lsoda()),
    Dict(:alg=>CVODE_BDF()),
    Dict(:alg=>CVODE_BDF(linear_solver = :LapackDense)),
    Dict(:alg=>CVODE_BDF(linear_solver = :GMRES))
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e9), numruns = 200)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (LapackDense)" "CVODE_BDF (GMRES)"]
plot(wp; label = names)


setups = [
    Dict(:alg=>TRBDF2()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>FBDF()),
    Dict(:alg=>KenCarp4()),
    Dict(:alg=>Rosenbrock23()),
    Dict(:alg=>Rodas4()),
    Dict(:alg=>Rodas5P())
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e12), dtmin = 1e-18, numruns = 200)

names = ["TRBDF2" "QNDF" "FBDF" "KenCarp4" "Rosenbrock23" "Rodas4" "Rodas5P"]
plot(wp; label = names)


setups = [
    Dict(:alg=>lsoda()),
    Dict(:alg=>Tsit5()),
    Dict(:alg=>BS5()),
    Dict(:alg=>VCABM()),
    Dict(:alg=>Vern6()),
    Dict(:alg=>Vern7()),
    Dict(:alg=>Vern8()),
    Dict(:alg=>Vern9()),
    Dict(:alg=>ROCK4())
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e9), numruns = 200)

names = ["lsoda" "Tsit5" "BS5" "VCABM" "Vern6" "Vern7" "Vern8" "Vern9" "ROCK4"]
plot(wp; label = names)


setups = [Dict(:alg=>CVODE_Adams()), Dict(:alg=>ROCK2())];
wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e9), numruns = 200)
names = ["CVODE_Adams" "ROCK2"]
plot(wp; label = names)


setups = [
    Dict(:alg=>lsoda()),
    Dict(:alg=>CVODE_BDF()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>KenCarp4()),
    Dict(:alg=>Rodas5P()),
    Dict(:alg=>Tsit5()),
    Dict(:alg=>BS5()),
    Dict(:alg=>VCABM()),
    Dict(:alg=>Vern7())
];


wp = WorkPrecisionSet(oprob, abstols, reltols, setups; error_estimate = :l2,
    saveat = tf/10000.0, appxsol = test_sol, maxiters = Int(1e9), numruns = 200)

names = ["lsoda" "CVODE_BDF" "QNDF" "KenCarp4" "Rodas5P" "Tsit5" "BS5" "VCABM" "Vern7"]
colors = [:seagreen1 :chartreuse1 :deepskyblue1 :lightskyblue :blue :orchid2 :thistle2 :lightsteelblue2 :mediumpurple1]
markershapes = [:star4 :circle :hexagon :star5 :heptagon :ltriangle :star8 :heptagon :star6]
plot(wp; label = names, left_margin = 10Plots.mm, right_margin = 10Plots.mm,
    xticks = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    yticks = [1e-3, 1e-2], color = colors, markershape = markershapes,
    legendfontsize = 15, tickfontsize = 15, guidefontsize = 15, legend = :topright,
    lw = 20, la = 0.8, markersize = 20, markerstrokealpha = 1.0,
    markerstrokewidth = 1.5, gridalpha = 0.3, gridlinewidth = 7.5, size = (1100, 1000))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

