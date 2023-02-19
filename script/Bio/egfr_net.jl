
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools,
      LinearSolve

gr()
const to = TimerOutput()
tf       = 10.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Models/egfr_net.net"))
show(to) 
rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = convert(ODESystem, rn)
show(to) 

tspan = (0.,tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem{true, SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[])
show(to);


@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem{true, SciMLBase.FullSpecialize}(osys, Float64[], tspan, Float64[], jac=true, sparse=true)
show(to)


@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Apprx. number of terms in the ODE
@show length(parameters(rn)); # Number of Parameters


u  = ModelingToolkit.varmap_to_vars(nothing, species(rn); defaults=ModelingToolkit.defaults(rn))
du = copy(u)
p  = ModelingToolkit.varmap_to_vars(nothing, parameters(rn); defaults=ModelingToolkit.defaults(rn))
@timeit to "ODE rhs Eval1" oprob.f(du,u,p,0.)
@timeit to "ODE rhs Eval2" oprob.f(du,u,p,0.)
sparsejacprob.f(du,u,p,0.)


@btime oprob.f($du,$u,$p,0.)


sol = solve(oprob, CVODE_BDF(), saveat=tf/1000., reltol=1e-5, abstol=1e-5)
plot(sol; idxs=obs, legend=false, fmt=:png)


@time sol = solve(oprob, CVODE_BDF(), abstol=1/10^14, reltol=1/10^14)
test_sol  = TestSolution(sol);


default(legendfontsize=7,framestyle=:box,gridalpha=0.3,gridlinewidth=2.5)


function plot_settings(wp)
    times = vcat(map(wp -> wp.times, wp.wps)...)
    errors = vcat(map(wp -> wp.errors, wp.wps)...)
    xlimit = 10 .^ (floor(log10(minimum(errors))), ceil(log10(maximum(errors))))
    ylimit = 10 .^ (floor(log10(minimum(times))), ceil(log10(maximum(times))))
    return xlimit,ylimit
end


abstols = 1.0 ./ 10.0 .^ (6:10)
reltols = 1.0 ./ 10.0 .^ (6:10);


setups = [
          Dict(:alg=>lsoda()),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>CVODE_BDF(linear_solver=:LapackDense)),
          Dict(:alg=>CVODE_Adams()),  
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>FBDF()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P())
          ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=100)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (Lapack Dense)" "CVODE_Adams" "TRBDF2" "QNDF" "FBDF" "KenCarp4" "Rosenbrock23" "Rodas4" "Rodas5P"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
          Dict(:alg=>lsoda()),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES)),
          Dict(:alg=>TRBDF2(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>QNDF(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>FBDF(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>KenCarp4(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>Rosenbrock23(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>Rodas4(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>Rodas5P(linsolve=KrylovJL_GMRES()))
          ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=100)

names = ["lsoda" "CVODE_BDF (GMRES)" "TRBDF2 (GMRES)" "QNDF (GMRES)" "FBDF (GMRES)" "KenCarp4 (GMRES)" "Rosenbrock23 (GMRES)" "Rodas4 (GMRES)" "Rodas5P (GMRES)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
          Dict(:alg=>CVODE_BDF(linear_solver=:KLU)),
          Dict(:alg=>TRBDF2(linsolve=KLUFactorization())),
          Dict(:alg=>QNDF(linsolve=KLUFactorization())),
          Dict(:alg=>FBDF(linsolve=KLUFactorization())),
          Dict(:alg=>KenCarp4(linsolve=KLUFactorization())),
          Dict(:alg=>Rosenbrock23(linsolve=KLUFactorization())),
          Dict(:alg=>Rodas4(linsolve=KLUFactorization())),
          Dict(:alg=>Rodas5P(linsolve=KLUFactorization()))
          ];


wp = WorkPrecisionSet(sparsejacprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=100)

names = ["CVODE_BDF (KLU, sparse jac)" "TRBDF2 (KLU, sparse jac)" "QNDF (KLU, sparse jac)" "FBDF (KLU, sparse jac)" "KenCarp4 (KLU, sparse jac)" "Rosenbrock23 (KLU, sparse jac)" "Rodas4 (KLU, sparse jac)" "Rodas5P (KLU, sparse jac)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
          Dict(:alg=>lsoda()), 
          Dict(:alg=>CVODE_Adams()), 
          Dict(:alg=>Tsit5()), 
          Dict(:alg=>BS5()), 
          Dict(:alg=>VCABM()),          
          Dict(:alg=>Vern6()), 
          Dict(:alg=>Vern7()), 
          Dict(:alg=>Vern8()), 
          Dict(:alg=>Vern9()),
          Dict(:alg=>ROCK4())
          ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=200)

names = ["lsoda" "CVODE_Adams" "Tsit5" "BS5" "VCABM" "Vern6" "Vern7" "Vern8" "Vern9" "ROCK4"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [Dict(:alg=>ROCK2())];
wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=200)
names = ["ROCK2"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
          Dict(:alg=>lsoda(), :prob_choice => 1),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES), :prob_choice => 1),
          Dict(:alg=>QNDF(linsolve=KrylovJL_GMRES()), :prob_choice => 2),

          Dict(:alg=>QNDF(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>BS5(), :prob_choice => 1),     
          Dict(:alg=>Vern6(), :prob_choice => 1), 
          Dict(:alg=>ROCK4(), :prob_choice => 1)
          ];


wp = WorkPrecisionSet([oprob,sparsejacprob],abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=[test_sol,test_sol],maxiters=Int(1e9),numruns=200)

names = ["lsoda" "CVODE_BDF (GMRES)" "QNDF (GMRES)" "BS5" "Vern6" "ROCK4"]
colors = [:seagreen1 :darkgreen :deepskyblue1 :thistle2 :lightslateblue :purple4]
markershapes = [:star4, :rect, :hexagon, :star8, :rtriangle, :square]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit,xticks=[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3],yticks=[],color=colors,markershape=markershapes,legendfontsize=15,tickfontsize=15,guidefontsize=15, legend=:topright, lw=20, la=0.8, markersize=20,markerstrokealpha=1.0, markerstrokewidth=1.5, gridalpha=0.3, gridlinewidth=7.5,size=(1100,1000))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

