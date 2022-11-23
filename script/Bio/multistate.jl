
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools

gr()
const to = TimerOutput()
tf       = 20.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Models/multistate.net"))
show(to) 
rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = convert(ODESystem, rn)
show(to) 

tspan = (0.,tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem(osys, Float64[], tspan, Float64[])
show(to);


@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem(osys, Float64[], tspan, Float64[], jac=true, sparse=true)
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


@time sol = solve(oprob, CVODE_BDF(), reltol=1e-15, abstol=1e-15)
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
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES))
          ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=200)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (LapackDense)" "CVODE_BDF (GMRES)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>FBDF()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P())
          ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e12),dtmin=1e-18,numruns=200)

names = ["TRBDF2" "QNDF" "FBDF" "KenCarp4" "Rosenbrock23" "Rodas4" "Rodas5P"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


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


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=200)

names = ["lsoda" "Tsit5" "BS5" "VCABM" "Vern6" "Vern7" "Vern8" "Vern9" "ROCK4"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [Dict(:alg=>CVODE_Adams()), Dict(:alg=>ROCK2())];
wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e9),numruns=200)
names = ["CVODE_Adams" "ROCK2"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

