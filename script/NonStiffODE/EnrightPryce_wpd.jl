
using OrdinaryDiffEq, ParameterizedFunctions, ODEInterface,
      ODEInterfaceDiffEq, LSODA, Sundials, DiffEqDevTools,
      StaticArrays, OrdinaryDiffEqSIMDRK
using Plots
gr()

# Load the problems from the Enright-Pryce suite
include(joinpath(@__DIR__, "enright_pryce.jl"))

abstols = 1.0 ./ 10.0 .^ (6:13)
reltols = 1.0 ./ 10.0 .^ (3:10)


simdrkalgs = [
    Dict(:alg=>MER5v2()),
    Dict(:alg=>MER6v2()),
    Dict(:alg=>RK6v4())
]


setups = [
    Dict(:alg=>Tsit5()),
    Dict(:alg=>Vern6()),
    Dict(:alg=>Vern7()),
    Dict(:alg=>Vern9()),
    simdrkalgs...
]


prob = NA_PROBLEMS[1]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NA1")


prob = NA_PROBLEMS[2]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NA2")


prob = NA_PROBLEMS[4]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NA4")


prob = NA_PROBLEMS[5]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NA5")


prob = NB_PROBLEMS[1]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NB1")


prob = NB_PROBLEMS[2]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NB2")


prob = NB_PROBLEMS[3]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NB3")


prob = NB_PROBLEMS[5]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NB5")


prob = NC_PROBLEMS[1]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NC1")


prob = NC_PROBLEMS[2]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NC2")


prob = NC_PROBLEMS[3]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NC3")


prob = NC_PROBLEMS[4]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NC4")


prob = NC_PROBLEMS[5]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="NC5")


prob = ND_PROBLEMS[1]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="ND1")


prob = ND_PROBLEMS[2]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="ND2")


prob = ND_PROBLEMS[3]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="ND3")


prob = ND_PROBLEMS[4]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="ND4")


prob = ND_PROBLEMS[5]
test_sol = solve(prob, Vern9(), abstol=1/10^14, reltol=1/10^14)
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol, save_everystep=false, numruns=100)
plot(wp; title="ND5")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

