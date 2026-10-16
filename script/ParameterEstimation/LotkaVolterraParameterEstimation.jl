
using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim, Optimization, ForwardDiff
using OptimizationBBO, OptimizationNLopt, Plots, RecursiveArrayTools, BenchmarkTools
using ModelingToolkit
using ModelingToolkitBase
using SciCompDSL
using ModelingToolkit: @mtkbuild, D_nounits as D, t_nounits as t
gr(fmt = :png)


loc_bounds = Tuple{Float64, Float64}[(0, 5), (0, 5), (
    0, 5), (0, 5)]
glo_bounds = Tuple{Float64, Float64}[(0, 10), (0, 10), (
    0, 10), (0, 10)]
loc_init = [1, 0.5, 3.5, 1.5]
glo_init = [5.0, 5.0, 5.0, 5.0]


@mtkmodel LotkaVolterraTest begin
    @parameters begin
        a = 1.5  # Growth rate of prey
        b = 1.0  # Predation rate
        c = 3.0  # Death rate of predators
        d = 1.0  # Reproduction rate of predators
    end
    @variables begin
        x(t) = 1.0  # Population of prey with initial condition
        y(t) = 1.0  # Population of predators with initial condition
    end
    @equations begin
        D(x) ~ a * x - b * x * y
        D(y) ~ -c * y + d * x * y
    end
end

@mtkbuild f = LotkaVolterraTest()


u0 = [1.0, 1.0]                          #initial values
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]                   #parameters used, these need to be estimated from the data
tspan = (0.0, 30.0)                     # sample of 3000 observations over the (0,30) timespan
# unknowns(f) is ordered [y, x] rather than in declaration order, so the
# initial condition and parameters are keyed by name instead of positionally.
varmap = merge(Dict([f.x, f.y] .=> u0), Dict([f.a, f.b, f.c, f.d] .=> p))
prob = ODEProblem(f, varmap, tspan)
tspan2 = (0.0, 3.0)                     # sample of 300 observations over the (0,3) timespan
prob_short = ODEProblem(f, varmap, tspan2)


dt = 30.0/3000
tf = 30.0
tinterval = 0:dt:tf
time_points = collect(tinterval)


h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)


#Generate Data
data_sol_short = solve(prob_short, Tsit5(), saveat = t_short, reltol = 1e-9, abstol = 1e-9)
data_short = convert(Array, data_sol_short)
data_sol = solve(prob, Tsit5(), saveat = time_points, reltol = 1e-9, abstol = 1e-9)
data = convert(Array, data_sol)


p1 = plot(data_sol_short)


p2 = plot(data_sol)


obj_short = build_loss_objective(prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))
short_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
short_solutions["BBO (Tsit5 default tol)"] = res1


obj_short = build_loss_objective(
    prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short, reltol = 1e-9)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
short_solutions["BBO (Tsit5 reltol=1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
short_solutions["BBO (Vern9 1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    Optimization.AutoForwardDiff(), tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))


opt = Opt(:GN_ORIG_DIRECT_L, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ORIG_DIRECT_L"] = res1


opt = Opt(:GN_CRS2_LM, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_CRS2_LM"] = res1


opt = Opt(:GN_ISRES, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ISRES"] = res1


opt = Opt(:GN_ESCH, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ESCH"] = res1


optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))


opt = Opt(:LN_BOBYQA, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_BOBYQA"] = res1


opt = Opt(:LN_NELDERMEAD, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_NELDERMEAD"] = res1


opt = Opt(:LD_SLSQP, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_SLSQP"] = res1


opt = Opt(:LN_COBYLA, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_COBYLA"] = res1


# LN_NEWUOA_BOUND is pathologically slow on this objective with NLopt_jll 2.11
# (https://github.com/jump-dev/NLopt.jl/issues/280); it is given a 300 s
# wall-time budget so the page terminates.
opt = Opt(:LN_NEWUOA_BOUND, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12, maxtime = 300)
short_solutions["LN_NEWUOA_BOUND"] = res1


opt = Opt(:LN_PRAXIS, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_PRAXIS"] = res1


opt = Opt(:LN_SBPLX, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_SBPLX"] = res1


opt = Opt(:LD_MMA, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_MMA"] = res1


opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_TNEWTON_PRECOND_RESTART"] = res1


for (name, sol) in sort!(collect(short_solutions); by = first)
    println(rpad(name, 28),
        "estimate=", round.(sol.u; digits = 3),
        "  retcode=", sol.retcode,
        "  loss=", sol.objective,
        "  max|est-true|=", round(maximum(abs.(sol.u .- p)); sigdigits = 3))
end
best_loss, best_loss_name = minimum((sol.objective, name) for (name, sol) in short_solutions)
best_dist, best_dist_name = minimum((maximum(abs.(sol.u .- p)), name) for (name, sol) in short_solutions)
println("lowest loss: ", best_loss_name, " (loss = ", best_loss, ")")
println("closest to the true parameters: ", best_dist_name, " (max|est-true| = ", best_dist, ")")


t_concrete = collect(0.0:dt:tf)
obj = build_loss_objective(prob, Vern9(), L2Loss(t_concrete, data),
    tstops = t_concrete, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))


long_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 4e3)
long_solutions["BBO_adaptive_de_rand_1_bin"] = res1


opt = Opt(:GN_ORIG_DIRECT_L, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
long_solutions["GN_ORIG_DIRECT_L"] = res1


opt = Opt(:GN_CRS2_LM, 4)
res1 = @btime solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12)
long_solutions["GN_CRS2_LM"] = res1


opt = Opt(:GN_ISRES, 4)
res1 = @btime solve(optprob, opt, maxiters = 50000, xtol_rel = 1e-12)
long_solutions["GN_ISRES"] = res1


opt = Opt(:GN_ESCH, 4)
res1 = @btime solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12)
long_solutions["GN_ESCH"] = res1


for (name, sol) in sort!(collect(long_solutions); by = first)
    println(rpad(name, 28),
        "estimate=", round.(sol.u; digits = 3),
        "  retcode=", sol.retcode,
        "  loss=", sol.objective,
        "  max|est-true|=", round(maximum(abs.(sol.u .- p)); sigdigits = 3))
end
best_loss, best_loss_name = minimum((sol.objective, name) for (name, sol) in long_solutions)
best_dist, best_dist_name = minimum((maximum(abs.(sol.u .- p)), name) for (name, sol) in long_solutions)
println("lowest loss: ", best_loss_name, " (loss = ", best_loss, ")")
println("closest to the true parameters: ", best_dist_name, " (max|est-true| = ", best_dist, ")")


obj = build_loss_objective(prob, Vern9(), L2Loss(time_points, data), Optimization.AutoForwardDiff(),
    tstops = time_points, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))


opt = Opt(:LN_BOBYQA, 4)
local_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12, maxtime = 300)
local_solutions["LN_BOBYQA"] = res1


opt = Opt(:LN_NELDERMEAD, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12, maxtime = 300)
local_solutions["LN_NELDERMEAD"] = res1


opt = Opt(:LD_SLSQP, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12, maxtime = 300)
local_solutions["LD_SLSQP"] = res1


for (name, sol) in sort!(collect(local_solutions); by = first)
    println(rpad(name, 28),
        "estimate=", round.(sol.u; digits = 3),
        "  retcode=", sol.retcode,
        "  loss=", sol.objective,
        "  max|est-true|=", round(maximum(abs.(sol.u .- p)); sigdigits = 3),
        "  strictly inside loc_bounds: ", all(first.(loc_bounds) .< sol.u .< last.(loc_bounds)))
end
best_loss, best_loss_name = minimum((sol.objective, name) for (name, sol) in local_solutions)
best_dist, best_dist_name = minimum((maximum(abs.(sol.u .- p)), name) for (name, sol) in local_solutions)
println("lowest loss: ", best_loss_name, " (loss = ", best_loss, ")")
println("closest to the true parameters: ", best_dist_name, " (max|est-true| = ", best_dist, ")")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

