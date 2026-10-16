
using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim, Optimization
using OptimizationBBO, OptimizationNLopt, Plots, ForwardDiff, BenchmarkTools
using ModelingToolkit
using ModelingToolkitBase
using SciCompDSL
using ModelingToolkit: @mtkbuild, D_nounits as D, t_nounits as t
gr(fmt = :png)


Xiang2015Bounds = Tuple{Float64, Float64}[(9, 11), (20, 30), (
    2, 3)] # for local optimizations
xlow_bounds = [9.0, 20.0, 2.0]
xhigh_bounds = [11.0, 30.0, 3.0]
LooserBounds = Tuple{Float64, Float64}[(0, 22), (0, 60), (
    0, 6)] # for global optimization
GloIniPar = [0.0, 0.5, 0.1] # for global optimizations
LocIniPar = [9.0, 20.0, 2.0] # for local optimization


@mtkmodel LorenzExample begin
    @parameters begin
        σ = 10.0  # Parameter: Prandtl number
        ρ = 28.0  # Parameter: Rayleigh number
        β = 8/3   # Parameter: Geometric factor
    end
    @variables begin
        x(t) = 1.0  # Initial condition for x
        y(t) = 1.0  # Initial condition for y
        z(t) = 1.0  # Initial condition for z
    end
    @equations begin
        D(x) ~ σ * (y - x)
        D(y) ~ x * (ρ - z) - y
        D(z) ~ x * y - β * z
    end
end

@mtkbuild g1 = LorenzExample()
p = [10.0, 28.0, 2.66] # Parameters used to construct the dataset
r0 = [1.0; 0.0; 0.0]                #[-11.8,-5.1,37.5] PODES Initial values of the system in space # [0.1, 0.0, 0.0]
# unknowns(g1) is ordered [z, y, x] rather than in declaration order, so the
# initial condition and parameters are keyed by name instead of positionally.
varmap = merge(Dict([g1.x, g1.y, g1.z] .=> r0), Dict([g1.σ, g1.ρ, g1.β] .=> p))
tspan = (0.0, 30.0)                 # PODES sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(g1, varmap, tspan)
tspan2 = (0.0, 3.0)                 # Xiang test sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(g1, varmap, tspan2)


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


# Generate Data
data_sol_short = solve(prob_short, Vern9(), saveat = t_short, reltol = 1e-9, abstol = 1e-9)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob, Vern9(), saveat = time_points, reltol = 1e-9, abstol = 1e-9)
data = convert(Array, data_sol)


plot(data_sol_short, vars = (1, 2, 3)) # the short solution
plot(data_sol, vars = (1, 2, 3)) # the longer solution
interpolation_sol = solve(prob, Vern7(), saveat = time_points, reltol = 1e-12, abstol = 1e-12)
plot(interpolation_sol, vars = (1, 2, 3))


xyzt = plot(data_sol_short, plotdensity = 10000, lw = 1.5)
xy = plot(data_sol_short, plotdensity = 10000, vars = (1, 2))
xz = plot(data_sol_short, plotdensity = 10000, vars = (1, 3))
yz = plot(data_sol_short, plotdensity = 10000, vars = (2, 3))
xyz = plot(data_sol_short, plotdensity = 10000, vars = (1, 2, 3))
plot(plot(xyzt, xyz), plot(xy, xz, yz, layout = (1, 3), w = 1), layout = (2, 1), size = (
    800, 600))


xyzt = plot(data_sol, plotdensity = 10000, lw = 1.5)
xy = plot(data_sol, plotdensity = 10000, vars = (1, 2))
xz = plot(data_sol, plotdensity = 10000, vars = (1, 3))
yz = plot(data_sol, plotdensity = 10000, vars = (2, 3))
xyz = plot(data_sol, plotdensity = 10000, vars = (1, 2, 3))
plot(plot(xyzt, xyz), plot(xy, xz, yz, layout = (1, 3), w = 1), layout = (2, 1), size = (
    800, 600))


obj_short = build_loss_objective(prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
short_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Tsit5 default tol)"] = res1


obj_short = build_loss_objective(
    prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short, reltol = 1e-9)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Tsit5 reltol=1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Vern9 1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    Optimization.AutoForwardDiff(), tstops = t_short, reltol = 1e-9, abstol = 1e-9)


opt = Opt(:GN_ORIG_DIRECT_L, 3)
optprob = OptimizationProblem(obj_short, GloIniPar, lb = first.(LooserBounds), ub = last.(LooserBounds))
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ORIG_DIRECT_L"] = res1


opt = Opt(:GN_CRS2_LM, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_CRS2_LM"] = res1


opt = Opt(:GN_ISRES, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ISRES"] = res1


opt = Opt(:GN_ESCH, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["GN_ESCH"] = res1


opt = Opt(:LN_BOBYQA, 3)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_BOBYQA"] = res1


opt = Opt(:LN_NELDERMEAD, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_NELDERMEAD"] = res1


opt = Opt(:LD_SLSQP, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_SLSQP"] = res1


opt = Opt(:LN_COBYLA, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_COBYLA"] = res1


opt = Opt(:LN_NEWUOA_BOUND, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_NEWUOA_BOUND"] = res1


opt = Opt(:LN_PRAXIS, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_PRAXIS"] = res1


opt = Opt(:LN_SBPLX, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LN_SBPLX"] = res1


opt = Opt(:LD_MMA, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_MMA"] = res1


opt = Opt(:LD_LBFGS, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
short_solutions["LD_LBFGS"] = res1


opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 3)
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


# BBO is given a smaller iteration budget on the long problem.
obj = build_loss_objective(prob, Vern9(), L2Loss(time_points, data),
    Optimization.AutoForwardDiff(), tstops = time_points, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj, GloIniPar, lb = first.(LooserBounds), ub = last.(LooserBounds))
global_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 4e3)
global_solutions["BBO_adaptive_de_rand_1_bin"] = res1
#res1 = @btime bboptimize(obj;SearchRange = LooserBounds, Method = :adaptive_de_rand_1_bin, MaxSteps = 4e3)
#res1 = @btime bboptimize(obj;SearchRange = LooserBounds, Method = :dxnes, MaxSteps = 2e3)
#res1 = @btime bboptimize(obj;SearchRange = LooserBounds, Method = :xnes, MaxSteps = 2e3)
#res1 = @btime bboptimize(obj;SearchRange = LooserBounds, Method = :de_rand_1_bin_radiuslimited, MaxSteps = 2e3)
#res1 = @btime bboptimize(obj;SearchRange = LooserBounds, Method = :generating_set_search, MaxSteps = 2e3)


# using Evolutionary
# N = 3
# @time result, fitness, cnt = cmaes(obj, N; μ = 3, λ = 12, iterations = 1000) # cmaes( rastrigin, N; μ = 15, λ = P, tol = 1e-8)


opt = Opt(:GN_ORIG_DIRECT_L, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12, maxtime = 300)
global_solutions["GN_ORIG_DIRECT_L"] = res1


opt = Opt(:GN_CRS2_LM, 3)
res1 = @btime solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12, maxtime = 300)
global_solutions["GN_CRS2_LM"] = res1


opt = Opt(:GN_ISRES, 3)
res1 = @btime solve(optprob, opt, maxiters = 50000, xtol_rel = 1e-12, maxtime = 300)
global_solutions["GN_ISRES"] = res1


opt = Opt(:GN_ESCH, 3)
res1 = @btime solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12, maxtime = 300)
global_solutions["GN_ESCH"] = res1


for (name, sol) in sort!(collect(global_solutions); by = first)
    println(rpad(name, 28),
        "estimate=", round.(sol.u; digits = 3),
        "  retcode=", sol.retcode,
        "  loss=", sol.objective,
        "  max|est-true|=", round(maximum(abs.(sol.u .- p)); sigdigits = 3),
        "  within LooserBounds: ", all(first.(LooserBounds) .<= sol.u .<= last.(LooserBounds)),
        "  within Xiang2015Bounds: ", all(xlow_bounds .<= sol.u .<= xhigh_bounds))
end
best_loss, best_loss_name = minimum((sol.objective, name) for (name, sol) in global_solutions)
best_dist, best_dist_name = minimum((maximum(abs.(sol.u .- p)), name) for (name, sol) in global_solutions)
println("lowest loss: ", best_loss_name, " (loss = ", best_loss, ")")
println("closest to the true parameters: ", best_dist_name, " (max|est-true| = ", best_dist, ")")
println("loss at the true parameters ", p, ": ", obj(p))
println("loss with σ shifted by +1e-6: ", obj([10.0 + 1e-6, 28.0, 2.66]))
println("loss with σ shifted by +1e-4: ", obj([10.0 + 1e-4, 28.0, 2.66]))


opt = Opt(:LN_BOBYQA, 3)
optprob = OptimizationProblem(obj, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
local_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
local_solutions["LN_BOBYQA"] = res1


opt = Opt(:LN_NELDERMEAD, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
local_solutions["LN_NELDERMEAD"] = res1


opt = Opt(:LD_SLSQP, 3)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
local_solutions["LD_SLSQP"] = res1


for (name, sol) in sort!(collect(local_solutions); by = first)
    println(rpad(name, 28),
        "estimate=", round.(sol.u; digits = 3),
        "  retcode=", sol.retcode,
        "  loss=", sol.objective,
        "  max|est-true|=", round(maximum(abs.(sol.u .- p)); sigdigits = 3),
        "  strictly inside Xiang2015Bounds: ", all(xlow_bounds .< sol.u .< xhigh_bounds))
end
best_loss, best_loss_name = minimum((sol.objective, name) for (name, sol) in local_solutions)
best_dist, best_dist_name = minimum((maximum(abs.(sol.u .- p)), name) for (name, sol) in local_solutions)
println("lowest loss: ", best_loss_name, " (loss = ", best_loss, ")")
println("closest to the true parameters: ", best_dist_name, " (max|est-true| = ", best_dist, ")")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

