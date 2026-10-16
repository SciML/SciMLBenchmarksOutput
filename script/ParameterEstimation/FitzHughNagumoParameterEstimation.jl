
using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim, Optimization
using OptimizationBBO, OptimizationNLopt, ForwardDiff, Plots, BenchmarkTools
using ModelingToolkit
using ModelingToolkitBase
using SciCompDSL
using ModelingToolkit: @mtkbuild, D_nounits as D, t_nounits as t
gr(fmt = :png)


loc_bounds = Tuple{Float64, Float64}[(0, 1), (0, 1), (
    0, 1), (0, 1)]
glo_bounds = Tuple{Float64, Float64}[(0, 5), (0, 5), (
    0, 5), (0, 5)]
loc_init = [0.5, 0.5, 0.5, 0.5]
glo_init = [2.5, 2.5, 2.5, 2.5]


@mtkmodel FitzHughNagumo begin
    @parameters begin
        a = 0.7      # Parameter for excitability
        b = 0.8      # Recovery rate parameter
        τinv = 0.08  # Inverse of the time constant
        l = 0.5      # External stimulus
    end
    @variables begin
        v(t) = 1.0  # Membrane potential with initial condition
        w(t) = 1.0  # Recovery variable with initial condition
    end
    @equations begin
        D(v) ~ v - v^3 / 3 - w + l
        D(w) ~ τinv * (v + a - b * w)
    end
end

@mtkbuild fitz = FitzHughNagumo()


p = [0.7, 0.8, 0.08, 0.5]              # Parameters used to construct the dataset
r0 = [1.0; 1.0]                     # initial value
tspan = (0.0, 30.0)                 # sample of 3000 observations over the (0,30) timespan
# unknowns(fitz) is ordered [w, v] rather than in declaration order, so the
# initial condition and parameters are keyed by name instead of positionally.
varmap = merge(Dict([fitz.v, fitz.w] .=> r0), Dict([fitz.a, fitz.b, fitz.τinv, fitz.l] .=> p))
prob = ODEProblem(fitz, varmap, tspan)
tspan2 = (0.0, 3.0)                 # sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(fitz, varmap, tspan2)


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
data_sol_short = solve(prob_short, Vern9(), saveat = t_short, reltol = 1e-9, abstol = 1e-9)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob, Vern9(), saveat = time_points, reltol = 1e-9, abstol = 1e-9)
data = convert(Array, data_sol)


plot(data_sol_short)


plot(data_sol)


obj_short = build_loss_objective(prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))
short_solutions = Dict{String, Any}()
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Tsit5 default tol)"] = res1


obj_short = build_loss_objective(
    prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short, reltol = 1e-9)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Tsit5 reltol=1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))
res1 = @btime solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
short_solutions["BBO (Vern9 1e-9)"] = res1


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    tstops = t_short, reltol = 1e-9, abstol = 1e-9)


opt = Opt(:GN_ORIG_DIRECT_L, 4)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))
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


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    Optimization.AutoForwardDiff(), tstops = t_short, reltol = 1e-9, abstol = 1e-9)
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


opt = Opt(:LN_NEWUOA_BOUND, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
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


obj = build_loss_objective(
    prob, Vern9(), L2Loss(time_points, data), Optimization.AutoForwardDiff(),
    tstops = time_points, reltol = 1e-9, abstol = 1e-9)
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


optprob = OptimizationProblem(obj, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))


opt = Opt(:LN_BOBYQA, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
long_solutions["LN_BOBYQA"] = res1


opt = Opt(:LN_NELDERMEAD, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-9)
long_solutions["LN_NELDERMEAD"] = res1


opt = Opt(:LD_SLSQP, 4)
res1 = @btime solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)
long_solutions["LD_SLSQP"] = res1


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


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

