
using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim, Optimization, ForwardDiff
using OptimizationBBO, OptimizationNLopt, Plots, RecursiveArrayTools, BenchmarkTools
using ModelingToolkit: t_nounits as t, D_nounits as D
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
        e = 0.0  # Additional parameter (if needed)
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
p = [1.5, 1.0, 3.0, 1, 0]                   #parameters used, these need to be estimated from the data
tspan = (0.0, 30.0)                     # sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(f, u0, tspan, p)
tspan2 = (0.0, 3.0)                     # sample of 3000 observations over the (0,30) timespan
prob_short = ODEProblem(f, u0, tspan2, p)


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
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
# Lower tolerance could lead to smaller fitness (more accuracy)


obj_short = build_loss_objective(
    prob_short, Tsit5(), L2Loss(t_short, data_short), tstops = t_short, reltol = 1e-9)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
# Change in tolerance makes it worse


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 7e3)
# using the moe accurate Vern9() reduces the fitness marginally and leads to some increase in time taken


obj_short = build_loss_objective(prob_short, Vern9(), L2Loss(t_short, data_short),
    Optimization.AutoForwardDiff(), tstops = t_short, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))


opt = Opt(:GN_ORIG_DIRECT_L, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_CRS2_LM, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_ISRES, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_ESCH, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))


opt = Opt(:LN_BOBYQA, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_NELDERMEAD, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_SLSQP, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_COBYLA, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_NEWUOA_BOUND, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_PRAXIS, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_SBPLX, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_MMA, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


t_concrete = collect(0.0:dt:tf)
obj = build_loss_objective(prob, Vern9(), L2Loss(t_concrete, data),
    tstops = t_concrete, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj, glo_init, lb = first.(glo_bounds), ub = last.(glo_bounds))


@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 4e3)


opt = Opt(:GN_ORIG_DIRECT_L, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_CRS2_LM, 4)
@btime res1 = solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12)


opt = Opt(:GN_ISRES, 4)
@btime res1 = solve(optprob, opt, maxiters = 50000, xtol_rel = 1e-12)


opt = Opt(:GN_ESCH, 4)
@btime res1 = solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12)


obj = build_loss_objective(prob, Vern9(), L2Loss(t, data), Optimization.AutoForwardDiff(),
    tstops = t, reltol = 1e-9, abstol = 1e-9)
optprob = OptimizationProblem(obj_short, loc_init, lb = first.(loc_bounds), ub = last.(loc_bounds))


opt = Opt(:LN_BOBYQA, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_NELDERMEAD, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_SLSQP, 4)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

