
using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim, Optimization
using OptimizationBBO, OptimizationNLopt, Plots, ForwardDiff, BenchmarkTools
using ModelingToolkit: t_nounits as t, D_nounits as D
gr(fmt=:png)


Xiang2015Bounds = Tuple{Float64, Float64}[(9, 11), (20, 30), (2, 3)] # for local optimizations
xlow_bounds = [9.0,20.0,2.0]
xhigh_bounds = [11.0,30.0,3.0]
LooserBounds = Tuple{Float64, Float64}[(0, 22), (0, 60), (0, 6)] # for global optimization
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
p = [10.0,28.0,2.66] # Parameters used to construct the dataset
r0 = [1.0; 0.0; 0.0]                #[-11.8,-5.1,37.5] PODES Initial values of the system in space # [0.1, 0.0, 0.0]
tspan = (0.0, 30.0)                 # PODES sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(g1, r0, tspan,p)
tspan2 = (0.0, 3.0)                 # Xiang test sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(g1, r0, tspan2,p)


dt = 30.0/3000
tf = 30.0
tinterval = 0:dt:tf
time_points  = collect(tinterval)


h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)


# Generate Data
data_sol_short = solve(prob_short,Vern9(),saveat=t_short,reltol=1e-9,abstol=1e-9)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob,Vern9(),saveat=time_points,reltol=1e-9,abstol=1e-9)
data = convert(Array, data_sol)


plot(data_sol_short,vars=(1,2,3)) # the short solution
plot(data_sol,vars=(1,2,3)) # the longer solution
interpolation_sol = solve(prob,Vern7(),saveat=t,reltol=1e-12,abstol=1e-12)
plot(interpolation_sol,vars=(1,2,3))


xyzt = plot(data_sol_short, plotdensity=10000,lw=1.5)
xy = plot(data_sol_short, plotdensity=10000, vars=(1,2))
xz = plot(data_sol_short, plotdensity=10000, vars=(1,3))
yz = plot(data_sol_short, plotdensity=10000, vars=(2,3))
xyz = plot(data_sol_short, plotdensity=10000, vars=(1,2,3))
plot(plot(xyzt,xyz),plot(xy, xz, yz, layout=(1,3),w=1), layout=(2,1), size=(800,600))


xyzt = plot(data_sol, plotdensity=10000,lw=1.5)
xy = plot(data_sol, plotdensity=10000, vars=(1,2))
xz = plot(data_sol, plotdensity=10000, vars=(1,3))
yz = plot(data_sol, plotdensity=10000, vars=(2,3))
xyz = plot(data_sol, plotdensity=10000, vars=(1,2,3))
plot(plot(xyzt,xyz),plot(xy, xz, yz, layout=(1,3),w=1), layout=(2,1), size=(800,600))


obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
# Tolernace is still too high to get close enough


obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short,reltol=1e-9)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
# With the tolerance lower, it achieves the correct solution in 3.5 seconds.


obj_short = build_loss_objective(prob_short,Vern9(),L2Loss(t_short,data_short),tstops=t_short,reltol=1e-9,abstol=1e-9)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 7e3)
# With the more accurate solver Vern9 in the solution of the ODE, the convergence is less efficient!

# Fastest BlackBoxOptim: 3.5 seconds


obj_short = build_loss_objective(prob_short,Vern9(),L2Loss(t_short,data_short),Optimization.AutoForwardDiff(),tstops=t_short,reltol=1e-9,abstol=1e-9)


opt = Opt(:GN_ORIG_DIRECT_L, 3)
optprob = OptimizationProblem(obj_short, GloIniPar, lb = first.(LooserBounds), ub = last.(LooserBounds))
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_CRS2_LM, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_ISRES, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12) # Accurate to single precision 8.2 seconds


opt = Opt(:GN_ESCH, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12) # Approximately accurate, good starting values for local optimization


opt = Opt(:LN_BOBYQA, 3)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_NELDERMEAD, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_SLSQP, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_COBYLA, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_NEWUOA_BOUND, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_PRAXIS, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LN_SBPLX, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_MMA, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_LBFGS, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


# BB with Vern9 converges very slowly. The final values are within the NarrowBounds.
obj = build_loss_objective(prob,Vern9(),L2Loss(time_points,data),tstops=time_points,reltol=1e-9,abstol=1e-9)
optprob = OptimizationProblem(obj, GloIniPar, lb = first.(LooserBounds), ub = last.(LooserBounds))
@btime res1 = solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters = 4e3) # Default adaptive_de_rand_1_bin_radiuslimited 33 sec [10.2183, 24.6711, 2.28969]
#@btime res1 = bboptimize(obj;SearchRange = LooserBounds, Method = :adaptive_de_rand_1_bin, MaxSteps = 4e3) # Method 32 sec [13.2222, 25.8589, 2.56176]
#@btime res1 = bboptimize(obj;SearchRange = LooserBounds, Method = :dxnes, MaxSteps = 2e3) # Method dxnes 119 sec  [16.8648, 24.393, 2.29119]
#@btime res1 = bboptimize(obj;SearchRange = LooserBounds, Method = :xnes, MaxSteps = 2e3) # Method xnes 304 sec  [19.1647, 24.9479, 2.39467]
#@btime res1 = bboptimize(obj;SearchRange = LooserBounds, Method = :de_rand_1_bin_radiuslimited, MaxSteps = 2e3) # Method 44 sec  [13.805, 24.6054, 2.37274]
#@btime res1 = bboptimize(obj;SearchRange = LooserBounds, Method = :generating_set_search, MaxSteps = 2e3) # Method 195 sec [19.1847, 24.9492, 2.39412]


# using Evolutionary
# N = 3
# @time result, fitness, cnt = cmaes(obj, N; μ = 3, λ = 12, iterations = 1000) # cmaes( rastrigin, N; μ = 15, λ = P, tol = 1e-8)


opt = Opt(:GN_ORIG_DIRECT_L, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12)


opt = Opt(:GN_CRS2_LM, 3)
@btime res1 = solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12) # Hit and miss. converge approximately accurate values for local opt.91 seconds


opt = Opt(:GN_ISRES, 3)
@btime res1 = solve(optprob, opt, maxiters = 50000, xtol_rel = 1e-12) # Approximately accurate within local bounds


opt = Opt(:GN_ESCH, 3)
@btime res1 = solve(optprob, opt, maxiters = 20000, xtol_rel = 1e-12) # Approximately accurate


opt = Opt(:LN_BOBYQA, 3)
optprob = OptimizationProblem(obj_short, LocIniPar, lb = xlow_bounds, ub = xhigh_bounds)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12) # Claims SUCCESS but does not iterate to the true values.


opt = Opt(:LN_NELDERMEAD, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12) # Inaccurate final values


opt = Opt(:LD_SLSQP, 3)
@btime res1 = solve(optprob, opt, maxiters = 10000, xtol_rel = 1e-12) # Inaccurate final values


minimum(root)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

