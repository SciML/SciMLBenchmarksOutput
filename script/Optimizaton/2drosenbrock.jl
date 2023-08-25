
# Define the problem to solve
using Optimization, ForwardDiff, Zygote, BenchmarkTools

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]

f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
l1 = rosenbrock(x0, _p)
prob = OptimizationProblem(f, x0, _p)


using OptimizationOptimJL


@btime sol = solve(prob, SimulatedAnnealing())


prob = OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
@btime sol = solve(prob, SAMIN())


l1 = rosenbrock(x0, _p)
prob = OptimizationProblem(rosenbrock, x0, _p)
@btime sol = solve(prob, NelderMead())


optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, _p)
@btime sol = solve(prob, BFGS())


@btime sol = solve(prob, Newton())


@btime sol = solve(prob, Optim.KrylovTrustRegion())


cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons)

prob = OptimizationProblem(optf, x0, _p, lcons = [-Inf], ucons = [Inf])
@btime sol = solve(prob, IPNewton()) # Note that -Inf < x[1]^2 + x[2]^2 < Inf is always true


prob = OptimizationProblem(optf, x0, _p, lcons = [-5.0], ucons = [10.0])
@btime sol = solve(prob, IPNewton()) # Again, -5.0 < x[1]^2 + x[2]^2 < 10.0


prob = OptimizationProblem(optf, x0, _p, lcons = [-Inf], ucons = [Inf],
    lb = [-500.0, -500.0], ub = [50.0, 50.0])
@btime sol = solve(prob, IPNewton())


prob = OptimizationProblem(optf, x0, _p, lcons = [0.5], ucons = [0.5],
    lb = [-500.0, -500.0], ub = [50.0, 50.0])
@btime sol = solve(prob, IPNewton()) # Notice now that x[1]^2 + x[2]^2 ≈ 0.5:
# cons(sol.u, _p) = 0.49999999999999994

function con_c(res, x, p)
    res .= [x[1]^2 + x[2]^2]
end

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = con_c)
prob = OptimizationProblem(optf, x0, _p, lcons = [-Inf], ucons = [0.25^2])
@btime sol = solve(prob, IPNewton()) # -Inf < cons_circ(sol.u, _p) = 0.25^2


function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
end

optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = con2_c)
prob = OptimizationProblem(optf, x0, _p, lcons = [-Inf, -Inf], ucons = [Inf, Inf])
@btime sol = solve(prob, IPNewton())


using OptimizationOptimisers
optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(optf, x0, _p)
@btime sol = solve(prob, Adam(0.05), maxiters = 1000, progress = false)


using OptimizationCMAEvolutionStrategy
@btime sol = solve(prob, CMAEvolutionStrategyOpt())


using OptimizationNLopt, ModelingToolkit
optf = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit())
prob = OptimizationProblem(optf, x0, _p)

@btime sol = solve(prob, Opt(:LN_BOBYQA, 2))


@btime sol = solve(prob, Opt(:LD_LBFGS, 2))


prob = OptimizationProblem(optf, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
@btime sol = solve(prob, Opt(:LD_LBFGS, 2))


using OptimizationEvolutionary
@btime sol = solve(prob, CMAES(μ = 40, λ = 100), abstol = 1e-15) # -1.0 ≤ x[1], x[2] ≤ 0.8


using OptimizationBBO
prob = Optimization.OptimizationProblem(rosenbrock, x0, _p, lb = [-1.0, -1.0],
    ub = [0.8, 0.43])
@btime sol = solve(prob, BBO_adaptive_de_rand_1_bin()) # -1.0 ≤ x[1] ≤ 0.8, 0.2 ≤ x[2] ≤ 0.43


Threads.nthreads()


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

