
using NonlinearSolve, StaticArrays, BenchmarkTools

f(u, p) = u .* u .- p
u0 = @SVector[1.0, 1.0]
p = 2.0
static_prob = NonlinearProblem(f, u0, p)

iip_f(du, u, p) = (du .= u .* u .- p)
prob = NonlinearProblem(iip_f, Vector(u0), p)


@btime sol = solve(static_prob, NewtonRaphson(), reltol = 1e-9)


@btime sol = solve(static_prob, TrustRegion(), reltol = 1e-9)


using SimpleNonlinearSolve
@btime sol = solve(static_prob, SimpleNewtonRaphson(), reltol = 1e-9)


@btime sol = solve(static_prob, Halley(), reltol = 1e-9)


@btime sol = solve(static_prob, Klement(), reltol = 1e-9)


@btime sol = solve(static_prob, SimpleTrustRegion(), reltol = 1e-9)


@btime sol = solve(static_prob, SimpleDFSane(), reltol = 1e-9)


@btime sol = solve(static_prob, Broyden(), reltol = 1e-9)


@btime sol = solve(static_prob, LBroyden(), reltol = 1e-9)


@btime sol = solve(prob, NewtonRaphson(), reltol = 1e-9)


@btime sol = solve(prob, TrustRegion(), reltol = 1e-9)


using NonlinearSolveMINPACK
@btime sol = solve(prob, CMINPACK(method=:hybr), reltol = 1e-9)


@btime sol = solve(prob, CMINPACK(method=:lm), reltol = 1e-9)


@btime sol = solve(prob, CMINPACK(method=:lmdif), reltol = 1e-9)


@btime sol = solve(prob, CMINPACK(method=:hybrid), reltol = 1e-9)


using SciMLNLSolve
@btime sol = solve(prob, NLSolveJL(), reltol = 1e-9)


@btime sol = solve(prob, NLSolveJL(method=:newton), reltol = 1e-9)


using Sundials
@btime sol = solve(prob, KINSOL(), reltol = 1e-9)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

