
using BoundaryValueDiffEq, OrdinaryDiffEq, BenchmarkTools

y0 = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    -5528.612564911408,
    1216.8442360202787,
    4845.114446429901,
]
init_val = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    7.0526926403748598E+06,
    -7.9650476230388973E+05,
    -1.1911128863666430E+06,
]
J2 = 1.08262668E-3
req = 6378137
myu = 398600.4418E+9
t0 = 86400 * 2.3577475462484435E+04
t1 = 86400 * 2.3577522023524125E+04
tspan = (t0, t1)

# ODE solver
function orbital(dy, y, p, t)
    r2 = (y[1]^2 + y[2]^2 + y[3]^2)
    r3 = r2^(3 / 2)
    w = 1 + 1.5J2 * (req * req / r2) * (1 - 5y[3] * y[3] / r2)
    w2 = 1 + 1.5J2 * (req * req / r2) * (3 - 5y[3] * y[3] / r2)
    dy[1] = y[4]
    dy[2] = y[5]
    dy[3] = y[6]
    dy[4] = -myu * y[1] * w / r3
    dy[5] = -myu * y[2] * w / r3
    dy[6] = -myu * y[3] * w2 / r3
end

function bc!_generator(resid, sol, init_val)
    resid[1] = sol[1][1] - init_val[1]
    resid[2] = sol[1][2] - init_val[2]
    resid[3] = sol[1][3] - init_val[3]
    resid[4] = sol[end][1] - init_val[4]
    resid[5] = sol[end][2] - init_val[5]
    resid[6] = sol[end][3] - init_val[6]
end
cur_bc! = (resid, sol, p, t) -> bc!_generator(resid, sol, init_val)
resid_f = Array{Float64}(undef, 6)
bvp = BVProblem(orbital, cur_bc!, y0, tspan)


@btime sol = solve(bvp, Shooting(DP5()), abstol = 1e-13, reltol = 1e-13)


@btime sol = solve(bvp, MIRK2(), abstol = 1e-13, reltol = 1e-13)


@btime sol = solve(bvp, MIRK3(), abstol = 1e-13, reltol = 1e-13)


@btime sol = solve(bvp, MIRK4(), abstol = 1e-13, reltol = 1e-13)


@btime sol = solve(bvp, MIRK5(), abstol = 1e-13, reltol = 1e-13)


@btime sol = solve(bvp, MIRK6(), abstol = 1e-13, reltol = 1e-13)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

