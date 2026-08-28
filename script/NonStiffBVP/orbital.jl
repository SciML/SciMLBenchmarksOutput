using BoundaryValueDiffEq, OrdinaryDiffEq, BenchmarkTools, SciMLBase
using OrdinaryDiffEqLowOrderRK

y0 = [
    -4.7763169762853989e+6,
    -3.838639870444152e+5,
    -5.3500183933132319e+6,
    -5528.612564911408,
    1216.8442360202787,
    4845.114446429901,
]
init_val = [
    -4.7763169762853989e+6,
    -3.838639870444152e+5,
    -5.3500183933132319e+6,
    7.0526926403748598e+6,
    -7.9650476230388973e+5,
    -1.191112886366643e+6,
]
J2 = 1.08262668e-3
req = 6378137
mu = 398600.4418e+9
t0 = 86400 * 2.3577475462484435e+4
t1 = 86400 * 2.3577522023524125e+4
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
    dy[4] = -mu * y[1] * w / r3
    dy[5] = -mu * y[2] * w / r3
    return dy[6] = -mu * y[3] * w2 / r3
end

function bc!_generator(resid, sol, init_val)
    resid[1] = sol.u[1][1] - init_val[1]
    resid[2] = sol.u[1][2] - init_val[2]
    resid[3] = sol.u[1][3] - init_val[3]
    resid[4] = sol.u[end][1] - init_val[4]
    resid[5] = sol.u[end][2] - init_val[5]
    return resid[6] = sol.u[end][3] - init_val[6]
end
cur_bc! = (resid, sol, p, t) -> bc!_generator(resid, sol, init_val)
resid_f = Array{Float64}(undef, 6)
bvp = BVProblem(orbital, cur_bc!, y0, tspan; nlls = Val(false))

function validate_solution(sol, tolerance)
    SciMLBase.successful_retcode(sol) || error("solve failed with retcode $(sol.retcode)")
    residual_norm = maximum(abs, sol.resid)
    residual_norm < tolerance ||
        error("boundary residual $residual_norm exceeds tolerance $tolerance")
    return sol
end


tolerance = 1.0e-6
sol = @btime solve(
    bvp,
    Shooting(DP5());
    force_dtmin = true,
    abstol = $tolerance,
    reltol = $tolerance,
    verbose = false,
    odesolve_kwargs = (
        abstol = $tolerance,
        reltol = $tolerance,
        adaptive = false,
        dt = 10000,
    ),
)
validate_solution(sol, tolerance)


dt = (t1 - t0) / 100
sol = @btime solve(
    bvp, MIRK2(); dt = $dt, abstol = $tolerance, reltol = $tolerance, verbose = false
)
validate_solution(sol, tolerance)


sol = @btime solve(
    bvp, MIRK3(); dt = $dt, abstol = $tolerance, reltol = $tolerance, verbose = false
)
validate_solution(sol, tolerance)


sol = @btime solve(
    bvp, MIRK4(); dt = $dt, abstol = $tolerance, reltol = $tolerance, verbose = false
)
validate_solution(sol, tolerance)


sol = @btime solve(
    bvp, MIRK5(); dt = $dt, abstol = $tolerance, reltol = $tolerance, verbose = false
)
validate_solution(sol, tolerance)


sol = @btime solve(
    bvp, MIRK6(); dt = $dt, abstol = $tolerance, reltol = $tolerance, verbose = false
)
validate_solution(sol, tolerance)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
