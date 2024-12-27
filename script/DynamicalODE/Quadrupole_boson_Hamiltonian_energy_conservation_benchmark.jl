
using OrdinaryDiffEq, Plots, DiffEqCallbacks, LinearAlgebra
using TaylorIntegration
using ParameterizedFunctions
using StaticArrays
gr()
default(fmt=:png)

T(p) = A / 2 * norm(p)^2
V(q) = A / 2 * (q[1]^2 + q[2]^2) + B / sqrt(2) * q[1] * (3 * q[2]^2 - q[1]^2) + D / 4 * (q[1]^2 + q[2]^2)^2
H(p, q, params) = T(p) + V(q)

const A, B, D = 1., 0.55, 0.4

function iip_dq(dq, p, q, params, t)
    dq[1] = A * p[1]
    dq[2] = A * p[2]
end

function iip_dp(dp, p, q, params, t)
    dp[1] = -A * q[1] - 3 * B / sqrt(2) * (q[2]^2 - q[1]^2) - D * q[1] * (q[1]^2 + q[2]^2)
    dp[2] = -q[2] * (A + 3 * sqrt(2) * B * q[1] + D * (q[1]^2 + q[2]^2))
end

const iip_q0 = [4.919080920016389, 2.836942666663649]
const iip_p0 = [0., 0.]
const iip_u0 = vcat(iip_p0,iip_q0)

function oop_dq(p, q, params, t)
    p
end

function oop_dp(p, q, params, t)
    dp1 = -A * q[1] - 3 * B / sqrt(2) * (q[2]^2 - q[1]^2) - D * q[1] * (q[1]^2 + q[2]^2)
    dp2 = -q[2] * (A + 3 * sqrt(2) * B * q[1] + D * (q[1]^2 + q[2]^2))
    @SVector [dp1, dp2]
end

const oop_q0 = @SVector [4.919080920016389, 2.836942666663649]
const oop_p0 = @SVector [0., 0.]
const oop_u0 = vcat(oop_p0,oop_q0)

function hamilton(z, params, t)
    SVector(
    -A * z[3] - 3 * B / sqrt(2) * (z[4]^2 - z[3]^2) - D * z[3] * (z[3]^2 + z[4]^2),
    -z[4] * (A + 3 * sqrt(2) * B * z[3] + D * (z[3]^2 + z[4]^2)),
    z[1],
    z[2]
    )
end

function g(resid, u, p)
    resid[1] = H([u[1],u[2]],[u[3],u[4]],nothing) - E
    resid[2:4] .= 0
end

function g_jacobian(J, u, p)
    J[1, 1] = u[1]
    J[1, 2] = u[2]
    J[1, 3] = u[3]
    J[1, 4] = u[4]
    J[2:4, :] .= 0
end

const E = H(iip_p0, iip_q0, nothing)
const resid_prototype = zeros(4)
const cb = ManifoldProjection(g, manifold_jacobian=g_jacobian, resid_prototype=resid_prototype, nlopts=Dict(:ftol=>1e-13))


energy_err(sol) = map(i->H([sol[1,i], sol[2,i]], [sol[3,i], sol[4,i]],nothing)-E, 1:length(sol.u))
abs_energy_err(sol) = [abs.(H([sol[1,j], sol[2,j]], [sol[3,j], sol[4,j]],nothing) - E) for j=1:length(sol.u)]

function compare(mode=:inplace, all=true, plt=nothing; tmax=1e2)
    if mode == :inplace
        prob = DynamicalODEProblem(iip_dp, iip_dq, iip_p0, iip_q0, (0., tmax))
    else
        prob = DynamicalODEProblem(oop_dp, oop_dq, oop_p0, oop_q0, (0., tmax))
    end
    prob_linear = ODEProblem(hamilton, vcat(iip_p0, iip_q0), (0., tmax))

    GC.gc()
    (mode == :inplace  && all) && @time sol1 = solve(prob, Vern9(), callback=cb, abstol=1e-14, reltol=1e-14)
    GC.gc()
    @time sol2 = solve(prob, KahanLi8(), dt=1e-2, maxiters=1e10)
    GC.gc()
    @time sol3 = solve(prob, SofSpa10(), dt=1e-2, maxiters=1e8)
    GC.gc()
    @time sol4 = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14)
    GC.gc()
    @time sol5 = solve(prob, DPRKN12(), abstol=1e-14, reltol=1e-14)
    GC.gc()
    (mode == :inplace && all) && @time sol6 = solve(prob_linear, TaylorMethod(50), abstol=1e-20)

    (mode == :inplace && all) && println("Vern9 + ManifoldProjection max energy error:\t"*
        "$(maximum(abs_energy_err(sol1)))\tin\t$(length(sol1.u))\tsteps.")
    println("KahanLi8 max energy error:\t\t\t$(maximum(abs_energy_err(sol2)))\tin\t$(length(sol2.u))\tsteps.")
    println("SofSpa10 max energy error:\t\t\t$(maximum(abs_energy_err(sol3)))\tin\t$(length(sol3.u))\tsteps.")
    println("Vern9 max energy error:\t\t\t\t$(maximum(abs_energy_err(sol4)))\tin\t$(length(sol4.u))\tsteps.")
    println("DPRKN12 max energy error:\t\t\t$(maximum(abs_energy_err(sol5)))\tin\t$(length(sol5.u))\tsteps.")
    (mode == :inplace && all) && println("TaylorMethod max energy error:\t\t\t$(maximum(abs_energy_err(sol6)))"*
        "\tin\t$(length(sol6.u))\tsteps.")

    if plt == nothing
        plt = plot(xlabel="t", ylabel="Energy error")
    end

    (mode == :inplace && all) && plot!(sol1.t, energy_err(sol1), label="Vern9 + ManifoldProjection")
    plot!(sol2.t, energy_err(sol2), label="KahanLi8", ls=mode==:inplace ? :solid : :dash)
    plot!(sol3.t, energy_err(sol3), label="SofSpa10", ls=mode==:inplace ? :solid : :dash)
    plot!(sol4.t, energy_err(sol4), label="Vern9", ls=mode==:inplace ? :solid : :dash)
    plot!(sol5.t, energy_err(sol5), label="DPRKN12", ls=mode==:inplace ? :solid : :dash)
    (mode == :inplace && all) && plot!(sol6.t, energy_err(sol6), label="TaylorMethod")

    return plt
end


compare(tmax=1e2)


compare(tmax=1e3)


compare(tmax=1e4)


compare(tmax=2e4)


function in_vs_out(;all=false, tmax=1e2)
    println("In place versions:")
    plt = compare(:inplace, all, tmax=tmax)
    println("\nOut of place versions:")
    plt = compare(:oop, false, plt; tmax=tmax)
end


in_vs_out(all=true, tmax=1e3)


in_vs_out(tmax=1e2)


in_vs_out(tmax=1e3)


in_vs_out(tmax=1e4)


in_vs_out(tmax=2e4)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

