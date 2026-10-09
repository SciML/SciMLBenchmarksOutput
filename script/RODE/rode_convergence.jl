
using StochasticDiffEq, DiffEqNoiseProcess, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5,
      OrdinaryDiffEqVerner, BenchmarkTools, Plots, Random, Statistics, Printf
gr()

const SEED = 20260921
const TEND = 1.0
const FINE = 2^18
const FDT = TEND / FINE
const GRID = collect(range(0.0, TEND; length = FINE + 1))
const STEPS = [16, 32, 64, 128, 256]
const PATHS = 16


wiener_path(rng) = (W = zeros(FINE + 1); W[2:end] .= cumsum(sqrt(FDT) .* randn(rng, FINE)); W)

function cumulative_trapezoid(g, dt)
    I = zeros(length(g)); acc = zero(eltype(g))
    @inbounds for k in 1:(length(g) - 1)
        acc += dt * (g[k] + g[k + 1]) / 2
        I[k + 1] = acc
    end
    return I
end

function fitted_slope(ns, errs)
    x = log2.(1 ./ ns); y = log2.(errs); n = length(x)
    return (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

rode_f(u, p, t, W) = -u * cos(5W)

# W(t) by linear interpolation of the stored path, the way a practitioner hands a
# sampled path to an ODE solver
function make_W(path)
    return function (t)
        s = t / FDT
        i = clamp(floor(Int, s), 0, FINE - 1)
        theta = s - i
        return (1 - theta) * path[i + 1] + theta * path[i + 2]
    end
end


function strong_errors(solve_one)
    rng = MersenneTwister(SEED)
    E = zeros(PATHS, length(STEPS))
    floors = zeros(PATHS)
    for m in 1:PATHS
        path = wiener_path(rng)
        g = cos.(5 .* path)
        exact = exp.(-cumulative_trapezoid(g, FDT))
        floors[m] = maximum(abs, exp.(-cumulative_trapezoid(g[1:4:end], 4FDT)) .- exact[1:4:end])
        for (j, n) in enumerate(STEPS)
            u = solve_one(path, n)
            stride = FINE ÷ n
            E[m, j] = maximum(abs(u[k + 1] - exact[k * stride + 1]) for k in 0:n)
        end
    end
    return [sqrt(mean(E[:, j] .^ 2)) for j in eachindex(STEPS)], maximum(floors)
end

rode_solver(alg) = (path, n) -> solve(
    RODEProblem{false}(rode_f, 1.0, (0.0, TEND), noise = NoiseGrid(GRID, path)),
    alg, dt = TEND / n, adaptive = false, save_everystep = true).u

rode_algs = ["RandomEM" => RandomEM(), "RandomHeun" => RandomHeun(),
             "RandomTamedEM" => RandomTamedEM()]
rode_results = Dict(name => strong_errors(rode_solver(alg)) for (name, alg) in rode_algs)

for (name, _) in rode_algs
    errs, fl = rode_results[name]
    @printf("%-16s slope %.3f   floor margin %.1fx\n", name,
            fitted_slope(STEPS, errs), minimum(errs) / fl)
end


dts = TEND ./ STEPS
plt = plot(xscale = :log10, yscale = :log10, xlabel = "dt", ylabel = "strong error",
           title = "RODE solvers on a Wiener-driven RODE", legend = :bottomright)
for (name, _) in rode_algs
    plot!(plt, dts, rode_results[name][1], marker = :circle, label = name)
end
plot!(plt, dts, dts .* (rode_results["RandomEM"][1][end] / dts[end]),
      linestyle = :dash, color = :black, label = "slope 1")
plt


ode_solver(alg) = function (path, n)
    Wf = make_W(path)
    odef(u, p, t) = -u * cos(5 * Wf(t))
    return solve(ODEProblem(odef, 1.0, (0.0, TEND)), alg,
                 dt = TEND / n, adaptive = false, save_everystep = true).u
end

ode_algs = ["Euler (order 1)" => Euler(), "Tsit5 (order 5)" => Tsit5(),
            "Vern9 (order 9)" => Vern9()]
ode_results = Dict(name => strong_errors(ode_solver(alg)) for (name, alg) in ode_algs)

for (name, _) in ode_algs
    errs, fl = ode_results[name]
    @printf("%-16s slope %.3f   error at dt=1/256 %.3g\n", name,
            fitted_slope(STEPS, errs), errs[end])
end

@printf("\nmax |Euler - RandomEM| over the step counts: %.3g\n",
        maximum(abs, ode_results["Euler (order 1)"][1] .- rode_results["RandomEM"][1]))


plt = plot(xscale = :log10, yscale = :log10, xlabel = "dt", ylabel = "strong error",
           title = "Classical Runge-Kutta on the same RODE", legend = :bottomright)
for (name, _) in ode_algs
    plot!(plt, dts, ode_results[name][1], marker = :square, label = name)
end
plot!(plt, dts, rode_results["RandomEM"][1], marker = :circle, linestyle = :dot,
      label = "RandomEM")
plot!(plt, dts, dts .* (ode_results["Euler (order 1)"][1][end] / dts[end]),
      linestyle = :dash, color = :black, label = "slope 1")
plt


rode_problem(alg) = path -> (
    RODEProblem{false}(rode_f, 1.0, (0.0, TEND), noise = NoiseGrid(GRID, path)), alg)
ode_problem(alg) = function (path)
    Wf = make_W(path)
    odef(u, p, t) = -u * cos(5 * Wf(t))
    return ODEProblem(odef, 1.0, (0.0, TEND)), alg
end

function timed_errors(build)
    rng = MersenneTwister(SEED)
    path = wiener_path(rng)
    g = cos.(5 .* path)
    exact = exp.(-cumulative_trapezoid(g, FDT))
    errs = zeros(length(STEPS)); times = zeros(length(STEPS))
    for (j, n) in enumerate(STEPS)
        prob, alg = build(path)
        dt = TEND / n
        u = solve(prob, alg, dt = dt, adaptive = false, save_everystep = true).u
        times[j] = @belapsed solve($prob, $alg, dt = $dt, adaptive = false,
                                   save_everystep = true) seconds = 1
        stride = FINE ÷ n
        errs[j] = maximum(abs(u[k + 1] - exact[k * stride + 1]) for k in 0:n)
    end
    return errs, times
end

wp = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
for (name, alg) in rode_algs
    wp[name] = timed_errors(rode_problem(alg))
end
for (name, alg) in ode_algs
    wp[name] = timed_errors(ode_problem(alg))
end

@printf("%-16s %-14s %-12s\n", "method", "time at dt=1/256", "error")
for (name, _) in vcat(rode_algs, ode_algs)
    @printf("%-16s %-14.3g %-12.3g\n", name, wp[name][2][end], wp[name][1][end])
end

plt = plot(xscale = :log10, yscale = :log10, xlabel = "time (s)", ylabel = "error",
           title = "Work-precision, single path", legend = :bottomleft)
for (name, _) in rode_algs
    plot!(plt, wp[name][2], wp[name][1], marker = :circle, label = name)
end
for (name, _) in ode_algs
    plot!(plt, wp[name][2], wp[name][1], marker = :square, linestyle = :dash, label = name)
end
plt


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

