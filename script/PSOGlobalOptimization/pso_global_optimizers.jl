
using Random; Random.seed!(42)
using BlackBoxOptimizationBenchmarking, CairoMakie, Optimization, Memoize, Statistics
using StaticArrays, LinearAlgebra, ForwardDiff, KernelAbstractions, CUDA
using OptimizationBBO, OptimizationOptimJL, OptimizationEvolutionary, OptimizationNLopt
using OptimizationMetaheuristics, OptimizationSciPy
using OptimizationOptimJL: Optim          # NelderMead and SAMIN live here
using ParallelParticleSwarms
CairoMakie.activate!()

import BlackBoxOptimizationBenchmarking: Chain, BenchmarkSetup, BenchmarkResults,
    BBOBFunction, FunctionCallsCounter, solve_problem, pinit, compute_CI
const BBOB = BlackBoxOptimizationBenchmarking
const SciMLBase = Optimization.SciMLBase

const PSOKernel = ParallelParticleSwarms.ParallelPSOKernel
const SyncPSOKernel = ParallelParticleSwarms.ParallelSyncPSOKernel
const SerialPSO = ParallelParticleSwarms.SerialPSO
const HybridPSO = ParallelParticleSwarms.HybridPSO
const BACKEND = CUDABackend()

const DIMENSIONS = (3, 5, 10)
const NTRIALS = 10
const Δf = 1.0e-6
const NUM_PARTICLES = 50_000    # GPU swarm
const SERIAL_PARTICLES = 512       # CPU swarm
const RUN_LENGTH = round.(Int, 10 .^ LinRange(1, 4, 15))   # budget experiment
const MAX_TTS_ITERS = 10_000    # time-to-success experiment
const TTS_CHUNK = 20        # GPU: read back the best cost every this many iterations
# f4, f7, f10 crash the GPU kernels and are excluded.
suite(D) = filter(f -> nameof(f.f) ∉ (:f4, :f7, :f10), BBOB.bbob_suite(Val(D)))
const SUITES = (suite(3), suite(5), suite(10))
const TEST_FUNCTIONS = first(SUITES)   # names/order for the heatmap; rates pool all three D


chain(t; isboxed = false) =
    Chain(BenchmarkSetup(t; isboxed), BenchmarkSetup(Optim.NelderMead(); isboxed = false), 0.9)

setup = Dict(
    "NelderMead" => () -> BenchmarkSetup(Optim.NelderMead()),
    "NLopt.GN_CRS2_LM()" => () -> chain(NLopt.GN_CRS2_LM(), isboxed = true),
    "NLopt.GN_DIRECT()" => () -> chain(NLopt.GN_DIRECT(), isboxed = true),
    "NLopt.GN_ESCH()" => () -> chain(NLopt.GN_ESCH(), isboxed = true),
    "OptimizationEvolutionary.GA()" => () -> chain(OptimizationEvolutionary.GA(), isboxed = true),
    "OptimizationEvolutionary.DE()" => () -> chain(OptimizationEvolutionary.DE(), isboxed = true),
    "OptimizationEvolutionary.ES()" => () -> chain(OptimizationEvolutionary.ES(), isboxed = true),
    "Optim.SAMIN" => () -> chain(Optim.SAMIN(verbosity = 0), isboxed = true),
    "BBO_adaptive_de_rand_1_bin" => () -> chain(BBO_adaptive_de_rand_1_bin(), isboxed = true),
    "BBO_de_rand_2_bin" => () -> chain(BBO_de_rand_2_bin(), isboxed = true),
    "OptimizationMetaheuristics.ECA" => () -> chain(OptimizationMetaheuristics.ECA(), isboxed = true),
    "OptimizationMetaheuristics.DE" => () -> chain(OptimizationMetaheuristics.DE(), isboxed = true),
    "ScipyDifferentialEvolution" => () -> chain(ScipyDifferentialEvolution(), isboxed = true),
    "SerialPSO" => () -> SerialPSO(SERIAL_PARTICLES),
    "PSOKernel" => () -> PSOKernel(NUM_PARTICLES; backend = BACKEND, global_update = true),
    "SyncPSOKernel" => () -> SyncPSOKernel(NUM_PARTICLES; backend = BACKEND),
    "HybridPSO_LBFGS" => () -> HybridPSO(pso = SyncPSOKernel(NUM_PARTICLES; backend = BACKEND); backend = BACKEND),
)

const LABELS = collect(keys(setup))
const PSO_KEYS = Set(["SerialPSO", "PSOKernel", "SyncPSOKernel", "HybridPSO_LBFGS"])
particles_of(algo) = algo == "SerialPSO" ? SERIAL_PARTICLES : NUM_PARTICLES


_value(x::Real) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_to_f64(x) = Float64(_value(x))

function pso_objective(f::BBOBFunction, x)
    any(xi -> !isfinite(_value(xi)) || abs(_value(xi)) > 15, x) && return zero(first(x)) + 1.0e10
    return f(x)
end

# `obj` is any x -> objective callable. u0 only fixes type and dimension; the swarm is
# sampled from the box.
function pso_problem(obj, ::Val{D}) where {D}
    optf = OptimizationFunction{false}((x, p) -> obj(x), SciMLBase.NoAD())
    lb = SVector{D, Float64}(ntuple(_ -> -5.5, Val(D)))     # same box as the baselines
    ub = SVector{D, Float64}(ntuple(_ -> 5.5, Val(D)))
    return OptimizationProblem{false}(optf, SVector{D, Float64}(pinit(D)), nothing; lb, ub)
end
pso_problem(obj, f::BBOBFunction{F, N}) where {F, N} = pso_problem(obj, Val(N))

pso_solve(opt, prob, budget) = opt isa HybridPSO ?
    solve(prob, opt; maxiters = budget, local_maxiters = 50, abstol = 1.0e-8, reltol = 1.0e-8) :
    solve(prob, opt; maxiters = budget)


function run_one(algo, f::BBOBFunction, budget::Int)
    if algo in PSO_KEYS
        sol = pso_solve(setup[algo](), pso_problem(x -> pso_objective(f, x), f), budget)
        u = sol.u isa AbstractVector ? sol.u : sol.u[]
        return _to_f64(sol.objective), u, budget * particles_of(algo)
    else
        counted = FunctionCallsCounter(f)
        sol = solve_problem(setup[algo](), counted, length(f.x_opt), budget)
        return sol.objective, sol.u, counted.count
    end
end

function benchmark(algo)
    Nf = length(TEST_FUNCTIONS)
    Nd = length(SUITES)
    success = zeros(Nf, length(RUN_LENGTH)); dist = zeros(Nf, length(RUN_LENGTH))
    calls = zeros(Nf, length(RUN_LENGTH))
    for funcs in SUITES, (fi, f) in enumerate(funcs)
        run_one(algo, f, 10)                               # warm-up per (function, D): compile, discard
        for (ri, rl) in enumerate(RUN_LENGTH), _ in 1:NTRIALS
            fval, u, n = run_one(algo, f, rl)
            success[fi, ri] += fval < f.f_opt + Δf
            dist[fi, ri] += norm(u .- f.x_opt)
            calls[fi, ri] += n
        end
    end
    success ./= NTRIALS * Nd
    dist ./= NTRIALS * Nd
    calls ./= NTRIALS * Nd
    sr = vec(mean(success, dims = 1))
    Neff = NTRIALS * Nf * Nd
    qlow, qhigh = compute_CI(sr, Neff, 0.25)
    return BenchmarkResults(;
        run_length = collect(RUN_LENGTH),
        success_count = round.(Int, sr .* Neff),
        success_rate = sr,
        success_rate_qlow = qlow,
        success_rate_qhigh = qhigh,
        distance_to_minimizer = vec(mean(dist, dims = 1)),
        minimum = fill(NaN, length(RUN_LENGTH)),
        runtime = 0.0,
        Neffective = Neff,
        callcount = vec(mean(calls, dims = 1)),
        success_rate_per_function = success[:, end],
    )
end

@memoize run_bench(algo) = benchmark(algo)

results = Dict{String, BenchmarkResults}()
for algo in LABELS
    algo in PSO_KEYS && (results[algo] = run_bench(algo))
end   # GPU first
for algo in LABELS
    algo in PSO_KEYS || (results[algo] = run_bench(algo))
end
[results[l] for l in LABELS]


# Wrap an objective so the first evaluation below `target` records the elapsed time.
function tracked(obj, target, t0)
    hit = Ref(Inf)
    g(x) = (v = obj(x); v < target && hit[] == Inf && (hit[] = time() - t0); v)
    return g, hit
end

# Advance a PSO cache in chunks; return the time of the first chunk whose best cost is
# below `target`, or Inf. (solve! resets the inertia weight each call; harmless with the
# default wdamp = 1.)
function chunked!(cache, target, t0)
    for _ in 1:cld(MAX_TTS_ITERS, TTS_CHUNK)
        sol = SciMLBase.solve!(cache; maxiters = TTS_CHUNK)
        if cache.alg isa SyncPSOKernel                       # sync kernel does not store gbest back
            obj = _to_f64(sol.objective)
            cache.gbest = ParallelParticleSwarms.SPSOGBest(sol.u, obj)
        end
        _to_f64(sol.objective) < target && return time() - t0
    end
    return Inf
end

function time_to_success_one(algo, f::BBOBFunction)
    opt, target = setup[algo](), f.f_opt + Δf
    t0 = time()
    if !(algo in PSO_KEYS)
        g, hit = tracked(f, target, t0)
        solve_problem(opt, g, length(f.x_opt), MAX_TTS_ITERS)
        return hit[]
    elseif opt isa SerialPSO
        g, hit = tracked(x -> pso_objective(f, x), target, t0)
        pso_solve(opt, pso_problem(g, f), MAX_TTS_ITERS)
        return hit[]
    elseif opt isa HybridPSO
        cache = SciMLBase.init(pso_problem(x -> pso_objective(f, x), f), opt)
        t = chunked!(cache.pso_cache, target, t0)             # PSO phase
        isfinite(t) && return t
        sol = SciMLBase.solve!(cache; maxiters = 0, local_maxiters = 50, abstol = 1.0e-8, reltol = 1.0e-8)
        return _to_f64(sol.objective) < target ? time() - t0 : Inf   # L-BFGS phase only
    else
        cache = SciMLBase.init(pso_problem(x -> pso_objective(f, x), f), opt)
        return chunked!(cache, target, t0)
    end
end

function time_to_success(algo)
    times = Float64[]
    for funcs in SUITES, f in funcs
        time_to_success_one(algo, f)                       # warm-up per (function, D): compile, discard
        for _ in 1:NTRIALS
            push!(times, time_to_success_one(algo, f))
        end
    end
    return times
end

@memoize run_tts(algo) = time_to_success(algo)

tts = Dict{String, Vector{Float64}}()
for algo in LABELS
    algo in PSO_KEYS && (tts[algo] = run_tts(algo))
end
for algo in LABELS
    algo in PSO_KEYS || (tts[algo] = run_tts(algo))
end


using Printf
for l in sort(collect(keys(results)))
    v = filter(isfinite, tts[l])
    @printf "%-28s final success = %.3f   solved = %3d/%d   median TTS = %s\n" l results[l].success_rate[end] length(v) length(tts[l]) (isempty(v) ? "never" : @sprintf("%.3f s", median(v)))
end


const MARKERS = [
    :circle, :rect, :utriangle, :diamond, :dtriangle, :pentagon, :cross,
    :xcross, :star4, :star5, :hexagon, :star6, :ltriangle, :rtriangle,
]
const LINESTYLES = [:solid, :dash, :dot, :dashdot, (:dot, :dense)]
const STYLE = Dict(l => (MARKERS[mod1(i, end)], LINESTYLES[mod1(i, end)]) for (i, l) in enumerate(LABELS))

# xs, ys: Dict label => vector. Legend ordered by each curve's final y value.
function plot_curves(xs, ys; xlabel, ylabel, xlims = (nothing, nothing), ylims = (0, 1), best_is_high = true)
    order = sort(collect(keys(ys)); by = l -> ys[l][end], rev = best_is_high)
    fig = Figure(size = (1100, 450))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel, ylabel, limits = (xlims..., ylims...))
    for l in order
        marker, linestyle = STYLE[l]
        scatterlines!(ax, xs[l], ys[l]; label = l, marker, linestyle, linewidth = 2, markersize = 6)
    end
    Legend(fig[1, 2], ax; framevisible = false)
    return fig
end


plot_curves(
    Dict(l => Float64.(RUN_LENGTH) for l in LABELS),
    Dict(l => results[l].success_rate for l in LABELS);
    xlabel = "Iterations", ylabel = "Success rate", xlims = (1, maximum(RUN_LENGTH))
)


evals_labels = filter(!=("HybridPSO_LBFGS"), LABELS)
plot_curves(
    Dict(l => results[l].callcount for l in evals_labels),
    Dict(l => results[l].success_rate for l in evals_labels);
    xlabel = "Function evaluations", ylabel = "Success rate", xlims = (1, 1.0e9)
)


finite = filter(isfinite, reduce(vcat, values(tts); init = Float64[]))
isempty(finite) && (finite = [1.0e-3, 1.0e3])          # nothing succeeded; keep the plot alive
thresholds = 10 .^ range(log10(minimum(finite) / 2), log10(maximum(finite) * 2), length = 50)
cdf(times) = [count(<=(T), times) / length(times) for T in thresholds]

plot_curves(
    Dict(l => thresholds for l in LABELS),
    Dict(l => cdf(tts[l]) for l in LABELS);
    xlabel = "Wall time (s)", ylabel = "Success rate"
)


M = reduce(hcat, results[l].success_rate_per_function for l in LABELS)  # functions × optimizers
order = sortperm(vec(mean(M, dims = 1)))
fig = Figure(size = (1150, 600))
ax = Axis(
    fig[1, 1]; xticks = (1:length(TEST_FUNCTIONS), string.(TEST_FUNCTIONS)),
    yticks = (1:length(LABELS), LABELS[order]), xticklabelrotation = π / 4
)
hm = heatmap!(ax, M[:, order]; colormap = :RdYlGn, colorrange = (0, 1))
Colorbar(fig[1, 2], hm; label = "Success rate")
fig


plot_curves(
    Dict(l => Float64.(RUN_LENGTH) for l in LABELS),
    Dict(l => results[l].distance_to_minimizer for l in LABELS);
    xlabel = "Iterations", ylabel = "Mean distance to minimizer",
    xlims = (1, maximum(RUN_LENGTH)), ylims = (0, 5), best_is_high = false
)


med(l) = (v = filter(isfinite, tts[l]); isempty(v) ? NaN : median(v))
rt = [med(l) for l in LABELS]
rt ./= med("NelderMead")
fig = Figure(size = (1050, 520))
ax = Axis(
    fig[1, 1]; yscale = log10, ylabel = "Median time to success relative to Nelder-Mead",
    xticks = (1:length(LABELS), LABELS), xticklabelrotation = π / 4
)
barplot!(ax, 1:length(LABELS), rt)
fig


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

