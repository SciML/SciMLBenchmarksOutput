
using NonlinearSolve, LinearAlgebra, BenchmarkTools, CairoMakie, PrettyTables

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5;


const CALLS = Ref(0)
reset_calls!() = (CALLS[] = 0)


h_quad(u, p, λ) = (CALLS[] += 1; [(1 - λ) * (u[1] - 4.0) + λ * (u[1]^2 - 4.0)])
d_quad(u, p) = (CALLS[] += 1; [u[1]^2 - 4.0])
u0_quad = [4.0]
uref_quad = [2.0]


h_fold(u, p, λ) = (CALLS[] += 1; [u[1]^3 - 3u[1] - (-3 + 6λ)])
d_fold(u, p) = (CALLS[] += 1; [u[1]^3 - 3u[1] - 3.0])
u0_fold = [-2.1038034027355366]
uref_fold = [2.1038034027355366]


us = range(-2.5, 2.5; length = 500)
λs = @. (us^3 - 3us + 3) / 6
fig = Figure(; size = (800, 500))
ax = Axis(fig[1, 1]; xlabel = "λ", ylabel = "u",
    title = "S-curve homotopy path: H(u, λ) = u³ - 3u - (-3 + 6λ) = 0")
lines!(ax, λs, us; linewidth = 3)
vlines!(ax, [0.0, 1.0]; color = :gray, linestyle = :dash)
scatter!(ax, [0.0], [-2.1038034027355366]; color = :blue, markersize = 16,
    label = "start (λ=0)")
scatter!(ax, [1.0], [2.1038034027355366]; color = :green, markersize = 16,
    label = "target (λ=1)")
scatter!(ax, [5 / 6, 1 / 6], [-1.0, 1.0]; color = :red, marker = :xcross,
    markersize = 16, label = "folds")
axislegend(ax; position = :lt)
fig


const N_CUBIC = 50
const c_cubic = [1.0 + 0.25 * (i > 1) + 0.25 * (i < N_CUBIC) + 1.0 for i in 1:N_CUBIC]
function h_cubic!(du, u, p, λ)
    CALLS[] += 1
    for i in 1:N_CUBIC
        du[i] = u[i] + 0.25 * (i > 1 ? u[i - 1] : 0.0) +
                0.25 * (i < N_CUBIC ? u[i + 1] : 0.0) + λ * u[i]^3 - c_cubic[i]
    end
    return nothing
end
d_cubic!(du, u, p) = h_cubic!(du, u, p, 1.0)
uref_cubic = ones(N_CUBIC)


h_ramp(u, p, λ) = (CALLS[] += 1; [u[1]^3 - 1.0 - 7λ])
d_ramp(u, p) = (CALLS[] += 1; [u[1]^3 - 8.0])
u0_ramp = [100.0]
uref_ramp = [2.0]


homotopy_solvers = [
    (; name = "HomotopySweep (default inner)", alg = HomotopySweep()),
    (; name = "HomotopySweep (NewtonRaphson)", alg = HomotopySweep(inner = NewtonRaphson())),
    (; name = "ArcLengthContinuation (secant)", alg = ArcLengthContinuation()),
    (; name = "ArcLengthContinuation (tangent)",
        alg = ArcLengthContinuation(predictor = :tangent)),
    (; name = "solve(prob) default", alg = nothing),
];

direct_solvers = [
    (; name = "NewtonRaphson", alg = NewtonRaphson()),
    (; name = "TrustRegion", alg = TrustRegion()),
    (; name = "LevenbergMarquardt", alg = LevenbergMarquardt()),
    (; name = "Broyden", alg = Broyden()),
    (; name = "Default PolyAlgorithm", alg = nothing),
];


function run_solver(prob, alg, uref)
    reset_calls!()
    local sol
    try
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
    catch err
        Base.printstyled("[Warn] Solver threw $(typeof(err)).\n"; color = :red)
        return (; retcode = "exception", success = false, correct = false,
            calls = CALLS[], err = NaN, time = NaN)
    end
    calls = CALLS[]
    success = SciMLBase.successful_retcode(sol.retcode)
    err = norm(sol.u .- uref, Inf)
    correct = success && err < 1e-6
    time = alg === nothing ? (@belapsed solve($prob)) : (@belapsed solve($prob, $alg))
    return (; retcode = string(sol.retcode), success, correct, calls, err, time)
end

function benchmark_case(hprob, dprob, uref)
    results = []
    for s in homotopy_solvers
        push!(results, (; s.name, kind = "homotopy", run_solver(hprob, s.alg, uref)...))
    end
    for s in direct_solvers
        push!(results, (; s.name, kind = "direct", run_solver(dprob, s.alg, uref)...))
    end
    return results
end

fmt_time(t) = isnan(t) ? "—" : t < 1e-3 ? string(round(t * 1e6; digits = 1), " μs") :
              string(round(t * 1e3; digits = 2), " ms")
fmt_err(e) = isnan(e) ? "—" : string(round(e; sigdigits = 3))

function result_table(results)
    data = permutedims(reduce(hcat,
        [[r.name, r.kind, r.retcode, r.correct ? "yes" : "no", r.calls,
             fmt_err(r.err), fmt_time(r.time)] for r in results]))
    io = IOBuffer()
    println(io, "```@raw html")
    pretty_table(io, data; backend = :html, alignment = :c,
        column_labels = ["Solver", "Kind", "Return Code", "Correct Root",
            "Residual Calls", "‖u - u*‖∞", "Time"])
    println(io, "```")
    return Base.Text(String(take!(io)))
end


res_quad = benchmark_case(HomotopyProblem(h_quad, copy(u0_quad)),
    NonlinearProblem(d_quad, copy(u0_quad)), uref_quad);


result_table(res_quad)


res_fold = benchmark_case(HomotopyProblem(h_fold, copy(u0_fold)),
    NonlinearProblem(d_fold, copy(u0_fold)), uref_fold);


result_table(res_fold)


res_cubic_good = benchmark_case(HomotopyProblem{true}(h_cubic!, ones(N_CUBIC)),
    NonlinearProblem{true}(d_cubic!, ones(N_CUBIC)), uref_cubic);


result_table(res_cubic_good)


res_cubic_bad = benchmark_case(HomotopyProblem{true}(h_cubic!, 10 .* ones(N_CUBIC)),
    NonlinearProblem{true}(d_cubic!, 10 .* ones(N_CUBIC)), uref_cubic);


result_table(res_cubic_bad)


res_ramp = benchmark_case(HomotopyProblem(h_ramp, copy(u0_ramp)),
    NonlinearProblem(d_ramp, copy(u0_ramp)), uref_ramp);


result_table(res_ramp)


problem_names = ["Quadratic", "S-curve fold", "Cubic n=50 (good u₀)",
    "Cubic n=50 (bad u₀)", "Hard ramp"]
all_results = [res_quad, res_fold, res_cubic_good, res_cubic_bad, res_ramp]
solver_names = [r.name for r in all_results[1]]
solver_kinds = [r.kind for r in all_results[1]]

fig = begin
    nsolver = length(solver_names)
    nprob = length(problem_names)
    xs = Int[]
    dodge = Int[]
    ys = Float64[]
    for (pi, res) in enumerate(all_results), (si, r) in enumerate(res)
        # failed / wrong-root solves are omitted, leaving a visible gap in the group
        r.correct || continue
        push!(xs, pi)
        push!(dodge, si)
        push!(ys, r.time)
    end
    # log-scale axes cannot render bars that start at 0, so anchor them just below
    # the smallest measured time
    lo = 10.0^floor(log10(minimum(ys)) - 0.3)

    colors = cgrad(:tableau_20, nsolver; categorical = true)
    fig = Figure(; size = (1400, 700))
    ax = Axis(fig[1, 1]; yscale = log10, ylabel = "Time (s), log scale",
        title = "Homotopy continuation vs direct solvers (missing bar = wrong/failed solve)",
        xticks = (1:nprob, problem_names), xticklabelrotation = π / 12)
    barplot!(ax, xs, ys; dodge = dodge, n_dodge = nsolver, color = colors[dodge],
        strokewidth = 1, fillto = lo)
    ylims!(ax, lo, 10.0^ceil(log10(maximum(ys)) + 0.3))

    elements = [PolyElement(; polycolor = colors[i]) for i in 1:nsolver]
    Legend(fig[1, 2], elements,
        [n * (k == "homotopy" ? " [H]" : " [D]")
         for (n, k) in zip(solver_names, solver_kinds)],
        "Solver"; framevisible = true)
    fig
end


save("homotopy_summary.svg", fig)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

