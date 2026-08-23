
using BenchmarkTools, Random
using LinearAlgebra, SparseArrays, LinearSolve, Sparspak
# PureUMFPACK backs PureUMFPACKFactorization via LinearSolvePureUMFPACKExt.
# Use `import` (not `using`): PureUMFPACK ≤0.1 exports `solve`, which collides
# with LinearSolve/CommonSolve. PureKLU / SupernodalLU need no extra load.
import PureUMFPACK
import Pardiso
import ParU_jll
using Plots

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

# Sparse matrix generation on an n-dimensional rectangular grid. After
# https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/135
# by A. Braunstein (same generator as SparsePDE.jmd).

A ⊕ B = kron(I(size(B, 1)), A) + kron(B, I(size(A, 1)))

function lattice(n; Tv = Float64)
    d = fill(2 * one(Tv), n)
    d[1] = one(Tv)
    d[end] = one(Tv)
    spdiagm(1 => -ones(Tv, n - 1), 0 => d, -1 => -ones(Tv, n - 1))
end

lattice(L...; Tv = Float64) = lattice(L[1]; Tv) ⊕ lattice(L[2:end]...; Tv)

#
# Matrix like a finite difference discretization of ``-Δu + δu`` in a
# `dim`-dimensional unit cube with approximately N unknowns; strictly diagonally
# dominant, so every method here should succeed — and we verify that they do.
#
function fdmatrix(N; dim = 2, Tv = Float64, δ = 1.0e-2)
    n = N^(1 / dim) |> ceil |> Int
    lattice([n for i in 1:dim]...; Tv) + Tv(δ) * I
end

# `nothing` = LinearSolve's automatic default selection.
algs = [
    ("UMFPACK", UMFPACKFactorization()),
    ("KLU", KLUFactorization()),
    ("Pardiso", MKLPardisoFactorize()),
    ("Sparspak", SparspakFactorization()),
    ("PureKLU", PureKLUFactorization()),
    ("PureUMFPACK", PureUMFPACKFactorization()),
    # ParU is EXCLUDED from this sweep. Its METIS-based analysis routes every
    # allocation through Julia's counted-malloc path, and across many
    # factorizations this ratchets the GC's allocation accounting upward
    # irreversibly (GC.gc(true) does not reset it). Once tripped, EVERY
    # allocation-heavy code path in the process crawls — a later sweep wedged
    # inside SupernodalLU's symbolic analysis purely as collateral. Sub-second
    # solves become hours. See the LinearSolve.jl issue for the reproducer.
    ("SupernodalLU", SupernodalLUFactorization()),
    ("SupernodalLU (threaded)", SupernodalLUFactorization(threaded = true)),
    ("Default", nothing),
]
algnames = first.(algs)
cols = [:red, :blue, :green, :magenta, :gold, :brown, :turquoise, :orange, :black]


function bench_alg(A, b, alg)
    prob = LinearProblem(A, b)
    mk() = alg === nothing ? init(prob) : init(prob, alg)

    # Correctness gate before any timing.
    cache = mk()
    sol = solve!(cache)
    res = norm(A * sol.u - b) / norm(b)
    if !(res < 1e-8)
        @warn "correctness gate failed — omitting from plot" alg res
        return (first = NaN, resolve = NaN)
    end

    t_first = @belapsed solve!(c) setup=(c = $mk()) evals=1
    # `cache` is factorized above; repeated solve! reuses the factorization.
    t_resolve = @belapsed solve!($cache)

    return (first = t_first, resolve = t_resolve)
end

# kmax=12 gives ≈ 40_000 unknowns max — the historical bound this folder's sweeps
# have used (SparsePDE.jmd), chosen so 3-D KLU stays tractable.
function sweep(dim; kmax = 12)
    ns = [10 * 2^k for k in 0:kmax]
    tfirst = fill(NaN, length(ns), length(algs))
    tresolve = fill(NaN, length(ns), length(algs))
    sizes = zeros(Int, length(ns))
    for (i, N) in enumerate(ns)
        rng = MersenneTwister(123)
        A = fdmatrix(N; dim)
        n = size(A, 1)
        sizes[i] = n
        b = rand(rng, n)
        @info "dim=$dim: $n × $n, nnz=$(nnz(A))"
        for (j, (name, alg)) in enumerate(algs)
            try
                r = bench_alg(A, b, alg)
                tfirst[i, j] = r.first
                tresolve[i, j] = r.resolve
            catch e
                @warn "$(name) failed at n=$(n)" exception=(e,)
            end
        end
    end
    return (; sizes, tfirst, tresolve)
end

function plot_sweep(sizes, times, dim, phase)
    p = plot(;
        ylabel = "Time / s", xlabel = "N",
        yscale = :log10, xscale = :log10,
        title = "$(phase), $(dim)D FD matrix",
        legend = :outertopright)
    for j in 1:length(algs)
        mask = .!isnan.(times[:, j])
        any(mask) && plot!(p, sizes[mask], times[mask, j];
            linecolor = cols[j], marker = :circle, markersize = 2,
            label = algnames[j])
    end
    p
end


r1 = sweep(1)
plot_sweep(r1.sizes, r1.tfirst, 1, "Factor + first solve")


plot_sweep(r1.sizes, r1.tresolve, 1, "Cached re-solve")


r2 = sweep(2)
plot_sweep(r2.sizes, r2.tfirst, 2, "Factor + first solve")


plot_sweep(r2.sizes, r2.tresolve, 2, "Cached re-solve")


r3 = sweep(3)
plot_sweep(r3.sizes, r3.tfirst, 3, "Factor + first solve")


plot_sweep(r3.sizes, r3.tresolve, 3, "Cached re-solve")


using Printf
# Winner-per-regime table: for each dimension, which algorithm is fastest at the
# largest size, for each phase. A concrete recommendation, not just curves.
println("dim | phase           | fastest at N=max | time (s)")
println("----+-----------------+------------------+---------")
for (dim, r) in ((1, r1), (2, r2), (3, r3))
    for (phase, times) in (("first solve", r.tfirst), ("cached re-solve", r.tresolve))
        row = times[end, :]
        valid = findall(!isnan, row)
        isempty(valid) && continue
        j = valid[argmin(row[valid])]
        @printf("%3d | %-15s | %-16s | %.3g\n", dim, phase, algnames[j], row[j])
    end
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

