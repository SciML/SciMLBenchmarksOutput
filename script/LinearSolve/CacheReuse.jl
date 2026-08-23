
using BenchmarkTools, Random
using LinearAlgebra, SparseArrays, LinearSolve, Sparspak
using RecursiveFactorization, FastLapackInterface
import Pardiso
import ParU_jll
using Plots

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

# Same FD generator as the folder's other sparse documents.
A ⊕ B = kron(I(size(B, 1)), A) + kron(B, I(size(A, 1)))
function lattice(n; Tv = Float64)
    d = fill(2 * one(Tv), n); d[1] = one(Tv); d[end] = one(Tv)
    spdiagm(1 => -ones(Tv, n - 1), 0 => d, -1 => -ones(Tv, n - 1))
end
lattice(L...; Tv = Float64) = lattice(L[1]; Tv) ⊕ lattice(L[2:end]...; Tv)
function fdmatrix(N; dim = 2, Tv = Float64, δ = 1.0e-2)
    n = N^(1 / dim) |> ceil |> Int
    lattice([n for i in 1:dim]...; Tv) + Tv(δ) * I
end

# (family, name, alg). Dense algs get a dense A; sparse algs the 2-D FD matrix.
algs = [
    (:dense,  "LU",           LUFactorization()),
    (:dense,  "RFLU",         RFLUFactorization()),
    (:dense,  "MKL LU",       MKLLUFactorization()),
    (:sparse, "UMFPACK",      UMFPACKFactorization()),
    (:sparse, "KLU",          KLUFactorization()),
    (:sparse, "SupernodalLU", SupernodalLUFactorization()),
    (:sparse, "Sparspak",     SparspakFactorization()),
    (:sparse, "ParU",         ParUFactorization()),
]

const DENSE_N = 500
const SPARSE_N = 40_000    # ~200×200 grid, 2-D

function make_problem(family)
    rng = MersenneTwister(123)
    A = family === :dense ? (rand(rng, DENSE_N, DENSE_N) + DENSE_N * I) :
        fdmatrix(SPARSE_N; dim = 2)
    n = size(A, 1)
    b = rand(rng, n)
    # Same-sparsity update: scale values, keep the pattern (dense: fresh matrix).
    A2 = family === :dense ? (rand(rng, n, n) + n * I) :
        SparseMatrixCSC(n, n, copy(A.colptr), copy(A.rowval), 1.1 .* A.nzval)
    b2 = rand(rng, n)
    return A, A2, b, b2
end


function bench_reuse(family, alg)
    A, A2, b, b2 = make_problem(family)

    # Correctness gate on both the fresh solve and the A-update path — a cache
    # that silently returns the OLD factorization's answer after `cache.A = A2`
    # would be fast and wrong, which is the failure mode this guards against.
    # NOTE: `cache.A = X` hands the backend the array itself, and several
    # factorizations (RFLU, MKL) factorize it IN PLACE — the caller's matrix is
    # destroyed. Others (LU) copy into an internal workspace. Always assign a copy
    # if you still need the matrix; this benchmark does, so its residual checks
    # test the solve rather than the wreckage.
    cache = init(LinearProblem(copy(A), b), alg)
    u1 = copy(solve!(cache).u)
    cache.b = b2
    u2 = copy(solve!(cache).u)
    cache.A = copy(A2)
    u3 = copy(solve!(cache).u)
    r1 = norm(A * u1 - b) / norm(b)
    r2 = norm(A * u2 - b2) / norm(b2)
    r3 = norm(A2 * u3 - b2) / norm(b2)
    if !(r1 < 1e-8 && r2 < 1e-8 && r3 < 1e-8)
        @warn "correctness gate failed" alg r1 r2 r3
        return nothing
    end

    t_naive = @belapsed solve(LinearProblem($A, $b), $alg).u

    t_newb = @belapsed solve!(c).u setup=(
        c = init(LinearProblem($A, $b), $alg); solve!(c); c.b = $b2) evals=1

    t_newA = @belapsed solve!(c).u setup=(
        c = init(LinearProblem($A, $b), $alg); solve!(c); c.A = copy($A2)) evals=1

    # Steady-state allocations: repeated solve! on a warm cache, same b.
    warm = init(LinearProblem(A, b), alg)
    solve!(warm); solve!(warm)
    allocs = @allocated solve!(warm)

    return (; t_naive, t_newb, t_newA, allocs)
end

results = []
for (family, name, alg) in algs
    r = try
        bench_reuse(family, alg)
    catch e
        @warn "$name failed" exception=(e,)
        nothing
    end
    r === nothing || push!(results, (; family, name, r...))
    r === nothing || @info name t_naive=r.t_naive t_newb=r.t_newb t_newA=r.t_newA allocs=r.allocs
end


using Printf
println("alg           | family |  naive (s) | new-b (s) | new-A (s) | b-speedup | A-speedup | allocs/solve")
println("--------------+--------+------------+-----------+-----------+-----------+-----------+-------------")
for r in results
    @printf("%-13s | %-6s | %10.3g | %9.3g | %9.3g | %8.1fx | %8.2fx | %d\n",
        r.name, r.family, r.t_naive, r.t_newb, r.t_newA,
        r.t_naive / r.t_newb, r.t_naive / r.t_newA, r.allocs)
end


sparse_res = filter(r -> r.family === :sparse, results)
vals = hcat([ [r.t_naive for r in sparse_res],
              [r.t_newA  for r in sparse_res],
              [r.t_newb  for r in sparse_res] ]...)
p = plot()   # grouped bars via repeated bar! calls keep deps minimal
xs = 1:length(sparse_res)
w = 0.25
for (k, (lab, col)) in enumerate(zip(("naive", "new A (refactor)", "new b (backsolve)"),
                                     (:gray, :steelblue, :seagreen)))
    bar!(p, xs .+ (k - 2) * w, vals[:, k]; bar_width = w, label = lab, color = col)
end
plot!(p; xticks = (xs, [r.name for r in sparse_res]), yscale = :log10,
    ylabel = "time / s (log)", title = "Sparse (N = $(SPARSE_N)): cost per solve by reuse mode",
    legend = :topright)
p


A = fdmatrix(SPARSE_N; dim = 2)
n = size(A, 1)
rng = MersenneTwister(42)
b = rand(rng, n)
db = 0.01 .* rand(rng, n)   # small perturbation: the "time-stepping" regime

for (label, ws) in (("WarmStart.Previous", KrylovJL_GMRES(warm_start = WarmStart.Previous)),
                    ("WarmStart.None", KrylovJL_GMRES(warm_start = WarmStart.None)))
    cache = init(LinearProblem(A, b), ws; reltol = 1e-8)
    iters = Int[]
    for step in 1:5
        sol = solve!(cache)
        push!(iters, sol.iters)
        cache.b = cache.b .+ db
    end
    println(rpad(label, 18), " iters per step: ", iters)
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

