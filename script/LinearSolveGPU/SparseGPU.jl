
using BenchmarkTools, Random, Printf
using LinearAlgebra, SparseArrays, LinearSolve
using CUDA, CUDSS

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

@assert CUDA.functional() "This benchmark requires a functional CUDA GPU"
println("GPU: ", CUDA.name(CUDA.device()))

# 2-D FD Laplacian (same generator family as the CPU sparse documents).
lap1d(k) = spdiagm(1 => -ones(k - 1), 0 => fill(2.0, k), -1 => -ones(k - 1))
function fd2d(m)
    kron(I(m), lap1d(m)) + kron(lap1d(m), I(m)) + 0.01I
end

ms = [50, 100, 200, 320, 450]          # grid sides → N = 2.5e3 … 2.0e5


function bench_cpu(A, b, alg, ref)
    cache = init(LinearProblem(A, b), alg)
    sol = solve!(cache)
    err = norm(sol.u - ref) / norm(ref)
    err < 1e-8 || return nothing
    t_first = @belapsed solve!(c) setup=(
        c = init(LinearProblem($A, $b), $alg)) evals=1
    t_re = @belapsed solve!($cache)
    return (; t_first, t_re)
end

function bench_cudss(A, b, ref)
    Ag = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
    bg = CuArray(b)
    cache = init(LinearProblem(Ag, bg), LUFactorization())
    sol = solve!(cache)
    err = norm(Array(sol.u) - ref) / norm(ref)
    err < 1e-8 || return nothing
    t_first = @belapsed CUDA.@sync(solve!(c)) setup=(
        c = init(LinearProblem($Ag, $bg), LUFactorization())) evals=1
    t_re = @belapsed CUDA.@sync(solve!($cache))
    t_h2d = @belapsed begin
        G = CUDA.CUSPARSE.CuSparseMatrixCSR($A)
        CUDA.@sync G
    end evals=1
    return (; t_first, t_re, t_h2d)
end

rows = []
for m in ms
    A = fd2d(m); n = size(A, 1)
    rng = MersenneTwister(123)
    b = rand(rng, n)
    ref = A \ b
    @info "grid $m×$m → n=$n, nnz=$(nnz(A))"
    umf = bench_cpu(A, b, UMFPACKFactorization(), ref)
    klu = bench_cpu(A, b, KLUFactorization(), ref)
    gpu = bench_cudss(A, b, ref)
    push!(rows, (; n, umf, klu, gpu))
end


using Plots
p = plot(; xlabel = "N", ylabel = "time / s", xscale = :log10, yscale = :log10,
    title = "Sparse factor+solve: CPU vs CUDSS", legend = :topleft)
for (label, f) in (("UMFPACK", r -> r.umf), ("KLU", r -> r.klu), ("CUDSS (GPU)", r -> r.gpu))
    xs = [r.n for r in rows if f(r) !== nothing]
    ys = [f(r).t_first for r in rows if f(r) !== nothing]
    isempty(xs) || plot!(p, xs, ys; marker = :circle, label = label)
end
gx = [r.n for r in rows if r.gpu !== nothing]
plot!(p, gx, [r.gpu.t_h2d for r in rows if r.gpu !== nothing];
    linestyle = :dash, color = :gray, label = "CSR H2D transfer only")
p


println("    N   | UMF first | KLU first | GPU first | UMF re   | KLU re   | GPU re   | H2D")
println("--------+-----------+-----------+-----------+----------+----------+----------+--------")
for r in rows
    f(x, fld) = x === nothing ? NaN : getfield(x, fld)
    @printf("%7d | %9.3g | %9.3g | %9.3g | %8.3g | %8.3g | %8.3g | %.3g\n",
        r.n, f(r.umf, :t_first), f(r.klu, :t_first), f(r.gpu, :t_first),
        f(r.umf, :t_re), f(r.klu, :t_re), f(r.gpu, :t_re), f(r.gpu, :t_h2d))
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

