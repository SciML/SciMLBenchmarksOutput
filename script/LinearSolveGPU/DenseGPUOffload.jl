
using BenchmarkTools, Random, Printf
using LinearAlgebra, LinearSolve, RecursiveFactorization, MKL_jll
using CUDA

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

@assert CUDA.functional() "This benchmark requires a functional CUDA GPU"
println("GPU: ", CUDA.name(CUDA.device()))
println("CPU: ", Sys.cpu_info()[1].model, " — ", Sys.CPU_THREADS,
    " hardware threads, ", BLAS.get_num_threads(), " BLAS threads, ",
    Threads.nthreads(), " Julia threads")

# (name, algorithm, gate tolerance, device class); `nothing` is LinearSolve's
# default algorithm choice for the problem — the baseline a non-expert gets.
algs = Any[
    ("CPU OpenBLAS LU", LUFactorization(), 1e-10, :cpu),
    ("CPU RFLU", RFLUFactorization(), 1e-10, :cpu),
    ("CPU default choice", nothing, 1e-10, :cpu),
    ("GPU LU offload", CudaOffloadLUFactorization(), 1e-10, :gpu),
    ("GPU QR offload", CudaOffloadQRFactorization(), 1e-10, :gpu),
    ("GPU 32-mixed LU", CUDAOffload32MixedLUFactorization(), 1e-4, :gpu),
]

# LinearSolve deliberately does not load MKL on CPUs where it defaults away
# from it — notably AMD EPYC, this folder's runner CPU (LinearSolve.jl#518,
# the LoadMKL_JLL preference). Probe once and include the MKL row only where
# it actually runs, so the CPU baselines are exactly the options a user of
# this machine has; a NaN column with load errors is not a baseline.
mkl_works = try
    solve(LinearProblem(Matrix(8.0I, 8, 8), ones(8)), MKLLUFactorization())
    true
catch
    false
end
if mkl_works
    insert!(algs, 2, ("CPU MKL LU", MKLLUFactorization(), 1e-10, :cpu))
else
    println("MKL LU excluded: LinearSolve does not load MKL_jll on this CPU ",
        "by default (LoadMKL_JLL preference; see LinearSolve.jl#518).")
end
cpu_idx = findall(a -> a[4] == :cpu, algs)
gpu_idx = findall(a -> a[4] == :gpu, algs)

ns = [256, 512, 1024, 2048, 4096, 8192]


res_time = fill(NaN, length(ns), length(algs))
res_xfer = fill(NaN, length(ns))

for (i, n) in enumerate(ns)
    rng = MersenneTwister(123)
    A = rand(rng, n, n) + n * I
    b = rand(rng, n)
    ref = A \ b
    @info "n=$n"

    for (j, (name, alg, tol, _)) in enumerate(algs)
        try
            sol = solve(LinearProblem(A, b), alg)
            err = norm(sol.u - ref) / norm(ref)
            if !(err < tol)
                @warn "correctness gate failed — omitted" name n err
                continue
            end
            res_time[i, j] = @belapsed solve(LinearProblem($A, $b), $alg).u evals=1
        catch e
            @warn "$name failed at n=$n" exception=(e,)
        end
    end

    # Round-trip transfer for the same data, isolated.
    res_xfer[i] = @belapsed begin
        Ag = CuArray($A)
        bg = CuArray($b)
        CUDA.@sync Ag
        Array(bg)
    end evals=1
end


using Plots
p = plot(; xlabel = "N", ylabel = "time / s", xscale = :log2, yscale = :log10,
    title = "Dense solve: CPU vs GPU offload", legend = :topleft)
for (j, (name, _, _, class)) in enumerate(algs)
    mask = .!isnan.(res_time[:, j])
    any(mask) && plot!(p, ns[mask], res_time[mask, j];
        marker = class == :cpu ? :circle : :diamond, label = name)
end
plot!(p, ns, res_xfer; linestyle = :dash, color = :gray,
    label = "transfer round-trip only")
p


println("   N   | " * join([rpad(a[1], 18) for a in algs], "| ") * "| transfer (s)")
println("-"^(9 + 20 * length(algs) + 13))
for (i, n) in enumerate(ns)
    vals = join([@sprintf("%17.4g ", res_time[i, j]) for j in 1:length(algs)], "| ")
    @printf("%6d | %s| %12.4g\n", n, vals, res_xfer[i])
end

# Crossover: first size where the best GPU variant beats the BEST CPU option
# (not just one CPU algorithm — a crossover against a slow baseline is not a
# crossover). The best-CPU column names which algorithm set the bar.
best_cpu = [minimum(filter(!isnan, res_time[i, cpu_idx]); init = Inf) for i in 1:length(ns)]
best_gpu = [minimum(filter(!isnan, res_time[i, gpu_idx]); init = Inf) for i in 1:length(ns)]
println()
for (i, n) in enumerate(ns)
    j = cpu_idx[argmin(replace(res_time[i, cpu_idx], NaN => Inf))]
    @printf("%6d | best CPU: %-18s %.4g s | best GPU: %.4g s | GPU/CPU: %.2fx\n",
        n, algs[j][1], best_cpu[i], best_gpu[i], best_cpu[i] / best_gpu[i])
end
cross = findfirst(i -> best_gpu[i] < best_cpu[i], 1:length(ns))
println()
println(cross === nothing ?
    "No crossover in the measured range — the best CPU option wins throughout." :
    "Crossover: GPU offload first beats the best CPU option at N = $(ns[cross]).")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

