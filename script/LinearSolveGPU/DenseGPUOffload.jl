
using BenchmarkTools, Random, Printf
using LinearAlgebra, LinearSolve
using CUDA

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

@assert CUDA.functional() "This benchmark requires a functional CUDA GPU"
println("GPU: ", CUDA.name(CUDA.device()))

algs = [
    ("CPU LU", LUFactorization(), 1e-10),
    ("GPU LU offload", CudaOffloadLUFactorization(), 1e-10),
    ("GPU QR offload", CudaOffloadQRFactorization(), 1e-10),
    ("GPU 32-mixed LU", CUDAOffload32MixedLUFactorization(), 1e-4),
]

ns = [256, 512, 1024, 2048, 4096, 8192]


res_time = fill(NaN, length(ns), length(algs))
res_xfer = fill(NaN, length(ns))

for (i, n) in enumerate(ns)
    rng = MersenneTwister(123)
    A = rand(rng, n, n) + n * I
    b = rand(rng, n)
    ref = A \ b
    @info "n=$n"

    for (j, (name, alg, tol)) in enumerate(algs)
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
for (j, (name, _, _)) in enumerate(algs)
    mask = .!isnan.(res_time[:, j])
    any(mask) && plot!(p, ns[mask], res_time[mask, j];
        marker = :circle, label = name)
end
plot!(p, ns, res_xfer; linestyle = :dash, color = :gray,
    label = "transfer round-trip only")
p


println("   N   | CPU LU (s) | GPU LU (s) | 32-mixed (s) | transfer (s) | transfer % of GPU LU")
println("-------+------------+------------+--------------+--------------+---------------------")
for (i, n) in enumerate(ns)
    tcpu, tgpu, tmix = res_time[i, 1], res_time[i, 2], res_time[i, 4]
    @printf("%6d | %10.4g | %10.4g | %12.4g | %12.4g | %18.1f%%\n",
        n, tcpu, tgpu, tmix, res_xfer[i],
        isnan(tgpu) ? NaN : 100 * res_xfer[i] / tgpu)
end

# Crossover: first size where the best GPU variant beats CPU.
best_gpu = [minimum(filter(!isnan, res_time[i, 2:end]); init = Inf) for i in 1:length(ns)]
cross = findfirst(i -> best_gpu[i] < res_time[i, 1], 1:length(ns))
println()
println(cross === nothing ?
    "No crossover in the measured range — CPU wins throughout." :
    "Crossover: GPU offload first beats CPU at N = $(ns[cross]).")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

