
using MPI            # provides mpiexec()
using Plots
using Printf

const WORKER = joinpath(@__DIR__, "run_solve.jl")
const PROJECT = Base.active_project()

# MPICH_jll 5.x hydra can fail PMI bootstrap on a single node; `-launcher fork`
# spawns ranks with fork() instead and is harmless where hydra is healthy.
const MPIEXEC_ARGS = `-launcher fork`

# Run run_solve.jl under `mpiexec -n P`; parse the CSV line rank 0 prints:
#   ranks,N,nnz,solver,pc,time_s,residual,iters,retcode
# One thread per rank so parallelism comes only from the rank count.
function run_ranks(P; N, solver = "cg", pc = "gamg")
    cmd = `$(mpiexec()) $(MPIEXEC_ARGS) -n $P $(Base.julia_cmd()) --project=$(PROJECT) $(WORKER) $N $solver $pc`
    out = read(addenv(cmd, "OMP_NUM_THREADS" => "1"), String)
    line = strip(last(filter(!isempty, split(out, '\n'))))
    f = split(line, ',')
    return (ranks = parse(Int, f[1]), N = parse(Int, f[2]), nnz = parse(Int, f[3]),
        time = parse(Float64, f[6]), residual = parse(Float64, f[7]),
        iters = parse(Int, f[8]), retcode = f[9])
end


const N_PER_RANK = 10_000
const RANKS = [1, 2, 4]

results = [run_ranks(P; N = N_PER_RANK * P) for P in RANKS]


t1 = results[1].time
weak_eff = [t1 / r.time for r in results]

println("ranks |    N    |  time (s)  | iters | weak eff. | residual  | retcode")
println("------+---------+------------+-------+-----------+-----------+--------")
for (r, e) in zip(results, weak_eff)
    @printf("%5d | %7d | %10.4g | %5d | %8.1f%% | %9.2e | %s\n",
        r.ranks, r.N, r.time, r.iters, 100 * e, r.residual, r.retcode)
end

# Auditability annotations, mirroring the strong-scaling document: iteration
# growth means the preconditioner is not algorithmically scaling (the time
# column then conflates solver work with parallel overhead), and efficiency
# above ~110% at these sizes usually reflects cache effects on a generic
# (JLL) PETSc build rather than real scaling.
itset = [r.iters for r in results]
if maximum(itset) - minimum(itset) > 0.25 * minimum(itset)
    @warn "GAMG iteration count grows >25% across the sweep — algorithmic " *
          "scalability is not holding at these sizes." iters = itset
end
if maximum(weak_eff) > 1.10
    @warn "Weak efficiency exceeds 110% — treat as a cache/working-set artifact " *
          "at small N, not a real result." weak_eff
end


p = plot(RANKS, 100 .* weak_eff;
    marker = :square, label = "PETSc CG + GAMG",
    xlabel = "MPI ranks (N = $(N_PER_RANK) × ranks)",
    ylabel = "weak efficiency (%)",
    title = "Weak scaling: constant work per rank",
    ylims = (0, 130), legend = :bottomleft)
hline!(p, [100]; linestyle = :dash, color = :gray, label = "ideal (100%)")
p


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

