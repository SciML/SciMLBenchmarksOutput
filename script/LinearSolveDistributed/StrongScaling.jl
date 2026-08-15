
using MPI            # provides mpiexec()
using Plots

const WORKER = joinpath(@__DIR__, "run_solve.jl")
const PROJECT = Base.active_project()

# MPICH_jll 5.x hydra fails to bootstrap PMI on a single node; `-launcher fork`
# makes it spawn ranks with fork() instead. One place, reused by every mpiexec call.
const MPIEXEC_ARGS = `-launcher fork`

# Run run_solve.jl under `mpiexec -n P`, return the CSV line rank 0 prints:
#   ranks,N,nnz,solver,pc,time_s,residual,iters,retcode
# OMP_NUM_THREADS=1 forces one thread per rank so parallelism comes only from the
# rank count — otherwise the -n 1 baseline oversubscribes all cores and fakes
# superlinear speedup (see run_solve.jl for the full rationale).
function run_ranks(P; N, solver = "cg", pc = "none")
    cmd = `$(mpiexec()) $(MPIEXEC_ARGS) -n $P $(Base.julia_cmd()) --project=$(PROJECT) $(WORKER) $N $solver $pc`
    out = read(addenv(cmd, "OMP_NUM_THREADS" => "1"), String)
    line = strip(last(filter(!isempty, split(out, '\n'))))
    fields = split(line, ',')
    return (ranks = parse(Int, fields[1]),
            N = parse(Int, fields[2]),
            nnz = parse(Int, fields[3]),
            time = parse(Float64, fields[6]),
            residual = parse(Float64, fields[7]),
            iters = parse(Int, fields[8]),
            retcode = fields[9])
end


const N = 40_000           # ~200×200 grid; deliberately modest for the first runs
const RANKS = [1, 2, 4]    # capped: the replicated-matrix path holds a full copy
                           # per rank, so memory grows with the rank count.
                           # Raise both once the pipeline is proven on the runner.

results = [run_ranks(P; N = N, solver = "cg", pc = "gamg") for P in RANKS]


t1 = results[1].time
speedup    = [t1 / r.time for r in results]
efficiency = [t1 / (r.ranks * r.time) for r in results]

using Printf
println("ranks |  time (s)  | iters | speedup | efficiency | residual  | retcode")
println("------+------------+-------+---------+------------+-----------+--------")
for (r, s, e) in zip(results, speedup, efficiency)
    @printf("%5d | %10.4g | %5d | %7.2f | %9.1f%% | %9.2e | %s\n",
            r.ranks, r.time, r.iters, s, 100 * e, r.residual, r.retcode)
end

# Sanity annotations (informational, not failures). Two things worth surfacing on
# every run so a reader can judge the numbers rather than trust them blindly:
#
#  * iters spread: GAMG's aggregation is partition-dependent, so the iteration
#    count can drift a little with rank count. A *small* spread is expected; a
#    large one means the preconditioner strength is changing with P and the
#    timing ratios are contaminated by algorithm change, not just parallel work.
#  * superlinear efficiency: with the generic (JLL) PETSc build, small
#    problems can show >100% efficiency from cache/working-set effects (each
#    rank's slice fits in a faster level of cache). It is a real effect but NOT a
#    marketing number — the honest scaling curve needs large N on optimized PETSc
#    (dedicated benchmark hardware), where compute dominates these constant factors.
itset = [r.iters for r in results]
iters_spread = maximum(itset) - minimum(itset)
super = maximum(efficiency) > 1.10
@info "iteration counts across ranks" iters = itset spread = iters_spread
if iters_spread > 0.25 * minimum(itset)
    @warn "GAMG iteration count varies >25% across ranks — preconditioner strength is " *
          "partition-dependent here; treat speedup as approximate." iters = itset
end
if super
    @warn "Superlinear efficiency (>110%) — expected for small N on non-optimized PETSc " *
          "(cache effects). Reproduce at large N on optimized PETSc before quoting." efficiency
end


p1 = plot(RANKS, speedup;
    marker = :circle, label = "PETSc CG + GAMG",
    xlabel = "MPI ranks", ylabel = "speedup (T₁ / T_P)",
    title = "Strong scaling: speedup (N = $N)", legend = :topleft)
plot!(p1, RANKS, RANKS; linestyle = :dash, color = :gray, label = "ideal (linear)")
p1


p2 = plot(RANKS, 100 .* efficiency;
    marker = :square, label = "PETSc CG + GAMG",
    xlabel = "MPI ranks", ylabel = "parallel efficiency (%)",
    title = "Strong scaling: efficiency (N = $N)",
    ylims = (0, 130), legend = :bottomleft)
hline!(p2, [100]; linestyle = :dash, color = :gray, label = "ideal (100%)")
p2


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

