
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


const RANKS = [1, 2, 4, 8]
const N_SMALL = 40_000       # cache-artifact regime, kept deliberately (see text)
const N_LARGE = 1_000_000    # honest regime: working set exceeds cache everywhere

res_small = [run_ranks(P; N = N_SMALL, solver = "cg", pc = "gamg") for P in RANKS]
res_large = [run_ranks(P; N = N_LARGE, solver = "cg", pc = "gamg") for P in RANKS]


using Printf

function report(results, N)
    t1 = results[1].time
    speedup = [t1 / r.time for r in results]
    efficiency = [t1 / (r.ranks * r.time) for r in results]
    println("N = $N")
    println("ranks |  time (s)  | iters | speedup | efficiency | residual  | retcode")
    println("------+------------+-------+---------+------------+-----------+--------")
    for (r, s, e) in zip(results, speedup, efficiency)
        @printf("%5d | %10.4g | %5d | %7.2f | %9.1f%% | %9.2e | %s\n",
            r.ranks, r.time, r.iters, s, 100 * e, r.residual, r.retcode)
    end
    # Auditability annotations (informational, not failures):
    #  * iters spread: GAMG's aggregation is partition-dependent, so iteration
    #    counts can drift a little with rank count; a large spread means the
    #    preconditioner strength is changing with P and the ratios are
    #    contaminated by algorithm change, not just parallel work.
    #  * superlinear efficiency: expected for the small-N series (cache and
    #    working-set effects on a generic JLL PETSc build); if the LARGE series
    #    trips it too, the honest regime has not been reached yet.
    itset = [r.iters for r in results]
    if maximum(itset) - minimum(itset) > 0.25 * minimum(itset)
        @warn "GAMG iteration count varies >25% across ranks at N=$N" iters = itset
    end
    if maximum(efficiency) > 1.10
        @warn "Superlinear efficiency at N=$N — cache/working-set regime, not a marketing number." efficiency
    end
    return speedup, efficiency
end

sp_small, eff_small = report(res_small, N_SMALL)
println()
sp_large, eff_large = report(res_large, N_LARGE)


p1 = plot(RANKS, sp_small;
    marker = :circle, label = "N = $(N_SMALL) (cache-artifact regime)",
    xlabel = "MPI ranks", ylabel = "speedup (T₁ / T_P)",
    title = "Strong scaling: speedup", legend = :topleft)
plot!(p1, RANKS, sp_large; marker = :diamond, linewidth = 2,
    label = "N = $(N_LARGE)")
plot!(p1, RANKS, RANKS; linestyle = :dash, color = :gray, label = "ideal (linear)")
p1


p2 = plot(RANKS, 100 .* eff_small;
    marker = :circle, label = "N = $(N_SMALL) (cache-artifact regime)",
    xlabel = "MPI ranks", ylabel = "parallel efficiency (%)",
    title = "Strong scaling: efficiency", legend = :topleft)
plot!(p2, RANKS, 100 .* eff_large; marker = :diamond, linewidth = 2,
    label = "N = $(N_LARGE)")
hline!(p2, [100]; linestyle = :dash, color = :gray, label = "ideal (100%)")
p2


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

