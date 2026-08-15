
using MPI            # provides mpiexec()

const WORKER = joinpath(@__DIR__, "run_solve.jl")
const PROJECT = Base.active_project()

# MPICH_jll 5.x's default hydra process manager fails to bootstrap PMI on a
# single node (PMI_Init returns -1 / "Broken pipe"). `-launcher fork` makes
# hydra spawn ranks with plain fork() instead, which works. Every mpiexec call
# in this benchmark folder routes through here so the flag lives in one place;
# if a future runner needs a different bootstrap, change it once.
const MPIEXEC_ARGS = `-launcher fork`

# Run run_solve.jl under `mpiexec -n P` and return the single CSV line rank 0 prints:
#   ranks,N,nnz,solver,pc,time_s,residual,retcode
function run_ranks(P; N = 10_000, solver = "gmres", pc = "none")
    cmd = `$(mpiexec()) $(MPIEXEC_ARGS) -n $P $(Base.julia_cmd()) --project=$(PROJECT) $(WORKER) $N $solver $pc`
    out = read(cmd, String)
    line = strip(last(filter(!isempty, split(out, '\n'))))
    return line
end


rows = String[]
for P in (1, 2)
    push!(rows, run_ranks(P; N = 10_000, solver = "gmres", pc = "jacobi"))
end
rows


using Printf
println("ranks |     N   |   nnz   | solver | pc     |  time (s)  |  residual  | retcode")
println("------+---------+---------+--------+--------+------------+------------+---------")
for r in rows
    ranks, N, nnz, solver, pc, t, res, rc = split(r, ',')
    @printf("%5s | %7s | %7s | %-6s | %-6s | %10.4g | %10.2e | %s\n",
            ranks, N, nnz, solver, pc, parse(Float64, t), parse(Float64, res), rc)
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

