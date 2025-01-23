
using JumpProcesses, Plots, StableRNGs, BenchmarkTools, ReactionNetworkImporters, StatsPlots, Catalyst


tf = 12.
rng = StableRNG(53124)
algs = [NRM(), CCNRM(), DirectCR(), RSSACR()]
egfr_net = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Data/egfr_net.net"));
dprob = DiscreteProblem(complete(egfr_net.rn), egfr_net.u0, (0., tf), egfr_net.p)
dprob = remake(dprob,u0=Int64.(dprob.u0))

plt = plot(title="Dimer concentrations")
for alg in algs
    jprob = JumpProblem(complete(egfr_net.rn), dprob, alg)
    sol = solve(jprob, SSAStepper(), saveat = tf/200)
    plot!(plt, sol, idxs = :Dimers, label="$alg")
end
plot!(plt)


function benchmark_and_bar_plot(model, end_time, algs)
    times = Vector{Float64}()
    alg_names = ["$s"[15:end-2] for s in algs]

    benchmarks = Vector{BenchmarkTools.Trial}(undef, length(algs))
    for (i, alg) in enumerate(algs)
        alg_name = alg_names[i]
        println("Benchmarking $alg_name")
        dprob = DiscreteProblem(complete(model.rn), model.u0, (0., end_time), model.p)
        dprob = remake(dprob,u0 = Int64.(dprob.u0))
        jprob = JumpProblem(complete(model.rn), dprob, alg; rng, save_positions = (false, false))

        b = @benchmarkable solve($jprob; saveat = $end_time) samples = 5 seconds = 7200
        bm = run(b)
        push!(times, median(bm).time/1e9)
    end

    bar(alg_names, times, xlabel = "Algorithm", ylabel = "Average Time (s)", title = "SSA Runtime for EGFR network", legend = false)
end


tf = 12. 
rng = StableRNG(53124)
algs = [NRM(), CCNRM(), DirectCR(), RSSACR()]

plt = benchmark_and_bar_plot(egfr_net, tf, algs)
plt


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

