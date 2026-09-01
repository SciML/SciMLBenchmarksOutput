using JumpProcesses, Plots, StableRNGs, BenchmarkTools, ReactionNetworkImporters,
    StatsPlots, Catalyst


tf = 12.0
rng = StableRNG(53124)
algs = [NRM(), CCNRM(), DirectCR(), RSSACR()]
egfr_net = complete(loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Data/egfr_net.net")))
egfr_u0 = Catalyst.get_u0_map(egfr_net)
egfr_parameters = Catalyst.get_parameter_map(egfr_net)

plt = plot(title = "Dimer concentrations")
for alg in algs
    jprob = JumpProblem(
        egfr_net, egfr_u0, (0.0, tf), egfr_parameters;
        aggregator = alg, u0_eltype = Int64
    )
    sol = solve(jprob, SSAStepper(), saveat = tf / 200)
    plot!(plt, sol, idxs = :Dimers, label = "$alg")
end
plot!(plt)


function benchmark_and_bar_plot(model, end_time, algs)
    times = Vector{Float64}()
    alg_names = ["$s"[15:(end - 2)] for s in algs]
    u0 = Catalyst.get_u0_map(model)
    parameters = Catalyst.get_parameter_map(model)

    benchmarks = Vector{BenchmarkTools.Trial}(undef, length(algs))
    for (i, alg) in enumerate(algs)
        alg_name = alg_names[i]
        println("Benchmarking $alg_name")
        jprob = JumpProblem(
            model, u0, (0.0, end_time), parameters;
            aggregator = alg, rng, save_positions = (false, false), u0_eltype = Int64
        )

        b = @benchmarkable solve($jprob; saveat = $end_time) samples = 5 seconds = 7200
        bm = run(b)
        push!(times, median(bm).time / 1.0e9)
    end

    return bar(
        alg_names, times, xlabel = "Algorithm", ylabel = "Average Time (s)",
        title = "SSA Runtime for EGFR network", legend = false
    )
end


tf = 12.0
rng = StableRNG(53124)
algs = [NRM(), CCNRM(), DirectCR(), RSSACR()]

plt = benchmark_and_bar_plot(egfr_net, tf, algs)
plt


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
