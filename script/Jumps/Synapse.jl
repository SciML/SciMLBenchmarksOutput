
using Synapse
using PiecewiseDeterministicMarkovProcesses, JumpProcesses, OrdinaryDiffEq, Plots
using BenchmarkTools
fmt = :png


p_synapse = SynapseParams(t_end = 1000.0);
glu = 0.0;
events_sorted_times = [500.0];
is_pre_or_post_event = [true];
events_bap = events_sorted_times[is_pre_or_post_event.==false];
bap_by_epsp = Float64[];
nu = buildTransitionMatrix();


xc0 = initial_conditions_continuous_temp(p_synapse);
xd0 = initial_conditions_discrete(p_synapse);


solver = AutoTsit5(Rosenbrock23());
algorithms = [
    (
        label = "PDMP",
        agg = nothing,
        solver = (CHV(solver), CHV(solver)),
        saveat = [],
    ),
    (
        label = "Coevolve",
        agg = Coevolve(),
        solver = (solver, solver),
        saveat = 1 / p_synapse.sampling_rate,
    ),
];


results = []

for algo in algorithms
    push!(
        results,
        evolveSynapse(
            xc0,
            xd0,
            p_synapse,
            events_sorted_times,
            is_pre_or_post_event,
            bap_by_epsp,
            [true],
            nu,
            algo.solver,
            algo.agg;
            save_positions = (false, true),
            saveat = algo.saveat,
            save_everystep = false,
        ),
    )
end


fig = plot(xlabel = "Voltage", ylabel = "Time");
for (i, algo) in enumerate(algorithms)
    res = results[i]
    plot!(res.t, res.Vsp, label = algo.label)
end
title!("Vsp")


fig = plot(xlabel = "N", ylabel = "Time");
for (i, algo) in enumerate(algorithms)
    res = results[i]
    plot!(res.t, res.XD[1, :], label = algo.label)
end
title!("2line-Go, AMPA")


bs = Vector{BenchmarkTools.Trial}()

for algo in algorithms
    push!(
        bs,
        @benchmark(
            evolveSynapse(
                xc0,
                xd0,
                p_synapse,
                events_sorted_times,
                is_pre_or_post_event,
                bap_by_epsp,
                [true],
                nu,
                $(algo).solver,
                $(algo).agg;
                save_positions = (false, true),
                saveat = $(algo).saveat,
                save_everystep = false,
            ),
            samples = 50,
            evals = 1,
            seconds = 500,
        )
    )
end


labels = [a.label for a in algorithms]
medtimes = [text(string(round(median(b).time/1e9, digits=3),"s"), :center, 12) for b in bs]
relmedtimes = [median(b).time for b in bs]
relmedtimes ./= relmedtimes[1]
bar(labels, relmedtimes, markeralpha=0, series_annotation=medtimes, fmt=fmt)
title!("evolveSynapse (Median time)")


medmem = [text(string(round(median(b).memory/1e6, digits=3),"Mb"), :center, 12) for b in bs]
relmedmem = Float64[median(b).memory for b in bs]
relmedmem ./= relmedmem[1]
bar(labels, relmedmem, markeralpha=0, series_annotation=medmem, fmt=fmt)
title!("evolveSynapse (Median memory)")

