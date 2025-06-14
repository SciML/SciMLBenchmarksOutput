
using BlackBoxOptimizationBenchmarking, Plots, Optimization, Memoize, Statistics
import BlackBoxOptimizationBenchmarking.Chain
const BBOB = BlackBoxOptimizationBenchmarking

using OptimizationBBO, OptimizationOptimJL, OptimizationEvolutionary, OptimizationNLopt
using OptimizationMetaheuristics, OptimizationNOMAD, OptimizationPRIMA, OptimizationOptimisers


chain = (t; isboxed=false) -> Chain(
    BenchmarkSetup(t, isboxed = isboxed),
    BenchmarkSetup(NelderMead(), isboxed = false),
    0.9
)

test_functions = BBOB.list_functions()
dimension = 3
run_length = round.(Int, 10 .^ LinRange(1,5,30))

@memoize run_bench(algo) = BBOB.benchmark(setup[algo], test_functions, run_length, Ntrials=40, dimension = dimension)


setup = Dict(
    "NelderMead" => NelderMead(),
    #Optim.BFGS(),
    #"NLopt.GN_MLSL_LDS" => chain(NLopt.GN_MLSL_LDS(), isboxed=true), # gives me errors
    "NLopt.GN_CRS2_LM()" => chain(NLopt.GN_CRS2_LM(), isboxed=true),
    "NLopt.GN_DIRECT()" => chain(NLopt.GN_DIRECT(), isboxed=true),
    "NLopt.GN_ESCH()"  => chain(NLopt.GN_ESCH(), isboxed=true),
    "OptimizationEvolutionary.GA()" => chain(OptimizationEvolutionary.GA(), isboxed=true),
    "OptimizationEvolutionary.DE()" => chain(OptimizationEvolutionary.DE(), isboxed=true),
    "OptimizationEvolutionary.ES()" => chain(OptimizationEvolutionary.ES(), isboxed=true),
    "Optim.SAMIN" => chain(SAMIN(verbosity=0), isboxed=true),
    "BBO_adaptive_de_rand_1_bin" => chain(BBO_adaptive_de_rand_1_bin(), isboxed=true),
    "BBO_adaptive_de_rand_1_bin_radiuslimited" => chain(BBO_adaptive_de_rand_1_bin_radiuslimited(), isboxed=true), # same as BBO_adaptive_de_rand_1_bin
    "BBO_separable_nes" => chain(BBO_separable_nes(), isboxed=true),
    "BBO_de_rand_2_bin" => chain(BBO_de_rand_2_bin(), isboxed=true),
    #"BBO_xnes" => chain(BBO_xnes(), isboxed=true), # good but slow
    #"BBO_dxnes" => chain(BBO_dxnes(), isboxed=true), 
    "OptimizationMetaheuristics.ECA" => chain(OptimizationMetaheuristics.ECA(), isboxed=true),
    #"OptimizationMetaheuristics.CGSA" => () -> chain(OptimizationMetaheuristics.CGSA(), isboxed=true), #give me strange results
    "OptimizationMetaheuristics.DE" => chain(OptimizationMetaheuristics.DE(), isboxed=true),
    "Optimisers.AdamW" => chain(Optimisers.AdamW(), isboxed=false),
    "Optimisers.RMSProp" => chain(Optimisers.RMSProp(), isboxed=false),
    # "NOMADOpt" => chain(NOMADOpt()), too much printing
    # "OptimizationPRIMA.UOBYQA()" => chain(OptimizationPRIMA.UOBYQA()), :StackOverflowError?
    # "OptimizationPRIMA.NEWUOA()" => OptimizationPRIMA.UOBYQA(),
    #
)


@time b = BBOB.benchmark(
    chain(OptimizationMetaheuristics.CGSA(), isboxed=true),
    test_functions[1:10], 100:500:10_000, Ntrials=10, dimension = 3
)

plot(b)


Δf = 1e-6
f = test_functions[3]

single_setup = BenchmarkSetup(NLopt.GN_CRS2_LM(), isboxed=true)

sol = [BBOB.solve_problem(single_setup, f, 3, 5_000) for in in 1:10]
@info [sol.objective < Δf + f.f_opt for sol in sol]

p = plot(f, size = (600,600), zoom = 1.5)
for sol in sol
    scatter!(sol.u[1:1], sol.u[2:2], label="", c="blue", marker = :xcross, markersize=5, markerstrokewidth=0)
end
p


results = Array{BBOB.BenchmarkResults}(undef, length(setup))

Threads.@threads for (i,algo) in collect(enumerate(keys(setup)))
    results[i] = run_bench(algo)
end

results


labels = collect(keys(setup))
idx = sortperm([b.success_rate[end] for b in results], rev=true)

p = plot(xscale = :log10, legend = :outerright, size = (700,350), margin=10Plots.px, dpi=200)
for i in idx
    plot!(results[i], label = labels[i], showribbon=false, lw=2.5, xlim = (1,1e5), x = :run_length)
end
p


success_rate_per_function = reduce(hcat, b.success_rate_per_function for b in results)

idx = sortperm(mean(success_rate_per_function, dims=1)[:], rev=false)
idxfunc = sortperm(mean(success_rate_per_function, dims=2)[:], rev=true)
idxfunc = 1:length(test_functions)

p = heatmap(
    string.(test_functions)[idxfunc], labels[idx], success_rate_per_function[idxfunc, idx]',
    cmap = :RdYlGn,
    xticks = :all,
    yticks = :all,
    xrotation = 45,
    dpi = 200,
)


labels = collect(keys(setup))
idx = sortperm([b.distance_to_minimizer[end] for b in results], rev=false)

p = plot(xscale = :log10, legend = :outerright, size = (900,500), margin=10Plots.px, ylim = (0,5))
for i in idx
    plot!(
        results[i].run_length, results[i].distance_to_minimizer, label = labels[i],
        showribbon=false, lw=2, xlim = (1,1e5),
        xlabel = "Iterations", ylabel = "Mean distance to minimum"
    )
end
p


ref = findfirst("NelderMead" .== labels)
runtimes = getfield.(results, :runtime)
runtimes = runtimes ./ runtimes[ref]

bar(
    labels, runtimes, xrotation = :45, xticks = :all, ylabel = "Run time relative to NM",
    yscale = :log10, yticks = [0.1,1,10,100],
    legend = false, margin = 25Plots.px
)

