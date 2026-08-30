
using DiffEqBayes, BenchmarkTools


using OrdinaryDiffEq, RecursiveArrayTools, Distributions, ParameterizedFunctions,
      StanSample, DynamicHMC
using Plots, StaticArrays, Turing, LinearAlgebra


"""Display ESS/s (effective samples per second) from a Turing chain."""
function display_ess_per_sec(chain, elapsed)
    stats = summarystats(chain)
    ess_bulk = stats[:, :ess_bulk]
    println("Elapsed time: $(round(elapsed; digits=2)) seconds\n")
    println("ESS/s (effective samples per second, bulk):")
    for (i, param) in enumerate(stats[:, :parameters])
        println("  $param: $(round(ess_bulk[i] / elapsed; digits=1))")
    end
    println("\nMinimum ESS/s (bulk): $(round(minimum(ess_bulk) / elapsed; digits=1))")
end

"""Extract and display Stan's internal timing from its CSV output files."""
function display_stan_timing(stan_result)
    sample_files = stan_result.model.sample_file
    for (chain_idx, f) in enumerate(sample_files)
        isfile(f) || continue
        lines = readlines(f)
        println("Chain $chain_idx timing (from Stan CSV):")
        for line in lines
            if startswith(line, "#") && occursin("Elapsed Time", line)
                println("  ", strip(line[2:end]))
            elseif startswith(line, "#") && occursin("seconds", line)
                println("  ", strip(line[2:end]))
            end
        end
    end
end


gr(fmt = :png)


fitz = @ode_def FitzhughNagumo begin
    dv = v - 0.33*v^3 - w + l
    dw = τinv*(v + a - b*w)
end a b τinv l


prob_ode_fitzhughnagumo = ODEProblem(fitz, [1.0, 1.0], (0.0, 10.0), [0.7, 0.8, 1/12.5, 0.5])
sol = solve(prob_ode_fitzhughnagumo, Tsit5())


sprob_ode_fitzhughnagumo = ODEProblem{false, SciMLBase.FullSpecialize}(
    fitz, SA[1.0, 1.0], (0.0, 10.0), SA[0.7, 0.8, 1 / 12.5, 0.5])
sol = solve(sprob_ode_fitzhughnagumo, Tsit5())


t = collect(range(1, stop = 10, length = 10))
sig = 0.20
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(2)) for i in 1:length(t)]))


scatter(t, data[1, :])
scatter!(t, data[2, :])
plot!(sol)


priors = [truncated(Normal(1.0, 0.5), 0, 1.5), truncated(Normal(1.0, 0.5), 0, 1.5),
    truncated(Normal(0.0, 0.5), 0.0, 0.5), truncated(Normal(0.5, 0.5), 0, 1)]


bayesian_result_stan = @time stan_inference(
    prob_ode_fitzhughnagumo, :rk45, t, data, priors;
    print_summary = false,
    sample_kwargs = Dict(:delta => 0.85, :num_samples => 10_000),
    vars = (DiffEqBayes.StanODEData(), InverseGamma(2, 3)))


display_stan_timing(bayesian_result_stan)


@model function fitfhn(data, prob)
    # Prior distributions.
    σ ~ filldist(InverseGamma(2, 3), 2)
    a ~ truncated(Normal(1.0, 0.5), 0, 1.5)
    b ~ truncated(Normal(1.0, 0.5), 0, 1.5)
    τinv ~ truncated(Normal(0.0, 0.5), 0.0, 0.5)
    l ~ truncated(Normal(0.5, 0.5), 0, 1)

    # Simulate FitzHugh-Nagumo model.
    p = SA[a, b, τinv, l]
    _prob = remake(prob, p = p)
    predicted = solve(_prob, Tsit5(); saveat = t)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], Diagonal(σ .^ 2))
    end

    return nothing
end

model = fitfhn(data, sprob_ode_fitzhughnagumo)

# Warmup run to compile all code paths before timing
sample(model, Turing.NUTS(0.85), 10; progress = false)

elapsed_turing_direct = @elapsed chain = sample(model, Turing.NUTS(0.85), 10_000; progress = false)
chain


display_ess_per_sec(chain, elapsed_turing_direct)


@btime bayesian_result_turing = turing_inference(
    prob_ode_fitzhughnagumo, Tsit5(), t, data, priors;
    sample_args = (sampler = Turing.NUTS(0.85), num_samples = 10_000),
    likelihood = (u, p, t, σ) -> MvNormal(u, Diagonal(σ .^ 2)),
    likelihood_dist_priors = [InverseGamma(2, 3), InverseGamma(2, 3)])


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

