
using DiffEqBayes, StanSample, DynamicHMC, Turing


using Distributions, BenchmarkTools, StaticArrays
using OrdinaryDiffEq, RecursiveArrayTools, ParameterizedFunctions
using Plots, LinearAlgebra


gr(fmt = :png)


f = @ode_def LotkaVolterraTest begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d


u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1, 0]


prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5())


su0 = SA[1.0, 1.0]
sp = SA[1.5, 1.0, 3.0, 1, 0]
sprob = ODEProblem{false, SciMLBase.FullSpecialize}(f, su0, tspan, sp)
sol = solve(sprob, Tsit5())


t = collect(range(1, stop = 10, length = 10))
sig = 0.49
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(2)) for i in 1:length(t)]))


scatter(t, data[1, :], lab = "#prey (data)")
scatter!(t, data[2, :], lab = "#predator (data)")
plot!(sol)


priors = [truncated(Normal(1.5, 0.5), 0.5, 2.5), truncated(Normal(1.2, 0.5), 0, 2),
    truncated(Normal(3.0, 0.5), 1, 4), truncated(Normal(1.0, 0.5), 0, 2)]


@btime bayesian_result_stan = stan_inference(
    prob, t, data, priors, num_samples = 10_000, print_summary = false,
    delta = 0.65, vars = (DiffEqBayes.StanODEData(), InverseGamma(2, 3)))


@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)

    # Simulate Lotka-Volterra model. 
    p = SA[α, β, γ, δ]
    _prob = remake(prob, p = p)
    predicted = solve(_prob, Tsit5(); saveat = t)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(data, sprob)

@time chain = sample(model, Turing.NUTS(0.65), 10000; progress = false)


@btime bayesian_result_turing = turing_inference(
    prob, Tsit5(), t, data, priors, num_samples = 10_000)


@btime bayesian_result_dynamichmc = dynamichmc_inference(
    prob, Tsit5(), t, data, priors, num_samples = 10_000)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

