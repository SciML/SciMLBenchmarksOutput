
using DiffEqBayes
using DiffEqCallbacks, StaticArrays
using Distributions, StanSample, DynamicHMC, Turing
using OrdinaryDiffEq, RecursiveArrayTools, ParameterizedFunctions, DiffEqCallbacks
using Plots, LinearAlgebra


gr(fmt = :png)


g1 = @ode_def LorenzExample begin
    dx = σ*(y-x)
    dy = x*(ρ-z) - y
    dz = x*y - β*z
end σ ρ β


r0 = [1.0; 0.0; 0.0]
tspan = (0.0, 30.0)
p = [10.0, 28.0, 2.66]


prob = ODEProblem(g1, r0, tspan, p)
sol = solve(prob, Tsit5())


sr0 = SA[1.0; 0.0; 0.0]
tspan = (0.0, 30.0)
sp = SA[10.0, 28.0, 2.66]
sprob = ODEProblem{false, SciMLBase.FullSpecialize}(g1, sr0, tspan, sp)
sol = solve(sprob, Tsit5())


t = collect(range(1, stop = 30, length = 30))
sig = 0.49
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(3)) for i in 1:length(t)]))


Plots.scatter(t, data[1, :], markersize = 4, color = :purple)
Plots.scatter!(t, data[2, :], markersize = 4, color = :yellow)
Plots.scatter!(t, data[3, :], markersize = 4, color = :black)
plot!(sol)


cb = AdaptiveProbIntsUncertainty(5)
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 100, callback = cb, reltol = 1e-5, abstol = 1e-5)
plot(sim, vars = (0, 1), linealpha = 0.4)


cb = AdaptiveProbIntsUncertainty(5)
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 100, callback = cb, reltol = 1e-6, abstol = 1e-6)
plot(sim, vars = (0, 1), linealpha = 0.4)


cb = AdaptiveProbIntsUncertainty(5)
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 100, callback = cb, reltol = 1e-8, abstol = 1e-8)
plot(sim, vars = (0, 1), linealpha = 0.4)


priors = [truncated(Normal(10, 2), 1, 15), truncated(Normal(30, 5), 1, 45),
    truncated(Normal(2.5, 0.5), 1, 4)]


@time bayesian_result_stan = stan_inference(
    prob, t, data, priors; delta = 0.65, reltol = 1e-8, abstol = 1e-8,
    vars = (DiffEqBayes.StanODEData(), InverseGamma(2, 3)))


@model function fitlv(data, prob)
    # Prior distributions.
    α ~ InverseGamma(2, 3)
    σ ~ truncated(Normal(10, 2), 1, 15)
    ρ ~ truncated(Normal(30, 5), 1, 45)
    β ~ truncated(Normal(2.5, 0.5), 1, 4)

    # Simulate Lotka-Volterra model. 
    p = SA[σ, ρ, β]
    _prob = remake(prob, p = p)
    predicted = solve(_prob, Vern9(); saveat = t)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], α^2 * I)
    end

    return nothing
end

model = fitlv(data, sprob)

@time chain = sample(model, Turing.NUTS(0.65), 10000; progress = false)


@time bayesian_result_turing = turing_inference(
    prob, Vern9(), t, data, priors; reltol = 1e-8, abstol = 1e-8,
    likelihood = (u, p, t, σ) -> MvNormal(u, Diagonal((σ) .^ 2 .* ones(length(u)))),
    likelihood_dist_priors = [InverseGamma(2, 3), InverseGamma(2, 3), InverseGamma(2, 3)])


@time bayesian_result_dynamichmc = dynamichmc_inference(
    prob, Tsit5(), t, data, priors; solve_kwargs = (reltol = 1e-8, abstol = 1e-8))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

