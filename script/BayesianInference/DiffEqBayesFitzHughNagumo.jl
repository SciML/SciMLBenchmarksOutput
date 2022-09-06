
using DiffEqBayes, BenchmarkTools


using OrdinaryDiffEq, RecursiveArrayTools, Distributions, ParameterizedFunctions, StanSample, DynamicHMC
using Plots, StaticArrays, Turing, LinearAlgebra


gr(fmt=:png)


fitz = @ode_def FitzhughNagumo begin
  dv = v - 0.33*v^3 -w + l
  dw = τinv*(v +  a - b*w)
end a b τinv l


prob_ode_fitzhughnagumo = ODEProblem(fitz, [1.0,1.0], (0.0,10.0), [0.7,0.8,1/12.5,0.5])
sol = solve(prob_ode_fitzhughnagumo, Tsit5())


sprob_ode_fitzhughnagumo = ODEProblem{false,SciMLBase.FullSpecialize}(fitz, SA[1.0,1.0], (0.0,10.0), SA[0.7,0.8,1/12.5,0.5])
sol = solve(sprob_ode_fitzhughnagumo, Tsit5())


t = collect(range(1,stop=10,length=10))
sig = 0.20
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(2)) for i in 1:length(t)]))


scatter(t, data[1,:])
scatter!(t, data[2,:])
plot!(sol)


priors = [truncated(Normal(1.0,0.5),0,1.5), truncated(Normal(1.0,0.5),0,1.5), truncated(Normal(0.0,0.5),0.0,0.5), truncated(Normal(0.5,0.5),0,1)]


@time bayesian_result_stan = stan_inference(prob_ode_fitzhughnagumo,t,data,priors; delta = 0.65, num_samples = 10_000, print_summary=false, vars=(DiffEqBayes.StanODEData(), InverseGamma(2, 3)))


@model function fitlv(data, prob)

    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    a ~ truncated(Normal(1.0,0.5),0,1.5)
    b ~ truncated(Normal(1.0,0.5),0,1.5)
    τinv ~ truncated(Normal(0.0,0.5),0.0,0.5)
    l ~ truncated(Normal(0.5,0.5),0,1)

    # Simulate Lotka-Volterra model. 
    p = SA[a,b,τinv,l]
    _prob = remake(prob, p = p)
    predicted = solve(_prob, Tsit5(); saveat=t)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(data, sprob_ode_fitzhughnagumo)

@time chain = sample(model, NUTS(0.65), 10000; progress=false)


@time chain = sample(model, NUTS(0.65), 10000; progress=false)


@time bayesian_result_turing = turing_inference(prob_ode_fitzhughnagumo,Tsit5(),t,data,priors;num_samples = 10_000)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

