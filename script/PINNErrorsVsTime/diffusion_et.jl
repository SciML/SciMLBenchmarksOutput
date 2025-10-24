
using NeuralPDE
using Integrals, Cubature, Cuba
using ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots
using OptimizationOptimisers
using DelimitedFiles
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum


function diffusion(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) - Dxx(u(x, t)) ~ -exp(-t) * (sin(pi * x) - pi^2 * sin(pi * x))

    bcs = [u(x, 0) ~ sin(pi*x),
        u(-1, t) ~ 0.0,
        u(1, t) ~ 0.0]

    domains = [x ∈ Interval(-1.0, 1.0),
        t ∈ Interval(0.0, 1.0)]

    dx = 0.2;
    dt = 0.1
    xs,
    ts = [infimum(domain.domain):(dx / 10):supremum(domain.domain)
          for (dx, domain) in zip([dx, dt], domains)]

    indvars = [x, t]
    depvars = [u(x, t)]

    chain = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 10, tanh), Lux.Dense(10, 1))

    losses = []
    error = []
    times = []

    dx_err = [0.2, 0.1]

    error_strategy = GridTraining(dx_err)

    discretization_ = PhysicsInformedNN(chain, error_strategy)
    @named pde_system_ = PDESystem(eq, bcs, domains, indvars, depvars)
    prob_ = discretize(pde_system_, discretization_)

    function loss_function_(θ, p)
        params = θ.u
        return prob_.f.f(params, nothing)
    end

    cb_ = function (p, l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        loss_ = loss_function_(p, nothing)
        append!(error, loss_)

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    discretization = PhysicsInformedNN(chain, strategy)

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training

    res = Optimization.solve(prob, minimizer; callback = cb_, maxiters = maxIters)
    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [x, t]

    u_predict = reshape([first(phi([x, t], res.minimizer)) for x in xs for t in ts], (
        length(xs), length(ts)))

    return [error, params, domain, times, u_predict, losses]
end


maxIters = [(5000, 5000, 5000, 5000, 5000), (300, 300, 300, 300, 300)] #iters for ADAM/LBFGS
# maxIters = [(5,5,5,5,5,5),(3,3,3,3,3,3)] #iters for ADAM/LBFGS

strategies = [
    NeuralPDE.GridTraining([0.2, 0.1]),
    NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol = 1e-4, abstol = 1e-5, maxiters = 1100),
    NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol = 1e-4, abstol = 1e-5, maxiters = 1100),
    NeuralPDE.StochasticTraining(400; bcs_points = 50),
    NeuralPDE.QuasiRandomTraining(400; bcs_points = 50)
]

strategies_short_name = [#"CubaCuhre",
    #"HCubatureJL",
    "CubatureJLh",
    "CubatureJLp",
    #"CubaVegas",
    #"CubaSUAVE"]
    "GridTraining",
    "StochasticTraining",
    "QuasiRandomTraining"]

minimizers = [Optimisers.ADAM(0.001),
    #BFGS()]
    LBFGS()]

minimizers_short_name = ["ADAM",
    "LBFGS"]
# "BFGS"]

# Run models
error_res = Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()
prediction = Dict()
losses_res = Dict()


print("Starting run")
## Convergence

for min in 1:length(minimizers) # minimizer
    for strat in 1:length(strategies) # strategy
        # println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
        res = diffusion(strategies[strat], minimizers[min], maxIters[min][strat])
        push!(error_res, string(strat, min) => res[1])
        push!(params_res, string(strat, min) => res[2])
        push!(domains, string(strat, min) => res[3])
        push!(times, string(strat, min) => res[4])
        push!(prediction, string(strat, min) => res[5])
        push!(losses_res, string(strat, min) => res[6])
    end
end


current_label = string(strategies_short_name[1], " + ", minimizers_short_name[1])
error = Plots.plot(times["11"], error_res["11"], yaxis = :log10, label = current_label)
plot!(error, times["21"], error_res["21"], yaxis = :log10,
    label = string(strategies_short_name[2], " + ", minimizers_short_name[1]))
plot!(error, times["31"], error_res["31"], yaxis = :log10,
    label = string(strategies_short_name[3], " + ", minimizers_short_name[1]))
plot!(error, times["41"], error_res["41"], yaxis = :log10,
    label = string(strategies_short_name[4], " + ", minimizers_short_name[1]))
plot!(error, times["51"], error_res["51"], yaxis = :log10,
    label = string(strategies_short_name[5], " + ", minimizers_short_name[1]))

plot!(error, times["12"], error_res["12"], yaxis = :log10,
    label = string(strategies_short_name[1], " + ", minimizers_short_name[2]))
plot!(error, times["22"], error_res["22"], yaxis = :log10,
    label = string(strategies_short_name[2], " + ", minimizers_short_name[2]))
plot!(error, times["32"], error_res["32"], yaxis = :log10,
    label = string(strategies_short_name[3], " + ", minimizers_short_name[2]))
plot!(error, times["42"], error_res["42"], yaxis = :log10,
    label = string(strategies_short_name[4], " + ", minimizers_short_name[2]))
plot!(error, times["52"], error_res["52"], yaxis = :log10,
    title = string("Diffusion convergence ADAM/LBFGS"),
    ylabel = "log(error)", xlabel = "t",
    label = string(strategies_short_name[5], " + ", minimizers_short_name[2]))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

