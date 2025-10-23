
using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum


# Physical and numerical parameters (fixed)
nu = 0.07
nx = 10001 #101
x_max = 2.0 * pi
dx = x_max / (nx - 1.0)
nt = 2 #10
dt = dx * nu
t_max = dt * nt

# Analytic function
function analytic_sol_func(t, x)
    -2*nu*(-(-8*t + 2*x)*exp(-(-4*t + x)^2/(4*nu*(t + 1))) /
           (4*nu*(t + 1)) -
           (-8*t + 2*x - 12.5663706143592) *
           exp(-(-4*t + x - 6.28318530717959)^2/(4*nu*(t + 1))) /
           (4*nu*(t +
                  1)))/(exp(-(-4*t + x - 6.28318530717959)^2 /
                            (4*nu*(t + 1))) + exp(-(-4*t + x)^2/(4*nu*(t + 1)))) + 4
end


function burgers(strategy, minimizer)
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) ~ nu * Dxx(u(x, t))

    bcs = [u(x, 0.0) ~ analytic_sol_func(x, 0.0),
        u(0.0, t) ~ u(x_max, t)]

    domains = [x ∈ Interval(0.0, x_max),
        t ∈ Interval(0.0, t_max)]

    chain = Lux.Chain(Lux.Dense(2, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 1))
    discretization = PhysicsInformedNN(chain, strategy)

    indvars = [x, t]   # physically independent variables
    depvars = [u]      # dependent (target) variable

    dim = length(domains)

    losses = []
    error = []
    times = []

    dx_err = 0.00005

    error_strategy = GridTraining(dx_err)

    discretization_ = PhysicsInformedNN(chain, error_strategy)
    @named pde_system_ = PDESystem(eq, bcs, domains, indvars, depvars)
    prob_ = discretize(pde_system_, discretization_)

    function loss_function__(θ)
        return prob_.f.f(θ, nothing)
    end

    cb = function (p, l)
        timeCounter = 0.0
        deltaT_s = time_ns() # Start a clock when the callback begins

        ctime = time_ns() - startTime - timeCounter # Time for the time benchmark plot
        append!(times, ctime / 10^9) # Conversion nanosec to seconds
        append!(losses, l)
        append!(error, l)

        timeCounter = timeCounter + time_ns() - deltaT_s # Sum all delays due to the callback functions

        return false
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    startTime = time_ns() # Fix initial time (t=0) before starting the training

    if minimizer == "both"
        res = Optimization.solve(prob, ADAM(); callback = cb, maxiters = 5)
        prob = remake(prob, u0 = res.minimizer)
        res = Optimization.solve(prob, BFGS(); callback = cb, maxiters = 15)
    else
        res = Optimization.solve(prob, minimizer; callback = cb, maxiters = 500)
    end

    phi = discretization.phi

    params = res.minimizer

    return [error, params, times, losses]
end


# Settings:
#maxIters = [(0,0,0,0,0,0,20000),(300,300,300,300,300,300,300)] #iters

strategies = [NeuralPDE.QuadratureTraining()]

strategies_short_name = ["QuadratureTraining"]

minimizers = [Optimisers.ADAM(),
    Optimisers.ADAM(0.000005),
    Optimisers.ADAM(0.0005),
    Optimisers.RMSProp(),
    Optimisers.RMSProp(0.00005),
    Optimisers.RMSProp(0.05),
    OptimizationOptimJL.BFGS(),
    OptimizationOptimJL.LBFGS()]

minimizers_short_name = ["ADAM",
    "ADAM(0.000005)",
    "ADAM(0.0005)",
    "RMS",
    "RMS(0.00005)",
    "RMS(0.05)",
    "BFGS",
    "LBFGS"]


# Run models
error_res = Dict()
params_res = Dict()
times = Dict()
losses_res = Dict()

print("Starting run \n")

for min in 1:length(minimizers) # minimizer
    for strat in 1:length(strategies) # strategy
        #println(string(strategies_short_name[1], "  ", minimizers_short_name[min]))
        res = burgers(strategies[strat], minimizers[min])
        push!(error_res, string(strat, min) => res[1])
        push!(params_res, string(strat, min) => res[2])
        push!(times, string(strat, min) => res[3])
        push!(losses_res, string(strat, min) => res[4])
    end
end


#PLOT ERROR VS ITER: to compare to compare between minimizers, keeping the same strategy (easily adjustable to compare between strategies)
error_iter = Plots.plot(1:length(error_res["11"]), error_res["11"], yaxis = :log10,
    title = string("Burger error vs iter"), ylabel = "Error",
    label = string(minimizers_short_name[1]), ylims = (0.0001, 1))
plot!(error_iter, 1:length(error_res["12"]), error_res["12"],
    yaxis = :log10, label = string(minimizers_short_name[2]))
plot!(error_iter, 1:length(error_res["13"]), error_res["13"],
    yaxis = :log10, label = string(minimizers_short_name[3]))
plot!(error_iter, 1:length(error_res["14"]), error_res["14"],
    yaxis = :log10, label = string(minimizers_short_name[4]))
plot!(error_iter, 1:length(error_res["15"]), error_res["15"],
    yaxis = :log10, label = string(minimizers_short_name[5]))
plot!(error_iter, 1:length(error_res["16"]), error_res["16"],
    yaxis = :log10, label = string(minimizers_short_name[6]))
plot!(error_iter, 1:length(error_res["17"]), error_res["17"],
    yaxis = :log10, label = string(minimizers_short_name[7]))
plot!(error_iter, 1:length(error_res["18"]), error_res["18"],
    yaxis = :log10, label = string(minimizers_short_name[8]))

Plots.plot(error_iter)


#Use after having modified the analysis setting correctly --> Error vs iter: to compare different strategies, keeping the same minimizer
#error_iter = Plots.plot(1:length(error_res["11"]), error_res["11"], yaxis=:log10, title = string("Burger error vs iter"), ylabel = "Error", label = string(strategies_short_name[1]), ylims = (0.0001,1))
#plot!(error_iter, 1:length(error_res["21"]), error_res["21"], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error_iter, 1:length(error_res["31"]), error_res["31"], yaxis=:log10, label = string(strategies_short_name[3]))
#plot!(error_iter, 1:length(error_res["41"]), error_res["41"], yaxis=:log10, label = string(strategies_short_name[4]))
#plot!(error_iter, 1:length(error_res["51"]), error_res["51"], yaxis=:log10, label = string(strategies_short_name[5]))
#plot!(error_iter, 1:length(error_res["61"]), error_res["61"], yaxis=:log10, label = string(strategies_short_name[6]))
#plot!(error_iter, 1:length(error_res["71"]), error_res["71"], yaxis=:log10, label = string(strategies_short_name[7]))


#PLOT ERROR VS TIME: to compare to compare between minimizers, keeping the same strategy
error_time = plot(
    times["11"], error_res["11"], yaxis = :log10, label = string(minimizers_short_name[1]),
    title = string("Burger error vs time"), ylabel = "Error", size = (1500, 500))
plot!(error_time, times["12"], error_res["12"], yaxis = :log10,
    label = string(minimizers_short_name[2]))
plot!(error_time, times["13"], error_res["13"], yaxis = :log10,
    label = string(minimizers_short_name[3]))
plot!(error_time, times["14"], error_res["14"], yaxis = :log10,
    label = string(minimizers_short_name[4]))
plot!(error_time, times["15"], error_res["15"], yaxis = :log10,
    label = string(minimizers_short_name[5]))
plot!(error_time, times["16"], error_res["16"], yaxis = :log10,
    label = string(minimizers_short_name[6]))
plot!(error_time, times["17"], error_res["17"], yaxis = :log10,
    label = string(minimizers_short_name[7]))
plot!(error_time, times["18"], error_res["18"], yaxis = :log10,
    label = string(minimizers_short_name[7]))

Plots.plot(error_time)


#Use after having modified the analysis setting correctly --> Error vs time: to compare different strategies, keeping the same minimizer
#error_time = plot(times["11"], error_res["11"], yaxis=:log10, label = string(strategies_short_name[1]),title = string("Burger error vs time"), ylabel = "Error", size = (1500,500))
#plot!(error_time, times["21"], error_res["21"], yaxis=:log10, label = string(strategies_short_name[2]))
#plot!(error_time, times["31"], error_res["31"], yaxis=:log10, label = string(strategies_short_name[3]))
#plot!(error_time, times["41"], error_res["41"], yaxis=:log10, label = string(strategies_short_name[4]))
#plot!(error_time, times["51"], error_res["51"], yaxis=:log10, label = string(strategies_short_name[5]))
#plot!(error_time, times["61"], error_res["61"], yaxis=:log10, label = string(strategies_short_name[6]))
#plot!(error_time, times["71"], error_res["71"], yaxis=:log10, label = string(strategies_short_name[7]))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

