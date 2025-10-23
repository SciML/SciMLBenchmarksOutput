
using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum


function solve(opt)
    strategy = QuadratureTraining()

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi*x)*sin(pi*y)

    # Boundary conditions
    bcs = [u(0, y) ~ 0.0f0, u(1, y) ~ -sin(pi*1)*sin(pi*y),
        u(x, 0) ~ 0.0f0, u(x, 1) ~ -sin(pi*x)*sin(pi*1)]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    # Neural network
    dim = 2 # number of dimensions
    chain = Lux.Chain(Lux.Dense(dim, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 1))

    discretization = PhysicsInformedNN(chain, strategy)

    indvars = [x, y]   #physically independent variables
    depvars = [u(x, y)]       #dependent (target) variable

    loss = []
    initial_time = nothing

    times = []

    cb = function (p, l)
        if initial_time == nothing
            initial_time = time()
        end
        push!(times, time() - initial_time)
        #println("Current loss for $opt is: $l")
        push!(loss, l)
        return false
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    if opt == "both"
        res = Optimization.solve(prob, ADAM(); callback = cb, maxiters = 50)
        prob = remake(prob, u0 = res.minimizer)
        res = Optimization.solve(prob, BFGS(); callback = cb, maxiters = 150)
    else
        res = Optimization.solve(prob, opt; callback = cb, maxiters = 200)
    end

    times[1] = 0.001

    return loss, times #add numeric solution
end


opt1 = Optimisers.ADAM()
opt2 = Optimisers.ADAM(0.005)
opt3 = Optimisers.ADAM(0.05)
opt4 = Optimisers.RMSProp()
opt5 = Optimisers.RMSProp(0.005)
opt6 = Optimisers.RMSProp(0.05)
opt7 = OptimizationOptimJL.BFGS()
opt8 = OptimizationOptimJL.LBFGS()


loss_1, times_1 = solve(opt1)
loss_2, times_2 = solve(opt2)
loss_3, times_3 = solve(opt3)
loss_4, times_4 = solve(opt4)
loss_5, times_5 = solve(opt5)
loss_6, times_6 = solve(opt6)
loss_7, times_7 = solve(opt7)
loss_8, times_8 = solve(opt8)
loss_9, times_9 = solve("both")


p = plot([times_1, times_2, times_3, times_4, times_5, times_6, times_7, times_8, times_9],
    [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9],
    xlabel = "time (s)",
    ylabel = "loss",
    xscale = :log10,
    yscale = :log10,
    labels = ["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"],
    legend = :bottomleft,
    linecolor = ["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])


p = plot([loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9],
    xlabel = "iterations",
    ylabel = "loss",
    yscale = :log10,
    labels = ["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"],
    legend = :bottomleft,
    linecolor = ["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])


@show loss_1[end], loss_2[end], loss_3[end], loss_4[end], loss_5[end],
loss_6[end], loss_7[end], loss_8[end], loss_9[end]


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

