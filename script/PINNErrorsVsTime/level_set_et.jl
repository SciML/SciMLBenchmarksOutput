
using NeuralPDE
using Integrals, Cubature, Cuba
using ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots
using OptimizationOptimisers
using DelimitedFiles
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum


function level_set(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters t x y
    @variables u(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    # Discretization
    xwidth = 1.0      #ft
    ywidth = 1.0
    tmax = 1.0      #min
    xScale = 1.0
    yScale = 1.0
    xMeshNum = 10
    yMeshNum = 10
    tMeshNum = 10
    dx = xwidth / xMeshNum
    dy = ywidth / yMeshNum
    dt = tmax / tMeshNum

    domains = [t ∈ Interval(0.0, tmax),
        x ∈ Interval(0.0, xwidth),
        y ∈ Interval(0.0, ywidth)]

    xs = 0.0:dx:xwidth
    ys = 0.0:dy:ywidth
    ts = 0.0:dt:tmax

    # Definitions
    x0 = 0.5
    y0 = 0.5
    Uwind = [0.0, 2.0]  #wind vector

    # Operators
    gn = (Dx(u(t, x, y))^2 + Dy(u(t, x, y))^2)^0.5  #gradient's norm
    ∇u = [Dx(u(t, x, y)), Dy(u(t, x, y))]
    n = ∇u / gn              #normal versor
    #U    = ((Uwind[1]*n[1] + Uwind[2]*n[2])^2)^0.5 #inner product between wind and normal vector

    R0 = 0.112471
    ϕw = 0#0.156927*max((0.44*U)^0.04086,1.447799)
    ϕs = 0
    S = R0 * (1 + ϕw + ϕs)

    # Equation
    eq = Dt(u(t, x, y)) + S * gn ~ 0  #LEVEL SET EQUATION

    initialCondition = (xScale * (x - x0)^2 + (yScale * (y - y0)^2))^0.5 - 0.2   #Distance from ignition

    bcs = [u(0, x, y) ~ initialCondition]  #from literature

    ## NEURAL NETWORK
    n = 10   #neuron number

    chain = Lux.Chain(Lux.Dense(3, n, tanh), Lux.Dense(n, n, tanh), Lux.Dense(n, 1))   #Neural network from OptimizationFlux library

    indvars = [t, x, y]   #physically independent variables
    depvars = [u(t, x, y)]       #dependent (target) variable

    dim = length(domains)

    losses = []
    error = []
    times = []

    dx_err = 0.1

    error_strategy = GridTraining(dx_err)

    discretization_ = PhysicsInformedNN(chain, error_strategy)
    @named pde_system_ = PDESystem(eq, bcs, domains, indvars, depvars)
    prob_ = discretize(pde_system_, discretization_)

    function loss_function_(θ, p)
        params = θ.u
        return prob_.f.f(params, nothing)
    end

    function cb_(p, l)
        try
            deltaT_s = time_ns()
            ctime = time_ns() - startTime - timeCounter

            push!(times, ctime / 1e9)
            push!(losses, l)

            # Extract parameters for loss calculation
            params = p.u
            loss_ = loss_function_(p, nothing)
            push!(error, loss_)

            timeCounter += time_ns() - deltaT_s
            return false
        catch e
            @warn "Callback error: $e"
            return false
        end
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
    prob = NeuralPDE.discretize(pde_system, discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training
    res = Optimization.solve(prob, minimizer, callback = cb_, maxiters = maxIters)

    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [ts, xs, ys]

    u_predict = [reshape([first(phi([t, x, y], res.minimizer)) for x in xs for y in ys],
                     (length(xs), length(ys))) for t in ts]  #matrix of model's prediction

    return [error, params, domain, times, losses] #add numeric solution
end

#level_set(NeuralPDE.QuadratureTraining(algorithm = CubaCuhre(), reltol = 1e-8, abstol = 1e-8, maxiters = 100), ADAM(0.01), 500)

maxIters = [(1, 1, 1, 1000, 1000, 1000, 1000), (1, 1, 1, 500, 500, 500, 500)] #iters for ADAM/LBFGS
# maxIters = [(1,1,1,2,2,2,2),(1,1,1,2,2,2,2)] #iters for ADAM/LBFGS

strategies = [
    NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1e-4, abstol = 1e-4, maxiters = 1100),
    #NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol = 1e-4, abstol = 1e-4, maxiters = 1100, batch = 0),
    NeuralPDE.GridTraining(0.1),
    NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol = 1e-4, abstol = 1e-4, maxiters = 1100),
    NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol = 1e-4, abstol = 1e-4, maxiters = 1100),
    NeuralPDE.GridTraining(0.1),
    NeuralPDE.StochasticTraining(400; bcs_points = 50),
    NeuralPDE.QuasiRandomTraining(400; bcs_points = 50)]

strategies_short_name = ["CubaCuhre",
    "HCubatureJL",
    "CubatureJLh",
    "CubatureJLp",
    "GridTraining",
    "StochasticTraining",
    "QuasiRandomTraining"]

minimizers = [Optimisers.ADAM(0.005),
    #BFGS()]
    LBFGS()]

minimizers_short_name = ["ADAM",
    "LBFGS"]
# "BFGS"]

# Run models
prediction_res = Dict()
error_res = Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()
losses_res = Dict()


## Convergence
for min in 1:length(minimizers) # minimizer
    for strat in 1:length(strategies) # strategy
        # println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
        res = level_set(strategies[strat], minimizers[min], maxIters[min][strat])
        push!(error_res, string(strat, min) => res[1])
        push!(params_res, string(strat, min) => res[2])
        push!(domains, string(strat, min) => res[3])
        push!(times, string(strat, min) => res[4])
        push!(losses_res, string(strat, min) => res[5])
    end
end


#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1], " + ", minimizers_short_name[1])
error = Plots.plot(times["11"], error_res["11"], yaxis = :log10, label = current_label)# xlims = (0,10))#legend = true)#, size=(1200,700))
plot!(error, times["21"], error_res["21"], yaxis = :log10,
    label = string(strategies_short_name[2], " + ", minimizers_short_name[1]))
plot!(error, times["31"], error_res["31"], yaxis = :log10,
    label = string(strategies_short_name[3], " + ", minimizers_short_name[1]))
plot!(error, times["41"], error_res["41"], yaxis = :log10,
    label = string(strategies_short_name[4], " + ", minimizers_short_name[1]))
plot!(error, times["51"], error_res["51"], yaxis = :log10,
    label = string(strategies_short_name[5], " + ", minimizers_short_name[1]))
plot!(error, times["61"], error_res["61"], yaxis = :log10,
    label = string(strategies_short_name[6], " + ", minimizers_short_name[1]))
plot!(error, times["71"], error_res["71"], yaxis = :log10,
    label = string(strategies_short_name[7], " + ", minimizers_short_name[1]))

plot!(error, times["12"], error_res["12"], yaxis = :log10,
    label = string(strategies_short_name[1], " + ", minimizers_short_name[2]))
plot!(error, times["22"], error_res["22"], yaxis = :log10,
    label = string(strategies_short_name[2], " + ", minimizers_short_name[2]))
plot!(error, times["32"], error_res["32"], yaxis = :log10,
    label = string(strategies_short_name[3], " + ", minimizers_short_name[2]))
plot!(error, times["42"], error_res["42"], yaxis = :log10,
    label = string(strategies_short_name[4], " + ", minimizers_short_name[2]))
plot!(error, times["52"], error_res["52"], yaxis = :log10,
    label = string(strategies_short_name[5], " + ", minimizers_short_name[2]))
plot!(error, times["62"], error_res["62"], yaxis = :log10,
    label = string(strategies_short_name[6], " + ", minimizers_short_name[2]))
plot!(error, times["72"], error_res["72"], yaxis = :log10,
    title = string("Level Set convergence ADAM/LBFGS"),
    ylabel = "log(error)", xlabel = "t",
    label = string(strategies_short_name[7], " + ", minimizers_short_name[2]))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

