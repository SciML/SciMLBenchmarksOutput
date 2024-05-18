
using NeuralPDE
using Integrals, IntegralsCubature, IntegralsCuba
using OptimizationFlux, ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots
using DelimitedFiles
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum


function hamilton_jacobi(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters t x1 x2 x3 x4
    @variables u(..)

    Dt = Differential(t)

    Dx1 = Differential(x1)
    Dx2 = Differential(x2)
    Dx3 = Differential(x3)
    Dx4 = Differential(x4)

    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dxx3 = Differential(x3)^2
    Dxx4 = Differential(x4)^2

    # Discretization
    tmax = 1.0
    x1width = 1.0
    x2width = 1.0
    x3width = 1.0
    x4width = 1.0

    tMeshNum = 10
    x1MeshNum = 10
    x2MeshNum = 10
    x3MeshNum = 10
    x4MeshNum = 10

    dt = tmax / tMeshNum
    dx1 = x1width / x1MeshNum
    dx2 = x2width / x2MeshNum
    dx3 = x3width / x3MeshNum
    dx4 = x4width / x4MeshNum

    domains = [t ∈ Interval(0.0, tmax),
        x1 ∈ Interval(0.0, x1width),
        x2 ∈ Interval(0.0, x2width),
        x3 ∈ Interval(0.0, x3width),
        x4 ∈ Interval(0.0, x4width)]

    ts = 0.0:dt:tmax
    x1s = 0.0:dx1:x1width
    x2s = 0.0:dx2:x2width
    x3s = 0.0:dx3:x3width
    x4s = 0.0:dx4:x4width

    λ = 1.0f0

    # Operators
    Δu = Dxx1(u(t, x1, x2, x3, x4)) + Dxx2(u(t, x1, x2, x3, x4)) + Dxx3(u(t, x1, x2, x3, x4)) + Dxx4(u(t, x1, x2, x3, x4)) # Laplacian
    ∇u = [Dx1(u(t, x1, x2, x3, x4)), Dx2(u(t, x1, x2, x3, x4)), Dx3(u(t, x1, x2, x3, x4)), Dx4(u(t, x1, x2, x3, x4))]

    # Equation
    eq = Dt(u(t, x1, x2, x3, x4)) + Δu - λ * sum(∇u .^ 2) ~ 0  #HAMILTON-JACOBI-BELLMAN EQUATION

    terminalCondition = log((1 + x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4) / 2) # see PNAS paper

    bcs = [u(tmax, x1, x2, x3, x4) ~ terminalCondition]  #PNAS paper again

    ## NEURAL NETWORK
    n = 10   #neuron number

    chain = Lux.Chain(Lux.Dense(5, n, tanh), Lux.Dense(n, n, tanh), Lux.Dense(n, 1))   #Neural network from OptimizationFlux library

    indvars = [t, x1, x2, x3, x4]   #physically independent variables
    depvars = [u(t, x1, x2, x3, x4)]       #dependent (target) variable

    dim = length(domains)

    losses = []
    error = []
    times = []

    dx_err = 0.2

    error_strategy = GridTraining(dx_err)

    discretization_ = PhysicsInformedNN(chain, error_strategy)
    @named pde_system_ = PDESystem(eq, bcs, domains, indvars, depvars)
    prob_ = discretize(pde_system_, discretization_)

    function loss_function_(θ, p)
        return prob_.f.f(θ, nothing)
    end

    cb_ = function (p, l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime / 10^9) #Conversion nanosec to seconds
        append!(losses, l)
        loss_ = loss_function_(p, nothing)
        append!(error, loss_)

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
    prob = NeuralPDE.discretize(pde_system, discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training
    res = Optimization.solve(prob, minimizer, callback=cb_, maxiters=maxIters)

    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [ts, x1s, x2s, x3s, x4s]

    u_predict = [reshape([first(phi([t, x1, x2, x3, x4], res.minimizer)) for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s], (length(x1s), length(x2s), length(x3s), length(x4s))) for t in ts]  #matrix of model's prediction

    return [error, params, domain, times, losses]
end

maxIters = [(1,1,1,1000,1000,1000,1000),(1,1,1,300,300,300,300)] #iters for ADAM/LBFGS
# maxIters = [(1,1,1,1,1,2,2),(1,1,1,3,3,3,3)] #iters for ADAM/LBFGS

strategies = [NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1e-4, abstol = 1e-4, maxiters = 100),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol = 1e-4, abstol = 1e-4, maxiters = 100, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol = 1e-4, abstol = 1e-4, maxiters = 100),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol = 1e-4, abstol = 1e-4, maxiters = 100),
              NeuralPDE.GridTraining(0.2),
              NeuralPDE.StochasticTraining(400 ; bcs_points= 50),
              NeuralPDE.QuasiRandomTraining(400 ; bcs_points= 50)]

strategies_short_name = ["CubaCuhre",
                        "HCubatureJL",
                        "CubatureJLh",
                        "CubatureJLp",
                        "GridTraining",
                        "StochasticTraining",
                        "QuasiRandomTraining"]

minimizers = [ADAM(0.005),
              #BFGS()]
              LBFGS()]


minimizers_short_name = ["ADAM",
                         "LBFGS"]
                        #"BFGS"]


# Run models
error_res =  Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()
losses_res = Dict()


print("Starting run")
## Convergence

for min =1:length(minimizers) # minimizer
      for strat=1:length(strategies) # strategy
            # println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
            res = hamilton_jacobi(strategies[strat], minimizers[min], maxIters[min][strat])
            push!(error_res, string(strat,min)     => res[1])
            push!(params_res, string(strat,min) => res[2])
            push!(domains, string(strat,min)        => res[3])
            push!(times, string(strat,min)        => res[4])
            push!(losses_res, string(strat,min)   => res[5])
      end
end


#Plotting the first strategy with the first minimizer out from the loop to initialize the canvas
current_label = string(strategies_short_name[1], " + " , minimizers_short_name[1])
error = Plots.plot(times["11"], error_res["11"], yaxis=:log10, label = current_label)#, xlims = (0,10))#legend = true)#, size=(1200,700))
plot!(error, times["21"], error_res["21"], yaxis=:log10, label = string(strategies_short_name[2], " + " , minimizers_short_name[1]))
plot!(error, times["31"], error_res["31"], yaxis=:log10, label = string(strategies_short_name[3], " + " , minimizers_short_name[1]))
plot!(error, times["41"], error_res["41"], yaxis=:log10, label = string(strategies_short_name[4], " + " , minimizers_short_name[1]))
plot!(error, times["51"], error_res["51"], yaxis=:log10, label = string(strategies_short_name[5], " + " , minimizers_short_name[1]))
plot!(error, times["61"], error_res["61"], yaxis=:log10, label = string(strategies_short_name[6], " + " , minimizers_short_name[1]))
plot!(error, times["71"], error_res["71"], yaxis=:log10, label = string(strategies_short_name[7], " + " , minimizers_short_name[1]))


plot!(error, times["12"], error_res["12"], yaxis=:log10, label = string(strategies_short_name[1], " + " , minimizers_short_name[2]))
plot!(error, times["22"], error_res["22"], yaxis=:log10, label = string(strategies_short_name[2], " + " , minimizers_short_name[2]))
plot!(error, times["32"], error_res["32"], yaxis=:log10, label = string(strategies_short_name[3], " + " , minimizers_short_name[2]))
plot!(error, times["42"], error_res["42"], yaxis=:log10, label = string(strategies_short_name[4], " + " , minimizers_short_name[2]))
plot!(error, times["52"], error_res["52"], yaxis=:log10, label = string(strategies_short_name[5], " + " , minimizers_short_name[2]))
plot!(error, times["62"], error_res["62"], yaxis=:log10, label = string(strategies_short_name[6], " + " , minimizers_short_name[2]))
plot!(error, times["72"], error_res["72"], yaxis=:log10, title = string("Hamilton Jacobi convergence ADAM/LBFGS"), ylabel = "log(error)",xlabel = "t", label = string(strategies_short_name[7], " + " , minimizers_short_name[2]))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

