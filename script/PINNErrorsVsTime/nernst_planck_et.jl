
using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using DelimitedFiles
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum


function nernst_planck(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters t x y z
    @variables c(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2


    ## DOMAINS AND OPERATORS

    # Discretization
    xwidth      = 1.0
    ywidth      = 1.0
    zwidth      = 1.0
    tmax        = 1.0
    xMeshNum    = 10
    yMeshNum    = 10
    zMeshNum    = 10
    tMeshNum    = 10

    dx  = xwidth/xMeshNum
    dy  = ywidth/yMeshNum
    dz  = zwidth/zMeshNum
    dt  = tmax/tMeshNum

    domains = [t ∈ Interval(0.0,tmax),
               x ∈ Interval(0.0,xwidth),
               y ∈ Interval(0.0,ywidth),
               z ∈ Interval(0.0,zwidth)]

    xs = 0.0 : dx : xwidth
    ys = 0.0 : dy : ywidth
    zs = 0.0 : dz : zwidth
    ts = 0.0 : dt : tmax

    # Constants
    D = 1  #dummy
    ux = 10 #dummy
    uy = 10 #dummy
    uz = 10 #dummy

    # Operators
    div = - D*(Dxx(c(t,x,y,z)) + Dyy(c(t,x,y,z)) + Dzz(c(t,x,y,z)))
          + (ux*Dx(c(t,x,y,z)) + uy*Dy(c(t,x,y,z)) + uz*Dz(c(t,x,y,z)))

    # Equation
    eq = Dt(c(t,x,y,z)) + div ~ 0      #NERNST-PLANCK EQUATION

    # Boundary conditions
    bcs = [c(0,x,y,z) ~ 0]

    ## NEURAL NETWORK
    n = 16   #neuron number

    chain = FastChain(FastDense(4,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

    indvars = [t,x,y,z]   #independent variables
    depvars = [c(t,x,y,z)]       #dependent (target) variable

    dim = length(domains)

    losses = []
    error = []
    times = []

    dx_err = 0.2

    error_strategy = GridTraining(dx_err)

    initθ = Float64.(DiffEqFlux.initial_params(chain))
    eltypeθ = eltype(initθ)
    parameterless_type_θ = DiffEqBase.parameterless_type(initθ)

    phi = NeuralPDE.get_phi(chain,parameterless_type_θ)
    derivative = NeuralPDE.get_numeric_derivative()

    _pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                             phi,derivative,nothing,chain,initθ,error_strategy)

    bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
    _bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                              phi,derivative,nothing,chain,initθ,error_strategy,
                                              bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    train_sets = NeuralPDE.generate_training_sets(domains,dx_err,[eq],bcs,eltypeθ,indvars,depvars)
    train_domain_set, train_bound_set = train_sets


    pde_loss_functions = [NeuralPDE.get_loss_function(_pde_loss_function,
                                          train_domain_set[1],eltypeθ,
                                          parameterless_type_θ,error_strategy)]

    bc_loss_functions = [NeuralPDE.get_loss_function(bc,
                                          train_set,eltypeθ,
                                          parameterless_type_θ,error_strategy) for (train_set,bc) in zip(train_bound_set, _bc_loss_functions)]
    loss_functions =  [pde_loss_functions;bc_loss_functions]
    function loss_function_(θ,p)
        return sum(map(l->l(θ) ,loss_functions))
    end

    cb_ = function (p,l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        loss_ = loss_function_(p,nothing)
        append!(error, loss_)

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy; init_params =initθ)
    prob = NeuralPDE.discretize(pde_system,discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training
    res = GalacticOptim.solve(prob, minimizer, cb = cb_, maxiters=maxIters)

    phi = discretization.phi

    params = res.minimizer


    # Model prediction
    domain = [ts, xs, ys, zs]

    u_predict  = [reshape([phi([t,x,y,z],res.minimizer) for x in xs for y in ys for z in zs],
                 (length(xs),length(ys),length(zs))) for t in ts]


    return [error, params, domain, times]
end

maxIters = [(1,1,1,1000,1000,1000,1000),(1,1,1,300,300,300,300)] #iters for ADAM/LBFGS
# maxIters = [(1,1,1,10,10,10,10),(1,1,1,3,3,3,3)] #iters for ADAM/LBFGS

strategies = [NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1e-4, abstol = 1e-4, maxiters = 50),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol = 1e-4, abstol = 1e-4, maxiters = 50, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol = 1e-4, abstol = 1e-4, maxiters = 50),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol = 1e-4, abstol = 1e-4, maxiters = 50),
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
                        # "BFGS"]


# Run models
error_res =  Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()


## Convergence

for strat=1:length(strategies) # strategy
      for min =1:length(minimizers) # minimizer
            # println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
            res = nernst_planck(strategies[strat], minimizers[min], maxIters[min][strat])
            push!(error_res, string(strat,min)     => res[1])
            push!(params_res, string(strat,min) => res[2])
            push!(domains, string(strat,min)        => res[3])
            push!(times, string(strat,min)        => res[4])
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
plot!(error, times["72"], error_res["72"], yaxis=:log10, title = string("Nernst Planck convergence ADAM/LBFGS"), ylabel = "log(error)", xlabel = "t",label = string(strategies_short_name[7], " + " , minimizers_short_name[2]))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

