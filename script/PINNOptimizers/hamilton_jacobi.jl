
using NeuralPDE, OptimizationFlux, ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots
import ModelingToolkit: Interval, infimum, supremum


function solve(opt)
    strategy = QuadratureTraining()

    ##  DECLARATIONS
    @parameters  t x1 x2 x3 x4
    @variables   u(..)

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
    tmax         = 1.0
    x1width      = 1.0
    x2width      = 1.0
    x3width      = 1.0
    x4width      = 1.0

    tMeshNum     = 10
    x1MeshNum    = 10
    x2MeshNum    = 10
    x3MeshNum    = 10
    x4MeshNum    = 10

    dt   = tmax/tMeshNum
    dx1  = x1width/x1MeshNum
    dx2  = x2width/x2MeshNum
    dx3  = x3width/x3MeshNum
    dx4  = x4width/x4MeshNum

    domains = [t ∈ Interval(0.0,tmax),
               x1 ∈ Interval(0.0,x1width),
               x2 ∈ Interval(0.0,x2width),
               x3 ∈ Interval(0.0,x3width),
               x4 ∈ Interval(0.0,x4width)]

    ts  = 0.0 : dt : tmax
    x1s = 0.0 : dx1 : x1width
    x2s = 0.0 : dx2 : x2width
    x3s = 0.0 : dx3 : x3width
    x4s = 0.0 : dx4 : x4width

    λ = 1.0f0

    # Operators
    Δu = Dxx1(u(t,x1,x2,x3,x4)) + Dxx2(u(t,x1,x2,x3,x4)) + Dxx3(u(t,x1,x2,x3,x4)) + Dxx4(u(t,x1,x2,x3,x4)) # Laplacian
    ∇u = [Dx1(u(t,x1,x2,x3,x4)), Dx2(u(t,x1,x2,x3,x4)),Dx3(u(t,x1,x2,x3,x4)),Dx4(u(t,x1,x2,x3,x4))]

    # Equation
    eq = Dt(u(t,x1,x2,x3,x4)) + Δu - λ*sum(∇u.^2) ~ 0  #HAMILTON-JACOBI-BELLMAN EQUATION

    terminalCondition =  log((1 + x1*x1 + x2*x2 + x3*x3 + x4*x4)/2) # see PNAS paper

    bcs = [u(tmax,x1,x2,x3,x4) ~ terminalCondition]  #PNAS paper again

    ## NEURAL NETWORK
    n = 20   #neuron number

    chain = Lux.Chain(Lux.Dense(5,n,tanh),Lux.Dense(n,n,tanh),Lux.Dense(n,1))   #Neural network from OptimizationFlux library

    discretization = PhysicsInformedNN(chain, strategy)

    indvars = [t,x1,x2,x3,x4]   #physically independent variables
    depvars = [u]       #dependent (target) variable

    loss = []
    initial_time = 0

    times = []

    cb = function (p,l)
        if initial_time == 0
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
        res = Optimization.solve(prob, ADAM(); callback = cb, maxiters=50)
        prob = remake(prob,u0=res.minimizer)
        res = Optimization.solve(prob, BFGS(); callback = cb, maxiters=150)
    else
        res = Optimization.solve(prob, opt; callback = cb, maxiters=200)
    end

    times[1] = 0.001

    return loss, times #add numeric solution
end


opt1 = ADAM()
opt2 = ADAM(0.005)
opt3 = ADAM(0.05)
opt4 = RMSProp()
opt5 = RMSProp(0.005)
opt6 = RMSProp(0.05)
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


p = plot([times_1, times_2, times_3, times_4, times_5, times_6, times_7, times_8, times_9], [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9],xlabel="time (s)", ylabel="loss", xscale=:log10, yscale=:log10, labels=["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"], legend=:bottomleft, linecolor=["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])


p = plot([loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_9], xlabel="iterations", ylabel="loss", yscale=:log10, labels=["ADAM(0.001)" "ADAM(0.005)" "ADAM(0.05)" "RMSProp(0.001)" "RMSProp(0.005)" "RMSProp(0.05)" "BFGS()" "LBFGS()" "ADAM + BFGS"], legend=:bottomleft, linecolor=["#2660A4" "#4CD0F4" "#FEC32F" "#F763CD" "#44BD79" "#831894" "#A6ED18" "#980000" "#FF912B"])


@show loss_1[end], loss_2[end], loss_3[end], loss_4[end], loss_5[end], loss_6[end], loss_7[end], loss_8[end], loss_9[end]


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

