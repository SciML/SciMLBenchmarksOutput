
using NeuralPDE, OptimizationFlux, ModelingToolkit, Optimization, OptimizationOptimJL
using Lux, Plots
import ModelingToolkit: Interval, infimum, supremum


function solve(opt)
    strategy = QuadratureTraining()

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x,t)) - Dxx(u(x,t)) ~ -exp(-t) * (sin(pi * x) - pi^2 * sin(pi * x))

    bcs = [u(x,0) ~ sin(pi*x),
           u(-1,t) ~ 0.,
           u(1,t) ~ 0.]

    domains = [x ∈ Interval(-1.0,1.0),
               t ∈ Interval(0.0,1.0)]

    chain = Lux.Chain(Lux.Dense(2,18,tanh),Lux.Dense(18,18,tanh),Lux.Dense(18,1))

    discretization = PhysicsInformedNN(chain,strategy)

    indvars = [x, t]   #physically independent variables
    depvars = [u(x,t)]       #dependent (target) variable

    loss = []
    initial_time = nothing

    times = []

    cb_ = function (p,l)
        if initial_time == nothing
            initial_time = time()
        end
        push!(times, time() - initial_time)
        #println("Current loss for $opt is: $l")
        push!(loss, l)
      #  println(l )
      #  println(time() - initial_time)
        return false
    end

    @named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
    prob = discretize(pde_system, discretization)

    if opt == "both"
        res = Optimization.solve(prob, ADAM(); callback = cb_, maxiters=50)
        prob = remake(prob,u0=res.minimizer)
        res = Optimization.solve(prob, BFGS(); callback = cb_, maxiters=150)
    else
        res = Optimization.solve(prob, opt; callback = cb_, maxiters=200)
    end

    times[1] = 0.01

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

