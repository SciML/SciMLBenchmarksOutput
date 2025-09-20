
using OrdinaryDiffEq, DiffEqDevTools, Sundials, Plots, ODEInterfaceDiffEq, LSODA
using LinearAlgebra, StaticArrays, RecursiveFactorization
using OrdinaryDiffEqFIRK
gr()

# E5 Problem from Stiff Test Set
# Exact implementation based on the Fortran feval subroutine
# 4-dimensional stiff ODE system with widely separated rate constants

function e5!(du, u, p, t)
    # E5 problem - exact translation from Fortran feval subroutine:
    # prod1=7.89D-10*y(1)
    # prod2=1.1D7*y(1)*y(3)  
    # prod3=1.13D9*y(2)*y(3)
    # prod4=1.13D3*y(4)
    # f(1)=-prod1-prod2
    # f(2)=prod1-prod3
    # f(4)=prod2-prod4
    # f(3)=f(2)-f(4)

    prod1 = 7.89e-10 * u[1]
    prod2 = 1.1e7 * u[1] * u[3]
    prod3 = 1.13e9 * u[2] * u[3]
    prod4 = 1.13e3 * u[4]

    du[1] = -prod1 - prod2        # f(1)
    du[2] = prod1 - prod3         # f(2) 
    du[4] = prod2 - prod4         # f(4)
    du[3] = du[2] - du[4]         # f(3) = f(2) - f(4)
end

function e5(u, p, t)
    # E5 problem - exact translation from Fortran feval subroutine:
    # prod1=7.89D-10*y(1)
    # prod2=1.1D7*y(1)*y(3)  
    # prod3=1.13D9*y(2)*y(3)
    # prod4=1.13D3*y(4)
    # f(1)=-prod1-prod2
    # f(2)=prod1-prod3
    # f(4)=prod2-prod4
    # f(3)=f(2)-f(4)

    prod1 = 7.89e-10 * u[1]
    prod2 = 1.1e7 * u[1] * u[3]
    prod3 = 1.13e9 * u[2] * u[3]
    prod4 = 1.13e3 * u[4]

    du1 = -prod1 - prod2        # f(1)
    du2 = prod1 - prod3         # f(2) 
    du4 = prod2 - prod4         # f(4)
    du3 = du2 - du4         # f(3) = f(2) - f(4)
    SA[du1, du2, du3, du4]
end

# Initial conditions - system starts with all mass in species A
u0 = [1.76e-3, 0.0, 0.0, 0.0]

# Time span - long enough to see both fast and slow time scales
tspan = (0.0, 1e10)

prob = ODEProblem{true, SciMLBase.FullSpecialize}(e5!, u0, tspan)
probstatic = ODEProblem{false}(e5, SVector{4}(u0), tspan)
probbig = ODEProblem{true, SciMLBase.FullSpecialize}(e5!, big.(u0), big.(tspan))

# Generate reference solution
sol = solve(probbig, Rodas5P(), abstol = 1/10^60, reltol = 1/10^30)
probs = [prob, probstatic]
test_sol = [sol, sol]

abstols = [1.7e-28 for i in 1:2]
reltols = 10.0 .^ -(6 .+ (1:2) ./ 4)


ylabels = 10.0 .^ (-30:2:0)
plot(sol, xscale = :log10, yscale = :log10, tspan = (1e-5, 1e10),
    labels = ["A" "B" "C" "D"], yticks = ylabels)


setups = [Dict(:alg=>Rosenbrock23()),
    Dict(:alg=>Rosenbrock23(), :prob_choice => 2),
    Dict(:alg=>lsoda()),
    Dict(:alg=>Rodas5P()),
    Dict(:alg=>Rodas5P(), :prob_choice => 2),
    Dict(:alg=>FBDF()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>TRBDF2()),
    Dict(:alg=>CVODE_BDF()),
    #Dict(:alg=>rodas()),
    #Dict(:alg=>radau()),
    Dict(:alg=>RadauIIA5()),
    Dict(:alg=>ROS34PW1a())]
wp = WorkPrecisionSet(
    probs, abstols, reltols, setups; error_estimate = :l2, verbose = false, dense = false,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1e5), numruns = 10)
plot(wp)


setups = [Dict(:alg=>FBDF()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>CVODE_BDF()),
    Dict(:alg=>lsoda()),
    Dict(:alg=>ddebdf()),
    Dict(:alg=>Rodas5P()),
    Dict(:alg=>Rodas5P(), :prob_choice => 2),
    Dict(:alg=>rodas()),
    Dict(:alg=>radau()),
    Dict(:alg=>RadauIIA5())
]
wp = WorkPrecisionSet(probs, abstols, reltols, setups; verbose = false, dense = false,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1e5), numruns = 10)
plot(wp)


setups = [Dict(:alg=>Kvaerno4()),
    Dict(:alg=>Kvaerno5()),
    Dict(:alg=>CVODE_BDF()),
    Dict(:alg=>KenCarp4()),
    Dict(:alg=>KenCarp47()),
    Dict(:alg=>KenCarp47(), :prob_choice => 2),
    Dict(:alg=>KenCarp5()),
    Dict(:alg=>lsoda()),
    Dict(:alg=>Rodas5P()),
    Dict(:alg=>Rodas5P(), :prob_choice => 2),
    Dict(:alg=>ImplicitEulerExtrapolation(
        min_order = 4, init_order = 11, threading = OrdinaryDiffEq.PolyesterThreads())),
    Dict(:alg=>ImplicitEulerExtrapolation(min_order = 4, init_order = 11, threading = false)),
    Dict(:alg=>ImplicitEulerBarycentricExtrapolation(
        min_order = 4, init_order = 11, threading = OrdinaryDiffEq.PolyesterThreads())),
    Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 4, init_order = 11, threading = false)),
    Dict(:alg=>ImplicitHairerWannerExtrapolation(
        min_order = 3, init_order = 111, threading = OrdinaryDiffEq.PolyesterThreads())),
    Dict(:alg=>ImplicitHairerWannerExtrapolation(min_order = 3, init_order = 11, threading = false))
]
wp = WorkPrecisionSet(probs, abstols, reltols, setups; verbose = false, dense = false,
    save_everystep = false, appxsol = test_sol, maxiters = Int(1e5), numruns = 10)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

