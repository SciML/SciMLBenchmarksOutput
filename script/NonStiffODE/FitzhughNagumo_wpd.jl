
using OrdinaryDiffEq, ParameterizedFunctions, ODEInterface,
      ODEInterfaceDiffEq, LSODA, Sundials, DiffEqDevTools,
      StaticArrays
using Plots; gr()

f = @ode_def FitzhughNagumo begin
  dv = v - v^3/3 -w + l
  dw = τinv*(v +  a - b*w)
end a b τinv l

p = SA[0.7,0.8,1/12.5,0.5]
prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,[1.0;1.0],(0.0,10.0),p)
probstatic = ODEProblem{false}(f,SA[1.0;1.0],(0.0,10.0),p)

abstols = 1.0 ./ 10.0 .^ (6:13)
reltols = 1.0 ./ 10.0 .^ (3:10);

sol = solve(prob,Vern7(),abstol=1/10^14,reltol=1/10^14)
sol2 = solve(probstatic,Vern7(),abstol=1/10^14,reltol=1/10^14)
probs = [prob,probstatic]
test_sol = [sol,sol2];


plot(sol)


setups = [Dict(:alg=>DP5())
          #Dict(:alg=>ode45()) #fails
          Dict(:alg=>dopri5())
          Dict(:alg=>BS5())
          Dict(:alg=>Tsit5())
          Dict(:alg=>Vern6())
          Dict(:alg=>Tsit5(), :prob_choice => 2)
          Dict(:alg=>Vern6(), :prob_choice => 2)
]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,save_everystep=false,numruns=100,maxiters=1000)
plot(wp)


setups = [Dict(:alg=>DP5())
          #Dict(:alg=>ode45()) # fails
          Dict(:alg=>BS5())
          Dict(:alg=>Tsit5())
          Dict(:alg=>Vern6())
          Dict(:alg=>Tsit5(), :prob_choice => 2)
          Dict(:alg=>Vern6(), :prob_choice => 2)
]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,numruns=100,maxiters=10000,error_estimate=:L2,dense_errors=true)
plot(wp)


setups = [Dict(:alg=>DP8())
          Dict(:alg=>dop853())
          #Dict(:alg=>ode78()) # fails
          Dict(:alg=>Vern6())
          Dict(:alg=>Vern7())
          Dict(:alg=>Vern8())
          Dict(:alg=>Vern9())
          Dict(:alg=>Vern6(), :prob_choice => 2)
          Dict(:alg=>Vern7(), :prob_choice => 2)
          Dict(:alg=>Vern8(), :prob_choice => 2)
          Dict(:alg=>Vern9(), :prob_choice => 2)
]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,save_everystep=false,numruns=100,maxiters=1000)
plot(wp)


setups = [Dict(:alg=>DP8())
          Dict(:alg=>Vern7())
          Dict(:alg=>Vern7(), :prob_choice => 2)
          Dict(:alg=>CVODE_Adams())
          Dict(:alg=>ARKODE(Sundials.Explicit(),order=6))
          Dict(:alg=>lsoda())
          Dict(:alg=>odex())
          Dict(:alg=>ddeabm())
]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,save_everystep=false,numruns=100,maxiters=1000)
plot(wp)


setups = [Dict(:alg=>DP8())
          #Dict(:alg=>ode78()) # fails
          Dict(:alg=>Vern6())
          Dict(:alg=>Vern7())
          Dict(:alg=>Vern8())
          Dict(:alg=>Vern9())
          Dict(:alg=>Vern6(), :prob_choice => 2)
          Dict(:alg=>Vern7(), :prob_choice => 2)
          Dict(:alg=>Vern8(), :prob_choice => 2)
          Dict(:alg=>Vern9(), :prob_choice => 2)
]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,numruns=100,maxiters=1000,error_estimate=:L2,dense_errors=true)
plot(wp)


setups = [Dict(:alg=>Tsit5())
          Dict(:alg=>Vern9())
          Dict(:alg=>VCABM())
          Dict(:alg=>Vern9(), :prob_choice => 2)
          Dict(:alg=>VCABM(), :prob_choice => 2)
          Dict(:alg=>AitkenNeville(min_order=1, max_order=9, init_order=4, threading=true))
          Dict(:alg=>ExtrapolationMidpointDeuflhard(min_order=1, max_order=9, init_order=4, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, threading=true))]
solnames = ["Tsit5","Vern9","VCABM","Vern9 Static","VCABM Static","AitkenNeville","Midpoint Deuflhard","Midpoint Hairer Wanner"]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,names=solnames,
                      save_everystep=false,verbose=false,numruns=100)
plot(wp)


setups = [Dict(:alg=>ExtrapolationMidpointDeuflhard(min_order=1, max_order=9, init_order=9, threading=false))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, threading=false))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, sequence = :romberg, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, sequence = :bulirsch, threading=true))]
solnames = ["Deuflhard","No threads","standard","Romberg","Bulirsch"]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,names=solnames,
                      save_everystep=false,verbose=false,numruns=100)
plot(wp)


setups = [Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=10, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=11, init_order=4, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=5, max_order=11, init_order=10, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=2, max_order=15, init_order=10, threading=true))
          Dict(:alg=>ExtrapolationMidpointHairerWanner(min_order=5, max_order=7, init_order=6, threading=true))]
solnames = ["1","2","3","4","5"]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;appxsol=test_sol,names=solnames,
                      save_everystep=false,verbose=false,numruns=100)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

