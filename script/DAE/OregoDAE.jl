
using OrdinaryDiffEq, DiffEqDevTools, Sundials, ModelingToolkit, ODEInterfaceDiffEq,
      Plots, DASSL, DASKR
using LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables y1(t)=1.0 y2(t)=2.0 y3(t)=3.0
@parameters p1=77.27 p2=8.375e-6 p3=0.161

eqs = [
  D(y1) ~ p1*(y2+y1*(1-p2*y1-y2))
  D(y2) ~ (y3-(1+y1)*y2)/p1
  D(y3) ~ p3*(y1-y3)
]
@mtkbuild sys = ODESystem(eqs, t)
mtkprob = ODEProblem(sys,[],(0.0,30.0))
daeprob = DAEProblem(sys,[D(y1)=>77.26935286375,
                              D(y2)=>-0.012941633234114146,
                              D(y3)=>-0.322],[],(0.0,30.0))
odaeprob = ODAEProblem(sys,[],(0.0,30.0))

ref_sol = solve(daeprob,IDA(),abstol=1/10^14,reltol=1/10^14);
ode_ref_sol = solve(odaeprob,CVODE_BDF(),abstol=1/10^14,reltol=1/10^14);

probs = [mtkprob,daeprob,odaeprob]
refs = [ref_sol,ref_sol,ode_ref_sol];


plot(ref_sol)


abstols = 1.0 ./ 10.0 .^ (6:9)
reltols = 1.0 ./ 10.0 .^ (2:5);
setups = [Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
          Dict(:prob_choice => 1, :alg=>rodas()),
          Dict(:prob_choice => 1, :alg=>radau()),
          Dict(:prob_choice => 1, :alg=>RadauIIA5()),
          Dict(:prob_choice => 2, :alg=>DFBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
]

wp = WorkPrecisionSet(probs,abstols,reltols,setups;print_names=true,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


setups = [Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 3, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>CVODE_BDF()),
          Dict(:prob_choice => 3, :alg=>TRBDF2()),
          Dict(:prob_choice => 3, :alg=>KenCarp4()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (2:4);
setups = [Dict(:prob_choice => 3, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 2, :alg=>DASSL.dassl()),
          Dict(:prob_choice => 2, :alg=>DASKR.daskr()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:9)
reltols = 1.0 ./ 10.0 .^ (2:5);
setups = [Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
          Dict(:prob_choice => 1, :alg=>rodas()),
          Dict(:prob_choice => 1, :alg=>radau()),
          Dict(:prob_choice => 1, :alg=>RadauIIA5()),
          Dict(:prob_choice => 2, :alg=>DFBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          ]
gr()
wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:9)
reltols = 1.0 ./ 10.0 .^ (2:5);
setups = [Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 3, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>CVODE_BDF()),
          Dict(:prob_choice => 3, :alg=>TRBDF2()),
          Dict(:prob_choice => 3, :alg=>KenCarp4()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (7:12)
reltols = 1.0 ./ 10.0 .^ (4:9)

setups = [Dict(:prob_choice => 1, :alg=>Rodas5()),
          Dict(:prob_choice => 3, :alg=>Rodas5()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
          Dict(:prob_choice => 1, :alg=>rodas()),
          Dict(:prob_choice => 1, :alg=>radau()),
          Dict(:prob_choice => 1, :alg=>RadauIIA5()),
          Dict(:prob_choice => 2, :alg=>DFBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          ]
gr()
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

