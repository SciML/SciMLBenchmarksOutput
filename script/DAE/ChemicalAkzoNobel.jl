
using OrdinaryDiffEq, DiffEqDevTools, Sundials, ModelingToolkit, ODEInterfaceDiffEq,
      Plots, DASSL, DASKR
using LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D

ModelingToolkit.@parameters begin
      k₁=18.7
      k₂=0.58
      k₃=0.09
      k₄=0.42
      kbig=34.4
      kla=3.3
      ks=115.83
      po2=0.9
      hen=737
end

@variables begin
      y₁(t)  = 0.444
      y₂(t)  = 0.00123
      y₃(t)  = 0.0
      y₄(t)  = 0.007
      y₅(t)  = 1.0
      y₆(t)  = 115.83*0.444*0.007 # ks*y₁*y₄
end

r₁  = k₁ * (y₁^4.)*sqrt(abs(y₂))
r₂  = k₂ * y₃ * y₄
r₃  = k₂/kbig * y₁ * y₅
r₄  = k₃*y₁*(y₄^2)
r₅  = k₄*(y₆^2)*sqrt(abs(y₂))
fin = kla*(po2/hen-y₂)

eqs = [
      D(y₁) ~ -2. * r₁ + r₂ - r₃ - r₄
      D(y₂) ~ -0.5 * r₁ - r₄ - 0.5*r₅ + fin
      D(y₃) ~ r₁ - r₂ + r₃
      D(y₄) ~ -r₂ + r₃ - 2. * r₄
      D(y₅) ~ r₂ - r₃ + r₅
      0. ~ ks * y₁ * y₄ - y₆
]

ModelingToolkit.@mtkbuild sys = ModelingToolkit.ODESystem(eqs, t)

tspan = (0.0, 180.0)
mtkprob = ODEProblem(sys, [], tspan)
sol = solve(mtkprob, Rodas4(),abstol=1/10^14,reltol=1/10^14)

odaeprob = ODAEProblem(sys,[],tspan)
ode_ref_sol = solve(odaeprob,CVODE_BDF(),abstol=1/10^14,reltol=1/10^14);

du = mtkprob.f(mtkprob.u0,mtkprob.p,0.0)
du0 = D.(unknowns(sys)) .=> du
daeprob = DAEProblem(sys,du0,[],tspan)
ref_sol = solve(daeprob,IDA(),abstol=1/10^14,reltol=1/10^14);

function akzo(du, u, p, t)
    y₁, y₂, y₃, y₄, y₅, y₆ = u
    k₁=18.7
    k₂=0.58
    k₃=0.09
    k₄=0.42
    kbig=34.4
    kla=3.3
    ks=115.83
    po2=0.9
    hen=737

    r₁  = k₁ * (y₁^4.)*sqrt(abs(y₂))
    r₂  = k₂ * y₃ * y₄
    r₃  = k₂/kbig * y₁ * y₅
    r₄  = k₃*y₁*(y₄^2)
    r₅  = k₄*(y₆^2)*sqrt(abs(y₂))
    fin = kla*(po2/hen-y₂)

    du[1] = -2. * r₁ + r₂ - r₃ - r₄
    du[2] = -0.5 * r₁ - r₄ - 0.5*r₅ + fin
    du[3] = r₁ - r₂ + r₃
    du[4] = -r₂ + r₃ - 2. * r₄
    du[5] = r₂ - r₃ + r₅
    du[6] = ks * y₁ * y₄ - y₆
    nothing
end
M = Matrix{Float64}(I, 6,6); M[6,6] = 0;
mmf = ODEFunction(akzo, mass_matrix = M)
mmprob = ODEProblem(mmf, [0.444,0.00123,0.0,0.007,1.0,115.83*0.444*0.007], tspan)
mm_refsol = solve(mmprob, Rodas5(), reltol = 1e-12, abstol = 1e-12)

probs = [mtkprob,daeprob,odaeprob,mmprob]
refs = [ref_sol,ref_sol,ode_ref_sol,mm_refsol];


plot(ref_sol, idxs = [y₁,y₂,y₃,y₄,y₅,y₆])


plot(mm_refsol)


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
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

wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (2:4);
setups = [Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 3, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>CVODE_BDF()),
          Dict(:prob_choice => 3, :alg=>TRBDF2()),
          Dict(:prob_choice => 3, :alg=>KenCarp4()),
          Dict(:prob_choice => 4, :alg=>Rodas4()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (3:5);
setups = [Dict(:prob_choice => 3, :alg=>Rosenbrock23()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 2, :alg=>DASSL.dassl()),
          Dict(:prob_choice => 2, :alg=>DASKR.daskr()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
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
wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (2:4);
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
          Dict(:prob_choice => 4, :alg=>Rodas5()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 4, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
          Dict(:prob_choice => 1, :alg=>rodas()),
          Dict(:prob_choice => 1, :alg=>radau()),
          Dict(:prob_choice => 1, :alg=>RadauIIA5()),
          Dict(:prob_choice => 2, :alg=>DFBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 2, :alg=>DASKR.daskr()),
          ]

wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

