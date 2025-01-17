
using DiffEqDevTools, Sundials, ODEInterfaceDiffEq,
      Plots, DASSL, DASKR
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra

@parameters begin
      Ub=6.0
      UF=0.026
      α=0.99
      β=1e-6
      R₀=1e3
      R₁=9e3
      R₂=9e3
      R₃=9e3
      R₄=9e3
      R₅=9e3
      R₆=9e3
      R₇=9e3
      R₈=9e3
      R₉=9e3
      C₁=1e-6
      C₂=2e-6
      C₃=3e-6
      C₄=4e-6
      C₅=5e-6
end

@variables begin
      y₁(t)  = 0.0
      y₂(t)  = 3.0 # Ub/(R₂/R₁ + 1)
      y₃(t)  = 3.0
      y₄(t)  = 6.0
      y₅(t)  = 3.0 # Ub/(R₆/R₅ + 1)
      y₆(t)  = 3.0
      y₇(t)  = 6.0
      y₈(t)  = 0.0
      tmp1(t)
      tmp2(t)
      tmp3(t)
      tmp4(t)
      tmp5(t)
      tmp6(t)
end

Uₑ = 0.1sin(200π * t)
g(x) = β * (exp(x / UF) - 1)

eqs = [
     tmp1 ~ (-Uₑ / R₀ + y₁ / R₀) / C₁
     tmp2 ~ (-Ub / R₂ + y₂ * (1 / R₁ + 1 / R₂) - (α - 1) * g(y₂ - y₃))/C₁
     D(y₂) - D(y₁) ~ tmp1
     D(y₁) - D(y₂) ~ tmp2
     -C₂ * D(y₃)                ~ -g(y₂ - y₃) + y₃/R₃
     tmp5 ~ (-Ub / R₄ + y₄ / R₄ + α * g(y₂ - y₃))/C₃
     tmp6 ~ (-Ub / R₆ + y₅ * (1 / R₅ + 1 / R₆) - (α - 1) * g(y₅ - y₆))/C₃
     D(y₅) - D(y₄)   ~ tmp5
     -D(y₅) + D(y₄)   ~ tmp6
     -C₄ * D(y₆)                ~ -g(y₅ - y₆) + y₆ / R₇
     tmp3 ~ (-Ub / R₈ + y₇ / R₈ + α * g(y₅ - y₆) )/C₅
     tmp4 ~ (y₈ / R₉) / C₅
     -D(y₇) + D(y₈)   ~ tmp3
     D(y₇) - D(y₈)   ~ tmp4
]

u0 = [y₁  => 0.0
      y₂  => 3.0
      y₃  => 3.0
      y₄  => 6.0
      y₅  => 3.0
      y₆  => 3.0
      y₇  => 6.0
      y₈  => 0.0
]

@mtkbuild sys = ODESystem(eqs, t)
tspan = (0.0, 0.2)
mtkprob   = ODEProblem(sys, u0, tspan)
ref_sol = solve(mtkprob, Rodas5P(), abstol = 1e-10, reltol = 1e-10)

du = mtkprob.f(mtkprob.u0, mtkprob.p, 0.0)
du0 = D.(unknowns(sys)) .=> du
daeprob = DAEProblem(sys, du0, [], tspan)
dae_ref_sol = solve(daeprob, IDA(), abstol = 1/10^7, reltol = 1/10^7)

function transamp(du, u, p, t)
    y₁, y₂, y₃, y₄, y₅, y₆, y₇, y₈ = u
    Uₑ = 0.1sin(200π * t)
    Ub=6.0
    UF=0.026
    α=0.99
    β=1e-6
    R₀=1e3
    R₁=9e3
    R₂=9e3
    R₃=9e3
    R₄=9e3
    R₅=9e3
    R₆=9e3
    R₇=9e3
    R₈=9e3
    R₉=9e3
    C₁=1e-6
    C₂=2e-6
    C₃=3e-6
    C₄=4e-6
    C₅=5e-6
    g(x) = β * (exp(x / UF) - 1)

    du[1] = -Uₑ / R₀ + y₁ / R₀
    du[2] = -Ub / R₂ + y₂ * (1 / R₁ + 1 / R₂) - (α - 1) * g(y₂ - y₃)
    du[3] = -g(y₂ - y₃) + y₃/R₃
    du[4] = -Ub / R₄ + y₄ / R₄ + α * g(y₂ - y₃)
    du[5] = -Ub / R₆ + y₅ * (1 / R₅ + 1 / R₆) - (α - 1) * g(y₅ - y₆)
    du[6] = -g(y₅ - y₆) + y₆ / R₇
    du[7] = -Ub / R₈ + y₇ / R₈ + α * g(y₅ - y₆)
    du[8] = y₈ / R₉
    nothing
end

dirMassMatrix =
    Float64.(ModelingToolkit.unwrap.(substitute.(
                [-C₁  C₁  0   0   0   0   0   0
                  C₁ -C₁  0   0   0   0   0   0
                  0   0  -C₂  0   0   0   0   0
                  0   0   0  -C₃  C₃  0   0   0
                  0   0   0   C₃ -C₃  0   0   0
                  0   0   0   0   0 -C₄   0   0
                  0   0   0   0   0   0  -C₅  C₅
                  0   0   0   0   0   0   C₅ -C₅],
                 (parameters(sys) .=> ModelingToolkit.getdefault.(parameters(sys)),))))
mmf = ODEFunction(transamp, mass_matrix = dirMassMatrix)
mmprob = ODEProblem(mmf, [0.0,3.0,3.0,6.0,3.0,3.0,6.0,0.0], tspan)
mm_refsol = solve(mmprob, Rodas5(), reltol = 1e-12, abstol = 1e-12)

probs = [mtkprob,daeprob,mmprob]
refs = [ref_sol,ref_sol,mm_refsol];


plot(ref_sol, idxs = [y₁,y₂,y₃,y₄,y₅,y₆,y₇,y₈])


plot(mm_refsol)


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
setups = [Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
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
          Dict(:prob_choice => 3, :alg=>Rodas5P()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>rodas()),
          Dict(:prob_choice => 3, :alg=>FBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
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
          Dict(:prob_choice => 3, :alg=>Rodas5P()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>rodas()),
          Dict(:prob_choice => 3, :alg=>FBDF()),
          Dict(:prob_choice => 2, :alg=>IDA()),
          Dict(:prob_choice => 2, :alg=>DASKR.daskr()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;error_estimate = :l2,
                      save_everystep=false,appxsol=refs,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (7:12)
reltols = 1.0 ./ 10.0 .^ (4:9)

setups = [Dict(:prob_choice => 1, :alg=>Rodas5P()),
          Dict(:prob_choice => 3, :alg=>Rodas5P()),
          Dict(:prob_choice => 1, :alg=>Rodas4()),
          Dict(:prob_choice => 3, :alg=>Rodas4()),
          Dict(:prob_choice => 1, :alg=>FBDF()),
          Dict(:prob_choice => 1, :alg=>QNDF()),
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

