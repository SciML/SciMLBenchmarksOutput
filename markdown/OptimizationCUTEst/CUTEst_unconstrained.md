---
author: "Alonso M. Cisneros"
title: "CUTEst Unconstrained Nonlinear Optimization Benchmarks"
---


# Introduction

CUTEst, the Constrained and Unconstrained Testing Environment, is a collection of around 1500 problems for general nonlinear optimization used to test optimization routines. The wrapper [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) provides convenient access to the problem collection, which we can leverage to test the optimizers made available by Optimization.jl.


## Unconstrained problems

CUTEst contains 286 unconstrained problems. We will compare how the optimizers behave in
terms of the time to solution with respect to the number of variables.


```julia
using Optimization
using OptimizationNLPModels
using CUTEst
using OptimizationOptimJL
using OptimizationOptimisers
using OptimizationOptimJL: LBFGS, ConjugateGradient, NelderMead, SimulatedAnnealing, ParticleSwarm
using Ipopt
using OptimizationMOI
using OptimizationMOI: MOI as MOI
using DataFrames
using Plots
using StatsPlots
using StatsBase: countmap


optimizers = [
    ("LBFGS", LBFGS()),
    ("ConjugateGradient", ConjugateGradient()),
    ("NelderMead", NelderMead()),
    ("SimulatedAnnealing", SimulatedAnnealing()),
    ("ParticleSwarm", ParticleSwarm()),
]

function get_stats(sol, optimizer_name)
    if hasfield(typeof(sol), :stats) && hasfield(typeof(sol.stats), :time)
        solve_time = sol.stats.time
    else
        solve_time = NaN
    end
    return (length(sol.u), solve_time, optimizer_name, Symbol(sol.retcode))
end

function run_benchmarks(problems, optimizers; chunk_size=1)
    problem = String[]
    n_vars = Int64[]
    secs = Float64[]
    solver = String[]
    retcode = Symbol[]
    optz = length(optimizers)
    n = length(problems)
    @info "Processing $(n) problems with $(optz) optimizers in chunks of $(chunk_size)"
    broadcast(c -> sizehint!(c, optz * n), [problem, n_vars, secs, solver, retcode])
    for chunk_start in 1:chunk_size:n
        chunk_end = min(chunk_start + chunk_size - 1, n)
        chunk_problems = problems[chunk_start:chunk_end]
        @info "Processing chunk $(div(chunk_start-1, chunk_size)+1)/$(div(n-1, chunk_size)+1): problems $(chunk_start)-$(chunk_end)"
        for (idx, prob_name) in enumerate(chunk_problems)
            current_problem = chunk_start + idx - 1
            @info "Problem $(current_problem)/$(n): $(prob_name)"
            nlp_prob = nothing
            try
                nlp_prob = CUTEstModel(prob_name)
                if nlp_prob.meta.nvar > 10000
                    @info "  Skipping $(prob_name) (too large: $(nlp_prob.meta.nvar) variables)"
                    finalize(nlp_prob)
                    continue
                end
                prob = OptimizationNLPModels.OptimizationProblem(nlp_prob, Optimization.AutoFiniteDiff())
                for (optimizer_name, optimizer) in optimizers
                    try
                        sol = solve(prob, optimizer; maxiters = 1000, maxtime = 30.0)
                        @info "✓ Solved $(prob_name) with $(optimizer_name) - Status: $(sol.retcode)"
                        vars, time, alg, code = get_stats(sol, optimizer_name)
                        push!(problem, prob_name)
                        push!(n_vars, vars)
                        push!(secs, time)
                        push!(solver, alg)
                        push!(retcode, code)
                    catch e
                        push!(problem, prob_name)
                        push!(n_vars, nlp_prob !== nothing ? nlp_prob.meta.nvar : -1)
                        push!(secs, NaN)
                        push!(solver, optimizer_name)
                        push!(retcode, :FAILED)
                    end
                end
            catch e
                for (optimizer_name, optimizer) in optimizers
                    push!(problem, prob_name)
                    push!(n_vars, -1)
                    push!(secs, NaN)
                    push!(solver, optimizer_name)
                    push!(retcode, :LOAD_FAILED)
                end
            finally
                if nlp_prob !== nothing
                    try
                        finalize(nlp_prob)
                    catch e
                    end
                end
            end
        end
        GC.gc()
        @info "Completed chunk, memory usage cleaned up"
    end
    return DataFrame(problem = problem, n_vars = n_vars, secs = secs, solver = solver, retcode = retcode)
end

unc_problems = collect(CUTEst.select_sif_problems(contype="unc"))
println("Number of problems: ", length(unc_problems))
println("First 5 problems: ", unc_problems[1:min(5, end)])
unc_problems = unc_problems[1:min(50, length(unc_problems))]
println("Limited to ", length(unc_problems), " problems for comprehensive testing")
unc_results =  run_benchmarks(unc_problems, optimizers)
@show unc_results
successful_codes = [:Success, :MaxIters, :MaxTime, :FirstOrderOptimal]
successful_results = filter(row -> row.retcode in successful_codes, unc_results)
total_attempts = nrow(unc_results)
successful_attempts = nrow(successful_results)
success_rate = total_attempts > 0 ? round(successful_attempts / total_attempts * 100, digits=1) : 0
println("SUCCESS RATE ANALYSIS:")
println("Total attempts: ", total_attempts)
println("Successful attempts: ", successful_attempts)
println("Success rate: ", success_rate, "%")
println("Return code distribution:")
if total_attempts > 0
    for (code, count) in sort(collect(pairs(countmap(unc_results.retcode))), by=x->x[2], rev=true)
        println("  ", code, ": ", count, " occurrences")
    end
else
    println("  No results to analyze")
end
@df unc_results scatter(:n_vars, :secs,
        group = :solver,
        xlabel = "n. variables",
        ylabel = "secs.",
        title = "Time to solution by optimizer and number of vars",
    )
```

```
Number of problems: 293
First 5 problems: ["LIARWHD", "SCHMVETT", "LUKSAN13LS", "VAREIGVL", "JUDGE"
]
Limited to 50 problems for comprehensive testing
unc_results = 240×5 DataFrame
 Row │ problem     n_vars  secs           solver              retcode
     │ String      Int64   Float64        String              Symbol
─────┼────────────────────────────────────────────────────────────────
   1 │ LIARWHD       5000    1.28568      LBFGS               Success
   2 │ LIARWHD       5000    0.67915      ConjugateGradient   Success
   3 │ LIARWHD       5000   13.9192       NelderMead          Failure
   4 │ LIARWHD       5000    0.343245     SimulatedAnnealing  Failure
   5 │ LIARWHD       5000   33.5615       ParticleSwarm       Failure
   6 │ SCHMVETT      5000    0.13994      LBFGS               Success
   7 │ SCHMVETT      5000    0.095603     ConjugateGradient   Success
   8 │ SCHMVETT      5000   14.124        NelderMead          Failure
   9 │ SCHMVETT      5000    0.422547     SimulatedAnnealing  Failure
  10 │ SCHMVETT      5000   37.0636       ParticleSwarm       Failure
  11 │ LUKSAN13LS      98    0.0036242    LBFGS               Success
  12 │ LUKSAN13LS      98    0.00366998   ConjugateGradient   Success
  13 │ LUKSAN13LS      98    0.0117111    NelderMead          Failure
  14 │ LUKSAN13LS      98    0.00560713   SimulatedAnnealing  Failure
  15 │ LUKSAN13LS      98    0.442637     ParticleSwarm       Failure
  16 │ VAREIGVL      5000    0.0725961    LBFGS               Success
  17 │ VAREIGVL      5000    0.707599     ConjugateGradient   Success
  18 │ VAREIGVL      5000   13.5195       NelderMead          Failure
  19 │ VAREIGVL      5000    0.165379     SimulatedAnnealing  Failure
  20 │ VAREIGVL      5000   34.2195       ParticleSwarm       Failure
  21 │ JUDGE            2    0.000153065  LBFGS               Success
  22 │ JUDGE            2    0.000109911  ConjugateGradient   Success
  23 │ JUDGE            2    0.000216961  NelderMead          Success
  24 │ JUDGE            2    0.00303984   SimulatedAnnealing  Failure
  25 │ JUDGE            2    0.00658894   ParticleSwarm       Failure
  26 │ DIXMAANJ      3000    0.652868     LBFGS               Failure
  27 │ DIXMAANJ      3000    0.277309     ConjugateGradient   Failure
  28 │ DIXMAANJ      3000    5.12163      NelderMead          Failure
  29 │ DIXMAANJ      3000    0.0579059    SimulatedAnnealing  Failure
  30 │ DIXMAANJ      3000   31.0944       ParticleSwarm       Failure
  31 │ FBRAIN3LS        6    3.57757      LBFGS               Failure
  32 │ FBRAIN3LS        6    2.07655      ConjugateGradient   Failure
  33 │ FBRAIN3LS        6    0.559859     NelderMead          Failure
  34 │ FBRAIN3LS        6    0.362412     SimulatedAnnealing  Failure
  35 │ FBRAIN3LS        6    2.52477      ParticleSwarm       Failure
  36 │ SPIN2LS        102    0.039355     LBFGS               Success
  37 │ SPIN2LS        102    0.016027     ConjugateGradient   Success
  38 │ SPIN2LS        102    0.235501     NelderMead          Failure
  39 │ SPIN2LS        102    0.0204189    SimulatedAnnealing  Failure
  40 │ SPIN2LS        102    1.95174      ParticleSwarm       Failure
  41 │ SBRYBND       5000    2.17019      LBFGS               Failure
  42 │ SBRYBND       5000    0.958153     ConjugateGradient   Failure
  43 │ SBRYBND       5000   13.7104       NelderMead          Failure
  44 │ SBRYBND       5000    0.156268     SimulatedAnnealing  Failure
  45 │ SBRYBND       5000   33.7482       ParticleSwarm       Failure
  46 │ ARGLINC        200    0.05039      LBFGS               Failure
  47 │ ARGLINC        200    5.68835      ConjugateGradient   Failure
  48 │ ARGLINC        200    0.209398     NelderMead          Failure
  49 │ ARGLINC        200    0.0962369    SimulatedAnnealing  Failure
  50 │ ARGLINC        200   19.7177       ParticleSwarm       Failure
  51 │ TOINTGOR        50    0.00340319   LBFGS               Success
  52 │ TOINTGOR        50    0.00258589   ConjugateGradient   Success
  53 │ TOINTGOR        50    0.00784397   NelderMead          Failure
  54 │ TOINTGOR        50    0.00489497   SimulatedAnnealing  Failure
  55 │ TOINTGOR        50    0.147895     ParticleSwarm       Failure
  56 │ DIXMAANC      3000    0.00581217   LBFGS               Success
  57 │ DIXMAANC      3000    0.00369191   ConjugateGradient   Success
  58 │ DIXMAANC      3000    5.29081      NelderMead          Failure
  59 │ DIXMAANC      3000    0.058157     SimulatedAnnealing  Failure
  60 │ DIXMAANC      3000   30.9962       ParticleSwarm       Failure
  61 │ WAYSEA2          2    0.00019908   LBFGS               Success
  62 │ WAYSEA2          2    0.000185966  ConjugateGradient   Success
  63 │ WAYSEA2          2    0.000154018  NelderMead          Success
  64 │ WAYSEA2          2    0.00249887   SimulatedAnnealing  Failure
  65 │ WAYSEA2          2    0.00578904   ParticleSwarm       Failure
  66 │ BROWNDEN         4    0.000207901  LBFGS               Success
  67 │ BROWNDEN         4    0.000404119  ConjugateGradient   Success
  68 │ BROWNDEN         4    0.00077796   NelderMead          Success
  69 │ BROWNDEN         4    0.00321293   SimulatedAnnealing  Failure
  70 │ BROWNDEN         4    0.0077672    ParticleSwarm       Failure
  71 │ HILBERTA         2    5.91278e-5   LBFGS               Success
  72 │ HILBERTA         2    3.40939e-5   ConjugateGradient   Success
  73 │ HILBERTA         2    0.000133991  NelderMead          Success
  74 │ HILBERTA         2    0.00270796   SimulatedAnnealing  Failure
  75 │ HILBERTA         2    0.00593019   ParticleSwarm       Failure
  76 │ DMN37142LS      66   11.9498       LBFGS               Failure
  77 │ DMN37142LS      66    8.76718      ConjugateGradient   Failure
  78 │ DMN37142LS      66    0.977412     NelderMead          Failure
  79 │ DMN37142LS      66    0.617031     SimulatedAnnealing  Failure
  80 │ DMN37142LS      66   30.0747       ParticleSwarm       Failure
  81 │ DIXMAANE1     3000    0.128823     LBFGS               Success
  82 │ DIXMAANE1     3000    0.0775619    ConjugateGradient   Success
  83 │ DIXMAANE1     3000    5.15211      NelderMead          Failure
  84 │ DIXMAANE1     3000    0.0447609    SimulatedAnnealing  Failure
  85 │ DIXMAANE1     3000   30.8996       ParticleSwarm       Failure
  86 │ PALMER5D         4    0.000102043  LBFGS               Success
  87 │ PALMER5D         4    8.82149e-5   ConjugateGradient   Success
  88 │ PALMER5D         4    0.00101995   NelderMead          Success
  89 │ PALMER5D         4    0.00291109   SimulatedAnnealing  Failure
  90 │ PALMER5D         4    0.00706482   ParticleSwarm       Failure
  91 │ BOXBODLS         2    0.000302792  LBFGS               Success
  92 │ BOXBODLS         2    0.000472069  ConjugateGradient   Success
  93 │ BOXBODLS         2    0.00027895   NelderMead          Success
  94 │ BOXBODLS         2    0.00458908   SimulatedAnnealing  Failure
  95 │ BOXBODLS         2    0.01073      ParticleSwarm       Failure
  96 │ HIMMELBB         2    0.000123978  LBFGS               Success
  97 │ HIMMELBB         2    0.000169039  ConjugateGradient   Success
  98 │ HIMMELBB         2    9.20296e-5   NelderMead          Success
  99 │ HIMMELBB         2    0.00303292   SimulatedAnnealing  Failure
 100 │ HIMMELBB         2    0.00590086   ParticleSwarm       Failure
 101 │ ENGVAL2          3    0.000246048  LBFGS               Success
 102 │ ENGVAL2          3    0.000607967  ConjugateGradient   Success
 103 │ ENGVAL2          3    0.000550032  NelderMead          Success
 104 │ ENGVAL2          3    0.00315118   SimulatedAnnealing  Failure
 105 │ ENGVAL2          3    0.00574422   ParticleSwarm       Failure
 106 │ MUONSINELS       1    0.0962291    LBFGS               Failure
 107 │ MUONSINELS       1    0.000657082  ConjugateGradient   Success
 108 │ MUONSINELS       1    0.000166178  NelderMead          Success
 109 │ MUONSINELS       1    0.015193     SimulatedAnnealing  Failure
 110 │ MUONSINELS       1    0.05567      ParticleSwarm       Failure
 111 │ ENSOLS           9    0.00483084   LBFGS               Success
 112 │ ENSOLS           9    0.00836897   ConjugateGradient   Success
 113 │ ENSOLS           9    0.0357702    NelderMead          Success
 114 │ ENSOLS           9    0.0264409    SimulatedAnnealing  Failure
 115 │ ENSOLS           9    0.239312     ParticleSwarm       Failure
 116 │ PRICE4           2    0.00012207   LBFGS               Success
 117 │ PRICE4           2    0.00013113   ConjugateGradient   Success
 118 │ PRICE4           2    0.000201941  NelderMead          Success
 119 │ PRICE4           2    0.00308895   SimulatedAnnealing  Failure
 120 │ PRICE4           2    0.00597906   ParticleSwarm       Failure
 121 │ EIGENALS      2550   12.3316       LBFGS               Failure
 122 │ EIGENALS      2550    4.76671      ConjugateGradient   Failure
 123 │ EIGENALS      2550    5.6711       NelderMead          Failure
 124 │ EIGENALS      2550    0.565485     SimulatedAnnealing  Failure
 125 │ EIGENALS      2550   31.6766       ParticleSwarm       Failure
 126 │ CERI651ELS       7    0.0300519    LBFGS               Success
 127 │ CERI651ELS       7    0.068429     ConjugateGradient   Failure
 128 │ CERI651ELS       7    6.31809e-5   NelderMead          Failure
 129 │ CERI651ELS       7    0.00650382   SimulatedAnnealing  Failure
 130 │ CERI651ELS       7    0.036571     ParticleSwarm       Failure
 131 │ GENHUMPS      5000    2.76571      LBFGS               Failure
 132 │ GENHUMPS      5000    2.07636      ConjugateGradient   Failure
 133 │ GENHUMPS      5000   15.3474       NelderMead          Failure
 134 │ GENHUMPS      5000    0.444946     SimulatedAnnealing  Failure
 135 │ GENHUMPS      5000   32.9061       ParticleSwarm       Failure
 136 │ OSCIPATH       500    0.00153208   LBFGS               Success
 137 │ OSCIPATH       500    0.00119996   ConjugateGradient   Success
 138 │ OSCIPATH       500    0.102505     NelderMead          Failure
 139 │ OSCIPATH       500    0.0119679    SimulatedAnnealing  Failure
 140 │ OSCIPATH       500    9.81063      ParticleSwarm       Failure
 141 │ FLETCBV2      5000    1.93324      LBFGS               Failure
 142 │ FLETCBV2      5000    0.843986     ConjugateGradient   Failure
 143 │ FLETCBV2      5000   14.1034       NelderMead          Failure
 144 │ FLETCBV2      5000    0.261064     SimulatedAnnealing  Failure
 145 │ FLETCBV2      5000   35.4844       ParticleSwarm       Failure
 146 │ DIXMAAND      3000    0.00723696   LBFGS               Success
 147 │ DIXMAAND      3000    0.00511599   ConjugateGradient   Success
 148 │ DIXMAAND      3000    5.36648      NelderMead          Failure
 149 │ DIXMAAND      3000    0.0588238    SimulatedAnnealing  Failure
 150 │ DIXMAAND      3000   30.771        ParticleSwarm       Failure
 151 │ SISSER           2    0.00010705   LBFGS               Success
 152 │ SISSER           2    0.000102997  ConjugateGradient   Success
 153 │ SISSER           2    9.48906e-5   NelderMead          Success
 154 │ SISSER           2    0.00296092   SimulatedAnnealing  Failure
 155 │ SISSER           2    0.00613713   ParticleSwarm       Failure
 156 │ TRIGON1         10    0.000524044  LBFGS               Success
 157 │ TRIGON1         10    0.00088501   ConjugateGradient   Success
 158 │ TRIGON1         10    0.00461102   NelderMead          Success
 159 │ TRIGON1         10    0.00534678   SimulatedAnnealing  Failure
 160 │ TRIGON1         10    0.02597      ParticleSwarm       Failure
 161 │ S308NE           2  NaN            LBFGS               FAILED
 162 │ S308NE           2  NaN            ConjugateGradient   FAILED
 163 │ S308NE           2  NaN            NelderMead          FAILED
 164 │ S308NE           2  NaN            SimulatedAnnealing  FAILED
 165 │ S308NE           2  NaN            ParticleSwarm       FAILED
 166 │ PENALTY1      1000    0.0107739    LBFGS               Success
 167 │ PENALTY1      1000    0.0050981    ConjugateGradient   Success
 168 │ PENALTY1      1000    0.246242     NelderMead          Failure
 169 │ PENALTY1      1000    0.0176408    SimulatedAnnealing  Failure
 170 │ PENALTY1      1000   30.0495       ParticleSwarm       Failure
 171 │ LRW8A          300   21.9441       LBFGS               Failure
 172 │ LRW8A          300   10.074        ConjugateGradient   Failure
 173 │ LRW8A          300    3.87486      NelderMead          Failure
 174 │ LRW8A          300    2.68621      SimulatedAnnealing  Failure
 175 │ LRW8A          300   30.8375       ParticleSwarm       Failure
 176 │ SPMSRTLS      4999    0.460211     LBFGS               Success
 177 │ SPMSRTLS      4999    0.200958     ConjugateGradient   Success
 178 │ SPMSRTLS      4999   14.2561       NelderMead          Failure
 179 │ SPMSRTLS      4999    0.128738     SimulatedAnnealing  Failure
 180 │ SPMSRTLS      4999   33.1972       ParticleSwarm       Failure
 181 │ NONCVXUN      5000    2.37591      LBFGS               Failure
 182 │ NONCVXUN      5000    1.05323      ConjugateGradient   Failure
 183 │ NONCVXUN      5000   14.6716       NelderMead          Failure
 184 │ NONCVXUN      5000    0.304772     SimulatedAnnealing  Failure
 185 │ NONCVXUN      5000   34.0975       ParticleSwarm       Failure
 186 │ BRYBND        5000    0.105322     LBFGS               Success
 187 │ BRYBND        5000    0.037384     ConjugateGradient   Success
 188 │ BRYBND        5000   14.3121       NelderMead          Failure
 189 │ BRYBND        5000    0.141034     SimulatedAnnealing  Failure
 190 │ BRYBND        5000   33.1568       ParticleSwarm       Failure
 191 │ GROWTHLS         3    6.79493e-5   LBFGS               Success
 192 │ GROWTHLS         3    3.69549e-5   ConjugateGradient   Success
 193 │ GROWTHLS         3    0.00138998   NelderMead          Success
 194 │ GROWTHLS         3    0.003443     SimulatedAnnealing  Failure
 195 │ GROWTHLS         3    0.0117629    ParticleSwarm       Failure
 196 │ SINEVAL          2    0.000437021  LBFGS               Success
 197 │ SINEVAL          2    0.000370026  ConjugateGradient   Success
 198 │ SINEVAL          2    0.000664949  NelderMead          Success
 199 │ SINEVAL          2    0.00316405   SimulatedAnnealing  Failure
 200 │ SINEVAL          2    0.00663614   ParticleSwarm       Failure
 201 │ GAUSS2LS         8    0.00978899   LBFGS               Success
 202 │ GAUSS2LS         8    0.0712101    ConjugateGradient   Failure
 203 │ GAUSS2LS         8    0.0122428    NelderMead          Success
 204 │ GAUSS2LS         8    0.014828     SimulatedAnnealing  Failure
 205 │ GAUSS2LS         8    0.12431      ParticleSwarm       Failure
 206 │ STRATEC         10    3.23459      LBFGS               Success
 207 │ STRATEC         10    8.13048      ConjugateGradient   Failure
 208 │ STRATEC         10    0.460869     NelderMead          Failure
 209 │ STRATEC         10    0.324659     SimulatedAnnealing  Failure
 210 │ STRATEC         10    3.54688      ParticleSwarm       Failure
 211 │ NELSONLS         3    0.0416131    LBFGS               Failure
 212 │ NELSONLS         3    0.002249     ConjugateGradient   Success
 213 │ NELSONLS         3    0.00459981   NelderMead          Success
 214 │ NELSONLS         3    0.00610399   SimulatedAnnealing  Failure
 215 │ NELSONLS         3    0.015919     ParticleSwarm       Failure
 216 │ HYDCAR6LS       29    0.0344939    LBFGS               Failure
 217 │ HYDCAR6LS       29    0.0175531    ConjugateGradient   Failure
 218 │ HYDCAR6LS       29    0.00751591   NelderMead          Failure
 219 │ HYDCAR6LS       29    0.00569606   SimulatedAnnealing  Failure
 220 │ HYDCAR6LS       29    0.0998991    ParticleSwarm       Failure
 221 │ DQRTIC        5000    0.0313861    LBFGS               Success
 222 │ DQRTIC        5000    0.00918698   ConjugateGradient   Success
 223 │ DQRTIC        5000   13.8571       NelderMead          Failure
 224 │ DQRTIC        5000    0.06336      SimulatedAnnealing  Failure
 225 │ DQRTIC        5000   34.256        ParticleSwarm       Failure
 226 │ MISRA1ALS        2    0.000996113  LBFGS               Success
 227 │ MISRA1ALS        2    0.00155616   ConjugateGradient   Success
 228 │ MISRA1ALS        2    0.00100899   NelderMead          Success
 229 │ MISRA1ALS        2    0.00509691   SimulatedAnnealing  Failure
 230 │ MISRA1ALS        2    0.01249      ParticleSwarm       Failure
 231 │ WAYSEA1          2    0.000146151  LBFGS               Success
 232 │ WAYSEA1          2    0.000123978  ConjugateGradient   Success
 233 │ WAYSEA1          2    0.000144005  NelderMead          Success
 234 │ WAYSEA1          2    0.00302601   SimulatedAnnealing  Failure
 235 │ WAYSEA1          2    0.00728106   ParticleSwarm       Failure
 236 │ BOX          10000    0.0220361    LBFGS               Success
 237 │ BOX          10000    0.042733     ConjugateGradient   Success
 238 │ BOX          10000   33.1561       NelderMead          Failure
 239 │ BOX          10000    0.386975     SimulatedAnnealing  Failure
 240 │ BOX          10000   38.1831       ParticleSwarm       Failure
SUCCESS RATE ANALYSIS:
Total attempts: 240
Successful attempts: 86
Success rate: 35.8%
Return code distribution:
  Failure: 149 occurrences
  Success: 86 occurrences
  FAILED: 5 occurrences
```


![](figures/CUTEst_unconstrained_1_1.png)


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/OptimizationCUTEst","CUTEst_unconstrained.jmd")
```

Computer Information:

```
Julia Version 1.10.10
Commit 95f30e51f41 (2025-06-27 09:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 128 default, 0 interactive, 64 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953:

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/OptimizationCUTEst/Project.toml`
⌃ [1b53aba6] CUTEst v1.3.2
⌃ [a93c6f00] DataFrames v1.7.0
⌃ [b6b21f68] Ipopt v1.10.6
⌃ [b8f27783] MathOptInterface v1.42.1
⌃ [a4795742] NLPModels v0.21.5
⌅ [7f7a1694] Optimization v4.5.0
⌃ [fd9f6733] OptimizationMOI v0.5.5
⌅ [064b21be] OptimizationNLPModels v0.0.2
⌃ [36348300] OptimizationOptimJL v0.4.3
⌃ [42dfb2eb] OptimizationOptimisers v0.3.8
⌃ [91a5bcdd] Plots v1.40.17
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [2913bbd2] StatsBase v0.34.6
⌃ [f3b207a7] StatsPlots v0.15.7
  [de0858da] Printf
  [10745b16] Statistics v1.10.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/OptimizationCUTEst/Manifest.toml`
⌃ [47edcb42] ADTypes v1.16.0
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
⌃ [7d9f7c33] Accessors v0.1.42
⌃ [79e6a3ab] Adapt v4.3.0
  [66dad0bd] AliasTables v1.1.3
  [ec485272] ArnoldiMethod v0.4.0
  [7d9fca2a] Arpack v0.5.4
⌃ [4fba245c] ArrayInterface v7.19.0
⌃ [4c555306] ArrayLayouts v1.11.2
  [13072b0f] AxisAlgorithms v1.1.0
⌃ [6e4b80f9] BenchmarkTools v1.6.0
  [e2ed5e7c] Bijections v0.2.2
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [8e7c35d0] BlockArrays v1.7.0
⌃ [70df07ce] BracketingNonlinearSolve v1.3.0
⌃ [2a0fbf3d] CPUSummary v0.2.6
⌃ [1b53aba6] CUTEst v1.3.2
⌃ [d360d2e6] ChainRulesCore v1.25.2
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [aaaa29a8] Clustering v0.15.8
  [523fee87] CodecBzip2 v0.8.5
  [944b1d66] CodecZlib v0.7.8
⌃ [35d6a980] ColorSchemes v3.30.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌃ [861a8166] Combinatorics v1.0.3
⌅ [a80b9123] CommonMark v0.9.1
⌃ [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
⌃ [34da2185] Compat v4.18.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.5.0
⌃ [8f4d0f93] Conda v1.10.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
⌃ [a93c6f00] DataFrames v1.7.0
⌅ [864edb3b] DataStructures v0.18.22
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v6.181.0
⌃ [459566f4] DiffEqCallbacks v4.8.0
⌃ [77a26b50] DiffEqNoiseProcess v5.24.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌃ [a0c0ee7d] DifferentiationInterface v0.7.4
⌃ [8d63f2c5] DispatchDoctor v0.4.26
  [b4f34e82] Distances v0.10.12
⌃ [31c24e10] Distributions v0.25.120
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.7.16
⌃ [7c1d4256] DynamicPolynomials v0.6.2
⌃ [06fc5a27] DynamicQuantities v1.8.0
  [4e289a0a] EnumX v1.0.5
⌃ [f151be2c] EnzymeCore v0.8.12
  [460bff9d] ExceptionUnwrapping v0.1.11
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
⌃ [c87230d0] FFMPEG v0.4.4
⌃ [7a1cc6ca] FFTW v1.9.0
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
⌃ [a4df4552] FastPower v1.1.3
⌃ [1a297f60] FillArrays v1.13.0
⌃ [64ca27bc] FindFirstFunctions v1.4.1
⌃ [6a86dc24] FiniteDiff v2.27.0
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
⌃ [f6369f11] ForwardDiff v1.0.1
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.5.2
  [46192b85] GPUArraysCore v0.2.0
⌃ [28b8d3ca] GR v0.73.17
⌃ [d7ba0133] Git v1.4.0
⌃ [c27321d9] Glob v1.3.1
⌃ [86223c79] Graphs v1.13.0
  [42e2da0e] Grisu v1.0.2
⌃ [cd3eb016] HTTP v1.10.17
  [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.28
⌃ [7073ff75] IJulia v1.29.2
  [615f187c] IfElse v0.1.1
⌅ [3263718b] ImplicitDiscreteSolve v0.1.3
  [d25df0c9] Inflate v0.1.5
⌃ [842dd82b] InlineStrings v1.4.4
  [18e54dd8] IntegerMathUtils v0.1.3
⌅ [a98d9a8b] Interpolations v0.15.1
⌃ [8197267c] IntervalSets v0.7.11
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
⌃ [b6b21f68] Ipopt v1.10.6
⌃ [92d709cd] IrrationalConstants v0.2.4
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.7.1
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [ae98c720] Jieko v0.2.1
⌃ [98e50ef6] JuliaFormatter v2.1.6
⌅ [70703baa] JuliaSyntax v0.4.10
⌃ [ccbc3e58] JumpProcesses v9.16.1
⌃ [5ab0869b] KernelDensity v0.6.10
⌃ [ba0b0d4f] Krylov v0.10.1
  [5be7bae1] LBFGSB v0.4.1
  [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.8
  [10f19ff3] LayoutPointers v0.1.17
⌃ [5078a376] LazyArrays v2.6.2
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.1
⌃ [87fe0de2] LineSearch v0.1.4
⌃ [d3d80556] LineSearches v7.4.0
⌃ [5c8ed15e] LinearOperators v2.10.0
⌃ [7ed4a6bd] LinearSolve v3.25.0
  [2ab3a3ac] LogExpFunctions v0.3.29
⌃ [e6f89c97] LoggingExtras v1.1.0
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
⌃ [b8f27783] MathOptInterface v1.42.1
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
⌃ [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.2.0
⌅ [961ee093] ModelingToolkit v10.14.0
  [2e0e35c7] Moshi v0.3.7
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [102ac46a] MultivariatePolynomials v0.5.9
  [6f286f6a] MultivariateStats v0.10.3
  [ffc61752] Mustache v1.0.21
⌃ [d8a4904e] MutableArithmetics v1.6.4
⌃ [a4795742] NLPModels v0.21.5
⌅ [d41bc354] NLSolversBase v7.10.0
  [77ba4419] NaNMath v1.1.3
⌃ [b8a86587] NearestNeighbors v0.4.22
⌃ [8913a72c] NonlinearSolve v4.10.0
⌅ [be0214bd] NonlinearSolveBase v1.13.0
⌃ [5959db7a] NonlinearSolveFirstOrder v1.6.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.7.0
⌃ [26075421] NonlinearSolveSpectralMethods v1.2.0
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.17.0
⌃ [4d8831e6] OpenSSL v1.5.0
⌅ [429524aa] Optim v1.13.2
⌃ [3bd65402] Optimisers v0.4.6
⌅ [7f7a1694] Optimization v4.5.0
⌅ [bca83a33] OptimizationBase v2.10.0
⌃ [fd9f6733] OptimizationMOI v0.5.5
⌅ [064b21be] OptimizationNLPModels v0.0.2
⌃ [36348300] OptimizationOptimJL v0.4.3
⌃ [42dfb2eb] OptimizationOptimisers v0.3.8
  [bac558e1] OrderedCollections v1.8.1
⌅ [bbf590c4] OrdinaryDiffEqCore v1.26.2
⌃ [90014a1f] PDMats v0.11.35
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.3
  [ccf2f8ad] PlotThemes v3.3.0
⌃ [995b91a9] PlotUtils v1.4.3
⌃ [91a5bcdd] Plots v1.40.17
⌃ [e409e4f3] PoissonRandom v0.4.6
  [f517fe37] Polyester v0.7.18
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
⌅ [aea7be01] PrecompileTools v1.2.1
⌃ [21216c6a] Preferences v1.4.3
⌅ [08abe8d2] PrettyTables v2.4.0
  [27ebfcd6] Primes v0.5.7
⌃ [33c8b6b6] ProgressLogging v0.1.5
⌃ [92933f4c] ProgressMeter v1.10.4
  [43287f4e] PtrArrays v1.3.0
  [1fd47b50] QuadGK v2.11.2
  [be4d8f0f] Quadmath v0.5.13
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌃ [731186ca] RecursiveArrayTools v3.36.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [ae5879a3] ResettableStacks v1.1.1
⌅ [79098fc4] Rmath v0.8.0
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.15
⌃ [9dfe8606] SCCNonlinearSolve v1.4.0
  [94e857df] SIMDTypes v0.1.0
⌅ [0bca4576] SciMLBase v2.107.0
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [19f34311] SciMLJacobianOperators v0.1.8
⌃ [c0aeaf25] SciMLOperators v1.4.0
⌃ [431bcebd] SciMLPublic v1.0.0
⌃ [53ae85a6] SciMLStructures v1.7.0
  [6c6a2e73] Scratch v1.3.0
⌃ [91c51154] SentinelArrays v1.4.8
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.7.0
⌃ [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.2
⌃ [9f842d2f] SparseConnectivityTracer v1.0.0
⌃ [0a514795] SparseMatrixColorings v0.4.21
⌃ [276daf66] SpecialFunctions v2.5.1
⌃ [860ef19b] StableRNGs v1.0.3
⌃ [aedffcd0] Static v1.2.0
  [0d7ed370] StaticArrayInterface v1.8.0
⌃ [90137ffa] StaticArrays v1.9.14
⌃ [1e83bf80] StaticArraysCore v1.4.3
⌃ [82ae8749] StatsAPI v1.7.1
⌃ [2913bbd2] StatsBase v0.34.6
⌃ [4c63d2b9] StatsFuns v1.5.0
⌃ [f3b207a7] StatsPlots v0.15.7
⌃ [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
⌃ [892a3eda] StringManipulation v0.4.1
  [856f2bd8] StructTypes v1.11.0
⌃ [2efcf032] SymbolicIndexingInterface v0.3.42
⌅ [19f23fe9] SymbolicLimits v0.2.2
⌅ [d1185830] SymbolicUtils v3.29.0
⌅ [0c5d862f] Symbolics v6.48.1
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.1
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [5d786b92] TerminalLoggers v0.1.7
  [1c621080] TestItems v1.0.0
  [8290d209] ThreadingUtilities v0.5.5
  [a759f4b9] TimerOutputs v0.5.29
  [3bb67fe8] TranscodingStreams v0.11.3
⌃ [410a4b4d] Tricks v0.1.11
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.6.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌅ [1986cc42] Unitful v1.24.0
  [45397f5d] UnitfulLatexify v1.7.0
  [a7c27f48] Unityper v0.1.6
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [897b6980] WeakValueDicts v0.1.0
  [44d3d7a6] Weave v0.10.12
  [cc8bc4a8] Widgets v0.6.7
⌃ [efce3f68] WoodburyMatrices v1.0.0
⌃ [ddb6d928] YAML v0.4.14
⌃ [c2297ded] ZMQ v1.4.1
  [ae81ac8f] ASL_jll v0.1.3+0
⌅ [68821587] Arpack_jll v3.5.1+1
  [6e34b625] Bzip2_jll v1.0.9+0
⌃ [bb5f6f25] CUTEst_jll v2.5.3+0
  [83423d85] Cairo_jll v1.18.5+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.6.5+0
⌅ [b22a6f82] FFMPEG_jll v7.1.1+0
  [f5851436] FFTW_jll v3.3.11+0
⌃ [a3f928ae] Fontconfig_jll v2.16.0+0
  [d7e528f0] FreeType2_jll v2.13.4+0
  [559328eb] FriBidi_jll v1.0.17+0
⌃ [0656b61e] GLFW_jll v3.4.0+2
⌅ [d2c73de3] GR_jll v0.73.17+0
  [b0724c58] GettextRuntime_jll v0.22.4+0
⌃ [f8c6e375] Git_jll v2.50.1+0
⌃ [7746bdde] Glib_jll v2.84.3+0
  [3b182d85] Graphite2_jll v1.3.15+0
  [2e76f6c2] HarfBuzz_jll v8.5.1+0
⌃ [e33a78d0] Hwloc_jll v2.12.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
⌅ [9cc047cb] Ipopt_jll v300.1400.1701+0
⌃ [aacddb02] JpegTurbo_jll v3.1.1+0
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.0.1+0
  [1d63c593] LLVMOpenMP_jll v18.1.8+0
  [dd4b983a] LZO_jll v2.10.3+0
  [81d17ec3] L_BFGS_B_jll v3.0.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
⌃ [4b2f31a3] Libmount_jll v2.41.0+0
⌃ [89763e89] Libtiff_jll v4.7.1+0
⌃ [38a345b3] Libuuid_jll v2.41.0+0
  [d00139f3] METIS_jll v5.1.3+0
  [856f044c] MKL_jll v2025.2.0+0
⌅ [d7ed1dd3] MUMPS_seq_jll v500.800.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
⌅ [656ef2d0] OpenBLAS32_jll v0.3.24+0
⌃ [9bd350c2] OpenSSH_jll v10.0.1+0
⌅ [458c3c95] OpenSSL_jll v3.5.1+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
⌃ [91d4177d] Opus_jll v1.5.2+0
⌃ [36c8627f] Pango_jll v1.56.3+0
⌅ [30392449] Pixman_jll v0.44.2+0
⌃ [c0090381] Qt6Base_jll v6.8.2+1
  [629bc702] Qt6Declarative_jll v6.8.2+1
  [ce943373] Qt6ShaderTools_jll v6.8.2+1
⌃ [e99dba38] Qt6Wayland_jll v6.8.2+1
  [f50d1b31] Rmath_jll v0.5.1+0
⌅ [54dcf436] SIFDecode_jll v2.6.3+0
⌅ [319450e9] SPRAL_jll v2025.5.20+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
⌃ [ffd25f8a] XZ_jll v5.8.1+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.12+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.7+0
⌃ [d091e8ba] Xorg_libXfixes_jll v6.0.1+0
  [a51aa0fd] Xorg_libXi_jll v1.8.3+0
  [d1454406] Xorg_libXinerama_jll v1.1.6+0
  [ec84b674] Xorg_libXrandr_jll v1.5.5+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.3+0
⌃ [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.5+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.44.0+0
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
  [214eeab7] fzf_jll v0.61.1+0
⌃ [a4ae2306] libaom_jll v3.12.1+0
  [0ac62f75] libass_jll v0.17.4+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
⌃ [b53b4c65] libpng_jll v1.6.50+0
  [a9144af2] libsodium_jll v1.0.21+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
⌃ [1317d2d5] oneTBB_jll v2022.0.0+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
⌃ [d8fb68d0] xkbcommon_jll v1.9.2+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.10.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

