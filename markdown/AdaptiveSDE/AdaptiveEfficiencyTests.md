---
author: "Chris Rackauckas"
title: "Adaptive Efficiency Tests"
---
```julia

using Distributed
addprocs(2)

p1 = Vector{Any}(undef, 3)
p2 = Vector{Any}(undef, 3)
p3 = Vector{Any}(undef, 3)

@everywhere begin
    using StochasticDiffEq, SDEProblemLibrary, DiffEqNoiseProcess, Plots,
          ParallelDataTransfer
    import SDEProblemLibrary: prob_sde_additive,
                              prob_sde_linear, prob_sde_wave
end

using StochasticDiffEq, SDEProblemLibrary, DiffEqNoiseProcess, Plots, ParallelDataTransfer
import Statistics
import SDEProblemLibrary: prob_sde_additive,
                          prob_sde_linear, prob_sde_wave

function final_error_stats(sim)
    errors = [sol.errors[:final] for sol in sim.u]
    return (;
        elapsed_time = sim.elapsedTime, mean = Statistics.mean(errors),
        median = Statistics.median(errors),
    )
end

probs = Matrix{SDEProblem}(undef, 3, 3)
## Problem 1
prob = prob_sde_linear
probs[1, 1] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM1)))
probs[1, 2] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM2)))
probs[1, 3] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM3)))
## Problem 2
prob = prob_sde_wave
probs[2, 1] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM1)))
probs[2, 2] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM2)))
probs[2, 3] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM3)))
## Problem 3
prob = prob_sde_additive
probs[3, 1] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM1)))
probs[3, 2] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM2)))
probs[3, 3] = SDEProblem(prob.f, prob.g, prob.u0, prob.tspan, prob.p,
    noise = WienerProcess(0.0, 0.0, 0.0, rswm = RSWM(adaptivealg = :RSwM3)))

fullMeans = Vector{Array}(undef, 3)
fullMedians = Vector{Array}(undef, 3)
fullElapsed = Vector{Array}(undef, 3)
fullTols = Vector{Array}(undef, 3)
offset = 0

Ns = [17, 23,
    17]
```

```
From worker 3:	Precompiling StochasticDiffEq...
      From worker 2:	Precompiling StochasticDiffEq...
      From worker 2:	    StochasticDiffEqLevyArea Being precompiled by anot
her process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/
.julia/compiled/v1.11/StochasticDiffEqLevyArea/kQAQL_LSg15.ji.pidfile)
      From worker 2:	    NonlinearSolveSpectralMethodsForwardDiffExt Being 
precompiled by another process (pid: 3964812, pidfile: /home/crackauc/githu
b-runners/amdci8-1/.julia/compiled/v1.11/NonlinearSolveSpectralMethodsForwa
rdDiffExt/jMsv6_LSg15.ji.pidfile)
      From worker 2:	    SciMLBaseDistributionsExt Being precompiled by ano
ther process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1
/.julia/compiled/v1.11/SciMLBaseDistributionsExt/JC1Lw_LSg15.ji.pidfile)
      From worker 2:	    NonlinearSolveQuasiNewtonForwardDiffExt Being prec
ompiled by another process (pid: 3964812, pidfile: /home/crackauc/github-ru
nners/amdci8-1/.julia/compiled/v1.11/NonlinearSolveQuasiNewtonForwardDiffEx
t/PT5v2_LSg15.ji.pidfile)
      From worker 2:	    OrdinaryDiffEqDifferentiation Being precompiled by
 another process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdc
i8-1/.julia/compiled/v1.11/OrdinaryDiffEqDifferentiation/zTnwv_LSg15.ji.pid
file)
      From worker 3:	   1138.1 ms  ✓ StochasticDiffEqLevyArea
      From worker 2:	   1170.3 ms  ✓ StochasticDiffEqLevyArea
      From worker 3:	   1623.1 ms  ✓ NonlinearSolveSpectralMethods → Nonlin
earSolveSpectralMethodsForwardDiffExt
      From worker 2:	   1181.4 ms  ✓ NonlinearSolveSpectralMethods → Nonlin
earSolveSpectralMethodsForwardDiffExt
      From worker 3:	   1653.0 ms  ✓ SciMLBase → SciMLBaseDistributionsExt
      From worker 2:	   1598.5 ms  ✓ SciMLBase → SciMLBaseDistributionsExt
      From worker 2:	    DiffEqNoiseProcess Being precompiled by another pr
ocess (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.julia
/compiled/v1.11/DiffEqNoiseProcess/NSRab_LSg15.ji.pidfile)
      From worker 3:	   2087.7 ms  ✓ NonlinearSolveQuasiNewton → NonlinearS
olveQuasiNewtonForwardDiffExt
      From worker 2:	   2127.9 ms  ✓ NonlinearSolveQuasiNewton → NonlinearS
olveQuasiNewtonForwardDiffExt
      From worker 2:	    NonlinearSolve Being precompiled by another proces
s (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.julia/com
piled/v1.11/NonlinearSolve/8YWaI_LSg15.ji.pidfile)
      From worker 3:	   3611.4 ms  ✓ OrdinaryDiffEqDifferentiation
      From worker 2:	   4069.5 ms  ✓ OrdinaryDiffEqDifferentiation
      From worker 2:	    OrdinaryDiffEqDifferentiationSparseArraysExt Being
 precompiled by another process (pid: 3964812, pidfile: /home/crackauc/gith
ub-runners/amdci8-1/.julia/compiled/v1.11/OrdinaryDiffEqDifferentiationSpar
seArraysExt/Rhw3a_LSg15.ji.pidfile)
      From worker 3:	   2990.5 ms  ✓ DiffEqNoiseProcess
      From worker 2:	   3158.4 ms  ✓ DiffEqNoiseProcess
      From worker 2:	    StochasticDiffEqCore Being precompiled by another 
process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.jul
ia/compiled/v1.11/StochasticDiffEqCore/vTWRy_LSg15.ji.pidfile)
      From worker 3:	   1824.8 ms  ✓ OrdinaryDiffEqDifferentiation → Ordina
ryDiffEqDifferentiationSparseArraysExt
      From worker 2:	   1844.2 ms  ✓ OrdinaryDiffEqDifferentiation → Ordina
ryDiffEqDifferentiationSparseArraysExt
      From worker 3:	   3065.6 ms  ✓ StochasticDiffEqCore
      From worker 2:	   3366.2 ms  ✓ StochasticDiffEqCore
      From worker 2:	    StochasticDiffEqRODE Being precompiled by another 
process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.jul
ia/compiled/v1.11/StochasticDiffEqRODE/4d6QD_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqIIF Being precompiled by another p
rocess (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.juli
a/compiled/v1.11/StochasticDiffEqIIF/FUGvq_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqROCK Being precompiled by another 
process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.jul
ia/compiled/v1.11/StochasticDiffEqROCK/xJtmQ_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqMilstein Being precompiled by anot
her process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/
.julia/compiled/v1.11/StochasticDiffEqMilstein/vnwsX_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqHighOrder Being precompiled by ano
ther process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1
/.julia/compiled/v1.11/StochasticDiffEqHighOrder/toXso_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqLowOrder Being precompiled by anot
her process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/
.julia/compiled/v1.11/StochasticDiffEqLowOrder/OsQtK_LSg15.ji.pidfile)
      From worker 3:	   3079.9 ms  ✓ StochasticDiffEqRODE
      From worker 3:	   3070.8 ms  ✓ StochasticDiffEqIIF
      From worker 3:	   3161.0 ms  ✓ StochasticDiffEqLowOrder
      From worker 3:	   3445.7 ms  ✓ StochasticDiffEqMilstein
      From worker 3:	   3468.8 ms  ✓ StochasticDiffEqHighOrder
      From worker 3:	   3856.6 ms  ✓ StochasticDiffEqROCK
      From worker 2:	   3109.4 ms  ✓ StochasticDiffEqRODE
      From worker 2:	   5689.8 ms  ✓ StochasticDiffEqIIF
      From worker 2:	   5688.7 ms  ✓ StochasticDiffEqLowOrder
      From worker 2:	   5689.4 ms  ✓ StochasticDiffEqMilstein
      From worker 2:	   5689.2 ms  ✓ StochasticDiffEqHighOrder
      From worker 2:	   6381.5 ms  ✓ StochasticDiffEqROCK
      From worker 3:	  34958.7 ms  ✓ NonlinearSolve
      From worker 2:	  35419.2 ms  ✓ NonlinearSolve
      From worker 2:	    OrdinaryDiffEqNonlinearSolve Being precompiled by 
another process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci
8-1/.julia/compiled/v1.11/OrdinaryDiffEqNonlinearSolve/jGadk_LSg15.ji.pidfi
le)
      From worker 3:	   4443.1 ms  ✓ OrdinaryDiffEqNonlinearSolve
      From worker 2:	   4605.6 ms  ✓ OrdinaryDiffEqNonlinearSolve
      From worker 2:	    StochasticDiffEqLeaping Being precompiled by anoth
er process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.
julia/compiled/v1.11/StochasticDiffEqLeaping/mGXg0_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqWeak Being precompiled by another 
process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.jul
ia/compiled/v1.11/StochasticDiffEqWeak/0p3bI_LSg15.ji.pidfile)
      From worker 2:	    StochasticDiffEqImplicit Being precompiled by anot
her process (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/
.julia/compiled/v1.11/StochasticDiffEqImplicit/N1COH_LSg15.ji.pidfile)
      From worker 3:	   4766.9 ms  ✓ StochasticDiffEqLeaping
      From worker 3:	   4878.8 ms  ✓ StochasticDiffEqImplicit
      From worker 3:	   6103.3 ms  ✓ StochasticDiffEqWeak
      From worker 2:	   5046.5 ms  ✓ StochasticDiffEqLeaping
      From worker 2:	   7001.2 ms  ✓ StochasticDiffEqImplicit
      From worker 2:	   7001.8 ms  ✓ StochasticDiffEqWeak
      From worker 2:	    StochasticDiffEq Being precompiled by another proc
ess (pid: 3964812, pidfile: /home/crackauc/github-runners/amdci8-1/.julia/c
ompiled/v1.11/StochasticDiffEq/36ZnY_LSg15.ji.pidfile)
      From worker 3:	   4294.0 ms  ✓ StochasticDiffEq
      From worker 3:	  20 dependencies successfully precompiled in 55 secon
ds. 188 already precompiled.
      From worker 3:	  1 dependency had output during precompilation:
      From worker 3:	┌ StochasticDiffEqCore
      From worker 3:	│  WARNING: using DiffEqNoiseProcess.save_noise! in mo
dule StochasticDiffEqCore conflicts with an existing identifier.
      From worker 3:	└  
      From worker 2:	   3961.2 ms  ✓ StochasticDiffEq
      From worker 2:	  20 dependencies successfully precompiled in 57 secon
ds. 188 already precompiled.
      From worker 3:	Precompiling Plots...
      From worker 2:	Precompiling Plots...
      From worker 2:	    Plots Being precompiled by another process (pid: 3
964812, pidfile: /home/crackauc/github-runners/amdci8-1/.julia/compiled/v1.
11/Plots/ld3vC_LSg15.ji.pidfile)
      From worker 3:	  47891.3 ms  ✓ Plots
      From worker 3:	  1 dependency successfully precompiled in 49 seconds.
 177 already precompiled.
      From worker 2:	  48242.6 ms  ✓ Plots
      From worker 2:	  1 dependency successfully precompiled in 49 seconds.
 177 already precompiled.
3-element Vector{Int64}:
 17
 23
 17
```





Timings are only valid if no workers die. Workers die if you run out of memory.

```julia
for k in 1:size(probs, 1)
    global probs, Ns, fullMeans, fullMedians, fullElapsed, fullTols
    println("Problem $k")
    ## Setup
    N = Ns[k]

    msims = Vector{Any}(undef, N)
    elapsed = Array{Float64}(undef, N, 3)
    medians = Array{Float64}(undef, N, 3)
    means = Array{Float64}(undef, N, 3)
    tols = Array{Float64}(undef, N, 3)

    #Compile
    prob = probs[k, 1]
    ParallelDataTransfer.sendto(workers(), prob = prob)
    monte_prob = EnsembleProblem(prob)
    solve(monte_prob, SRIW1(), dt = 1/2^(4), adaptive = true,
        trajectories = 1000, abstol = 2.0^(-1), reltol = 0)

    println("RSwM1")
    for i in (1 + offset):(N + offset)
        tols[i - offset, 1] = 2.0^(-i-1)
        msims[i - offset] = final_error_stats(solve(monte_prob, SRIW1(),
            trajectories = 1000, abstol = 2.0^(-i-1),
            reltol = 0, force_dtmin = true))
        elapsed[i - offset, 1] = msims[i - offset].elapsed_time
        medians[i - offset, 1] = msims[i - offset].median
        means[i - offset, 1] = msims[i - offset].mean
    end

    println("RSwM2")
    prob = probs[k, 2]

    ParallelDataTransfer.sendto(workers(), prob = prob)
    monte_prob = EnsembleProblem(prob)
    solve(monte_prob, SRIW1(), dt = 1/2^(4), adaptive = true,
        trajectories = 1000, abstol = 2.0^(-1), reltol = 0)

    for i in (1 + offset):(N + offset)
        tols[i - offset, 2] = 2.0^(-i-1)
        msims[i - offset] = final_error_stats(solve(monte_prob, SRIW1(),
            trajectories = 1000, abstol = 2.0^(-i-1),
            reltol = 0, force_dtmin = true))
        elapsed[i - offset, 2] = msims[i - offset].elapsed_time
        medians[i - offset, 2] = msims[i - offset].median
        means[i - offset, 2] = msims[i - offset].mean
    end

    println("RSwM3")
    prob = probs[k, 3]
    ParallelDataTransfer.sendto(workers(), prob = prob)
    monte_prob = EnsembleProblem(prob)
    solve(monte_prob, SRIW1(), dt = 1/2^(4), adaptive = true,
        trajectories = 1000, abstol = 2.0^(-1), reltol = 0)

    for i in (1 + offset):(N + offset)
        tols[i - offset, 3] = 2.0^(-i-1)
        msims[i - offset] = final_error_stats(solve(monte_prob, SRIW1(),
            adaptive = true, trajectories = 1000, abstol = 2.0^(-i-1),
            reltol = 0, force_dtmin = true))
        elapsed[i - offset, 3] = msims[i - offset].elapsed_time
        medians[i - offset, 3] = msims[i - offset].median
        means[i - offset, 3] = msims[i - offset].mean
    end

    fullMeans[k] = means
    fullMedians[k] = medians
    fullElapsed[k] = elapsed
    fullTols[k] = tols
end
```

```
Problem 1
RSwM1
RSwM2
RSwM3
Problem 2
RSwM1
RSwM2
RSwM3
Problem 3
RSwM1
RSwM2
RSwM3
```



```julia
gr(fmt = :svg)
lw=3
leg = permutedims(String["RSwM1", "RSwM2", "RSwM3"])

titleFontSize = 16
guideFontSize = 14
legendFontSize = 14
tickFontSize = 12

for k in 1:size(probs, 1)
    global probs, Ns, fullMeans, fullMedians, fullElapsed, fullTols
    p1[k] = Plots.plot(fullTols[k], fullMeans[k], xscale = :log10, yscale = :log10,
        xguide = "Absolute Tolerance", yguide = "Mean Final Error",
        title = "Example $k", linewidth = lw, grid = false, lab = leg,
        titlefont = font(titleFontSize), legendfont = font(legendFontSize),
        tickfont = font(tickFontSize), guidefont = font(guideFontSize))
    p2[k] = Plots.plot(fullTols[k], fullMedians[k], xscale = :log10, yscale = :log10,
        xguide = "Absolute Tolerance", yguide = "Median Final Error",
        title = "Example $k", linewidth = lw, grid = false, lab = leg,
        titlefont = font(titleFontSize), legendfont = font(legendFontSize),
        tickfont = font(tickFontSize), guidefont = font(guideFontSize))
    p3[k] = Plots.plot(fullTols[k], fullElapsed[k], xscale = :log10, yscale = :log10,
        xguide = "Absolute Tolerance", yguide = "Elapsed Time",
        title = "Example $k", linewidth = lw, grid = false, lab = leg,
        titlefont = font(titleFontSize), legendfont = font(legendFontSize),
        tickfont = font(tickFontSize), guidefont = font(guideFontSize))
end

Plots.plot!(p1[1])
Plots.plot(p1[1], p1[2], p1[3], layout = (3, 1), size = (1000, 800))
```

![](figures/AdaptiveEfficiencyTests_3_1.png)

```julia
#savefig("meanvstol.png")
#savefig("meanvstol.pdf")
```


```julia
plot(p3[1], p3[2], p3[3], layout = (3, 1), size = (1000, 800))
#savefig("timevstol.png")
#savefig("timevstol.pdf")
```

![](figures/AdaptiveEfficiencyTests_5_1.png)

```julia
plot(p1[1], p3[1], p1[2], p3[2], p1[3], p3[3], layout = (3, 2), size = (1000, 800))
```

![](figures/AdaptiveEfficiencyTests_6_1.png)

```julia

using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
```


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AdaptiveSDE","AdaptiveEfficiencyTests.jmd")
```

Computer Information:

```
Julia Version 1.11.9
Commit 53a02c0720c (2026-02-06 00:27 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)
Threads: 128 default, 0 interactive, 64 GC (on 128 virtual cores)
Environment:
  JULIA_DEPOT_PATH = /home/crackauc/github-runners/amdci8-1/.julia
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `~/github-runners/amdci8-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AdaptiveSDE/Project.toml`
⌃ [77a26b50] DiffEqNoiseProcess v5.31.0
  [2dcacdae] ParallelDataTransfer v0.5.1
⌃ [91a5bcdd] Plots v1.41.6
⌃ [c72e72a9] SDEProblemLibrary v1.2.0
⌃ [31c91b34] SciMLBenchmarks v0.1.3
⌅ [a6db7da4] SciMLLogging v1.9.1
⌃ [10745b16] Statistics v1.11.1
⌃ [789caeaf] StochasticDiffEq v7.0.0
  [8ba89e20] Distributed v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `~/github-runners/amdci8-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AdaptiveSDE/Manifest.toml`
⌃ [47edcb42] ADTypes v1.22.0
⌃ [7d9f7c33] Accessors v0.1.44
⌃ [79e6a3ab] Adapt v4.5.2
  [66dad0bd] AliasTables v1.1.3
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.24.0
⌃ [d1d4a3ce] BitFlags v0.1.9
⌃ [70df07ce] BracketingNonlinearSolve v1.12.1
⌃ [944b1d66] CodecZlib v0.7.8
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌃ [38540f10] CommonSolve v0.2.6
  [bbf7d656] CommonSubexpressions v0.3.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
⌃ [2569d6c7] ConcreteStructs v0.2.3
⌃ [f0e56b4a] ConcurrentUtilities v2.5.1
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [9a962f9c] DataAPI v1.16.0
⌃ [864edb3b] DataStructures v0.19.4
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v7.1.0
⌃ [459566f4] DiffEqCallbacks v4.17.0
⌃ [77a26b50] DiffEqNoiseProcess v5.31.0
  [163ba53b] DiffResults v1.1.0
⌃ [b552c78f] DiffRules v1.15.1
⌃ [a0c0ee7d] DifferentiationInterface v0.7.17
⌃ [31c24e10] Distributions v0.25.125
  [ffbed154] DocStringExtensions v0.9.5
  [4e289a0a] EnumX v1.0.7
⌃ [f151be2c] EnzymeCore v0.8.20
  [460bff9d] ExceptionUnwrapping v0.1.11
⌃ [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.5
⌃ [7034ab61] FastBroadcast v1.3.2
  [9aa1b823] FastClosures v0.3.2
⌃ [a4df4552] FastPower v1.3.1
⌃ [1a297f60] FillArrays v1.16.0
⌃ [6a86dc24] FiniteDiff v2.31.0
⌅ [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
⌃ [f6369f11] ForwardDiff v1.3.3
  [069b7b12] FunctionWrappers v1.1.3
⌃ [77dc65aa] FunctionWrappersWrappers v1.8.0
  [46192b85] GPUArraysCore v0.2.0
⌃ [28b8d3ca] GR v0.73.24
  [d7ba0133] Git v1.5.0
  [86223c79] Graphs v1.14.0
  [42e2da0e] Grisu v1.0.2
⌅ [cd3eb016] HTTP v1.11.0
⌅ [eafb193a] Highlights v0.5.3
⌃ [34004b35] HypergeometricFunctions v0.3.28
  [7073ff75] IJulia v1.34.4
  [d25df0c9] Inflate v0.1.5
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
⌃ [692b3bcd] JLLWrappers v1.7.1
⌅ [682c06a0] JSON v0.21.4
⌃ [ccbc3e58] JumpProcesses v9.28.0
⌃ [ba0b0d4f] Krylov v0.10.6
⌃ [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.10
⌃ [87fe0de2] LineSearch v0.1.9
⌅ [7ed4a6bd] LinearSolve v3.75.0
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
⌃ [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.10
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
⌃ [46d2c3a1] MuladdMacro v0.2.4
  [ffc61752] Mustache v1.0.21
⌃ [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.19.0
⌅ [be0214bd] NonlinearSolveBase v2.25.0
⌃ [5959db7a] NonlinearSolveFirstOrder v2.1.1
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.13.1
⌃ [26075421] NonlinearSolveSpectralMethods v1.7.1
  [4d8831e6] OpenSSL v1.6.1
⌅ [bac558e1] OrderedCollections v1.8.1
⌃ [bbf590c4] OrdinaryDiffEqCore v4.0.0
⌃ [4302a76b] OrdinaryDiffEqDifferentiation v3.0.0
⌃ [127b3ac7] OrdinaryDiffEqNonlinearSolve v2.0.0
⌃ [90014a1f] PDMats v0.11.37
  [2dcacdae] ParallelDataTransfer v0.5.1
⌅ [69de0a69] Parsers v2.8.4
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
⌃ [91a5bcdd] Plots v1.41.6
⌃ [e409e4f3] PoissonRandom v0.4.7
⌃ [d236fae5] PreallocationTools v1.2.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [43287f4e] PtrArrays v1.4.0
  [1fd47b50] QuadGK v2.11.3
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌃ [731186ca] RecursiveArrayTools v4.3.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [ae5879a3] ResettableStacks v1.2.0
  [79098fc4] Rmath v0.9.0
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.18
⌃ [c72e72a9] SDEProblemLibrary v1.2.0
⌅ [0bca4576] SciMLBase v3.7.1
⌃ [31c91b34] SciMLBenchmarks v0.1.3
⌃ [19f34311] SciMLJacobianOperators v0.1.13
⌅ [a6db7da4] SciMLLogging v1.9.1
⌃ [c0aeaf25] SciMLOperators v1.17.0
⌃ [431bcebd] SciMLPublic v1.0.1
⌃ [53ae85a6] SciMLStructures v1.10.0
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
⌃ [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.11.1
⌃ [699a6c99] SimpleTraits v0.9.5
⌃ [a2af1166] SortingAlgorithms v1.2.2
  [0a514795] SparseMatrixColorings v0.4.27
⌃ [276daf66] SpecialFunctions v2.7.2
  [860ef19b] StableRNGs v1.0.4
⌃ [90137ffa] StaticArrays v1.9.18
  [1e83bf80] StaticArraysCore v1.4.4
⌃ [10745b16] Statistics v1.11.1
  [82ae8749] StatsAPI v1.8.0
⌃ [2913bbd2] StatsBase v0.34.10
⌅ [4c63d2b9] StatsFuns v1.5.2
⌃ [789caeaf] StochasticDiffEq v7.0.0
⌃ [19c5a474] StochasticDiffEqCore v2.0.0
⌃ [0520c28c] StochasticDiffEqHighOrder v2.0.0
⌃ [ebf54054] StochasticDiffEqIIF v2.0.0
⌃ [5080b986] StochasticDiffEqImplicit v2.0.0
⌃ [aefaaa88] StochasticDiffEqLeaping v2.0.0
⌃ [90dbc90e] StochasticDiffEqLevyArea v2.0.0
⌃ [d15fe365] StochasticDiffEqLowOrder v2.0.0
⌃ [8c95a807] StochasticDiffEqMilstein v2.0.0
⌃ [db241ea8] StochasticDiffEqROCK v2.0.0
⌃ [49714585] StochasticDiffEqRODE v2.0.0
⌃ [af2a2fcd] StochasticDiffEqWeak v2.0.0
  [69024149] StringEncodings v0.3.7
⌃ [2efcf032] SymbolicIndexingInterface v0.3.46
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.12.1
  [62fd8b95] TensorCore v0.1.1
⌅ [a759f4b9] TimerOutputs v0.5.29
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [5c2747f8] URIs v1.6.1
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [6e34b625] Bzip2_jll v1.0.9+0
⌃ [83423d85] Cairo_jll v1.18.6+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.8.0+0
⌅ [b22a6f82] FFMPEG_jll v8.1.0+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
⌃ [0656b61e] GLFW_jll v3.4.1+1
⌅ [d2c73de3] GR_jll v0.73.24+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
⌃ [020c3dae] Git_LFS_jll v3.7.0+0
⌃ [f8c6e375] Git_jll v2.54.0+0
⌃ [7746bdde] Glib_jll v2.86.3+0
⌃ [3b182d85] Graphite2_jll v1.3.15+0
⌅ [2e76f6c2] HarfBuzz_jll v8.5.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
⌃ [aacddb02] JpegTurbo_jll v3.1.5+0
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.1.0+0
⌃ [1d63c593] LLVMOpenMP_jll v18.1.8+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
⌃ [89763e89] Libtiff_jll v4.7.2+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [856f044c] MKL_jll v2025.2.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
⌃ [9bd350c2] OpenSSH_jll v10.3.1+0
⌃ [458c3c95] OpenSSL_jll v3.5.6+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
⌃ [36c8627f] Pango_jll v1.57.1+0
⌅ [30392449] Pixman_jll v0.44.2+0
⌃ [c0090381] Qt6Base_jll v6.10.2+1
⌃ [629bc702] Qt6Declarative_jll v6.10.2+1
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
⌃ [f50d1b31] Rmath_jll v0.5.1+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
  [ffd25f8a] XZ_jll v5.8.3+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.13+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.8+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.2+0
⌃ [a51aa0fd] Xorg_libXi_jll v1.8.3+0
  [d1454406] Xorg_libXinerama_jll v1.1.7+0
  [ec84b674] Xorg_libXrandr_jll v1.5.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [a65dc6b1] Xorg_libpciaccess_jll v0.19.0+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.2.0+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.6+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
⌃ [33bec58e] Xorg_xkeyboard_config_jll v2.47.0+0
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
⌅ [214eeab7] fzf_jll v0.61.1+0
⌃ [a4ae2306] libaom_jll v3.13.3+0
⌃ [0ac62f75] libass_jll v0.17.4+0
  [1183f4f0] libdecor_jll v0.2.2+0
⌃ [8e53e030] libdrm_jll v2.4.125+1
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.58+0
  [a9144af2] libsodium_jll v1.0.21+0
  [9a156e7d] libva_jll v2.23.0+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
⌃ [1317d2d5] oneTBB_jll v2022.0.0+1
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [37e2e46d] LinearAlgebra v1.11.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.11.0
  [de0858da] Printf v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.11.0
  [f489334b] StyledStrings v1.11.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.6.0+0
  [e37daf67] LibGit2_jll v1.7.2+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.6+0
  [14a3606d] MozillaCACerts_jll v2023.12.12
  [4536629a] OpenBLAS_jll v0.3.27+1
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.7.0+0
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

