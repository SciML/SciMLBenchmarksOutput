---
author: "Samuel Isaacson and Chris Rackauckas"
title: "BCR Work-Precision Diagrams"
---


The following benchmark is of 1122 ODEs with 24388 terms that describe a stiff
chemical reaction network modeling the BCR signaling network from [Barua et
al.](https://doi.org/10.4049/jimmunol.1102003). We use
[`ReactionNetworkImporters`](https://github.com/isaacsas/ReactionNetworkImporters.jl)
to load the BioNetGen model files as a
[Catalyst](https://github.com/SciML/Catalyst.jl) model, and then use
[ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl) to convert the
Catalyst network model to ODEs.


```julia
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
      Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
      LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools

gr()
datadir  = joinpath(dirname(pathof(ReactionNetworkImporters)),"../data/bcr")
const to = TimerOutput()
tf       = 100000.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(datadir, "bcr.net"))
rn    = prnbng.rn
@timeit to "Create ODESys" osys = convert(ODESystem, rn)

u₀    = prnbng.u₀
p     = prnbng.p
tspan = (0.,tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem(osys, u₀, tspan, p)
@timeit to "ODEProb DenseJac" densejacprob = ODEProblem(osys, u₀, tspan, p, jac=true)
```

```
Parsing parameters...done
Adding parameters...done
Parsing species...done
Adding species...done
Creating ModelingToolkit versions of species and parameters...done
Parsing and adding reactions...done
Parsing groups...done
ODEProblem with uType Vector{Float64} and tType Float64. In-place: true
timespan: (0.0, 100000.0)
u0: 1122-element Vector{Float64}:
 299717.8348854
  47149.15480798
  46979.01102231
 290771.2428252
 299980.7396749
 300000.0
    141.3151575495
      0.1256496403614
      0.4048783555301
    140.8052338618
      ⋮
      1.005585387399e-24
      6.724953378237e-17
      3.395560698281e-16
      1.787990228838e-5
      8.761844379939e-13
      0.0002517949074779
      0.0005539124513976
      2.281251822741e-14
      1.78232055967e-8
```



```julia
@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem(osys, u₀, tspan, p, jac=true, sparse=true)
show(to)
```

```
──────────────────────────────────────────────────────────────────────────
──
                                     Time                   Allocations    
  
                             ──────────────────────   ─────────────────────
──
      Tot / % measured:            342s / 100%            63.6GiB / 100%   
  

 Section             ncalls     time   %tot     avg     alloc   %tot      a
vg
 ──────────────────────────────────────────────────────────────────────────
──
 ODEProb DenseJac         1     274s  80.0%    274s   45.8GiB  72.1%  45.8G
iB
 ODEProb SparseJac        1    59.3s  17.3%   59.3s   15.3GiB  24.1%  15.3G
iB
 ODEProb No Jac           1    6.52s  1.91%   6.52s   1.72GiB  2.70%  1.72G
iB
 Create ODESys            1    1.39s  0.41%   1.39s    507MiB  0.78%   507M
iB
 Parse Network            1    1.15s  0.34%   1.15s    220MiB  0.34%   220M
iB
 ──────────────────────────────────────────────────────────────────────────
──
```



```julia
@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Apprx. number of terms in the ODE
@show numparams(rn) # Number of Parameters
```

```
numspecies(rn) = 1122
numreactions(rn) = 24388
numparams(rn) = 128
128
```





## Time ODE derivative function compilation
As compiling the ODE derivative functions has in the past taken longer than
running a simulation, we first force compilation by evaluating these functions
one time.
```julia
u  = copy(u₀)
du = similar(u)
@timeit to "ODE rhs Eval1" oprob.f(du,u,p,0.)
@timeit to "ODE rhs Eval2" oprob.f(du,u,p,0.)
densejacprob.f(du,u,p,0.)
sparsejacprob.f(du,u,p,0.)
```




We also time the ODE rhs function with BenchmarkTools as it is more accurate
given how fast evaluating `f` is:
```julia
@btime oprob.f($du,$u,$p,0.)
```

```
27.090 μs (3 allocations: 512 bytes)
```





Now we time the Jacobian functions, including compilation time in the first
evaluations
```julia
J = zeros(length(u),length(u))
@timeit to "DenseJac Eval1" densejacprob.f.jac(J,u,p,0.)
@timeit to "DenseJac Eval2" densejacprob.f.jac(J,u,p,0.)
```

```
Error: syntax: expression too large
```



```julia
Js = similar(sparsejacprob.f.jac_prototype)
@timeit to "SparseJac Eval1" sparsejacprob.f.jac(Js,u,p,0.)
@timeit to "SparseJac Eval2" sparsejacprob.f.jac(Js,u,p,0.)
show(to)
```

```
──────────────────────────────────────────────────────────────────────────
──
                                     Time                   Allocations    
  
                             ──────────────────────   ─────────────────────
──
      Tot / % measured:            360s / 95.5%           65.0GiB / 100%   
  

 Section             ncalls     time   %tot     avg     alloc   %tot      a
vg
 ──────────────────────────────────────────────────────────────────────────
──
 ODEProb DenseJac         1     274s  79.6%    274s   45.8GiB  70.7%  45.8G
iB
 ODEProb SparseJac        1    59.3s  17.3%   59.3s   15.3GiB  23.6%  15.3G
iB
 ODEProb No Jac           1    6.52s  1.90%   6.52s   1.72GiB  2.64%  1.72G
iB
 DenseJac Eval1           1    1.58s  0.46%   1.58s   1.30GiB  2.01%  1.30G
iB
 Create ODESys            1    1.39s  0.41%   1.39s    507MiB  0.76%   507M
iB
 Parse Network            1    1.15s  0.34%   1.15s    220MiB  0.33%   220M
iB
 ODE rhs Eval1            1    355μs  0.00%   355μs      688B  0.00%     68
8B
 SparseJac Eval1          1    301μs  0.00%   301μs      912B  0.00%     91
2B
 ODE rhs Eval2            1   31.9μs  0.00%  31.9μs      688B  0.00%     68
8B
 SparseJac Eval2          1   27.7μs  0.00%  27.7μs      912B  0.00%     91
2B
 ──────────────────────────────────────────────────────────────────────────
──
```





## Picture of the solution

```julia
sol = solve(oprob, CVODE_BDF(), saveat=tf/1000., reltol=1e-5, abstol=1e-5)
plot(sol, legend=false, fmt=:png)
```

![](figures/BCR_8_1.png)



For these benchmarks we will be using the time-series error with these saving
points since the final time point is not well-indicative of the solution
behavior (capturing the oscillation is the key!).

## Generate Test Solution

```julia
@time sol = solve(oprob, CVODE_BDF(), abstol=1/10^12, reltol=1/10^12)
test_sol  = TestSolution(sol)
```

```
632.475680 seconds (3.80 M allocations: 2.145 GiB, 0.23% gc time)
retcode: Success
Interpolation: 3rd order Hermite
t: nothing
u: nothing
```





## Setups

```julia
abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (5:8);
setups = [
          #Dict(:alg=>Rosenbrock23(autodiff=false)),
          Dict(:alg=>TRBDF2(autodiff=false)),
          Dict(:alg=>QNDF(autodiff=false)),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>CVODE_BDF(linear_solver=:LapackDense)),
          #Dict(:alg=>rodas()),
          #Dict(:alg=>radau()),
          #Dict(:alg=>Rodas4(autodiff=false)),
          #Dict(:alg=>Rodas5(autodiff=false)),
          Dict(:alg=>KenCarp4(autodiff=false)),
          Dict(:alg=>KenCarp47(autodiff=false)),
          #Dict(:alg=>RadauIIA5(autodiff=false)),
          #Dict(:alg=>lsoda()),
          ]
```

```
6-element Vector{Dict{Symbol, V} where V}:
 Dict{Symbol, OrdinaryDiffEq.TRBDF2{0, false, DiffEqBase.DefaultLinSolve, D
iffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}, Val{
:forward}}}(:alg => OrdinaryDiffEq.TRBDF2{0, false, DiffEqBase.DefaultLinSo
lve, DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}
, Val{:forward}}(DiffEqBase.DefaultLinSolve(nothing, nothing, nothing), Dif
fEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}(1//100,
 10, 1//5, 1//5), true, :linear, :PI))
 Dict{Symbol, OrdinaryDiffEq.QNDF{5, 0, false, DiffEqBase.DefaultLinSolve, 
DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}, Val
{:forward}, Nothing, Nothing, NTuple{5, Float64}}}(:alg => OrdinaryDiffEq.Q
NDF{5, 0, false, DiffEqBase.DefaultLinSolve, DiffEqBase.NLNewton{Rational{I
nt64}, Rational{Int64}, Rational{Int64}}, Val{:forward}, Nothing, Nothing, 
NTuple{5, Float64}}(Val{5}(), DiffEqBase.DefaultLinSolve(nothing, nothing, 
nothing), DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{In
t64}}(1//100, 10, 1//5, 1//5), nothing, nothing, :linear, (-0.185, -0.11111
11111111111, -0.0823, -0.0415, 0.0), :Standard))
 Dict{Symbol, Sundials.CVODE_BDF{:Newton, :Dense, Nothing, Nothing}}(:alg =
> Sundials.CVODE_BDF{:Newton, :Dense, Nothing, Nothing}(0, 0, 0, false, 10,
 5, 7, 3, 10, nothing, nothing, 0))
 Dict{Symbol, Sundials.CVODE_BDF{:Newton, :LapackDense, Nothing, Nothing}}(
:alg => Sundials.CVODE_BDF{:Newton, :LapackDense, Nothing, Nothing}(0, 0, 0
, false, 10, 5, 7, 3, 10, nothing, nothing, 0))
 Dict{Symbol, OrdinaryDiffEq.KenCarp4{0, false, DiffEqBase.DefaultLinSolve,
 DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}, Va
l{:forward}}}(:alg => OrdinaryDiffEq.KenCarp4{0, false, DiffEqBase.DefaultL
inSolve, DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int
64}}, Val{:forward}}(DiffEqBase.DefaultLinSolve(nothing, nothing, nothing),
 DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}(1//
100, 10, 1//5, 1//5), true, :linear, :PI))
 Dict{Symbol, OrdinaryDiffEq.KenCarp47{0, false, DiffEqBase.DefaultLinSolve
, DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}, V
al{:forward}}}(:alg => OrdinaryDiffEq.KenCarp47{0, false, DiffEqBase.Defaul
tLinSolve, DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{I
nt64}}, Val{:forward}}(DiffEqBase.DefaultLinSolve(nothing, nothing, nothing
), DiffEqBase.NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}}(1
//100, 10, 1//5, 1//5), true, :linear, :PI))
```





## Automatic Jacobian Solves

Due to the computational cost of the problem, we are only going to focus on the
methods which demonstrated computational efficiency on the smaller biochemical
benchmark problems. This excludes the exponential integrator, stabilized explicit,
and extrapolation classes of methods.

First we test using auto-generated Jacobians (finite difference)
```julia
wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e5),numruns=1)
plot(wp)
```

![](figures/BCR_11_1.png)



## Analytical Jacobian
Now we test using the generated analytic Jacobian function.
```julia
wp = WorkPrecisionSet(densejacprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e5),numruns=1)
plot(wp)
```

```
Error: syntax: expression too large
```






## Sparse Jacobian
Finally we test using the generated sparse analytic Jacobian function.
```julia
setups = [
          #Dict(:alg=>Rosenbrock23(autodiff=false)),
          Dict(:alg=>TRBDF2(autodiff=false)),
          Dict(:alg=>QNDF(autodiff=false)),
          #Dict(:alg=>CVODE_BDF(linear_solver=:KLU)), # Fails!
          #Dict(:alg=>rodas()),
          #Dict(:alg=>radau()),
          #Dict(:alg=>Rodas4(autodiff=false)),
          #Dict(:alg=>Rodas5(autodiff=false)),
          Dict(:alg=>KenCarp4(autodiff=false)),
          Dict(:alg=>KenCarp47(autodiff=false)),
          #Dict(:alg=>RadauIIA5(autodiff=false)),
          #Dict(:alg=>lsoda()),
          ]
wp = WorkPrecisionSet(sparsejacprob,abstols,reltols,setups;error_estimate=:l2,
                      saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e5),numruns=1)
plot(wp)
```

![](figures/BCR_13_1.png)


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/Bio","BCR.jmd")
```

Computer Information:

```
Julia Version 1.6.3
Commit ae8452a9e0 (2021-09-23 17:34 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
      Status `/var/lib/buildkite-agent/builds/amdci3-julia-csail-mit-edu/julialang/scimlbenchmarks-dot-jl/benchmarks/Bio/Project.toml`
  [6e4b80f9] BenchmarkTools v1.2.0
  [479239e8] Catalyst v9.0.0
  [2b5f629d] DiffEqBase v6.73.2
  [f3b72e0c] DiffEqDevTools v2.27.2
  [7f56f5a3] LSODA v0.7.0
  [961ee093] ModelingToolkit v6.5.0
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.10.1
  [1dea7af3] OrdinaryDiffEq v5.64.0
  [91a5bcdd] Plots v1.22.1
  [b4db0fb7] ReactionNetworkImporters v0.11.1
  [31c91b34] SciMLBenchmarks v0.1.0
  [c3572dad] Sundials v4.5.3
  [a759f4b9] TimerOutputs v0.5.13
```

And the full manifest:

```
      Status `/var/lib/buildkite-agent/builds/amdci3-julia-csail-mit-edu/julialang/scimlbenchmarks-dot-jl/benchmarks/Bio/Manifest.toml`
  [c3fe647b] AbstractAlgebra v0.21.0
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.1
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.33
  [6e4b80f9] BenchmarkTools v1.2.0
  [e2ed5e7c] Bijections v0.1.3
  [9e28174c] BinDeps v1.0.2
  [62783981] BitTwiddlingConvenienceFunctions v0.1.0
  [fa961155] CEnum v0.4.1
  [2a0fbf3d] CPUSummary v0.1.3
  [00ebfdb7] CSTParser v3.2.4
  [479239e8] Catalyst v9.0.0
  [d360d2e6] ChainRulesCore v1.6.0
  [fb6a15b2] CloseOpenIntervals v0.1.2
  [35d6a980] ColorSchemes v3.14.0
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.2
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.38.0
  [b152e2b5] CompositeTypes v0.1.2
  [8f4d0f93] Conda v1.5.2
  [187b0558] ConstructionBase v1.3.0
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.0.4
  [754358af] DEDataArrays v0.2.0
  [9a962f9c] DataAPI v1.9.0
  [864edb3b] DataStructures v0.18.10
  [e2d170a0] DataValueInterfaces v1.0.0
  [2b5f629d] DiffEqBase v6.73.2
  [459566f4] DiffEqCallbacks v2.17.0
  [f3b72e0c] DiffEqDevTools v2.27.2
  [c894b116] DiffEqJump v7.3.0
  [77a26b50] DiffEqNoiseProcess v5.9.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.3.1
  [b4f34e82] Distances v0.10.4
  [31c24e10] Distributions v0.25.16
  [ffbed154] DocStringExtensions v0.8.5
  [5b8099bc] DomainSets v0.5.7
  [7c1d4256] DynamicPolynomials v0.3.20
  [da5c29d0] EllipsisNotation v1.1.0
  [d4d017d3] ExponentialUtilities v1.9.0
  [e2ba6199] ExprTools v0.1.6
  [c87230d0] FFMPEG v0.4.1
  [7034ab61] FastBroadcast v0.1.8
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.12.5
  [6a86dc24] FiniteDiff v2.8.1
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.19
  [069b7b12] FunctionWrappers v1.1.2
  [28b8d3ca] GR v0.59.0
  [5c1252a2] GeometryBasics v0.4.1
  [d7ba0133] Git v1.2.1
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v0.9.14
  [eafb193a] Highlights v0.4.5
  [3e5b6fbb] HostCPUFeatures v0.1.4
  [0e44f5e4] Hwloc v2.0.0
  [7073ff75] IJulia v1.23.2
  [615f187c] IfElse v0.1.0
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.0
  [8197267c] IntervalSets v0.5.3
  [92d709cd] IrrationalConstants v0.1.0
  [c8e1da08] IterTools v1.3.0
  [42fd0dbc] IterativeSolvers v0.9.1
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.2
  [98e50ef6] JuliaFormatter v0.15.11
  [7f56f5a3] LSODA v0.7.0
  [b964fa9f] LaTeXStrings v1.2.1
  [2ee39098] LabelledArrays v1.6.4
  [23fbe1c1] Latexify v0.15.6
  [10f19ff3] LayoutPointers v0.1.3
  [093fc24a] LightGraphs v1.3.5
  [d3d80556] LineSearches v7.1.1
  [2ab3a3ac] LogExpFunctions v0.3.3
  [bdcacae8] LoopVectorization v0.12.76
  [1914dd2f] MacroTools v0.5.8
  [d125e4d3] ManualMemory v0.1.6
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [e1d29d7a] Missings v1.0.2
  [961ee093] ModelingToolkit v6.5.0
  [46d2c3a1] MuladdMacro v0.2.2
  [102ac46a] MultivariatePolynomials v0.3.18
  [ffc61752] Mustache v1.0.10
  [d8a4904e] MutableArithmetics v0.2.20
  [d41bc354] NLSolversBase v7.8.1
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v0.3.5
  [8913a72c] NonlinearSolve v0.3.11
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.10.1
  [6fe1bfb0] OffsetArrays v1.10.7
  [429524aa] Optim v1.4.1
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v5.64.0
  [90014a1f] PDMats v0.11.1
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.0.3
  [ccf2f8ad] PlotThemes v2.0.1
  [995b91a9] PlotUtils v1.0.14
  [91a5bcdd] Plots v1.22.1
  [e409e4f3] PoissonRandom v0.4.0
  [f517fe37] Polyester v0.5.1
  [1d0040c9] PolyesterWeave v0.1.0
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.1.1
  [21216c6a] Preferences v1.2.2
  [1fd47b50] QuadGK v2.4.2
  [74087812] Random123 v1.4.2
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.5.3
  [b4db0fb7] ReactionNetworkImporters v0.11.1
  [3cdcf5f2] RecipesBase v1.1.2
  [01d81517] RecipesPipeline v0.4.1
  [731186ca] RecursiveArrayTools v2.17.2
  [f2c3362d] RecursiveFactorization v0.2.4
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.0
  [ae029012] Requires v1.1.3
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.7.0
  [47965b36] RootedTrees v1.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.3
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.27
  [1bc83da4] SafeTestsets v0.0.1
  [0bca4576] SciMLBase v1.19.0
  [31c91b34] SciMLBenchmarks v0.1.0
  [6c6a2e73] Scratch v1.1.0
  [efcf1570] Setfield v0.7.1
  [992d4aef] Showoff v1.0.3
  [699a6c99] SimpleTraits v0.9.4
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.1
  [47a9eef4] SparseDiffTools v1.16.5
  [276daf66] SpecialFunctions v1.6.2
  [aedffcd0] Static v0.3.3
  [90137ffa] StaticArrays v1.2.12
  [82ae8749] StatsAPI v1.0.0
  [2913bbd2] StatsBase v0.33.10
  [4c63d2b9] StatsFuns v0.9.10
  [7792a7ef] StrideArraysCore v0.2.4
  [69024149] StringEncodings v0.3.5
  [09ab397b] StructArrays v0.6.3
  [c3572dad] Sundials v4.5.3
  [d1185830] SymbolicUtils v0.16.0
  [0c5d862f] Symbolics v3.4.1
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.5.2
  [8ea1fca8] TermInterface v0.1.8
  [8290d209] ThreadingUtilities v0.4.6
  [a759f4b9] TimerOutputs v0.5.13
  [0796e94c] Tokenize v0.5.21
  [a2a6695c] TreeViews v0.3.0
  [d5829a12] TriangularSolve v0.1.6
  [30578b45] URIParser v0.4.1
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1986cc42] Unitful v1.9.0
  [3d5dd08c] VectorizationBase v0.21.9
  [81def892] VersionParsing v1.2.0
  [19fa3120] VertexSafeGraphs v0.1.2
  [44d3d7a6] Weave v0.10.10
  [ddb6d928] YAML v0.4.7
  [c2297ded] ZMQ v1.2.1
  [700de1a5] ZygoteRules v0.2.1
  [6e34b625] Bzip2_jll v1.0.6+5
  [83423d85] Cairo_jll v1.16.0+6
  [5ae413db] EarCut_jll v2.2.3+0
  [2e619515] Expat_jll v2.2.10+0
  [b22a6f82] FFMPEG_jll v4.3.1+4
  [a3f928ae] Fontconfig_jll v2.13.1+14
  [d7e528f0] FreeType2_jll v2.10.1+5
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.5+0
  [d2c73de3] GR_jll v0.58.1+0
  [78b55507] Gettext_jll v0.20.1+7
  [f8c6e375] Git_jll v2.31.0+0
  [7746bdde] Glib_jll v2.59.0+4
  [e33a78d0] Hwloc_jll v2.5.0+0
  [aacddb02] JpegTurbo_jll v2.1.0+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [aae0fff6] LSODA_jll v0.1.1+0
  [dd4b983a] LZO_jll v2.10.1+0
  [dd192d2f] LibVPX_jll v1.10.0+0
  [e9f186c6] Libffi_jll v3.2.2+0
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.3.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
  [c771fb93] ODEInterface_jll v0.0.1+0
  [e7412a2a] Ogg_jll v1.3.5+0
  [458c3c95] OpenSSL_jll v1.1.10+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.2+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [fb77eaff] Sundials_jll v5.2.0+1
  [a2964d1f] Wayland_jll v1.19.0+0
  [2381bf8a] Wayland_protocols_jll v1.18.0+4
  [02c8fc9c] XML2_jll v2.9.12+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3
  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] Xorg_xtrans_jll v1.4.0+3
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [3161d3a3] Zstd_jll v1.5.0+0
  [0ac62f75] libass_jll v0.14.0+4
  [f638f0a6] libfdk_aac_jll v0.1.6+4
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+0
  [1270edf5] x264_jll v2020.7.14+2
  [dfaa095f] x265_jll v3.0.0+3
  [d8fb68d0] xkbcommon_jll v0.9.1+5
  [0dad84c5] ArgTools
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [f43a241f] Downloads
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [b27032c2] LibCURL
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions
  [44cfe95a] Pkg
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML
  [a4e569a6] Tar
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll
  [deac9b47] LibCURL_jll
  [29816b5a] LibSSH2_jll
  [c8ffd9c3] MbedTLS_jll
  [14a3606d] MozillaCACerts_jll
  [4536629a] OpenBLAS_jll
  [05823500] OpenLibm_jll
  [efcefdf7] PCRE2_jll
  [bea87d4a] SuiteSparse_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

