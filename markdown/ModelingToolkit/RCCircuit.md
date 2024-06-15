---
author: "Avinash Subramanian, Yingbo Ma, Chris Elrod"
title: "RC Circuit"
---


When a model is defined using repeated components, JuliaSimCompiler is able to take advantage of this 
to scale efficiently by rerolling equations into loops. This option can be disabled by setting `loop=false`. 
Here, we build an RC circuit model with variable numbers of components to show scaling of compile and 
runtimes of MTK vs JuliaSimCompiler's three backends with and without loop rerolling.

## Setup Model Code

```julia
using JuliaSimCompiler, ModelingToolkit, OrdinaryDiffEq, BenchmarkTools, ModelingToolkitStandardLibrary, OMJulia, CairoMakie
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkitStandardLibrary.Electrical
```

```
Error: ArgumentError: Package ModelingToolkitStandardLibrary not found in c
urrent path.
- Run `import Pkg; Pkg.add("ModelingToolkitStandardLibrary")` to install th
e ModelingToolkitStandardLibrary package.
```



```julia
# ModelingToolkit and JuliaSimCompiler
const t = Blocks.t

function build_system(n)
  systems = @named begin
    sine = Sine(frequency = 10)
    source = Voltage()
    resistors[1:n] = Resistor()
    capacitors[1:n] = Capacitor()
    ground = Ground()
  end
  systems = reduce(vcat, systems)
  eqs = [connect(sine.output, source.V)
       connect(source.p, resistors[1].p)
       [connect(resistors[i].n, resistors[i + 1].p, capacitors[i].p)
        for i in 1:(n - 1)]
       connect(resistors[end].n, capacitors[end].p)
       [connect(capacitors[i].n, source.n) for i in 1:n]
       connect(source.n, ground.g)]
  @named sys = ODESystem(eqs, t; systems)
  u0 = [capacitors[i].v => float(i) for i in 1:n];
  ps = [[resistors[i].R => 1 / i for i in 1:n];
        [capacitors[i].C => 1 / i^2 for i in 1:n]]
  return sys, u0, ps
end

function compile_run_problem(sys, u0, ps; target=JuliaSimCompiler.JuliaTarget(), duref=nothing)
  tspan = (0.0, 10.0)
  t0 = time()
  prob = if target === JuliaSimCompiler.JuliaTarget()
    ODEProblem(sys, u0, tspan, ps; sparse = true)
  else
    ODEProblem(sys, target, u0, tspan, ps; sparse = true)
  end
  (; f, u0, p) = prob
  ff = f.f
  du = similar(u0)
  ff(du, u0, p, 0.0)
  t_fode = time() - t0
  duref === nothing || @assert duref ≈ du
  t_run = @belapsed $ff($du, $u0, $p, 0.0)
  t_solve = @elapsed solve(prob, Rodas5(autodiff = false))
  (t_fode, t_run, t_solve), du
end

const C = JuliaSimCompiler.CTarget();
const LLVM = JuliaSimCompiler.llvm.LLVMTarget();

function run_and_time_julia!(ss_times, times, max_sizes, i, n)
  sys, u0, ps = build_system(n);
  if n <= max_sizes[1]
    ss_times[i, 1] = @elapsed sys_mtk = structural_simplify(sys)
    times[i, 1], _ = compile_run_problem(sys_mtk, u0, ps)
  end
  ss_times[i, 2] = @elapsed sys_jsir_scalar = structural_simplify(IRSystem(sys), loop=false)
  ss_times[i, 3] = @elapsed sys_jsir_loop = structural_simplify(JuliaSimCompiler.compressed_connection_expansion(sys))
  oderef = daeref = nothing
  n <= max_sizes[2] && ((times[i, 2], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; duref = oderef))
  n <= max_sizes[3] && ((times[i, 3], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; target=C, duref = oderef))
  n <= max_sizes[4] && ((times[i, 4], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; target=LLVM, duref = oderef))
  n <= max_sizes[5] && ((times[i, 5], deeref) = compile_run_problem(sys_jsir_loop, u0, ps; duref = daeref))
  n <= max_sizes[6] && ((times[i, 6], daeref) = compile_run_problem(sys_jsir_loop, u0, ps; target=C, duref = daeref))
  n <= max_sizes[7] && ((times[i, 7], daeref) = compile_run_problem(sys_jsir_loop, u0, ps; target=LLVM, duref = daeref))
  for j = 1:7
    ss_time = j == 1 ? ss_times[i,1] : ss_times[i, 2 + (j >= 5)]
    t_fode, t_run, t_solve = times[i,j]
    total_times[i, j] = ss_time + t_fode + t_solve
  end
end
```

```
Error: UndefVarError: `Blocks` not defined
```



```julia
N = [5, 10, 20, 40, 60, 80, 160, 320, 480, 640, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000];

# max size we test per method
max_sizes = [4_000, 8_000, 20_000, 20_000, 20_000, 20_000, 20_000, 9000];

# NaN-initialize so Makie will ignore incomplete
ss_times = fill(NaN, length(N), 3);
times = fill((NaN,NaN,NaN), length(N), length(max_sizes) - 1);
total_times = fill(NaN, length(N), length(max_sizes));
```




## Julia Timings

```julia
@time run_and_time_julia!(ss_times, times, max_sizes, 1, 4); # precompile
```

```
Error: UndefVarError: `run_and_time_julia!` not defined
```





## OpenModelica Timings

```julia
# OMJ
const omod = OMJulia.OMCSession();
OMJulia.sendExpression(omod, "getVersion()")
OMJulia.sendExpression(omod, "installPackage(Modelica)")
const modelicafile = joinpath(@__DIR__, "RC_Circuit.mo")

function time_open_modelica(n::Int)
  totaltime = @elapsed res = begin
    @sync ModelicaSystem(omod, modelicafile, "RC_Circuit.Test.RC_Circuit_MTK_test_$n")
    sendExpression(omod, "simulate(RC_Circuit.Test.RC_Circuit_MTK_test_$n)")
  end
  @assert res["messages"][1:11] == "LOG_SUCCESS"
  return totaltime
end

function run_and_time_om!(ss_times, times, max_sizes, i, n)
  run_and_time_julia!(ss_times, times, max_sizes, i, n)
  if n <= max_sizes[8]
    total_times[i, 8] = time_open_modelica(n)
  end 
  @views println("n = $(n)\nstructural_simplify_times = $(ss_times[i,:])\ncomponent times = $(times[i, :])\ntotal times = $(total_times[i, :])")
end

for (i, n) in enumerate(N)
  @time run_and_time_om!(ss_times, times, max_sizes, i, n)
end

OMJulia.quit(omod)
```

```
Error: UndefVarError: `OMJulia` not defined
```





## Dymola Timings

Dymola requires a license server and thus cannot be hosted. This was run locally for the
following times:

```julia
translation_and_total_times = [
7.027, 7.237
11.295, 11.798
16.681, 17.646
22.125, 23.839
27.529, 29.82
33.282, 36.622
39.007, 43.088
44.825, 51.601
50.281, 56.676
] # TODO: I will add other times once the Dymola license server is back up.
#total_times[:, 6] = translation_and_total_times[1:length(N_x),2]
```

```
Error: ParseError:
# Error @ /cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/
benchmarks/ModelingToolkit/RCCircuit.jmd:4:1
7.027, 7.237
┌─────────────
11.295, 11.798
16.681, 17.646
⋮
44.825, 51.601
50.281, 56.676
#────────────┘ ── Expected `]`
```





## Results

```julia
f = Figure(size=(800,1200));
ss_names = ["MTK", "JSIR-Scalar", "JSIR-Loop"];
let ax = Axis(f[1, 1]; yscale = log10, xscale = log10, title="Structural Simplify Time")
  _lines = map(eachcol(ss_times)) do ts
    lines!(N, ts)
  end
  Legend(f[1,2], _lines, ss_names)
end
method_names = ["MTK", "JSIR - Scalar - Julia", "JSIR - Scalar - C", "JSIR - Scalar - LLVM", "JSIR - Loop - Julia", "JSIR - Loop - C", "JSIR - Loop - LLVM"];
for (i, timecat) in enumerate(("ODEProblem + f!", "Run", "Solve"))
  title = timecat * " Time"
  ax = Axis(f[i+1, 1]; yscale = log10, xscale = log10, title)
  _lines = map(eachcol(times)) do ts
    lines!(N, getindex.(ts, i))
  end
  Legend(f[i+1, 2], _lines, method_names)
end
let method_names_m = vcat(method_names, "OpenModelica");
  ax = Axis(f[5, 1]; yscale = log10, xscale = log10, title = "Total Time")
  _lines = map(Base.Fix1(lines!, N), eachcol(total_times))
  Legend(f[5, 2], _lines, method_names_m)
end
f
```

```
Error: UndefVarError: `Figure` not defined
```





All three backends compiled more quickly with loops, but the C and LLVM backends are so much quicker to compile than the Julia backend that this made much less difference for them.
The impact on runtime was more varied.

## Appendix


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/ModelingToolkit","RCCircuit.jmd")
```

Computer Information:

```
Julia Version 1.10.4
Commit 48d4fd48430 (2024-06-04 10:41 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
  JULIA_PKG_SERVER = juliahub.com

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Project.toml`
  [6e4b80f9] BenchmarkTools v1.5.0
  [336ed68f] CSV v0.10.14
⌅ [13f3f980] CairoMakie v0.11.11
  [a93c6f00] DataFrames v1.6.1
  [8391cb6b] JuliaSimCompiler v0.1.12
  [9cbdfd5a] JuliaSimCompilerRuntime v1.0.2
  [7ed4a6bd] LinearSolve v2.30.1
  [961ee093] ModelingToolkit v9.19.0
  [0f4fe800] OMJulia v0.3.2
  [1dea7af3] OrdinaryDiffEq v6.83.1
  [f27b6e38] Polynomials v4.0.11
  [ba661fbb] PreferenceTools v0.1.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [0c5d862f] Symbolics v5.30.3
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Manifest.toml`
  [47edcb42] ADTypes v1.3.0
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.36
  [79e6a3ab] Adapt v4.0.4
  [66dad0bd] AliasTables v1.1.3
  [27a7e980] Animations v0.4.1
  [ec485272] ArnoldiMethod v0.4.0
  [4fba245c] ArrayInterface v7.11.0
  [4c555306] ArrayLayouts v1.9.4
  [67c07d97] Automa v1.0.3
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.7
  [6e4b80f9] BenchmarkTools v1.5.0
  [e2ed5e7c] Bijections v0.1.6
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.5
  [00ebfdb7] CSTParser v3.4.3
  [336ed68f] CSV v0.10.14
  [159f3aea] Cairo v1.0.5
⌅ [13f3f980] CairoMakie v0.11.11
  [49dc2e85] Calculus v0.5.1
  [082447d4] ChainRules v1.69.0
  [d360d2e6] ChainRulesCore v1.24.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [944b1d66] CodecZlib v0.7.4
  [a2cac450] ColorBrewer v0.4.0
  [35d6a980] ColorSchemes v3.25.0
  [3da002f7] ColorTypes v0.11.5
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.11
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.15.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.5
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.6.1
  [864edb3b] DataStructures v0.18.20
  [e2d170a0] DataValueInterfaces v1.0.0
  [927a84f5] DelaunayTriangulation v1.0.3
  [2b5f629d] DiffEqBase v6.151.3
  [459566f4] DiffEqCallbacks v3.6.2
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [a0c0ee7d] DifferentiationInterface v0.5.5
  [31c24e10] Distributions v0.25.109
  [ffbed154] DocStringExtensions v0.9.3
  [5b8099bc] DomainSets v0.7.14
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.5.7
⌅ [06fc5a27] DynamicQuantities v0.13.2
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.7.5
  [429591f6] ExactPredicates v2.2.8
  [d4d017d3] ExponentialUtilities v1.26.1
  [e2ba6199] ExprTools v0.1.10
⌅ [6b7a57c9] Expronicon v0.8.5
  [411431e0] Extents v0.1.3
  [7a1cc6ca] FFTW v1.8.0
  [7034ab61] FastBroadcast v0.3.3
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.4
  [5789e2e9] FileIO v1.16.3
  [8fc22ac5] FilePaths v0.8.3
  [48062228] FilePathsBase v0.9.21
  [1a297f60] FillArrays v1.11.0
  [64ca27bc] FindFirstFunctions v1.2.0
  [6a86dc24] FiniteDiff v2.23.1
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.36
  [b38be410] FreeType v4.1.1
  [663a7486] FreeTypeAbstraction v0.10.3
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.4.11
  [46192b85] GPUArraysCore v0.1.6
  [61eb1bfa] GPUCompiler v0.26.5
  [c145ed77] GenericSchur v0.5.4
  [cf35fbd7] GeoInterface v1.3.4
  [5c1252a2] GeometryBasics v0.4.11
  [d7ba0133] Git v1.3.1
  [c27321d9] Glob v1.3.1
  [a2bd30eb] Graphics v1.1.2
  [86223c79] Graphs v1.11.1
⌅ [3955a311] GridLayoutBase v0.10.2
  [42e2da0e] Grisu v1.0.2
  [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [2803e5a7] ImageAxes v0.6.11
  [c817782e] ImageBase v0.1.7
  [a09fc81d] ImageCore v0.10.2
  [82e4d734] ImageIO v0.6.8
  [bc367c6b] ImageMetadata v0.9.9
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.5
  [842dd82b] InlineStrings v1.4.0
  [a98d9a8b] Interpolations v0.15.1
  [d1acc4aa] IntervalArithmetic v0.22.14
  [8197267c] IntervalSets v0.7.10
  [3587e190] InverseFunctions v0.1.14
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [b835a17e] JpegTurbo v0.1.5
  [98e50ef6] JuliaFormatter v1.0.56
  [9c9cc66b] JuliaSimBase v0.1.0
  [8391cb6b] JuliaSimCompiler v0.1.12
  [9cbdfd5a] JuliaSimCompilerRuntime v1.0.2
  [ccbc3e58] JumpProcesses v9.11.1
  [ef3ab10e] KLU v0.6.0
  [5ab0869b] KernelDensity v0.6.9
  [ba0b0d4f] Krylov v0.9.6
  [929cbde3] LLVM v7.2.1
  [b964fa9f] LaTeXStrings v1.3.1
  [2ee39098] LabelledArrays v1.16.0
  [984bce1d] LambertW v0.4.6
  [23fbe1c1] Latexify v0.16.3
  [10f19ff3] LayoutPointers v0.1.15
  [5078a376] LazyArrays v2.0.4
  [8cdb02fc] LazyModules v0.3.1
  [9c8b4983] LightXML v0.9.1
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.30.1
  [2ab3a3ac] LogExpFunctions v0.3.28
  [bdcacae8] LoopVectorization v0.12.170
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.13
⌅ [ee78f7c6] Makie v0.20.10
⌅ [20f20a25] MakieCore v0.7.3
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
⌅ [0a4f8689] MathTeXEngine v0.5.7
  [bb5d69b7] MaybeInplace v0.1.3
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
  [961ee093] ModelingToolkit v9.19.0
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.6
  [ffc61752] Mustache v1.0.19
  [d8a4904e] MutableArithmetics v1.4.5
  [d41bc354] NLSolversBase v7.8.3
  [77ba4419] NaNMath v1.0.2
  [f09324ee] Netpbm v1.1.1
  [8913a72c] NonlinearSolve v3.13.0
  [0f4fe800] OMJulia v0.3.2
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.14.0
  [52e1d378] OpenEXR v0.3.2
  [bac558e1] OrderedCollections v1.6.3
  [1dea7af3] OrdinaryDiffEq v6.83.1
  [90014a1f] PDMats v0.11.31
  [f57f5aa1] PNGFiles v0.4.3
  [65ce6f38] PackageExtensionCompat v1.0.2
  [19eb6ba3] Packing v0.5.0
  [5432bcbf] PaddedViews v0.5.12
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [eebad327] PkgVersion v0.3.3
  [995b91a9] PlotUtils v1.4.1
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.14
  [1d0040c9] PolyesterWeave v0.2.1
  [647866c9] PolygonOps v0.1.2
  [f27b6e38] Polynomials v4.0.11
  [2dfb63ee] PooledArrays v1.4.3
  [d236fae5] PreallocationTools v0.4.22
  [aea7be01] PrecompileTools v1.2.1
  [ba661fbb] PreferenceTools v0.1.2
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.3.2
  [92933f4c] ProgressMeter v1.10.0
  [43287f4e] PtrArrays v1.2.0
  [4b34888f] QOI v1.0.0
  [1fd47b50] QuadGK v2.9.4
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [731186ca] RecursiveArrayTools v3.24.0
  [f2c3362d] RecursiveFactorization v0.2.23
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.13
  [fdea26ae] SIMD v3.5.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
  [0bca4576] SciMLBase v2.41.3
  [31c91b34] SciMLBenchmarks v0.1.3
  [c0aeaf25] SciMLOperators v0.3.8
  [53ae85a6] SciMLStructures v1.3.0
  [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.3
  [efcf1570] Setfield v1.1.1
  [65257c39] ShaderAbstractions v0.4.1
  [992d4aef] Showoff v1.0.3
  [73760f76] SignedDistanceFields v0.4.0
  [727e6d20] SimpleNonlinearSolve v1.10.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [45858cf5] Sixel v0.1.3
  [66db9d55] SnoopPrecompile v1.0.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.19.0
  [dc90abb0] SparseInverseSubset v0.1.2
  [0a514795] SparseMatrixColorings v0.3.3
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.4.0
  [cae243ae] StackViews v0.1.1
  [aedffcd0] Static v0.8.10
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.5
  [1e83bf80] StaticArraysCore v1.4.3
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.3
  [4c63d2b9] StatsFuns v1.3.1
  [7792a7ef] StrideArraysCore v0.5.6
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [09ab397b] StructArrays v0.6.18
  [2efcf032] SymbolicIndexingInterface v0.3.23
  [19f23fe9] SymbolicLimits v0.2.1
  [d1185830] SymbolicUtils v2.0.2
  [0c5d862f] Symbolics v5.30.3
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [62fd8b95] TensorCore v0.1.1
⌅ [8ea1fca8] TermInterface v0.4.1
  [8290d209] ThreadingUtilities v0.5.2
  [731e570b] TiffImages v0.10.0
  [a759f4b9] TimerOutputs v0.5.24
  [0796e94c] Tokenize v0.5.29
  [3bb67fe8] TranscodingStreams v0.10.9
  [d5829a12] TriangularSolve v0.2.0
  [410a4b4d] Tricks v0.1.8
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.20.0
  [a7c27f48] Unityper v0.1.6
  [3d5dd08c] VectorizationBase v0.21.68
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [ea10d353] WeakRefStrings v1.4.2
  [44d3d7a6] Weave v0.10.12
  [efce3f68] WoodburyMatrices v1.0.0
  [76eceee3] WorkerUtilities v1.6.1
  [ddb6d928] YAML v0.4.11
  [c2297ded] ZMQ v1.2.6
  [6e34b625] Bzip2_jll v1.0.8+1
  [4e9b3aee] CRlibm_jll v1.0.1+0
  [83423d85] Cairo_jll v1.18.0+2
  [5ae413db] EarCut_jll v2.2.4+0
  [2e619515] Expat_jll v2.6.2+0
  [b22a6f82] FFMPEG_jll v6.1.1+0
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.96+0
  [d7e528f0] FreeType2_jll v2.13.2+0
  [559328eb] FriBidi_jll v1.0.14+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.44.0+2
  [7746bdde] Glib_jll v2.80.2+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [905a6f67] Imath_jll v3.1.11+0
  [1d5cc7b8] IntelOpenMP_jll v2024.1.0+0
  [aacddb02] JpegTurbo_jll v3.0.3+0
  [c1c5ebd0] LAME_jll v3.100.2+0
⌅ [dad2f222] LLVMExtra_jll v0.0.29+0
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [dd4b983a] LZO_jll v2.10.2+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.11+0
  [7add5ba3] Libgpg_error_jll v1.49.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.40.1+0
  [38a345b3] Libuuid_jll v2.40.1+0
  [856f044c] MKL_jll v2024.1.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [18a262bb] OpenEXR_jll v3.2.4+0
  [458c3c95] OpenSSL_jll v3.0.14+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [36c8627f] Pango_jll v1.52.2+0
  [30392449] Pixman_jll v0.43.4+0
  [f50d1b31] Rmath_jll v0.4.2+0
  [02c8fc9c] XML2_jll v2.12.7+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.11+0
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.5+0
  [9a68df92] isoband_jll v0.2.3+0
  [a4ae2306] libaom_jll v3.9.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.43+1
  [075b6546] libsixel_jll v1.10.3+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1317d2d5] oneTBB_jll v2021.12.0+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [8bf52ea8] CRC32c
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
  [05823500] OpenLibm_jll v0.8.1+2
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.8.0+1
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`
```

