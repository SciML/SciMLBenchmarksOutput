---
author: "Chris Rackauckas"
title: "Simple Interval Rootfinding (NonlinearSolve.jl vs Roots.jl vs MATLAB)"
---


This example comes from 
[MATLAB's documentation showing improved rootfinding performance](https://twitter.com/walkingrandomly/status/1544615360833507329),
and thus can be assumed to be considered optimized from MATLAB's perspective. MATLAB's results are:

![](https://user-images.githubusercontent.com/1814174/262883161-d0ad6826-42fe-49c2-9645-0f08cfc3a723.png)

In comparison, Roots.jl:

```julia
using Roots, BenchmarkTools, Random

Random.seed!(42)

const N = 100_000;
levels = 1.5 .* rand(N);
out = zeros(N);
myfun(x, lv) = x * sin(x) - lv
function froots(out, levels, u0)
    for i in 1:N
        out[i] = find_zero(myfun, u0, levels[i])
    end
end

@btime froots(out, levels, (0, 2))
```

```
207.226 ms (0 allocations: 0 bytes)
```



```julia
using NonlinearSolve, BenchmarkTools

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]), ITP()).u
    end
end

function f2(out, levels, u0)
    for i in 1:N
        out[i] = solve(IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]), NonlinearSolve.Bisection()).u
    end
end

function f3(out, levels, u0)
    for i in 1:N
        out[i] = solve(NonlinearProblem{false}(NonlinearFunction{false}(myfun),
                u0, levels[i]), SimpleNewtonRaphson()).u
    end
end

@btime f(out, levels, (0.0, 2.0))
@btime f2(out, levels, (0.0, 2.0))
@btime f3(out, levels, 1.0)
```

```
33.746 ms (0 allocations: 0 bytes)
  97.575 ms (0 allocations: 0 bytes)
  12.851 ms (0 allocations: 0 bytes)
```





MATLAB 2022a reportedly achieves 1.66s. Try this code yourself: we receive ~0.05 seconds, or a 33x speedup.


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/IntervalNonlinearProblem","simpleintervalrootfind.jmd")
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
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/IntervalNonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.6.0
⌃ [8913a72c] NonlinearSolve v4.8.0
⌃ [f2b01f46] Roots v2.2.7
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
⌃ [727e6d20] SimpleNonlinearSolve v2.3.0
  [de0858da] Printf
  [9a3f8284] Random
  [10745b16] Statistics v1.10.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/IntervalNonlinearProblem/Manifest.toml`
⌃ [47edcb42] ADTypes v1.14.0
  [7d9f7c33] Accessors v0.1.42
  [79e6a3ab] Adapt v4.3.0
⌃ [4fba245c] ArrayInterface v7.18.0
  [4c555306] ArrayLayouts v1.11.1
  [6e4b80f9] BenchmarkTools v1.6.0
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [70df07ce] BracketingNonlinearSolve v1.2.0
  [2a0fbf3d] CPUSummary v0.2.6
  [336ed68f] CSV v0.10.15
⌃ [d360d2e6] ChainRulesCore v1.25.1
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [944b1d66] CodecZlib v0.7.8
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
⌃ [34da2185] Compat v4.16.0
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [8f4d0f93] Conda v1.10.2
⌃ [187b0558] ConstructionBase v1.5.8
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.7.0
  [864edb3b] DataStructures v0.18.22
  [e2d170a0] DataValueInterfaces v1.0.0
⌃ [2b5f629d] DiffEqBase v6.170.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌅ [a0c0ee7d] DifferentiationInterface v0.6.52
⌃ [ffbed154] DocStringExtensions v0.9.4
  [4e289a0a] EnumX v1.0.5
⌃ [f151be2c] EnzymeCore v0.8.8
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
⌃ [a4df4552] FastPower v1.1.2
  [48062228] FilePathsBase v0.9.24
  [1a297f60] FillArrays v1.13.0
  [6a86dc24] FiniteDiff v2.27.0
  [f6369f11] ForwardDiff v1.0.1
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.2.0
⌃ [d7ba0133] Git v1.3.1
  [eafb193a] Highlights v0.5.3
⌃ [7073ff75] IJulia v1.27.0
  [615f187c] IfElse v0.1.1
⌃ [842dd82b] InlineStrings v1.4.3
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.4
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.7.0
  [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
  [ba0b0d4f] Krylov v0.10.1
  [b964fa9f] LaTeXStrings v1.4.0
  [10f19ff3] LayoutPointers v0.1.17
  [5078a376] LazyArrays v2.6.1
⌃ [9c8b4983] LightXML v0.9.1
  [87fe0de2] LineSearch v0.1.4
⌃ [7ed4a6bd] LinearSolve v3.9.0
  [2ab3a3ac] LogExpFunctions v0.3.29
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
⌃ [2e0e35c7] Moshi v0.3.5
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [ffc61752] Mustache v1.0.20
  [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.8.0
⌃ [be0214bd] NonlinearSolveBase v1.6.0
⌃ [5959db7a] NonlinearSolveFirstOrder v1.4.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.3.0
  [26075421] NonlinearSolveSpectralMethods v1.2.0
  [0f4fe800] OMJulia v0.3.2
⌃ [bac558e1] OrderedCollections v1.8.0
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.3
⌃ [f517fe37] Polyester v0.7.16
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.4.0
  [3cdcf5f2] RecipesBase v1.3.4
⌃ [731186ca] RecursiveArrayTools v3.33.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [f2b01f46] Roots v2.2.7
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.14
  [94e857df] SIMDTypes v0.1.0
⌃ [0bca4576] SciMLBase v2.86.2
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
⌃ [19f34311] SciMLJacobianOperators v0.1.3
⌅ [c0aeaf25] SciMLOperators v0.3.13
  [53ae85a6] SciMLStructures v1.7.0
⌃ [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.8
  [efcf1570] Setfield v1.1.2
⌃ [727e6d20] SimpleNonlinearSolve v2.3.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
⌃ [0a514795] SparseMatrixColorings v0.4.19
  [276daf66] SpecialFunctions v2.5.1
  [aedffcd0] Static v1.2.0
  [0d7ed370] StaticArrayInterface v1.8.0
  [1e83bf80] StaticArraysCore v1.4.3
  [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.4.1
⌃ [2efcf032] SymbolicIndexingInterface v0.3.40
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.12.0
⌃ [8290d209] ThreadingUtilities v0.5.3
⌃ [a759f4b9] TimerOutputs v0.5.28
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
  [3a884ed6] UnPack v1.0.2
  [81def892] VersionParsing v1.3.0
  [ea10d353] WeakRefStrings v1.4.2
  [44d3d7a6] Weave v0.10.12
  [76eceee3] WorkerUtilities v1.6.1
⌃ [ddb6d928] YAML v0.4.13
⌃ [c2297ded] ZMQ v1.4.0
  [2e619515] Expat_jll v2.6.5+0
⌃ [f8c6e375] Git_jll v2.49.0+0
  [1d5cc7b8] IntelOpenMP_jll v2025.0.4+0
  [94ce4f54] Libiconv_jll v1.18.0+0
  [856f044c] MKL_jll v2025.0.1+1
⌃ [458c3c95] OpenSSL_jll v3.5.0+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
⌅ [02c8fc9c] XML2_jll v2.13.6+1
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [a9144af2] libsodium_jll v1.0.21+0
  [1317d2d5] oneTBB_jll v2022.0.0+0
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
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.1+4
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

