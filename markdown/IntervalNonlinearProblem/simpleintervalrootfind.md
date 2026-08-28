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
        out[i] = solve(ZeroProblem(myfun, u0), levels[i])
    end
    return
end

@btime froots(out, levels, (0, 2))
```

```
107.438 ms (0 allocations: 0 bytes)
```



```julia
using BracketingNonlinearSolve, SimpleNonlinearSolve, BenchmarkTools
using BracketingNonlinearSolve: Bisection # Roots also exports Bisection leading to a name conflict

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(
            IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]
            ),
            ITP()
        ).u
    end
    return
end

function f2(out, levels, u0)
    for i in 1:N
        out[i] = solve(
            IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]
            ),
            Bisection()
        ).u
    end
    return
end

function f3(out, levels, u0)
    for i in 1:N
        out[i] = solve(
            NonlinearProblem{false}(
                NonlinearFunction{false}(myfun),
                u0, levels[i]
            ),
            SimpleNewtonRaphson()
        ).u
    end
    return
end

@btime f(out, levels, (0.0, 2.0))
@btime f2(out, levels, (0.0, 2.0))
@btime f3(out, levels, 1.0)
```

```
46.298 ms (0 allocations: 0 bytes)
  128.755 ms (0 allocations: 0 bytes)
  17.460 ms (0 allocations: 0 bytes)
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
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/IntervalNonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.8.0
  [70df07ce] BracketingNonlinearSolve v1.12.6
  [f2b01f46] Roots v3.0.7
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [10745b16] Statistics v1.11.1
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

And the full manifest:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/IntervalNonlinearProblem/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [4fba245c] ArrayInterface v7.30.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [70df07ce] BracketingNonlinearSolve v1.12.6
  [38540f10] CommonSolve v0.2.14
  [bbf7d656] CommonSubexpressions v0.3.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.8
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [e2d170a0] DataValueInterfaces v1.0.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [ffbed154] DocStringExtensions v0.9.5
  [4e289a0a] EnumX v1.0.7
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [9aa1b823] FastClosures v0.3.2
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
  [f6369f11] ForwardDiff v1.4.5
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [46192b85] GPUArraysCore v0.2.0
  [d7ba0133] Git v1.5.0
⌅ [eafb193a] Highlights v0.5.3
  [7073ff75] IJulia v1.34.4
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [b964fa9f] LaTeXStrings v1.4.1
  [87fe0de2] LineSearch v0.1.16
  [2ab3a3ac] LogExpFunctions v1.0.1
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [bb5d69b7] MaybeInplace v0.1.8
  [ffc61752] Mustache v1.0.21
  [77ba4419] NaNMath v1.1.4
⌃ [be0214bd] NonlinearSolveBase v2.48.0
  [bac558e1] OrderedCollections v2.0.1
⌅ [69de0a69] Parsers v2.8.7
⌃ [d236fae5] PreallocationTools v1.7.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [08abe8d2] PrettyTables v3.4.8
  [3cdcf5f2] RecipesBase v1.3.4
  [731186ca] RecursiveArrayTools v4.5.1
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [9fe22ead] RespecializeParams v1.3.0
  [f2b01f46] Roots v3.0.7
  [7e49a35a] RuntimeGeneratedFunctions v0.5.25
⌃ [0bca4576] SciMLBase v3.49.2
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.18
  [a6db7da4] SciMLLogging v2.1.0
  [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
  [53ae85a6] SciMLStructures v1.10.5
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [276daf66] SpecialFunctions v2.9.0
  [1e83bf80] StaticArraysCore v1.4.4
  [10745b16] Statistics v1.11.1
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.5.0
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [a759f4b9] TimerOutputs v1.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [2e619515] Expat_jll v2.8.3+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.55.0+0
  [94ce4f54] Libiconv_jll v1.18.0+0
  [9bd350c2] OpenSSH_jll v10.5.1+0
  [458c3c95] OpenSSL_jll v3.5.8+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [a9144af2] libsodium_jll v1.0.21+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
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
  [9abbd945] Profile v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [f489334b] StyledStrings v1.11.0
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
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
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

