---
author: "Oscar Dowson"
title: "CLNLBEAM Nonlinear Optimization Benchmark"
---


## Introduction

This benchmark is the clnlbeam example, adapted from H. Maurer and H.D.
Mittelman, "The non-linear beam via optimal control with bound state variables,"
Optimal Control Applications and Methods 12, pp.19-31, 1991.

This benchmark uses the following packages:

```julia
import Enzyme
import ForwardDiff
import Ipopt
import JuMP
import ModelingToolkit as MTK
import Optimization
import OptimizationMOI
import Plots
import ReverseDiff
import Test
```




## Optimization.jl

```julia
function run_optimization(N::Int, automatic_differentiation)
    h = 1 / N
    alpha = 350
    x_offset = N + 1
    u_offset = 2(N + 1)
    function objective_fn(x, p)
        return sum(
            0.5 * h * (x[x_offset + i + 1]^2 + x[x_offset + i]^2) +
            0.5 * alpha * h * (cos(x[i + 1]) + cos(x[i])) for i in 1:N
        )
    end
    function constraint_fn(res, x, p)
        for i in 1:N
            res[i] = x[x_offset + i + 1] - x[x_offset + i] -
                     0.5 * h * (sin(x[i + 1]) + sin(x[i]))
        end
        for i in 1:N
            res[N + i] = x[i + 1] - x[i] - 0.5 * h * x[u_offset + i + 1] -
                         0.5 * h * x[u_offset + i]
        end
        return
    end
    prob = Optimization.OptimizationProblem(
        Optimization.OptimizationFunction(
            objective_fn,
            automatic_differentiation;
            cons = constraint_fn
        ),
        zeros(3 * (N + 1)),
        nothing;
        lb = vcat(fill(-1.0, N+1), fill(-0.05, N+1), fill(-Inf, N+1)),
        ub = vcat(fill(1.0, N+1), fill(0.05, N+1), fill(Inf, N+1)),
        lcons = zeros(2 * N),
        ucons = zeros(2 * N)
    )
    sol = Optimization.solve(prob, Ipopt.Optimizer(); print_level = 0)
    Test.@test ≈(sol.objective, 350.0; atol = 1e-6)
    Test.@test ≈(sol.u, zeros(3 * (N + 1)); atol = 1e-6)
    return
end
```

```
run_optimization (generic function with 1 method)
```





We test three different backends to Optimization.jl:

```julia
function run_enzyme_diff(N::Int)
    run_optimization(N, Optimization.AutoEnzyme())
    return
end

function run_forward_diff(N::Int)
    run_optimization(N, Optimization.AutoSparse(Optimization.AutoForwardDiff()))
    return
end

function run_reverse_diff(N::Int)
    run_optimization(N, Optimization.AutoSparse(Optimization.AutoReverseDiff(true)))
    return
end
```

```
run_reverse_diff (generic function with 1 method)
```





## JuMP.jl

```julia
function run_jump(N::Int)
    h = 1 / N
    alpha = 350
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_attribute(model, "print_level", 0)
    JuMP.@variables(model, begin
        -1 <= t[1:(N + 1)] <= 1
        -0.05 <= x[1:(N + 1)] <= 0.05
        u[1:(N + 1)]
    end)
    JuMP.@objective(model,
        Min,
        sum(
            0.5 * h * (u[i + 1]^2 + u[i]^2) +
            0.5 * alpha * h * (cos(t[i + 1]) + cos(t[i])) for i in 1:N
        ),)
    JuMP.@constraint(model,
        [i = 1:N],
        x[i + 1] - x[i] - 0.5 * h * (sin(t[i + 1]) + sin(t[i])) == 0,)
    JuMP.@constraint(model,
        [i = 1:N],
        t[i + 1] - t[i] - 0.5 * h * u[i + 1] - 0.5 * h * u[i] == 0,)
    JuMP.optimize!(model)
    Test.@test ≈(JuMP.objective_value(model), 350.0; atol = 1e-6)
    Test.@test ≈(JuMP.value.(t), zeros((N + 1)); atol = 1e-6)
    Test.@test ≈(JuMP.value.(x), zeros((N + 1)); atol = 1e-6)
    Test.@test ≈(JuMP.value.(u), zeros((N + 1)); atol = 1e-6)
    return
end
```

```
run_jump (generic function with 1 method)
```





## ModelingToolkit.jl

```julia
function run_modelingtoolkit(N::Int, use_structural_simplify::Bool = true)
    h = 1 / N
    alpha = 350
    MTK.@variables t[1:(N + 1)]
    MTK.@variables x[1:(N + 1)]
    MTK.@variables u[1:(N + 1)]
    t = [MTK.ModelingToolkitBase.setbounds(ti, (-1.0, 1.0)) for ti in collect(t)]
    x = [MTK.ModelingToolkitBase.setbounds(xi, (-0.05, 0.05)) for xi in collect(x)]
    u = collect(u)
    loss = sum(
        0.5 * h * (u[i + 1]^2 + u[i]^2) +
        0.5 * alpha * h * (cos(t[i + 1]) + cos(t[i])) for i in 1:N
    )
    cons = vcat(
        [x[i + 1] - x[i] - 0.5 * h * (sin(t[i + 1]) + sin(t[i])) ~ 0 for i in 1:N],
        [t[i + 1] - t[i] - 0.5 * h * u[i + 1] - 0.5 * h * u[i] ~ 0 for i in 1:N]
    )
    vars = vcat(t, x, u)
    system = MTK.complete(MTK.OptimizationSystem(
        loss,
        vars,
        [];
        constraints = cons,
        name = :clnlbeam
    ))
    if use_structural_simplify
        system = MTK.mtkcompile(system)
    end
    prob = Optimization.OptimizationProblem(
        system,
        Dict(k => 0.0 for k in MTK.unknowns(system));
        grad = true,
        hess = true,
        cons_j = true,
        cons_h = true,
        cons_sparse = true,
        sparse = true
    )
    sol = Optimization.solve(prob, Ipopt.Optimizer(); print_level = 0)
    Test.@test ≈(sol[loss], 350.0; atol = 1e-6)
    Test.@test ≈(sol[vars], zeros(3 * (N + 1)); atol = 1e-6)
    return
end

function run_modelingtoolkit_no_simplify(N::Int)
    run_modelingtoolkit(N, false)
    return
end
```

```
run_modelingtoolkit_no_simplify (generic function with 1 method)
```





## Benchmark

```julia
function run_benchmark(N; time_limit::Float64 = 1.0)
    function _elapsed(f::F, n::Int) where {F <: Function}
        # We use the minimum of three runs here. We could also use
        # `return BenchmarkTools.@belapsed \$f(\$n)` but it took much longer to
        # run.
        return minimum(@elapsed f(n) for _ in 1:3)
    end
    benchmarks = (
        run_enzyme_diff,
        run_forward_diff,
        run_reverse_diff,
        run_modelingtoolkit,
        run_modelingtoolkit_no_simplify,
        run_jump
    )
    data = fill(NaN, length(N), length(benchmarks))
    for (i, n) in enumerate(N), (j, f) in enumerate(benchmarks)

        if i == 1 || data[i - 1, j] < time_limit
            @info "Running $f($n)"
            data[i, j] = _elapsed(f, n)
        end
    end
    return Plots.plot(
        N,
        data;
        labels = ["Optimization(Enzyme)" "Optimization(ForwardDiff)" "Optimization(ReverseDiff)" "MTK(simplify)" "MTK(no simplify)" "JuMP"],
        xlabel = "N",
        ylabel = "Total time [seconds]",
        ylims = (0, time_limit)
    )
end
```

```
run_benchmark (generic function with 1 method)
```



```julia
plt = run_benchmark(vcat(1:10, 20:20:200))
```

```
***************************************************************************
***
This program contains Ipopt, a library for large-scale nonlinear optimizati
on.
 Ipopt is released as open source code under the Eclipse Public License (EP
L).
         For more information visit https://github.com/coin-or/Ipopt
***************************************************************************
***
```


![](figures/clnlbeam_7_1.png)


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/OptimizationFrameworks","clnlbeam.jmd")
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
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/OptimizationFrameworks/Project.toml`
  [54578032] ADNLPModels v0.8.13
  [6e4b80f9] BenchmarkTools v1.8.0
  [2569d6c7] ConcreteStructs v0.2.8
⌃ [992eb4ea] CondaPkg v0.2.33
  [a93c6f00] DataFrames v1.8.2
⌃ [7da242da] Enzyme v0.13.203
  [f6369f11] ForwardDiff v1.4.6
  [b6b21f68] Ipopt v1.16.0
  [4076af6c] JuMP v1.31.2
  [961ee093] ModelingToolkit v11.43.1
  [f4238b75] NLPModelsIpopt v0.11.3
⌅ [429524aa] Optim v1.13.3
  [7f7a1694] Optimization v5.9.1
  [bca83a33] OptimizationBase v5.6.1
  [fd9f6733] OptimizationMOI v1.4.1
  [91a5bcdd] Plots v1.41.7
⌃ [c36e90e8] PowerModels v0.21.5
  [08abe8d2] PrettyTables v3.4.8
  [6099a3de] PythonCall v0.9.35
  [37e2e3b7] ReverseDiff v1.17.0
  [31c91b34] SciMLBenchmarks v0.2.1
  [860ef19b] StableRNGs v1.0.4
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [0c5d862f] Symbolics v7.39.2
  [76f85450] LibGit2 v1.11.0
  [8dfed614] Test v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/OptimizationFrameworks/Manifest.toml`
  [54578032] ADNLPModels v0.8.13
  [47edcb42] ADTypes v1.24.0
  [14f7f29c] AMD v0.5.4
  [6e696c72] AbstractPlutoDingetjes v1.4.1
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [66dad0bd] AliasTables v1.1.3
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.30.1
  [4c555306] ArrayLayouts v1.12.2
  [aae01518] BandedMatrices v1.12.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [e2ed5e7c] Bijections v0.2.2
  [b2a6c25c] BinaryHeaps v1.1.0
  [caf10ac8] BipartiteGraphs v0.1.14
  [8e7c35d0] BlockArrays v1.10.0
  [70df07ce] BracketingNonlinearSolve v1.12.7
  [fa961155] CEnum v0.5.0
  [d360d2e6] ChainRulesCore v1.26.1
  [523fee87] CodecBzip2 v0.8.5
  [944b1d66] CodecZlib v0.7.9
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌅ [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.14
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.2.2
  [34da2185] Compat v4.18.1
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.8
⌃ [992eb4ea] CondaPkg v0.2.33
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.8.2
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v7.21.1
  [459566f4] DiffEqCallbacks v4.19.4
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [b4f34e82] Distances v0.10.12
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.8.1
  [7c1d4256] DynamicPolynomials v0.6.8
  [4e289a0a] EnumX v1.0.7
⌃ [7da242da] Enzyme v0.13.203
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [55351af7] ExproniconLite v0.10.14
  [c87230d0] FFMPEG v0.4.5
  [7034ab61] FastBroadcast v1.4.0
  [9aa1b823] FastClosures v0.3.2
  [a4df4552] FastPower v1.5.0
  [1a297f60] FillArrays v1.17.0
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.4.6
  [a85aefff] FunctionMaps v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [46192b85] GPUArraysCore v0.2.0
⌅ [61eb1bfa] GPUCompiler v1.23.0
  [28b8d3ca] GR v0.73.27
  [86223c79] Graphs v1.15.0
  [076d061b] HashArrayMappedTries v0.2.0
⌅ [eafb193a] Highlights v0.5.3
  [3263718b] ImplicitDiscreteSolve v2.3.0
  [d25df0c9] Inflate v0.1.5
  [2030c09a] InfrastructureModels v0.7.9
⌅ [842dd82b] InlineStrings v1.4.6
  [18e54dd8] IntegerMathUtils v0.1.4
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [b6b21f68] Ipopt v1.16.0
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [ae98c720] Jieko v0.2.1
  [4076af6c] JuMP v1.31.2
⌃ [ccbc3e58] JumpProcesses v9.32.3
  [ba0b0d4f] Krylov v0.10.10
  [2faa5264] LHLFactorization v2.2.2
  [929cbde3] LLVM v9.13.1
  [b964fa9f] LaTeXStrings v1.4.1
  [23fbe1c1] Latexify v0.16.12
  [1d6d02ad] LeftChildRightSiblingTrees v0.3.0
  [87fe0de2] LineSearch v0.1.18
⌃ [d3d80556] LineSearches v7.5.1
  [5c8ed15e] LinearOperators v2.14.2
⌃ [7ed4a6bd] LinearSolve v5.17.3
  [2ab3a3ac] LogExpFunctions v1.0.1
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [b8f27783] MathOptInterface v1.53.0
  [bb5d69b7] MaybeInplace v0.1.8
  [442fdcdd] Measures v0.3.3
  [f28f55f0] Memento v1.5.0
  [0b3b1443] MicroMamba v0.1.15
  [e1d29d7a] Missings v1.2.0
  [961ee093] ModelingToolkit v11.43.1
⌃ [7771a370] ModelingToolkitBase v1.70.0
  [6bb917b9] ModelingToolkitTearing v1.20.6
⌅ [2e0e35c7] Moshi v0.3.9
  [46d2c3a1] MuladdMacro v0.2.7
  [102ac46a] MultivariatePolynomials v0.5.19
  [ffc61752] Mustache v1.0.21
  [d8a4904e] MutableArithmetics v1.8.0
  [a4795742] NLPModels v0.21.12
  [f4238b75] NLPModelsIpopt v0.11.3
  [e01155f1] NLPModelsModifiers v0.8.0
⌅ [d41bc354] NLSolversBase v7.10.0
⌅ [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.1.4
⌃ [be0214bd] NonlinearSolveBase v2.48.0
⌃ [5959db7a] NonlinearSolveFirstOrder v2.6.1
  [d8793406] ObjectFile v0.5.1
  [6fe1bfb0] OffsetArrays v1.17.0
⌅ [429524aa] Optim v1.13.3
  [7f7a1694] Optimization v5.9.1
  [bca83a33] OptimizationBase v5.6.1
  [fd9f6733] OptimizationMOI v1.4.1
  [bac558e1] OrderedCollections v2.0.1
⌃ [bbf590c4] OrdinaryDiffEqCore v4.17.2
⌅ [69de0a69] Parsers v2.8.8
  [fa939f87] Pidfile v1.3.0
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.7
  [e409e4f3] PoissonRandom v0.4.13
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
⌃ [c36e90e8] PowerModels v0.21.5
  [d236fae5] PreallocationTools v1.7.1
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.6.0
  [08abe8d2] PrettyTables v3.4.8
  [27ebfcd6] Primes v0.5.7
  [33c8b6b6] ProgressLogging v0.1.6
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.5.0
  [6099a3de] PythonCall v0.9.35
  [988b38a3] ReadOnlyArrays v0.2.0
  [795d4caa] ReadOnlyDicts v1.0.1
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v4.5.1
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [9fe22ead] RespecializeParams v1.3.0
  [37e2e3b7] ReverseDiff v1.17.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.26
  [9dfe8606] SCCNonlinearSolve v1.15.3
⌃ [0bca4576] SciMLBase v3.54.0
  [31c91b34] SciMLBenchmarks v0.2.1
  [19f34311] SciMLJacobianOperators v0.1.19
  [a6db7da4] SciMLLogging v2.1.0
⌃ [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
  [53ae85a6] SciMLStructures v1.10.5
  [7e506255] ScopedValues v1.6.2
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.10
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.1.1
  [727e6d20] SimpleNonlinearSolve v2.14.5
  [699a6c99] SimpleTraits v0.9.6
  [ff4d7338] SolverCore v0.3.10
  [a2af1166] SortingAlgorithms v1.2.3
  [a57abbd0] SparseColumnPivotedQR v2.1.8
  [9f842d2f] SparseConnectivityTracer v1.2.3
  [0a514795] SparseMatrixColorings v0.4.28
  [276daf66] SpecialFunctions v2.9.0
  [860ef19b] StableRNGs v1.0.4
  [0c0c59c1] StarAlgebras v0.3.0
  [64909d44] StateSelection v1.11.1
  [90137ffa] StaticArrays v1.9.20
  [1e83bf80] StaticArraysCore v1.4.4
  [10745b16] Statistics v1.11.5
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.13
  [69024149] StringEncodings v0.3.7
⌅ [892a3eda] StringManipulation v0.5.0
  [53d494c1] StructIO v0.3.1
  [856f2bd8] StructTypes v1.11.0
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [19f23fe9] SymbolicLimits v1.2.1
  [d1185830] SymbolicUtils v4.46.6
  [0c5d862f] Symbolics v7.39.2
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [5d786b92] TerminalLoggers v0.1.8
⌅ [a759f4b9] TimerOutputs v0.5.29
  [e689c965] Tracy v0.1.6
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [e17b2a0c] UnsafePointers v1.0.0
  [41fe7b60] Unzip v0.2.0
  [d30d5f5c] WeakCacheSets v0.1.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [ae81ac8f] ASL_jll v0.1.5+0
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [7cc45869] Enzyme_jll v0.0.293+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.8.4+0
⌅ [b22a6f82] FFMPEG_jll v8.1.2+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.5.1+0
  [d2c73de3] GR_jll v0.73.27+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [7746bdde] Glib_jll v2.88.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
  [2e76f6c2] HarfBuzz_jll v100.14004.0+0
  [e33a78d0] Hwloc_jll v2.14.0+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
⌅ [9cc047cb] Ipopt_jll v300.1400.1902+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.2.0+0
  [dad2f222] LLVMExtra_jll v0.0.47+0
  [1d63c593] LLVMOpenMP_jll v23.1.1+0
  [ad6e5548] LibTracyClient_jll v0.13.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [d00139f3] METIS_jll v5.1.4+0
  [856f044c] MKL_jll v2025.2.0+0
  [d7ed1dd3] MUMPS_seq_jll v500.900.100+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [656ef2d0] OpenBLAS32_jll v0.3.34+0
  [458c3c95] OpenSSL_jll v3.5.8+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.58.2+0
  [30392449] Pixman_jll v0.46.4+0
  [c0090381] Qt6Base_jll v6.10.2+2
  [629bc702] Qt6Declarative_jll v6.10.2+2
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
  [319450e9] SPRAL_jll v2025.9.18+1
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
⌅ [02c8fc9c] XML2_jll v2.13.9+0
  [ffd25f8a] XZ_jll v5.8.4+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.13+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.8+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.2+0
  [a51aa0fd] Xorg_libXi_jll v1.8.4+0
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
  [33bec58e] Xorg_xkeyboard_config_jll v2.47.0+2
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
⌅ [214eeab7] fzf_jll v0.61.1+0
  [a4ae2306] libaom_jll v3.14.1+0
  [0ac62f75] libass_jll v0.17.5+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [8e53e030] libdrm_jll v2.4.134+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.58+0
  [9a156e7d] libva_jll v2.23.0+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [f8abcde7] micromamba_jll v2.3.1+0
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.3.0+0
  [4d7b5844] pixi_jll v0.76.2+0
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
  [9abbd945] Profile v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.11.0
  [f489334b] StyledStrings v1.11.0
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

