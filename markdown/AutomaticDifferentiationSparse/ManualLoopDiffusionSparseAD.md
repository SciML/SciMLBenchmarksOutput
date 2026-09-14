---
author: "Yolhan Mannes"
title: "Diffusion operator loop sparse AD benchmarks"
---
```julia
using DifferentiationInterface
using DifferentiationInterfaceTest
using Chairmarks
using DataFrames
using LinearAlgebra
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings
import Chairmarks
import Enzyme, ForwardDiff, Mooncake
import Markdown, PrettyTables, Printf
```




## Backends tested

```julia
bcks = [
    AutoEnzyme(mode = Enzyme.Reverse),
    AutoEnzyme(mode = Enzyme.Forward),
    AutoMooncake(config = nothing),
    AutoForwardDiff(),
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    ),
    AutoSparse(
        AutoEnzyme(mode = Enzyme.Forward);
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
]
```

```
6-element Vector{ADTypes.AbstractADType}:
 ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, false, Enzyme
Core.FFIABI, false, false}())
 ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, f
alse, false, false}())
 ADTypes.AutoMooncake()
 ADTypes.AutoForwardDiff()
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=S
parseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Sparse
MatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColor
ings.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode
{false, EnzymeCore.FFIABI, false, false, false}()), sparsity_detector=Spars
eConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMatr
ixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings
.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))
```





## Diffusion operator simple loop

```julia
uin() = 0.0
uout() = 0.0
function Diffusion(u)
    du = zero(u)
    for i in eachindex(du, u)
        if i == 1
            ug = uin()
            ud = u[i + 1]
        elseif i == length(u)
            ug = u[i - 1]
            ud = uout()
        else
            ug = u[i - 1]
            ud = u[i + 1]
        end
        du[i] = ug + ud - 2*u[i]
    end
    return du
end;
```




## Manual jacobian

```julia
function DDiffusion(u)
    A = diagm(
        -1 => ones(length(u)-1),
        0=>-2 .* ones(length(u)),
        1 => ones(length(u)-1))
    return A
end;
```




## Define Scenarios

```julia
u = rand(1000)
scenarios = [Scenario{:jacobian, :out}(Diffusion, u; res1 = DDiffusion(u))];
```




## Run Benchmarks

```julia
df = DataFrame(benchmark_differentiation(bcks, scenarios))
table = PrettyTables.pretty_table(
    String,
    df;
    backend = :markdown,
    column_labels = names(df),
    formatters = [PrettyTables.fmt__printf("%.1e")]
)

Markdown.parse(table)
```

```
Test Summary:                                                              
                                                                           
                                                                           
                                                                           
                                                                  | Pass  T
otal     Time
Testing benchmarks                                                         
                                                                           
                                                                           
                                                                           
                                                                  |   12   
  12  3m17.2s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, false, Enzym
eCore.FFIABI, false, false}())                                             
                                                                           
                                                                           
                                                                  |    2   
   2  1m00.6s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, 
false, false, false}())                                                    
                                                                           
                                                                           
                                                                  |    2   
   2  1m06.8s
  ADTypes.AutoMooncake()                                                   
                                                                           
                                                                           
                                                                           
                                                                  |    2   
   2    55.2s
  ADTypes.AutoForwardDiff()                                                
                                                                           
                                                                           
                                                                           
                                                                  |    2   
   2     4.6s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=
SparseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Spars
eMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColo
rings.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))      
                                                                  |    2   
   2     5.3s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMod
e{false, EnzymeCore.FFIABI, false, false, false}()), sparsity_detector=Spar
seConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMat
rixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColoring
s.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false)) |    2   
   2     4.3s
```



|                                                                                                                                                                                                                                **backend** |                                                             **scenario** |       **operator** | **prepared** | **calls** | **samples** | **evals** | **time** | **allocs** | **bytes** | **gc_fraction** | **compile_fraction** |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| ------------------------------------------------------------------------:| ------------------:| ------------:| ---------:| -----------:| ---------:| --------:| ----------:| ---------:| ---------------:| --------------------:|
|                                                                                                                                                                  AutoEnzyme(mode=ReverseMode{false, false, false, FFIABI, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   6.4e+01 |     5.0e+00 |   1.0e+00 |  1.7e-01 |    6.6e+03 |   2.8e+08 |         4.8e-01 |              0.0e+00 |
|                                                                                                                                                                  AutoEnzyme(mode=ReverseMode{false, false, false, FFIABI, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   6.3e+01 |     6.0e+00 |   1.0e+00 |  1.0e-01 |    6.6e+03 |   2.8e+08 |         3.2e-01 |              0.0e+00 |
|                                                                                                                                                                         AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   6.3e+01 |     4.4e+01 |   1.0e+00 |  5.5e-03 |    5.3e+03 |   1.8e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                         AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   6.3e+01 |     3.0e+01 |   1.0e+00 |  5.6e-03 |    5.3e+03 |   1.8e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                             AutoMooncake() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   1.0e+00 |     1.1e+01 |   1.0e+00 |  7.2e-02 |    1.1e+04 |   3.3e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                             AutoMooncake() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   0.0e+00 |     1.1e+01 |   1.0e+00 |  7.0e-02 |    1.1e+04 |   3.3e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                          AutoForwardDiff() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   8.5e+01 |     7.9e+01 |   1.0e+00 |  3.8e-03 |    2.6e+02 |   1.7e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                          AutoForwardDiff() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   8.4e+01 |     1.3e+02 |   1.0e+00 |  3.8e-03 |    2.6e+02 |   1.7e+07 |         0.0e+00 |              0.0e+00 |
|                                                  AutoSparse(dense_ad=AutoForwardDiff(), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   2.0e+00 |     1.5e+04 |   1.0e+00 |  2.0e-05 |    1.5e+01 |   9.6e+04 |         0.0e+00 |              0.0e+00 |
|                                                  AutoSparse(dense_ad=AutoForwardDiff(), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   1.0e+00 |     2.0e+04 |   1.0e+00 |  1.7e-05 |    1.2e+01 |   8.8e+04 |         0.0e+00 |              0.0e+00 |
| AutoSparse(dense_ad=AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   2.0e+00 |     1.1e+04 |   1.0e+00 |  2.0e-05 |    2.3e+01 |   9.7e+04 |         0.0e+00 |              0.0e+00 |
| AutoSparse(dense_ad=AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   1.0e+00 |     1.8e+04 |   1.0e+00 |  1.5e-05 |    2.0e+01 |   8.9e+04 |         0.0e+00 |              0.0e+00 |



## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AutomaticDifferentiationSparse","ManualLoopDiffusionSparseAD.jmd")
```

Computer Information:

```
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver2)
  GC: Built with stock GC
Threads: 128 default, 1 interactive, 128 GC (on 128 virtual cores)
Environment:
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationSparse/Project.toml`
  [47edcb42] ADTypes v1.24.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [0ca39b1e] Chairmarks v1.3.1
  [a93c6f00] DataFrames v1.8.2
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [a82114a7] DifferentiationInterfaceTest v0.11.0
⌃ [7da242da] Enzyme v0.13.199
⌃ [f6369f11] ForwardDiff v1.4.5
⌃ [da2b9cff] Mooncake v0.5.48
  [91a5bcdd] Plots v1.41.7
  [d236fae5] PreallocationTools v1.7.1
  [08abe8d2] PrettyTables v3.4.8
  [31c91b34] SciMLBenchmarks v0.2.1 [loaded: `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/src/SciMLBenchmarks.jl` (v0.2.1) expected `/home/crackauc/.julia/packages/SciMLBenchmarks/ceJyd/src/SciMLBenchmarks.jl` (v0.2.1)]
  [9f842d2f] SparseConnectivityTracer v1.2.3
⌃ [0a514795] SparseMatrixColorings v0.4.27
⌃ [0c5d862f] Symbolics v7.39.0
  [44d3d7a6] Weave v0.10.12
⌃ [e88e6eb3] Zygote v0.7.12
  [37e2e46d] LinearAlgebra v1.12.0
  [2f01184e] SparseArrays v1.12.0
  [8dfed614] Test v1.11.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

And the full manifest:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationSparse/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
  [621f4979] AbstractFFTs v1.5.0
⌃ [6e696c72] AbstractPlutoDingetjes v1.4.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [66dad0bd] AliasTables v1.1.3
  [9b6a8646] AllocCheck v0.2.6
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.30.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [e2ed5e7c] Bijections v0.2.2
  [fa961155] CEnum v0.5.0
  [082447d4] ChainRules v1.73.0
  [d360d2e6] ChainRulesCore v1.26.1
  [0ca39b1e] Chairmarks v1.3.1
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌅ [861a8166] Combinatorics v1.0.2
  [bbf7d656] CommonSubexpressions v0.3.1
⌃ [f70d9fcc] CommonWorldInvalidations v1.2.0
  [34da2185] Compat v4.18.1
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.8.2
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [a82114a7] DifferentiationInterfaceTest v0.11.0
  [8d63f2c5] DispatchDoctor v0.4.28
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.8.1
⌃ [7c1d4256] DynamicPolynomials v0.6.7
  [4e289a0a] EnumX v1.0.7
⌃ [7da242da] Enzyme v0.13.199
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [55351af7] ExproniconLite v0.10.14
  [c87230d0] FFMPEG v0.4.5
  [1a297f60] FillArrays v1.17.0
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
⌃ [f6369f11] ForwardDiff v1.4.5
  [a85aefff] FunctionMaps v0.1.2
  [46192b85] GPUArraysCore v0.2.0
⌅ [61eb1bfa] GPUCompiler v1.23.0
  [28b8d3ca] GR v0.73.27
⌃ [86223c79] Graphs v1.14.0
  [42e2da0e] Grisu v1.0.2
⌅ [eafb193a] Highlights v0.5.3
  [7869d1d1] IRTools v0.4.20
  [d25df0c9] Inflate v0.1.5
⌅ [842dd82b] InlineStrings v1.4.5
  [18e54dd8] IntegerMathUtils v0.1.4
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
  [929cbde3] LLVM v9.13.1
  [b964fa9f] LaTeXStrings v1.4.1
  [23fbe1c1] Latexify v0.16.12
  [2ab3a3ac] LogExpFunctions v1.0.1
  [1914dd2f] MacroTools v0.5.16
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
⌃ [da2b9cff] Mooncake v0.5.48
  [2e0e35c7] Moshi v0.3.12
  [102ac46a] MultivariatePolynomials v0.5.19
  [ffc61752] Mustache v1.0.21
  [d8a4904e] MutableArithmetics v1.8.0
  [77ba4419] NaNMath v1.1.4
  [d8793406] ObjectFile v0.5.1
  [bac558e1] OrderedCollections v2.0.1
⌅ [69de0a69] Parsers v2.8.7 [loaded: v2.8.8]
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.7
  [2dfb63ee] PooledArrays v1.4.3
  [d236fae5] PreallocationTools v1.7.1
  [aea7be01] PrecompileTools v1.3.4
⌃ [21216c6a] Preferences v1.5.2 [loaded: v1.6.0]
  [08abe8d2] PrettyTables v3.4.8
  [27ebfcd6] Primes v0.5.7
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [988b38a3] ReadOnlyArrays v0.2.0
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.25
  [31c91b34] SciMLBenchmarks v0.2.1 [loaded: `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/src/SciMLBenchmarks.jl` (v0.2.1) expected `/home/crackauc/.julia/packages/SciMLBenchmarks/ceJyd/src/SciMLBenchmarks.jl` (v0.2.1)]
  [431bcebd] SciMLPublic v1.3.0
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.10
  [efcf1570] Setfield v1.1.2
⌃ [992d4aef] Showoff v1.0.3
  [699a6c99] SimpleTraits v0.9.6
  [a2af1166] SortingAlgorithms v1.2.3
  [9f842d2f] SparseConnectivityTracer v1.2.3
  [dc90abb0] SparseInverseSubset v0.1.3
⌃ [0a514795] SparseMatrixColorings v0.4.27
  [276daf66] SpecialFunctions v2.9.0
  [860ef19b] StableRNGs v1.0.4
  [0c0c59c1] StarAlgebras v0.3.0
⌃ [90137ffa] StaticArrays v1.9.19
  [1e83bf80] StaticArraysCore v1.4.4
⌃ [10745b16] Statistics v1.11.4
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.13
  [69024149] StringEncodings v0.3.7
⌅ [892a3eda] StringManipulation v0.5.0
  [09ab397b] StructArrays v0.7.3
  [53d494c1] StructIO v0.3.1
  [2efcf032] SymbolicIndexingInterface v0.3.55
⌃ [19f23fe9] SymbolicLimits v1.2.0
⌃ [d1185830] SymbolicUtils v4.46.1
⌃ [0c5d862f] Symbolics v7.39.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [e689c965] Tracy v0.1.6
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.2.0
  [d30d5f5c] WeakCacheSets v0.1.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
⌃ [e88e6eb3] Zygote v0.7.12
  [700de1a5] ZygoteRules v0.2.8
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
⌅ [7cc45869] Enzyme_jll v0.0.290+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.8.3+0
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
⌃ [2e76f6c2] HarfBuzz_jll v100.14003.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
⌃ [88015f11] LERC_jll v4.1.0+0
  [dad2f222] LLVMExtra_jll v0.0.47+0
⌃ [1d63c593] LLVMOpenMP_jll v22.1.7+0
  [ad6e5548] LibTracyClient_jll v0.13.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.58.2+0
  [30392449] Pixman_jll v0.46.4+0
  [c0090381] Qt6Base_jll v6.10.2+2
  [629bc702] Qt6Declarative_jll v6.10.2+2
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
⌃ [ffd25f8a] XZ_jll v5.8.3+0
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
  [009596ad] mtdev_jll v1.1.7+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.7.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [ac6e5ff7] JuliaSyntaxHighlighting v1.12.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [37e2e46d] LinearAlgebra v1.12.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.3.0
  [44cfe95a] Pkg v1.12.1
  [de0858da] Printf v1.11.0
  [9abbd945] Profile v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.12.0
  [f489334b] StyledStrings v1.11.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [e66e0078] CompilerSupportLibraries_jll v1.3.1+2
  [deac9b47] LibCURL_jll v8.15.0+0
  [e37daf67] LibGit2_jll v1.9.0+0
  [29816b5a] LibSSH2_jll v1.11.3+1
  [14a3606d] MozillaCACerts_jll v2025.11.4
  [4536629a] OpenBLAS_jll v0.3.29+0
  [05823500] OpenLibm_jll v0.8.7+0
  [458c3c95] OpenSSL_jll v3.5.6+0
  [efcefdf7] PCRE2_jll v10.44.0+1
  [bea87d4a] SuiteSparse_jll v7.8.3+2
  [83775a58] Zlib_jll v1.3.1+2
  [8e850b90] libblastrampoline_jll v5.15.0+0
  [8e850ede] nghttp2_jll v1.64.0+1
  [3f19e933] p7zip_jll v17.7.0+0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

