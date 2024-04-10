---
author: "Chris Rackauckas"
title: "Orbital Dynamics BVP Benchmark"
---


This is a benchmark of a small non-stiff BVP.

```julia
using BoundaryValueDiffEq, OrdinaryDiffEq, BenchmarkTools

y0 = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    -5528.612564911408,
    1216.8442360202787,
    4845.114446429901,
]
init_val = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    7.0526926403748598E+06,
    -7.9650476230388973E+05,
    -1.1911128863666430E+06,
]
J2 = 1.08262668E-3
req = 6378137
myu = 398600.4418E+9
t0 = 86400 * 2.3577475462484435E+04
t1 = 86400 * 2.3577522023524125E+04
tspan = (t0, t1)

# ODE solver
function orbital(dy, y, p, t)
    r2 = (y[1]^2 + y[2]^2 + y[3]^2)
    r3 = r2^(3 / 2)
    w = 1 + 1.5J2 * (req * req / r2) * (1 - 5y[3] * y[3] / r2)
    w2 = 1 + 1.5J2 * (req * req / r2) * (3 - 5y[3] * y[3] / r2)
    dy[1] = y[4]
    dy[2] = y[5]
    dy[3] = y[6]
    dy[4] = -myu * y[1] * w / r3
    dy[5] = -myu * y[2] * w / r3
    dy[6] = -myu * y[3] * w2 / r3
end

function bc!_generator(resid, sol, init_val)
    resid[1] = sol[1][1] - init_val[1]
    resid[2] = sol[1][2] - init_val[2]
    resid[3] = sol[1][3] - init_val[3]
    resid[4] = sol[end][1] - init_val[4]
    resid[5] = sol[end][2] - init_val[5]
    resid[6] = sol[end][3] - init_val[6]
end
cur_bc! = (resid, sol, p, t) -> bc!_generator(resid, sol, init_val)
resid_f = Array{Float64}(undef, 6)
bvp = BVProblem(orbital, cur_bc!, y0, tspan)
```

```
BVProblem with uType Vector{Float64} and tType Float64. In-place: true
timespan: (2.037093879958655e9, 2.0370979028324845e9)
u0: 6-element Vector{Float64}:
      -4.776316976285399e6
 -383863.9870444152
      -5.350018393313232e6
   -5528.612564911408
    1216.8442360202787
    4845.114446429901
```



```julia
@btime sol = solve(bvp, Shooting(DP5()), abstol = 1e-13, reltol = 1e-13)
```

```
9.767 s (271864850 allocations: 8.87 GiB)
retcode: MaxIters
Interpolation: specialized 4th order "free" interpolation
t: 597-element Vector{Float64}:
 2.037093879958655e9
 2.0370938799622977e9
 2.0370938799987223e9
 2.0370938803629692e9
 2.037093884005438e9
 2.0370938890873458e9
 2.03709389546286e9
 2.0370939024988694e9
 2.0370939099517152e9
 2.0370939175970695e9
 ⋮
 2.0370978542279081e9
 2.0370978607705305e9
 2.037097867387156e9
 2.0370978740753317e9
 2.0370978808328161e9
 2.0370978876575065e9
 2.0370978945474555e9
 2.037097901500842e9
 2.0370979028324845e9
u: 597-element Vector{Vector{Float64}}:
 [-4.776316976285399e6, -383863.9870444152, -5.350018393313232e6, -5528.612
563131059, 1216.844236158524, 4845.114447868596]
 [-4.77633711405097e6, -383859.5547243374, -5.350000745096112e6, -5528.5938
87445432, 1216.8457370778892, 4845.135420386226]
 [-4.776538487965223e6, -383815.2312228838, -5.349824258723416e6, -5528.407
126238267, 1216.8607453199836, 4845.345141780103]
 [-4.778551852902747e6, -383371.9661535982, -5.348058974897674e6, -5526.539
079130501, 1217.0107325805616, 4847.441977415121]
 [-4.79864802319921e6, -378936.32281508803, -5.33036417780586e6, -5507.8151
581240045, 1218.5010846666296, 4868.372441826212]
 [-4.826571583990289e6, -372738.78855618887, -5.305549594173777e6, -5481.56
0247393949, 1220.551444320358, 4897.458720261269]
 [-4.86141379061026e6, -364949.0695039659, -5.27420995050537e6, -5448.40625
572443, 1223.075941847303, 4933.757460009457]
 [-4.899619302616308e6, -356333.87416909187, -5.239355781758246e6, -5411.54
0315888837, 1225.8001177308527, 4973.567526515767]
 [-4.939804187099205e6, -347187.64117364015, -5.202132299252132e6, -5372.17
5157344959, 1228.6147371926625, 5015.448621990125]
 [-4.980720938554404e6, -337783.6504205913, -5.163624168182001e6, -5331.458
688215044, 1231.4260409005406, 5058.101630739833]
 ⋮
 [7.109033523333406e6, -748210.6568061096, -837179.6570194897, -974.6743068
149754, -1013.5002623133091, -7306.80857481939]
 [7.102493770741549e6, -754824.4133920531, -884965.7508691706, -1024.440881
8853058, -1008.2368654432507, -7300.7625224396925]
 [7.095549052690655e6, -761477.7997669737, -933250.9953676062, -1074.723125
757611, -1002.8668944382293, -7294.306917660723]
 [7.088191289422171e6, -768166.8647894585, -982013.8106573044, -1125.498415
1746946, -997.3910362163293, -7287.433138057781]
 [7.080412568718434e6, -774887.8893992639, -1.0312340623652438e6, -1176.745
6360545077, -991.8098745700049, -7280.132754525507]
 [7.072205195586522e6, -781637.3103360656, -1.0808925254399125e6, -1228.444
6248447714, -986.1239455917315, -7272.397574765741]
 [7.06356164595553e6, -788411.7357021254, -1.1309710145915258e6, -1280.5763
040478375, -980.3337189473176, -7264.219597537158]
 [7.05447457808568e6, -795207.912410005, -1.1814521611918462e6, -1333.12244
96677469, -974.4396195194423, -7255.591021233254]
 [7.052692640363215e6, -796504.7623123724, -1.1911128864298302e6, -1343.177
819382454, -973.3050714432662, -7253.895584392121]
```



```julia
@btime sol = solve(bvp, MIRK2(), abstol = 1e-13, reltol = 1e-13)
```

```
Error: ArgumentError: dt must be positive
```



```julia
@btime sol = solve(bvp, MIRK3(), abstol = 1e-13, reltol = 1e-13)
```

```
Error: ArgumentError: dt must be positive
```



```julia
@btime sol = solve(bvp, MIRK4(), abstol = 1e-13, reltol = 1e-13)
```

```
Error: ArgumentError: dt must be positive
```



```julia
@btime sol = solve(bvp, MIRK5(), abstol = 1e-13, reltol = 1e-13)
```

```
Error: ArgumentError: dt must be positive
```



```julia
@btime sol = solve(bvp, MIRK6(), abstol = 1e-13, reltol = 1e-13)
```

```
Error: ArgumentError: dt must be positive
```





## Appendix


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonStiffBVP","orbital.jmd")
```

Computer Information:

```
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
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
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonStiffBVP/Project.toml`
  [ded0fc24] BVProblemLibrary v0.1.4
  [6e4b80f9] BenchmarkTools v1.5.0
  [764a87c0] BoundaryValueDiffEq v5.7.1
  [f3b72e0c] DiffEqDevTools v2.44.2
  [54ca160b] ODEInterface v0.5.0
  [1dea7af3] OrdinaryDiffEq v6.74.1
  [91a5bcdd] Plots v1.40.2
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonStiffBVP/Manifest.toml`
  [47edcb42] ADTypes v0.2.7
  [79e6a3ab] Adapt v4.0.4
⌅ [ec485272] ArnoldiMethod v0.2.0
  [4fba245c] ArrayInterface v7.9.0
  [4c555306] ArrayLayouts v1.8.0
  [ded0fc24] BVProblemLibrary v0.1.4
  [aae01518] BandedMatrices v1.6.1
  [6e4b80f9] BenchmarkTools v1.5.0
  [d1d4a3ce] BitFlags v0.1.8
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [764a87c0] BoundaryValueDiffEq v5.7.1
  [2a0fbf3d] CPUSummary v0.2.4
  [49dc2e85] Calculus v0.5.1
  [d360d2e6] ChainRulesCore v1.23.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [944b1d66] CodecZlib v0.7.4
  [35d6a980] ColorSchemes v3.24.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.14.0
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.4.1
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.5
  [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [9a962f9c] DataAPI v1.16.0
  [864edb3b] DataStructures v0.18.18
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v6.148.0
  [f3b72e0c] DiffEqDevTools v2.44.2
  [77a26b50] DiffEqNoiseProcess v5.21.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.11
  [31c24e10] Distributions v0.25.107
  [ffbed154] DocStringExtensions v0.9.3
  [fa6b7ba4] DualNumbers v0.6.8
  [4e289a0a] EnumX v1.0.4
⌃ [f151be2c] EnzymeCore v0.6.5
  [460bff9d] ExceptionUnwrapping v0.1.10
  [d4d017d3] ExponentialUtilities v1.26.1
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [9d29842c] FastAlmostBandedMatrices v0.1.1
  [7034ab61] FastBroadcast v0.2.8
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.2
  [1a297f60] FillArrays v1.9.3
  [6a86dc24] FiniteDiff v2.23.0
  [53c48c17] FixedPointNumbers v0.8.4
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.36
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.1.6
  [28b8d3ca] GR v0.73.3
  [c145ed77] GenericSchur v0.5.4
  [d7ba0133] Git v1.3.1
  [86223c79] Graphs v1.9.0
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.10.5
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.4
  [92d709cd] IrrationalConstants v0.2.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.7
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [ef3ab10e] KLU v0.6.0
  [ba0b0d4f] Krylov v0.9.5
  [b964fa9f] LaTeXStrings v1.3.1
  [23fbe1c1] Latexify v0.16.2
  [10f19ff3] LayoutPointers v0.1.15
  [5078a376] LazyArrays v1.8.3
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.28.0
  [2ab3a3ac] LogExpFunctions v0.3.27
  [e6f89c97] LoggingExtras v1.0.3
  [bdcacae8] LoopVectorization v0.12.166
  [1914dd2f] MacroTools v0.5.13
  [d125e4d3] ManualMemory v0.1.8
  [a3b82374] MatrixFactorizations v2.1.0
  [bb5d69b7] MaybeInplace v0.1.1
  [739be429] MbedTLS v1.1.9
  [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.1.0
  [46d2c3a1] MuladdMacro v0.2.4
  [ffc61752] Mustache v1.0.19
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.2
  [8913a72c] NonlinearSolve v3.8.3
  [54ca160b] ODEInterface v0.5.0
  [6fe1bfb0] OffsetArrays v1.13.0
  [4d8831e6] OpenSSL v1.4.2
  [429524aa] Optim v1.9.3
  [bac558e1] OrderedCollections v1.6.3
  [1dea7af3] OrdinaryDiffEq v6.74.1
  [90014a1f] PDMats v0.11.31
  [65ce6f38] PackageExtensionCompat v1.0.2
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [b98c9c47] Pipe v1.3.0
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.4.1
  [91a5bcdd] Plots v1.40.2
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.10
  [1d0040c9] PolyesterWeave v0.2.1
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.20
  [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [1fd47b50] QuadGK v2.9.4
  [74087812] Random123 v1.7.0
  [e6cf234a] RandomNumbers v1.5.3
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v3.13.0
  [f2c3362d] RecursiveFactorization v0.2.21
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.7.1
  [47965b36] RootedTrees v2.21.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.12
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
  [0bca4576] SciMLBase v2.31.0
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
  [c0aeaf25] SciMLOperators v0.3.8
  [53ae85a6] SciMLStructures v1.1.0
  [6c6a2e73] Scratch v1.2.1
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [727e6d20] SimpleNonlinearSolve v1.6.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.17.0
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.1
  [aedffcd0] Static v0.8.10
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.3
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.2
  [4c63d2b9] StatsFuns v1.3.1
  [7792a7ef] StrideArraysCore v0.5.2
  [69024149] StringEncodings v0.3.7
  [09ab397b] StructArrays v0.6.18
  [2efcf032] SymbolicIndexingInterface v0.3.13
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [62fd8b95] TensorCore v0.1.1
  [8290d209] ThreadingUtilities v0.5.2
  [a759f4b9] TimerOutputs v0.5.23
  [3bb67fe8] TranscodingStreams v0.10.6
  [d5829a12] TriangularSolve v0.1.20
  [410a4b4d] Tricks v0.1.8
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.19.0
  [45397f5d] UnitfulLatexify v1.6.3
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.65
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [6e34b625] Bzip2_jll v1.0.8+1
  [83423d85] Cairo_jll v1.18.0+1
  [2702e6a9] EpollShim_jll v0.0.20230411+0
  [2e619515] Expat_jll v2.5.0+0
⌅ [b22a6f82] FFMPEG_jll v4.4.4+1
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.9+0
  [d2c73de3] GR_jll v0.73.3+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.44.0+1
  [7746bdde] Glib_jll v2.80.0+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2024.0.2+0
  [aacddb02] JpegTurbo_jll v3.0.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.39.3+0
⌅ [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.39.3+1
  [856f044c] MKL_jll v2024.0.0+0
  [c771fb93] ODEInterface_jll v0.0.1+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v3.0.13+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.5.3+1
  [f50d1b31] Rmath_jll v0.4.0+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.21.0+1
  [2381bf8a] Wayland_protocols_jll v1.31.0+0
  [02c8fc9c] XML2_jll v2.12.5+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.6.1+0
  [f67eecfb] Xorg_libICE_jll v1.0.10+1
  [c834827a] Xorg_libSM_jll v1.2.3+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.4+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.5+0
  [3161d3a3] Zstd_jll v1.5.5+0
  [35ca27e7] eudev_jll v3.2.9+0
  [214eeab7] fzf_jll v0.43.0+0
  [1a1c6b14] gperf_jll v3.1.1+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [2db6ffa8] libevdev_jll v1.11.0+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [36db933b] libinput_jll v1.18.0+0
  [b53b4c65] libpng_jll v1.6.43+1
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [009596ad] mtdev_jll v1.1.6+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+1
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
  [e66e0078] CompilerSupportLibraries_jll v1.1.0+0
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
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

