---
author: "Sebastian Micluța-Câmpeanu, Mikhail Vaganov"
title: "Acceleration function benchmarks"
---


Solving the equations of notions for an N-body problem implies solving a (large)
system of differential equations. In `DifferentialEquations.jl` these are represented
through ODE or SDE problems. To build the problem we need a function that
describe the equations. In the case of N-body problems, this function
gives the accelerations for the particles in the system.

Here we will test the performance of several acceleration functions used
in N-body simulations. The systems that will be used are not necessarily realistic
as we are not solving the problem, we just time how fast is an acceleration
function call.

```julia
using BenchmarkTools, NBodySimulator
using NBodySimulator: gather_bodies_initial_coordinates, gather_accelerations_for_potentials,
    gather_simultaneous_acceleration, gather_group_accelerations
using StaticArrays

const SUITE = BenchmarkGroup();

function acceleration(simulation)

    (u0, v0, n) = gather_bodies_initial_coordinates(simulation)

    acceleration_functions = gather_accelerations_for_potentials(simulation)
    simultaneous_acceleration = gather_simultaneous_acceleration(simulation)

    function soode_system!(dv, v, u, p, t)
        @inbounds for i = 1:n
            a = MVector(0.0, 0.0, 0.0)
            for acceleration! in acceleration_functions
                acceleration!(a, u, v, t, i);
            end
            dv[:, i] .= a
        end
        for acceleration! in simultaneous_acceleration
            acceleration!(dv, u, v, t);
        end
    end

    return soode_system!
end
```

```
acceleration (generic function with 1 method)
```





## Gravitational potential

```julia
let SUITE=SUITE
    G = 6.67e-11 # m^3/kg/s^2
    N = 200 # number of bodies/particles
    m = 1.0 # mass of each of them
    v = 10.0 # mean velocity
    L = 20.0 # size of the cell side

    bodies = generate_bodies_in_cell_nodes(N, m, v, L)
    g_parameters = GravitationalParameters(G)
    system = PotentialNBodySystem(bodies, Dict(:gravitational => g_parameters))
    tspan = (0.0, 1.0)
    simulation = NBodySimulation(system, tspan)

    f = acceleration(simulation)
    u0, v0, n = gather_bodies_initial_coordinates(simulation)
    dv = zero(v0)

    b = @benchmarkable $f(dv, $v0, $u0, $g_parameters, 0.) setup=(dv=zero($v0)) evals=1

    SUITE["gravitational"] = b
end
```

```
Benchmark(evals=1, seconds=5.0, samples=10000)
```





## Coulomb potential

```julia
let SUITE=SUITE
    n = 200
    bodies = ChargedParticle[]
    L = 20.0
    m = 1.0
    q = 1.0
    count = 1
    dL = L / (ceil(n^(1 / 3)) + 1)
    for x = dL / 2:dL:L, y = dL / 2:dL:L, z = dL / 2:dL:L
        if count > n
            break
        end
        r = SVector(x, y, z)
        v = SVector(.0, .0, .0)
        body = ChargedParticle(r, v, m, q)
        push!(bodies, body)
        count += 1
    end

    k = 9e9
    τ = 0.01 * dL / sqrt(2 * k * q * q / (dL * m))
    t1 = 0.0
    t2 = 1000 * τ

    potential = ElectrostaticParameters(k, 0.45 * L)
    system = PotentialNBodySystem(bodies, Dict(:electrostatic => potential))
    pbc = CubicPeriodicBoundaryConditions(L)
    simulation = NBodySimulation(system, (t1, t2), pbc)

    f = acceleration(simulation)
    u0, v0, n = gather_bodies_initial_coordinates(simulation)
    dv = zero(v0)

    b = @benchmarkable $f(dv, $v0, $u0, $potential, 0.) setup=(dv=zero($v0)) evals=1

    SUITE["coulomb"] = b
end
```

```
Benchmark(evals=1, seconds=5.0, samples=10000)
```





## Magnetic dipole potential

```julia
let SUITE=SUITE
    n = 200
    bodies = MagneticParticle[]
    L = 20.0
    m = 1.0
    count = 1
    dL = L / (ceil(n^(1 / 3)) + 1)
    for x = dL / 2:dL:L, y = dL / 2:dL:L, z = dL / 2:dL:L
        if count > n
            break
        end
        r = SVector(x, y, z)
        v = SVector(.0, .0, .0)
        mm = rand(SVector{3})
        body = MagneticParticle(r, v, m, mm)
        push!(bodies, body)
        count += 1
    end

    μ_4π = 1e-7
    t1 = 0.0  # s
    t2 = 1.0 # s
    τ = (t2 - t1) / 100

    parameters = MagnetostaticParameters(μ_4π)
    system = PotentialNBodySystem(bodies, Dict(:magnetic => parameters))
    simulation = NBodySimulation(system, (t1, t2))

    f = acceleration(simulation)
    u0, v0, n = gather_bodies_initial_coordinates(simulation)
    dv = zero(v0)

    b = @benchmarkable $f(dv, $v0, $u0, $parameters, 0.) setup=(dv=zero($v0)) evals=1

    SUITE["magnetic_dipole"] = b
end
```

```
Benchmark(evals=1, seconds=5.0, samples=10000)
```





## Lennard Jones potential

```julia
let SUITE=SUITE
    T = 120.0 # K
    T0 = 90.0 # K
    kb = 8.3144598e-3 # kJ/(K*mol)
    ϵ = T * kb
    σ = 0.34 # nm
    ρ = 1374/1.6747# Da/nm^3
    N = 200
    m = 39.95# Da = 216 # number of bodies/particles
    L = (m*N/ρ)^(1/3)#10.229σ
    R = 0.5*L
    v_dev = sqrt(kb * T / m)
    bodies = generate_bodies_in_cell_nodes(N, m, v_dev, L)

    τ = 0.5e-3 # ps or 1e-12 s
    t1 = 0.0
    t2 = 2000τ

    lj_parameters = LennardJonesParameters(ϵ, σ, R)
    lj_system = PotentialNBodySystem(bodies, Dict(:lennard_jones => lj_parameters));

    pbc = CubicPeriodicBoundaryConditions(L)
    simulation = NBodySimulation(lj_system, (t1, t2), pbc, kb)

    f = acceleration(simulation)
    u0, v0, n = gather_bodies_initial_coordinates(simulation)
    dv = zero(v0)

    b = @benchmarkable $f(dv, $v0, $u0, $lj_parameters, 0.) setup=(dv=zero($v0)) evals=1

    SUITE["lennard_jones"] = b
end
```

```
Benchmark(evals=1, seconds=5.0, samples=10000)
```





## WaterSPCFw model

```julia
function acceleration(simulation::NBodySimulation{<:WaterSPCFw})

    (u0, v0, n) = gather_bodies_initial_coordinates(simulation)

    (o_acelerations, h_acelerations) = gather_accelerations_for_potentials(simulation)
    group_accelerations = gather_group_accelerations(simulation)
    simultaneous_acceleration = gather_simultaneous_acceleration(simulation)

    function soode_system!(dv, v, u, p, t)
        @inbounds for i = 1:n
            a = MVector(0.0, 0.0, 0.0)
            for acceleration! in o_acelerations
                acceleration!(a, u, v, t, 3 * (i - 1) + 1);
            end
            dv[:, 3 * (i - 1) + 1]  .= a
        end
        @inbounds for i in 1:n, j in (2, 3)
            a = MVector(0.0, 0.0, 0.0)
            for acceleration! in h_acelerations
                acceleration!(a, u, v, t, 3 * (i - 1) + j);
            end
            dv[:, 3 * (i - 1) + j]   .= a
        end
        @inbounds for i = 1:n
            for acceleration! in group_accelerations
                acceleration!(dv, u, v, t, i);
            end
        end
        for acceleration! in simultaneous_acceleration
            acceleration!(dv, u, v, t);
        end
    end

    return soode_system!
end

let SUITE=SUITE
    T = 370 # K
    T0 = 275 # K
    kb = 8.3144598e-3 # kJ/(K*mol)
    ϵOO = 0.1554253*4.184 # kJ
    σOO = 0.3165492 # nm
    ρ = 997/1.6747# Da/nm^3
    mO = 15.999 # Da
    mH = 1.00794 # Da
    mH2O = mO+2*mH
    N = 200
    L = (mH2O*N/ρ)^(1/3)
    R = 0.9 # ~3*σOO
    Rel = 0.49*L
    v_dev = sqrt(kb * T /mH2O)
    τ = 0.5e-3 # ps
    t1 = 0τ
    t2 = 5τ # ps
    k_bond = 1059.162*4.184*1e2 # kJ/(mol*nm^2)
    k_angle = 75.90*4.184 # kJ/(mol*rad^2)
    rOH = 0.1012 # nm
    ∠HOH = 113.24*pi/180 # rad
    qH = 0.41
    qO = -0.82
    k = 138.935458 #
    bodies = generate_bodies_in_cell_nodes(N, mH2O, v_dev, L)
    jl_parameters = LennardJonesParameters(ϵOO, σOO, R)
    e_parameters = ElectrostaticParameters(k, Rel)
    spc_parameters = SPCFwParameters(rOH, ∠HOH, k_bond, k_angle)
    pbc = CubicPeriodicBoundaryConditions(L)
    water = WaterSPCFw(bodies, mH, mO, qH, qO,  jl_parameters, e_parameters, spc_parameters);
    simulation = NBodySimulation(water, (t1, t2), pbc, kb);

    f = acceleration(simulation)
    u0, v0, n = gather_bodies_initial_coordinates(simulation)
    dv = zero(v0)

    b = @benchmarkable $f(dv, $v0, $u0, $spc_parameters, 0.) setup=(dv=zero($v0)) evals=1

    SUITE["water_spcfw"] = b
end
```

```
Benchmark(evals=1, seconds=5.0, samples=10000)
```





Here are the results of the benchmarks
```julia
r = run(SUITE)

minimum(r)
```

```
5-element BenchmarkTools.BenchmarkGroup:
  tags: []
  "gravitational" => TrialEstimate(5.784 ms)
  "coulomb" => TrialEstimate(599.765 μs)
  "lennard_jones" => TrialEstimate(493.896 μs)
  "water_spcfw" => TrialEstimate(6.916 ms)
  "magnetic_dipole" => TrialEstimate(23.494 ms)
```




and
```julia
memory(r)
```

```
5-element BenchmarkTools.BenchmarkGroup:
  tags: []
  "gravitational" => 7670400
  "coulomb" => 9600
  "lennard_jones" => 9600
  "water_spcfw" => 105712
  "magnetic_dipole" => 25564800
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NBodySimulator","acceleration_functions.jmd")
```

Computer Information:

```
Julia Version 1.7.3
Commit 742b9abb4d (2022-05-06 12:58 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, znver2)
Environment:
  JULIA_CPU_THREADS = 128
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
      Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NBodySimulator/Project.toml`
  [6e4b80f9] BenchmarkTools v1.3.1
  [a93c6f00] DataFrames v1.3.4
  [0e6f8da7] NBodySimulator v1.8.1
  [1dea7af3] OrdinaryDiffEq v6.19.1
  [91a5bcdd] Plots v1.31.4
  [33c8b6b6] ProgressLogging v0.1.4
  [31c91b34] SciMLBenchmarks v0.1.0
  [90137ffa] StaticArrays v1.5.2
  [f3b207a7] StatsPlots v0.15.0
```

And the full manifest:

```
      Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NBodySimulator/Manifest.toml`
  [621f4979] AbstractFFTs v1.2.1
  [79e6a3ab] Adapt v3.3.3
  [ec485272] ArnoldiMethod v0.2.0
  [7d9fca2a] Arpack v0.5.3
  [4fba245c] ArrayInterface v6.0.21
  [30b0a656] ArrayInterfaceCore v0.1.15
  [6ba088a2] ArrayInterfaceGPUArrays v0.2.1
  [015c0d05] ArrayInterfaceOffsetArrays v0.1.6
  [b0d46f97] ArrayInterfaceStaticArrays v0.1.4
  [dd5226c6] ArrayInterfaceStaticArraysCore v0.1.0
  [13072b0f] AxisAlgorithms v1.0.1
  [6e4b80f9] BenchmarkTools v1.3.1
  [62783981] BitTwiddlingConvenienceFunctions v0.1.4
  [2a0fbf3d] CPUSummary v0.1.25
  [49dc2e85] Calculus v0.5.1
  [d360d2e6] ChainRulesCore v1.15.3
  [9e997f8a] ChangesOfVariables v0.1.4
  [fb6a15b2] CloseOpenIntervals v0.1.10
  [aaaa29a8] Clustering v0.14.2
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.19.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.9.9
  [5ae59095] Colors v0.12.8
  [38540f10] CommonSolve v0.2.1
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.45.0
  [8f4d0f93] Conda v1.7.0
  [187b0558] ConstructionBase v1.4.0
  [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.10.0
  [a93c6f00] DataFrames v1.3.4
  [864edb3b] DataStructures v0.18.13
  [e2d170a0] DataValueInterfaces v1.0.0
  [e7dc6d0d] DataValues v0.4.13
  [b429d917] DensityInterface v0.4.0
  [2b5f629d] DiffEqBase v6.94.4
  [459566f4] DiffEqCallbacks v2.23.1
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.11.0
  [b4f34e82] Distances v0.10.7
  [31c24e10] Distributions v0.25.66
  [ffbed154] DocStringExtensions v0.8.6
  [fa6b7ba4] DualNumbers v0.6.8
  [d4d017d3] ExponentialUtilities v1.18.0
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.5.0
  [7034ab61] FastBroadcast v0.2.1
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v1.1.0
  [5789e2e9] FileIO v1.14.0
  [1a297f60] FillArrays v0.13.2
  [6a86dc24] FiniteDiff v2.14.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.30
  [069b7b12] FunctionWrappers v1.1.2
  [46192b85] GPUArraysCore v0.1.1
  [28b8d3ca] GR v0.66.0
  [c145ed77] GenericSchur v0.5.3
  [5c1252a2] GeometryBasics v0.4.2
  [d7ba0133] Git v1.2.1
  [86223c79] Graphs v1.7.1
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.2.0
  [eafb193a] Highlights v0.4.5
  [3e5b6fbb] HostCPUFeatures v0.1.8
  [34004b35] HypergeometricFunctions v0.3.11
  [7073ff75] IJulia v1.23.3
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.1
  [a98d9a8b] Interpolations v0.13.6
  [3587e190] InverseFunctions v0.1.7
  [41ab1584] InvertedIndices v1.1.0
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [42fd0dbc] IterativeSolvers v0.9.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.3
  [ef3ab10e] KLU v0.3.0
  [5ab0869b] KernelDensity v0.6.4
  [ba0b0d4f] Krylov v0.8.2
  [0b1a1467] KrylovKit v0.5.4
  [b964fa9f] LaTeXStrings v1.3.0
  [23fbe1c1] Latexify v0.15.16
  [10f19ff3] LayoutPointers v0.1.10
  [d3d80556] LineSearches v7.1.1
  [7ed4a6bd] LinearSolve v1.23.0
  [2ab3a3ac] LogExpFunctions v0.3.16
  [e6f89c97] LoggingExtras v0.4.9
  [bdcacae8] LoopVectorization v0.12.120
  [1914dd2f] MacroTools v0.5.9
  [d125e4d3] ManualMemory v0.1.8
  [739be429] MbedTLS v1.1.1
  [442fdcdd] Measures v0.3.1
  [e1d29d7a] Missings v1.0.2
  [46d2c3a1] MuladdMacro v0.2.2
  [6f286f6a] MultivariateStats v0.9.1
  [ffc61752] Mustache v1.0.14
  [0e6f8da7] NBodySimulator v1.8.1
  [d41bc354] NLSolversBase v7.8.2
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v0.3.7
  [b8a86587] NearestNeighbors v0.4.11
  [8913a72c] NonlinearSolve v0.3.21
  [510215fc] Observables v0.5.1
  [6fe1bfb0] OffsetArrays v1.12.7
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v6.19.1
  [90014a1f] PDMats v0.11.16
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.3.2
  [ccf2f8ad] PlotThemes v3.0.0
  [995b91a9] PlotUtils v1.3.0
  [91a5bcdd] Plots v1.31.4
  [f517fe37] Polyester v0.6.14
  [1d0040c9] PolyesterWeave v0.1.7
  [2dfb63ee] PooledArrays v1.4.2
  [d236fae5] PreallocationTools v0.4.0
  [21216c6a] Preferences v1.3.0
  [08abe8d2] PrettyTables v1.3.1
  [33c8b6b6] ProgressLogging v0.1.4
  [1fd47b50] QuadGK v2.4.2
  [c84ed2f1] Ratios v0.4.3
  [3cdcf5f2] RecipesBase v1.2.1
  [01d81517] RecipesPipeline v0.6.2
  [731186ca] RecursiveArrayTools v2.31.2
  [f2c3362d] RecursiveFactorization v0.2.11
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.3
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.0
  [3cdde19b] SIMDDualNumbers v0.1.1
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.33
  [0bca4576] SciMLBase v1.44.1
  [31c91b34] SciMLBenchmarks v0.1.0
  [6c6a2e73] Scratch v1.1.1
  [91c51154] SentinelArrays v1.3.13
  [efcf1570] Setfield v1.0.0
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [699a6c99] SimpleTraits v0.9.4
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.1
  [47a9eef4] SparseDiffTools v1.24.0
  [276daf66] SpecialFunctions v2.1.7
  [aedffcd0] Static v0.7.6
  [90137ffa] StaticArrays v1.5.2
  [1e83bf80] StaticArraysCore v1.0.1
  [82ae8749] StatsAPI v1.2.2
  [2913bbd2] StatsBase v0.33.20
  [4c63d2b9] StatsFuns v1.0.1
  [f3b207a7] StatsPlots v0.15.0
  [7792a7ef] StrideArraysCore v0.3.15
  [69024149] StringEncodings v0.3.5
  [09ab397b] StructArrays v0.6.11
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.7.0
  [62fd8b95] TensorCore v0.1.1
  [8290d209] ThreadingUtilities v0.5.0
  [3bb67fe8] TranscodingStreams v0.9.6
  [d5829a12] TriangularSolve v0.1.12
  [5c2747f8] URIs v1.4.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.1.2
  [3d5dd08c] VectorizationBase v0.21.43
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.10
  [cc8bc4a8] Widgets v0.6.6
  [efce3f68] WoodburyMatrices v0.5.5
  [ddb6d928] YAML v0.4.7
  [c2297ded] ZMQ v1.2.1
  [700de1a5] ZygoteRules v0.2.2
  [68821587] Arpack_jll v3.5.1+1
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [5ae413db] EarCut_jll v2.2.3+0
  [2e619515] Expat_jll v2.4.8+0
  [b22a6f82] FFMPEG_jll v4.4.2+0
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.10.4+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.6+0
  [d2c73de3] GR_jll v0.66.0+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.34.1+0
  [7746bdde] Glib_jll v2.68.3+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.1.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dd4b983a] LZO_jll v2.10.1+0
  [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.4.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2022.0.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v1.1.17+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.3+1
  [f50d1b31] Rmath_jll v0.3.0+0
  [a2964d1f] Wayland_jll v1.19.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.9.14+0
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
  [3161d3a3] Zstd_jll v1.5.2+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v0.9.1+5
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.8.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.0
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v0.5.2+0
  [deac9b47] LibCURL_jll v7.81.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.0+0
  [14a3606d] MozillaCACerts_jll v2022.2.1
  [4536629a] OpenBLAS_jll v0.3.20+0
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.40.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+0
  [83775a58] Zlib_jll v1.2.12+3
  [8e850b90] libblastrampoline_jll v5.1.0+0
  [8e850ede] nghttp2_jll v1.41.0+1
  [3f19e933] p7zip_jll v17.4.0+0
```

