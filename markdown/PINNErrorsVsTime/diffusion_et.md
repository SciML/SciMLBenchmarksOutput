---
author: "Kirill Zubov, Zoe McCarthy, Yingbo Ma, Francesco Calisto, Valerio Pagliarino, Simone Azeglio, Luca Bottero, Emmanuel Luján, Valentin Sulzer, Ashutosh Bharambe, Nand Vinchhi, Kaushik Balakrishnan, Devesh Upadhyay, Chris Rackauckas"
title: "Diffusion PDE Physics-Informed Neural Network (PINN) Loss Function Error vs Time Benchmarks"
---


Adapted from [NeuralPDE: Automating Physics-Informed Neural Networks (PINNs) with Error Approximations](https://arxiv.org/abs/2107.09443).
Uses the [NeuralPDE.jl](https://neuralpde.sciml.ai/dev/) library from the
[SciML Scientific Machine Learning Open Source Organization](https://sciml.ai/)
for the implementation of physics-informed neural networks (PINNs) and other
science-guided AI techniques.

## Setup

```julia
using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
#using Plots
using DelimitedFiles
using QuasiMonteCarlo
```


```julia
function diffusion(strategy, minimizer, maxIters)

    ##  DECLARATIONS
    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(x,t)) - Dxx(u(x,t)) ~ -exp(-t) * (sin(pi * x) - pi^2 * sin(pi * x))

    bcs = [u(x,0) ~ sin(pi*x),
           u(-1,t) ~ 0.,
           u(1,t) ~ 0.]

    domains = [x ∈ IntervalDomain(-1.0,1.0),
               t ∈ IntervalDomain(0.0,1.0)]

    dx = 0.2; dt = 0.1
    xs,ts = [domain.domain.lower:dx/10:domain.domain.upper for (dx,domain) in zip([dx,dt],domains)]

    indvars = [x,t]
    depvars = [u]

    chain = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))

    losses = []
    error = []
    times = []

    dx_err = 0.9

    error_strategy = GridTraining(dx_err)

    initθ = DiffEqFlux.initial_params(chain)
    eltypeθ = eltype(initθ)
    parameterless_type_θ = DiffEqBase.parameterless_type(initθ)

    phi = NeuralPDE.get_phi(chain,parameterless_type_θ)
    derivative = NeuralPDE.get_numeric_derivative()

    _pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                             phi,derivative,chain,initθ,error_strategy)

    bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
    _bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                              phi,derivative,chain,initθ,error_strategy,
                                              bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    train_sets = NeuralPDE.generate_training_sets(domains,dx_err,[eq],bcs,indvars,depvars)
    train_domain_set, train_bound_set = train_sets


    pde_loss_function = NeuralPDE.get_loss_function([_pde_loss_function],
                                          train_domain_set,
                                          error_strategy)

    bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                         train_bound_set,
                                         error_strategy)

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    cb_ = function (p,l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        append!(error, pde_loss_function(p) + bc_loss_function(p))
        #println(length(losses), " Current loss is: ", l, " uniform error is, ",  pde_loss_function(p) + bc_loss_function(p))

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    discretization = PhysicsInformedNN(chain,strategy)

    @named pde_system = PDESystem(eq,bcs,domains,indvars,depvars)
    prob = discretize(pde_system,discretization)


    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training

    res = GalacticOptim.solve(prob, minimizer; cb = cb_, maxiters = maxIters)
    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [x,t]

    u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

    return [error, params, domain, times, u_predict, losses]
end
```

```
diffusion (generic function with 1 method)
```



```julia
maxIters = [(0,0,0,0,0,0),(5000,5000,5000,5000,5000,5000)] #iters for ADAM/LBFGS

strategies = [#NeuralPDE.QuadratureTraining(quadrature_alg = CubaCuhre(), reltol = 1, abstol = 1e-3, maxiters = 10, batch = 10),
              NeuralPDE.QuadratureTraining(quadrature_alg = HCubatureJL(), reltol=1, abstol=1e-5, maxiters=100, batch = 0),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLh(), reltol=1, abstol=1e-5, maxiters=100),
              NeuralPDE.QuadratureTraining(quadrature_alg = CubatureJLp(), reltol=1, abstol=1e-5, maxiters=100),
              NeuralPDE.GridTraining([0.2,0.1]),
              NeuralPDE.StochasticTraining(100),
              NeuralPDE.QuasiRandomTraining(100; sampling_alg = UniformSample(), minibatch = 100)]

strategies_short_name = [#"CubaCuhre",
                        "HCubatureJL",
                        "CubatureJLh",
                        "CubatureJLp",
                        #"CubaVegas",
                        #"CubaSUAVE"]
                        "GridTraining",
                        "StochasticTraining",
                        "QuasiRandomTraining"]

minimizers = [ADAM(0.001),
              #BFGS()]
              LBFGS()]


minimizers_short_name = ["ADAM",
                         "LBFGS"]
                        # "BFGS"]


# Run models
error_res =  Dict()
domains = Dict()
params_res = Dict()  #to use same params for the next run
times = Dict()
prediction = Dict()
losses_res = Dict()
```

```
Dict{Any, Any}()
```





## Solve

```julia
print("Starting run")
## Convergence

for min =1:length(minimizers) # minimizer
      for strat=1:length(strategies) # strategy
            #println(string(strategies_short_name[strat], "  ", minimizers_short_name[min]))
            res = diffusion(strategies[strat], minimizers[min], maxIters[min][strat])
            push!(error_res, string(strat,min)     => res[1])
            push!(params_res, string(strat,min) => res[2])
            push!(domains, string(strat,min)        => res[3])
            push!(times, string(strat,min)        => res[4])
            push!(prediction, string(strat,min)        => res[5])
            push!(losses_res, string(strat,min)        => res[6])

      end
end
```

```
Starting runError: MethodError: no method matching build_loss_function(::Sy
mbolics.Equation, ::Vector{Symbolics.Num}, ::Vector{Symbolics.CallWithMetad
ata{SymbolicUtils.FnType{Tuple, Real}, Base.ImmutableDict{DataType, Any}}},
 ::NeuralPDE.var"#270#272"{DiffEqFlux.FastChain{Tuple{DiffEqFlux.FastDense{
typeof(NNlib.σ), DiffEqFlux.var"#initial_params#90"{Vector{Float32}}}, Diff
EqFlux.FastDense{typeof(NNlib.σ), DiffEqFlux.var"#initial_params#90"{Vector
{Float32}}}, DiffEqFlux.FastDense{typeof(identity), DiffEqFlux.var"#initial
_params#90"{Vector{Float32}}}}}, UnionAll}, ::NeuralPDE.var"#276#277", ::Di
ffEqFlux.FastChain{Tuple{DiffEqFlux.FastDense{typeof(NNlib.σ), DiffEqFlux.v
ar"#initial_params#90"{Vector{Float32}}}, DiffEqFlux.FastDense{typeof(NNlib
.σ), DiffEqFlux.var"#initial_params#90"{Vector{Float32}}}, DiffEqFlux.FastD
ense{typeof(identity), DiffEqFlux.var"#initial_params#90"{Vector{Float32}}}
}}, ::Vector{Float32}, ::NeuralPDE.GridTraining)
Closest candidates are:
  build_loss_function(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::An
y, !Matched::Any; bc_indvars, eq_params, param_estim, default_p) at /cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/packages
/NeuralPDE/nxvU8/src/pinns_pde_solve.jl:559
  build_loss_function(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::An
y, !Matched::Any, !Matched::Any, !Matched::Any, !Matched::Any; bc_indvars, 
integration_indvars, eq_params, param_estim, default_p) at /cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/packages/NeuralPD
E/nxvU8/src/pinns_pde_solve.jl:578
```





## Results

```julia
current_label = string(strategies_short_name[1], " + " , minimizers_short_name[1])
error = Plots.plot(times["11"], error_res["11"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = current_label, xlims = (0,10))#legend = true)#, size=(1200,700))
plot!(error, times["21"], error_res["21"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[2], " + " , minimizers_short_name[1]))
plot!(error, times["31"], error_res["31"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[3], " + " , minimizers_short_name[1]))
plot!(error, times["41"], error_res["41"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[4], " + " , minimizers_short_name[1]))
plot!(error, times["51"], error_res["51"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[5], " + " , minimizers_short_name[1]))
plot!(error, times["61"], error_res["61"], yaxis=:log10, title = string("Diffusion convergence"), ylabel = "log(error)", label = string(strategies_short_name[7], " + " , minimizers_short_name[1]))


plot!(error, times["12"], error_res["12"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[1], " + " , minimizers_short_name[2]))
plot!(error, times["22"], error_res["22"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[2], " + " , minimizers_short_name[2]))
plot!(error, times["32"], error_res["32"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[3], " + " , minimizers_short_name[2]))
plot!(error, times["42"], error_res["42"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[4], " + " , minimizers_short_name[2]))
plot!(error, times["52"], error_res["52"], yaxis=:log10, title = string("Level Set"), ylabel = "log(error)", label = string(strategies_short_name[5], " + " , minimizers_short_name[2]))
plot!(error, times["62"], error_res["62"], yaxis=:log10, title = string("Diffusion convergence ADAM/LBFGS"), ylabel = "log(error)", label = string(strategies_short_name[7], " + " , minimizers_short_name[2]))
```

```
Error: KeyError: key "11" not found
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/PINNErrorsVsTime","diffusion_et.jmd")
```

Computer Information:

```
Julia Version 1.6.4
Commit 35f0c911f4 (2021-11-19 03:54 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
      Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/PINNErrorsVsTime/Project.toml`
  [8a292aeb] Cuba v2.2.0
  [667455a9] Cubature v1.5.1
  [aae7a2af] DiffEqFlux v1.44.0
  [587475ba] Flux v0.12.8
  [a75be94c] GalacticOptim v2.2.0
  [961ee093] ModelingToolkit v6.7.1
  [315f7962] NeuralPDE v4.0.1
  [429524aa] Optim v1.5.0
  [67601950] Quadrature v1.12.0
  [8a4e6c94] QuasiMonteCarlo v0.2.3
  [31c91b34] SciMLBenchmarks v0.1.0
  [8bb1440f] DelimitedFiles
```

And the full manifest:

```
      Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/PINNErrorsVsTime/Manifest.toml`
  [621f4979] AbstractFFTs v1.0.1
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.1
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.40
  [13072b0f] AxisAlgorithms v1.0.1
  [ab4f0b2a] BFloat16s v0.2.0
  [e2ed5e7c] Bijections v0.1.3
  [62783981] BitTwiddlingConvenienceFunctions v0.1.1
  [fa961155] CEnum v0.4.1
  [2a0fbf3d] CPUSummary v0.1.6
  [00ebfdb7] CSTParser v3.3.0
  [052768ef] CUDA v3.5.0
  [7057c7e9] Cassette v0.3.9
  [082447d4] ChainRules v1.14.0
  [d360d2e6] ChainRulesCore v1.11.1
  [9e997f8a] ChangesOfVariables v0.1.1
  [fb6a15b2] CloseOpenIntervals v0.1.4
  [944b1d66] CodecZlib v0.7.0
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.3
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.40.0
  [b152e2b5] CompositeTypes v0.1.2
  [8f4d0f93] Conda v1.5.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.3.0
  [a8cc5b0e] Crayons v4.0.4
  [8a292aeb] Cuba v2.2.0
  [667455a9] Cubature v1.5.1
  [754358af] DEDataArrays v0.2.0
  [9a962f9c] DataAPI v1.9.0
  [82cc6244] DataInterpolations v3.6.1
  [864edb3b] DataStructures v0.18.10
  [e2d170a0] DataValueInterfaces v1.0.0
  [b429d917] DensityInterface v0.4.0
  [2b5f629d] DiffEqBase v6.76.0
  [459566f4] DiffEqCallbacks v2.17.0
  [aae7a2af] DiffEqFlux v1.44.0
  [c894b116] DiffEqJump v7.3.1
  [77a26b50] DiffEqNoiseProcess v5.9.0
  [41bf760c] DiffEqSensitivity v6.60.3
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.5.0
  [b4f34e82] Distances v0.10.6
  [31c24e10] Distributions v0.25.34
  [ced4e74d] DistributionsAD v0.6.34
  [ffbed154] DocStringExtensions v0.8.6
  [5b8099bc] DomainSets v0.5.9
  [7c1d4256] DynamicPolynomials v0.3.21
  [da5c29d0] EllipsisNotation v1.1.1
  [7da242da] Enzyme v0.7.2
  [d4d017d3] ExponentialUtilities v1.10.2
  [e2ba6199] ExprTools v0.1.6
  [7a1cc6ca] FFTW v1.4.5
  [7034ab61] FastBroadcast v0.1.11
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.12.7
  [6a86dc24] FiniteDiff v2.8.1
  [53c48c17] FixedPointNumbers v0.8.4
  [587475ba] Flux v0.12.8
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.23
  [069b7b12] FunctionWrappers v1.1.2
  [d9f16b24] Functors v0.2.7
  [0c68f7d7] GPUArrays v8.1.2
  [61eb1bfa] GPUCompiler v0.13.8
  [a75be94c] GalacticOptim v2.2.0
  [d7ba0133] Git v1.2.1
  [af5da776] GlobalSensitivity v1.2.2
  [86223c79] Graphs v1.4.1
  [19dc6840] HCubature v1.5.0
  [eafb193a] Highlights v0.4.5
  [3e5b6fbb] HostCPUFeatures v0.1.5
  [0e44f5e4] Hwloc v2.0.0
  [7073ff75] IJulia v1.23.2
  [7869d1d1] IRTools v0.4.4
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.2
  [a98d9a8b] Interpolations v0.13.4
  [8197267c] IntervalSets v0.5.3
  [3587e190] InverseFunctions v0.1.2
  [92d709cd] IrrationalConstants v0.1.1
  [42fd0dbc] IterativeSolvers v0.9.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.2
  [98e50ef6] JuliaFormatter v0.18.1
  [e5e0dc1b] Juno v0.8.4
  [5ab0869b] KernelDensity v0.6.3
  [929cbde3] LLVM v4.7.0
  [b964fa9f] LaTeXStrings v1.3.0
  [2ee39098] LabelledArrays v1.6.7
  [23fbe1c1] Latexify v0.15.9
  [a5e1c1ea] LatinHypercubeSampling v1.8.0
  [73f95e8e] LatticeRules v0.0.1
  [10f19ff3] LayoutPointers v0.1.4
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.2
  [093fc24a] LightGraphs v1.3.5
  [d3d80556] LineSearches v7.1.1
  [2ab3a3ac] LogExpFunctions v0.3.5
  [e6f89c97] LoggingExtras v0.4.7
  [bdcacae8] LoopVectorization v0.12.98
  [1914dd2f] MacroTools v0.5.9
  [d125e4d3] ManualMemory v0.1.6
  [739be429] MbedTLS v1.0.3
  [e89f7d12] Media v0.5.0
  [e1d29d7a] Missings v1.0.2
  [961ee093] ModelingToolkit v6.7.1
  [4886b29c] MonteCarloIntegration v0.0.3
  [46d2c3a1] MuladdMacro v0.2.2
  [102ac46a] MultivariatePolynomials v0.3.18
  [ffc61752] Mustache v1.0.12
  [d8a4904e] MutableArithmetics v0.2.22
  [d41bc354] NLSolversBase v7.8.2
  [2774e3e8] NLsolve v4.5.1
  [872c559c] NNlib v0.7.31
  [a00861dc] NNlibCUDA v0.1.10
  [77ba4419] NaNMath v0.3.5
  [315f7962] NeuralPDE v4.0.1
  [8913a72c] NonlinearSolve v0.3.11
  [d8793406] ObjectFile v0.3.7
  [6fe1bfb0] OffsetArrays v1.10.8
  [429524aa] Optim v1.5.0
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v5.68.0
  [90014a1f] PDMats v0.11.5
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.1.2
  [e409e4f3] PoissonRandom v0.4.0
  [f517fe37] Polyester v0.5.4
  [1d0040c9] PolyesterWeave v0.1.2
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.2.0
  [21216c6a] Preferences v1.2.2
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.1
  [1fd47b50] QuadGK v2.4.2
  [67601950] Quadrature v1.12.0
  [8a4e6c94] QuasiMonteCarlo v0.2.3
  [74087812] Random123 v1.4.2
  [e6cf234a] RandomNumbers v1.5.3
  [c84ed2f1] Ratios v0.4.2
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.2.1
  [731186ca] RecursiveArrayTools v2.20.0
  [f2c3362d] RecursiveFactorization v0.2.5
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v0.1.3
  [ae029012] Requires v1.1.3
  [ae5879a3] ResettableStacks v1.1.1
  [37e2e3b7] ReverseDiff v1.10.0
  [79098fc4] Rmath v0.7.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.3
  [3cdde19b] SIMDDualNumbers v0.1.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.28
  [1bc83da4] SafeTestsets v0.0.1
  [0bca4576] SciMLBase v1.19.5
  [31c91b34] SciMLBenchmarks v0.1.0
  [6c6a2e73] Scratch v1.1.0
  [efcf1570] Setfield v0.8.0
  [699a6c99] SimpleTraits v0.9.4
  [ed01d8cd] Sobol v1.5.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.1
  [47a9eef4] SparseDiffTools v1.18.1
  [276daf66] SpecialFunctions v1.8.1
  [860ef19b] StableRNGs v1.0.0
  [aedffcd0] Static v0.4.0
  [90137ffa] StaticArrays v1.2.13
  [82ae8749] StatsAPI v1.1.0
  [2913bbd2] StatsBase v0.33.13
  [4c63d2b9] StatsFuns v0.9.14
  [789caeaf] StochasticDiffEq v6.41.0
  [7792a7ef] StrideArraysCore v0.2.9
  [69024149] StringEncodings v0.3.5
  [53d494c1] StructIO v0.3.0
  [d1185830] SymbolicUtils v0.16.0
  [0c5d862f] Symbolics v3.5.1
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.6.0
  [8ea1fca8] TermInterface v0.1.8
  [5d786b92] TerminalLoggers v0.1.5
  [8290d209] ThreadingUtilities v0.4.6
  [a759f4b9] TimerOutputs v0.5.13
  [0796e94c] Tokenize v0.5.21
  [9f7883ad] Tracker v0.2.16
  [3bb67fe8] TranscodingStreams v0.9.6
  [592b5752] Trapz v2.0.3
  [a2a6695c] TreeViews v0.3.0
  [d5829a12] TriangularSolve v0.1.8
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1986cc42] Unitful v1.9.2
  [3d5dd08c] VectorizationBase v0.21.21
  [81def892] VersionParsing v1.2.1
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.10
  [efce3f68] WoodburyMatrices v0.5.5
  [ddb6d928] YAML v0.4.7
  [c2297ded] ZMQ v1.2.1
  [a5390f91] ZipFile v0.9.4
  [e88e6eb3] Zygote v0.6.32
  [700de1a5] ZygoteRules v0.2.2
  [3bed1096] Cuba_jll v4.2.2+0
  [7bc98958] Cubature_jll v1.0.5+0
  [7cc45869] Enzyme_jll v0.0.22+0
  [2e619515] Expat_jll v2.2.10+0
  [f5851436] FFTW_jll v3.3.10+0
  [78b55507] Gettext_jll v0.20.1+7
  [f8c6e375] Git_jll v2.31.0+0
  [e33a78d0] Hwloc_jll v2.5.0+0
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [dad2f222] LLVMExtra_jll v0.0.13+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [856f044c] MKL_jll v2021.1.1+2
  [458c3c95] OpenSSL_jll v1.1.10+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [02c8fc9c] XML2_jll v2.9.12+0
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [a9144af2] libsodium_jll v1.0.20+0
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
  [4af54fe1] LazyArtifacts
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
  [05823500] OpenLibm_jll
  [efcefdf7] PCRE2_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

