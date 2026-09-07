---
author: "Qingyu Qu"
title: "Linear BVP Benchmarks"
---


This benchmark compares the runtime and error of BVP solvers, including MIRK solvers, FIRK solvers, Shooting solvers and FORTRAN BVP solvers on linear boundary value problems.
The testing BVPs are a set of standard BVP test problems as described [here](https://archimede.uniba.it/%7Ebvpsolvers/testsetbvpsolvers/?page_id=29).
The problems are implemented in [BVProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/master/lib/BVProblemLibrary/src/BVProblemLibrary.jl), where you can find the problem function declarations.
For each problem, we test the following solvers:

  - BoundaryValueDiffEq.jl's MIRK methods(including `MIRK4`, `MIRK5`, `MIRK6`).
  - BoundaryValueDiffEq.jl's Shooting methods(including `Shooting`, `MultipleShooting`).
  - BoundaryValueDiffEq.jl's FIRK methods(including `RadauIIa3`, `RadauIIa5`, `RadauIIa7`, `LobattoIIIa4`, `LobattoIIIa5`, `LobattoIIIb4`, `LobattoIIIb5`, `LobattoIIIc4`, `LobattoIIIc5`).
  - SimpleBoundaryValueDiffEq.jl's MIRK methods(including `SimpleMIRK4`, `SimpleMIRK5`, `SimpleMIRK6`).
  - FORTRAN BVP solvers from ODEInterface.jl(including `BVPM2` and `COLNEW`).

# Setup

Fetch required packages.

```julia
using BoundaryValueDiffEq, SimpleBoundaryValueDiffEq, OrdinaryDiffEq, ODEInterface,
      DiffEqDevTools, BenchmarkTools,
      BVProblemLibrary, CairoMakie, NonlinearSolveFirstOrder
```




Set up the benchmarked solvers.

```julia
solvers_all = [
    (; pkg = :boundaryvaluediffeq, type = :mirk, name = "MIRK4",
        solver = Dict(:alg => MIRK4(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :mirk, name = "MIRK5",
        solver = Dict(:alg => MIRK5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :mirk, name = "MIRK6",
        solver = Dict(:alg => MIRK6(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "RadauIIa3",
        solver = Dict(:alg => RadauIIa3(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "RadauIIa5",
        solver = Dict(:alg => RadauIIa5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "RadauIIa7",
        solver = Dict(:alg => RadauIIa7(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIa4",
        solver = Dict(:alg => LobattoIIIa4(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIa5",
        solver = Dict(:alg => LobattoIIIa5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIb4",
        solver = Dict(:alg => LobattoIIIb4(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIb5",
        solver = Dict(:alg => LobattoIIIb5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIc4",
        solver = Dict(:alg => LobattoIIIc4(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :firk, name = "LobattoIIIc5",
        solver = Dict(:alg => LobattoIIIc5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,
        type = :shooting,
        name = "Single Shooting",
        solver = Dict(:alg => Shooting(Tsit5(), NewtonRaphson()), :dts=>1.0 ./
                                                                        10.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq, type = :shooting, name = "Multiple Shooting",
        solver = Dict(:alg => MultipleShooting(10, Tsit5()), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :simpleboundaryvaluediffeq, type = :simplemirk, name = "SimpleMIRK4",
        solver = Dict(:alg => SimpleMIRK4(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :simpleboundaryvaluediffeq, type = :simplemirk, name = "SimpleMIRK5",
        solver = Dict(:alg => SimpleMIRK5(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :simpleboundaryvaluediffeq, type = :simplemirk, name = "SimpleMIRK6",
        solver = Dict(:alg => SimpleMIRK6(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :wrapper, type = :general, name = "BVPM2",
        solver = Dict(:alg => BVPM2(), :dts=>1.0 ./ 10.0 .^ (1:4))),
    (; pkg = :wrapper, type = :general, name = "COLNEW",
        solver = Dict(:alg => COLNEW(), :dts=>1.0 ./ 10.0 .^ (1:4)))
];

solver_tracker = [];
wp_general_tracker = [];
```




Sets tolerances.

```julia
abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3);
```




Prepares helper function for benchmarking a specific problem.

```julia
function benchmark(prob)
    sol = solve(prob, Shooting(Vern7()), abstol = 1e-14, reltol = 1e-14)
    testsol = TestSolution(sol)
    wps = WorkPrecisionSet(prob, abstols, reltols, getfield.(solvers_all, :solver);
        names = getfield.(solvers_all, :name), appxsol = testsol, maxiters = Int(1e4))
    push!(wp_general_tracker, wps)
    return wps
end

function plot_wpd(wp_set)
    fig = begin
        LINESTYLES = Dict(:boundaryvaluediffeq => :solid, :simpleboundaryvaluediffeq => :dash, :wrapper => :dot)
        ASPECT_RATIO = 0.7
        WIDTH = 1200
        HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
        STROKEWIDTH = 2.5

        colors = cgrad(:seaborn_bright, length(solvers_all); categorical = true)
        cycle = Cycle([:marker], covary = true)
        plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

        with_theme(plot_theme) do
            fig = Figure(; size = (WIDTH, HEIGHT))
            ax = Axis(fig[1, 1], ylabel = L"Time $\mathbf{(s)}$",
                xlabelsize = 22, ylabelsize = 22,
                xlabel = L"Error: $\mathbf{||f(u^\ast)||_\infty}$",
                xscale = log10, yscale = log10, xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                xticklabelsize = 20, yticklabelsize = 20)

            idxs = sortperm(median.(getfield.(wp_set.wps, :times)))

            ls, scs = [], []

            for (i, (wp, solver)) in enumerate(zip(wp_set.wps[idxs], solvers_all[idxs]))
                (; name, times, errors) = wp
                errors = [err.l∞ for err in errors]
                l = lines!(
                    ax, errors, times; linestyle = LINESTYLES[solver.pkg], label = name,
                    linewidth = 5, color = colors[i])
                sc = scatter!(
                    ax, errors, times; label = name, markersize = 16, strokewidth = 2,
                    color = colors[i])
                push!(ls, l)
                push!(scs, sc)
            end

            xlims!(ax; high = 1)
            ylims!(ax; low = 5e-6)

            axislegend(ax, [[l, sc] for (l, sc) in zip(ls, scs)],
                [solver.name for solver in solvers_all[idxs]], "BVP Solvers";
                framevisible = true, framewidth = STROKEWIDTH, position = :rb,
                titlesize = 20, labelsize = 16, patchsize = (40.0f0, 20.0f0))

            fig[0, :] = Label(fig, "Linear BVP Benchmark",
                fontsize = 24, tellwidth = false, font = :bold)
            fig
        end
    end
end
```

```
plot_wpd (generic function with 1 method)
```





# Benchmarks

We here run benchmarks for each of the 18 test problems.

### Linear BVP 1

```julia
prob_1 = BVProblemLibrary.prob_bvp_linear_1
wps = benchmark(prob_1)
plot_wpd(wps)
```

![](figures/linear_wpd_5_1.png)



### Linear BVP 2

```julia
prob_2 = BVProblemLibrary.prob_bvp_linear_2
wps = benchmark(prob_2)
plot_wpd(wps)
```

![](figures/linear_wpd_6_1.png)



### Linear BVP 3

```julia
prob_3 = BVProblemLibrary.prob_bvp_linear_3
wps = benchmark(prob_3)
plot_wpd(wps)
```

![](figures/linear_wpd_7_1.png)



### Linear BVP 4

```julia
prob_4 = BVProblemLibrary.prob_bvp_linear_4
wps = benchmark(prob_4)
plot_wpd(wps)
```

![](figures/linear_wpd_8_1.png)



### Linear BVP 5

```julia
prob_5 = BVProblemLibrary.prob_bvp_linear_5
wps = benchmark(prob_5)
plot_wpd(wps)
```

![](figures/linear_wpd_9_1.png)



### Linear BVP 6

```julia
prob_6 = BVProblemLibrary.prob_bvp_linear_6
wps = benchmark(prob_6)
plot_wpd(wps)
```

![](figures/linear_wpd_10_1.png)



### Linear BVP 7

```julia
prob_7 = BVProblemLibrary.prob_bvp_linear_7
wps = benchmark(prob_7)
plot_wpd(wps)
```

![](figures/linear_wpd_11_1.png)



### Linear BVP 8

```julia
prob_8 = BVProblemLibrary.prob_bvp_linear_8
wps = benchmark(prob_8)
plot_wpd(wps)
```

![](figures/linear_wpd_12_1.png)



### Linear BVP 9

```julia
prob_9 = BVProblemLibrary.prob_bvp_linear_9
wps = benchmark(prob_9)
plot_wpd(wps)
```

![](figures/linear_wpd_13_1.png)



### Linear BVP 10

```julia
prob_10 = BVProblemLibrary.prob_bvp_linear_10
wps = benchmark(prob_10)
plot_wpd(wps)
```

![](figures/linear_wpd_14_1.png)



### Linear BVP 11

```julia
prob_11 = BVProblemLibrary.prob_bvp_linear_11
wps = benchmark(prob_11)
plot_wpd(wps)
```

![](figures/linear_wpd_15_1.png)



### Linear BVP 12

```julia
prob_12 = BVProblemLibrary.prob_bvp_linear_12
wps = benchmark(prob_12)
plot_wpd(wps)
```

![](figures/linear_wpd_16_1.png)



### Linear BVP 13

```julia
prob_13 = BVProblemLibrary.prob_bvp_linear_13
wps = benchmark(prob_13)
plot_wpd(wps)
```

![](figures/linear_wpd_17_1.png)



### Linear BVP 14

```julia
prob_14 = BVProblemLibrary.prob_bvp_linear_14
wps = benchmark(prob_14)
plot_wpd(wps)
```

![](figures/linear_wpd_18_1.png)



### Linear BVP 15

```julia
prob_15 = BVProblemLibrary.prob_bvp_linear_15
wps = benchmark(prob_15)
plot_wpd(wps)
```

![](figures/linear_wpd_19_1.png)



### Linear BVP 16

```julia
prob_16 = BVProblemLibrary.prob_bvp_linear_16
wps = benchmark(prob_16)
plot_wpd(wps)
```

![](figures/linear_wpd_20_1.png)



### Linear BVP 17

```julia
prob_17 = BVProblemLibrary.prob_bvp_linear_17
wps = benchmark(prob_17)
plot_wpd(wps)
```

![](figures/linear_wpd_21_1.png)



### Linear BVP 18

```julia
prob_18 = BVProblemLibrary.prob_bvp_linear_18
wps = benchmark(prob_18)
plot_wpd(wps)
```

![](figures/linear_wpd_22_1.png)



# Summary of General Solvers Performance on All Problems

```julia
fig = begin
    LINESTYLES = Dict(:boundaryvaluediffeq => :solid, :simpleboundaryvaluediffeq => :dash, :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1800
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:seaborn_bright, length(solvers_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    with_theme(plot_theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        axs = Matrix{Any}(undef, 5, 5)

        ls = []
        scs = []
        labels = []
        solver_times = []

        for i in 1:4, j in 1:5

            idx = 5 * (i - 1) + j

            idx > length(wp_general_tracker) && break

            wp = wp_general_tracker[idx]

            ax = Axis(fig[i, j],
                xscale = log10, yscale = log10,
                xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                title = "No. $(idx) Linear BVP benchmarking", titlegap = 10,
                xticklabelsize = 16, yticklabelsize = 16)

            for wpᵢ in wp.wps
                idx = findfirst(s -> s.name == wpᵢ.name, solvers_all)
                errs = getindex.(wpᵢ.errors, :l∞)
                times = wpᵢ.times

                l = lines!(ax, errs, times; color = colors[idx], linewidth = 3,
                    linestyle = LINESTYLES[solvers_all[idx].pkg], alpha = 0.8,
                    label = wpᵢ.name)
                sc = scatter!(ax, errs, times; color = colors[idx], markersize = 16,
                    strokewidth = 2, marker = Cycled(idx), alpha = 0.8, label = wpᵢ.name)

                if wpᵢ.name ∉ labels
                    push!(ls, l)
                    push!(scs, sc)
                    push!(labels, wpᵢ.name)
                end
            end
        end

        fig[0, :] = Label(fig, "Work-Precision Diagram for 18 Test Problems",
            fontsize = 24, tellwidth = false, font = :bold)

        fig[:, 0] = Label(fig, "Time (s)", fontsize = 20, tellheight = false, font = :bold,
            rotation = π / 2)
        fig[end + 1, :] = Label(fig,
            L"Error: $\mathbf{||f(u^\ast)||_\infty}$",
            fontsize = 20, tellwidth = false, font = :bold)

        Legend(fig[4, 4:5], [[l, sc] for (l, sc) in zip(ls, scs)],
            labels, "BVP Solvers";
            framevisible = true, framewidth = STROKEWIDTH, orientation = :horizontal,
            titlesize = 20, nbanks = 9, labelsize = 20, halign = :center,
            tellheight = false, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        return fig
    end
end
```

![](figures/linear_wpd_23_1.png)

```julia
save("summary_wp_18test_problems.svg", fig)
```

```
CairoMakie.Screen{SVG}
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonStiffBVP","linear_wpd.jmd")
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
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

Package Information:

```
Status `~/sandbox/tmp_20260825_180339_53321/nonstiffbvp-current/benchmarks/NonStiffBVP/Project.toml`
  [ded0fc24] BVProblemLibrary v0.1.13
  [6e4b80f9] BenchmarkTools v1.8.0
  [764a87c0] BoundaryValueDiffEq v5.25.0
  [13f3f980] CairoMakie v0.15.14
⌃ [f3b72e0c] DiffEqDevTools v3.6.1
  [5959db7a] NonlinearSolveFirstOrder v2.5.0
  [54ca160b] ODEInterface v0.5.2
  [1dea7af3] OrdinaryDiffEq v7.8.1
  [1344f307] OrdinaryDiffEqLowOrderRK v2.2.5
  [91a5bcdd] Plots v1.41.7
  [0bca4576] SciMLBase v3.50.2
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [be0294bd] SimpleBoundaryValueDiffEq v1.6.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

And the full manifest:

```
Status `~/sandbox/tmp_20260825_180339_53321/nonstiffbvp-current/benchmarks/NonStiffBVP/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
⌃ [14f7f29c] AMD v0.5.3
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [35492f91] AdaptivePredicates v1.2.0
  [66dad0bd] AliasTables v1.1.3
  [a95523ee] AlmostBlockDiagonals v0.1.10
  [27a7e980] Animations v0.4.2
  [4fba245c] ArrayInterface v7.30.1
  [4c555306] ArrayLayouts v1.12.2
  [67c07d97] Automa v1.2.0
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.8
  [ded0fc24] BVProblemLibrary v0.1.13
  [aae01518] BandedMatrices v1.12.0
  [18cc8868] BaseDirs v1.4.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [b2a6c25c] BinaryHeaps v1.1.0
  [764a87c0] BoundaryValueDiffEq v5.25.0
  [7227322d] BoundaryValueDiffEqAscher v1.16.1
  [56b672f2] BoundaryValueDiffEqCore v2.8.3
  [85d9eb09] BoundaryValueDiffEqFIRK v1.19.1
  [1a22d4ce] BoundaryValueDiffEqMIRK v1.18.1
  [9255f1d6] BoundaryValueDiffEqMIRKN v1.17.1
  [ed55bfe0] BoundaryValueDiffEqShooting v1.18.2
  [70df07ce] BracketingNonlinearSolve v1.12.6
  [fa961155] CEnum v0.5.0
  [96374032] CRlibm v1.0.2
  [159f3aea] Cairo v1.1.1
  [13f3f980] CairoMakie v0.15.14
  [d360d2e6] ChainRulesCore v1.26.1
  [6b39b394] CodecZstd v0.8.7
  [a2cac450] ColorBrewer v0.4.2
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
  [861a8166] Combinatorics v1.1.0
  [38540f10] CommonSolve v0.2.14
  [bbf7d656] CommonSubexpressions v0.3.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [95dc2771] ComputePipeline v0.1.8
  [2569d6c7] ConcreteStructs v0.2.8
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [b7a15901] CoreMath v0.1.0
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [927a84f5] DelaunayTriangulation v1.6.6
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v7.20.0
⌃ [f3b72e0c] DiffEqDevTools v3.6.1
  [77a26b50] DiffEqNoiseProcess v5.36.2
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [31c24e10] Distributions v0.25.131
  [ffbed154] DocStringExtensions v0.9.5
  [4e289a0a] EnumX v1.0.7
  [f151be2c] EnzymeCore v0.8.21
  [429591f6] ExactPredicates v2.2.9
  [e2ba6199] ExprTools v0.1.11
  [c87230d0] FFMPEG v0.4.5
  [b86e33f2] FFTA v0.3.1
  [9d29842c] FastAlmostBandedMatrices v0.1.12
  [7034ab61] FastBroadcast v1.4.0
  [9aa1b823] FastClosures v0.3.2
  [a4df4552] FastPower v1.5.0
  [5789e2e9] FileIO v1.20.0
  [8fc22ac5] FilePaths v0.9.0
  [48062228] FilePathsBase v0.9.24
  [1a297f60] FillArrays v1.17.0
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.4.5
  [b38be410] FreeType v4.1.1
  [663a7486] FreeTypeAbstraction v0.10.8
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [46192b85] GPUArraysCore v0.2.0
  [28b8d3ca] GR v0.73.27
  [a0844989] Gamma v1.2.0
  [5c1252a2] GeometryBasics v0.5.12
  [d7ba0133] Git v1.5.0
  [a2bd30eb] Graphics v1.1.3
  [3955a311] GridLayoutBase v0.11.3
  [19dc6840] HCubature v1.8.0
⌅ [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.30
  [7073ff75] IJulia v1.34.4
  [2803e5a7] ImageAxes v0.6.12
  [c817782e] ImageBase v0.1.7
  [a09fc81d] ImageCore v0.10.5
  [82e4d734] ImageIO v0.6.10
  [bc367c6b] ImageMetadata v0.9.10
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.5
  [18e54dd8] IntegerMathUtils v0.1.4
  [de52edbc] Integrals v5.5.0
  [a98d9a8b] Interpolations v0.16.3
  [d1acc4aa] IntervalArithmetic v1.0.11
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.6
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [b835a17e] JpegTurbo v0.1.6
  [5ab0869b] KernelDensity v0.6.12
  [ba0b0d4f] Krylov v0.10.9
  [2faa5264] LHLFactorization v2.2.2
  [b964fa9f] LaTeXStrings v1.4.1
  [23fbe1c1] Latexify v0.16.12
  [73f95e8e] LatticeRules v0.0.1
  [5078a376] LazyArrays v2.12.0
  [8cdb02fc] LazyModules v0.3.1
  [87fe0de2] LineSearch v0.1.16
  [7ed4a6bd] LinearSolve v5.15.1
  [2ab3a3ac] LogExpFunctions v1.0.1
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [ee78f7c6] Makie v0.24.14
  [dbb5928d] MappedArrays v0.4.3
  [0a4f8689] MathTeXEngine v0.6.9
  [a3b82374] MatrixFactorizations v3.1.3
  [bb5d69b7] MaybeInplace v0.1.8
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [4886b29c] MonteCarloIntegration v0.2.0
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.7
  [ffc61752] Mustache v1.0.21
  [77ba4419] NaNMath v1.1.4
  [f09324ee] Netpbm v1.1.1
  [8913a72c] NonlinearSolve v4.29.1
  [be0214bd] NonlinearSolveBase v2.49.2
  [5959db7a] NonlinearSolveFirstOrder v2.5.0
  [9a2c21bd] NonlinearSolveQuasiNewton v1.15.3
  [26075421] NonlinearSolveSpectralMethods v1.8.1
  [54ca160b] ODEInterface v0.5.2
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.17.0
  [52e1d378] OpenEXR v0.3.3
  [bca83a33] OptimizationBase v5.5.3
  [bac558e1] OrderedCollections v2.0.1
  [1dea7af3] OrdinaryDiffEq v7.8.1
  [6ad6398a] OrdinaryDiffEqBDF v2.4.6
  [bbf590c4] OrdinaryDiffEqCore v4.16.0
  [50262376] OrdinaryDiffEqDefault v2.6.0
  [4302a76b] OrdinaryDiffEqDifferentiation v3.11.4
  [1344f307] OrdinaryDiffEqLowOrderRK v2.2.5
  [127b3ac7] OrdinaryDiffEqNonlinearSolve v2.9.4
  [43230ef6] OrdinaryDiffEqRosenbrock v2.7.1
  [b4bd8bb3] OrdinaryDiffEqRosenbrockTableaus v2.4.2
  [2d112036] OrdinaryDiffEqSDIRK v2.9.2
  [b1df2697] OrdinaryDiffEqTsit5 v2.1.4
  [79d7bb75] OrdinaryDiffEqVerner v2.4.1
  [90014a1f] PDMats v0.11.41
  [f57f5aa1] PNGFiles v0.4.5
  [19eb6ba3] Packing v0.5.1
  [5432bcbf] PaddedViews v0.5.12
⌅ [69de0a69] Parsers v2.8.7
  [eebad327] PkgVersion v0.3.3
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.7
  [e409e4f3] PoissonRandom v0.4.13
  [647866c9] PolygonOps v0.1.2
  [d236fae5] PreallocationTools v1.7.1
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [08abe8d2] PrettyTables v3.4.8
  [27ebfcd6] Primes v0.5.7
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.4.1
  [4b34888f] QOI v1.0.2
  [1fd47b50] QuadGK v2.11.3
⌅ [8a4e6c94] QuasiMonteCarlo v0.3.11
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v4.5.1
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.4.0
  [9fe22ead] RespecializeParams v1.3.0
  [79098fc4] Rmath v0.9.0
  [47965b36] RootedTrees v2.27.0
  [f2b01f46] Roots v3.0.8
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.25
  [fdea26ae] SIMD v3.7.2
  [0bca4576] SciMLBase v3.50.2
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.18
  [a6db7da4] SciMLLogging v2.1.0
  [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
  [53ae85a6] SciMLStructures v1.10.5
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
  [65257c39] ShaderAbstractions v0.5.0
  [992d4aef] Showoff v1.1.1
  [73760f76] SignedDistanceFields v0.4.1
  [be0294bd] SimpleBoundaryValueDiffEq v1.6.0
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [699a6c99] SimpleTraits v0.9.6
  [45858cf5] Sixel v0.1.5
  [ed01d8cd] Sobol v1.5.0
  [a2af1166] SortingAlgorithms v1.2.3
⌃ [a57abbd0] SparseColumnPivotedQR v2.1.7
  [9f842d2f] SparseConnectivityTracer v1.2.3
⌃ [0a514795] SparseMatrixColorings v0.4.27
  [276daf66] SpecialFunctions v2.9.0
  [860ef19b] StableRNGs v1.0.4
  [cae243ae] StackViews v0.1.2
  [90137ffa] StaticArrays v1.9.20
  [1e83bf80] StaticArraysCore v1.4.4
  [10745b16] Statistics v1.11.5
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.13
  [4c63d2b9] StatsFuns v2.2.1
  [69024149] StringEncodings v0.3.7
⌅ [892a3eda] StringManipulation v0.5.0
  [09ab397b] StructArrays v0.7.3
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [62fd8b95] TensorCore v0.1.1
  [731e570b] TiffImages v0.11.9
  [a759f4b9] TimerOutputs v1.2.1
  [3bb67fe8] TranscodingStreams v0.11.3
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [1cfade01] UnicodeFun v0.4.1
⌃ [1986cc42] Unitful v1.28.0
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [e3aaa7dc] WebP v0.1.3
  [efce3f68] WoodburyMatrices v1.1.0
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [6e34b625] Bzip2_jll v1.0.9+0
  [4e9b3aee] CRlibm_jll v1.0.1+0
  [83423d85] Cairo_jll v1.18.7+0
  [a38c48d9] CoreMath_jll v0.1.0+0
  [ee1fde0b] Dbus_jll v1.16.2+0
⌅ [5ae413db] EarCut_jll v2.2.4+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.8.3+0
⌅ [b22a6f82] FFMPEG_jll v8.1.2+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.5.1+0
  [d2c73de3] GR_jll v0.73.27+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
⌅ [59f7168a] Giflib_jll v5.2.3+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.55.0+0
  [7746bdde] Glib_jll v2.88.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
  [2e76f6c2] HarfBuzz_jll v100.14003.0+0
  [905a6f67] Imath_jll v3.2.2+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.1.0+0
  [1d63c593] LLVMOpenMP_jll v22.1.7+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [856f044c] MKL_jll v2025.2.0+0
  [c771fb93] ODEInterface_jll v0.0.2+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [6cdc7f73] OpenBLASConsistentFPCSR_jll v0.3.34+0
  [18a262bb] OpenEXR_jll v3.4.14+0
  [9bd350c2] OpenSSH_jll v10.5.1+0
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
  [f50d1b31] Rmath_jll v0.5.2+0
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
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
⌅ [214eeab7] fzf_jll v0.61.1+0
  [9a68df92] isoband_jll v0.2.3+0
  [a4ae2306] libaom_jll v3.14.1+0
  [0ac62f75] libass_jll v0.17.5+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [8e53e030] libdrm_jll v2.4.134+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.58+0
  [075b6546] libsixel_jll v1.10.5+0
  [a9144af2] libsodium_jll v1.0.21+0
  [9a156e7d] libva_jll v2.23.0+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [c5f90fcd] libwebp_jll v1.6.0+0
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.3.0+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [8bf52ea8] CRC32c v1.11.0
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
  [1a1011a3] SharedArrays v1.11.0
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

