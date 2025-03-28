---
author: "Avik Pal"
title: "Ill-Conditioned Nonlinear System Work-Precision Diagrams"
---


# Setup

Fetch required packages

```julia
using NonlinearSolve, SparseDiffTools, LinearAlgebra, SparseArrays, DiffEqDevTools,
    CairoMakie, Symbolics, BenchmarkTools, PolyesterForwardDiff, LinearSolve, Sundials,
    Enzyme, SparseConnectivityTracer, DifferentiationInterface
import NLsolve, MINPACK, PETSc, RecursiveFactorization

const RUS = RadiusUpdateSchemes;
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.2;
```




Define a utility to timeout the benchmark after a certain time.

```julia
# Taken from ReTestItems.jl
function timeout(f, timeout)
    cond = Threads.Condition()
    timer = Timer(timeout) do tm
        close(tm)
        ex = ErrorException("timed out after $timeout seconds")
        @lock cond notify(cond, ex; error=false)
    end
    Threads.@spawn begin
        try
            ret = $f()
            isopen(timer) && @lock cond notify(cond, ret)
        catch e
            isopen(timer) && @lock cond notify(cond, CapturedException(e, catch_backtrace()); error=true)
        finally
            close(timer)
        end
    end
    return @lock cond wait(cond) # will throw if we timeout
end
```

```
timeout (generic function with 1 method)
```





Define the Brussletor problem.

```julia
brusselator_f(x, y) = (((x - 3 // 10) ^ 2 + (y - 6 // 10) ^ 2) ≤ 0.01) * 5

limit(a, N) = ifelse(a == N + 1, 1, ifelse(a == 0, N, a))

function init_brusselator_2d(xyd, N)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end

function generate_brusselator_problem(N::Int; sparsity = nothing, kwargs...)
    xyd_brusselator = range(0; stop = 1, length = N)

    function brusselator_2d_loop(du_, u_, p)
        A, B, α, δx = p
        α = α / δx ^ 2

        du = reshape(du_, N, N, 2)
        u = reshape(u_, N, N, 2)

        @inbounds @simd for I in CartesianIndices((N, N))
            i, j = Tuple(I)
            x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
            ip1, im1 = limit(i + 1, N), limit(i - 1, N)
            jp1, jm1 = limit(j + 1, N), limit(j - 1, N)

            du[i, j, 1] = α * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                               4u[i, j, 1]) +
                          B + u[i, j, 1] ^ 2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
                          brusselator_f(x, y)

            du[i, j, 2] = α * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                               4u[i, j, 2]) +
                            A * u[i, j, 1] - u[i, j, 1] ^ 2 * u[i, j, 2]
        end
        return nothing
    end

    return NonlinearProblem(
        NonlinearFunction(brusselator_2d_loop; sparsity),
        vec(init_brusselator_2d(xyd_brusselator, N)),
        (3.4, 1.0, 10.0, step(xyd_brusselator));
        kwargs...
    )
end
```

```
generate_brusselator_problem (generic function with 1 method)
```



```julia
function get_ordering(x::AbstractMatrix)
    idxs = Vector{Int}(undef, size(x, 1))
    placed = zeros(Bool, size(x, 1))
    idx = 1
    for j in size(x, 2):-1:1
        row = view(x, :, j)
        idxs_row = sortperm(row; by = x -> isnan(x) ? Inf : (x == -1 ? Inf : x))
        for i in idxs_row
            if !placed[i] && !isnan(row[i]) && row[i] ≠ -1
                idxs[idx] = i
                placed[i] = true
                idx += 1
                idx > length(idxs) && break
            end
        end
        idx > length(idxs) && break
    end
    return idxs
end
```

```
get_ordering (generic function with 1 method)
```





# Scaling of Sparsity Detection Algorithm

We increase the problem size, and compute the jacobian 10 times similar to a real workload
where the jacobian is computed several times and amortizes the cost for computing the
sparsity pattern.

```julia
test_problem = generate_brusselator_problem(4)
bruss_f!, u0 = (du, u) -> test_problem.f(du, u, test_problem.p), test_problem.u0
y = similar(u0)

J = Float64.(ADTypes.jacobian_sparsity(bruss_f!, y, u0, TracerSparsityDetector()))
colors = matrix_colors(J)

begin
    J_ = similar(J)
    rows = rowvals(J)
    vals = nonzeros(J)
    for j in 1:size(J, 2)
        for i in nzrange(J, j)
            row = rows[i]
            J_[j, row] = colors[j]  # spy does a ordering I can't figure out. so transposing it here
        end
    end
end

function cache_and_compute_10_jacobians(adtype, f!::F, y, x, p) where {F}
    prep = DifferentiationInterface.prepare_jacobian(f!, y, adtype, x, Constant(p))
    J = DifferentiationInterface.jacobian(f!, y, prep, adtype, x, Constant(p))
    for _ in 1:9
        DifferentiationInterface.jacobian!(f!, y, J, prep, adtype, x, Constant(p))
    end
    return J
end

# XXX: run till 8 on CI
# Ns = [2^i for i in 1:8]
Ns = [2^i for i in 1:6]

adtypes = [
    (
        AutoSparse(AutoFiniteDiff(); sparsity_detector=TracerSparsityDetector()),
        [:finitediff, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoPolyesterForwardDiff(; chunksize=8);
            sparsity_detector=TracerSparsityDetector()
        ),
        [:forwarddiff, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoEnzyme(; mode=Enzyme.Forward);
            sparsity_detector=TracerSparsityDetector()
        ),
        [:enzyme, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoFiniteDiff();
            sparsity_detector=DenseSparsityDetector(AutoFiniteDiff(); atol=1e-5)
        ),
        [:finitediff, :approx_sparse]
    ),
    (
        AutoSparse(
            AutoPolyesterForwardDiff(; chunksize=8);
            sparsity_detector=DenseSparsityDetector(
                AutoPolyesterForwardDiff(; chunksize=8); atol=1e-5
            )
        ),
        [:polyester, :approx_sparse]
    ),
    (
        AutoSparse(
            AutoEnzyme(; mode=Enzyme.Forward);
            sparsity_detector=DenseSparsityDetector(
                AutoEnzyme(; mode=Enzyme.Forward); atol=1e-5
            )
        ),
        [:enzyme, :approx_sparse]
    ),
    (
        AutoPolyesterForwardDiff(; chunksize=8),
        [:polyester, :none]
    ),
]

times = Matrix{Float64}(undef, length(Ns), length(adtypes))

for (i, N) in enumerate(Ns)
    @info N
    test_problem = generate_brusselator_problem(N)
    bruss_f! = test_problem.f
    u0 = test_problem.u0
    y = similar(u0)

    for (j, (adtype, _)) in enumerate(adtypes)
        times[i, j] = @belapsed begin
            $(cache_and_compute_10_jacobians)(
                $(adtype), $(bruss_f!), $(y), $(u0), $(test_problem.p)
            )
        end
        @info times[i, j]
    end
end
```




Plotting the results.

```julia
symbol_to_adname = Dict(
    :finitediff => "Finite Diff",
    :forwarddiff => "Forward Mode AD",
    :polyester => "Threaded Forward Mode AD",
    :enzyme => "Forward Mode AD (Enzyme)",
)

fig = begin
    cycle = Cycle([:marker], covary=true)
    plot_theme = Theme(Lines=(; cycle), Scatter=(; cycle))

    with_theme(plot_theme) do
        fig = Figure(; size=(1400, 1400 * 0.5))

        ax = Axis(fig[1, 1]; title="Sparsity Pattern for 2D Brusselator Jacobian",
            titlesize=22, titlegap=10,
            xticksize=20, yticksize=20, xticklabelsize=20, yticklabelsize=20,
            xtickwidth=2.5, ytickwidth=2.5, spinewidth=2.5, yreversed=true)

        spy!(ax, J_; markersize=1, marker=:circle, framecolor=:lightgray,
            colormap=:seaborn_bright)

        ax = Axis(fig[1, 2]; title="Scaling of Sparse Jacobian Computation",
            titlesize=22, titlegap=10, xscale=log10, yscale=log10,
            xticksize=20, yticksize=20, xticklabelsize=20, yticklabelsize=20,
            xtickwidth=2.5, ytickwidth=2.5, spinewidth=2.5,
            xlabel=L"Input Dimension ($\mathbf{N}$)", ylabel=L"Time $\mathbf{(s)}$", xlabelsize=22,
            ylabelsize=22, yaxisposition=:right)

        colors = cgrad(:seaborn_bright, length(adtypes); categorical=true)

        line_list = []
        scatter_list = []
        Ns_ = Ns .^ 2 .* 2
        linestyles = [:solid, :solid, :solid, :dash, :dash, :dash, :dot, :dot]

        for (i, times) in enumerate(eachcol(times))
            l = lines!(Ns_, times; linewidth=5, color=colors[i], linestyle=linestyles[i])
            push!(line_list, l)
            sc = scatter!(Ns_, times; markersize=16, strokewidth=2, color=colors[i])
            push!(scatter_list, sc)
        end

        tracer_idxs = [idx for idx in 1:length(adtypes) if :exact_sparse ∈ adtypes[idx][2]]
        group_tracer = [
            [
                LineElement(;
                    color=line_list[idx].color,
                    linestyle=line_list[idx].linestyle,
                    linewidth=line_list[idx].linewidth,
                ),
                MarkerElement(;
                    color=scatter_list[idx].color,
                    marker=scatter_list[idx].marker,
                    strokewidth=scatter_list[idx].strokewidth,
                    markersize=scatter_list[idx].markersize,
                ),
            ] for idx in tracer_idxs
        ]

        local_sparse_idxs = [idx for idx in 1:length(adtypes) if :approx_sparse ∈ adtypes[idx][2]]
        group_local_sparse = [
            [
                LineElement(;
                    color=line_list[idx].color,
                    linestyle=line_list[idx].linestyle,
                    linewidth=line_list[idx].linewidth,
                ),
                MarkerElement(;
                    color=scatter_list[idx].color,
                    marker=scatter_list[idx].marker,
                    strokewidth=scatter_list[idx].strokewidth,
                    markersize=scatter_list[idx].markersize,
                ),
            ] for idx in local_sparse_idxs
        ]

        non_sparse_idxs = [idx for idx in 1:length(adtypes) if :none ∈ adtypes[idx][2]]
        group_nonsparse = [
            [
                LineElement(;
                    color=line_list[idx].color,
                    linestyle=line_list[idx].linestyle,
                    linewidth=line_list[idx].linewidth,
                ),
                MarkerElement(;
                    color=scatter_list[idx].color,
                    marker=scatter_list[idx].marker,
                    strokewidth=scatter_list[idx].strokewidth,
                    markersize=scatter_list[idx].markersize,
                ),
            ] for idx in non_sparse_idxs
        ]

        axislegend(
            ax,
            [group_tracer, group_local_sparse, group_nonsparse],
            [
                [symbol_to_adname[adtypes[idx][2][1]] for idx in tracer_idxs],
                [symbol_to_adname[adtypes[idx][2][1]] for idx in local_sparse_idxs],
                [symbol_to_adname[adtypes[idx][2][1]] for idx in non_sparse_idxs],
            ],
            ["Exact Sparsity", "Approx. Local Sparsity", "No Sparsity"];
            position=:rb, framevisible=true, framewidth=2.5, titlesize=18,
            labelsize=16, patchsize=(40.0f0, 20.0f0)
        )

        fig
    end
end
```

![](figures/bruss_6_1.png)

```julia
save("brusselator_sparse_jacobian_scaling.svg", fig)
```

```
CairoMakie.Screen{SVG}
```





# Scaling with Problem Size

First, let us experiment the scaling of each algorithm with the problem size.

```julia
Ns = vcat(collect(2 .^ (2:7)), [150, 175, 200])

# XXX: PETSc just segfaults
solvers_scaling = [
    (; pkg = :nonlinearsolve,       sparsity = :none,   name = "NR (No Sparsity)",                    alg = NewtonRaphson()),
    # (; pkg = :nonlinearsolve,       sparsity = :approx, name = "NR (Approx. Local Sparsity)",         alg = NewtonRaphson()),
    (; pkg = :nonlinearsolve,       sparsity = :exact,  name = "NR (Exact Sparsity)",                 alg = NewtonRaphson()),
    (; pkg = :wrapper,              sparsity = :none,   name = "NR [NLsolve.jl]",                     alg = NLsolveJL(; method = :newton, autodiff = :forward)),
    (; pkg = :wrapper,              sparsity = :none,   name = "Mod. NR [Sundials]",                  alg = KINSOL()),
    # (; pkg = :wrapper,              sparsity = :none,   name = "NR [PETSc] (No Sparsity)",            alg = PETScSNES(; snes_type = "newtonls")),
    # (; pkg = :wrapper,              sparsity = :none,   name = "NR [PETSc] (Approx. Local Sparsity)", alg = PETScSNES(; snes_type = "newtonls")),
    # (; pkg = :wrapper,              sparsity = :none,   name = "NR [PETSc] (Exact Sparsity)",         alg = PETScSNES(; snes_type = "newtonls")),

    (; pkg = :nonlinearsolve,       sparsity = :none,   name = "TR (No Sparsity)",                    alg = TrustRegion(; radius_update_scheme = RUS.NLsolve)),
    # (; pkg = :nonlinearsolve,       sparsity = :approx, name = "TR (Approx. Local Sparsity)",         alg = TrustRegion(; radius_update_scheme = RUS.NLsolve)),
    (; pkg = :nonlinearsolve,       sparsity = :exact,  name = "TR (Exact Sparsity)",                 alg = TrustRegion(; radius_update_scheme = RUS.NLsolve)),
    (; pkg = :wrapper,              sparsity = :none,   name = "TR [NLsolve.jl]",                     alg = NLsolveJL(; autodiff = :forward)),
    # (; pkg = :wrapper,              sparsity = :none,   name = "TR [PETSc] (No Sparsity)",            alg = PETScSNES(; snes_type = "newtontr")),
    # (; pkg = :wrapper,              sparsity = :none,   name = "TR [PETSc] (Approx. Local Sparsity)", alg = PETScSNES(; snes_type = "newtontr")),
    # (; pkg = :wrapper,              sparsity = :none,   name = "TR [PETSc] (Exact Sparsity)",         alg = PETScSNES(; snes_type = "newtontr")),

    (; pkg = :wrapper,              sparsity = :none,   name = "Mod. Powell [MINPACK]",               alg = CMINPACK()),
]

runtimes_scaling = fill(-1.0, length(solvers_scaling), length(Ns))

for (i, N) in enumerate(Ns)
    prob_dense = generate_brusselator_problem(N)
    prob_exact_sparse = generate_brusselator_problem(N;
        sparsity = TracerSparsityDetector()
    )

    @info "Benchmarking N = $N"

    for (j, solver) in enumerate(solvers_scaling)
        ptype = solver.sparsity
        alg = solver.alg
        name = solver.name

        prob = if ptype == :none
            prob_dense
        elseif ptype == :approx
            # With Tracing based sparsity detection, we dont need this any more
            error("Approximate Sparsity not implemented")
        elseif ptype == :exact
            prob_exact_sparse
        end

        if (j > 1 && runtimes_scaling[j - 1, i] == -1) ||
            (alg isa CMINPACK && N > 32) ||
            (alg isa KINSOL && N > 64) ||
            (alg isa NLsolveJL && N > 64 && alg.method == :trust_region) ||
            (alg isa GeneralizedFirstOrderAlgorithm && alg.name == :TrustRegion && N > 64) ||
            (alg isa NLsolveJL && N > 128 && alg.method == :newton) ||
            (alg isa GeneralizedFirstOrderAlgorithm && alg.name == :NewtonRaphson && N > 128 && ptype == :none) ||
            (alg isa GeneralizedFirstOrderAlgorithm && alg.name == :NewtonRaphson && N > 150 && ptype == :approx)
            # The last benchmark failed so skip this too
            runtimes_scaling[j, i] = NaN
            @warn "$(name): Would Have Timed out"
        else
            function benchmark_function()
                termination_condition = (alg isa PETScSNES || alg isa KINSOL) ?
                                        nothing :
                                        NonlinearSolveBase.AbsNormTerminationMode(Base.Fix1(maximum, abs))
                sol = solve(prob, alg; abstol=1e-6, reltol=1e-6, termination_condition)
                runtimes_scaling[j, i] = @belapsed solve($prob, $alg; abstol=1e-6,
                    reltol=1e-6, termination_condition=$termination_condition)
                @info "$(name): $(runtimes_scaling[j, i]) | $(norm(sol.resid, Inf)) | $(sol.retcode)"
            end

            timeout(benchmark_function, 600)

            if runtimes_scaling[j, i] == -1
                @warn "$(name): Timed out"
                runtimes_scaling[j, i] = NaN
            end
        end
    end

    println()
end
```




Plot the results.

```julia
fig = begin
    ASPECT_RATIO = 0.7
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    cycle = Cycle([:marker], covary = true)
    colors = cgrad(:seaborn_bright, length(solvers_scaling); categorical = true)
    theme = Theme(Lines = (cycle = cycle,), Scatter = (cycle = cycle,))
    LINESTYLES = Dict(
        (:nonlinearsolve, :none) => :solid,
        # (:nonlinearsolve, :approx) => :dash,
        (:nonlinearsolve, :exact) => :dashdot,
        # (:simplenonlinearsolve, :none) => :solid,
        (:wrapper, :none) => :dot,
    )

    Ns_ = Ns .^ 2 .* 2

    with_theme(theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        ax = Axis(fig[1, 1:3], ylabel = L"Time ($s$)", xlabel = L"Problem Size ($N$)",
            xscale = log10, yscale = log10, xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 20, yticklabelsize = 20, xtickwidth = STROKEWIDTH,
            ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH)

        idxs = get_ordering(runtimes_scaling)

        ls, scs = [], []
        for (i, solver) in zip(idxs, solvers_scaling[idxs])
            linestyle = LINESTYLES[(solver.pkg, solver.sparsity)]
            l = lines!(Ns_, runtimes_scaling[i, :]; linewidth = 5, color = colors[i],
                linestyle)
            sc = scatter!(Ns_, runtimes_scaling[i, :]; markersize = 16, strokewidth = 2,
                color = colors[i])
            push!(ls, l)
            push!(scs, sc)
        end

        main_legend = [
            [
                LineElement(; color = ls[idx].color, linestyle = ls[idx].linestyle,
                    linewidth = ls[idx].linewidth),
                MarkerElement(; color = scs[idx].color, marker = scs[idx].marker,
                    markersize = scs[idx].markersize, strokewidth = scs[idx].strokewidth)
            ]
            for idx in 1:length(solvers_scaling)
        ]

        sparsity_legend = [
            LineElement(; linestyle = :solid, linewidth = 5),
            # LineElement(; linestyle = :dash, linewidth = 5),
            LineElement(; linestyle = :dashdot, linewidth = 5),
        ]

        axislegend(ax, main_legend, [s.name for s in solvers_scaling[idxs]],
            "Successful Solvers\n(Fastest to Slowest)";
            framevisible=true, framewidth = STROKEWIDTH, orientation = :vertical,
            titlesize = 20, nbanks = 1, labelsize = 16,
            tellheight = true, tellwidth = false, patchsize = (60.0f0, 20.0f0),
            position = :rb)

        axislegend(ax, sparsity_legend,
            [
                "No Sparsity Detection",
                # "Approx. Sparsity",
                "Exact Sparsity"
            ],
            "Sparsity Detection"; framevisible=true, framewidth = STROKEWIDTH,
            orientation = :vertical, titlesize = 20, nbanks = 1, labelsize = 16,
            tellheight = true, tellwidth = false, patchsize = (60.0f0, 20.0f0),
            position = :lt)

        fig[0, :] = Label(fig,
            "Brusselator 2D: Scaling of First-Order Nonlinear Solvers with Problem Size",
            fontsize = 24, tellwidth = false, font = :bold)

        return fig
    end
end
```

![](figures/bruss_9_1.png)

```julia
save("brusselator_scaling.svg", fig)
```

```
CairoMakie.Screen{SVG}
```





# Jacobian-Free Newton / TR Krylov Methods

In this section, we will benchmark jacobian-free nonlinear solvers with Krylov methods. We
will use preconditioning from `AlgebraicMultigrid.jl` and `IncompleteLU.jl`. Unfortunately,
our ability to use 3rd party software is limited here, since only `Sundials.jl` supports
jacobian-free methods via `:GMRES`.

```julia
using AlgebraicMultigrid, IncompleteLU

incompletelu(W, p = nothing) = ilu(W, τ = 50.0), LinearAlgebra.I

function algebraicmultigrid(W, p = nothing)
    return aspreconditioner(ruge_stuben(convert(AbstractMatrix, W))), LinearAlgebra.I
end

function algebraicmultigrid_jacobi(W, p = nothing)
    A = convert(AbstractMatrix, W)
    Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(
        A, presmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1))),
        postsmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1)))
    ))
    return Pl, LinearAlgebra.I
end

Ns = 2 .^ (2:7)

solvers_scaling_jacobian_free = [
    (; pkg = :nonlinearsolve,  name = "Newton Krylov",                    alg = NewtonRaphson(; linsolve = KrylovJL_GMRES())),
    (; pkg = :nonlinearsolve,  name = "Newton Krylov (ILU)",              alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(; precs = incompletelu), concrete_jac = true)),
    (; pkg = :nonlinearsolve,  name = "Newton Krylov (AMG)",              alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid), concrete_jac = true)),
    (; pkg = :nonlinearsolve,  name = "Newton Krylov (AMG Jacobi)",       alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid_jacobi), concrete_jac = true)),
    (; pkg = :nonlinearsolve,  name = "TR Krylov",                        alg = TrustRegion(; linsolve = KrylovJL_GMRES())),
    (; pkg = :nonlinearsolve,  name = "TR Krylov (ILU)",                  alg = TrustRegion(; linsolve = KrylovJL_GMRES(; precs = incompletelu), concrete_jac = true)),
    (; pkg = :nonlinearsolve,  name = "TR Krylov (AMG)",                  alg = TrustRegion(; linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid), concrete_jac = true)),
    (; pkg = :nonlinearsolve,  name = "TR Krylov (AMG Jacobi)",           alg = TrustRegion(; linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid_jacobi), concrete_jac = true)),
    (; pkg = :wrapper,         name = "Newton Krylov [Sundials]",         alg = KINSOL(; linear_solver = :GMRES)),
]

runtimes_scaling = zeros(length(solvers_scaling_jacobian_free), length(Ns)) .- 1

for (i, N) in enumerate(Ns)
    prob = generate_brusselator_problem(
        N; sparsity = TracerSparsityDetector()
    )

    @info "Benchmarking N = $N"

    for (j, solver) in enumerate(solvers_scaling_jacobian_free)
        alg = solver.alg
        name = solver.name

        if (j > 1 && runtimes_scaling[j - 1, i] == -1)
            # The last benchmark failed so skip this too
            runtimes_scaling[j, i] = NaN
            @warn "$(name): Would Have Timed out"
        else
            function benchmark_function()
                termination_condition = (alg isa PETScSNES || alg isa KINSOL) ?
                                        nothing :
                                        NonlinearSolveBase.AbsNormTerminationMode(Base.Fix1(maximum, abs))
                sol = solve(prob, alg; abstol=1e-6, reltol=1e-6,
                    linsolve_kwargs = (; abstol = 1e-9, reltol = 1e-9),
                    termination_condition)
                if SciMLBase.successful_retcode(sol) || norm(sol.resid, Inf) ≤ 1e-5
                    runtimes_scaling[j, i] = @belapsed solve($prob, $alg; abstol=1e-6,
                        reltol=1e-6, 
                        linsolve_kwargs = (; abstol = 1e-9, reltol = 1e-9),
                        termination_condition=$termination_condition)
                else
                    runtimes_scaling[j, i] = NaN
                end
                @info "$(name): $(runtimes_scaling[j, i]) | $(norm(sol.resid, Inf)) | $(sol.retcode)"
            end

            timeout(benchmark_function, 600)

            if runtimes_scaling[j, i] == -1
                @warn "$(name): Timed out"
                runtimes_scaling[j, i] = NaN
            end
        end
    end

    println()
end
```




Plot the results.

```julia
fig = begin
    ASPECT_RATIO = 0.7
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    cycle = Cycle([:marker], covary = true)
    colors = cgrad(:seaborn_bright, length(solvers_scaling_jacobian_free); categorical = true)
    theme = Theme(Lines = (cycle = cycle,), Scatter = (cycle = cycle,))
    LINESTYLES = Dict(
        (:nonlinearsolve, :none) => :solid,
        (:nonlinearsolve, :amg) => :dot,
        (:nonlinearsolve, :amg_jacobi) => :dash,
        (:nonlinearsolve, :ilu) => :dashdot,
    )

    Ns_ = Ns .^ 2 .* 2

    with_theme(theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        ax = Axis(fig[1, 1:2], ylabel = L"Time ($s$)", xlabel = L"Problem Size ($N$)",
            xscale = log10, yscale = log10, xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 20, yticklabelsize = 20, xtickwidth = STROKEWIDTH,
            ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH)

        idxs = get_ordering(runtimes_scaling)

        ls, scs, labels = [], [], []
        for (i, solver) in zip(idxs, solvers_scaling_jacobian_free[idxs])
            all(isnan, runtimes_scaling[i, :]) && continue
            precon = occursin("AMG Jacobi", solver.name) ? :amg_jacobi : occursin("AMG", solver.name) ? :amg : occursin("ILU", solver.name) ? :ilu : :none
            linestyle = LINESTYLES[(solver.pkg, precon)]
            l = lines!(Ns_, runtimes_scaling[i, :]; linewidth = 5, color = colors[i],
                linestyle)
            sc = scatter!(Ns_, runtimes_scaling[i, :]; markersize = 16, strokewidth = 2,
                color = colors[i])
            push!(ls, l)
            push!(scs, sc)
            push!(labels, solver.name)
        end

        axislegend(ax, [[l, sc] for (l, sc) in zip(ls, scs)], labels,
            "Successful Solvers\n(Fastest to Slowest)";
            framevisible=true, framewidth = STROKEWIDTH, orientation = :vertical,
            titlesize = 20, labelsize = 16, position = :rb,
            tellheight = true, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        axislegend(ax, [
                LineElement(; linestyle = :solid, linewidth = 5),
                LineElement(; linestyle = :dot, linewidth = 5),
                LineElement(; linestyle = :dash, linewidth = 5),
                LineElement(; linestyle = :dashdot, linewidth = 5),
            ], ["No Preconditioning", "AMG", "AMG Jacobi", "Incomplete LU"],
            "Preconditioning"; framevisible=true, framewidth = STROKEWIDTH,
            orientation = :vertical, titlesize = 20, labelsize = 16,
            tellheight = true, tellwidth = true, patchsize = (40.0f0, 20.0f0),
            position = :lt)

        fig[0, :] = Label(fig,
            "Brusselator 2D: Scaling of Jacobian-Free Nonlinear Solvers with Problem Size",
            fontsize = 24, tellwidth = false, font = :bold)

        return fig
    end
end
```

```
Error: BoundsError: attempt to access 9-element Vector{NamedTuple{(:pkg, :n
ame, :alg)}} at index [[3, 7, 5, 1, 8, 4, 6, 2, 139988625266448]]
```



```julia
save("brusselator_krylov_methods_scaling.svg", fig)
```

```
CairoMakie.Screen{SVG}
```





# Work-Precision Diagram

In this section, we will generate the work-precision of the solvers. All solvers that can
exploit sparsity will automatically do so.

```julia
solvers_all = [
    (; pkg = :nonlinearsolve,       name = "Default PolyAlg",                  solver = Dict(:alg => FastShortcutNonlinearPolyalg())),
    (; pkg = :nonlinearsolve,       name = "RobustMultiNewton (GMRES)",        solver = Dict(:alg => RobustMultiNewton(; linsolve = KrylovJL_GMRES()))),
    (; pkg = :nonlinearsolve,       name = "Newton Raphson",                   solver = Dict(:alg => NewtonRaphson(; linsolve = nothing))),
    (; pkg = :nonlinearsolve,       name = "Newton Krylov",                    solver = Dict(:alg => NewtonRaphson(; linsolve = KrylovJL_GMRES()))),
    (; pkg = :nonlinearsolve,       name = "Trust Region",                     solver = Dict(:alg => TrustRegion())),
    (; pkg = :nonlinearsolve,       name = "TR Krylov",                        solver = Dict(:alg => TrustRegion(; linsolve = KrylovJL_GMRES()))),
    (; pkg = :wrapper,              name = "NR [NLsolve.jl]",                  solver = Dict(:alg => NLsolveJL(; method = :newton, autodiff = :forward))),
    (; pkg = :wrapper,              name = "TR [NLsolve.jl]",                  solver = Dict(:alg => NLsolveJL(; autodiff = :forward))),
    (; pkg = :wrapper,              name = "NR [Sundials]",                    solver = Dict(:alg => KINSOL())),
    (; pkg = :wrapper,              name = "Newton Krylov [Sundials]",         solver = Dict(:alg => KINSOL(; linear_solver = :GMRES))),

    (; pkg = :wrapper,              name = "Mod. Powell [MINPACK]",            solver = Dict(:alg => CMINPACK())),
];
```


```julia
prob_wpd = generate_brusselator_problem(32; sparsity = TracerSparsityDetector())

abstols = 1.0 ./ 10 .^ (2:10)
reltols = 1.0 ./ 10 .^ (2:10)

function check_solver(prob, solver)
    try
        sol = solve(prob, solver.solver[:alg]; abstol = 1e-4, reltol = 1e-4,
            maxiters = 10000)
        err = norm(sol.resid, Inf)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("[Warn] Solver $(solver.name) returned retcode $(sol.retcode) with an residual norm = $(norm(sol.resid)).\n";
                color = :red)
            return false
        elseif err > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm = $(norm(sol.resid))).\n";
                color = :red)
            return false
        elseif isinf(err) || isnan(err)
            Base.printstyled("[Warn] Solver $(solver.name) had a residual of $(err).\n";
                color = :red)
            return false
        end
        Base.printstyled("[Info] Solver $(solver.name) successfully solved the problem (norm = $(norm(sol.resid))).\n";
            color = :green)
    catch e
        Base.printstyled("[Warn] Solver $(solver.name) threw an error: $e.\n"; color = :red)
        return false
    end
    return true
end

function generate_wpset(prob, solvers)
    # Finds the solvers that can solve the problem
    successful_solvers = filter(solver -> check_solver(prob, solver), solvers)

    return WorkPrecisionSet(prob, abstols, reltols,
        getfield.(successful_solvers, :solver);
        names = getfield.(successful_solvers, :name), numruns = 10, error_estimate = :l∞,
        maxiters = 1000, verbose = true), successful_solvers
end
```

```
generate_wpset (generic function with 1 method)
```



```julia
wp_set, successful_solvers = generate_wpset(prob_wpd, solvers_all);
```

```
[Warn] Solver Default PolyAlg threw an error: MethodError(Core.kwcall, ((fu
ll = false, alg = LinearAlgebra.DivideAndConquer()), LinearAlgebra.svd!, sp
arse([1, 2, 32, 33, 993, 1025, 1, 2, 3, 34, 994, 1026, 2, 3, 4, 35, 995, 10
27, 3, 4, 5, 36, 996, 1028, 4, 5, 6, 37, 997, 1029, 5, 6, 7, 38, 998, 1030,
 6, 7, 8, 39, 999, 1031, 7, 8, 9, 40, 1000, 1032, 8, 9, 10, 41, 1001, 1033,
 9, 10, 11, 42, 1002, 1034, 10, 11, 12, 43, 1003, 1035, 11, 12, 13, 44, 100
4, 1036, 12, 13, 14, 45, 1005, 1037, 13, 14, 15, 46, 1006, 1038, 14, 15, 16
, 47, 1007, 1039, 15, 16, 17, 48, 1008, 1040, 16, 17, 18, 49, 1009, 1041, 1
7, 18, 19, 50, 1010, 1042, 18, 19, 20, 51, 1011, 1043, 19, 20, 21, 52, 1012
, 1044, 20, 21, 22, 53, 1013, 1045, 21, 22, 23, 54, 1014, 1046, 22, 23, 24,
 55, 1015, 1047, 23, 24, 25, 56, 1016, 1048, 24, 25, 26, 57, 1017, 1049, 25
, 26, 27, 58, 1018, 1050, 26, 27, 28, 59, 1019, 1051, 27, 28, 29, 60, 1020,
 1052, 28, 29, 30, 61, 1021, 1053, 29, 30, 31, 62, 1022, 1054, 30, 31, 32, 
63, 1023, 1055, 1, 31, 32, 64, 1024, 1056, 1, 33, 34, 64, 65, 1057, 2, 33, 
34, 35, 66, 1058, 3, 34, 35, 36, 67, 1059, 4, 35, 36, 37, 68, 1060, 5, 36, 
37, 38, 69, 1061, 6, 37, 38, 39, 70, 1062, 7, 38, 39, 40, 71, 1063, 8, 39, 
40, 41, 72, 1064, 9, 40, 41, 42, 73, 1065, 10, 41, 42, 43, 74, 1066, 11, 42
, 43, 44, 75, 1067, 12, 43, 44, 45, 76, 1068, 13, 44, 45, 46, 77, 1069, 14,
 45, 46, 47, 78, 1070, 15, 46, 47, 48, 79, 1071, 16, 47, 48, 49, 80, 1072, 
17, 48, 49, 50, 81, 1073, 18, 49, 50, 51, 82, 1074, 19, 50, 51, 52, 83, 107
5, 20, 51, 52, 53, 84, 1076, 21, 52, 53, 54, 85, 1077, 22, 53, 54, 55, 86, 
1078, 23, 54, 55, 56, 87, 1079, 24, 55, 56, 57, 88, 1080, 25, 56, 57, 58, 8
9, 1081, 26, 57, 58, 59, 90, 1082, 27, 58, 59, 60, 91, 1083, 28, 59, 60, 61
, 92, 1084, 29, 60, 61, 62, 93, 1085, 30, 61, 62, 63, 94, 1086, 31, 62, 63,
 64, 95, 1087, 32, 33, 63, 64, 96, 1088, 33, 65, 66, 96, 97, 1089, 34, 65, 
66, 67, 98, 1090, 35, 66, 67, 68, 99, 1091, 36, 67, 68, 69, 100, 1092, 37, 
68, 69, 70, 101, 1093, 38, 69, 70, 71, 102, 1094, 39, 70, 71, 72, 103, 1095
, 40, 71, 72, 73, 104, 1096, 41, 72, 73, 74, 105, 1097, 42, 73, 74, 75, 106
, 1098, 43, 74, 75, 76, 107, 1099, 44, 75, 76, 77, 108, 1100, 45, 76, 77, 7
8, 109, 1101, 46, 77, 78, 79, 110, 1102, 47, 78, 79, 80, 111, 1103, 48, 79,
 80, 81, 112, 1104, 49, 80, 81, 82, 113, 1105, 50, 81, 82, 83, 114, 1106, 5
1, 82, 83, 84, 115, 1107, 52, 83, 84, 85, 116, 1108, 53, 84, 85, 86, 117, 1
109, 54, 85, 86, 87, 118, 1110, 55, 86, 87, 88, 119, 1111, 56, 87, 88, 89, 
120, 1112, 57, 88, 89, 90, 121, 1113, 58, 89, 90, 91, 122, 1114, 59, 90, 91
, 92, 123, 1115, 60, 91, 92, 93, 124, 1116, 61, 92, 93, 94, 125, 1117, 62, 
93, 94, 95, 126, 1118, 63, 94, 95, 96, 127, 1119, 64, 65, 95, 96, 128, 1120
, 65, 97, 98, 128, 129, 1121, 66, 97, 98, 99, 130, 1122, 67, 98, 99, 100, 1
31, 1123, 68, 99, 100, 101, 132, 1124, 69, 100, 101, 102, 133, 1125, 70, 10
1, 102, 103, 134, 1126, 71, 102, 103, 104, 135, 1127, 72, 103, 104, 105, 13
6, 1128, 73, 104, 105, 106, 137, 1129, 74, 105, 106, 107, 138, 1130, 75, 10
6, 107, 108, 139, 1131, 76, 107, 108, 109, 140, 1132, 77, 108, 109, 110, 14
1, 1133, 78, 109, 110, 111, 142, 1134, 79, 110, 111, 112, 143, 1135, 80, 11
1, 112, 113, 144, 1136, 81, 112, 113, 114, 145, 1137, 82, 113, 114, 115, 14
6, 1138, 83, 114, 115, 116, 147, 1139, 84, 115, 116, 117, 148, 1140, 85, 11
6, 117, 118, 149, 1141, 86, 117, 118, 119, 150, 1142, 87, 118, 119, 120, 15
1, 1143, 88, 119, 120, 121, 152, 1144, 89, 120, 121, 122, 153, 1145, 90, 12
1, 122, 123, 154, 1146, 91, 122, 123, 124, 155, 1147, 92, 123, 124, 125, 15
6, 1148, 93, 124, 125, 126, 157, 1149, 94, 125, 126, 127, 158, 1150, 95, 12
6, 127, 128, 159, 1151, 96, 97, 127, 128, 160, 1152, 97, 129, 130, 160, 161
, 1153, 98, 129, 130, 131, 162, 1154, 99, 130, 131, 132, 163, 1155, 100, 13
1, 132, 133, 164, 1156, 101, 132, 133, 134, 165, 1157, 102, 133, 134, 135, 
166, 1158, 103, 134, 135, 136, 167, 1159, 104, 135, 136, 137, 168, 1160, 10
5, 136, 137, 138, 169, 1161, 106, 137, 138, 139, 170, 1162, 107, 138, 139, 
140, 171, 1163, 108, 139, 140, 141, 172, 1164, 109, 140, 141, 142, 173, 116
5, 110, 141, 142, 143, 174, 1166, 111, 142, 143, 144, 175, 1167, 112, 143, 
144, 145, 176, 1168, 113, 144, 145, 146, 177, 1169, 114, 145, 146, 147, 178
, 1170, 115, 146, 147, 148, 179, 1171, 116, 147, 148, 149, 180, 1172, 117, 
148, 149, 150, 181, 1173, 118, 149, 150, 151, 182, 1174, 119, 150, 151, 152
, 183, 1175, 120, 151, 152, 153, 184, 1176, 121, 152, 153, 154, 185, 1177, 
122, 153, 154, 155, 186, 1178, 123, 154, 155, 156, 187, 1179, 124, 155, 156
, 157, 188, 1180, 125, 156, 157, 158, 189, 1181, 126, 157, 158, 159, 190, 1
182, 127, 158, 159, 160, 191, 1183, 128, 129, 159, 160, 192, 1184, 129, 161
, 162, 192, 193, 1185, 130, 161, 162, 163, 194, 1186, 131, 162, 163, 164, 1
95, 1187, 132, 163, 164, 165, 196, 1188, 133, 164, 165, 166, 197, 1189, 134
, 165, 166, 167, 198, 1190, 135, 166, 167, 168, 199, 1191, 136, 167, 168, 1
69, 200, 1192, 137, 168, 169, 170, 201, 1193, 138, 169, 170, 171, 202, 1194
, 139, 170, 171, 172, 203, 1195, 140, 171, 172, 173, 204, 1196, 141, 172, 1
73, 174, 205, 1197, 142, 173, 174, 175, 206, 1198, 143, 174, 175, 176, 207,
 1199, 144, 175, 176, 177, 208, 1200, 145, 176, 177, 178, 209, 1201, 146, 1
77, 178, 179, 210, 1202, 147, 178, 179, 180, 211, 1203, 148, 179, 180, 181,
 212, 1204, 149, 180, 181, 182, 213, 1205, 150, 181, 182, 183, 214, 1206, 1
51, 182, 183, 184, 215, 1207, 152, 183, 184, 185, 216, 1208, 153, 184, 185,
 186, 217, 1209, 154, 185, 186, 187, 218, 1210, 155, 186, 187, 188, 219, 12
11, 156, 187, 188, 189, 220, 1212, 157, 188, 189, 190, 221, 1213, 158, 189,
 190, 191, 222, 1214, 159, 190, 191, 192, 223, 1215, 160, 161, 191, 192, 22
4, 1216, 161, 193, 194, 224, 225, 1217, 162, 193, 194, 195, 226, 1218, 163,
 194, 195, 196, 227, 1219, 164, 195, 196, 197, 228, 1220, 165, 196, 197, 19
8, 229, 1221, 166, 197, 198, 199, 230, 1222, 167, 198, 199, 200, 231, 1223,
 168, 199, 200, 201, 232, 1224, 169, 200, 201, 202, 233, 1225, 170, 201, 20
2, 203, 234, 1226, 171, 202, 203, 204, 235, 1227, 172, 203, 204, 205, 236, 
1228, 173, 204, 205, 206, 237, 1229, 174, 205, 206, 207, 238, 1230, 175, 20
6, 207, 208, 239, 1231, 176, 207, 208, 209, 240, 1232, 177, 208, 209, 210, 
241, 1233, 178, 209, 210, 211, 242, 1234, 179, 210, 211, 212, 243, 1235, 18
0, 211, 212, 213, 244, 1236, 181, 212, 213, 214, 245, 1237, 182, 213, 214, 
215, 246, 1238, 183, 214, 215, 216, 247, 1239, 184, 215, 216, 217, 248, 124
0, 185, 216, 217, 218, 249, 1241, 186, 217, 218, 219, 250, 1242, 187, 218, 
219, 220, 251, 1243, 188, 219, 220, 221, 252, 1244, 189, 220, 221, 222, 253
, 1245, 190, 221, 222, 223, 254, 1246, 191, 222, 223, 224, 255, 1247, 192, 
193, 223, 224, 256, 1248, 193, 225, 226, 256, 257, 1249, 194, 225, 226, 227
, 258, 1250, 195, 226, 227, 228, 259, 1251, 196, 227, 228, 229, 260, 1252, 
197, 228, 229, 230, 261, 1253, 198, 229, 230, 231, 262, 1254, 199, 230, 231
, 232, 263, 1255, 200, 231, 232, 233, 264, 1256, 201, 232, 233, 234, 265, 1
257, 202, 233, 234, 235, 266, 1258, 203, 234, 235, 236, 267, 1259, 204, 235
, 236, 237, 268, 1260, 205, 236, 237, 238, 269, 1261, 206, 237, 238, 239, 2
70, 1262, 207, 238, 239, 240, 271, 1263, 208, 239, 240, 241, 272, 1264, 209
, 240, 241, 242, 273, 1265, 210, 241, 242, 243, 274, 1266, 211, 242, 243, 2
44, 275, 1267, 212, 243, 244, 245, 276, 1268, 213, 244, 245, 246, 277, 1269
, 214, 245, 246, 247, 278, 1270, 215, 246, 247, 248, 279, 1271, 216, 247, 2
48, 249, 280, 1272, 217, 248, 249, 250, 281, 1273, 218, 249, 250, 251, 282,
 1274, 219, 250, 251, 252, 283, 1275, 220, 251, 252, 253, 284, 1276, 221, 2
52, 253, 254, 285, 1277, 222, 253, 254, 255, 286, 1278, 223, 254, 255, 256,
 287, 1279, 224, 225, 255, 256, 288, 1280, 225, 257, 258, 288, 289, 1281, 2
26, 257, 258, 259, 290, 1282, 227, 258, 259, 260, 291, 1283, 228, 259, 260,
 261, 292, 1284, 229, 260, 261, 262, 293, 1285, 230, 261, 262, 263, 294, 12
86, 231, 262, 263, 264, 295, 1287, 232, 263, 264, 265, 296, 1288, 233, 264,
 265, 266, 297, 1289, 234, 265, 266, 267, 298, 1290, 235, 266, 267, 268, 29
9, 1291, 236, 267, 268, 269, 300, 1292, 237, 268, 269, 270, 301, 1293, 238,
 269, 270, 271, 302, 1294, 239, 270, 271, 272, 303, 1295, 240, 271, 272, 27
3, 304, 1296, 241, 272, 273, 274, 305, 1297, 242, 273, 274, 275, 306, 1298,
 243, 274, 275, 276, 307, 1299, 244, 275, 276, 277, 308, 1300, 245, 276, 27
7, 278, 309, 1301, 246, 277, 278, 279, 310, 1302, 247, 278, 279, 280, 311, 
1303, 248, 279, 280, 281, 312, 1304, 249, 280, 281, 282, 313, 1305, 250, 28
1, 282, 283, 314, 1306, 251, 282, 283, 284, 315, 1307, 252, 283, 284, 285, 
316, 1308, 253, 284, 285, 286, 317, 1309, 254, 285, 286, 287, 318, 1310, 25
5, 286, 287, 288, 319, 1311, 256, 257, 287, 288, 320, 1312, 257, 289, 290, 
320, 321, 1313, 258, 289, 290, 291, 322, 1314, 259, 290, 291, 292, 323, 131
5, 260, 291, 292, 293, 324, 1316, 261, 292, 293, 294, 325, 1317, 262, 293, 
294, 295, 326, 1318, 263, 294, 295, 296, 327, 1319, 264, 295, 296, 297, 328
, 1320, 265, 296, 297, 298, 329, 1321, 266, 297, 298, 299, 330, 1322, 267, 
298, 299, 300, 331, 1323, 268, 299, 300, 301, 332, 1324, 269, 300, 301, 302
, 333, 1325, 270, 301, 302, 303, 334, 1326, 271, 302, 303, 304, 335, 1327, 
272, 303, 304, 305, 336, 1328, 273, 304, 305, 306, 337, 1329, 274, 305, 306
, 307, 338, 1330, 275, 306, 307, 308, 339, 1331, 276, 307, 308, 309, 340, 1
332, 277, 308, 309, 310, 341, 1333, 278, 309, 310, 311, 342, 1334, 279, 310
, 311, 312, 343, 1335, 280, 311, 312, 313, 344, 1336, 281, 312, 313, 314, 3
45, 1337, 282, 313, 314, 315, 346, 1338, 283, 314, 315, 316, 347, 1339, 284
, 315, 316, 317, 348, 1340, 285, 316, 317, 318, 349, 1341, 286, 317, 318, 3
19, 350, 1342, 287, 318, 319, 320, 351, 1343, 288, 289, 319, 320, 352, 1344
, 289, 321, 322, 352, 353, 1345, 290, 321, 322, 323, 354, 1346, 291, 322, 3
23, 324, 355, 1347, 292, 323, 324, 325, 356, 1348, 293, 324, 325, 326, 357,
 1349, 294, 325, 326, 327, 358, 1350, 295, 326, 327, 328, 359, 1351, 296, 3
27, 328, 329, 360, 1352, 297, 328, 329, 330, 361, 1353, 298, 329, 330, 331,
 362, 1354, 299, 330, 331, 332, 363, 1355, 300, 331, 332, 333, 364, 1356, 3
01, 332, 333, 334, 365, 1357, 302, 333, 334, 335, 366, 1358, 303, 334, 335,
 336, 367, 1359, 304, 335, 336, 337, 368, 1360, 305, 336, 337, 338, 369, 13
61, 306, 337, 338, 339, 370, 1362, 307, 338, 339, 340, 371, 1363, 308, 339,
 340, 341, 372, 1364, 309, 340, 341, 342, 373, 1365, 310, 341, 342, 343, 37
4, 1366, 311, 342, 343, 344, 375, 1367, 312, 343, 344, 345, 376, 1368, 313,
 344, 345, 346, 377, 1369, 314, 345, 346, 347, 378, 1370, 315, 346, 347, 34
8, 379, 1371, 316, 347, 348, 349, 380, 1372, 317, 348, 349, 350, 381, 1373,
 318, 349, 350, 351, 382, 1374, 319, 350, 351, 352, 383, 1375, 320, 321, 35
1, 352, 384, 1376, 321, 353, 354, 384, 385, 1377, 322, 353, 354, 355, 386, 
1378, 323, 354, 355, 356, 387, 1379, 324, 355, 356, 357, 388, 1380, 325, 35
6, 357, 358, 389, 1381, 326, 357, 358, 359, 390, 1382, 327, 358, 359, 360, 
391, 1383, 328, 359, 360, 361, 392, 1384, 329, 360, 361, 362, 393, 1385, 33
0, 361, 362, 363, 394, 1386, 331, 362, 363, 364, 395, 1387, 332, 363, 364, 
365, 396, 1388, 333, 364, 365, 366, 397, 1389, 334, 365, 366, 367, 398, 139
0, 335, 366, 367, 368, 399, 1391, 336, 367, 368, 369, 400, 1392, 337, 368, 
369, 370, 401, 1393, 338, 369, 370, 371, 402, 1394, 339, 370, 371, 372, 403
, 1395, 340, 371, 372, 373, 404, 1396, 341, 372, 373, 374, 405, 1397, 342, 
373, 374, 375, 406, 1398, 343, 374, 375, 376, 407, 1399, 344, 375, 376, 377
, 408, 1400, 345, 376, 377, 378, 409, 1401, 346, 377, 378, 379, 410, 1402, 
347, 378, 379, 380, 411, 1403, 348, 379, 380, 381, 412, 1404, 349, 380, 381
, 382, 413, 1405, 350, 381, 382, 383, 414, 1406, 351, 382, 383, 384, 415, 1
407, 352, 353, 383, 384, 416, 1408, 353, 385, 386, 416, 417, 1409, 354, 385
, 386, 387, 418, 1410, 355, 386, 387, 388, 419, 1411, 356, 387, 388, 389, 4
20, 1412, 357, 388, 389, 390, 421, 1413, 358, 389, 390, 391, 422, 1414, 359
, 390, 391, 392, 423, 1415, 360, 391, 392, 393, 424, 1416, 361, 392, 393, 3
94, 425, 1417, 362, 393, 394, 395, 426, 1418, 363, 394, 395, 396, 427, 1419
, 364, 395, 396, 397, 428, 1420, 365, 396, 397, 398, 429, 1421, 366, 397, 3
98, 399, 430, 1422, 367, 398, 399, 400, 431, 1423, 368, 399, 400, 401, 432,
 1424, 369, 400, 401, 402, 433, 1425, 370, 401, 402, 403, 434, 1426, 371, 4
02, 403, 404, 435, 1427, 372, 403, 404, 405, 436, 1428, 373, 404, 405, 406,
 437, 1429, 374, 405, 406, 407, 438, 1430, 375, 406, 407, 408, 439, 1431, 3
76, 407, 408, 409, 440, 1432, 377, 408, 409, 410, 441, 1433, 378, 409, 410,
 411, 442, 1434, 379, 410, 411, 412, 443, 1435, 380, 411, 412, 413, 444, 14
36, 381, 412, 413, 414, 445, 1437, 382, 413, 414, 415, 446, 1438, 383, 414,
 415, 416, 447, 1439, 384, 385, 415, 416, 448, 1440, 385, 417, 418, 448, 44
9, 1441, 386, 417, 418, 419, 450, 1442, 387, 418, 419, 420, 451, 1443, 388,
 419, 420, 421, 452, 1444, 389, 420, 421, 422, 453, 1445, 390, 421, 422, 42
3, 454, 1446, 391, 422, 423, 424, 455, 1447, 392, 423, 424, 425, 456, 1448,
 393, 424, 425, 426, 457, 1449, 394, 425, 426, 427, 458, 1450, 395, 426, 42
7, 428, 459, 1451, 396, 427, 428, 429, 460, 1452, 397, 428, 429, 430, 461, 
1453, 398, 429, 430, 431, 462, 1454, 399, 430, 431, 432, 463, 1455, 400, 43
1, 432, 433, 464, 1456, 401, 432, 433, 434, 465, 1457, 402, 433, 434, 435, 
466, 1458, 403, 434, 435, 436, 467, 1459, 404, 435, 436, 437, 468, 1460, 40
5, 436, 437, 438, 469, 1461, 406, 437, 438, 439, 470, 1462, 407, 438, 439, 
440, 471, 1463, 408, 439, 440, 441, 472, 1464, 409, 440, 441, 442, 473, 146
5, 410, 441, 442, 443, 474, 1466, 411, 442, 443, 444, 475, 1467, 412, 443, 
444, 445, 476, 1468, 413, 444, 445, 446, 477, 1469, 414, 445, 446, 447, 478
, 1470, 415, 446, 447, 448, 479, 1471, 416, 417, 447, 448, 480, 1472, 417, 
449, 450, 480, 481, 1473, 418, 449, 450, 451, 482, 1474, 419, 450, 451, 452
, 483, 1475, 420, 451, 452, 453, 484, 1476, 421, 452, 453, 454, 485, 1477, 
422, 453, 454, 455, 486, 1478, 423, 454, 455, 456, 487, 1479, 424, 455, 456
, 457, 488, 1480, 425, 456, 457, 458, 489, 1481, 426, 457, 458, 459, 490, 1
482, 427, 458, 459, 460, 491, 1483, 428, 459, 460, 461, 492, 1484, 429, 460
, 461, 462, 493, 1485, 430, 461, 462, 463, 494, 1486, 431, 462, 463, 464, 4
95, 1487, 432, 463, 464, 465, 496, 1488, 433, 464, 465, 466, 497, 1489, 434
, 465, 466, 467, 498, 1490, 435, 466, 467, 468, 499, 1491, 436, 467, 468, 4
69, 500, 1492, 437, 468, 469, 470, 501, 1493, 438, 469, 470, 471, 502, 1494
, 439, 470, 471, 472, 503, 1495, 440, 471, 472, 473, 504, 1496, 441, 472, 4
73, 474, 505, 1497, 442, 473, 474, 475, 506, 1498, 443, 474, 475, 476, 507,
 1499, 444, 475, 476, 477, 508, 1500, 445, 476, 477, 478, 509, 1501, 446, 4
77, 478, 479, 510, 1502, 447, 478, 479, 480, 511, 1503, 448, 449, 479, 480,
 512, 1504, 449, 481, 482, 512, 513, 1505, 450, 481, 482, 483, 514, 1506, 4
51, 482, 483, 484, 515, 1507, 452, 483, 484, 485, 516, 1508, 453, 484, 485,
 486, 517, 1509, 454, 485, 486, 487, 518, 1510, 455, 486, 487, 488, 519, 15
11, 456, 487, 488, 489, 520, 1512, 457, 488, 489, 490, 521, 1513, 458, 489,
 490, 491, 522, 1514, 459, 490, 491, 492, 523, 1515, 460, 491, 492, 493, 52
4, 1516, 461, 492, 493, 494, 525, 1517, 462, 493, 494, 495, 526, 1518, 463,
 494, 495, 496, 527, 1519, 464, 495, 496, 497, 528, 1520, 465, 496, 497, 49
8, 529, 1521, 466, 497, 498, 499, 530, 1522, 467, 498, 499, 500, 531, 1523,
 468, 499, 500, 501, 532, 1524, 469, 500, 501, 502, 533, 1525, 470, 501, 50
2, 503, 534, 1526, 471, 502, 503, 504, 535, 1527, 472, 503, 504, 505, 536, 
1528, 473, 504, 505, 506, 537, 1529, 474, 505, 506, 507, 538, 1530, 475, 50
6, 507, 508, 539, 1531, 476, 507, 508, 509, 540, 1532, 477, 508, 509, 510, 
541, 1533, 478, 509, 510, 511, 542, 1534, 479, 510, 511, 512, 543, 1535, 48
0, 481, 511, 512, 544, 1536, 481, 513, 514, 544, 545, 1537, 482, 513, 514, 
515, 546, 1538, 483, 514, 515, 516, 547, 1539, 484, 515, 516, 517, 548, 154
0, 485, 516, 517, 518, 549, 1541, 486, 517, 518, 519, 550, 1542, 487, 518, 
519, 520, 551, 1543, 488, 519, 520, 521, 552, 1544, 489, 520, 521, 522, 553
, 1545, 490, 521, 522, 523, 554, 1546, 491, 522, 523, 524, 555, 1547, 492, 
523, 524, 525, 556, 1548, 493, 524, 525, 526, 557, 1549, 494, 525, 526, 527
, 558, 1550, 495, 526, 527, 528, 559, 1551, 496, 527, 528, 529, 560, 1552, 
497, 528, 529, 530, 561, 1553, 498, 529, 530, 531, 562, 1554, 499, 530, 531
, 532, 563, 1555, 500, 531, 532, 533, 564, 1556, 501, 532, 533, 534, 565, 1
557, 502, 533, 534, 535, 566, 1558, 503, 534, 535, 536, 567, 1559, 504, 535
, 536, 537, 568, 1560, 505, 536, 537, 538, 569, 1561, 506, 537, 538, 539, 5
70, 1562, 507, 538, 539, 540, 571, 1563, 508, 539, 540, 541, 572, 1564, 509
, 540, 541, 542, 573, 1565, 510, 541, 542, 543, 574, 1566, 511, 542, 543, 5
44, 575, 1567, 512, 513, 543, 544, 576, 1568, 513, 545, 546, 576, 577, 1569
, 514, 545, 546, 547, 578, 1570, 515, 546, 547, 548, 579, 1571, 516, 547, 5
48, 549, 580, 1572, 517, 548, 549, 550, 581, 1573, 518, 549, 550, 551, 582,
 1574, 519, 550, 551, 552, 583, 1575, 520, 551, 552, 553, 584, 1576, 521, 5
52, 553, 554, 585, 1577, 522, 553, 554, 555, 586, 1578, 523, 554, 555, 556,
 587, 1579, 524, 555, 556, 557, 588, 1580, 525, 556, 557, 558, 589, 1581, 5
26, 557, 558, 559, 590, 1582, 527, 558, 559, 560, 591, 1583, 528, 559, 560,
 561, 592, 1584, 529, 560, 561, 562, 593, 1585, 530, 561, 562, 563, 594, 15
86, 531, 562, 563, 564, 595, 1587, 532, 563, 564, 565, 596, 1588, 533, 564,
 565, 566, 597, 1589, 534, 565, 566, 567, 598, 1590, 535, 566, 567, 568, 59
9, 1591, 536, 567, 568, 569, 600, 1592, 537, 568, 569, 570, 601, 1593, 538,
 569, 570, 571, 602, 1594, 539, 570, 571, 572, 603, 1595, 540, 571, 572, 57
3, 604, 1596, 541, 572, 573, 574, 605, 1597, 542, 573, 574, 575, 606, 1598,
 543, 574, 575, 576, 607, 1599, 544, 545, 575, 576, 608, 1600, 545, 577, 57
8, 608, 609, 1601, 546, 577, 578, 579, 610, 1602, 547, 578, 579, 580, 611, 
1603, 548, 579, 580, 581, 612, 1604, 549, 580, 581, 582, 613, 1605, 550, 58
1, 582, 583, 614, 1606, 551, 582, 583, 584, 615, 1607, 552, 583, 584, 585, 
616, 1608, 553, 584, 585, 586, 617, 1609, 554, 585, 586, 587, 618, 1610, 55
5, 586, 587, 588, 619, 1611, 556, 587, 588, 589, 620, 1612, 557, 588, 589, 
590, 621, 1613, 558, 589, 590, 591, 622, 1614, 559, 590, 591, 592, 623, 161
5, 560, 591, 592, 593, 624, 1616, 561, 592, 593, 594, 625, 1617, 562, 593, 
594, 595, 626, 1618, 563, 594, 595, 596, 627, 1619, 564, 595, 596, 597, 628
, 1620, 565, 596, 597, 598, 629, 1621, 566, 597, 598, 599, 630, 1622, 567, 
598, 599, 600, 631, 1623, 568, 599, 600, 601, 632, 1624, 569, 600, 601, 602
, 633, 1625, 570, 601, 602, 603, 634, 1626, 571, 602, 603, 604, 635, 1627, 
572, 603, 604, 605, 636, 1628, 573, 604, 605, 606, 637, 1629, 574, 605, 606
, 607, 638, 1630, 575, 606, 607, 608, 639, 1631, 576, 577, 607, 608, 640, 1
632, 577, 609, 610, 640, 641, 1633, 578, 609, 610, 611, 642, 1634, 579, 610
, 611, 612, 643, 1635, 580, 611, 612, 613, 644, 1636, 581, 612, 613, 614, 6
45, 1637, 582, 613, 614, 615, 646, 1638, 583, 614, 615, 616, 647, 1639, 584
, 615, 616, 617, 648, 1640, 585, 616, 617, 618, 649, 1641, 586, 617, 618, 6
19, 650, 1642, 587, 618, 619, 620, 651, 1643, 588, 619, 620, 621, 652, 1644
, 589, 620, 621, 622, 653, 1645, 590, 621, 622, 623, 654, 1646, 591, 622, 6
23, 624, 655, 1647, 592, 623, 624, 625, 656, 1648, 593, 624, 625, 626, 657,
 1649, 594, 625, 626, 627, 658, 1650, 595, 626, 627, 628, 659, 1651, 596, 6
27, 628, 629, 660, 1652, 597, 628, 629, 630, 661, 1653, 598, 629, 630, 631,
 662, 1654, 599, 630, 631, 632, 663, 1655, 600, 631, 632, 633, 664, 1656, 6
01, 632, 633, 634, 665, 1657, 602, 633, 634, 635, 666, 1658, 603, 634, 635,
 636, 667, 1659, 604, 635, 636, 637, 668, 1660, 605, 636, 637, 638, 669, 16
61, 606, 637, 638, 639, 670, 1662, 607, 638, 639, 640, 671, 1663, 608, 609,
 639, 640, 672, 1664, 609, 641, 642, 672, 673, 1665, 610, 641, 642, 643, 67
4, 1666, 611, 642, 643, 644, 675, 1667, 612, 643, 644, 645, 676, 1668, 613,
 644, 645, 646, 677, 1669, 614, 645, 646, 647, 678, 1670, 615, 646, 647, 64
8, 679, 1671, 616, 647, 648, 649, 680, 1672, 617, 648, 649, 650, 681, 1673,
 618, 649, 650, 651, 682, 1674, 619, 650, 651, 652, 683, 1675, 620, 651, 65
2, 653, 684, 1676, 621, 652, 653, 654, 685, 1677, 622, 653, 654, 655, 686, 
1678, 623, 654, 655, 656, 687, 1679, 624, 655, 656, 657, 688, 1680, 625, 65
6, 657, 658, 689, 1681, 626, 657, 658, 659, 690, 1682, 627, 658, 659, 660, 
691, 1683, 628, 659, 660, 661, 692, 1684, 629, 660, 661, 662, 693, 1685, 63
0, 661, 662, 663, 694, 1686, 631, 662, 663, 664, 695, 1687, 632, 663, 664, 
665, 696, 1688, 633, 664, 665, 666, 697, 1689, 634, 665, 666, 667, 698, 169
0, 635, 666, 667, 668, 699, 1691, 636, 667, 668, 669, 700, 1692, 637, 668, 
669, 670, 701, 1693, 638, 669, 670, 671, 702, 1694, 639, 670, 671, 672, 703
, 1695, 640, 641, 671, 672, 704, 1696, 641, 673, 674, 704, 705, 1697, 642, 
673, 674, 675, 706, 1698, 643, 674, 675, 676, 707, 1699, 644, 675, 676, 677
, 708, 1700, 645, 676, 677, 678, 709, 1701, 646, 677, 678, 679, 710, 1702, 
647, 678, 679, 680, 711, 1703, 648, 679, 680, 681, 712, 1704, 649, 680, 681
, 682, 713, 1705, 650, 681, 682, 683, 714, 1706, 651, 682, 683, 684, 715, 1
707, 652, 683, 684, 685, 716, 1708, 653, 684, 685, 686, 717, 1709, 654, 685
, 686, 687, 718, 1710, 655, 686, 687, 688, 719, 1711, 656, 687, 688, 689, 7
20, 1712, 657, 688, 689, 690, 721, 1713, 658, 689, 690, 691, 722, 1714, 659
, 690, 691, 692, 723, 1715, 660, 691, 692, 693, 724, 1716, 661, 692, 693, 6
94, 725, 1717, 662, 693, 694, 695, 726, 1718, 663, 694, 695, 696, 727, 1719
, 664, 695, 696, 697, 728, 1720, 665, 696, 697, 698, 729, 1721, 666, 697, 6
98, 699, 730, 1722, 667, 698, 699, 700, 731, 1723, 668, 699, 700, 701, 732,
 1724, 669, 700, 701, 702, 733, 1725, 670, 701, 702, 703, 734, 1726, 671, 7
02, 703, 704, 735, 1727, 672, 673, 703, 704, 736, 1728, 673, 705, 706, 736,
 737, 1729, 674, 705, 706, 707, 738, 1730, 675, 706, 707, 708, 739, 1731, 6
76, 707, 708, 709, 740, 1732, 677, 708, 709, 710, 741, 1733, 678, 709, 710,
 711, 742, 1734, 679, 710, 711, 712, 743, 1735, 680, 711, 712, 713, 744, 17
36, 681, 712, 713, 714, 745, 1737, 682, 713, 714, 715, 746, 1738, 683, 714,
 715, 716, 747, 1739, 684, 715, 716, 717, 748, 1740, 685, 716, 717, 718, 74
9, 1741, 686, 717, 718, 719, 750, 1742, 687, 718, 719, 720, 751, 1743, 688,
 719, 720, 721, 752, 1744, 689, 720, 721, 722, 753, 1745, 690, 721, 722, 72
3, 754, 1746, 691, 722, 723, 724, 755, 1747, 692, 723, 724, 725, 756, 1748,
 693, 724, 725, 726, 757, 1749, 694, 725, 726, 727, 758, 1750, 695, 726, 72
7, 728, 759, 1751, 696, 727, 728, 729, 760, 1752, 697, 728, 729, 730, 761, 
1753, 698, 729, 730, 731, 762, 1754, 699, 730, 731, 732, 763, 1755, 700, 73
1, 732, 733, 764, 1756, 701, 732, 733, 734, 765, 1757, 702, 733, 734, 735, 
766, 1758, 703, 734, 735, 736, 767, 1759, 704, 705, 735, 736, 768, 1760, 70
5, 737, 738, 768, 769, 1761, 706, 737, 738, 739, 770, 1762, 707, 738, 739, 
740, 771, 1763, 708, 739, 740, 741, 772, 1764, 709, 740, 741, 742, 773, 176
5, 710, 741, 742, 743, 774, 1766, 711, 742, 743, 744, 775, 1767, 712, 743, 
744, 745, 776, 1768, 713, 744, 745, 746, 777, 1769, 714, 745, 746, 747, 778
, 1770, 715, 746, 747, 748, 779, 1771, 716, 747, 748, 749, 780, 1772, 717, 
748, 749, 750, 781, 1773, 718, 749, 750, 751, 782, 1774, 719, 750, 751, 752
, 783, 1775, 720, 751, 752, 753, 784, 1776, 721, 752, 753, 754, 785, 1777, 
722, 753, 754, 755, 786, 1778, 723, 754, 755, 756, 787, 1779, 724, 755, 756
, 757, 788, 1780, 725, 756, 757, 758, 789, 1781, 726, 757, 758, 759, 790, 1
782, 727, 758, 759, 760, 791, 1783, 728, 759, 760, 761, 792, 1784, 729, 760
, 761, 762, 793, 1785, 730, 761, 762, 763, 794, 1786, 731, 762, 763, 764, 7
95, 1787, 732, 763, 764, 765, 796, 1788, 733, 764, 765, 766, 797, 1789, 734
, 765, 766, 767, 798, 1790, 735, 766, 767, 768, 799, 1791, 736, 737, 767, 7
68, 800, 1792, 737, 769, 770, 800, 801, 1793, 738, 769, 770, 771, 802, 1794
, 739, 770, 771, 772, 803, 1795, 740, 771, 772, 773, 804, 1796, 741, 772, 7
73, 774, 805, 1797, 742, 773, 774, 775, 806, 1798, 743, 774, 775, 776, 807,
 1799, 744, 775, 776, 777, 808, 1800, 745, 776, 777, 778, 809, 1801, 746, 7
77, 778, 779, 810, 1802, 747, 778, 779, 780, 811, 1803, 748, 779, 780, 781,
 812, 1804, 749, 780, 781, 782, 813, 1805, 750, 781, 782, 783, 814, 1806, 7
51, 782, 783, 784, 815, 1807, 752, 783, 784, 785, 816, 1808, 753, 784, 785,
 786, 817, 1809, 754, 785, 786, 787, 818, 1810, 755, 786, 787, 788, 819, 18
11, 756, 787, 788, 789, 820, 1812, 757, 788, 789, 790, 821, 1813, 758, 789,
 790, 791, 822, 1814, 759, 790, 791, 792, 823, 1815, 760, 791, 792, 793, 82
4, 1816, 761, 792, 793, 794, 825, 1817, 762, 793, 794, 795, 826, 1818, 763,
 794, 795, 796, 827, 1819, 764, 795, 796, 797, 828, 1820, 765, 796, 797, 79
8, 829, 1821, 766, 797, 798, 799, 830, 1822, 767, 798, 799, 800, 831, 1823,
 768, 769, 799, 800, 832, 1824, 769, 801, 802, 832, 833, 1825, 770, 801, 80
2, 803, 834, 1826, 771, 802, 803, 804, 835, 1827, 772, 803, 804, 805, 836, 
1828, 773, 804, 805, 806, 837, 1829, 774, 805, 806, 807, 838, 1830, 775, 80
6, 807, 808, 839, 1831, 776, 807, 808, 809, 840, 1832, 777, 808, 809, 810, 
841, 1833, 778, 809, 810, 811, 842, 1834, 779, 810, 811, 812, 843, 1835, 78
0, 811, 812, 813, 844, 1836, 781, 812, 813, 814, 845, 1837, 782, 813, 814, 
815, 846, 1838, 783, 814, 815, 816, 847, 1839, 784, 815, 816, 817, 848, 184
0, 785, 816, 817, 818, 849, 1841, 786, 817, 818, 819, 850, 1842, 787, 818, 
819, 820, 851, 1843, 788, 819, 820, 821, 852, 1844, 789, 820, 821, 822, 853
, 1845, 790, 821, 822, 823, 854, 1846, 791, 822, 823, 824, 855, 1847, 792, 
823, 824, 825, 856, 1848, 793, 824, 825, 826, 857, 1849, 794, 825, 826, 827
, 858, 1850, 795, 826, 827, 828, 859, 1851, 796, 827, 828, 829, 860, 1852, 
797, 828, 829, 830, 861, 1853, 798, 829, 830, 831, 862, 1854, 799, 830, 831
, 832, 863, 1855, 800, 801, 831, 832, 864, 1856, 801, 833, 834, 864, 865, 1
857, 802, 833, 834, 835, 866, 1858, 803, 834, 835, 836, 867, 1859, 804, 835
, 836, 837, 868, 1860, 805, 836, 837, 838, 869, 1861, 806, 837, 838, 839, 8
70, 1862, 807, 838, 839, 840, 871, 1863, 808, 839, 840, 841, 872, 1864, 809
, 840, 841, 842, 873, 1865, 810, 841, 842, 843, 874, 1866, 811, 842, 843, 8
44, 875, 1867, 812, 843, 844, 845, 876, 1868, 813, 844, 845, 846, 877, 1869
, 814, 845, 846, 847, 878, 1870, 815, 846, 847, 848, 879, 1871, 816, 847, 8
48, 849, 880, 1872, 817, 848, 849, 850, 881, 1873, 818, 849, 850, 851, 882,
 1874, 819, 850, 851, 852, 883, 1875, 820, 851, 852, 853, 884, 1876, 821, 8
52, 853, 854, 885, 1877, 822, 853, 854, 855, 886, 1878, 823, 854, 855, 856,
 887, 1879, 824, 855, 856, 857, 888, 1880, 825, 856, 857, 858, 889, 1881, 8
26, 857, 858, 859, 890, 1882, 827, 858, 859, 860, 891, 1883, 828, 859, 860,
 861, 892, 1884, 829, 860, 861, 862, 893, 1885, 830, 861, 862, 863, 894, 18
86, 831, 862, 863, 864, 895, 1887, 832, 833, 863, 864, 896, 1888, 833, 865,
 866, 896, 897, 1889, 834, 865, 866, 867, 898, 1890, 835, 866, 867, 868, 89
9, 1891, 836, 867, 868, 869, 900, 1892, 837, 868, 869, 870, 901, 1893, 838,
 869, 870, 871, 902, 1894, 839, 870, 871, 872, 903, 1895, 840, 871, 872, 87
3, 904, 1896, 841, 872, 873, 874, 905, 1897, 842, 873, 874, 875, 906, 1898,
 843, 874, 875, 876, 907, 1899, 844, 875, 876, 877, 908, 1900, 845, 876, 87
7, 878, 909, 1901, 846, 877, 878, 879, 910, 1902, 847, 878, 879, 880, 911, 
1903, 848, 879, 880, 881, 912, 1904, 849, 880, 881, 882, 913, 1905, 850, 88
1, 882, 883, 914, 1906, 851, 882, 883, 884, 915, 1907, 852, 883, 884, 885, 
916, 1908, 853, 884, 885, 886, 917, 1909, 854, 885, 886, 887, 918, 1910, 85
5, 886, 887, 888, 919, 1911, 856, 887, 888, 889, 920, 1912, 857, 888, 889, 
890, 921, 1913, 858, 889, 890, 891, 922, 1914, 859, 890, 891, 892, 923, 191
5, 860, 891, 892, 893, 924, 1916, 861, 892, 893, 894, 925, 1917, 862, 893, 
894, 895, 926, 1918, 863, 894, 895, 896, 927, 1919, 864, 865, 895, 896, 928
, 1920, 865, 897, 898, 928, 929, 1921, 866, 897, 898, 899, 930, 1922, 867, 
898, 899, 900, 931, 1923, 868, 899, 900, 901, 932, 1924, 869, 900, 901, 902
, 933, 1925, 870, 901, 902, 903, 934, 1926, 871, 902, 903, 904, 935, 1927, 
872, 903, 904, 905, 936, 1928, 873, 904, 905, 906, 937, 1929, 874, 905, 906
, 907, 938, 1930, 875, 906, 907, 908, 939, 1931, 876, 907, 908, 909, 940, 1
932, 877, 908, 909, 910, 941, 1933, 878, 909, 910, 911, 942, 1934, 879, 910
, 911, 912, 943, 1935, 880, 911, 912, 913, 944, 1936, 881, 912, 913, 914, 9
45, 1937, 882, 913, 914, 915, 946, 1938, 883, 914, 915, 916, 947, 1939, 884
, 915, 916, 917, 948, 1940, 885, 916, 917, 918, 949, 1941, 886, 917, 918, 9
19, 950, 1942, 887, 918, 919, 920, 951, 1943, 888, 919, 920, 921, 952, 1944
, 889, 920, 921, 922, 953, 1945, 890, 921, 922, 923, 954, 1946, 891, 922, 9
23, 924, 955, 1947, 892, 923, 924, 925, 956, 1948, 893, 924, 925, 926, 957,
 1949, 894, 925, 926, 927, 958, 1950, 895, 926, 927, 928, 959, 1951, 896, 8
97, 927, 928, 960, 1952, 897, 929, 930, 960, 961, 1953, 898, 929, 930, 931,
 962, 1954, 899, 930, 931, 932, 963, 1955, 900, 931, 932, 933, 964, 1956, 9
01, 932, 933, 934, 965, 1957, 902, 933, 934, 935, 966, 1958, 903, 934, 935,
 936, 967, 1959, 904, 935, 936, 937, 968, 1960, 905, 936, 937, 938, 969, 19
61, 906, 937, 938, 939, 970, 1962, 907, 938, 939, 940, 971, 1963, 908, 939,
 940, 941, 972, 1964, 909, 940, 941, 942, 973, 1965, 910, 941, 942, 943, 97
4, 1966, 911, 942, 943, 944, 975, 1967, 912, 943, 944, 945, 976, 1968, 913,
 944, 945, 946, 977, 1969, 914, 945, 946, 947, 978, 1970, 915, 946, 947, 94
8, 979, 1971, 916, 947, 948, 949, 980, 1972, 917, 948, 949, 950, 981, 1973,
 918, 949, 950, 951, 982, 1974, 919, 950, 951, 952, 983, 1975, 920, 951, 95
2, 953, 984, 1976, 921, 952, 953, 954, 985, 1977, 922, 953, 954, 955, 986, 
1978, 923, 954, 955, 956, 987, 1979, 924, 955, 956, 957, 988, 1980, 925, 95
6, 957, 958, 989, 1981, 926, 957, 958, 959, 990, 1982, 927, 958, 959, 960, 
991, 1983, 928, 929, 959, 960, 992, 1984, 929, 961, 962, 992, 993, 1985, 93
0, 961, 962, 963, 994, 1986, 931, 962, 963, 964, 995, 1987, 932, 963, 964, 
965, 996, 1988, 933, 964, 965, 966, 997, 1989, 934, 965, 966, 967, 998, 199
0, 935, 966, 967, 968, 999, 1991, 936, 967, 968, 969, 1000, 1992, 937, 968,
 969, 970, 1001, 1993, 938, 969, 970, 971, 1002, 1994, 939, 970, 971, 972, 
1003, 1995, 940, 971, 972, 973, 1004, 1996, 941, 972, 973, 974, 1005, 1997,
 942, 973, 974, 975, 1006, 1998, 943, 974, 975, 976, 1007, 1999, 944, 975, 
976, 977, 1008, 2000, 945, 976, 977, 978, 1009, 2001, 946, 977, 978, 979, 1
010, 2002, 947, 978, 979, 980, 1011, 2003, 948, 979, 980, 981, 1012, 2004, 
949, 980, 981, 982, 1013, 2005, 950, 981, 982, 983, 1014, 2006, 951, 982, 9
83, 984, 1015, 2007, 952, 983, 984, 985, 1016, 2008, 953, 984, 985, 986, 10
17, 2009, 954, 985, 986, 987, 1018, 2010, 955, 986, 987, 988, 1019, 2011, 9
56, 987, 988, 989, 1020, 2012, 957, 988, 989, 990, 1021, 2013, 958, 989, 99
0, 991, 1022, 2014, 959, 990, 991, 992, 1023, 2015, 960, 961, 991, 992, 102
4, 2016, 1, 961, 993, 994, 1024, 2017, 2, 962, 993, 994, 995, 2018, 3, 963,
 994, 995, 996, 2019, 4, 964, 995, 996, 997, 2020, 5, 965, 996, 997, 998, 2
021, 6, 966, 997, 998, 999, 2022, 7, 967, 998, 999, 1000, 2023, 8, 968, 999
, 1000, 1001, 2024, 9, 969, 1000, 1001, 1002, 2025, 10, 970, 1001, 1002, 10
03, 2026, 11, 971, 1002, 1003, 1004, 2027, 12, 972, 1003, 1004, 1005, 2028,
 13, 973, 1004, 1005, 1006, 2029, 14, 974, 1005, 1006, 1007, 2030, 15, 975,
 1006, 1007, 1008, 2031, 16, 976, 1007, 1008, 1009, 2032, 17, 977, 1008, 10
09, 1010, 2033, 18, 978, 1009, 1010, 1011, 2034, 19, 979, 1010, 1011, 1012,
 2035, 20, 980, 1011, 1012, 1013, 2036, 21, 981, 1012, 1013, 1014, 2037, 22
, 982, 1013, 1014, 1015, 2038, 23, 983, 1014, 1015, 1016, 2039, 24, 984, 10
15, 1016, 1017, 2040, 25, 985, 1016, 1017, 1018, 2041, 26, 986, 1017, 1018,
 1019, 2042, 27, 987, 1018, 1019, 1020, 2043, 28, 988, 1019, 1020, 1021, 20
44, 29, 989, 1020, 1021, 1022, 2045, 30, 990, 1021, 1022, 1023, 2046, 31, 9
91, 1022, 1023, 1024, 2047, 32, 992, 993, 1023, 1024, 2048, 1, 1025, 1026, 
1056, 1057, 2017, 2, 1025, 1026, 1027, 1058, 2018, 3, 1026, 1027, 1028, 105
9, 2019, 4, 1027, 1028, 1029, 1060, 2020, 5, 1028, 1029, 1030, 1061, 2021, 
6, 1029, 1030, 1031, 1062, 2022, 7, 1030, 1031, 1032, 1063, 2023, 8, 1031, 
1032, 1033, 1064, 2024, 9, 1032, 1033, 1034, 1065, 2025, 10, 1033, 1034, 10
35, 1066, 2026, 11, 1034, 1035, 1036, 1067, 2027, 12, 1035, 1036, 1037, 106
8, 2028, 13, 1036, 1037, 1038, 1069, 2029, 14, 1037, 1038, 1039, 1070, 2030
, 15, 1038, 1039, 1040, 1071, 2031, 16, 1039, 1040, 1041, 1072, 2032, 17, 1
040, 1041, 1042, 1073, 2033, 18, 1041, 1042, 1043, 1074, 2034, 19, 1042, 10
43, 1044, 1075, 2035, 20, 1043, 1044, 1045, 1076, 2036, 21, 1044, 1045, 104
6, 1077, 2037, 22, 1045, 1046, 1047, 1078, 2038, 23, 1046, 1047, 1048, 1079
, 2039, 24, 1047, 1048, 1049, 1080, 2040, 25, 1048, 1049, 1050, 1081, 2041,
 26, 1049, 1050, 1051, 1082, 2042, 27, 1050, 1051, 1052, 1083, 2043, 28, 10
51, 1052, 1053, 1084, 2044, 29, 1052, 1053, 1054, 1085, 2045, 30, 1053, 105
4, 1055, 1086, 2046, 31, 1054, 1055, 1056, 1087, 2047, 32, 1025, 1055, 1056
, 1088, 2048, 33, 1025, 1057, 1058, 1088, 1089, 34, 1026, 1057, 1058, 1059,
 1090, 35, 1027, 1058, 1059, 1060, 1091, 36, 1028, 1059, 1060, 1061, 1092, 
37, 1029, 1060, 1061, 1062, 1093, 38, 1030, 1061, 1062, 1063, 1094, 39, 103
1, 1062, 1063, 1064, 1095, 40, 1032, 1063, 1064, 1065, 1096, 41, 1033, 1064
, 1065, 1066, 1097, 42, 1034, 1065, 1066, 1067, 1098, 43, 1035, 1066, 1067,
 1068, 1099, 44, 1036, 1067, 1068, 1069, 1100, 45, 1037, 1068, 1069, 1070, 
1101, 46, 1038, 1069, 1070, 1071, 1102, 47, 1039, 1070, 1071, 1072, 1103, 4
8, 1040, 1071, 1072, 1073, 1104, 49, 1041, 1072, 1073, 1074, 1105, 50, 1042
, 1073, 1074, 1075, 1106, 51, 1043, 1074, 1075, 1076, 1107, 52, 1044, 1075,
 1076, 1077, 1108, 53, 1045, 1076, 1077, 1078, 1109, 54, 1046, 1077, 1078, 
1079, 1110, 55, 1047, 1078, 1079, 1080, 1111, 56, 1048, 1079, 1080, 1081, 1
112, 57, 1049, 1080, 1081, 1082, 1113, 58, 1050, 1081, 1082, 1083, 1114, 59
, 1051, 1082, 1083, 1084, 1115, 60, 1052, 1083, 1084, 1085, 1116, 61, 1053,
 1084, 1085, 1086, 1117, 62, 1054, 1085, 1086, 1087, 1118, 63, 1055, 1086, 
1087, 1088, 1119, 64, 1056, 1057, 1087, 1088, 1120, 65, 1057, 1089, 1090, 1
120, 1121, 66, 1058, 1089, 1090, 1091, 1122, 67, 1059, 1090, 1091, 1092, 11
23, 68, 1060, 1091, 1092, 1093, 1124, 69, 1061, 1092, 1093, 1094, 1125, 70,
 1062, 1093, 1094, 1095, 1126, 71, 1063, 1094, 1095, 1096, 1127, 72, 1064, 
1095, 1096, 1097, 1128, 73, 1065, 1096, 1097, 1098, 1129, 74, 1066, 1097, 1
098, 1099, 1130, 75, 1067, 1098, 1099, 1100, 1131, 76, 1068, 1099, 1100, 11
01, 1132, 77, 1069, 1100, 1101, 1102, 1133, 78, 1070, 1101, 1102, 1103, 113
4, 79, 1071, 1102, 1103, 1104, 1135, 80, 1072, 1103, 1104, 1105, 1136, 81, 
1073, 1104, 1105, 1106, 1137, 82, 1074, 1105, 1106, 1107, 1138, 83, 1075, 1
106, 1107, 1108, 1139, 84, 1076, 1107, 1108, 1109, 1140, 85, 1077, 1108, 11
09, 1110, 1141, 86, 1078, 1109, 1110, 1111, 1142, 87, 1079, 1110, 1111, 111
2, 1143, 88, 1080, 1111, 1112, 1113, 1144, 89, 1081, 1112, 1113, 1114, 1145
, 90, 1082, 1113, 1114, 1115, 1146, 91, 1083, 1114, 1115, 1116, 1147, 92, 1
084, 1115, 1116, 1117, 1148, 93, 1085, 1116, 1117, 1118, 1149, 94, 1086, 11
17, 1118, 1119, 1150, 95, 1087, 1118, 1119, 1120, 1151, 96, 1088, 1089, 111
9, 1120, 1152, 97, 1089, 1121, 1122, 1152, 1153, 98, 1090, 1121, 1122, 1123
, 1154, 99, 1091, 1122, 1123, 1124, 1155, 100, 1092, 1123, 1124, 1125, 1156
, 101, 1093, 1124, 1125, 1126, 1157, 102, 1094, 1125, 1126, 1127, 1158, 103
, 1095, 1126, 1127, 1128, 1159, 104, 1096, 1127, 1128, 1129, 1160, 105, 109
7, 1128, 1129, 1130, 1161, 106, 1098, 1129, 1130, 1131, 1162, 107, 1099, 11
30, 1131, 1132, 1163, 108, 1100, 1131, 1132, 1133, 1164, 109, 1101, 1132, 1
133, 1134, 1165, 110, 1102, 1133, 1134, 1135, 1166, 111, 1103, 1134, 1135, 
1136, 1167, 112, 1104, 1135, 1136, 1137, 1168, 113, 1105, 1136, 1137, 1138,
 1169, 114, 1106, 1137, 1138, 1139, 1170, 115, 1107, 1138, 1139, 1140, 1171
, 116, 1108, 1139, 1140, 1141, 1172, 117, 1109, 1140, 1141, 1142, 1173, 118
, 1110, 1141, 1142, 1143, 1174, 119, 1111, 1142, 1143, 1144, 1175, 120, 111
2, 1143, 1144, 1145, 1176, 121, 1113, 1144, 1145, 1146, 1177, 122, 1114, 11
45, 1146, 1147, 1178, 123, 1115, 1146, 1147, 1148, 1179, 124, 1116, 1147, 1
148, 1149, 1180, 125, 1117, 1148, 1149, 1150, 1181, 126, 1118, 1149, 1150, 
1151, 1182, 127, 1119, 1150, 1151, 1152, 1183, 128, 1120, 1121, 1151, 1152,
 1184, 129, 1121, 1153, 1154, 1184, 1185, 130, 1122, 1153, 1154, 1155, 1186
, 131, 1123, 1154, 1155, 1156, 1187, 132, 1124, 1155, 1156, 1157, 1188, 133
, 1125, 1156, 1157, 1158, 1189, 134, 1126, 1157, 1158, 1159, 1190, 135, 112
7, 1158, 1159, 1160, 1191, 136, 1128, 1159, 1160, 1161, 1192, 137, 1129, 11
60, 1161, 1162, 1193, 138, 1130, 1161, 1162, 1163, 1194, 139, 1131, 1162, 1
163, 1164, 1195, 140, 1132, 1163, 1164, 1165, 1196, 141, 1133, 1164, 1165, 
1166, 1197, 142, 1134, 1165, 1166, 1167, 1198, 143, 1135, 1166, 1167, 1168,
 1199, 144, 1136, 1167, 1168, 1169, 1200, 145, 1137, 1168, 1169, 1170, 1201
, 146, 1138, 1169, 1170, 1171, 1202, 147, 1139, 1170, 1171, 1172, 1203, 148
, 1140, 1171, 1172, 1173, 1204, 149, 1141, 1172, 1173, 1174, 1205, 150, 114
2, 1173, 1174, 1175, 1206, 151, 1143, 1174, 1175, 1176, 1207, 152, 1144, 11
75, 1176, 1177, 1208, 153, 1145, 1176, 1177, 1178, 1209, 154, 1146, 1177, 1
178, 1179, 1210, 155, 1147, 1178, 1179, 1180, 1211, 156, 1148, 1179, 1180, 
1181, 1212, 157, 1149, 1180, 1181, 1182, 1213, 158, 1150, 1181, 1182, 1183,
 1214, 159, 1151, 1182, 1183, 1184, 1215, 160, 1152, 1153, 1183, 1184, 1216
, 161, 1153, 1185, 1186, 1216, 1217, 162, 1154, 1185, 1186, 1187, 1218, 163
, 1155, 1186, 1187, 1188, 1219, 164, 1156, 1187, 1188, 1189, 1220, 165, 115
7, 1188, 1189, 1190, 1221, 166, 1158, 1189, 1190, 1191, 1222, 167, 1159, 11
90, 1191, 1192, 1223, 168, 1160, 1191, 1192, 1193, 1224, 169, 1161, 1192, 1
193, 1194, 1225, 170, 1162, 1193, 1194, 1195, 1226, 171, 1163, 1194, 1195, 
1196, 1227, 172, 1164, 1195, 1196, 1197, 1228, 173, 1165, 1196, 1197, 1198,
 1229, 174, 1166, 1197, 1198, 1199, 1230, 175, 1167, 1198, 1199, 1200, 1231
, 176, 1168, 1199, 1200, 1201, 1232, 177, 1169, 1200, 1201, 1202, 1233, 178
, 1170, 1201, 1202, 1203, 1234, 179, 1171, 1202, 1203, 1204, 1235, 180, 117
2, 1203, 1204, 1205, 1236, 181, 1173, 1204, 1205, 1206, 1237, 182, 1174, 12
05, 1206, 1207, 1238, 183, 1175, 1206, 1207, 1208, 1239, 184, 1176, 1207, 1
208, 1209, 1240, 185, 1177, 1208, 1209, 1210, 1241, 186, 1178, 1209, 1210, 
1211, 1242, 187, 1179, 1210, 1211, 1212, 1243, 188, 1180, 1211, 1212, 1213,
 1244, 189, 1181, 1212, 1213, 1214, 1245, 190, 1182, 1213, 1214, 1215, 1246
, 191, 1183, 1214, 1215, 1216, 1247, 192, 1184, 1185, 1215, 1216, 1248, 193
, 1185, 1217, 1218, 1248, 1249, 194, 1186, 1217, 1218, 1219, 1250, 195, 118
7, 1218, 1219, 1220, 1251, 196, 1188, 1219, 1220, 1221, 1252, 197, 1189, 12
20, 1221, 1222, 1253, 198, 1190, 1221, 1222, 1223, 1254, 199, 1191, 1222, 1
223, 1224, 1255, 200, 1192, 1223, 1224, 1225, 1256, 201, 1193, 1224, 1225, 
1226, 1257, 202, 1194, 1225, 1226, 1227, 1258, 203, 1195, 1226, 1227, 1228,
 1259, 204, 1196, 1227, 1228, 1229, 1260, 205, 1197, 1228, 1229, 1230, 1261
, 206, 1198, 1229, 1230, 1231, 1262, 207, 1199, 1230, 1231, 1232, 1263, 208
, 1200, 1231, 1232, 1233, 1264, 209, 1201, 1232, 1233, 1234, 1265, 210, 120
2, 1233, 1234, 1235, 1266, 211, 1203, 1234, 1235, 1236, 1267, 212, 1204, 12
35, 1236, 1237, 1268, 213, 1205, 1236, 1237, 1238, 1269, 214, 1206, 1237, 1
238, 1239, 1270, 215, 1207, 1238, 1239, 1240, 1271, 216, 1208, 1239, 1240, 
1241, 1272, 217, 1209, 1240, 1241, 1242, 1273, 218, 1210, 1241, 1242, 1243,
 1274, 219, 1211, 1242, 1243, 1244, 1275, 220, 1212, 1243, 1244, 1245, 1276
, 221, 1213, 1244, 1245, 1246, 1277, 222, 1214, 1245, 1246, 1247, 1278, 223
, 1215, 1246, 1247, 1248, 1279, 224, 1216, 1217, 1247, 1248, 1280, 225, 121
7, 1249, 1250, 1280, 1281, 226, 1218, 1249, 1250, 1251, 1282, 227, 1219, 12
50, 1251, 1252, 1283, 228, 1220, 1251, 1252, 1253, 1284, 229, 1221, 1252, 1
253, 1254, 1285, 230, 1222, 1253, 1254, 1255, 1286, 231, 1223, 1254, 1255, 
1256, 1287, 232, 1224, 1255, 1256, 1257, 1288, 233, 1225, 1256, 1257, 1258,
 1289, 234, 1226, 1257, 1258, 1259, 1290, 235, 1227, 1258, 1259, 1260, 1291
, 236, 1228, 1259, 1260, 1261, 1292, 237, 1229, 1260, 1261, 1262, 1293, 238
, 1230, 1261, 1262, 1263, 1294, 239, 1231, 1262, 1263, 1264, 1295, 240, 123
2, 1263, 1264, 1265, 1296, 241, 1233, 1264, 1265, 1266, 1297, 242, 1234, 12
65, 1266, 1267, 1298, 243, 1235, 1266, 1267, 1268, 1299, 244, 1236, 1267, 1
268, 1269, 1300, 245, 1237, 1268, 1269, 1270, 1301, 246, 1238, 1269, 1270, 
1271, 1302, 247, 1239, 1270, 1271, 1272, 1303, 248, 1240, 1271, 1272, 1273,
 1304, 249, 1241, 1272, 1273, 1274, 1305, 250, 1242, 1273, 1274, 1275, 1306
, 251, 1243, 1274, 1275, 1276, 1307, 252, 1244, 1275, 1276, 1277, 1308, 253
, 1245, 1276, 1277, 1278, 1309, 254, 1246, 1277, 1278, 1279, 1310, 255, 124
7, 1278, 1279, 1280, 1311, 256, 1248, 1249, 1279, 1280, 1312, 257, 1249, 12
81, 1282, 1312, 1313, 258, 1250, 1281, 1282, 1283, 1314, 259, 1251, 1282, 1
283, 1284, 1315, 260, 1252, 1283, 1284, 1285, 1316, 261, 1253, 1284, 1285, 
1286, 1317, 262, 1254, 1285, 1286, 1287, 1318, 263, 1255, 1286, 1287, 1288,
 1319, 264, 1256, 1287, 1288, 1289, 1320, 265, 1257, 1288, 1289, 1290, 1321
, 266, 1258, 1289, 1290, 1291, 1322, 267, 1259, 1290, 1291, 1292, 1323, 268
, 1260, 1291, 1292, 1293, 1324, 269, 1261, 1292, 1293, 1294, 1325, 270, 126
2, 1293, 1294, 1295, 1326, 271, 1263, 1294, 1295, 1296, 1327, 272, 1264, 12
95, 1296, 1297, 1328, 273, 1265, 1296, 1297, 1298, 1329, 274, 1266, 1297, 1
298, 1299, 1330, 275, 1267, 1298, 1299, 1300, 1331, 276, 1268, 1299, 1300, 
1301, 1332, 277, 1269, 1300, 1301, 1302, 1333, 278, 1270, 1301, 1302, 1303,
 1334, 279, 1271, 1302, 1303, 1304, 1335, 280, 1272, 1303, 1304, 1305, 1336
, 281, 1273, 1304, 1305, 1306, 1337, 282, 1274, 1305, 1306, 1307, 1338, 283
, 1275, 1306, 1307, 1308, 1339, 284, 1276, 1307, 1308, 1309, 1340, 285, 127
7, 1308, 1309, 1310, 1341, 286, 1278, 1309, 1310, 1311, 1342, 287, 1279, 13
10, 1311, 1312, 1343, 288, 1280, 1281, 1311, 1312, 1344, 289, 1281, 1313, 1
314, 1344, 1345, 290, 1282, 1313, 1314, 1315, 1346, 291, 1283, 1314, 1315, 
1316, 1347, 292, 1284, 1315, 1316, 1317, 1348, 293, 1285, 1316, 1317, 1318,
 1349, 294, 1286, 1317, 1318, 1319, 1350, 295, 1287, 1318, 1319, 1320, 1351
, 296, 1288, 1319, 1320, 1321, 1352, 297, 1289, 1320, 1321, 1322, 1353, 298
, 1290, 1321, 1322, 1323, 1354, 299, 1291, 1322, 1323, 1324, 1355, 300, 129
2, 1323, 1324, 1325, 1356, 301, 1293, 1324, 1325, 1326, 1357, 302, 1294, 13
25, 1326, 1327, 1358, 303, 1295, 1326, 1327, 1328, 1359, 304, 1296, 1327, 1
328, 1329, 1360, 305, 1297, 1328, 1329, 1330, 1361, 306, 1298, 1329, 1330, 
1331, 1362, 307, 1299, 1330, 1331, 1332, 1363, 308, 1300, 1331, 1332, 1333,
 1364, 309, 1301, 1332, 1333, 1334, 1365, 310, 1302, 1333, 1334, 1335, 1366
, 311, 1303, 1334, 1335, 1336, 1367, 312, 1304, 1335, 1336, 1337, 1368, 313
, 1305, 1336, 1337, 1338, 1369, 314, 1306, 1337, 1338, 1339, 1370, 315, 130
7, 1338, 1339, 1340, 1371, 316, 1308, 1339, 1340, 1341, 1372, 317, 1309, 13
40, 1341, 1342, 1373, 318, 1310, 1341, 1342, 1343, 1374, 319, 1311, 1342, 1
343, 1344, 1375, 320, 1312, 1313, 1343, 1344, 1376, 321, 1313, 1345, 1346, 
1376, 1377, 322, 1314, 1345, 1346, 1347, 1378, 323, 1315, 1346, 1347, 1348,
 1379, 324, 1316, 1347, 1348, 1349, 1380, 325, 1317, 1348, 1349, 1350, 1381
, 326, 1318, 1349, 1350, 1351, 1382, 327, 1319, 1350, 1351, 1352, 1383, 328
, 1320, 1351, 1352, 1353, 1384, 329, 1321, 1352, 1353, 1354, 1385, 330, 132
2, 1353, 1354, 1355, 1386, 331, 1323, 1354, 1355, 1356, 1387, 332, 1324, 13
55, 1356, 1357, 1388, 333, 1325, 1356, 1357, 1358, 1389, 334, 1326, 1357, 1
358, 1359, 1390, 335, 1327, 1358, 1359, 1360, 1391, 336, 1328, 1359, 1360, 
1361, 1392, 337, 1329, 1360, 1361, 1362, 1393, 338, 1330, 1361, 1362, 1363,
 1394, 339, 1331, 1362, 1363, 1364, 1395, 340, 1332, 1363, 1364, 1365, 1396
, 341, 1333, 1364, 1365, 1366, 1397, 342, 1334, 1365, 1366, 1367, 1398, 343
, 1335, 1366, 1367, 1368, 1399, 344, 1336, 1367, 1368, 1369, 1400, 345, 133
7, 1368, 1369, 1370, 1401, 346, 1338, 1369, 1370, 1371, 1402, 347, 1339, 13
70, 1371, 1372, 1403, 348, 1340, 1371, 1372, 1373, 1404, 349, 1341, 1372, 1
373, 1374, 1405, 350, 1342, 1373, 1374, 1375, 1406, 351, 1343, 1374, 1375, 
1376, 1407, 352, 1344, 1345, 1375, 1376, 1408, 353, 1345, 1377, 1378, 1408,
 1409, 354, 1346, 1377, 1378, 1379, 1410, 355, 1347, 1378, 1379, 1380, 1411
, 356, 1348, 1379, 1380, 1381, 1412, 357, 1349, 1380, 1381, 1382, 1413, 358
, 1350, 1381, 1382, 1383, 1414, 359, 1351, 1382, 1383, 1384, 1415, 360, 135
2, 1383, 1384, 1385, 1416, 361, 1353, 1384, 1385, 1386, 1417, 362, 1354, 13
85, 1386, 1387, 1418, 363, 1355, 1386, 1387, 1388, 1419, 364, 1356, 1387, 1
388, 1389, 1420, 365, 1357, 1388, 1389, 1390, 1421, 366, 1358, 1389, 1390, 
1391, 1422, 367, 1359, 1390, 1391, 1392, 1423, 368, 1360, 1391, 1392, 1393,
 1424, 369, 1361, 1392, 1393, 1394, 1425, 370, 1362, 1393, 1394, 1395, 1426
, 371, 1363, 1394, 1395, 1396, 1427, 372, 1364, 1395, 1396, 1397, 1428, 373
, 1365, 1396, 1397, 1398, 1429, 374, 1366, 1397, 1398, 1399, 1430, 375, 136
7, 1398, 1399, 1400, 1431, 376, 1368, 1399, 1400, 1401, 1432, 377, 1369, 14
00, 1401, 1402, 1433, 378, 1370, 1401, 1402, 1403, 1434, 379, 1371, 1402, 1
403, 1404, 1435, 380, 1372, 1403, 1404, 1405, 1436, 381, 1373, 1404, 1405, 
1406, 1437, 382, 1374, 1405, 1406, 1407, 1438, 383, 1375, 1406, 1407, 1408,
 1439, 384, 1376, 1377, 1407, 1408, 1440, 385, 1377, 1409, 1410, 1440, 1441
, 386, 1378, 1409, 1410, 1411, 1442, 387, 1379, 1410, 1411, 1412, 1443, 388
, 1380, 1411, 1412, 1413, 1444, 389, 1381, 1412, 1413, 1414, 1445, 390, 138
2, 1413, 1414, 1415, 1446, 391, 1383, 1414, 1415, 1416, 1447, 392, 1384, 14
15, 1416, 1417, 1448, 393, 1385, 1416, 1417, 1418, 1449, 394, 1386, 1417, 1
418, 1419, 1450, 395, 1387, 1418, 1419, 1420, 1451, 396, 1388, 1419, 1420, 
1421, 1452, 397, 1389, 1420, 1421, 1422, 1453, 398, 1390, 1421, 1422, 1423,
 1454, 399, 1391, 1422, 1423, 1424, 1455, 400, 1392, 1423, 1424, 1425, 1456
, 401, 1393, 1424, 1425, 1426, 1457, 402, 1394, 1425, 1426, 1427, 1458, 403
, 1395, 1426, 1427, 1428, 1459, 404, 1396, 1427, 1428, 1429, 1460, 405, 139
7, 1428, 1429, 1430, 1461, 406, 1398, 1429, 1430, 1431, 1462, 407, 1399, 14
30, 1431, 1432, 1463, 408, 1400, 1431, 1432, 1433, 1464, 409, 1401, 1432, 1
433, 1434, 1465, 410, 1402, 1433, 1434, 1435, 1466, 411, 1403, 1434, 1435, 
1436, 1467, 412, 1404, 1435, 1436, 1437, 1468, 413, 1405, 1436, 1437, 1438,
 1469, 414, 1406, 1437, 1438, 1439, 1470, 415, 1407, 1438, 1439, 1440, 1471
, 416, 1408, 1409, 1439, 1440, 1472, 417, 1409, 1441, 1442, 1472, 1473, 418
, 1410, 1441, 1442, 1443, 1474, 419, 1411, 1442, 1443, 1444, 1475, 420, 141
2, 1443, 1444, 1445, 1476, 421, 1413, 1444, 1445, 1446, 1477, 422, 1414, 14
45, 1446, 1447, 1478, 423, 1415, 1446, 1447, 1448, 1479, 424, 1416, 1447, 1
448, 1449, 1480, 425, 1417, 1448, 1449, 1450, 1481, 426, 1418, 1449, 1450, 
1451, 1482, 427, 1419, 1450, 1451, 1452, 1483, 428, 1420, 1451, 1452, 1453,
 1484, 429, 1421, 1452, 1453, 1454, 1485, 430, 1422, 1453, 1454, 1455, 1486
, 431, 1423, 1454, 1455, 1456, 1487, 432, 1424, 1455, 1456, 1457, 1488, 433
, 1425, 1456, 1457, 1458, 1489, 434, 1426, 1457, 1458, 1459, 1490, 435, 142
7, 1458, 1459, 1460, 1491, 436, 1428, 1459, 1460, 1461, 1492, 437, 1429, 14
60, 1461, 1462, 1493, 438, 1430, 1461, 1462, 1463, 1494, 439, 1431, 1462, 1
463, 1464, 1495, 440, 1432, 1463, 1464, 1465, 1496, 441, 1433, 1464, 1465, 
1466, 1497, 442, 1434, 1465, 1466, 1467, 1498, 443, 1435, 1466, 1467, 1468,
 1499, 444, 1436, 1467, 1468, 1469, 1500, 445, 1437, 1468, 1469, 1470, 1501
, 446, 1438, 1469, 1470, 1471, 1502, 447, 1439, 1470, 1471, 1472, 1503, 448
, 1440, 1441, 1471, 1472, 1504, 449, 1441, 1473, 1474, 1504, 1505, 450, 144
2, 1473, 1474, 1475, 1506, 451, 1443, 1474, 1475, 1476, 1507, 452, 1444, 14
75, 1476, 1477, 1508, 453, 1445, 1476, 1477, 1478, 1509, 454, 1446, 1477, 1
478, 1479, 1510, 455, 1447, 1478, 1479, 1480, 1511, 456, 1448, 1479, 1480, 
1481, 1512, 457, 1449, 1480, 1481, 1482, 1513, 458, 1450, 1481, 1482, 1483,
 1514, 459, 1451, 1482, 1483, 1484, 1515, 460, 1452, 1483, 1484, 1485, 1516
, 461, 1453, 1484, 1485, 1486, 1517, 462, 1454, 1485, 1486, 1487, 1518, 463
, 1455, 1486, 1487, 1488, 1519, 464, 1456, 1487, 1488, 1489, 1520, 465, 145
7, 1488, 1489, 1490, 1521, 466, 1458, 1489, 1490, 1491, 1522, 467, 1459, 14
90, 1491, 1492, 1523, 468, 1460, 1491, 1492, 1493, 1524, 469, 1461, 1492, 1
493, 1494, 1525, 470, 1462, 1493, 1494, 1495, 1526, 471, 1463, 1494, 1495, 
1496, 1527, 472, 1464, 1495, 1496, 1497, 1528, 473, 1465, 1496, 1497, 1498,
 1529, 474, 1466, 1497, 1498, 1499, 1530, 475, 1467, 1498, 1499, 1500, 1531
, 476, 1468, 1499, 1500, 1501, 1532, 477, 1469, 1500, 1501, 1502, 1533, 478
, 1470, 1501, 1502, 1503, 1534, 479, 1471, 1502, 1503, 1504, 1535, 480, 147
2, 1473, 1503, 1504, 1536, 481, 1473, 1505, 1506, 1536, 1537, 482, 1474, 15
05, 1506, 1507, 1538, 483, 1475, 1506, 1507, 1508, 1539, 484, 1476, 1507, 1
508, 1509, 1540, 485, 1477, 1508, 1509, 1510, 1541, 486, 1478, 1509, 1510, 
1511, 1542, 487, 1479, 1510, 1511, 1512, 1543, 488, 1480, 1511, 1512, 1513,
 1544, 489, 1481, 1512, 1513, 1514, 1545, 490, 1482, 1513, 1514, 1515, 1546
, 491, 1483, 1514, 1515, 1516, 1547, 492, 1484, 1515, 1516, 1517, 1548, 493
, 1485, 1516, 1517, 1518, 1549, 494, 1486, 1517, 1518, 1519, 1550, 495, 148
7, 1518, 1519, 1520, 1551, 496, 1488, 1519, 1520, 1521, 1552, 497, 1489, 15
20, 1521, 1522, 1553, 498, 1490, 1521, 1522, 1523, 1554, 499, 1491, 1522, 1
523, 1524, 1555, 500, 1492, 1523, 1524, 1525, 1556, 501, 1493, 1524, 1525, 
1526, 1557, 502, 1494, 1525, 1526, 1527, 1558, 503, 1495, 1526, 1527, 1528,
 1559, 504, 1496, 1527, 1528, 1529, 1560, 505, 1497, 1528, 1529, 1530, 1561
, 506, 1498, 1529, 1530, 1531, 1562, 507, 1499, 1530, 1531, 1532, 1563, 508
, 1500, 1531, 1532, 1533, 1564, 509, 1501, 1532, 1533, 1534, 1565, 510, 150
2, 1533, 1534, 1535, 1566, 511, 1503, 1534, 1535, 1536, 1567, 512, 1504, 15
05, 1535, 1536, 1568, 513, 1505, 1537, 1538, 1568, 1569, 514, 1506, 1537, 1
538, 1539, 1570, 515, 1507, 1538, 1539, 1540, 1571, 516, 1508, 1539, 1540, 
1541, 1572, 517, 1509, 1540, 1541, 1542, 1573, 518, 1510, 1541, 1542, 1543,
 1574, 519, 1511, 1542, 1543, 1544, 1575, 520, 1512, 1543, 1544, 1545, 1576
, 521, 1513, 1544, 1545, 1546, 1577, 522, 1514, 1545, 1546, 1547, 1578, 523
, 1515, 1546, 1547, 1548, 1579, 524, 1516, 1547, 1548, 1549, 1580, 525, 151
7, 1548, 1549, 1550, 1581, 526, 1518, 1549, 1550, 1551, 1582, 527, 1519, 15
50, 1551, 1552, 1583, 528, 1520, 1551, 1552, 1553, 1584, 529, 1521, 1552, 1
553, 1554, 1585, 530, 1522, 1553, 1554, 1555, 1586, 531, 1523, 1554, 1555, 
1556, 1587, 532, 1524, 1555, 1556, 1557, 1588, 533, 1525, 1556, 1557, 1558,
 1589, 534, 1526, 1557, 1558, 1559, 1590, 535, 1527, 1558, 1559, 1560, 1591
, 536, 1528, 1559, 1560, 1561, 1592, 537, 1529, 1560, 1561, 1562, 1593, 538
, 1530, 1561, 1562, 1563, 1594, 539, 1531, 1562, 1563, 1564, 1595, 540, 153
2, 1563, 1564, 1565, 1596, 541, 1533, 1564, 1565, 1566, 1597, 542, 1534, 15
65, 1566, 1567, 1598, 543, 1535, 1566, 1567, 1568, 1599, 544, 1536, 1537, 1
567, 1568, 1600, 545, 1537, 1569, 1570, 1600, 1601, 546, 1538, 1569, 1570, 
1571, 1602, 547, 1539, 1570, 1571, 1572, 1603, 548, 1540, 1571, 1572, 1573,
 1604, 549, 1541, 1572, 1573, 1574, 1605, 550, 1542, 1573, 1574, 1575, 1606
, 551, 1543, 1574, 1575, 1576, 1607, 552, 1544, 1575, 1576, 1577, 1608, 553
, 1545, 1576, 1577, 1578, 1609, 554, 1546, 1577, 1578, 1579, 1610, 555, 154
7, 1578, 1579, 1580, 1611, 556, 1548, 1579, 1580, 1581, 1612, 557, 1549, 15
80, 1581, 1582, 1613, 558, 1550, 1581, 1582, 1583, 1614, 559, 1551, 1582, 1
583, 1584, 1615, 560, 1552, 1583, 1584, 1585, 1616, 561, 1553, 1584, 1585, 
1586, 1617, 562, 1554, 1585, 1586, 1587, 1618, 563, 1555, 1586, 1587, 1588,
 1619, 564, 1556, 1587, 1588, 1589, 1620, 565, 1557, 1588, 1589, 1590, 1621
, 566, 1558, 1589, 1590, 1591, 1622, 567, 1559, 1590, 1591, 1592, 1623, 568
, 1560, 1591, 1592, 1593, 1624, 569, 1561, 1592, 1593, 1594, 1625, 570, 156
2, 1593, 1594, 1595, 1626, 571, 1563, 1594, 1595, 1596, 1627, 572, 1564, 15
95, 1596, 1597, 1628, 573, 1565, 1596, 1597, 1598, 1629, 574, 1566, 1597, 1
598, 1599, 1630, 575, 1567, 1598, 1599, 1600, 1631, 576, 1568, 1569, 1599, 
1600, 1632, 577, 1569, 1601, 1602, 1632, 1633, 578, 1570, 1601, 1602, 1603,
 1634, 579, 1571, 1602, 1603, 1604, 1635, 580, 1572, 1603, 1604, 1605, 1636
, 581, 1573, 1604, 1605, 1606, 1637, 582, 1574, 1605, 1606, 1607, 1638, 583
, 1575, 1606, 1607, 1608, 1639, 584, 1576, 1607, 1608, 1609, 1640, 585, 157
7, 1608, 1609, 1610, 1641, 586, 1578, 1609, 1610, 1611, 1642, 587, 1579, 16
10, 1611, 1612, 1643, 588, 1580, 1611, 1612, 1613, 1644, 589, 1581, 1612, 1
613, 1614, 1645, 590, 1582, 1613, 1614, 1615, 1646, 591, 1583, 1614, 1615, 
1616, 1647, 592, 1584, 1615, 1616, 1617, 1648, 593, 1585, 1616, 1617, 1618,
 1649, 594, 1586, 1617, 1618, 1619, 1650, 595, 1587, 1618, 1619, 1620, 1651
, 596, 1588, 1619, 1620, 1621, 1652, 597, 1589, 1620, 1621, 1622, 1653, 598
, 1590, 1621, 1622, 1623, 1654, 599, 1591, 1622, 1623, 1624, 1655, 600, 159
2, 1623, 1624, 1625, 1656, 601, 1593, 1624, 1625, 1626, 1657, 602, 1594, 16
25, 1626, 1627, 1658, 603, 1595, 1626, 1627, 1628, 1659, 604, 1596, 1627, 1
628, 1629, 1660, 605, 1597, 1628, 1629, 1630, 1661, 606, 1598, 1629, 1630, 
1631, 1662, 607, 1599, 1630, 1631, 1632, 1663, 608, 1600, 1601, 1631, 1632,
 1664, 609, 1601, 1633, 1634, 1664, 1665, 610, 1602, 1633, 1634, 1635, 1666
, 611, 1603, 1634, 1635, 1636, 1667, 612, 1604, 1635, 1636, 1637, 1668, 613
, 1605, 1636, 1637, 1638, 1669, 614, 1606, 1637, 1638, 1639, 1670, 615, 160
7, 1638, 1639, 1640, 1671, 616, 1608, 1639, 1640, 1641, 1672, 617, 1609, 16
40, 1641, 1642, 1673, 618, 1610, 1641, 1642, 1643, 1674, 619, 1611, 1642, 1
643, 1644, 1675, 620, 1612, 1643, 1644, 1645, 1676, 621, 1613, 1644, 1645, 
1646, 1677, 622, 1614, 1645, 1646, 1647, 1678, 623, 1615, 1646, 1647, 1648,
 1679, 624, 1616, 1647, 1648, 1649, 1680, 625, 1617, 1648, 1649, 1650, 1681
, 626, 1618, 1649, 1650, 1651, 1682, 627, 1619, 1650, 1651, 1652, 1683, 628
, 1620, 1651, 1652, 1653, 1684, 629, 1621, 1652, 1653, 1654, 1685, 630, 162
2, 1653, 1654, 1655, 1686, 631, 1623, 1654, 1655, 1656, 1687, 632, 1624, 16
55, 1656, 1657, 1688, 633, 1625, 1656, 1657, 1658, 1689, 634, 1626, 1657, 1
658, 1659, 1690, 635, 1627, 1658, 1659, 1660, 1691, 636, 1628, 1659, 1660, 
1661, 1692, 637, 1629, 1660, 1661, 1662, 1693, 638, 1630, 1661, 1662, 1663,
 1694, 639, 1631, 1662, 1663, 1664, 1695, 640, 1632, 1633, 1663, 1664, 1696
, 641, 1633, 1665, 1666, 1696, 1697, 642, 1634, 1665, 1666, 1667, 1698, 643
, 1635, 1666, 1667, 1668, 1699, 644, 1636, 1667, 1668, 1669, 1700, 645, 163
7, 1668, 1669, 1670, 1701, 646, 1638, 1669, 1670, 1671, 1702, 647, 1639, 16
70, 1671, 1672, 1703, 648, 1640, 1671, 1672, 1673, 1704, 649, 1641, 1672, 1
673, 1674, 1705, 650, 1642, 1673, 1674, 1675, 1706, 651, 1643, 1674, 1675, 
1676, 1707, 652, 1644, 1675, 1676, 1677, 1708, 653, 1645, 1676, 1677, 1678,
 1709, 654, 1646, 1677, 1678, 1679, 1710, 655, 1647, 1678, 1679, 1680, 1711
, 656, 1648, 1679, 1680, 1681, 1712, 657, 1649, 1680, 1681, 1682, 1713, 658
, 1650, 1681, 1682, 1683, 1714, 659, 1651, 1682, 1683, 1684, 1715, 660, 165
2, 1683, 1684, 1685, 1716, 661, 1653, 1684, 1685, 1686, 1717, 662, 1654, 16
85, 1686, 1687, 1718, 663, 1655, 1686, 1687, 1688, 1719, 664, 1656, 1687, 1
688, 1689, 1720, 665, 1657, 1688, 1689, 1690, 1721, 666, 1658, 1689, 1690, 
1691, 1722, 667, 1659, 1690, 1691, 1692, 1723, 668, 1660, 1691, 1692, 1693,
 1724, 669, 1661, 1692, 1693, 1694, 1725, 670, 1662, 1693, 1694, 1695, 1726
, 671, 1663, 1694, 1695, 1696, 1727, 672, 1664, 1665, 1695, 1696, 1728, 673
, 1665, 1697, 1698, 1728, 1729, 674, 1666, 1697, 1698, 1699, 1730, 675, 166
7, 1698, 1699, 1700, 1731, 676, 1668, 1699, 1700, 1701, 1732, 677, 1669, 17
00, 1701, 1702, 1733, 678, 1670, 1701, 1702, 1703, 1734, 679, 1671, 1702, 1
703, 1704, 1735, 680, 1672, 1703, 1704, 1705, 1736, 681, 1673, 1704, 1705, 
1706, 1737, 682, 1674, 1705, 1706, 1707, 1738, 683, 1675, 1706, 1707, 1708,
 1739, 684, 1676, 1707, 1708, 1709, 1740, 685, 1677, 1708, 1709, 1710, 1741
, 686, 1678, 1709, 1710, 1711, 1742, 687, 1679, 1710, 1711, 1712, 1743, 688
, 1680, 1711, 1712, 1713, 1744, 689, 1681, 1712, 1713, 1714, 1745, 690, 168
2, 1713, 1714, 1715, 1746, 691, 1683, 1714, 1715, 1716, 1747, 692, 1684, 17
15, 1716, 1717, 1748, 693, 1685, 1716, 1717, 1718, 1749, 694, 1686, 1717, 1
718, 1719, 1750, 695, 1687, 1718, 1719, 1720, 1751, 696, 1688, 1719, 1720, 
1721, 1752, 697, 1689, 1720, 1721, 1722, 1753, 698, 1690, 1721, 1722, 1723,
 1754, 699, 1691, 1722, 1723, 1724, 1755, 700, 1692, 1723, 1724, 1725, 1756
, 701, 1693, 1724, 1725, 1726, 1757, 702, 1694, 1725, 1726, 1727, 1758, 703
, 1695, 1726, 1727, 1728, 1759, 704, 1696, 1697, 1727, 1728, 1760, 705, 169
7, 1729, 1730, 1760, 1761, 706, 1698, 1729, 1730, 1731, 1762, 707, 1699, 17
30, 1731, 1732, 1763, 708, 1700, 1731, 1732, 1733, 1764, 709, 1701, 1732, 1
733, 1734, 1765, 710, 1702, 1733, 1734, 1735, 1766, 711, 1703, 1734, 1735, 
1736, 1767, 712, 1704, 1735, 1736, 1737, 1768, 713, 1705, 1736, 1737, 1738,
 1769, 714, 1706, 1737, 1738, 1739, 1770, 715, 1707, 1738, 1739, 1740, 1771
, 716, 1708, 1739, 1740, 1741, 1772, 717, 1709, 1740, 1741, 1742, 1773, 718
, 1710, 1741, 1742, 1743, 1774, 719, 1711, 1742, 1743, 1744, 1775, 720, 171
2, 1743, 1744, 1745, 1776, 721, 1713, 1744, 1745, 1746, 1777, 722, 1714, 17
45, 1746, 1747, 1778, 723, 1715, 1746, 1747, 1748, 1779, 724, 1716, 1747, 1
748, 1749, 1780, 725, 1717, 1748, 1749, 1750, 1781, 726, 1718, 1749, 1750, 
1751, 1782, 727, 1719, 1750, 1751, 1752, 1783, 728, 1720, 1751, 1752, 1753,
 1784, 729, 1721, 1752, 1753, 1754, 1785, 730, 1722, 1753, 1754, 1755, 1786
, 731, 1723, 1754, 1755, 1756, 1787, 732, 1724, 1755, 1756, 1757, 1788, 733
, 1725, 1756, 1757, 1758, 1789, 734, 1726, 1757, 1758, 1759, 1790, 735, 172
7, 1758, 1759, 1760, 1791, 736, 1728, 1729, 1759, 1760, 1792, 737, 1729, 17
61, 1762, 1792, 1793, 738, 1730, 1761, 1762, 1763, 1794, 739, 1731, 1762, 1
763, 1764, 1795, 740, 1732, 1763, 1764, 1765, 1796, 741, 1733, 1764, 1765, 
1766, 1797, 742, 1734, 1765, 1766, 1767, 1798, 743, 1735, 1766, 1767, 1768,
 1799, 744, 1736, 1767, 1768, 1769, 1800, 745, 1737, 1768, 1769, 1770, 1801
, 746, 1738, 1769, 1770, 1771, 1802, 747, 1739, 1770, 1771, 1772, 1803, 748
, 1740, 1771, 1772, 1773, 1804, 749, 1741, 1772, 1773, 1774, 1805, 750, 174
2, 1773, 1774, 1775, 1806, 751, 1743, 1774, 1775, 1776, 1807, 752, 1744, 17
75, 1776, 1777, 1808, 753, 1745, 1776, 1777, 1778, 1809, 754, 1746, 1777, 1
778, 1779, 1810, 755, 1747, 1778, 1779, 1780, 1811, 756, 1748, 1779, 1780, 
1781, 1812, 757, 1749, 1780, 1781, 1782, 1813, 758, 1750, 1781, 1782, 1783,
 1814, 759, 1751, 1782, 1783, 1784, 1815, 760, 1752, 1783, 1784, 1785, 1816
, 761, 1753, 1784, 1785, 1786, 1817, 762, 1754, 1785, 1786, 1787, 1818, 763
, 1755, 1786, 1787, 1788, 1819, 764, 1756, 1787, 1788, 1789, 1820, 765, 175
7, 1788, 1789, 1790, 1821, 766, 1758, 1789, 1790, 1791, 1822, 767, 1759, 17
90, 1791, 1792, 1823, 768, 1760, 1761, 1791, 1792, 1824, 769, 1761, 1793, 1
794, 1824, 1825, 770, 1762, 1793, 1794, 1795, 1826, 771, 1763, 1794, 1795, 
1796, 1827, 772, 1764, 1795, 1796, 1797, 1828, 773, 1765, 1796, 1797, 1798,
 1829, 774, 1766, 1797, 1798, 1799, 1830, 775, 1767, 1798, 1799, 1800, 1831
, 776, 1768, 1799, 1800, 1801, 1832, 777, 1769, 1800, 1801, 1802, 1833, 778
, 1770, 1801, 1802, 1803, 1834, 779, 1771, 1802, 1803, 1804, 1835, 780, 177
2, 1803, 1804, 1805, 1836, 781, 1773, 1804, 1805, 1806, 1837, 782, 1774, 18
05, 1806, 1807, 1838, 783, 1775, 1806, 1807, 1808, 1839, 784, 1776, 1807, 1
808, 1809, 1840, 785, 1777, 1808, 1809, 1810, 1841, 786, 1778, 1809, 1810, 
1811, 1842, 787, 1779, 1810, 1811, 1812, 1843, 788, 1780, 1811, 1812, 1813,
 1844, 789, 1781, 1812, 1813, 1814, 1845, 790, 1782, 1813, 1814, 1815, 1846
, 791, 1783, 1814, 1815, 1816, 1847, 792, 1784, 1815, 1816, 1817, 1848, 793
, 1785, 1816, 1817, 1818, 1849, 794, 1786, 1817, 1818, 1819, 1850, 795, 178
7, 1818, 1819, 1820, 1851, 796, 1788, 1819, 1820, 1821, 1852, 797, 1789, 18
20, 1821, 1822, 1853, 798, 1790, 1821, 1822, 1823, 1854, 799, 1791, 1822, 1
823, 1824, 1855, 800, 1792, 1793, 1823, 1824, 1856, 801, 1793, 1825, 1826, 
1856, 1857, 802, 1794, 1825, 1826, 1827, 1858, 803, 1795, 1826, 1827, 1828,
 1859, 804, 1796, 1827, 1828, 1829, 1860, 805, 1797, 1828, 1829, 1830, 1861
, 806, 1798, 1829, 1830, 1831, 1862, 807, 1799, 1830, 1831, 1832, 1863, 808
, 1800, 1831, 1832, 1833, 1864, 809, 1801, 1832, 1833, 1834, 1865, 810, 180
2, 1833, 1834, 1835, 1866, 811, 1803, 1834, 1835, 1836, 1867, 812, 1804, 18
35, 1836, 1837, 1868, 813, 1805, 1836, 1837, 1838, 1869, 814, 1806, 1837, 1
838, 1839, 1870, 815, 1807, 1838, 1839, 1840, 1871, 816, 1808, 1839, 1840, 
1841, 1872, 817, 1809, 1840, 1841, 1842, 1873, 818, 1810, 1841, 1842, 1843,
 1874, 819, 1811, 1842, 1843, 1844, 1875, 820, 1812, 1843, 1844, 1845, 1876
, 821, 1813, 1844, 1845, 1846, 1877, 822, 1814, 1845, 1846, 1847, 1878, 823
, 1815, 1846, 1847, 1848, 1879, 824, 1816, 1847, 1848, 1849, 1880, 825, 181
7, 1848, 1849, 1850, 1881, 826, 1818, 1849, 1850, 1851, 1882, 827, 1819, 18
50, 1851, 1852, 1883, 828, 1820, 1851, 1852, 1853, 1884, 829, 1821, 1852, 1
853, 1854, 1885, 830, 1822, 1853, 1854, 1855, 1886, 831, 1823, 1854, 1855, 
1856, 1887, 832, 1824, 1825, 1855, 1856, 1888, 833, 1825, 1857, 1858, 1888,
 1889, 834, 1826, 1857, 1858, 1859, 1890, 835, 1827, 1858, 1859, 1860, 1891
, 836, 1828, 1859, 1860, 1861, 1892, 837, 1829, 1860, 1861, 1862, 1893, 838
, 1830, 1861, 1862, 1863, 1894, 839, 1831, 1862, 1863, 1864, 1895, 840, 183
2, 1863, 1864, 1865, 1896, 841, 1833, 1864, 1865, 1866, 1897, 842, 1834, 18
65, 1866, 1867, 1898, 843, 1835, 1866, 1867, 1868, 1899, 844, 1836, 1867, 1
868, 1869, 1900, 845, 1837, 1868, 1869, 1870, 1901, 846, 1838, 1869, 1870, 
1871, 1902, 847, 1839, 1870, 1871, 1872, 1903, 848, 1840, 1871, 1872, 1873,
 1904, 849, 1841, 1872, 1873, 1874, 1905, 850, 1842, 1873, 1874, 1875, 1906
, 851, 1843, 1874, 1875, 1876, 1907, 852, 1844, 1875, 1876, 1877, 1908, 853
, 1845, 1876, 1877, 1878, 1909, 854, 1846, 1877, 1878, 1879, 1910, 855, 184
7, 1878, 1879, 1880, 1911, 856, 1848, 1879, 1880, 1881, 1912, 857, 1849, 18
80, 1881, 1882, 1913, 858, 1850, 1881, 1882, 1883, 1914, 859, 1851, 1882, 1
883, 1884, 1915, 860, 1852, 1883, 1884, 1885, 1916, 861, 1853, 1884, 1885, 
1886, 1917, 862, 1854, 1885, 1886, 1887, 1918, 863, 1855, 1886, 1887, 1888,
 1919, 864, 1856, 1857, 1887, 1888, 1920, 865, 1857, 1889, 1890, 1920, 1921
, 866, 1858, 1889, 1890, 1891, 1922, 867, 1859, 1890, 1891, 1892, 1923, 868
, 1860, 1891, 1892, 1893, 1924, 869, 1861, 1892, 1893, 1894, 1925, 870, 186
2, 1893, 1894, 1895, 1926, 871, 1863, 1894, 1895, 1896, 1927, 872, 1864, 18
95, 1896, 1897, 1928, 873, 1865, 1896, 1897, 1898, 1929, 874, 1866, 1897, 1
898, 1899, 1930, 875, 1867, 1898, 1899, 1900, 1931, 876, 1868, 1899, 1900, 
1901, 1932, 877, 1869, 1900, 1901, 1902, 1933, 878, 1870, 1901, 1902, 1903,
 1934, 879, 1871, 1902, 1903, 1904, 1935, 880, 1872, 1903, 1904, 1905, 1936
, 881, 1873, 1904, 1905, 1906, 1937, 882, 1874, 1905, 1906, 1907, 1938, 883
, 1875, 1906, 1907, 1908, 1939, 884, 1876, 1907, 1908, 1909, 1940, 885, 187
7, 1908, 1909, 1910, 1941, 886, 1878, 1909, 1910, 1911, 1942, 887, 1879, 19
10, 1911, 1912, 1943, 888, 1880, 1911, 1912, 1913, 1944, 889, 1881, 1912, 1
913, 1914, 1945, 890, 1882, 1913, 1914, 1915, 1946, 891, 1883, 1914, 1915, 
1916, 1947, 892, 1884, 1915, 1916, 1917, 1948, 893, 1885, 1916, 1917, 1918,
 1949, 894, 1886, 1917, 1918, 1919, 1950, 895, 1887, 1918, 1919, 1920, 1951
, 896, 1888, 1889, 1919, 1920, 1952, 897, 1889, 1921, 1922, 1952, 1953, 898
, 1890, 1921, 1922, 1923, 1954, 899, 1891, 1922, 1923, 1924, 1955, 900, 189
2, 1923, 1924, 1925, 1956, 901, 1893, 1924, 1925, 1926, 1957, 902, 1894, 19
25, 1926, 1927, 1958, 903, 1895, 1926, 1927, 1928, 1959, 904, 1896, 1927, 1
928, 1929, 1960, 905, 1897, 1928, 1929, 1930, 1961, 906, 1898, 1929, 1930, 
1931, 1962, 907, 1899, 1930, 1931, 1932, 1963, 908, 1900, 1931, 1932, 1933,
 1964, 909, 1901, 1932, 1933, 1934, 1965, 910, 1902, 1933, 1934, 1935, 1966
, 911, 1903, 1934, 1935, 1936, 1967, 912, 1904, 1935, 1936, 1937, 1968, 913
, 1905, 1936, 1937, 1938, 1969, 914, 1906, 1937, 1938, 1939, 1970, 915, 190
7, 1938, 1939, 1940, 1971, 916, 1908, 1939, 1940, 1941, 1972, 917, 1909, 19
40, 1941, 1942, 1973, 918, 1910, 1941, 1942, 1943, 1974, 919, 1911, 1942, 1
943, 1944, 1975, 920, 1912, 1943, 1944, 1945, 1976, 921, 1913, 1944, 1945, 
1946, 1977, 922, 1914, 1945, 1946, 1947, 1978, 923, 1915, 1946, 1947, 1948,
 1979, 924, 1916, 1947, 1948, 1949, 1980, 925, 1917, 1948, 1949, 1950, 1981
, 926, 1918, 1949, 1950, 1951, 1982, 927, 1919, 1950, 1951, 1952, 1983, 928
, 1920, 1921, 1951, 1952, 1984, 929, 1921, 1953, 1954, 1984, 1985, 930, 192
2, 1953, 1954, 1955, 1986, 931, 1923, 1954, 1955, 1956, 1987, 932, 1924, 19
55, 1956, 1957, 1988, 933, 1925, 1956, 1957, 1958, 1989, 934, 1926, 1957, 1
958, 1959, 1990, 935, 1927, 1958, 1959, 1960, 1991, 936, 1928, 1959, 1960, 
1961, 1992, 937, 1929, 1960, 1961, 1962, 1993, 938, 1930, 1961, 1962, 1963,
 1994, 939, 1931, 1962, 1963, 1964, 1995, 940, 1932, 1963, 1964, 1965, 1996
, 941, 1933, 1964, 1965, 1966, 1997, 942, 1934, 1965, 1966, 1967, 1998, 943
, 1935, 1966, 1967, 1968, 1999, 944, 1936, 1967, 1968, 1969, 2000, 945, 193
7, 1968, 1969, 1970, 2001, 946, 1938, 1969, 1970, 1971, 2002, 947, 1939, 19
70, 1971, 1972, 2003, 948, 1940, 1971, 1972, 1973, 2004, 949, 1941, 1972, 1
973, 1974, 2005, 950, 1942, 1973, 1974, 1975, 2006, 951, 1943, 1974, 1975, 
1976, 2007, 952, 1944, 1975, 1976, 1977, 2008, 953, 1945, 1976, 1977, 1978,
 2009, 954, 1946, 1977, 1978, 1979, 2010, 955, 1947, 1978, 1979, 1980, 2011
, 956, 1948, 1979, 1980, 1981, 2012, 957, 1949, 1980, 1981, 1982, 2013, 958
, 1950, 1981, 1982, 1983, 2014, 959, 1951, 1982, 1983, 1984, 2015, 960, 195
2, 1953, 1983, 1984, 2016, 961, 1953, 1985, 1986, 2016, 2017, 962, 1954, 19
85, 1986, 1987, 2018, 963, 1955, 1986, 1987, 1988, 2019, 964, 1956, 1987, 1
988, 1989, 2020, 965, 1957, 1988, 1989, 1990, 2021, 966, 1958, 1989, 1990, 
1991, 2022, 967, 1959, 1990, 1991, 1992, 2023, 968, 1960, 1991, 1992, 1993,
 2024, 969, 1961, 1992, 1993, 1994, 2025, 970, 1962, 1993, 1994, 1995, 2026
, 971, 1963, 1994, 1995, 1996, 2027, 972, 1964, 1995, 1996, 1997, 2028, 973
, 1965, 1996, 1997, 1998, 2029, 974, 1966, 1997, 1998, 1999, 2030, 975, 196
7, 1998, 1999, 2000, 2031, 976, 1968, 1999, 2000, 2001, 2032, 977, 1969, 20
00, 2001, 2002, 2033, 978, 1970, 2001, 2002, 2003, 2034, 979, 1971, 2002, 2
003, 2004, 2035, 980, 1972, 2003, 2004, 2005, 2036, 981, 1973, 2004, 2005, 
2006, 2037, 982, 1974, 2005, 2006, 2007, 2038, 983, 1975, 2006, 2007, 2008,
 2039, 984, 1976, 2007, 2008, 2009, 2040, 985, 1977, 2008, 2009, 2010, 2041
, 986, 1978, 2009, 2010, 2011, 2042, 987, 1979, 2010, 2011, 2012, 2043, 988
, 1980, 2011, 2012, 2013, 2044, 989, 1981, 2012, 2013, 2014, 2045, 990, 198
2, 2013, 2014, 2015, 2046, 991, 1983, 2014, 2015, 2016, 2047, 992, 1984, 19
85, 2015, 2016, 2048, 993, 1025, 1985, 2017, 2018, 2048, 994, 1026, 1986, 2
017, 2018, 2019, 995, 1027, 1987, 2018, 2019, 2020, 996, 1028, 1988, 2019, 
2020, 2021, 997, 1029, 1989, 2020, 2021, 2022, 998, 1030, 1990, 2021, 2022,
 2023, 999, 1031, 1991, 2022, 2023, 2024, 1000, 1032, 1992, 2023, 2024, 202
5, 1001, 1033, 1993, 2024, 2025, 2026, 1002, 1034, 1994, 2025, 2026, 2027, 
1003, 1035, 1995, 2026, 2027, 2028, 1004, 1036, 1996, 2027, 2028, 2029, 100
5, 1037, 1997, 2028, 2029, 2030, 1006, 1038, 1998, 2029, 2030, 2031, 1007, 
1039, 1999, 2030, 2031, 2032, 1008, 1040, 2000, 2031, 2032, 2033, 1009, 104
1, 2001, 2032, 2033, 2034, 1010, 1042, 2002, 2033, 2034, 2035, 1011, 1043, 
2003, 2034, 2035, 2036, 1012, 1044, 2004, 2035, 2036, 2037, 1013, 1045, 200
5, 2036, 2037, 2038, 1014, 1046, 2006, 2037, 2038, 2039, 1015, 1047, 2007, 
2038, 2039, 2040, 1016, 1048, 2008, 2039, 2040, 2041, 1017, 1049, 2009, 204
0, 2041, 2042, 1018, 1050, 2010, 2041, 2042, 2043, 1019, 1051, 2011, 2042, 
2043, 2044, 1020, 1052, 2012, 2043, 2044, 2045, 1021, 1053, 2013, 2044, 204
5, 2046, 1022, 1054, 2014, 2045, 2046, 2047, 1023, 1055, 2015, 2046, 2047, 
2048, 1024, 1056, 2016, 2017, 2047, 2048], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2
, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6
, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13
, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 1
7, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 
20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23,
 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26
, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 29, 29, 29, 2
9, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 
32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35,
 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38
, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 4
2, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 45, 
45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 48, 48,
 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 51, 51, 51
, 51, 51, 51, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54, 54, 5
4, 54, 54, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 
57, 57, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60,
 60, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63
, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 6
7, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 70, 
70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 73, 73,
 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 76, 76, 76
, 76, 76, 76, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 79, 79, 79, 7
9, 79, 79, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 
82, 82, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85,
 85, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 88, 88, 88, 88, 88, 88
, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 9
2, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 95, 
95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 97, 98, 98,
 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 101,
 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103,
 103, 103, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 106,
 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108,
 108, 108, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 111,
 111, 111, 111, 111, 111, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113,
 113, 113, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 116,
 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118,
 118, 118, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 121,
 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123,
 123, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 126,
 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128,
 128, 128, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 131,
 131, 131, 131, 131, 131, 132, 132, 132, 132, 132, 132, 133, 133, 133, 133,
 133, 133, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 136,
 136, 136, 136, 136, 136, 137, 137, 137, 137, 137, 137, 138, 138, 138, 138,
 138, 138, 139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 140, 140, 141,
 141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 143, 143, 143, 143,
 143, 143, 144, 144, 144, 144, 144, 144, 145, 145, 145, 145, 145, 145, 146,
 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148,
 148, 148, 149, 149, 149, 149, 149, 149, 150, 150, 150, 150, 150, 150, 151,
 151, 151, 151, 151, 151, 152, 152, 152, 152, 152, 152, 153, 153, 153, 153,
 153, 153, 154, 154, 154, 154, 154, 154, 155, 155, 155, 155, 155, 155, 156,
 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158,
 158, 158, 159, 159, 159, 159, 159, 159, 160, 160, 160, 160, 160, 160, 161,
 161, 161, 161, 161, 161, 162, 162, 162, 162, 162, 162, 163, 163, 163, 163,
 163, 163, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165, 165, 166,
 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168,
 168, 168, 169, 169, 169, 169, 169, 169, 170, 170, 170, 170, 170, 170, 171,
 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173,
 173, 173, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 176,
 176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178,
 178, 178, 179, 179, 179, 179, 179, 179, 180, 180, 180, 180, 180, 180, 181,
 181, 181, 181, 181, 181, 182, 182, 182, 182, 182, 182, 183, 183, 183, 183,
 183, 183, 184, 184, 184, 184, 184, 184, 185, 185, 185, 185, 185, 185, 186,
 186, 186, 186, 186, 186, 187, 187, 187, 187, 187, 187, 188, 188, 188, 188,
 188, 188, 189, 189, 189, 189, 189, 189, 190, 190, 190, 190, 190, 190, 191,
 191, 191, 191, 191, 191, 192, 192, 192, 192, 192, 192, 193, 193, 193, 193,
 193, 193, 194, 194, 194, 194, 194, 194, 195, 195, 195, 195, 195, 195, 196,
 196, 196, 196, 196, 196, 197, 197, 197, 197, 197, 197, 198, 198, 198, 198,
 198, 198, 199, 199, 199, 199, 199, 199, 200, 200, 200, 200, 200, 200, 201,
 201, 201, 201, 201, 201, 202, 202, 202, 202, 202, 202, 203, 203, 203, 203,
 203, 203, 204, 204, 204, 204, 204, 204, 205, 205, 205, 205, 205, 205, 206,
 206, 206, 206, 206, 206, 207, 207, 207, 207, 207, 207, 208, 208, 208, 208,
 208, 208, 209, 209, 209, 209, 209, 209, 210, 210, 210, 210, 210, 210, 211,
 211, 211, 211, 211, 211, 212, 212, 212, 212, 212, 212, 213, 213, 213, 213,
 213, 213, 214, 214, 214, 214, 214, 214, 215, 215, 215, 215, 215, 215, 216,
 216, 216, 216, 216, 216, 217, 217, 217, 217, 217, 217, 218, 218, 218, 218,
 218, 218, 219, 219, 219, 219, 219, 219, 220, 220, 220, 220, 220, 220, 221,
 221, 221, 221, 221, 221, 222, 222, 222, 222, 222, 222, 223, 223, 223, 223,
 223, 223, 224, 224, 224, 224, 224, 224, 225, 225, 225, 225, 225, 225, 226,
 226, 226, 226, 226, 226, 227, 227, 227, 227, 227, 227, 228, 228, 228, 228,
 228, 228, 229, 229, 229, 229, 229, 229, 230, 230, 230, 230, 230, 230, 231,
 231, 231, 231, 231, 231, 232, 232, 232, 232, 232, 232, 233, 233, 233, 233,
 233, 233, 234, 234, 234, 234, 234, 234, 235, 235, 235, 235, 235, 235, 236,
 236, 236, 236, 236, 236, 237, 237, 237, 237, 237, 237, 238, 238, 238, 238,
 238, 238, 239, 239, 239, 239, 239, 239, 240, 240, 240, 240, 240, 240, 241,
 241, 241, 241, 241, 241, 242, 242, 242, 242, 242, 242, 243, 243, 243, 243,
 243, 243, 244, 244, 244, 244, 244, 244, 245, 245, 245, 245, 245, 245, 246,
 246, 246, 246, 246, 246, 247, 247, 247, 247, 247, 247, 248, 248, 248, 248,
 248, 248, 249, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 251,
 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253,
 253, 253, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 256,
 256, 256, 256, 256, 256, 257, 257, 257, 257, 257, 257, 258, 258, 258, 258,
 258, 258, 259, 259, 259, 259, 259, 259, 260, 260, 260, 260, 260, 260, 261,
 261, 261, 261, 261, 261, 262, 262, 262, 262, 262, 262, 263, 263, 263, 263,
 263, 263, 264, 264, 264, 264, 264, 264, 265, 265, 265, 265, 265, 265, 266,
 266, 266, 266, 266, 266, 267, 267, 267, 267, 267, 267, 268, 268, 268, 268,
 268, 268, 269, 269, 269, 269, 269, 269, 270, 270, 270, 270, 270, 270, 271,
 271, 271, 271, 271, 271, 272, 272, 272, 272, 272, 272, 273, 273, 273, 273,
 273, 273, 274, 274, 274, 274, 274, 274, 275, 275, 275, 275, 275, 275, 276,
 276, 276, 276, 276, 276, 277, 277, 277, 277, 277, 277, 278, 278, 278, 278,
 278, 278, 279, 279, 279, 279, 279, 279, 280, 280, 280, 280, 280, 280, 281,
 281, 281, 281, 281, 281, 282, 282, 282, 282, 282, 282, 283, 283, 283, 283,
 283, 283, 284, 284, 284, 284, 284, 284, 285, 285, 285, 285, 285, 285, 286,
 286, 286, 286, 286, 286, 287, 287, 287, 287, 287, 287, 288, 288, 288, 288,
 288, 288, 289, 289, 289, 289, 289, 289, 290, 290, 290, 290, 290, 290, 291,
 291, 291, 291, 291, 291, 292, 292, 292, 292, 292, 292, 293, 293, 293, 293,
 293, 293, 294, 294, 294, 294, 294, 294, 295, 295, 295, 295, 295, 295, 296,
 296, 296, 296, 296, 296, 297, 297, 297, 297, 297, 297, 298, 298, 298, 298,
 298, 298, 299, 299, 299, 299, 299, 299, 300, 300, 300, 300, 300, 300, 301,
 301, 301, 301, 301, 301, 302, 302, 302, 302, 302, 302, 303, 303, 303, 303,
 303, 303, 304, 304, 304, 304, 304, 304, 305, 305, 305, 305, 305, 305, 306,
 306, 306, 306, 306, 306, 307, 307, 307, 307, 307, 307, 308, 308, 308, 308,
 308, 308, 309, 309, 309, 309, 309, 309, 310, 310, 310, 310, 310, 310, 311,
 311, 311, 311, 311, 311, 312, 312, 312, 312, 312, 312, 313, 313, 313, 313,
 313, 313, 314, 314, 314, 314, 314, 314, 315, 315, 315, 315, 315, 315, 316,
 316, 316, 316, 316, 316, 317, 317, 317, 317, 317, 317, 318, 318, 318, 318,
 318, 318, 319, 319, 319, 319, 319, 319, 320, 320, 320, 320, 320, 320, 321,
 321, 321, 321, 321, 321, 322, 322, 322, 322, 322, 322, 323, 323, 323, 323,
 323, 323, 324, 324, 324, 324, 324, 324, 325, 325, 325, 325, 325, 325, 326,
 326, 326, 326, 326, 326, 327, 327, 327, 327, 327, 327, 328, 328, 328, 328,
 328, 328, 329, 329, 329, 329, 329, 329, 330, 330, 330, 330, 330, 330, 331,
 331, 331, 331, 331, 331, 332, 332, 332, 332, 332, 332, 333, 333, 333, 333,
 333, 333, 334, 334, 334, 334, 334, 334, 335, 335, 335, 335, 335, 335, 336,
 336, 336, 336, 336, 336, 337, 337, 337, 337, 337, 337, 338, 338, 338, 338,
 338, 338, 339, 339, 339, 339, 339, 339, 340, 340, 340, 340, 340, 340, 341,
 341, 341, 341, 341, 341, 342, 342, 342, 342, 342, 342, 343, 343, 343, 343,
 343, 343, 344, 344, 344, 344, 344, 344, 345, 345, 345, 345, 345, 345, 346,
 346, 346, 346, 346, 346, 347, 347, 347, 347, 347, 347, 348, 348, 348, 348,
 348, 348, 349, 349, 349, 349, 349, 349, 350, 350, 350, 350, 350, 350, 351,
 351, 351, 351, 351, 351, 352, 352, 352, 352, 352, 352, 353, 353, 353, 353,
 353, 353, 354, 354, 354, 354, 354, 354, 355, 355, 355, 355, 355, 355, 356,
 356, 356, 356, 356, 356, 357, 357, 357, 357, 357, 357, 358, 358, 358, 358,
 358, 358, 359, 359, 359, 359, 359, 359, 360, 360, 360, 360, 360, 360, 361,
 361, 361, 361, 361, 361, 362, 362, 362, 362, 362, 362, 363, 363, 363, 363,
 363, 363, 364, 364, 364, 364, 364, 364, 365, 365, 365, 365, 365, 365, 366,
 366, 366, 366, 366, 366, 367, 367, 367, 367, 367, 367, 368, 368, 368, 368,
 368, 368, 369, 369, 369, 369, 369, 369, 370, 370, 370, 370, 370, 370, 371,
 371, 371, 371, 371, 371, 372, 372, 372, 372, 372, 372, 373, 373, 373, 373,
 373, 373, 374, 374, 374, 374, 374, 374, 375, 375, 375, 375, 375, 375, 376,
 376, 376, 376, 376, 376, 377, 377, 377, 377, 377, 377, 378, 378, 378, 378,
 378, 378, 379, 379, 379, 379, 379, 379, 380, 380, 380, 380, 380, 380, 381,
 381, 381, 381, 381, 381, 382, 382, 382, 382, 382, 382, 383, 383, 383, 383,
 383, 383, 384, 384, 384, 384, 384, 384, 385, 385, 385, 385, 385, 385, 386,
 386, 386, 386, 386, 386, 387, 387, 387, 387, 387, 387, 388, 388, 388, 388,
 388, 388, 389, 389, 389, 389, 389, 389, 390, 390, 390, 390, 390, 390, 391,
 391, 391, 391, 391, 391, 392, 392, 392, 392, 392, 392, 393, 393, 393, 393,
 393, 393, 394, 394, 394, 394, 394, 394, 395, 395, 395, 395, 395, 395, 396,
 396, 396, 396, 396, 396, 397, 397, 397, 397, 397, 397, 398, 398, 398, 398,
 398, 398, 399, 399, 399, 399, 399, 399, 400, 400, 400, 400, 400, 400, 401,
 401, 401, 401, 401, 401, 402, 402, 402, 402, 402, 402, 403, 403, 403, 403,
 403, 403, 404, 404, 404, 404, 404, 404, 405, 405, 405, 405, 405, 405, 406,
 406, 406, 406, 406, 406, 407, 407, 407, 407, 407, 407, 408, 408, 408, 408,
 408, 408, 409, 409, 409, 409, 409, 409, 410, 410, 410, 410, 410, 410, 411,
 411, 411, 411, 411, 411, 412, 412, 412, 412, 412, 412, 413, 413, 413, 413,
 413, 413, 414, 414, 414, 414, 414, 414, 415, 415, 415, 415, 415, 415, 416,
 416, 416, 416, 416, 416, 417, 417, 417, 417, 417, 417, 418, 418, 418, 418,
 418, 418, 419, 419, 419, 419, 419, 419, 420, 420, 420, 420, 420, 420, 421,
 421, 421, 421, 421, 421, 422, 422, 422, 422, 422, 422, 423, 423, 423, 423,
 423, 423, 424, 424, 424, 424, 424, 424, 425, 425, 425, 425, 425, 425, 426,
 426, 426, 426, 426, 426, 427, 427, 427, 427, 427, 427, 428, 428, 428, 428,
 428, 428, 429, 429, 429, 429, 429, 429, 430, 430, 430, 430, 430, 430, 431,
 431, 431, 431, 431, 431, 432, 432, 432, 432, 432, 432, 433, 433, 433, 433,
 433, 433, 434, 434, 434, 434, 434, 434, 435, 435, 435, 435, 435, 435, 436,
 436, 436, 436, 436, 436, 437, 437, 437, 437, 437, 437, 438, 438, 438, 438,
 438, 438, 439, 439, 439, 439, 439, 439, 440, 440, 440, 440, 440, 440, 441,
 441, 441, 441, 441, 441, 442, 442, 442, 442, 442, 442, 443, 443, 443, 443,
 443, 443, 444, 444, 444, 444, 444, 444, 445, 445, 445, 445, 445, 445, 446,
 446, 446, 446, 446, 446, 447, 447, 447, 447, 447, 447, 448, 448, 448, 448,
 448, 448, 449, 449, 449, 449, 449, 449, 450, 450, 450, 450, 450, 450, 451,
 451, 451, 451, 451, 451, 452, 452, 452, 452, 452, 452, 453, 453, 453, 453,
 453, 453, 454, 454, 454, 454, 454, 454, 455, 455, 455, 455, 455, 455, 456,
 456, 456, 456, 456, 456, 457, 457, 457, 457, 457, 457, 458, 458, 458, 458,
 458, 458, 459, 459, 459, 459, 459, 459, 460, 460, 460, 460, 460, 460, 461,
 461, 461, 461, 461, 461, 462, 462, 462, 462, 462, 462, 463, 463, 463, 463,
 463, 463, 464, 464, 464, 464, 464, 464, 465, 465, 465, 465, 465, 465, 466,
 466, 466, 466, 466, 466, 467, 467, 467, 467, 467, 467, 468, 468, 468, 468,
 468, 468, 469, 469, 469, 469, 469, 469, 470, 470, 470, 470, 470, 470, 471,
 471, 471, 471, 471, 471, 472, 472, 472, 472, 472, 472, 473, 473, 473, 473,
 473, 473, 474, 474, 474, 474, 474, 474, 475, 475, 475, 475, 475, 475, 476,
 476, 476, 476, 476, 476, 477, 477, 477, 477, 477, 477, 478, 478, 478, 478,
 478, 478, 479, 479, 479, 479, 479, 479, 480, 480, 480, 480, 480, 480, 481,
 481, 481, 481, 481, 481, 482, 482, 482, 482, 482, 482, 483, 483, 483, 483,
 483, 483, 484, 484, 484, 484, 484, 484, 485, 485, 485, 485, 485, 485, 486,
 486, 486, 486, 486, 486, 487, 487, 487, 487, 487, 487, 488, 488, 488, 488,
 488, 488, 489, 489, 489, 489, 489, 489, 490, 490, 490, 490, 490, 490, 491,
 491, 491, 491, 491, 491, 492, 492, 492, 492, 492, 492, 493, 493, 493, 493,
 493, 493, 494, 494, 494, 494, 494, 494, 495, 495, 495, 495, 495, 495, 496,
 496, 496, 496, 496, 496, 497, 497, 497, 497, 497, 497, 498, 498, 498, 498,
 498, 498, 499, 499, 499, 499, 499, 499, 500, 500, 500, 500, 500, 500, 501,
 501, 501, 501, 501, 501, 502, 502, 502, 502, 502, 502, 503, 503, 503, 503,
 503, 503, 504, 504, 504, 504, 504, 504, 505, 505, 505, 505, 505, 505, 506,
 506, 506, 506, 506, 506, 507, 507, 507, 507, 507, 507, 508, 508, 508, 508,
 508, 508, 509, 509, 509, 509, 509, 509, 510, 510, 510, 510, 510, 510, 511,
 511, 511, 511, 511, 511, 512, 512, 512, 512, 512, 512, 513, 513, 513, 513,
 513, 513, 514, 514, 514, 514, 514, 514, 515, 515, 515, 515, 515, 515, 516,
 516, 516, 516, 516, 516, 517, 517, 517, 517, 517, 517, 518, 518, 518, 518,
 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520, 520, 521,
 521, 521, 521, 521, 521, 522, 522, 522, 522, 522, 522, 523, 523, 523, 523,
 523, 523, 524, 524, 524, 524, 524, 524, 525, 525, 525, 525, 525, 525, 526,
 526, 526, 526, 526, 526, 527, 527, 527, 527, 527, 527, 528, 528, 528, 528,
 528, 528, 529, 529, 529, 529, 529, 529, 530, 530, 530, 530, 530, 530, 531,
 531, 531, 531, 531, 531, 532, 532, 532, 532, 532, 532, 533, 533, 533, 533,
 533, 533, 534, 534, 534, 534, 534, 534, 535, 535, 535, 535, 535, 535, 536,
 536, 536, 536, 536, 536, 537, 537, 537, 537, 537, 537, 538, 538, 538, 538,
 538, 538, 539, 539, 539, 539, 539, 539, 540, 540, 540, 540, 540, 540, 541,
 541, 541, 541, 541, 541, 542, 542, 542, 542, 542, 542, 543, 543, 543, 543,
 543, 543, 544, 544, 544, 544, 544, 544, 545, 545, 545, 545, 545, 545, 546,
 546, 546, 546, 546, 546, 547, 547, 547, 547, 547, 547, 548, 548, 548, 548,
 548, 548, 549, 549, 549, 549, 549, 549, 550, 550, 550, 550, 550, 550, 551,
 551, 551, 551, 551, 551, 552, 552, 552, 552, 552, 552, 553, 553, 553, 553,
 553, 553, 554, 554, 554, 554, 554, 554, 555, 555, 555, 555, 555, 555, 556,
 556, 556, 556, 556, 556, 557, 557, 557, 557, 557, 557, 558, 558, 558, 558,
 558, 558, 559, 559, 559, 559, 559, 559, 560, 560, 560, 560, 560, 560, 561,
 561, 561, 561, 561, 561, 562, 562, 562, 562, 562, 562, 563, 563, 563, 563,
 563, 563, 564, 564, 564, 564, 564, 564, 565, 565, 565, 565, 565, 565, 566,
 566, 566, 566, 566, 566, 567, 567, 567, 567, 567, 567, 568, 568, 568, 568,
 568, 568, 569, 569, 569, 569, 569, 569, 570, 570, 570, 570, 570, 570, 571,
 571, 571, 571, 571, 571, 572, 572, 572, 572, 572, 572, 573, 573, 573, 573,
 573, 573, 574, 574, 574, 574, 574, 574, 575, 575, 575, 575, 575, 575, 576,
 576, 576, 576, 576, 576, 577, 577, 577, 577, 577, 577, 578, 578, 578, 578,
 578, 578, 579, 579, 579, 579, 579, 579, 580, 580, 580, 580, 580, 580, 581,
 581, 581, 581, 581, 581, 582, 582, 582, 582, 582, 582, 583, 583, 583, 583,
 583, 583, 584, 584, 584, 584, 584, 584, 585, 585, 585, 585, 585, 585, 586,
 586, 586, 586, 586, 586, 587, 587, 587, 587, 587, 587, 588, 588, 588, 588,
 588, 588, 589, 589, 589, 589, 589, 589, 590, 590, 590, 590, 590, 590, 591,
 591, 591, 591, 591, 591, 592, 592, 592, 592, 592, 592, 593, 593, 593, 593,
 593, 593, 594, 594, 594, 594, 594, 594, 595, 595, 595, 595, 595, 595, 596,
 596, 596, 596, 596, 596, 597, 597, 597, 597, 597, 597, 598, 598, 598, 598,
 598, 598, 599, 599, 599, 599, 599, 599, 600, 600, 600, 600, 600, 600, 601,
 601, 601, 601, 601, 601, 602, 602, 602, 602, 602, 602, 603, 603, 603, 603,
 603, 603, 604, 604, 604, 604, 604, 604, 605, 605, 605, 605, 605, 605, 606,
 606, 606, 606, 606, 606, 607, 607, 607, 607, 607, 607, 608, 608, 608, 608,
 608, 608, 609, 609, 609, 609, 609, 609, 610, 610, 610, 610, 610, 610, 611,
 611, 611, 611, 611, 611, 612, 612, 612, 612, 612, 612, 613, 613, 613, 613,
 613, 613, 614, 614, 614, 614, 614, 614, 615, 615, 615, 615, 615, 615, 616,
 616, 616, 616, 616, 616, 617, 617, 617, 617, 617, 617, 618, 618, 618, 618,
 618, 618, 619, 619, 619, 619, 619, 619, 620, 620, 620, 620, 620, 620, 621,
 621, 621, 621, 621, 621, 622, 622, 622, 622, 622, 622, 623, 623, 623, 623,
 623, 623, 624, 624, 624, 624, 624, 624, 625, 625, 625, 625, 625, 625, 626,
 626, 626, 626, 626, 626, 627, 627, 627, 627, 627, 627, 628, 628, 628, 628,
 628, 628, 629, 629, 629, 629, 629, 629, 630, 630, 630, 630, 630, 630, 631,
 631, 631, 631, 631, 631, 632, 632, 632, 632, 632, 632, 633, 633, 633, 633,
 633, 633, 634, 634, 634, 634, 634, 634, 635, 635, 635, 635, 635, 635, 636,
 636, 636, 636, 636, 636, 637, 637, 637, 637, 637, 637, 638, 638, 638, 638,
 638, 638, 639, 639, 639, 639, 639, 639, 640, 640, 640, 640, 640, 640, 641,
 641, 641, 641, 641, 641, 642, 642, 642, 642, 642, 642, 643, 643, 643, 643,
 643, 643, 644, 644, 644, 644, 644, 644, 645, 645, 645, 645, 645, 645, 646,
 646, 646, 646, 646, 646, 647, 647, 647, 647, 647, 647, 648, 648, 648, 648,
 648, 648, 649, 649, 649, 649, 649, 649, 650, 650, 650, 650, 650, 650, 651,
 651, 651, 651, 651, 651, 652, 652, 652, 652, 652, 652, 653, 653, 653, 653,
 653, 653, 654, 654, 654, 654, 654, 654, 655, 655, 655, 655, 655, 655, 656,
 656, 656, 656, 656, 656, 657, 657, 657, 657, 657, 657, 658, 658, 658, 658,
 658, 658, 659, 659, 659, 659, 659, 659, 660, 660, 660, 660, 660, 660, 661,
 661, 661, 661, 661, 661, 662, 662, 662, 662, 662, 662, 663, 663, 663, 663,
 663, 663, 664, 664, 664, 664, 664, 664, 665, 665, 665, 665, 665, 665, 666,
 666, 666, 666, 666, 666, 667, 667, 667, 667, 667, 667, 668, 668, 668, 668,
 668, 668, 669, 669, 669, 669, 669, 669, 670, 670, 670, 670, 670, 670, 671,
 671, 671, 671, 671, 671, 672, 672, 672, 672, 672, 672, 673, 673, 673, 673,
 673, 673, 674, 674, 674, 674, 674, 674, 675, 675, 675, 675, 675, 675, 676,
 676, 676, 676, 676, 676, 677, 677, 677, 677, 677, 677, 678, 678, 678, 678,
 678, 678, 679, 679, 679, 679, 679, 679, 680, 680, 680, 680, 680, 680, 681,
 681, 681, 681, 681, 681, 682, 682, 682, 682, 682, 682, 683, 683, 683, 683,
 683, 683, 684, 684, 684, 684, 684, 684, 685, 685, 685, 685, 685, 685, 686,
 686, 686, 686, 686, 686, 687, 687, 687, 687, 687, 687, 688, 688, 688, 688,
 688, 688, 689, 689, 689, 689, 689, 689, 690, 690, 690, 690, 690, 690, 691,
 691, 691, 691, 691, 691, 692, 692, 692, 692, 692, 692, 693, 693, 693, 693,
 693, 693, 694, 694, 694, 694, 694, 694, 695, 695, 695, 695, 695, 695, 696,
 696, 696, 696, 696, 696, 697, 697, 697, 697, 697, 697, 698, 698, 698, 698,
 698, 698, 699, 699, 699, 699, 699, 699, 700, 700, 700, 700, 700, 700, 701,
 701, 701, 701, 701, 701, 702, 702, 702, 702, 702, 702, 703, 703, 703, 703,
 703, 703, 704, 704, 704, 704, 704, 704, 705, 705, 705, 705, 705, 705, 706,
 706, 706, 706, 706, 706, 707, 707, 707, 707, 707, 707, 708, 708, 708, 708,
 708, 708, 709, 709, 709, 709, 709, 709, 710, 710, 710, 710, 710, 710, 711,
 711, 711, 711, 711, 711, 712, 712, 712, 712, 712, 712, 713, 713, 713, 713,
 713, 713, 714, 714, 714, 714, 714, 714, 715, 715, 715, 715, 715, 715, 716,
 716, 716, 716, 716, 716, 717, 717, 717, 717, 717, 717, 718, 718, 718, 718,
 718, 718, 719, 719, 719, 719, 719, 719, 720, 720, 720, 720, 720, 720, 721,
 721, 721, 721, 721, 721, 722, 722, 722, 722, 722, 722, 723, 723, 723, 723,
 723, 723, 724, 724, 724, 724, 724, 724, 725, 725, 725, 725, 725, 725, 726,
 726, 726, 726, 726, 726, 727, 727, 727, 727, 727, 727, 728, 728, 728, 728,
 728, 728, 729, 729, 729, 729, 729, 729, 730, 730, 730, 730, 730, 730, 731,
 731, 731, 731, 731, 731, 732, 732, 732, 732, 732, 732, 733, 733, 733, 733,
 733, 733, 734, 734, 734, 734, 734, 734, 735, 735, 735, 735, 735, 735, 736,
 736, 736, 736, 736, 736, 737, 737, 737, 737, 737, 737, 738, 738, 738, 738,
 738, 738, 739, 739, 739, 739, 739, 739, 740, 740, 740, 740, 740, 740, 741,
 741, 741, 741, 741, 741, 742, 742, 742, 742, 742, 742, 743, 743, 743, 743,
 743, 743, 744, 744, 744, 744, 744, 744, 745, 745, 745, 745, 745, 745, 746,
 746, 746, 746, 746, 746, 747, 747, 747, 747, 747, 747, 748, 748, 748, 748,
 748, 748, 749, 749, 749, 749, 749, 749, 750, 750, 750, 750, 750, 750, 751,
 751, 751, 751, 751, 751, 752, 752, 752, 752, 752, 752, 753, 753, 753, 753,
 753, 753, 754, 754, 754, 754, 754, 754, 755, 755, 755, 755, 755, 755, 756,
 756, 756, 756, 756, 756, 757, 757, 757, 757, 757, 757, 758, 758, 758, 758,
 758, 758, 759, 759, 759, 759, 759, 759, 760, 760, 760, 760, 760, 760, 761,
 761, 761, 761, 761, 761, 762, 762, 762, 762, 762, 762, 763, 763, 763, 763,
 763, 763, 764, 764, 764, 764, 764, 764, 765, 765, 765, 765, 765, 765, 766,
 766, 766, 766, 766, 766, 767, 767, 767, 767, 767, 767, 768, 768, 768, 768,
 768, 768, 769, 769, 769, 769, 769, 769, 770, 770, 770, 770, 770, 770, 771,
 771, 771, 771, 771, 771, 772, 772, 772, 772, 772, 772, 773, 773, 773, 773,
 773, 773, 774, 774, 774, 774, 774, 774, 775, 775, 775, 775, 775, 775, 776,
 776, 776, 776, 776, 776, 777, 777, 777, 777, 777, 777, 778, 778, 778, 778,
 778, 778, 779, 779, 779, 779, 779, 779, 780, 780, 780, 780, 780, 780, 781,
 781, 781, 781, 781, 781, 782, 782, 782, 782, 782, 782, 783, 783, 783, 783,
 783, 783, 784, 784, 784, 784, 784, 784, 785, 785, 785, 785, 785, 785, 786,
 786, 786, 786, 786, 786, 787, 787, 787, 787, 787, 787, 788, 788, 788, 788,
 788, 788, 789, 789, 789, 789, 789, 789, 790, 790, 790, 790, 790, 790, 791,
 791, 791, 791, 791, 791, 792, 792, 792, 792, 792, 792, 793, 793, 793, 793,
 793, 793, 794, 794, 794, 794, 794, 794, 795, 795, 795, 795, 795, 795, 796,
 796, 796, 796, 796, 796, 797, 797, 797, 797, 797, 797, 798, 798, 798, 798,
 798, 798, 799, 799, 799, 799, 799, 799, 800, 800, 800, 800, 800, 800, 801,
 801, 801, 801, 801, 801, 802, 802, 802, 802, 802, 802, 803, 803, 803, 803,
 803, 803, 804, 804, 804, 804, 804, 804, 805, 805, 805, 805, 805, 805, 806,
 806, 806, 806, 806, 806, 807, 807, 807, 807, 807, 807, 808, 808, 808, 808,
 808, 808, 809, 809, 809, 809, 809, 809, 810, 810, 810, 810, 810, 810, 811,
 811, 811, 811, 811, 811, 812, 812, 812, 812, 812, 812, 813, 813, 813, 813,
 813, 813, 814, 814, 814, 814, 814, 814, 815, 815, 815, 815, 815, 815, 816,
 816, 816, 816, 816, 816, 817, 817, 817, 817, 817, 817, 818, 818, 818, 818,
 818, 818, 819, 819, 819, 819, 819, 819, 820, 820, 820, 820, 820, 820, 821,
 821, 821, 821, 821, 821, 822, 822, 822, 822, 822, 822, 823, 823, 823, 823,
 823, 823, 824, 824, 824, 824, 824, 824, 825, 825, 825, 825, 825, 825, 826,
 826, 826, 826, 826, 826, 827, 827, 827, 827, 827, 827, 828, 828, 828, 828,
 828, 828, 829, 829, 829, 829, 829, 829, 830, 830, 830, 830, 830, 830, 831,
 831, 831, 831, 831, 831, 832, 832, 832, 832, 832, 832, 833, 833, 833, 833,
 833, 833, 834, 834, 834, 834, 834, 834, 835, 835, 835, 835, 835, 835, 836,
 836, 836, 836, 836, 836, 837, 837, 837, 837, 837, 837, 838, 838, 838, 838,
 838, 838, 839, 839, 839, 839, 839, 839, 840, 840, 840, 840, 840, 840, 841,
 841, 841, 841, 841, 841, 842, 842, 842, 842, 842, 842, 843, 843, 843, 843,
 843, 843, 844, 844, 844, 844, 844, 844, 845, 845, 845, 845, 845, 845, 846,
 846, 846, 846, 846, 846, 847, 847, 847, 847, 847, 847, 848, 848, 848, 848,
 848, 848, 849, 849, 849, 849, 849, 849, 850, 850, 850, 850, 850, 850, 851,
 851, 851, 851, 851, 851, 852, 852, 852, 852, 852, 852, 853, 853, 853, 853,
 853, 853, 854, 854, 854, 854, 854, 854, 855, 855, 855, 855, 855, 855, 856,
 856, 856, 856, 856, 856, 857, 857, 857, 857, 857, 857, 858, 858, 858, 858,
 858, 858, 859, 859, 859, 859, 859, 859, 860, 860, 860, 860, 860, 860, 861,
 861, 861, 861, 861, 861, 862, 862, 862, 862, 862, 862, 863, 863, 863, 863,
 863, 863, 864, 864, 864, 864, 864, 864, 865, 865, 865, 865, 865, 865, 866,
 866, 866, 866, 866, 866, 867, 867, 867, 867, 867, 867, 868, 868, 868, 868,
 868, 868, 869, 869, 869, 869, 869, 869, 870, 870, 870, 870, 870, 870, 871,
 871, 871, 871, 871, 871, 872, 872, 872, 872, 872, 872, 873, 873, 873, 873,
 873, 873, 874, 874, 874, 874, 874, 874, 875, 875, 875, 875, 875, 875, 876,
 876, 876, 876, 876, 876, 877, 877, 877, 877, 877, 877, 878, 878, 878, 878,
 878, 878, 879, 879, 879, 879, 879, 879, 880, 880, 880, 880, 880, 880, 881,
 881, 881, 881, 881, 881, 882, 882, 882, 882, 882, 882, 883, 883, 883, 883,
 883, 883, 884, 884, 884, 884, 884, 884, 885, 885, 885, 885, 885, 885, 886,
 886, 886, 886, 886, 886, 887, 887, 887, 887, 887, 887, 888, 888, 888, 888,
 888, 888, 889, 889, 889, 889, 889, 889, 890, 890, 890, 890, 890, 890, 891,
 891, 891, 891, 891, 891, 892, 892, 892, 892, 892, 892, 893, 893, 893, 893,
 893, 893, 894, 894, 894, 894, 894, 894, 895, 895, 895, 895, 895, 895, 896,
 896, 896, 896, 896, 896, 897, 897, 897, 897, 897, 897, 898, 898, 898, 898,
 898, 898, 899, 899, 899, 899, 899, 899, 900, 900, 900, 900, 900, 900, 901,
 901, 901, 901, 901, 901, 902, 902, 902, 902, 902, 902, 903, 903, 903, 903,
 903, 903, 904, 904, 904, 904, 904, 904, 905, 905, 905, 905, 905, 905, 906,
 906, 906, 906, 906, 906, 907, 907, 907, 907, 907, 907, 908, 908, 908, 908,
 908, 908, 909, 909, 909, 909, 909, 909, 910, 910, 910, 910, 910, 910, 911,
 911, 911, 911, 911, 911, 912, 912, 912, 912, 912, 912, 913, 913, 913, 913,
 913, 913, 914, 914, 914, 914, 914, 914, 915, 915, 915, 915, 915, 915, 916,
 916, 916, 916, 916, 916, 917, 917, 917, 917, 917, 917, 918, 918, 918, 918,
 918, 918, 919, 919, 919, 919, 919, 919, 920, 920, 920, 920, 920, 920, 921,
 921, 921, 921, 921, 921, 922, 922, 922, 922, 922, 922, 923, 923, 923, 923,
 923, 923, 924, 924, 924, 924, 924, 924, 925, 925, 925, 925, 925, 925, 926,
 926, 926, 926, 926, 926, 927, 927, 927, 927, 927, 927, 928, 928, 928, 928,
 928, 928, 929, 929, 929, 929, 929, 929, 930, 930, 930, 930, 930, 930, 931,
 931, 931, 931, 931, 931, 932, 932, 932, 932, 932, 932, 933, 933, 933, 933,
 933, 933, 934, 934, 934, 934, 934, 934, 935, 935, 935, 935, 935, 935, 936,
 936, 936, 936, 936, 936, 937, 937, 937, 937, 937, 937, 938, 938, 938, 938,
 938, 938, 939, 939, 939, 939, 939, 939, 940, 940, 940, 940, 940, 940, 941,
 941, 941, 941, 941, 941, 942, 942, 942, 942, 942, 942, 943, 943, 943, 943,
 943, 943, 944, 944, 944, 944, 944, 944, 945, 945, 945, 945, 945, 945, 946,
 946, 946, 946, 946, 946, 947, 947, 947, 947, 947, 947, 948, 948, 948, 948,
 948, 948, 949, 949, 949, 949, 949, 949, 950, 950, 950, 950, 950, 950, 951,
 951, 951, 951, 951, 951, 952, 952, 952, 952, 952, 952, 953, 953, 953, 953,
 953, 953, 954, 954, 954, 954, 954, 954, 955, 955, 955, 955, 955, 955, 956,
 956, 956, 956, 956, 956, 957, 957, 957, 957, 957, 957, 958, 958, 958, 958,
 958, 958, 959, 959, 959, 959, 959, 959, 960, 960, 960, 960, 960, 960, 961,
 961, 961, 961, 961, 961, 962, 962, 962, 962, 962, 962, 963, 963, 963, 963,
 963, 963, 964, 964, 964, 964, 964, 964, 965, 965, 965, 965, 965, 965, 966,
 966, 966, 966, 966, 966, 967, 967, 967, 967, 967, 967, 968, 968, 968, 968,
 968, 968, 969, 969, 969, 969, 969, 969, 970, 970, 970, 970, 970, 970, 971,
 971, 971, 971, 971, 971, 972, 972, 972, 972, 972, 972, 973, 973, 973, 973,
 973, 973, 974, 974, 974, 974, 974, 974, 975, 975, 975, 975, 975, 975, 976,
 976, 976, 976, 976, 976, 977, 977, 977, 977, 977, 977, 978, 978, 978, 978,
 978, 978, 979, 979, 979, 979, 979, 979, 980, 980, 980, 980, 980, 980, 981,
 981, 981, 981, 981, 981, 982, 982, 982, 982, 982, 982, 983, 983, 983, 983,
 983, 983, 984, 984, 984, 984, 984, 984, 985, 985, 985, 985, 985, 985, 986,
 986, 986, 986, 986, 986, 987, 987, 987, 987, 987, 987, 988, 988, 988, 988,
 988, 988, 989, 989, 989, 989, 989, 989, 990, 990, 990, 990, 990, 990, 991,
 991, 991, 991, 991, 991, 992, 992, 992, 992, 992, 992, 993, 993, 993, 993,
 993, 993, 994, 994, 994, 994, 994, 994, 995, 995, 995, 995, 995, 995, 996,
 996, 996, 996, 996, 996, 997, 997, 997, 997, 997, 997, 998, 998, 998, 998,
 998, 998, 999, 999, 999, 999, 999, 999, 1000, 1000, 1000, 1000, 1000, 1000
, 1001, 1001, 1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002, 1002, 1002, 1
003, 1003, 1003, 1003, 1003, 1003, 1004, 1004, 1004, 1004, 1004, 1004, 1005
, 1005, 1005, 1005, 1005, 1005, 1006, 1006, 1006, 1006, 1006, 1006, 1007, 1
007, 1007, 1007, 1007, 1007, 1008, 1008, 1008, 1008, 1008, 1008, 1009, 1009
, 1009, 1009, 1009, 1009, 1010, 1010, 1010, 1010, 1010, 1010, 1011, 1011, 1
011, 1011, 1011, 1011, 1012, 1012, 1012, 1012, 1012, 1012, 1013, 1013, 1013
, 1013, 1013, 1013, 1014, 1014, 1014, 1014, 1014, 1014, 1015, 1015, 1015, 1
015, 1015, 1015, 1016, 1016, 1016, 1016, 1016, 1016, 1017, 1017, 1017, 1017
, 1017, 1017, 1018, 1018, 1018, 1018, 1018, 1018, 1019, 1019, 1019, 1019, 1
019, 1019, 1020, 1020, 1020, 1020, 1020, 1020, 1021, 1021, 1021, 1021, 1021
, 1021, 1022, 1022, 1022, 1022, 1022, 1022, 1023, 1023, 1023, 1023, 1023, 1
023, 1024, 1024, 1024, 1024, 1024, 1024, 1025, 1025, 1025, 1025, 1025, 1025
, 1026, 1026, 1026, 1026, 1026, 1026, 1027, 1027, 1027, 1027, 1027, 1027, 1
028, 1028, 1028, 1028, 1028, 1028, 1029, 1029, 1029, 1029, 1029, 1029, 1030
, 1030, 1030, 1030, 1030, 1030, 1031, 1031, 1031, 1031, 1031, 1031, 1032, 1
032, 1032, 1032, 1032, 1032, 1033, 1033, 1033, 1033, 1033, 1033, 1034, 1034
, 1034, 1034, 1034, 1034, 1035, 1035, 1035, 1035, 1035, 1035, 1036, 1036, 1
036, 1036, 1036, 1036, 1037, 1037, 1037, 1037, 1037, 1037, 1038, 1038, 1038
, 1038, 1038, 1038, 1039, 1039, 1039, 1039, 1039, 1039, 1040, 1040, 1040, 1
040, 1040, 1040, 1041, 1041, 1041, 1041, 1041, 1041, 1042, 1042, 1042, 1042
, 1042, 1042, 1043, 1043, 1043, 1043, 1043, 1043, 1044, 1044, 1044, 1044, 1
044, 1044, 1045, 1045, 1045, 1045, 1045, 1045, 1046, 1046, 1046, 1046, 1046
, 1046, 1047, 1047, 1047, 1047, 1047, 1047, 1048, 1048, 1048, 1048, 1048, 1
048, 1049, 1049, 1049, 1049, 1049, 1049, 1050, 1050, 1050, 1050, 1050, 1050
, 1051, 1051, 1051, 1051, 1051, 1051, 1052, 1052, 1052, 1052, 1052, 1052, 1
053, 1053, 1053, 1053, 1053, 1053, 1054, 1054, 1054, 1054, 1054, 1054, 1055
, 1055, 1055, 1055, 1055, 1055, 1056, 1056, 1056, 1056, 1056, 1056, 1057, 1
057, 1057, 1057, 1057, 1057, 1058, 1058, 1058, 1058, 1058, 1058, 1059, 1059
, 1059, 1059, 1059, 1059, 1060, 1060, 1060, 1060, 1060, 1060, 1061, 1061, 1
061, 1061, 1061, 1061, 1062, 1062, 1062, 1062, 1062, 1062, 1063, 1063, 1063
, 1063, 1063, 1063, 1064, 1064, 1064, 1064, 1064, 1064, 1065, 1065, 1065, 1
065, 1065, 1065, 1066, 1066, 1066, 1066, 1066, 1066, 1067, 1067, 1067, 1067
, 1067, 1067, 1068, 1068, 1068, 1068, 1068, 1068, 1069, 1069, 1069, 1069, 1
069, 1069, 1070, 1070, 1070, 1070, 1070, 1070, 1071, 1071, 1071, 1071, 1071
, 1071, 1072, 1072, 1072, 1072, 1072, 1072, 1073, 1073, 1073, 1073, 1073, 1
073, 1074, 1074, 1074, 1074, 1074, 1074, 1075, 1075, 1075, 1075, 1075, 1075
, 1076, 1076, 1076, 1076, 1076, 1076, 1077, 1077, 1077, 1077, 1077, 1077, 1
078, 1078, 1078, 1078, 1078, 1078, 1079, 1079, 1079, 1079, 1079, 1079, 1080
, 1080, 1080, 1080, 1080, 1080, 1081, 1081, 1081, 1081, 1081, 1081, 1082, 1
082, 1082, 1082, 1082, 1082, 1083, 1083, 1083, 1083, 1083, 1083, 1084, 1084
, 1084, 1084, 1084, 1084, 1085, 1085, 1085, 1085, 1085, 1085, 1086, 1086, 1
086, 1086, 1086, 1086, 1087, 1087, 1087, 1087, 1087, 1087, 1088, 1088, 1088
, 1088, 1088, 1088, 1089, 1089, 1089, 1089, 1089, 1089, 1090, 1090, 1090, 1
090, 1090, 1090, 1091, 1091, 1091, 1091, 1091, 1091, 1092, 1092, 1092, 1092
, 1092, 1092, 1093, 1093, 1093, 1093, 1093, 1093, 1094, 1094, 1094, 1094, 1
094, 1094, 1095, 1095, 1095, 1095, 1095, 1095, 1096, 1096, 1096, 1096, 1096
, 1096, 1097, 1097, 1097, 1097, 1097, 1097, 1098, 1098, 1098, 1098, 1098, 1
098, 1099, 1099, 1099, 1099, 1099, 1099, 1100, 1100, 1100, 1100, 1100, 1100
, 1101, 1101, 1101, 1101, 1101, 1101, 1102, 1102, 1102, 1102, 1102, 1102, 1
103, 1103, 1103, 1103, 1103, 1103, 1104, 1104, 1104, 1104, 1104, 1104, 1105
, 1105, 1105, 1105, 1105, 1105, 1106, 1106, 1106, 1106, 1106, 1106, 1107, 1
107, 1107, 1107, 1107, 1107, 1108, 1108, 1108, 1108, 1108, 1108, 1109, 1109
, 1109, 1109, 1109, 1109, 1110, 1110, 1110, 1110, 1110, 1110, 1111, 1111, 1
111, 1111, 1111, 1111, 1112, 1112, 1112, 1112, 1112, 1112, 1113, 1113, 1113
, 1113, 1113, 1113, 1114, 1114, 1114, 1114, 1114, 1114, 1115, 1115, 1115, 1
115, 1115, 1115, 1116, 1116, 1116, 1116, 1116, 1116, 1117, 1117, 1117, 1117
, 1117, 1117, 1118, 1118, 1118, 1118, 1118, 1118, 1119, 1119, 1119, 1119, 1
119, 1119, 1120, 1120, 1120, 1120, 1120, 1120, 1121, 1121, 1121, 1121, 1121
, 1121, 1122, 1122, 1122, 1122, 1122, 1122, 1123, 1123, 1123, 1123, 1123, 1
123, 1124, 1124, 1124, 1124, 1124, 1124, 1125, 1125, 1125, 1125, 1125, 1125
, 1126, 1126, 1126, 1126, 1126, 1126, 1127, 1127, 1127, 1127, 1127, 1127, 1
128, 1128, 1128, 1128, 1128, 1128, 1129, 1129, 1129, 1129, 1129, 1129, 1130
, 1130, 1130, 1130, 1130, 1130, 1131, 1131, 1131, 1131, 1131, 1131, 1132, 1
132, 1132, 1132, 1132, 1132, 1133, 1133, 1133, 1133, 1133, 1133, 1134, 1134
, 1134, 1134, 1134, 1134, 1135, 1135, 1135, 1135, 1135, 1135, 1136, 1136, 1
136, 1136, 1136, 1136, 1137, 1137, 1137, 1137, 1137, 1137, 1138, 1138, 1138
, 1138, 1138, 1138, 1139, 1139, 1139, 1139, 1139, 1139, 1140, 1140, 1140, 1
140, 1140, 1140, 1141, 1141, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142
, 1142, 1142, 1143, 1143, 1143, 1143, 1143, 1143, 1144, 1144, 1144, 1144, 1
144, 1144, 1145, 1145, 1145, 1145, 1145, 1145, 1146, 1146, 1146, 1146, 1146
, 1146, 1147, 1147, 1147, 1147, 1147, 1147, 1148, 1148, 1148, 1148, 1148, 1
148, 1149, 1149, 1149, 1149, 1149, 1149, 1150, 1150, 1150, 1150, 1150, 1150
, 1151, 1151, 1151, 1151, 1151, 1151, 1152, 1152, 1152, 1152, 1152, 1152, 1
153, 1153, 1153, 1153, 1153, 1153, 1154, 1154, 1154, 1154, 1154, 1154, 1155
, 1155, 1155, 1155, 1155, 1155, 1156, 1156, 1156, 1156, 1156, 1156, 1157, 1
157, 1157, 1157, 1157, 1157, 1158, 1158, 1158, 1158, 1158, 1158, 1159, 1159
, 1159, 1159, 1159, 1159, 1160, 1160, 1160, 1160, 1160, 1160, 1161, 1161, 1
161, 1161, 1161, 1161, 1162, 1162, 1162, 1162, 1162, 1162, 1163, 1163, 1163
, 1163, 1163, 1163, 1164, 1164, 1164, 1164, 1164, 1164, 1165, 1165, 1165, 1
165, 1165, 1165, 1166, 1166, 1166, 1166, 1166, 1166, 1167, 1167, 1167, 1167
, 1167, 1167, 1168, 1168, 1168, 1168, 1168, 1168, 1169, 1169, 1169, 1169, 1
169, 1169, 1170, 1170, 1170, 1170, 1170, 1170, 1171, 1171, 1171, 1171, 1171
, 1171, 1172, 1172, 1172, 1172, 1172, 1172, 1173, 1173, 1173, 1173, 1173, 1
173, 1174, 1174, 1174, 1174, 1174, 1174, 1175, 1175, 1175, 1175, 1175, 1175
, 1176, 1176, 1176, 1176, 1176, 1176, 1177, 1177, 1177, 1177, 1177, 1177, 1
178, 1178, 1178, 1178, 1178, 1178, 1179, 1179, 1179, 1179, 1179, 1179, 1180
, 1180, 1180, 1180, 1180, 1180, 1181, 1181, 1181, 1181, 1181, 1181, 1182, 1
182, 1182, 1182, 1182, 1182, 1183, 1183, 1183, 1183, 1183, 1183, 1184, 1184
, 1184, 1184, 1184, 1184, 1185, 1185, 1185, 1185, 1185, 1185, 1186, 1186, 1
186, 1186, 1186, 1186, 1187, 1187, 1187, 1187, 1187, 1187, 1188, 1188, 1188
, 1188, 1188, 1188, 1189, 1189, 1189, 1189, 1189, 1189, 1190, 1190, 1190, 1
190, 1190, 1190, 1191, 1191, 1191, 1191, 1191, 1191, 1192, 1192, 1192, 1192
, 1192, 1192, 1193, 1193, 1193, 1193, 1193, 1193, 1194, 1194, 1194, 1194, 1
194, 1194, 1195, 1195, 1195, 1195, 1195, 1195, 1196, 1196, 1196, 1196, 1196
, 1196, 1197, 1197, 1197, 1197, 1197, 1197, 1198, 1198, 1198, 1198, 1198, 1
198, 1199, 1199, 1199, 1199, 1199, 1199, 1200, 1200, 1200, 1200, 1200, 1200
, 1201, 1201, 1201, 1201, 1201, 1201, 1202, 1202, 1202, 1202, 1202, 1202, 1
203, 1203, 1203, 1203, 1203, 1203, 1204, 1204, 1204, 1204, 1204, 1204, 1205
, 1205, 1205, 1205, 1205, 1205, 1206, 1206, 1206, 1206, 1206, 1206, 1207, 1
207, 1207, 1207, 1207, 1207, 1208, 1208, 1208, 1208, 1208, 1208, 1209, 1209
, 1209, 1209, 1209, 1209, 1210, 1210, 1210, 1210, 1210, 1210, 1211, 1211, 1
211, 1211, 1211, 1211, 1212, 1212, 1212, 1212, 1212, 1212, 1213, 1213, 1213
, 1213, 1213, 1213, 1214, 1214, 1214, 1214, 1214, 1214, 1215, 1215, 1215, 1
215, 1215, 1215, 1216, 1216, 1216, 1216, 1216, 1216, 1217, 1217, 1217, 1217
, 1217, 1217, 1218, 1218, 1218, 1218, 1218, 1218, 1219, 1219, 1219, 1219, 1
219, 1219, 1220, 1220, 1220, 1220, 1220, 1220, 1221, 1221, 1221, 1221, 1221
, 1221, 1222, 1222, 1222, 1222, 1222, 1222, 1223, 1223, 1223, 1223, 1223, 1
223, 1224, 1224, 1224, 1224, 1224, 1224, 1225, 1225, 1225, 1225, 1225, 1225
, 1226, 1226, 1226, 1226, 1226, 1226, 1227, 1227, 1227, 1227, 1227, 1227, 1
228, 1228, 1228, 1228, 1228, 1228, 1229, 1229, 1229, 1229, 1229, 1229, 1230
, 1230, 1230, 1230, 1230, 1230, 1231, 1231, 1231, 1231, 1231, 1231, 1232, 1
232, 1232, 1232, 1232, 1232, 1233, 1233, 1233, 1233, 1233, 1233, 1234, 1234
, 1234, 1234, 1234, 1234, 1235, 1235, 1235, 1235, 1235, 1235, 1236, 1236, 1
236, 1236, 1236, 1236, 1237, 1237, 1237, 1237, 1237, 1237, 1238, 1238, 1238
, 1238, 1238, 1238, 1239, 1239, 1239, 1239, 1239, 1239, 1240, 1240, 1240, 1
240, 1240, 1240, 1241, 1241, 1241, 1241, 1241, 1241, 1242, 1242, 1242, 1242
, 1242, 1242, 1243, 1243, 1243, 1243, 1243, 1243, 1244, 1244, 1244, 1244, 1
244, 1244, 1245, 1245, 1245, 1245, 1245, 1245, 1246, 1246, 1246, 1246, 1246
, 1246, 1247, 1247, 1247, 1247, 1247, 1247, 1248, 1248, 1248, 1248, 1248, 1
248, 1249, 1249, 1249, 1249, 1249, 1249, 1250, 1250, 1250, 1250, 1250, 1250
, 1251, 1251, 1251, 1251, 1251, 1251, 1252, 1252, 1252, 1252, 1252, 1252, 1
253, 1253, 1253, 1253, 1253, 1253, 1254, 1254, 1254, 1254, 1254, 1254, 1255
, 1255, 1255, 1255, 1255, 1255, 1256, 1256, 1256, 1256, 1256, 1256, 1257, 1
257, 1257, 1257, 1257, 1257, 1258, 1258, 1258, 1258, 1258, 1258, 1259, 1259
, 1259, 1259, 1259, 1259, 1260, 1260, 1260, 1260, 1260, 1260, 1261, 1261, 1
261, 1261, 1261, 1261, 1262, 1262, 1262, 1262, 1262, 1262, 1263, 1263, 1263
, 1263, 1263, 1263, 1264, 1264, 1264, 1264, 1264, 1264, 1265, 1265, 1265, 1
265, 1265, 1265, 1266, 1266, 1266, 1266, 1266, 1266, 1267, 1267, 1267, 1267
, 1267, 1267, 1268, 1268, 1268, 1268, 1268, 1268, 1269, 1269, 1269, 1269, 1
269, 1269, 1270, 1270, 1270, 1270, 1270, 1270, 1271, 1271, 1271, 1271, 1271
, 1271, 1272, 1272, 1272, 1272, 1272, 1272, 1273, 1273, 1273, 1273, 1273, 1
273, 1274, 1274, 1274, 1274, 1274, 1274, 1275, 1275, 1275, 1275, 1275, 1275
, 1276, 1276, 1276, 1276, 1276, 1276, 1277, 1277, 1277, 1277, 1277, 1277, 1
278, 1278, 1278, 1278, 1278, 1278, 1279, 1279, 1279, 1279, 1279, 1279, 1280
, 1280, 1280, 1280, 1280, 1280, 1281, 1281, 1281, 1281, 1281, 1281, 1282, 1
282, 1282, 1282, 1282, 1282, 1283, 1283, 1283, 1283, 1283, 1283, 1284, 1284
, 1284, 1284, 1284, 1284, 1285, 1285, 1285, 1285, 1285, 1285, 1286, 1286, 1
286, 1286, 1286, 1286, 1287, 1287, 1287, 1287, 1287, 1287, 1288, 1288, 1288
, 1288, 1288, 1288, 1289, 1289, 1289, 1289, 1289, 1289, 1290, 1290, 1290, 1
290, 1290, 1290, 1291, 1291, 1291, 1291, 1291, 1291, 1292, 1292, 1292, 1292
, 1292, 1292, 1293, 1293, 1293, 1293, 1293, 1293, 1294, 1294, 1294, 1294, 1
294, 1294, 1295, 1295, 1295, 1295, 1295, 1295, 1296, 1296, 1296, 1296, 1296
, 1296, 1297, 1297, 1297, 1297, 1297, 1297, 1298, 1298, 1298, 1298, 1298, 1
298, 1299, 1299, 1299, 1299, 1299, 1299, 1300, 1300, 1300, 1300, 1300, 1300
, 1301, 1301, 1301, 1301, 1301, 1301, 1302, 1302, 1302, 1302, 1302, 1302, 1
303, 1303, 1303, 1303, 1303, 1303, 1304, 1304, 1304, 1304, 1304, 1304, 1305
, 1305, 1305, 1305, 1305, 1305, 1306, 1306, 1306, 1306, 1306, 1306, 1307, 1
307, 1307, 1307, 1307, 1307, 1308, 1308, 1308, 1308, 1308, 1308, 1309, 1309
, 1309, 1309, 1309, 1309, 1310, 1310, 1310, 1310, 1310, 1310, 1311, 1311, 1
311, 1311, 1311, 1311, 1312, 1312, 1312, 1312, 1312, 1312, 1313, 1313, 1313
, 1313, 1313, 1313, 1314, 1314, 1314, 1314, 1314, 1314, 1315, 1315, 1315, 1
315, 1315, 1315, 1316, 1316, 1316, 1316, 1316, 1316, 1317, 1317, 1317, 1317
, 1317, 1317, 1318, 1318, 1318, 1318, 1318, 1318, 1319, 1319, 1319, 1319, 1
319, 1319, 1320, 1320, 1320, 1320, 1320, 1320, 1321, 1321, 1321, 1321, 1321
, 1321, 1322, 1322, 1322, 1322, 1322, 1322, 1323, 1323, 1323, 1323, 1323, 1
323, 1324, 1324, 1324, 1324, 1324, 1324, 1325, 1325, 1325, 1325, 1325, 1325
, 1326, 1326, 1326, 1326, 1326, 1326, 1327, 1327, 1327, 1327, 1327, 1327, 1
328, 1328, 1328, 1328, 1328, 1328, 1329, 1329, 1329, 1329, 1329, 1329, 1330
, 1330, 1330, 1330, 1330, 1330, 1331, 1331, 1331, 1331, 1331, 1331, 1332, 1
332, 1332, 1332, 1332, 1332, 1333, 1333, 1333, 1333, 1333, 1333, 1334, 1334
, 1334, 1334, 1334, 1334, 1335, 1335, 1335, 1335, 1335, 1335, 1336, 1336, 1
336, 1336, 1336, 1336, 1337, 1337, 1337, 1337, 1337, 1337, 1338, 1338, 1338
, 1338, 1338, 1338, 1339, 1339, 1339, 1339, 1339, 1339, 1340, 1340, 1340, 1
340, 1340, 1340, 1341, 1341, 1341, 1341, 1341, 1341, 1342, 1342, 1342, 1342
, 1342, 1342, 1343, 1343, 1343, 1343, 1343, 1343, 1344, 1344, 1344, 1344, 1
344, 1344, 1345, 1345, 1345, 1345, 1345, 1345, 1346, 1346, 1346, 1346, 1346
, 1346, 1347, 1347, 1347, 1347, 1347, 1347, 1348, 1348, 1348, 1348, 1348, 1
348, 1349, 1349, 1349, 1349, 1349, 1349, 1350, 1350, 1350, 1350, 1350, 1350
, 1351, 1351, 1351, 1351, 1351, 1351, 1352, 1352, 1352, 1352, 1352, 1352, 1
353, 1353, 1353, 1353, 1353, 1353, 1354, 1354, 1354, 1354, 1354, 1354, 1355
, 1355, 1355, 1355, 1355, 1355, 1356, 1356, 1356, 1356, 1356, 1356, 1357, 1
357, 1357, 1357, 1357, 1357, 1358, 1358, 1358, 1358, 1358, 1358, 1359, 1359
, 1359, 1359, 1359, 1359, 1360, 1360, 1360, 1360, 1360, 1360, 1361, 1361, 1
361, 1361, 1361, 1361, 1362, 1362, 1362, 1362, 1362, 1362, 1363, 1363, 1363
, 1363, 1363, 1363, 1364, 1364, 1364, 1364, 1364, 1364, 1365, 1365, 1365, 1
365, 1365, 1365, 1366, 1366, 1366, 1366, 1366, 1366, 1367, 1367, 1367, 1367
, 1367, 1367, 1368, 1368, 1368, 1368, 1368, 1368, 1369, 1369, 1369, 1369, 1
369, 1369, 1370, 1370, 1370, 1370, 1370, 1370, 1371, 1371, 1371, 1371, 1371
, 1371, 1372, 1372, 1372, 1372, 1372, 1372, 1373, 1373, 1373, 1373, 1373, 1
373, 1374, 1374, 1374, 1374, 1374, 1374, 1375, 1375, 1375, 1375, 1375, 1375
, 1376, 1376, 1376, 1376, 1376, 1376, 1377, 1377, 1377, 1377, 1377, 1377, 1
378, 1378, 1378, 1378, 1378, 1378, 1379, 1379, 1379, 1379, 1379, 1379, 1380
, 1380, 1380, 1380, 1380, 1380, 1381, 1381, 1381, 1381, 1381, 1381, 1382, 1
382, 1382, 1382, 1382, 1382, 1383, 1383, 1383, 1383, 1383, 1383, 1384, 1384
, 1384, 1384, 1384, 1384, 1385, 1385, 1385, 1385, 1385, 1385, 1386, 1386, 1
386, 1386, 1386, 1386, 1387, 1387, 1387, 1387, 1387, 1387, 1388, 1388, 1388
, 1388, 1388, 1388, 1389, 1389, 1389, 1389, 1389, 1389, 1390, 1390, 1390, 1
390, 1390, 1390, 1391, 1391, 1391, 1391, 1391, 1391, 1392, 1392, 1392, 1392
, 1392, 1392, 1393, 1393, 1393, 1393, 1393, 1393, 1394, 1394, 1394, 1394, 1
394, 1394, 1395, 1395, 1395, 1395, 1395, 1395, 1396, 1396, 1396, 1396, 1396
, 1396, 1397, 1397, 1397, 1397, 1397, 1397, 1398, 1398, 1398, 1398, 1398, 1
398, 1399, 1399, 1399, 1399, 1399, 1399, 1400, 1400, 1400, 1400, 1400, 1400
, 1401, 1401, 1401, 1401, 1401, 1401, 1402, 1402, 1402, 1402, 1402, 1402, 1
403, 1403, 1403, 1403, 1403, 1403, 1404, 1404, 1404, 1404, 1404, 1404, 1405
, 1405, 1405, 1405, 1405, 1405, 1406, 1406, 1406, 1406, 1406, 1406, 1407, 1
407, 1407, 1407, 1407, 1407, 1408, 1408, 1408, 1408, 1408, 1408, 1409, 1409
, 1409, 1409, 1409, 1409, 1410, 1410, 1410, 1410, 1410, 1410, 1411, 1411, 1
411, 1411, 1411, 1411, 1412, 1412, 1412, 1412, 1412, 1412, 1413, 1413, 1413
, 1413, 1413, 1413, 1414, 1414, 1414, 1414, 1414, 1414, 1415, 1415, 1415, 1
415, 1415, 1415, 1416, 1416, 1416, 1416, 1416, 1416, 1417, 1417, 1417, 1417
, 1417, 1417, 1418, 1418, 1418, 1418, 1418, 1418, 1419, 1419, 1419, 1419, 1
419, 1419, 1420, 1420, 1420, 1420, 1420, 1420, 1421, 1421, 1421, 1421, 1421
, 1421, 1422, 1422, 1422, 1422, 1422, 1422, 1423, 1423, 1423, 1423, 1423, 1
423, 1424, 1424, 1424, 1424, 1424, 1424, 1425, 1425, 1425, 1425, 1425, 1425
, 1426, 1426, 1426, 1426, 1426, 1426, 1427, 1427, 1427, 1427, 1427, 1427, 1
428, 1428, 1428, 1428, 1428, 1428, 1429, 1429, 1429, 1429, 1429, 1429, 1430
, 1430, 1430, 1430, 1430, 1430, 1431, 1431, 1431, 1431, 1431, 1431, 1432, 1
432, 1432, 1432, 1432, 1432, 1433, 1433, 1433, 1433, 1433, 1433, 1434, 1434
, 1434, 1434, 1434, 1434, 1435, 1435, 1435, 1435, 1435, 1435, 1436, 1436, 1
436, 1436, 1436, 1436, 1437, 1437, 1437, 1437, 1437, 1437, 1438, 1438, 1438
, 1438, 1438, 1438, 1439, 1439, 1439, 1439, 1439, 1439, 1440, 1440, 1440, 1
440, 1440, 1440, 1441, 1441, 1441, 1441, 1441, 1441, 1442, 1442, 1442, 1442
, 1442, 1442, 1443, 1443, 1443, 1443, 1443, 1443, 1444, 1444, 1444, 1444, 1
444, 1444, 1445, 1445, 1445, 1445, 1445, 1445, 1446, 1446, 1446, 1446, 1446
, 1446, 1447, 1447, 1447, 1447, 1447, 1447, 1448, 1448, 1448, 1448, 1448, 1
448, 1449, 1449, 1449, 1449, 1449, 1449, 1450, 1450, 1450, 1450, 1450, 1450
, 1451, 1451, 1451, 1451, 1451, 1451, 1452, 1452, 1452, 1452, 1452, 1452, 1
453, 1453, 1453, 1453, 1453, 1453, 1454, 1454, 1454, 1454, 1454, 1454, 1455
, 1455, 1455, 1455, 1455, 1455, 1456, 1456, 1456, 1456, 1456, 1456, 1457, 1
457, 1457, 1457, 1457, 1457, 1458, 1458, 1458, 1458, 1458, 1458, 1459, 1459
, 1459, 1459, 1459, 1459, 1460, 1460, 1460, 1460, 1460, 1460, 1461, 1461, 1
461, 1461, 1461, 1461, 1462, 1462, 1462, 1462, 1462, 1462, 1463, 1463, 1463
, 1463, 1463, 1463, 1464, 1464, 1464, 1464, 1464, 1464, 1465, 1465, 1465, 1
465, 1465, 1465, 1466, 1466, 1466, 1466, 1466, 1466, 1467, 1467, 1467, 1467
, 1467, 1467, 1468, 1468, 1468, 1468, 1468, 1468, 1469, 1469, 1469, 1469, 1
469, 1469, 1470, 1470, 1470, 1470, 1470, 1470, 1471, 1471, 1471, 1471, 1471
, 1471, 1472, 1472, 1472, 1472, 1472, 1472, 1473, 1473, 1473, 1473, 1473, 1
473, 1474, 1474, 1474, 1474, 1474, 1474, 1475, 1475, 1475, 1475, 1475, 1475
, 1476, 1476, 1476, 1476, 1476, 1476, 1477, 1477, 1477, 1477, 1477, 1477, 1
478, 1478, 1478, 1478, 1478, 1478, 1479, 1479, 1479, 1479, 1479, 1479, 1480
, 1480, 1480, 1480, 1480, 1480, 1481, 1481, 1481, 1481, 1481, 1481, 1482, 1
482, 1482, 1482, 1482, 1482, 1483, 1483, 1483, 1483, 1483, 1483, 1484, 1484
, 1484, 1484, 1484, 1484, 1485, 1485, 1485, 1485, 1485, 1485, 1486, 1486, 1
486, 1486, 1486, 1486, 1487, 1487, 1487, 1487, 1487, 1487, 1488, 1488, 1488
, 1488, 1488, 1488, 1489, 1489, 1489, 1489, 1489, 1489, 1490, 1490, 1490, 1
490, 1490, 1490, 1491, 1491, 1491, 1491, 1491, 1491, 1492, 1492, 1492, 1492
, 1492, 1492, 1493, 1493, 1493, 1493, 1493, 1493, 1494, 1494, 1494, 1494, 1
494, 1494, 1495, 1495, 1495, 1495, 1495, 1495, 1496, 1496, 1496, 1496, 1496
, 1496, 1497, 1497, 1497, 1497, 1497, 1497, 1498, 1498, 1498, 1498, 1498, 1
498, 1499, 1499, 1499, 1499, 1499, 1499, 1500, 1500, 1500, 1500, 1500, 1500
, 1501, 1501, 1501, 1501, 1501, 1501, 1502, 1502, 1502, 1502, 1502, 1502, 1
503, 1503, 1503, 1503, 1503, 1503, 1504, 1504, 1504, 1504, 1504, 1504, 1505
, 1505, 1505, 1505, 1505, 1505, 1506, 1506, 1506, 1506, 1506, 1506, 1507, 1
507, 1507, 1507, 1507, 1507, 1508, 1508, 1508, 1508, 1508, 1508, 1509, 1509
, 1509, 1509, 1509, 1509, 1510, 1510, 1510, 1510, 1510, 1510, 1511, 1511, 1
511, 1511, 1511, 1511, 1512, 1512, 1512, 1512, 1512, 1512, 1513, 1513, 1513
, 1513, 1513, 1513, 1514, 1514, 1514, 1514, 1514, 1514, 1515, 1515, 1515, 1
515, 1515, 1515, 1516, 1516, 1516, 1516, 1516, 1516, 1517, 1517, 1517, 1517
, 1517, 1517, 1518, 1518, 1518, 1518, 1518, 1518, 1519, 1519, 1519, 1519, 1
519, 1519, 1520, 1520, 1520, 1520, 1520, 1520, 1521, 1521, 1521, 1521, 1521
, 1521, 1522, 1522, 1522, 1522, 1522, 1522, 1523, 1523, 1523, 1523, 1523, 1
523, 1524, 1524, 1524, 1524, 1524, 1524, 1525, 1525, 1525, 1525, 1525, 1525
, 1526, 1526, 1526, 1526, 1526, 1526, 1527, 1527, 1527, 1527, 1527, 1527, 1
528, 1528, 1528, 1528, 1528, 1528, 1529, 1529, 1529, 1529, 1529, 1529, 1530
, 1530, 1530, 1530, 1530, 1530, 1531, 1531, 1531, 1531, 1531, 1531, 1532, 1
532, 1532, 1532, 1532, 1532, 1533, 1533, 1533, 1533, 1533, 1533, 1534, 1534
, 1534, 1534, 1534, 1534, 1535, 1535, 1535, 1535, 1535, 1535, 1536, 1536, 1
536, 1536, 1536, 1536, 1537, 1537, 1537, 1537, 1537, 1537, 1538, 1538, 1538
, 1538, 1538, 1538, 1539, 1539, 1539, 1539, 1539, 1539, 1540, 1540, 1540, 1
540, 1540, 1540, 1541, 1541, 1541, 1541, 1541, 1541, 1542, 1542, 1542, 1542
, 1542, 1542, 1543, 1543, 1543, 1543, 1543, 1543, 1544, 1544, 1544, 1544, 1
544, 1544, 1545, 1545, 1545, 1545, 1545, 1545, 1546, 1546, 1546, 1546, 1546
, 1546, 1547, 1547, 1547, 1547, 1547, 1547, 1548, 1548, 1548, 1548, 1548, 1
548, 1549, 1549, 1549, 1549, 1549, 1549, 1550, 1550, 1550, 1550, 1550, 1550
, 1551, 1551, 1551, 1551, 1551, 1551, 1552, 1552, 1552, 1552, 1552, 1552, 1
553, 1553, 1553, 1553, 1553, 1553, 1554, 1554, 1554, 1554, 1554, 1554, 1555
, 1555, 1555, 1555, 1555, 1555, 1556, 1556, 1556, 1556, 1556, 1556, 1557, 1
557, 1557, 1557, 1557, 1557, 1558, 1558, 1558, 1558, 1558, 1558, 1559, 1559
, 1559, 1559, 1559, 1559, 1560, 1560, 1560, 1560, 1560, 1560, 1561, 1561, 1
561, 1561, 1561, 1561, 1562, 1562, 1562, 1562, 1562, 1562, 1563, 1563, 1563
, 1563, 1563, 1563, 1564, 1564, 1564, 1564, 1564, 1564, 1565, 1565, 1565, 1
565, 1565, 1565, 1566, 1566, 1566, 1566, 1566, 1566, 1567, 1567, 1567, 1567
, 1567, 1567, 1568, 1568, 1568, 1568, 1568, 1568, 1569, 1569, 1569, 1569, 1
569, 1569, 1570, 1570, 1570, 1570, 1570, 1570, 1571, 1571, 1571, 1571, 1571
, 1571, 1572, 1572, 1572, 1572, 1572, 1572, 1573, 1573, 1573, 1573, 1573, 1
573, 1574, 1574, 1574, 1574, 1574, 1574, 1575, 1575, 1575, 1575, 1575, 1575
, 1576, 1576, 1576, 1576, 1576, 1576, 1577, 1577, 1577, 1577, 1577, 1577, 1
578, 1578, 1578, 1578, 1578, 1578, 1579, 1579, 1579, 1579, 1579, 1579, 1580
, 1580, 1580, 1580, 1580, 1580, 1581, 1581, 1581, 1581, 1581, 1581, 1582, 1
582, 1582, 1582, 1582, 1582, 1583, 1583, 1583, 1583, 1583, 1583, 1584, 1584
, 1584, 1584, 1584, 1584, 1585, 1585, 1585, 1585, 1585, 1585, 1586, 1586, 1
586, 1586, 1586, 1586, 1587, 1587, 1587, 1587, 1587, 1587, 1588, 1588, 1588
, 1588, 1588, 1588, 1589, 1589, 1589, 1589, 1589, 1589, 1590, 1590, 1590, 1
590, 1590, 1590, 1591, 1591, 1591, 1591, 1591, 1591, 1592, 1592, 1592, 1592
, 1592, 1592, 1593, 1593, 1593, 1593, 1593, 1593, 1594, 1594, 1594, 1594, 1
594, 1594, 1595, 1595, 1595, 1595, 1595, 1595, 1596, 1596, 1596, 1596, 1596
, 1596, 1597, 1597, 1597, 1597, 1597, 1597, 1598, 1598, 1598, 1598, 1598, 1
598, 1599, 1599, 1599, 1599, 1599, 1599, 1600, 1600, 1600, 1600, 1600, 1600
, 1601, 1601, 1601, 1601, 1601, 1601, 1602, 1602, 1602, 1602, 1602, 1602, 1
603, 1603, 1603, 1603, 1603, 1603, 1604, 1604, 1604, 1604, 1604, 1604, 1605
, 1605, 1605, 1605, 1605, 1605, 1606, 1606, 1606, 1606, 1606, 1606, 1607, 1
607, 1607, 1607, 1607, 1607, 1608, 1608, 1608, 1608, 1608, 1608, 1609, 1609
, 1609, 1609, 1609, 1609, 1610, 1610, 1610, 1610, 1610, 1610, 1611, 1611, 1
611, 1611, 1611, 1611, 1612, 1612, 1612, 1612, 1612, 1612, 1613, 1613, 1613
, 1613, 1613, 1613, 1614, 1614, 1614, 1614, 1614, 1614, 1615, 1615, 1615, 1
615, 1615, 1615, 1616, 1616, 1616, 1616, 1616, 1616, 1617, 1617, 1617, 1617
, 1617, 1617, 1618, 1618, 1618, 1618, 1618, 1618, 1619, 1619, 1619, 1619, 1
619, 1619, 1620, 1620, 1620, 1620, 1620, 1620, 1621, 1621, 1621, 1621, 1621
, 1621, 1622, 1622, 1622, 1622, 1622, 1622, 1623, 1623, 1623, 1623, 1623, 1
623, 1624, 1624, 1624, 1624, 1624, 1624, 1625, 1625, 1625, 1625, 1625, 1625
, 1626, 1626, 1626, 1626, 1626, 1626, 1627, 1627, 1627, 1627, 1627, 1627, 1
628, 1628, 1628, 1628, 1628, 1628, 1629, 1629, 1629, 1629, 1629, 1629, 1630
, 1630, 1630, 1630, 1630, 1630, 1631, 1631, 1631, 1631, 1631, 1631, 1632, 1
632, 1632, 1632, 1632, 1632, 1633, 1633, 1633, 1633, 1633, 1633, 1634, 1634
, 1634, 1634, 1634, 1634, 1635, 1635, 1635, 1635, 1635, 1635, 1636, 1636, 1
636, 1636, 1636, 1636, 1637, 1637, 1637, 1637, 1637, 1637, 1638, 1638, 1638
, 1638, 1638, 1638, 1639, 1639, 1639, 1639, 1639, 1639, 1640, 1640, 1640, 1
640, 1640, 1640, 1641, 1641, 1641, 1641, 1641, 1641, 1642, 1642, 1642, 1642
, 1642, 1642, 1643, 1643, 1643, 1643, 1643, 1643, 1644, 1644, 1644, 1644, 1
644, 1644, 1645, 1645, 1645, 1645, 1645, 1645, 1646, 1646, 1646, 1646, 1646
, 1646, 1647, 1647, 1647, 1647, 1647, 1647, 1648, 1648, 1648, 1648, 1648, 1
648, 1649, 1649, 1649, 1649, 1649, 1649, 1650, 1650, 1650, 1650, 1650, 1650
, 1651, 1651, 1651, 1651, 1651, 1651, 1652, 1652, 1652, 1652, 1652, 1652, 1
653, 1653, 1653, 1653, 1653, 1653, 1654, 1654, 1654, 1654, 1654, 1654, 1655
, 1655, 1655, 1655, 1655, 1655, 1656, 1656, 1656, 1656, 1656, 1656, 1657, 1
657, 1657, 1657, 1657, 1657, 1658, 1658, 1658, 1658, 1658, 1658, 1659, 1659
, 1659, 1659, 1659, 1659, 1660, 1660, 1660, 1660, 1660, 1660, 1661, 1661, 1
661, 1661, 1661, 1661, 1662, 1662, 1662, 1662, 1662, 1662, 1663, 1663, 1663
, 1663, 1663, 1663, 1664, 1664, 1664, 1664, 1664, 1664, 1665, 1665, 1665, 1
665, 1665, 1665, 1666, 1666, 1666, 1666, 1666, 1666, 1667, 1667, 1667, 1667
, 1667, 1667, 1668, 1668, 1668, 1668, 1668, 1668, 1669, 1669, 1669, 1669, 1
669, 1669, 1670, 1670, 1670, 1670, 1670, 1670, 1671, 1671, 1671, 1671, 1671
, 1671, 1672, 1672, 1672, 1672, 1672, 1672, 1673, 1673, 1673, 1673, 1673, 1
673, 1674, 1674, 1674, 1674, 1674, 1674, 1675, 1675, 1675, 1675, 1675, 1675
, 1676, 1676, 1676, 1676, 1676, 1676, 1677, 1677, 1677, 1677, 1677, 1677, 1
678, 1678, 1678, 1678, 1678, 1678, 1679, 1679, 1679, 1679, 1679, 1679, 1680
, 1680, 1680, 1680, 1680, 1680, 1681, 1681, 1681, 1681, 1681, 1681, 1682, 1
682, 1682, 1682, 1682, 1682, 1683, 1683, 1683, 1683, 1683, 1683, 1684, 1684
, 1684, 1684, 1684, 1684, 1685, 1685, 1685, 1685, 1685, 1685, 1686, 1686, 1
686, 1686, 1686, 1686, 1687, 1687, 1687, 1687, 1687, 1687, 1688, 1688, 1688
, 1688, 1688, 1688, 1689, 1689, 1689, 1689, 1689, 1689, 1690, 1690, 1690, 1
690, 1690, 1690, 1691, 1691, 1691, 1691, 1691, 1691, 1692, 1692, 1692, 1692
, 1692, 1692, 1693, 1693, 1693, 1693, 1693, 1693, 1694, 1694, 1694, 1694, 1
694, 1694, 1695, 1695, 1695, 1695, 1695, 1695, 1696, 1696, 1696, 1696, 1696
, 1696, 1697, 1697, 1697, 1697, 1697, 1697, 1698, 1698, 1698, 1698, 1698, 1
698, 1699, 1699, 1699, 1699, 1699, 1699, 1700, 1700, 1700, 1700, 1700, 1700
, 1701, 1701, 1701, 1701, 1701, 1701, 1702, 1702, 1702, 1702, 1702, 1702, 1
703, 1703, 1703, 1703, 1703, 1703, 1704, 1704, 1704, 1704, 1704, 1704, 1705
, 1705, 1705, 1705, 1705, 1705, 1706, 1706, 1706, 1706, 1706, 1706, 1707, 1
707, 1707, 1707, 1707, 1707, 1708, 1708, 1708, 1708, 1708, 1708, 1709, 1709
, 1709, 1709, 1709, 1709, 1710, 1710, 1710, 1710, 1710, 1710, 1711, 1711, 1
711, 1711, 1711, 1711, 1712, 1712, 1712, 1712, 1712, 1712, 1713, 1713, 1713
, 1713, 1713, 1713, 1714, 1714, 1714, 1714, 1714, 1714, 1715, 1715, 1715, 1
715, 1715, 1715, 1716, 1716, 1716, 1716, 1716, 1716, 1717, 1717, 1717, 1717
, 1717, 1717, 1718, 1718, 1718, 1718, 1718, 1718, 1719, 1719, 1719, 1719, 1
719, 1719, 1720, 1720, 1720, 1720, 1720, 1720, 1721, 1721, 1721, 1721, 1721
, 1721, 1722, 1722, 1722, 1722, 1722, 1722, 1723, 1723, 1723, 1723, 1723, 1
723, 1724, 1724, 1724, 1724, 1724, 1724, 1725, 1725, 1725, 1725, 1725, 1725
, 1726, 1726, 1726, 1726, 1726, 1726, 1727, 1727, 1727, 1727, 1727, 1727, 1
728, 1728, 1728, 1728, 1728, 1728, 1729, 1729, 1729, 1729, 1729, 1729, 1730
, 1730, 1730, 1730, 1730, 1730, 1731, 1731, 1731, 1731, 1731, 1731, 1732, 1
732, 1732, 1732, 1732, 1732, 1733, 1733, 1733, 1733, 1733, 1733, 1734, 1734
, 1734, 1734, 1734, 1734, 1735, 1735, 1735, 1735, 1735, 1735, 1736, 1736, 1
736, 1736, 1736, 1736, 1737, 1737, 1737, 1737, 1737, 1737, 1738, 1738, 1738
, 1738, 1738, 1738, 1739, 1739, 1739, 1739, 1739, 1739, 1740, 1740, 1740, 1
740, 1740, 1740, 1741, 1741, 1741, 1741, 1741, 1741, 1742, 1742, 1742, 1742
, 1742, 1742, 1743, 1743, 1743, 1743, 1743, 1743, 1744, 1744, 1744, 1744, 1
744, 1744, 1745, 1745, 1745, 1745, 1745, 1745, 1746, 1746, 1746, 1746, 1746
, 1746, 1747, 1747, 1747, 1747, 1747, 1747, 1748, 1748, 1748, 1748, 1748, 1
748, 1749, 1749, 1749, 1749, 1749, 1749, 1750, 1750, 1750, 1750, 1750, 1750
, 1751, 1751, 1751, 1751, 1751, 1751, 1752, 1752, 1752, 1752, 1752, 1752, 1
753, 1753, 1753, 1753, 1753, 1753, 1754, 1754, 1754, 1754, 1754, 1754, 1755
, 1755, 1755, 1755, 1755, 1755, 1756, 1756, 1756, 1756, 1756, 1756, 1757, 1
757, 1757, 1757, 1757, 1757, 1758, 1758, 1758, 1758, 1758, 1758, 1759, 1759
, 1759, 1759, 1759, 1759, 1760, 1760, 1760, 1760, 1760, 1760, 1761, 1761, 1
761, 1761, 1761, 1761, 1762, 1762, 1762, 1762, 1762, 1762, 1763, 1763, 1763
, 1763, 1763, 1763, 1764, 1764, 1764, 1764, 1764, 1764, 1765, 1765, 1765, 1
765, 1765, 1765, 1766, 1766, 1766, 1766, 1766, 1766, 1767, 1767, 1767, 1767
, 1767, 1767, 1768, 1768, 1768, 1768, 1768, 1768, 1769, 1769, 1769, 1769, 1
769, 1769, 1770, 1770, 1770, 1770, 1770, 1770, 1771, 1771, 1771, 1771, 1771
, 1771, 1772, 1772, 1772, 1772, 1772, 1772, 1773, 1773, 1773, 1773, 1773, 1
773, 1774, 1774, 1774, 1774, 1774, 1774, 1775, 1775, 1775, 1775, 1775, 1775
, 1776, 1776, 1776, 1776, 1776, 1776, 1777, 1777, 1777, 1777, 1777, 1777, 1
778, 1778, 1778, 1778, 1778, 1778, 1779, 1779, 1779, 1779, 1779, 1779, 1780
, 1780, 1780, 1780, 1780, 1780, 1781, 1781, 1781, 1781, 1781, 1781, 1782, 1
782, 1782, 1782, 1782, 1782, 1783, 1783, 1783, 1783, 1783, 1783, 1784, 1784
, 1784, 1784, 1784, 1784, 1785, 1785, 1785, 1785, 1785, 1785, 1786, 1786, 1
786, 1786, 1786, 1786, 1787, 1787, 1787, 1787, 1787, 1787, 1788, 1788, 1788
, 1788, 1788, 1788, 1789, 1789, 1789, 1789, 1789, 1789, 1790, 1790, 1790, 1
790, 1790, 1790, 1791, 1791, 1791, 1791, 1791, 1791, 1792, 1792, 1792, 1792
, 1792, 1792, 1793, 1793, 1793, 1793, 1793, 1793, 1794, 1794, 1794, 1794, 1
794, 1794, 1795, 1795, 1795, 1795, 1795, 1795, 1796, 1796, 1796, 1796, 1796
, 1796, 1797, 1797, 1797, 1797, 1797, 1797, 1798, 1798, 1798, 1798, 1798, 1
798, 1799, 1799, 1799, 1799, 1799, 1799, 1800, 1800, 1800, 1800, 1800, 1800
, 1801, 1801, 1801, 1801, 1801, 1801, 1802, 1802, 1802, 1802, 1802, 1802, 1
803, 1803, 1803, 1803, 1803, 1803, 1804, 1804, 1804, 1804, 1804, 1804, 1805
, 1805, 1805, 1805, 1805, 1805, 1806, 1806, 1806, 1806, 1806, 1806, 1807, 1
807, 1807, 1807, 1807, 1807, 1808, 1808, 1808, 1808, 1808, 1808, 1809, 1809
, 1809, 1809, 1809, 1809, 1810, 1810, 1810, 1810, 1810, 1810, 1811, 1811, 1
811, 1811, 1811, 1811, 1812, 1812, 1812, 1812, 1812, 1812, 1813, 1813, 1813
, 1813, 1813, 1813, 1814, 1814, 1814, 1814, 1814, 1814, 1815, 1815, 1815, 1
815, 1815, 1815, 1816, 1816, 1816, 1816, 1816, 1816, 1817, 1817, 1817, 1817
, 1817, 1817, 1818, 1818, 1818, 1818, 1818, 1818, 1819, 1819, 1819, 1819, 1
819, 1819, 1820, 1820, 1820, 1820, 1820, 1820, 1821, 1821, 1821, 1821, 1821
, 1821, 1822, 1822, 1822, 1822, 1822, 1822, 1823, 1823, 1823, 1823, 1823, 1
823, 1824, 1824, 1824, 1824, 1824, 1824, 1825, 1825, 1825, 1825, 1825, 1825
, 1826, 1826, 1826, 1826, 1826, 1826, 1827, 1827, 1827, 1827, 1827, 1827, 1
828, 1828, 1828, 1828, 1828, 1828, 1829, 1829, 1829, 1829, 1829, 1829, 1830
, 1830, 1830, 1830, 1830, 1830, 1831, 1831, 1831, 1831, 1831, 1831, 1832, 1
832, 1832, 1832, 1832, 1832, 1833, 1833, 1833, 1833, 1833, 1833, 1834, 1834
, 1834, 1834, 1834, 1834, 1835, 1835, 1835, 1835, 1835, 1835, 1836, 1836, 1
836, 1836, 1836, 1836, 1837, 1837, 1837, 1837, 1837, 1837, 1838, 1838, 1838
, 1838, 1838, 1838, 1839, 1839, 1839, 1839, 1839, 1839, 1840, 1840, 1840, 1
840, 1840, 1840, 1841, 1841, 1841, 1841, 1841, 1841, 1842, 1842, 1842, 1842
, 1842, 1842, 1843, 1843, 1843, 1843, 1843, 1843, 1844, 1844, 1844, 1844, 1
844, 1844, 1845, 1845, 1845, 1845, 1845, 1845, 1846, 1846, 1846, 1846, 1846
, 1846, 1847, 1847, 1847, 1847, 1847, 1847, 1848, 1848, 1848, 1848, 1848, 1
848, 1849, 1849, 1849, 1849, 1849, 1849, 1850, 1850, 1850, 1850, 1850, 1850
, 1851, 1851, 1851, 1851, 1851, 1851, 1852, 1852, 1852, 1852, 1852, 1852, 1
853, 1853, 1853, 1853, 1853, 1853, 1854, 1854, 1854, 1854, 1854, 1854, 1855
, 1855, 1855, 1855, 1855, 1855, 1856, 1856, 1856, 1856, 1856, 1856, 1857, 1
857, 1857, 1857, 1857, 1857, 1858, 1858, 1858, 1858, 1858, 1858, 1859, 1859
, 1859, 1859, 1859, 1859, 1860, 1860, 1860, 1860, 1860, 1860, 1861, 1861, 1
861, 1861, 1861, 1861, 1862, 1862, 1862, 1862, 1862, 1862, 1863, 1863, 1863
, 1863, 1863, 1863, 1864, 1864, 1864, 1864, 1864, 1864, 1865, 1865, 1865, 1
865, 1865, 1865, 1866, 1866, 1866, 1866, 1866, 1866, 1867, 1867, 1867, 1867
, 1867, 1867, 1868, 1868, 1868, 1868, 1868, 1868, 1869, 1869, 1869, 1869, 1
869, 1869, 1870, 1870, 1870, 1870, 1870, 1870, 1871, 1871, 1871, 1871, 1871
, 1871, 1872, 1872, 1872, 1872, 1872, 1872, 1873, 1873, 1873, 1873, 1873, 1
873, 1874, 1874, 1874, 1874, 1874, 1874, 1875, 1875, 1875, 1875, 1875, 1875
, 1876, 1876, 1876, 1876, 1876, 1876, 1877, 1877, 1877, 1877, 1877, 1877, 1
878, 1878, 1878, 1878, 1878, 1878, 1879, 1879, 1879, 1879, 1879, 1879, 1880
, 1880, 1880, 1880, 1880, 1880, 1881, 1881, 1881, 1881, 1881, 1881, 1882, 1
882, 1882, 1882, 1882, 1882, 1883, 1883, 1883, 1883, 1883, 1883, 1884, 1884
, 1884, 1884, 1884, 1884, 1885, 1885, 1885, 1885, 1885, 1885, 1886, 1886, 1
886, 1886, 1886, 1886, 1887, 1887, 1887, 1887, 1887, 1887, 1888, 1888, 1888
, 1888, 1888, 1888, 1889, 1889, 1889, 1889, 1889, 1889, 1890, 1890, 1890, 1
890, 1890, 1890, 1891, 1891, 1891, 1891, 1891, 1891, 1892, 1892, 1892, 1892
, 1892, 1892, 1893, 1893, 1893, 1893, 1893, 1893, 1894, 1894, 1894, 1894, 1
894, 1894, 1895, 1895, 1895, 1895, 1895, 1895, 1896, 1896, 1896, 1896, 1896
, 1896, 1897, 1897, 1897, 1897, 1897, 1897, 1898, 1898, 1898, 1898, 1898, 1
898, 1899, 1899, 1899, 1899, 1899, 1899, 1900, 1900, 1900, 1900, 1900, 1900
, 1901, 1901, 1901, 1901, 1901, 1901, 1902, 1902, 1902, 1902, 1902, 1902, 1
903, 1903, 1903, 1903, 1903, 1903, 1904, 1904, 1904, 1904, 1904, 1904, 1905
, 1905, 1905, 1905, 1905, 1905, 1906, 1906, 1906, 1906, 1906, 1906, 1907, 1
907, 1907, 1907, 1907, 1907, 1908, 1908, 1908, 1908, 1908, 1908, 1909, 1909
, 1909, 1909, 1909, 1909, 1910, 1910, 1910, 1910, 1910, 1910, 1911, 1911, 1
911, 1911, 1911, 1911, 1912, 1912, 1912, 1912, 1912, 1912, 1913, 1913, 1913
, 1913, 1913, 1913, 1914, 1914, 1914, 1914, 1914, 1914, 1915, 1915, 1915, 1
915, 1915, 1915, 1916, 1916, 1916, 1916, 1916, 1916, 1917, 1917, 1917, 1917
, 1917, 1917, 1918, 1918, 1918, 1918, 1918, 1918, 1919, 1919, 1919, 1919, 1
919, 1919, 1920, 1920, 1920, 1920, 1920, 1920, 1921, 1921, 1921, 1921, 1921
, 1921, 1922, 1922, 1922, 1922, 1922, 1922, 1923, 1923, 1923, 1923, 1923, 1
923, 1924, 1924, 1924, 1924, 1924, 1924, 1925, 1925, 1925, 1925, 1925, 1925
, 1926, 1926, 1926, 1926, 1926, 1926, 1927, 1927, 1927, 1927, 1927, 1927, 1
928, 1928, 1928, 1928, 1928, 1928, 1929, 1929, 1929, 1929, 1929, 1929, 1930
, 1930, 1930, 1930, 1930, 1930, 1931, 1931, 1931, 1931, 1931, 1931, 1932, 1
932, 1932, 1932, 1932, 1932, 1933, 1933, 1933, 1933, 1933, 1933, 1934, 1934
, 1934, 1934, 1934, 1934, 1935, 1935, 1935, 1935, 1935, 1935, 1936, 1936, 1
936, 1936, 1936, 1936, 1937, 1937, 1937, 1937, 1937, 1937, 1938, 1938, 1938
, 1938, 1938, 1938, 1939, 1939, 1939, 1939, 1939, 1939, 1940, 1940, 1940, 1
940, 1940, 1940, 1941, 1941, 1941, 1941, 1941, 1941, 1942, 1942, 1942, 1942
, 1942, 1942, 1943, 1943, 1943, 1943, 1943, 1943, 1944, 1944, 1944, 1944, 1
944, 1944, 1945, 1945, 1945, 1945, 1945, 1945, 1946, 1946, 1946, 1946, 1946
, 1946, 1947, 1947, 1947, 1947, 1947, 1947, 1948, 1948, 1948, 1948, 1948, 1
948, 1949, 1949, 1949, 1949, 1949, 1949, 1950, 1950, 1950, 1950, 1950, 1950
, 1951, 1951, 1951, 1951, 1951, 1951, 1952, 1952, 1952, 1952, 1952, 1952, 1
953, 1953, 1953, 1953, 1953, 1953, 1954, 1954, 1954, 1954, 1954, 1954, 1955
, 1955, 1955, 1955, 1955, 1955, 1956, 1956, 1956, 1956, 1956, 1956, 1957, 1
957, 1957, 1957, 1957, 1957, 1958, 1958, 1958, 1958, 1958, 1958, 1959, 1959
, 1959, 1959, 1959, 1959, 1960, 1960, 1960, 1960, 1960, 1960, 1961, 1961, 1
961, 1961, 1961, 1961, 1962, 1962, 1962, 1962, 1962, 1962, 1963, 1963, 1963
, 1963, 1963, 1963, 1964, 1964, 1964, 1964, 1964, 1964, 1965, 1965, 1965, 1
965, 1965, 1965, 1966, 1966, 1966, 1966, 1966, 1966, 1967, 1967, 1967, 1967
, 1967, 1967, 1968, 1968, 1968, 1968, 1968, 1968, 1969, 1969, 1969, 1969, 1
969, 1969, 1970, 1970, 1970, 1970, 1970, 1970, 1971, 1971, 1971, 1971, 1971
, 1971, 1972, 1972, 1972, 1972, 1972, 1972, 1973, 1973, 1973, 1973, 1973, 1
973, 1974, 1974, 1974, 1974, 1974, 1974, 1975, 1975, 1975, 1975, 1975, 1975
, 1976, 1976, 1976, 1976, 1976, 1976, 1977, 1977, 1977, 1977, 1977, 1977, 1
978, 1978, 1978, 1978, 1978, 1978, 1979, 1979, 1979, 1979, 1979, 1979, 1980
, 1980, 1980, 1980, 1980, 1980, 1981, 1981, 1981, 1981, 1981, 1981, 1982, 1
982, 1982, 1982, 1982, 1982, 1983, 1983, 1983, 1983, 1983, 1983, 1984, 1984
, 1984, 1984, 1984, 1984, 1985, 1985, 1985, 1985, 1985, 1985, 1986, 1986, 1
986, 1986, 1986, 1986, 1987, 1987, 1987, 1987, 1987, 1987, 1988, 1988, 1988
, 1988, 1988, 1988, 1989, 1989, 1989, 1989, 1989, 1989, 1990, 1990, 1990, 1
990, 1990, 1990, 1991, 1991, 1991, 1991, 1991, 1991, 1992, 1992, 1992, 1992
, 1992, 1992, 1993, 1993, 1993, 1993, 1993, 1993, 1994, 1994, 1994, 1994, 1
994, 1994, 1995, 1995, 1995, 1995, 1995, 1995, 1996, 1996, 1996, 1996, 1996
, 1996, 1997, 1997, 1997, 1997, 1997, 1997, 1998, 1998, 1998, 1998, 1998, 1
998, 1999, 1999, 1999, 1999, 1999, 1999, 2000, 2000, 2000, 2000, 2000, 2000
, 2001, 2001, 2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002, 2002, 2002, 2
003, 2003, 2003, 2003, 2003, 2003, 2004, 2004, 2004, 2004, 2004, 2004, 2005
, 2005, 2005, 2005, 2005, 2005, 2006, 2006, 2006, 2006, 2006, 2006, 2007, 2
007, 2007, 2007, 2007, 2007, 2008, 2008, 2008, 2008, 2008, 2008, 2009, 2009
, 2009, 2009, 2009, 2009, 2010, 2010, 2010, 2010, 2010, 2010, 2011, 2011, 2
011, 2011, 2011, 2011, 2012, 2012, 2012, 2012, 2012, 2012, 2013, 2013, 2013
, 2013, 2013, 2013, 2014, 2014, 2014, 2014, 2014, 2014, 2015, 2015, 2015, 2
015, 2015, 2015, 2016, 2016, 2016, 2016, 2016, 2016, 2017, 2017, 2017, 2017
, 2017, 2017, 2018, 2018, 2018, 2018, 2018, 2018, 2019, 2019, 2019, 2019, 2
019, 2019, 2020, 2020, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2021
, 2021, 2022, 2022, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2023, 2
023, 2024, 2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025, 2025, 2025
, 2026, 2026, 2026, 2026, 2026, 2026, 2027, 2027, 2027, 2027, 2027, 2027, 2
028, 2028, 2028, 2028, 2028, 2028, 2029, 2029, 2029, 2029, 2029, 2029, 2030
, 2030, 2030, 2030, 2030, 2030, 2031, 2031, 2031, 2031, 2031, 2031, 2032, 2
032, 2032, 2032, 2032, 2032, 2033, 2033, 2033, 2033, 2033, 2033, 2034, 2034
, 2034, 2034, 2034, 2034, 2035, 2035, 2035, 2035, 2035, 2035, 2036, 2036, 2
036, 2036, 2036, 2036, 2037, 2037, 2037, 2037, 2037, 2037, 2038, 2038, 2038
, 2038, 2038, 2038, 2039, 2039, 2039, 2039, 2039, 2039, 2040, 2040, 2040, 2
040, 2040, 2040, 2041, 2041, 2041, 2041, 2041, 2041, 2042, 2042, 2042, 2042
, 2042, 2042, 2043, 2043, 2043, 2043, 2043, 2043, 2044, 2044, 2044, 2044, 2
044, 2044, 2045, 2045, 2045, 2045, 2045, 2045, 2046, 2046, 2046, 2046, 2046
, 2046, 2047, 2047, 2047, 2047, 2047, 2047, 2048, 2048, 2048, 2048, 2048, 2
048], [-38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.00000
0000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 961
0.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.00000000000
2, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000
002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.00
0000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 
9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.00000000
0002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000
000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610
.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.4000000000
1, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.00000
0000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000
000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9
610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.4000000
0001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.00
0000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.
000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002
, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.4000
0000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610
.000000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 96
10.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000
002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.4
0000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9
610.000000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002,
 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000
000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -3844
4.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4
, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0000000000
02, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000
000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, -3
8444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 
3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0000000
00002, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.
000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002,
 -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.00000000000
2, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0000
00000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 96
10.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.0000000000
02, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.00000000
0002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001,
 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.0000000
00002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.00000
0000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 961
0.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.00000000000
2, -38444.36385817808, 9610.000000000002, 9610.000000000002, 3.363858178071
0384, 9610.000000000002, 9610.000000000002, -38444.30284388282, 9610.000000
000002, 9610.000000000002, 3.30284388280787, 9610.000000000002, 9610.000000
000002, -38444.23066486092, 9610.000000000002, 9610.000000000002, 3.2306648
609131807, 9610.000000000002, 9610.000000000002, -38444.15313257167, 9610.0
00000000002, 9610.000000000002, 3.1531325716591234, 9610.000000000002, 9610
.000000000002, -38444.073980791916, 9610.000000000002, 9610.000000000002, 3
.073980791904124, 9610.000000000002, 9610.000000000002, -38443.99592214669,
 9610.000000000002, 9610.000000000002, 2.9959221466807455, 9610.00000000000
2, 9610.000000000002, -38443.92104789944, 9610.000000000002, 9610.000000000
002, 2.9210478994341713, 9610.000000000002, 9610.000000000002, -38443.85102
205872, 9610.000000000002, 9610.000000000002, 2.851022058711155, 9610.00000
0000002, 9610.000000000002, -38443.7871898568, 9610.000000000002, 9610.0000
00000002, 2.7871898567876396, 9610.000000000002, 9610.000000000002, -38443.
73064409074, 9610.000000000002, 9610.000000000002, 2.7306440907303067, 9610
.000000000002, 9610.000000000002, -38443.68226821267, 9610.000000000002, 96
10.000000000002, 2.682268212664803, 9610.000000000002, 9610.000000000002, -
38443.64276543411, 9610.000000000002, 9610.000000000002, 2.64276543410358, 
9610.000000000002, 9610.000000000002, -38443.61267879984, 9610.000000000002
, 9610.000000000002, 2.6126787998251304, 9610.000000000002, 9610.0000000000
02, -38443.59240504866, 9610.000000000002, 9610.000000000002, 2.59240504865
0707, 9610.000000000002, 9610.000000000002, -38443.582203924154, 9610.00000
0000002, 9610.000000000002, 2.5822039241470254, 9610.000000000002, 9610.000
000000002, -38443.582203924154, 9610.000000000002, 9610.000000000002, 2.582
2039241470254, 9610.000000000002, 9610.000000000002, -38443.59240504866, 96
10.000000000002, 9610.000000000002, 2.592405048650707, 9610.000000000002, 9
610.000000000002, -38443.61267879984, 9610.000000000002, 9610.000000000002,
 2.6126787998251304, 9610.000000000002, 9610.000000000002, -38443.642765434
11, 9610.000000000002, 9610.000000000002, 2.64276543410358, 9610.0000000000
02, 9610.000000000002, -38443.68226821267, 9610.000000000002, 9610.00000000
0002, 2.682268212664803, 9610.000000000002, 9610.000000000002, -38443.73064
409074, 9610.000000000002, 9610.000000000002, 2.7306440907303067, 9610.0000
00000002, 9610.000000000002, -38443.7871898568, 9610.000000000002, 9610.000
000000002, 2.78718985678764, 9610.000000000002, 9610.000000000002, -38443.8
5102205872, 9610.000000000002, 9610.000000000002, 2.851022058711155, 9610.0
00000000002, 9610.000000000002, -38443.92104789944, 9610.000000000002, 9610
.000000000002, 2.9210478994341713, 9610.000000000002, 9610.000000000002, -3
8443.99592214669, 9610.000000000002, 9610.000000000002, 2.9959221466807455,
 9610.000000000002, 9610.000000000002, -38444.073980791916, 9610.0000000000
02, 9610.000000000002, 3.0739807919041247, 9610.000000000002, 9610.00000000
0002, -38444.15313257167, 9610.000000000002, 9610.000000000002, 3.153132571
6591234, 9610.000000000002, 9610.000000000002, -38444.23066486092, 9610.000
000000002, 9610.000000000002, 3.2306648609131803, 9610.000000000002, 9610.0
00000000002, -38444.30284388282, 9610.000000000002, 9610.000000000002, 3.30
284388280787, 9610.000000000002, 9610.000000000002, -38444.36385817808, 961
0.000000000002, 9610.000000000002, 3.3638581780710384, 9610.000000000002, 9
610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002,
 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.000000
000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -3844
4.30284388282, 9610.000000000002, 9610.000000000002, 3.30284388280787, 9610
.000000000002, 9610.000000000002, -38444.13882580889, 9610.000000000002, 96
10.000000000002, 3.138825808881349, 9610.000000000002, 9610.000000000002, -
38443.944794818315, 9610.000000000002, 9610.000000000002, 2.944794818307675
3, 9610.000000000002, 9610.000000000002, -38443.73637322309, 9610.000000000
002, 9610.000000000002, 2.7363732230779934, 9610.000000000002, 9610.0000000
00002, -38443.52359812821, 9610.000000000002, 9610.000000000002, 2.52359812
82043167, 9610.000000000002, 9610.000000000002, -38443.31376158765, 9610.00
0000000002, 9610.000000000002, 2.3137615876423077, 9610.000000000002, 9610.
000000000002, -38443.112485316786, 9610.000000000002, 9610.000000000002, 2.
112485316776411, 9610.000000000002, 9610.000000000002, -38442.92424248827, 
9610.000000000002, 9610.000000000002, 1.9242424882567186, 9610.000000000002
, 9610.000000000002, -38442.75264934327, 9610.000000000002, 9610.0000000000
02, 1.7526493432597003, 9610.000000000002, 9610.000000000002, -38442.600643
52894, 9610.000000000002, 9610.000000000002, 1.6006435289268413, 9610.00000
0000002, 9610.000000000002, -38442.470599933244, 9610.000000000002, 9610.00
0000000002, 1.4705999332320159, 9610.000000000002, 9610.000000000002, -3844
2.36440892298, 9610.000000000002, 9610.000000000002, 1.3644089229710339, 96
10.000000000002, 9610.000000000002, -38442.28353030619, 9610.000000000002, 
9610.000000000002, 1.2835303061813406, 9610.000000000002, 9610.000000000002
, -38442.22903059256, 9610.000000000002, 9610.000000000002, 1.2290305925560
028, 9610.000000000002, 9610.000000000002, -38442.2016080224, 9610.00000000
0002, 9610.000000000002, 1.2016080223900816, 9610.000000000002, 9610.000000
000002, -38442.2016080224, 9610.000000000002, 9610.000000000002, 1.20160802
23900816, 9610.000000000002, 9610.000000000002, -38442.22903059256, 9610.00
0000000002, 9610.000000000002, 1.2290305925560023, 9610.000000000002, 9610.
000000000002, -38442.28353030619, 9610.000000000002, 9610.000000000002, 1.2
835303061813406, 9610.000000000002, 9610.000000000002, -38442.36440892298, 
9610.000000000002, 9610.000000000002, 1.3644089229710339, 9610.000000000002
, 9610.000000000002, -38442.470599933244, 9610.000000000002, 9610.000000000
002, 1.4705999332320159, 9610.000000000002, 9610.000000000002, -38442.60064
352894, 9610.000000000002, 9610.000000000002, 1.6006435289268413, 9610.0000
00000002, 9610.000000000002, -38442.75264934327, 9610.000000000002, 9610.00
0000000002, 1.7526493432597006, 9610.000000000002, 9610.000000000002, -3844
2.92424248827, 9610.000000000002, 9610.000000000002, 1.9242424882567186, 96
10.000000000002, 9610.000000000002, -38443.112485316786, 9610.000000000002,
 9610.000000000002, 2.112485316776411, 9610.000000000002, 9610.000000000002
, -38443.31376158765, 9610.000000000002, 9610.000000000002, 2.3137615876423
077, 9610.000000000002, 9610.000000000002, -38443.52359812821, 9610.0000000
00002, 9610.000000000002, 2.523598128204317, 9610.000000000002, 9610.000000
000002, -38443.73637322309, 9610.000000000002, 9610.000000000002, 2.7363732
230779934, 9610.000000000002, 9610.000000000002, -38443.944794818315, 9610.
000000000002, 9610.000000000002, 2.944794818307675, 9610.000000000002, 9610
.000000000002, -38444.13882580889, 9610.000000000002, 9610.000000000002, 3.
1388258088813488, 9610.000000000002, 9610.000000000002, -38444.30284388282,
 9610.000000000002, 9610.000000000002, 3.30284388280787, 9610.000000000002,
 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.00000000000
2, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0000
00000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -38
444.23066486092, 9610.000000000002, 9610.000000000002, 3.2306648609131807, 
9610.000000000002, 9610.000000000002, -38443.944794818315, 9610.00000000000
2, 9610.000000000002, 2.9447948183076753, 9610.000000000002, 9610.000000000
002, -38443.606614725024, 9610.000000000002, 9610.000000000002, 2.606614725
015434, 9610.000000000002, 9610.000000000002, -38443.24335298879, 9610.0000
00000002, 9610.000000000002, 2.2433529887819703, 9610.000000000002, 9610.00
0000000002, -38442.87250345995, 9610.000000000002, 9610.000000000002, 1.872
5034599417643, 9610.000000000002, 9610.000000000002, -38442.50677559012, 96
10.000000000002, 9610.000000000002, 1.5067755901124504, 9610.000000000002, 
9610.000000000002, -38442.155967567865, 9610.000000000002, 9610.00000000000
2, 1.1559675678594377, 9610.000000000002, 9610.000000000002, -38441.8278757
6603, 9610.000000000002, 9610.000000000002, 0.8278757660180558, 9610.000000
000002, 9610.000000000002, -38441.528802996196, 9610.000000000002, 9610.000
000000002, 0.5288029961897385, 9610.000000000002, 9610.000000000002, -38441
.26386933627, 9610.000000000002, 9610.000000000002, 0.26386933626219866, 96
10.000000000002, 9610.000000000002, -38441.03721402109, 9610.000000000002, 
9610.000000000002, 0.03721402107719518, 9610.000000000002, 9610.00000000000
2, -38440.85213180484, 9610.000000000002, 9610.000000000002, -0.14786819517
434013, 9610.000000000002, 9610.000000000002, -38440.71116701313, 9610.0000
00000002, 9610.000000000002, -0.28883298688322157, 9610.000000000002, 9610.
000000000002, -38440.61617848483, 9610.000000000002, 9610.000000000002, -0.
3838215151782367, 9610.000000000002, 9610.000000000002, -38440.56838319548,
 9610.000000000002, 9610.000000000002, -0.43161680452663065, 9610.000000000
002, 9610.000000000002, -38440.56838319548, 9610.000000000002, 9610.0000000
00002, -0.43161680452663065, 9610.000000000002, 9610.000000000002, -38440.6
1617848483, 9610.000000000002, 9610.000000000002, -0.38382151517823715, 961
0.000000000002, 9610.000000000002, -38440.71116701313, 9610.000000000002, 9
610.000000000002, -0.28883298688322157, 9610.000000000002, 9610.00000000000
2, -38440.85213180484, 9610.000000000002, 9610.000000000002, -0.14786819517
434013, 9610.000000000002, 9610.000000000002, -38441.03721402109, 9610.0000
00000002, 9610.000000000002, 0.03721402107719518, 9610.000000000002, 9610.0
00000000002, -38441.26386933627, 9610.000000000002, 9610.000000000002, 0.26
386933626219866, 9610.000000000002, 9610.000000000002, -38441.528802996196,
 9610.000000000002, 9610.000000000002, 0.5288029961897389, 9610.00000000000
2, 9610.000000000002, -38441.82787576603, 9610.000000000002, 9610.000000000
002, 0.8278757660180558, 9610.000000000002, 9610.000000000002, -38442.15596
7567865, 9610.000000000002, 9610.000000000002, 1.1559675678594372, 9610.000
000000002, 9610.000000000002, -38442.50677559012, 9610.000000000002, 9610.0
00000000002, 1.5067755901124502, 9610.000000000002, 9610.000000000002, -384
42.87250345995, 9610.000000000002, 9610.000000000002, 1.8725034599417647, 9
610.000000000002, 9610.000000000002, -38443.24335298879, 9610.000000000002,
 9610.000000000002, 2.2433529887819703, 9610.000000000002, 9610.00000000000
2, -38443.606614725024, 9610.000000000002, 9610.000000000002, 2.60661472501
5433, 9610.000000000002, 9610.000000000002, -38443.944794818315, 9610.00000
0000002, 9610.000000000002, 2.944794818307675, 9610.000000000002, 9610.0000
00000002, -38444.23066486092, 9610.000000000002, 9610.000000000002, 3.23066
48609131807, 9610.000000000002, 9610.000000000002, 9610.000000000002, -3844
4.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.4000000000
1, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.00000
0000002, 9610.000000000002, -38444.15313257167, 9610.000000000002, 9610.000
000000002, 3.1531325716591234, 9610.000000000002, 9610.000000000002, -38443
.73637322309, 9610.000000000002, 9610.000000000002, 2.7363732230779934, 961
0.000000000002, 9610.000000000002, -38443.24335298879, 9610.000000000002, 9
610.000000000002, 2.2433529887819708, 9610.000000000002, 9610.000000000002,
 -38442.71376715609, 9610.000000000002, 9610.000000000002, 1.71376715608236
44, 9610.000000000002, 9610.000000000002, -38442.17311936155, 9610.00000000
0002, 9610.000000000002, 1.1731193615376734, 9610.000000000002, 9610.000000
000002, -38441.63993824401, 9610.000000000002, 9610.000000000002, 0.6399382
439962102, 9610.000000000002, 9610.000000000002, -38441.12850821972, 9610.0
00000000002, 9610.000000000002, 0.12850821971431436, 9610.000000000002, 961
0.000000000002, -38440.65019533194, 9610.000000000002, 9610.000000000002, -
0.3498046680719562, 9610.000000000002, 9610.000000000002, -38440.2141882162
7, 9610.000000000002, 9610.000000000002, -0.7858117837388687, 9610.00000000
0002, 9610.000000000002, -38439.82795124467, 9610.000000000002, 9610.000000
000002, -1.1720487553441248, 9610.000000000002, 9610.000000000002, -38439.4
9751885431, 9610.000000000002, 9610.000000000002, -1.5024811456988805, 9610
.000000000002, 9610.000000000002, -38439.227694345325, 9610.000000000002, 9
610.000000000002, -1.7723056546817468, 9610.000000000002, 9610.000000000002
, -38439.0221869958, 9610.000000000002, 9610.000000000002, -1.9778130042102
862, 9610.000000000002, 9610.000000000002, -38438.883706738074, 9610.000000
000002, 9610.000000000002, -2.1162932619319412, 9610.000000000002, 9610.000
000000002, -38438.814027755725, 9610.000000000002, 9610.000000000002, -2.18
59722442853706, 9610.000000000002, 9610.000000000002, -38438.814027755725, 
9610.000000000002, 9610.000000000002, -2.1859722442853706, 9610.00000000000
2, 9610.000000000002, -38438.883706738074, 9610.000000000002, 9610.00000000
0002, -2.1162932619319412, 9610.000000000002, 9610.000000000002, -38439.022
1869958, 9610.000000000002, 9610.000000000002, -1.9778130042102862, 9610.00
0000000002, 9610.000000000002, -38439.227694345325, 9610.000000000002, 9610
.000000000002, -1.7723056546817468, 9610.000000000002, 9610.000000000002, -
38439.49751885431, 9610.000000000002, 9610.000000000002, -1.502481145698880
5, 9610.000000000002, 9610.000000000002, -38439.82795124467, 9610.000000000
002, 9610.000000000002, -1.1720487553441248, 9610.000000000002, 9610.000000
000002, -38440.21418821627, 9610.000000000002, 9610.000000000002, -0.785811
7837388678, 9610.000000000002, 9610.000000000002, -38440.65019533194, 9610.
000000000002, 9610.000000000002, -0.3498046680719562, 9610.000000000002, 96
10.000000000002, -38441.12850821972, 9610.000000000002, 9610.000000000002, 
0.12850821971431392, 9610.000000000002, 9610.000000000002, -38441.639938244
01, 9610.000000000002, 9610.000000000002, 0.6399382439962098, 9610.00000000
0002, 9610.000000000002, -38442.17311936155, 9610.000000000002, 9610.000000
000002, 1.1731193615376743, 9610.000000000002, 9610.000000000002, -38442.71
376715609, 9610.000000000002, 9610.000000000002, 1.7137671560823644, 9610.0
00000000002, 9610.000000000002, -38443.24335298879, 9610.000000000002, 9610
.000000000002, 2.24335298878197, 9610.000000000002, 9610.000000000002, -384
43.73637322309, 9610.000000000002, 9610.000000000002, 2.736373223077993, 96
10.000000000002, 9610.000000000002, -38444.15313257167, 9610.000000000002, 
9610.000000000002, 3.153132571659124, 9610.000000000002, 9610.000000000002,
 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000
000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, 3.4, 9610.000000000002, 9610.000000000002, -38444.073980791916, 9
610.000000000002, 9610.000000000002, 3.073980791904124, 9610.000000000002, 
9610.000000000002, -38443.52359812821, 9610.000000000002, 9610.000000000002
, 2.523598128204317, 9610.000000000002, 9610.000000000002, -38442.872503459
95, 9610.000000000002, 9610.000000000002, 1.8725034599417645, 9610.00000000
0002, 9610.000000000002, -38442.17311936155, 9610.000000000002, 9610.000000
000002, 1.1731193615376738, 9610.000000000002, 9610.000000000002, -38441.45
912656379, 9610.000000000002, 9610.000000000002, 0.45912656378041605, 9610.
000000000002, 9610.000000000002, -38440.75499444365, 9610.000000000002, 961
0.000000000002, -0.24500555636513743, 9610.000000000002, 9610.000000000002,
 -38440.07958735315, 9610.000000000002, 9610.000000000002, -0.9204126468566
689, 9610.000000000002, 9610.000000000002, -38439.44791556904, 9610.0000000
00002, 9610.000000000002, -1.5520844309642476, 9610.000000000002, 9610.0000
00000002, -38438.872113829915, 9610.000000000002, 9610.000000000002, -2.127
886170096947, 9610.000000000002, 9610.000000000002, -38438.36203976924, 961
0.000000000002, 9610.000000000002, -2.637960230763316, 9610.000000000002, 9
610.000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000002
, -3.074337386537523, 9610.000000000002, 9610.000000000002, -38437.56932571
907, 9610.000000000002, 9610.000000000002, -3.4306742809373207, 9610.000000
000002, 9610.000000000002, -38437.29792765007, 9610.000000000002, 9610.0000
00000002, -3.7020723499392134, 9610.000000000002, 9610.000000000002, -38437
.115047209896, 9610.000000000002, 9610.000000000002, -3.8849527901120937, 9
610.000000000002, 9610.000000000002, -38437.02302742907, 9610.000000000002,
 9610.000000000002, -3.976972570933176, 9610.000000000002, 9610.00000000000
2, -38437.02302742907, 9610.000000000002, 9610.000000000002, -3.97697257093
3176, 9610.000000000002, 9610.000000000002, -38437.115047209896, 9610.00000
0000002, 9610.000000000002, -3.8849527901120946, 9610.000000000002, 9610.00
0000000002, -38437.29792765007, 9610.000000000002, 9610.000000000002, -3.70
20723499392134, 9610.000000000002, 9610.000000000002, -38437.56932571907, 9
610.000000000002, 9610.000000000002, -3.4306742809373207, 9610.000000000002
, 9610.000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000
002, -3.074337386537523, 9610.000000000002, 9610.000000000002, -38438.36203
976924, 9610.000000000002, 9610.000000000002, -2.637960230763316, 9610.0000
00000002, 9610.000000000002, -38438.872113829915, 9610.000000000002, 9610.0
00000000002, -2.127886170096946, 9610.000000000002, 9610.000000000002, -384
39.44791556904, 9610.000000000002, 9610.000000000002, -1.5520844309642476, 
9610.000000000002, 9610.000000000002, -38440.07958735315, 9610.000000000002
, 9610.000000000002, -0.9204126468566698, 9610.000000000002, 9610.000000000
002, -38440.75499444365, 9610.000000000002, 9610.000000000002, -0.245005556
36513788, 9610.000000000002, 9610.000000000002, -38441.45912656379, 9610.00
0000000002, 9610.000000000002, 0.45912656378041694, 9610.000000000002, 9610
.000000000002, -38442.17311936155, 9610.000000000002, 9610.000000000002, 1.
1731193615376738, 9610.000000000002, 9610.000000000002, -38442.87250345995,
 9610.000000000002, 9610.000000000002, 1.8725034599417636, 9610.00000000000
2, 9610.000000000002, -38443.52359812821, 9610.000000000002, 9610.000000000
002, 2.523598128204316, 9610.000000000002, 9610.000000000002, -38444.073980
791916, 9610.000000000002, 9610.000000000002, 3.0739807919041247, 9610.0000
00000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.00
0000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 
9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.00000000
0002, -38443.99592214669, 9610.000000000002, 9610.000000000002, 2.995922146
6807455, 9610.000000000002, 9610.000000000002, -38443.31376158765, 9610.000
000000002, 9610.000000000002, 2.3137615876423077, 9610.000000000002, 9610.0
00000000002, -38442.50677559012, 9610.000000000002, 9610.000000000002, 1.50
67755901124507, 9610.000000000002, 9610.000000000002, -38441.63993824401, 9
610.000000000002, 9610.000000000002, 0.6399382439962102, 9610.000000000002,
 9610.000000000002, -38440.75499444365, 9610.000000000002, 9610.00000000000
2, -0.24500555636513743, 9610.000000000002, 9610.000000000002, -38439.88227
225889, 9610.000000000002, 9610.000000000002, -1.1177277411202078, 9610.000
000000002, 9610.000000000002, -38439.04515272585, 9610.000000000002, 9610.0
00000000002, -1.9548472741575442, 9610.000000000002, 9610.000000000002, -38
438.26224002566, 9610.000000000002, 9610.000000000002, -2.7377599743487298,
 9610.000000000002, 9610.000000000002, -38437.548574312386, 9610.0000000000
02, 9610.000000000002, -3.4514256876210934, 9610.000000000002, 9610.0000000
00002, -38436.91637342867, 9610.000000000002, 9610.000000000002, -4.0836265
71341076, 9610.000000000002, 9610.000000000002, -38436.37551466954, 9610.00
0000000002, 9610.000000000002, -4.6244853304696125, 9610.000000000002, 9610
.000000000002, -38435.933860178724, 9610.000000000002, 9610.000000000002, -
5.066139821284695, 9610.000000000002, 9610.000000000002, -38435.59748138142
, 9610.000000000002, 9610.000000000002, -5.4025186185886405, 9610.000000000
002, 9610.000000000002, -38435.3708139525, 9610.000000000002, 9610.00000000
0002, -5.629186047513276, 9610.000000000002, 9610.000000000002, -38435.2567
6191331, 9610.000000000002, 9610.000000000002, -5.743238086699128, 9610.000
000000002, 9610.000000000002, -38435.25676191331, 9610.000000000002, 9610.0
00000000002, -5.743238086699128, 9610.000000000002, 9610.000000000002, -384
35.3708139525, 9610.000000000002, 9610.000000000002, -5.629186047513278, 96
10.000000000002, 9610.000000000002, -38435.59748138142, 9610.000000000002, 
9610.000000000002, -5.4025186185886405, 9610.000000000002, 9610.00000000000
2, -38435.933860178724, 9610.000000000002, 9610.000000000002, -5.0661398212
84695, 9610.000000000002, 9610.000000000002, -38436.37551466954, 9610.00000
0000002, 9610.000000000002, -4.6244853304696125, 9610.000000000002, 9610.00
0000000002, -38436.91637342867, 9610.000000000002, 9610.000000000002, -4.08
3626571341076, 9610.000000000002, 9610.000000000002, -38437.548574312386, 9
610.000000000002, 9610.000000000002, -3.4514256876210916, 9610.000000000002
, 9610.000000000002, -38438.26224002566, 9610.000000000002, 9610.0000000000
02, -2.7377599743487298, 9610.000000000002, 9610.000000000002, -38439.04515
272585, 9610.000000000002, 9610.000000000002, -1.9548472741575442, 9610.000
000000002, 9610.000000000002, -38439.88227225889, 9610.000000000002, 9610.0
00000000002, -1.1177277411202087, 9610.000000000002, 9610.000000000002, -38
440.75499444365, 9610.000000000002, 9610.000000000002, -0.24500555636513655
, 9610.000000000002, 9610.000000000002, -38441.63993824401, 9610.0000000000
02, 9610.000000000002, 0.6399382439962102, 9610.000000000002, 9610.00000000
0002, -38442.50677559012, 9610.000000000002, 9610.000000000002, 1.506775590
1124495, 9610.000000000002, 9610.000000000002, -38443.31376158765, 9610.000
000000002, 9610.000000000002, 2.3137615876423068, 9610.000000000002, 9610.0
00000000002, -38443.99592214669, 9610.000000000002, 9610.000000000002, 2.99
5922146680746, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38
444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000
001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000
000000002, 9610.000000000002, -38443.92104789944, 9610.000000000002, 9610.0
00000000002, 2.9210478994341713, 9610.000000000002, 9610.000000000002, -384
43.112485316786, 9610.000000000002, 9610.000000000002, 2.112485316776411, 9
610.000000000002, 9610.000000000002, -38442.155967567865, 9610.000000000002
, 9610.000000000002, 1.155967567859438, 9610.000000000002, 9610.00000000000
2, -38441.12850821972, 9610.000000000002, 9610.000000000002, 0.128508219714
3148, 9610.000000000002, 9610.000000000002, -38440.07958735315, 9610.000000
000002, 9610.000000000002, -0.9204126468566689, 9610.000000000002, 9610.000
000000002, -38439.04515272585, 9610.000000000002, 9610.000000000002, -1.954
8472741575433, 9610.000000000002, 9610.000000000002, -38438.05291780013, 96
10.000000000002, 9610.000000000002, -2.94708219987653, 9610.000000000002, 9
610.000000000002, -38437.1249340484, 9610.000000000002, 9610.000000000002, 
-3.8750659516115467, 9610.000000000002, 9610.000000000002, -38436.279028513
94, 9610.000000000002, 9610.000000000002, -4.720971486066933, 9610.00000000
0002, 9610.000000000002, -38435.52968296394, 9610.000000000002, 9610.000000
000002, -5.470317036064268, 9610.000000000002, 9610.000000000002, -38434.88
8604922504, 9610.000000000002, 9610.000000000002, -6.111395077501429, 9610.
000000000002, 9610.000000000002, -38434.36511336175, 9610.000000000002, 961
0.000000000002, -6.634886638255411, 9610.000000000002, 9610.000000000002, -
38433.96640472125, 9610.000000000002, 9610.000000000002, -7.033595278759009
, 9610.000000000002, 9610.000000000002, -38433.69773659128, 9610.0000000000
02, 9610.000000000002, -7.302263408728351, 9610.000000000002, 9610.00000000
0002, -38433.562551098454, 9610.000000000002, 9610.000000000002, -7.4374489
01556442, 9610.000000000002, 9610.000000000002, -38433.562551098454, 9610.0
00000000002, 9610.000000000002, -7.437448901556442, 9610.000000000002, 9610
.000000000002, -38433.69773659128, 9610.000000000002, 9610.000000000002, -7
.3022634087283524, 9610.000000000002, 9610.000000000002, -38433.96640472125
, 9610.000000000002, 9610.000000000002, -7.033595278759009, 9610.0000000000
02, 9610.000000000002, -38434.36511336175, 9610.000000000002, 9610.00000000
0002, -6.634886638255411, 9610.000000000002, 9610.000000000002, -38434.8886
04922504, 9610.000000000002, 9610.000000000002, -6.111395077501429, 9610.00
0000000002, 9610.000000000002, -38435.52968296394, 9610.000000000002, 9610.
000000000002, -5.470317036064268, 9610.000000000002, 9610.000000000002, -38
436.27902851394, 9610.000000000002, 9610.000000000002, -4.720971486066931, 
9610.000000000002, 9610.000000000002, -38437.1249340484, 9610.000000000002,
 9610.000000000002, -3.8750659516115467, 9610.000000000002, 9610.0000000000
02, -38438.05291780013, 9610.000000000002, 9610.000000000002, -2.9470821998
76531, 9610.000000000002, 9610.000000000002, -38439.04515272585, 9610.00000
0000002, 9610.000000000002, -1.9548472741575442, 9610.000000000002, 9610.00
0000000002, -38440.07958735315, 9610.000000000002, 9610.000000000002, -0.92
04126468566671, 9610.000000000002, 9610.000000000002, -38441.12850821972, 9
610.000000000002, 9610.000000000002, 0.1285082197143148, 9610.000000000002,
 9610.000000000002, -38442.155967567865, 9610.000000000002, 9610.0000000000
02, 1.1559675678594368, 9610.000000000002, 9610.000000000002, -38443.112485
316786, 9610.000000000002, 9610.000000000002, 2.1124853167764095, 9610.0000
00000002, 9610.000000000002, -38443.92104789944, 9610.000000000002, 9610.00
0000000002, 2.9210478994341718, 9610.000000000002, 9610.000000000002, 9610.
000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002
, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.0000000000
02, 3.4, 9610.000000000002, 9610.000000000002, -38443.85102205872, 9610.000
000000002, 9610.000000000002, 2.851022058711155, 9610.000000000002, 9610.00
0000000002, -38442.92424248827, 9610.000000000002, 9610.000000000002, 1.924
2424882567188, 9610.000000000002, 9610.000000000002, -38441.82787576603, 96
10.000000000002, 9610.000000000002, 0.8278757660180562, 9610.000000000002, 
9610.000000000002, -38440.65019533194, 9610.000000000002, 9610.000000000002
, -0.34980466807195576, 9610.000000000002, 9610.000000000002, -38439.447915
56904, 9610.000000000002, 9610.000000000002, -1.5520844309642476, 9610.0000
00000002, 9610.000000000002, -38438.26224002566, 9610.000000000002, 9610.00
0000000002, -2.737759974348729, 9610.000000000002, 9610.000000000002, -3843
7.1249340484, 9610.000000000002, 9610.000000000002, -3.8750659516115458, 96
10.000000000002, 9610.000000000002, -38436.06127317505, 9610.000000000002, 
9610.000000000002, -4.938726824953864, 9610.000000000002, 9610.000000000002
, -38435.09169087595, 9610.000000000002, 9610.000000000002, -5.908309124063
818, 9610.000000000002, 9610.000000000002, -38434.23278624463, 9610.0000000
00002, 9610.000000000002, -6.767213755373509, 9610.000000000002, 9610.00000
0000002, -38433.49797851965, 9610.000000000002, 9610.000000000002, -7.50202
1480358705, 9610.000000000002, 9610.000000000002, -38432.89794916606, 9610.
000000000002, 9610.000000000002, -8.102050833952292, 9610.000000000002, 961
0.000000000002, -38432.440946788614, 9610.000000000002, 9610.000000000002, 
-8.559053211390776, 9610.000000000002, 9610.000000000002, -38432.1329976707
3, 9610.000000000002, 9610.000000000002, -8.867002329280114, 9610.000000000
002, 9610.000000000002, -38431.97804719978, 9610.000000000002, 9610.0000000
00002, -9.02195280022952, 9610.000000000002, 9610.000000000002, -38431.9780
4719978, 9610.000000000002, 9610.000000000002, -9.02195280022952, 9610.0000
00000002, 9610.000000000002, -38432.13299767073, 9610.000000000002, 9610.00
0000000002, -8.867002329280115, 9610.000000000002, 9610.000000000002, -3843
2.440946788614, 9610.000000000002, 9610.000000000002, -8.559053211390776, 9
610.000000000002, 9610.000000000002, -38432.89794916606, 9610.000000000002,
 9610.000000000002, -8.102050833952292, 9610.000000000002, 9610.00000000000
2, -38433.49797851965, 9610.000000000002, 9610.000000000002, -7.50202148035
8705, 9610.000000000002, 9610.000000000002, -38434.23278624463, 9610.000000
000002, 9610.000000000002, -6.767213755373509, 9610.000000000002, 9610.0000
00000002, -38435.09169087595, 9610.000000000002, 9610.000000000002, -5.9083
09124063818, 9610.000000000002, 9610.000000000002, -38436.06127317505, 9610
.000000000002, 9610.000000000002, -4.938726824953864, 9610.000000000002, 96
10.000000000002, -38437.1249340484, 9610.000000000002, 9610.000000000002, -
3.8750659516115467, 9610.000000000002, 9610.000000000002, -38438.2622400256
6, 9610.000000000002, 9610.000000000002, -2.737759974348729, 9610.000000000
002, 9610.000000000002, -38439.44791556904, 9610.000000000002, 9610.0000000
00002, -1.5520844309642459, 9610.000000000002, 9610.000000000002, -38440.65
019533194, 9610.000000000002, 9610.000000000002, -0.34980466807195576, 9610
.000000000002, 9610.000000000002, -38441.82787576603, 9610.000000000002, 96
10.000000000002, 0.8278757660180549, 9610.000000000002, 9610.000000000002, 
-38442.92424248827, 9610.000000000002, 9610.000000000002, 1.924242488256717
3, 9610.000000000002, 9610.000000000002, -38443.85102205872, 9610.000000000
002, 9610.000000000002, 2.8510220587111554, 9610.000000000002, 9610.0000000
00002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.
000000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 961
0.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -38443.787189856
8, 9610.000000000002, 9610.000000000002, 2.7871898567876396, 9610.000000000
002, 9610.000000000002, -38442.75264934327, 9610.000000000002, 9610.0000000
00002, 1.7526493432597003, 9610.000000000002, 9610.000000000002, -38441.528
802996196, 9610.000000000002, 9610.000000000002, 0.5288029961897389, 9610.0
00000000002, 9610.000000000002, -38440.21418821627, 9610.000000000002, 9610
.000000000002, -0.7858117837388678, 9610.000000000002, 9610.000000000002, -
38438.872113829915, 9610.000000000002, 9610.000000000002, -2.12788617009694
6, 9610.000000000002, 9610.000000000002, -38437.548574312386, 9610.00000000
0002, 9610.000000000002, -3.4514256876210925, 9610.000000000002, 9610.00000
0000002, -38436.27902851394, 9610.000000000002, 9610.000000000002, -4.72097
1486066933, 9610.000000000002, 9610.000000000002, -38435.09169087595, 9610.
000000000002, 9610.000000000002, -5.90830912406382, 9610.000000000002, 9610
.000000000002, -38434.00937076272, 9610.000000000002, 9610.000000000002, -6
.990629237288765, 9610.000000000002, 9610.000000000002, -38433.05059732115,
 9610.000000000002, 9610.000000000002, -7.949402678864468, 9610.00000000000
2, 9610.000000000002, -38432.23035010661, 9610.000000000002, 9610.000000000
002, -8.76964989339654, 9610.000000000002, 9610.000000000002, -38431.560552
56749, 9610.000000000002, 9610.000000000002, -9.439447432518133, 9610.00000
0000002, 9610.000000000002, -38431.05041241193, 9610.000000000002, 9610.000
000000002, -9.949587588075156, 9610.000000000002, 9610.000000000002, -38430
.70665662715, 9610.000000000002, 9610.000000000002, -10.293343372857397, 96
10.000000000002, 9610.000000000002, -38430.53368934896, 9610.000000000002, 
9610.000000000002, -10.466310651051492, 9610.000000000002, 9610.00000000000
2, -38430.53368934896, 9610.000000000002, 9610.000000000002, -10.4663106510
51492, 9610.000000000002, 9610.000000000002, -38430.70665662715, 9610.00000
0000002, 9610.000000000002, -10.293343372857398, 9610.000000000002, 9610.00
0000000002, -38431.05041241193, 9610.000000000002, 9610.000000000002, -9.94
9587588075156, 9610.000000000002, 9610.000000000002, -38431.56055256749, 96
10.000000000002, 9610.000000000002, -9.439447432518133, 9610.000000000002, 
9610.000000000002, -38432.23035010661, 9610.000000000002, 9610.000000000002
, -8.76964989339654, 9610.000000000002, 9610.000000000002, -38433.050597321
15, 9610.000000000002, 9610.000000000002, -7.949402678864468, 9610.00000000
0002, 9610.000000000002, -38434.00937076272, 9610.000000000002, 9610.000000
000002, -6.990629237288763, 9610.000000000002, 9610.000000000002, -38435.09
169087595, 9610.000000000002, 9610.000000000002, -5.90830912406382, 9610.00
0000000002, 9610.000000000002, -38436.27902851394, 9610.000000000002, 9610.
000000000002, -4.720971486066933, 9610.000000000002, 9610.000000000002, -38
437.548574312386, 9610.000000000002, 9610.000000000002, -3.4514256876210934
, 9610.000000000002, 9610.000000000002, -38438.872113829915, 9610.000000000
002, 9610.000000000002, -2.1278861700969443, 9610.000000000002, 9610.000000
000002, -38440.21418821627, 9610.000000000002, 9610.000000000002, -0.785811
7837388678, 9610.000000000002, 9610.000000000002, -38441.528802996196, 9610
.000000000002, 9610.000000000002, 0.5288029961897376, 9610.000000000002, 96
10.000000000002, -38442.75264934327, 9610.000000000002, 9610.000000000002, 
1.7526493432596986, 9610.000000000002, 9610.000000000002, -38443.7871898568
, 9610.000000000002, 9610.000000000002, 2.7871898567876405, 9610.0000000000
02, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.00000000
0002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 
-38443.73064409074, 9610.000000000002, 9610.000000000002, 2.730644090730306
7, 9610.000000000002, 9610.000000000002, -38442.60064352894, 9610.000000000
002, 9610.000000000002, 1.6006435289268413, 9610.000000000002, 9610.0000000
00002, -38441.26386933627, 9610.000000000002, 9610.000000000002, 0.26386933
62621991, 9610.000000000002, 9610.000000000002, -38439.82795124467, 9610.00
0000000002, 9610.000000000002, -1.1720487553441257, 9610.000000000002, 9610
.000000000002, -38438.36203976924, 9610.000000000002, 9610.000000000002, -2
.637960230763317, 9610.000000000002, 9610.000000000002, -38436.91637342867,
 9610.000000000002, 9610.000000000002, -4.083626571341076, 9610.00000000000
2, 9610.000000000002, -38435.52968296394, 9610.000000000002, 9610.000000000
002, -5.4703170360642694, 9610.000000000002, 9610.000000000002, -38434.2327
8624463, 9610.000000000002, 9610.000000000002, -6.767213755373509, 9610.000
000000002, 9610.000000000002, -38433.05059732115, 9610.000000000002, 9610.0
00000000002, -7.94940267886447, 9610.000000000002, 9610.000000000002, -3843
2.00335507838, 9610.000000000002, 9610.000000000002, -8.996644921633852, 96
10.000000000002, 9610.000000000002, -38431.10742127866, 9610.000000000002, 
9610.000000000002, -9.892578721343801, 9610.000000000002, 9610.000000000002
, -38430.37581958148, 9610.000000000002, 9610.000000000002, -10.62418041852
7155, 9610.000000000002, 9610.000000000002, -38429.81860731687, 9610.000000
000002, 9610.000000000002, -11.181392683141262, 9610.000000000002, 9610.000
000000002, -38429.443132190616, 9610.000000000002, 9610.000000000002, -11.5
5686780939076, 9610.000000000002, 9610.000000000002, -38429.254204720906, 9
610.000000000002, 9610.000000000002, -11.745795279100397, 9610.000000000002
, 9610.000000000002, -38429.254204720906, 9610.000000000002, 9610.000000000
002, -11.745795279100397, 9610.000000000002, 9610.000000000002, -38429.4431
32190616, 9610.000000000002, 9610.000000000002, -11.556867809390761, 9610.0
00000000002, 9610.000000000002, -38429.81860731687, 9610.000000000002, 9610
.000000000002, -11.181392683141262, 9610.000000000002, 9610.000000000002, -
38430.37581958148, 9610.000000000002, 9610.000000000002, -10.62418041852715
5, 9610.000000000002, 9610.000000000002, -38431.10742127866, 9610.000000000
002, 9610.000000000002, -9.892578721343801, 9610.000000000002, 9610.0000000
00002, -38432.00335507838, 9610.000000000002, 9610.000000000002, -8.9966449
21633852, 9610.000000000002, 9610.000000000002, -38433.05059732115, 9610.00
0000000002, 9610.000000000002, -7.949402678864468, 9610.000000000002, 9610.
000000000002, -38434.23278624463, 9610.000000000002, 9610.000000000002, -6.
767213755373509, 9610.000000000002, 9610.000000000002, -38435.52968296394, 
9610.000000000002, 9610.000000000002, -5.4703170360642694, 9610.00000000000
2, 9610.000000000002, -38436.91637342867, 9610.000000000002, 9610.000000000
002, -4.083626571341078, 9610.000000000002, 9610.000000000002, -38438.36203
976924, 9610.000000000002, 9610.000000000002, -2.6379602307633143, 9610.000
000000002, 9610.000000000002, -38439.82795124467, 9610.000000000002, 9610.0
00000000002, -1.1720487553441257, 9610.000000000002, 9610.000000000002, -38
441.26386933627, 9610.000000000002, 9610.000000000002, 0.26386933626219733,
 9610.000000000002, 9610.000000000002, -38442.60064352894, 9610.00000000000
2, 9610.000000000002, 1.6006435289268393, 9610.000000000002, 9610.000000000
002, -38443.73064409074, 9610.000000000002, 9610.000000000002, 2.7306440907
303076, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.400
00000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 96
10.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.0000000000
02, 9610.000000000002, -38443.68226821267, 9610.000000000002, 9610.00000000
0002, 2.682268212664803, 9610.000000000002, 9610.000000000002, -38442.47059
9933244, 9610.000000000002, 9610.000000000002, 1.4705999332320157, 9610.000
000000002, 9610.000000000002, -38441.03721402109, 9610.000000000002, 9610.0
00000000002, 0.03721402107719518, 9610.000000000002, 9610.000000000002, -38
439.49751885431, 9610.000000000002, 9610.000000000002, -1.5024811456988805,
 9610.000000000002, 9610.000000000002, -38437.925662613474, 9610.0000000000
02, 9610.000000000002, -3.074337386537524, 9610.000000000002, 9610.00000000
0002, -38436.37551466954, 9610.000000000002, 9610.000000000002, -4.62448533
04696125, 9610.000000000002, 9610.000000000002, -38434.888604922504, 9610.0
00000000002, 9610.000000000002, -6.111395077501431, 9610.000000000002, 9610
.000000000002, -38433.49797851965, 9610.000000000002, 9610.000000000002, -7
.502021480358708, 9610.000000000002, 9610.000000000002, -38432.23035010661,
 9610.000000000002, 9610.000000000002, -8.769649893396544, 9610.00000000000
2, 9610.000000000002, -38431.10742127866, 9610.000000000002, 9610.000000000
002, -9.892578721343803, 9610.000000000002, 9610.000000000002, -38430.14673
630003, 9610.000000000002, 9610.000000000002, -10.853263699984586, 9610.000
000000002, 9610.000000000002, -38429.362260079724, 9610.000000000002, 9610.
000000000002, -11.63773992028087, 9610.000000000002, 9610.000000000002, -38
428.764776813994, 9610.000000000002, 9610.000000000002, -12.235223186011535
, 9610.000000000002, 9610.000000000002, -38428.362165244085, 9610.000000000
002, 9610.000000000002, -12.637834755926546, 9610.000000000002, 9610.000000
000002, -38428.15958355528, 9610.000000000002, 9610.000000000002, -12.84041
6444731476, 9610.000000000002, 9610.000000000002, -38428.15958355528, 9610.
000000000002, 9610.000000000002, -12.840416444731476, 9610.000000000002, 96
10.000000000002, -38428.362165244085, 9610.000000000002, 9610.000000000002,
 -12.63783475592655, 9610.000000000002, 9610.000000000002, -38428.764776813
994, 9610.000000000002, 9610.000000000002, -12.235223186011535, 9610.000000
000002, 9610.000000000002, -38429.362260079724, 9610.000000000002, 9610.000
000000002, -11.63773992028087, 9610.000000000002, 9610.000000000002, -38430
.14673630003, 9610.000000000002, 9610.000000000002, -10.853263699984586, 96
10.000000000002, 9610.000000000002, -38431.10742127866, 9610.000000000002, 
9610.000000000002, -9.892578721343803, 9610.000000000002, 9610.000000000002
, -38432.23035010661, 9610.000000000002, 9610.000000000002, -8.769649893396
542, 9610.000000000002, 9610.000000000002, -38433.49797851965, 9610.0000000
00002, 9610.000000000002, -7.502021480358708, 9610.000000000002, 9610.00000
0000002, -38434.888604922504, 9610.000000000002, 9610.000000000002, -6.1113
95077501433, 9610.000000000002, 9610.000000000002, -38436.37551466954, 9610
.000000000002, 9610.000000000002, -4.624485330469614, 9610.000000000002, 96
10.000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000002,
 -3.0743373865375223, 9610.000000000002, 9610.000000000002, -38439.49751885
431, 9610.000000000002, 9610.000000000002, -1.5024811456988805, 9610.000000
000002, 9610.000000000002, -38441.03721402109, 9610.000000000002, 9610.0000
00000002, 0.037214021077193404, 9610.000000000002, 9610.000000000002, -3844
2.470599933244, 9610.000000000002, 9610.000000000002, 1.4705999332320137, 9
610.000000000002, 9610.000000000002, -38443.68226821267, 9610.000000000002,
 9610.000000000002, 2.6822682126648036, 9610.000000000002, 9610.00000000000
2, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.0000
00000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.00
0000000002, 3.4, 9610.000000000002, 9610.000000000002, -38443.64276543411, 
9610.000000000002, 9610.000000000002, 2.64276543410358, 9610.000000000002, 
9610.000000000002, -38442.36440892298, 9610.000000000002, 9610.000000000002
, 1.3644089229710339, 9610.000000000002, 9610.000000000002, -38440.85213180
484, 9610.000000000002, 9610.000000000002, -0.14786819517434013, 9610.00000
0000002, 9610.000000000002, -38439.227694345325, 9610.000000000002, 9610.00
0000000002, -1.7723056546817468, 9610.000000000002, 9610.000000000002, -384
37.56932571907, 9610.000000000002, 9610.000000000002, -3.4306742809373216, 
9610.000000000002, 9610.000000000002, -38435.933860178724, 9610.00000000000
2, 9610.000000000002, -5.066139821284695, 9610.000000000002, 9610.000000000
002, -38434.36511336175, 9610.000000000002, 9610.000000000002, -6.634886638
255413, 9610.000000000002, 9610.000000000002, -38432.89794916606, 9610.0000
00000002, 9610.000000000002, -8.102050833952296, 9610.000000000002, 9610.00
0000000002, -38431.56055256749, 9610.000000000002, 9610.000000000002, -9.43
9447432518136, 9610.000000000002, 9610.000000000002, -38430.37581958148, 96
10.000000000002, 9610.000000000002, -10.624180418527155, 9610.000000000002,
 9610.000000000002, -38429.362260079724, 9610.000000000002, 9610.0000000000
02, -11.63773992028087, 9610.000000000002, 9610.000000000002, -38428.534607
57692, 9610.000000000002, 9610.000000000002, -12.465392423087875, 9610.0000
00000002, 9610.000000000002, -38427.904239813724, 9610.000000000002, 9610.0
00000000002, -13.095760186282176, 9610.000000000002, 9610.000000000002, -38
427.47946916437, 9610.000000000002, 9610.000000000002, -13.520530835637695,
 9610.000000000002, 9610.000000000002, -38427.265737712194, 9610.0000000000
02, 9610.000000000002, -13.734262287814724, 9610.000000000002, 9610.0000000
00002, -38427.265737712194, 9610.000000000002, 9610.000000000002, -13.73426
2287814724, 9610.000000000002, 9610.000000000002, -38427.47946916437, 9610.
000000000002, 9610.000000000002, -13.520530835637699, 9610.000000000002, 96
10.000000000002, -38427.904239813724, 9610.000000000002, 9610.000000000002,
 -13.095760186282176, 9610.000000000002, 9610.000000000002, -38428.53460757
692, 9610.000000000002, 9610.000000000002, -12.465392423087875, 9610.000000
000002, 9610.000000000002, -38429.362260079724, 9610.000000000002, 9610.000
000000002, -11.63773992028087, 9610.000000000002, 9610.000000000002, -38430
.37581958148, 9610.000000000002, 9610.000000000002, -10.624180418527155, 96
10.000000000002, 9610.000000000002, -38431.56055256749, 9610.000000000002, 
9610.000000000002, -9.439447432518133, 9610.000000000002, 9610.000000000002
, -38432.89794916606, 9610.000000000002, 9610.000000000002, -8.102050833952
296, 9610.000000000002, 9610.000000000002, -38434.36511336175, 9610.0000000
00002, 9610.000000000002, -6.634886638255415, 9610.000000000002, 9610.00000
0000002, -38435.933860178724, 9610.000000000002, 9610.000000000002, -5.0661
39821284697, 9610.000000000002, 9610.000000000002, -38437.56932571907, 9610
.000000000002, 9610.000000000002, -3.43067428093732, 9610.000000000002, 961
0.000000000002, -38439.227694345325, 9610.000000000002, 9610.000000000002, 
-1.7723056546817468, 9610.000000000002, 9610.000000000002, -38440.852131804
84, 9610.000000000002, 9610.000000000002, -0.1478681951743419, 9610.0000000
00002, 9610.000000000002, -38442.36440892298, 9610.000000000002, 9610.00000
0000002, 1.3644089229710312, 9610.000000000002, 9610.000000000002, -38443.6
4276543411, 9610.000000000002, 9610.000000000002, 2.642765434103581, 9610.0
00000000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610
.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.00000000000
2, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.00000
0000002, -38443.61267879984, 9610.000000000002, 9610.000000000002, 2.612678
7998251304, 9610.000000000002, 9610.000000000002, -38442.28353030619, 9610.
000000000002, 9610.000000000002, 1.2835303061813406, 9610.000000000002, 961
0.000000000002, -38440.71116701313, 9610.000000000002, 9610.000000000002, -
0.2888329868832211, 9610.000000000002, 9610.000000000002, -38439.0221869958
, 9610.000000000002, 9610.000000000002, -1.9778130042102862, 9610.000000000
002, 9610.000000000002, -38437.29792765007, 9610.000000000002, 9610.0000000
00002, -3.7020723499392125, 9610.000000000002, 9610.000000000002, -38435.59
748138142, 9610.000000000002, 9610.000000000002, -5.402518618588639, 9610.0
00000000002, 9610.000000000002, -38433.96640472125, 9610.000000000002, 9610
.000000000002, -7.033595278759009, 9610.000000000002, 9610.000000000002, -3
8432.440946788614, 9610.000000000002, 9610.000000000002, -8.559053211390776
, 9610.000000000002, 9610.000000000002, -38431.05041241193, 9610.0000000000
02, 9610.000000000002, -9.949587588075156, 9610.000000000002, 9610.00000000
0002, -38429.81860731687, 9610.000000000002, 9610.000000000002, -11.1813926
8314126, 9610.000000000002, 9610.000000000002, -38428.764776813994, 9610.00
0000000002, 9610.000000000002, -12.235223186011531, 9610.000000000002, 9610
.000000000002, -38427.904239813724, 9610.000000000002, 9610.000000000002, -
13.095760186282172, 9610.000000000002, 9610.000000000002, -38427.2488261199
76, 9610.000000000002, 9610.000000000002, -13.751173880032615, 9610.0000000
00002, 9610.000000000002, -38426.807178376315, 9610.000000000002, 9610.0000
00000002, -14.192821623692792, 9610.000000000002, 9610.000000000002, -38426
.58495489231, 9610.000000000002, 9610.000000000002, -14.415045107698608, 96
10.000000000002, 9610.000000000002, -38426.58495489231, 9610.000000000002, 
9610.000000000002, -14.415045107698608, 9610.000000000002, 9610.00000000000
2, -38426.807178376315, 9610.000000000002, 9610.000000000002, -14.192821623
692792, 9610.000000000002, 9610.000000000002, -38427.248826119976, 9610.000
000000002, 9610.000000000002, -13.751173880032615, 9610.000000000002, 9610.
000000000002, -38427.904239813724, 9610.000000000002, 9610.000000000002, -1
3.095760186282172, 9610.000000000002, 9610.000000000002, -38428.76477681399
4, 9610.000000000002, 9610.000000000002, -12.235223186011531, 9610.00000000
0002, 9610.000000000002, -38429.81860731687, 9610.000000000002, 9610.000000
000002, -11.18139268314126, 9610.000000000002, 9610.000000000002, -38431.05
041241193, 9610.000000000002, 9610.000000000002, -9.949587588075154, 9610.0
00000000002, 9610.000000000002, -38432.440946788614, 9610.000000000002, 961
0.000000000002, -8.559053211390776, 9610.000000000002, 9610.000000000002, -
38433.96640472125, 9610.000000000002, 9610.000000000002, -7.033595278759011
, 9610.000000000002, 9610.000000000002, -38435.59748138142, 9610.0000000000
02, 9610.000000000002, -5.4025186185886405, 9610.000000000002, 9610.0000000
00002, -38437.29792765007, 9610.000000000002, 9610.000000000002, -3.7020723
499392107, 9610.000000000002, 9610.000000000002, -38439.0221869958, 9610.00
0000000002, 9610.000000000002, -1.9778130042102862, 9610.000000000002, 9610
.000000000002, -38440.71116701313, 9610.000000000002, 9610.000000000002, -0
.2888329868832229, 9610.000000000002, 9610.000000000002, -38442.28353030619
, 9610.000000000002, 9610.000000000002, 1.2835303061813383, 9610.0000000000
02, 9610.000000000002, -38443.61267879984, 9610.000000000002, 9610.00000000
0002, 2.6126787998251313, 9610.000000000002, 9610.000000000002, 9610.000000
000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -384
44.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.
4, 9610.000000000002, 9610.000000000002, -38443.59240504866, 9610.000000000
002, 9610.000000000002, 2.592405048650707, 9610.000000000002, 9610.00000000
0002, -38442.22903059256, 9610.000000000002, 9610.000000000002, 1.229030592
5560028, 9610.000000000002, 9610.000000000002, -38440.61617848483, 9610.000
000000002, 9610.000000000002, -0.3838215151782367, 9610.000000000002, 9610.
000000000002, -38438.883706738074, 9610.000000000002, 9610.000000000002, -2
.1162932619319412, 9610.000000000002, 9610.000000000002, -38437.11504720989
6, 9610.000000000002, 9610.000000000002, -3.8849527901120937, 9610.00000000
0002, 9610.000000000002, -38435.3708139525, 9610.000000000002, 9610.0000000
00002, -5.629186047513276, 9610.000000000002, 9610.000000000002, -38433.697
73659128, 9610.000000000002, 9610.000000000002, -7.3022634087283524, 9610.0
00000000002, 9610.000000000002, -38432.13299767073, 9610.000000000002, 9610
.000000000002, -8.867002329280117, 9610.000000000002, 9610.000000000002, -3
8430.70665662715, 9610.000000000002, 9610.000000000002, -10.293343372857398
, 9610.000000000002, 9610.000000000002, -38429.443132190616, 9610.000000000
002, 9610.000000000002, -11.55686780939076, 9610.000000000002, 9610.0000000
00002, -38428.362165244085, 9610.000000000002, 9610.000000000002, -12.63783
4755926546, 9610.000000000002, 9610.000000000002, -38427.47946916437, 9610.
000000000002, 9610.000000000002, -13.520530835637695, 9610.000000000002, 96
10.000000000002, -38426.807178376315, 9610.000000000002, 9610.000000000002,
 -14.192821623692792, 9610.000000000002, 9610.000000000002, -38426.35415807
4086, 9610.000000000002, 9610.000000000002, -14.645841925922104, 9610.00000
0000002, 9610.000000000002, -38426.12621227037, 9610.000000000002, 9610.000
000000002, -14.873787729635366, 9610.000000000002, 9610.000000000002, -3842
6.12621227037, 9610.000000000002, 9610.000000000002, -14.873787729635366, 9
610.000000000002, 9610.000000000002, -38426.354158074086, 9610.000000000002
, 9610.000000000002, -14.645841925922108, 9610.000000000002, 9610.000000000
002, -38426.807178376315, 9610.000000000002, 9610.000000000002, -14.1928216
23692792, 9610.000000000002, 9610.000000000002, -38427.47946916437, 9610.00
0000000002, 9610.000000000002, -13.520530835637695, 9610.000000000002, 9610
.000000000002, -38428.362165244085, 9610.000000000002, 9610.000000000002, -
12.637834755926546, 9610.000000000002, 9610.000000000002, -38429.4431321906
16, 9610.000000000002, 9610.000000000002, -11.55686780939076, 9610.00000000
0002, 9610.000000000002, -38430.70665662715, 9610.000000000002, 9610.000000
000002, -10.293343372857397, 9610.000000000002, 9610.000000000002, -38432.1
3299767073, 9610.000000000002, 9610.000000000002, -8.867002329280117, 9610.
000000000002, 9610.000000000002, -38433.69773659128, 9610.000000000002, 961
0.000000000002, -7.3022634087283524, 9610.000000000002, 9610.000000000002, 
-38435.3708139525, 9610.000000000002, 9610.000000000002, -5.629186047513278
, 9610.000000000002, 9610.000000000002, -38437.115047209896, 9610.000000000
002, 9610.000000000002, -3.884952790112092, 9610.000000000002, 9610.0000000
00002, -38438.883706738074, 9610.000000000002, 9610.000000000002, -2.116293
2619319412, 9610.000000000002, 9610.000000000002, -38440.61617848483, 9610.
000000000002, 9610.000000000002, -0.3838215151782385, 9610.000000000002, 96
10.000000000002, -38442.22903059256, 9610.000000000002, 9610.000000000002, 
1.2290305925560001, 9610.000000000002, 9610.000000000002, -38443.5924050486
6, 9610.000000000002, 9610.000000000002, 2.5924050486507078, 9610.000000000
002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.0000000
00002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.
000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002,
 -38443.582203924154, 9610.000000000002, 9610.000000000002, 2.5822039241470
254, 9610.000000000002, 9610.000000000002, -38442.2016080224, 9610.00000000
0002, 9610.000000000002, 1.2016080223900816, 9610.000000000002, 9610.000000
000002, -38440.56838319548, 9610.000000000002, 9610.000000000002, -0.431616
8045266302, 9610.000000000002, 9610.000000000002, -38438.814027755725, 9610
.000000000002, 9610.000000000002, -2.1859722442853706, 9610.000000000002, 9
610.000000000002, -38437.02302742907, 9610.000000000002, 9610.000000000002,
 -3.976972570933176, 9610.000000000002, 9610.000000000002, -38435.256761913
31, 9610.000000000002, 9610.000000000002, -5.743238086699128, 9610.00000000
0002, 9610.000000000002, -38433.562551098454, 9610.000000000002, 9610.00000
0000002, -7.437448901556442, 9610.000000000002, 9610.000000000002, -38431.9
7804719978, 9610.000000000002, 9610.000000000002, -9.021952800229522, 9610.
000000000002, 9610.000000000002, -38430.53368934896, 9610.000000000002, 961
0.000000000002, -10.466310651051492, 9610.000000000002, 9610.000000000002, 
-38429.254204720906, 9610.000000000002, 9610.000000000002, -11.745795279100
395, 9610.000000000002, 9610.000000000002, -38428.15958355528, 9610.0000000
00002, 9610.000000000002, -12.840416444731476, 9610.000000000002, 9610.0000
00000002, -38427.265737712194, 9610.000000000002, 9610.000000000002, -13.73
426228781472, 9610.000000000002, 9610.000000000002, -38426.58495489231, 961
0.000000000002, 9610.000000000002, -14.415045107698608, 9610.000000000002, 
9610.000000000002, -38426.12621227037, 9610.000000000002, 9610.000000000002
, -14.873787729635366, 9610.000000000002, 9610.000000000002, -38425.8953871
7238, 9610.000000000002, 9610.000000000002, -15.104612827628367, 9610.00000
0000002, 9610.000000000002, -38425.89538717238, 9610.000000000002, 9610.000
000000002, -15.104612827628367, 9610.000000000002, 9610.000000000002, -3842
6.12621227037, 9610.000000000002, 9610.000000000002, -14.873787729635366, 9
610.000000000002, 9610.000000000002, -38426.58495489231, 9610.000000000002,
 9610.000000000002, -14.415045107698608, 9610.000000000002, 9610.0000000000
02, -38427.265737712194, 9610.000000000002, 9610.000000000002, -13.73426228
781472, 9610.000000000002, 9610.000000000002, -38428.15958355528, 9610.0000
00000002, 9610.000000000002, -12.840416444731476, 9610.000000000002, 9610.0
00000000002, -38429.254204720906, 9610.000000000002, 9610.000000000002, -11
.745795279100395, 9610.000000000002, 9610.000000000002, -38430.53368934896,
 9610.000000000002, 9610.000000000002, -10.46631065105149, 9610.00000000000
2, 9610.000000000002, -38431.97804719978, 9610.000000000002, 9610.000000000
002, -9.021952800229522, 9610.000000000002, 9610.000000000002, -38433.56255
1098454, 9610.000000000002, 9610.000000000002, -7.437448901556444, 9610.000
000000002, 9610.000000000002, -38435.25676191331, 9610.000000000002, 9610.0
00000000002, -5.7432380866991295, 9610.000000000002, 9610.000000000002, -38
437.02302742907, 9610.000000000002, 9610.000000000002, -3.976972570933174, 
9610.000000000002, 9610.000000000002, -38438.814027755725, 9610.00000000000
2, 9610.000000000002, -2.1859722442853706, 9610.000000000002, 9610.00000000
0002, -38440.56838319548, 9610.000000000002, 9610.000000000002, -0.43161680
4526632, 9610.000000000002, 9610.000000000002, -38442.2016080224, 9610.0000
00000002, 9610.000000000002, 1.201608022390079, 9610.000000000002, 9610.000
000000002, -38443.582203924154, 9610.000000000002, 9610.000000000002, 2.582
2039241470263, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38
444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000
001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000
000000002, 9610.000000000002, -38443.582203924154, 9610.000000000002, 9610.
000000000002, 2.5822039241470254, 9610.000000000002, 9610.000000000002, -38
442.2016080224, 9610.000000000002, 9610.000000000002, 1.2016080223900816, 9
610.000000000002, 9610.000000000002, -38440.56838319548, 9610.000000000002,
 9610.000000000002, -0.4316168045266302, 9610.000000000002, 9610.0000000000
02, -38438.814027755725, 9610.000000000002, 9610.000000000002, -2.185972244
2853706, 9610.000000000002, 9610.000000000002, -38437.02302742907, 9610.000
000000002, 9610.000000000002, -3.976972570933176, 9610.000000000002, 9610.0
00000000002, -38435.25676191331, 9610.000000000002, 9610.000000000002, -5.7
43238086699128, 9610.000000000002, 9610.000000000002, -38433.562551098454, 
9610.000000000002, 9610.000000000002, -7.437448901556442, 9610.000000000002
, 9610.000000000002, -38431.97804719978, 9610.000000000002, 9610.0000000000
02, -9.021952800229522, 9610.000000000002, 9610.000000000002, -38430.533689
34896, 9610.000000000002, 9610.000000000002, -10.466310651051492, 9610.0000
00000002, 9610.000000000002, -38429.254204720906, 9610.000000000002, 9610.0
00000000002, -11.745795279100395, 9610.000000000002, 9610.000000000002, -38
428.15958355528, 9610.000000000002, 9610.000000000002, -12.840416444731476,
 9610.000000000002, 9610.000000000002, -38427.265737712194, 9610.0000000000
02, 9610.000000000002, -13.73426228781472, 9610.000000000002, 9610.00000000
0002, -38426.58495489231, 9610.000000000002, 9610.000000000002, -14.4150451
07698608, 9610.000000000002, 9610.000000000002, -38426.12621227037, 9610.00
0000000002, 9610.000000000002, -14.873787729635366, 9610.000000000002, 9610
.000000000002, -38425.89538717238, 9610.000000000002, 9610.000000000002, -1
5.104612827628367, 9610.000000000002, 9610.000000000002, -38425.89538717238
, 9610.000000000002, 9610.000000000002, -15.104612827628367, 9610.000000000
002, 9610.000000000002, -38426.12621227037, 9610.000000000002, 9610.0000000
00002, -14.873787729635366, 9610.000000000002, 9610.000000000002, -38426.58
495489231, 9610.000000000002, 9610.000000000002, -14.415045107698608, 9610.
000000000002, 9610.000000000002, -38427.265737712194, 9610.000000000002, 96
10.000000000002, -13.73426228781472, 9610.000000000002, 9610.000000000002, 
-38428.15958355528, 9610.000000000002, 9610.000000000002, -12.8404164447314
76, 9610.000000000002, 9610.000000000002, -38429.254204720906, 9610.0000000
00002, 9610.000000000002, -11.745795279100395, 9610.000000000002, 9610.0000
00000002, -38430.53368934896, 9610.000000000002, 9610.000000000002, -10.466
31065105149, 9610.000000000002, 9610.000000000002, -38431.97804719978, 9610
.000000000002, 9610.000000000002, -9.021952800229522, 9610.000000000002, 96
10.000000000002, -38433.562551098454, 9610.000000000002, 9610.000000000002,
 -7.437448901556444, 9610.000000000002, 9610.000000000002, -38435.256761913
31, 9610.000000000002, 9610.000000000002, -5.7432380866991295, 9610.0000000
00002, 9610.000000000002, -38437.02302742907, 9610.000000000002, 9610.00000
0000002, -3.976972570933174, 9610.000000000002, 9610.000000000002, -38438.8
14027755725, 9610.000000000002, 9610.000000000002, -2.1859722442853706, 961
0.000000000002, 9610.000000000002, -38440.56838319548, 9610.000000000002, 9
610.000000000002, -0.431616804526632, 9610.000000000002, 9610.000000000002,
 -38442.2016080224, 9610.000000000002, 9610.000000000002, 1.201608022390079
, 9610.000000000002, 9610.000000000002, -38443.582203924154, 9610.000000000
002, 9610.000000000002, 2.5822039241470263, 9610.000000000002, 9610.0000000
00002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.
000000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 961
0.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -38443.592405048
66, 9610.000000000002, 9610.000000000002, 2.592405048650707, 9610.000000000
002, 9610.000000000002, -38442.22903059256, 9610.000000000002, 9610.0000000
00002, 1.2290305925560028, 9610.000000000002, 9610.000000000002, -38440.616
17848483, 9610.000000000002, 9610.000000000002, -0.3838215151782367, 9610.0
00000000002, 9610.000000000002, -38438.883706738074, 9610.000000000002, 961
0.000000000002, -2.1162932619319412, 9610.000000000002, 9610.000000000002, 
-38437.115047209896, 9610.000000000002, 9610.000000000002, -3.8849527901120
937, 9610.000000000002, 9610.000000000002, -38435.3708139525, 9610.00000000
0002, 9610.000000000002, -5.629186047513276, 9610.000000000002, 9610.000000
000002, -38433.69773659128, 9610.000000000002, 9610.000000000002, -7.302263
4087283524, 9610.000000000002, 9610.000000000002, -38432.13299767073, 9610.
000000000002, 9610.000000000002, -8.867002329280117, 9610.000000000002, 961
0.000000000002, -38430.70665662715, 9610.000000000002, 9610.000000000002, -
10.293343372857398, 9610.000000000002, 9610.000000000002, -38429.4431321906
16, 9610.000000000002, 9610.000000000002, -11.55686780939076, 9610.00000000
0002, 9610.000000000002, -38428.362165244085, 9610.000000000002, 9610.00000
0000002, -12.637834755926546, 9610.000000000002, 9610.000000000002, -38427.
47946916437, 9610.000000000002, 9610.000000000002, -13.520530835637695, 961
0.000000000002, 9610.000000000002, -38426.807178376315, 9610.000000000002, 
9610.000000000002, -14.192821623692792, 9610.000000000002, 9610.00000000000
2, -38426.354158074086, 9610.000000000002, 9610.000000000002, -14.645841925
922104, 9610.000000000002, 9610.000000000002, -38426.12621227037, 9610.0000
00000002, 9610.000000000002, -14.873787729635366, 9610.000000000002, 9610.0
00000000002, -38426.12621227037, 9610.000000000002, 9610.000000000002, -14.
873787729635366, 9610.000000000002, 9610.000000000002, -38426.354158074086,
 9610.000000000002, 9610.000000000002, -14.645841925922108, 9610.0000000000
02, 9610.000000000002, -38426.807178376315, 9610.000000000002, 9610.0000000
00002, -14.192821623692792, 9610.000000000002, 9610.000000000002, -38427.47
946916437, 9610.000000000002, 9610.000000000002, -13.520530835637695, 9610.
000000000002, 9610.000000000002, -38428.362165244085, 9610.000000000002, 96
10.000000000002, -12.637834755926546, 9610.000000000002, 9610.000000000002,
 -38429.443132190616, 9610.000000000002, 9610.000000000002, -11.55686780939
076, 9610.000000000002, 9610.000000000002, -38430.70665662715, 9610.0000000
00002, 9610.000000000002, -10.293343372857397, 9610.000000000002, 9610.0000
00000002, -38432.13299767073, 9610.000000000002, 9610.000000000002, -8.8670
02329280117, 9610.000000000002, 9610.000000000002, -38433.69773659128, 9610
.000000000002, 9610.000000000002, -7.3022634087283524, 9610.000000000002, 9
610.000000000002, -38435.3708139525, 9610.000000000002, 9610.000000000002, 
-5.629186047513278, 9610.000000000002, 9610.000000000002, -38437.1150472098
96, 9610.000000000002, 9610.000000000002, -3.884952790112092, 9610.00000000
0002, 9610.000000000002, -38438.883706738074, 9610.000000000002, 9610.00000
0000002, -2.1162932619319412, 9610.000000000002, 9610.000000000002, -38440.
61617848483, 9610.000000000002, 9610.000000000002, -0.3838215151782385, 961
0.000000000002, 9610.000000000002, -38442.22903059256, 9610.000000000002, 9
610.000000000002, 1.2290305925560001, 9610.000000000002, 9610.000000000002,
 -38443.59240504866, 9610.000000000002, 9610.000000000002, 2.59240504865070
78, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.4000000
0001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.0
00000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 
9610.000000000002, -38443.61267879984, 9610.000000000002, 9610.000000000002
, 2.6126787998251304, 9610.000000000002, 9610.000000000002, -38442.28353030
619, 9610.000000000002, 9610.000000000002, 1.2835303061813406, 9610.0000000
00002, 9610.000000000002, -38440.71116701313, 9610.000000000002, 9610.00000
0000002, -0.2888329868832211, 9610.000000000002, 9610.000000000002, -38439.
0221869958, 9610.000000000002, 9610.000000000002, -1.9778130042102862, 9610
.000000000002, 9610.000000000002, -38437.29792765007, 9610.000000000002, 96
10.000000000002, -3.7020723499392125, 9610.000000000002, 9610.000000000002,
 -38435.59748138142, 9610.000000000002, 9610.000000000002, -5.4025186185886
39, 9610.000000000002, 9610.000000000002, -38433.96640472125, 9610.00000000
0002, 9610.000000000002, -7.033595278759009, 9610.000000000002, 9610.000000
000002, -38432.440946788614, 9610.000000000002, 9610.000000000002, -8.55905
3211390776, 9610.000000000002, 9610.000000000002, -38431.05041241193, 9610.
000000000002, 9610.000000000002, -9.949587588075156, 9610.000000000002, 961
0.000000000002, -38429.81860731687, 9610.000000000002, 9610.000000000002, -
11.18139268314126, 9610.000000000002, 9610.000000000002, -38428.76477681399
4, 9610.000000000002, 9610.000000000002, -12.235223186011531, 9610.00000000
0002, 9610.000000000002, -38427.904239813724, 9610.000000000002, 9610.00000
0000002, -13.095760186282172, 9610.000000000002, 9610.000000000002, -38427.
248826119976, 9610.000000000002, 9610.000000000002, -13.751173880032615, 96
10.000000000002, 9610.000000000002, -38426.807178376315, 9610.000000000002,
 9610.000000000002, -14.192821623692792, 9610.000000000002, 9610.0000000000
02, -38426.58495489231, 9610.000000000002, 9610.000000000002, -14.415045107
698608, 9610.000000000002, 9610.000000000002, -38426.58495489231, 9610.0000
00000002, 9610.000000000002, -14.415045107698608, 9610.000000000002, 9610.0
00000000002, -38426.807178376315, 9610.000000000002, 9610.000000000002, -14
.192821623692792, 9610.000000000002, 9610.000000000002, -38427.248826119976
, 9610.000000000002, 9610.000000000002, -13.751173880032615, 9610.000000000
002, 9610.000000000002, -38427.904239813724, 9610.000000000002, 9610.000000
000002, -13.095760186282172, 9610.000000000002, 9610.000000000002, -38428.7
64776813994, 9610.000000000002, 9610.000000000002, -12.235223186011531, 961
0.000000000002, 9610.000000000002, -38429.81860731687, 9610.000000000002, 9
610.000000000002, -11.18139268314126, 9610.000000000002, 9610.000000000002,
 -38431.05041241193, 9610.000000000002, 9610.000000000002, -9.9495875880751
54, 9610.000000000002, 9610.000000000002, -38432.440946788614, 9610.0000000
00002, 9610.000000000002, -8.559053211390776, 9610.000000000002, 9610.00000
0000002, -38433.96640472125, 9610.000000000002, 9610.000000000002, -7.03359
5278759011, 9610.000000000002, 9610.000000000002, -38435.59748138142, 9610.
000000000002, 9610.000000000002, -5.4025186185886405, 9610.000000000002, 96
10.000000000002, -38437.29792765007, 9610.000000000002, 9610.000000000002, 
-3.7020723499392107, 9610.000000000002, 9610.000000000002, -38439.022186995
8, 9610.000000000002, 9610.000000000002, -1.9778130042102862, 9610.00000000
0002, 9610.000000000002, -38440.71116701313, 9610.000000000002, 9610.000000
000002, -0.2888329868832229, 9610.000000000002, 9610.000000000002, -38442.2
8353030619, 9610.000000000002, 9610.000000000002, 1.2835303061813383, 9610.
000000000002, 9610.000000000002, -38443.61267879984, 9610.000000000002, 961
0.000000000002, 2.6126787998251313, 9610.000000000002, 9610.000000000002, 9
610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.00000000
0002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000
000002, 3.4, 9610.000000000002, 9610.000000000002, -38443.64276543411, 9610
.000000000002, 9610.000000000002, 2.64276543410358, 9610.000000000002, 9610
.000000000002, -38442.36440892298, 9610.000000000002, 9610.000000000002, 1.
3644089229710339, 9610.000000000002, 9610.000000000002, -38440.85213180484,
 9610.000000000002, 9610.000000000002, -0.14786819517434013, 9610.000000000
002, 9610.000000000002, -38439.227694345325, 9610.000000000002, 9610.000000
000002, -1.7723056546817468, 9610.000000000002, 9610.000000000002, -38437.5
6932571907, 9610.000000000002, 9610.000000000002, -3.4306742809373216, 9610
.000000000002, 9610.000000000002, -38435.933860178724, 9610.000000000002, 9
610.000000000002, -5.066139821284695, 9610.000000000002, 9610.000000000002,
 -38434.36511336175, 9610.000000000002, 9610.000000000002, -6.6348866382554
13, 9610.000000000002, 9610.000000000002, -38432.89794916606, 9610.00000000
0002, 9610.000000000002, -8.102050833952296, 9610.000000000002, 9610.000000
000002, -38431.56055256749, 9610.000000000002, 9610.000000000002, -9.439447
432518136, 9610.000000000002, 9610.000000000002, -38430.37581958148, 9610.0
00000000002, 9610.000000000002, -10.624180418527155, 9610.000000000002, 961
0.000000000002, -38429.362260079724, 9610.000000000002, 9610.000000000002, 
-11.63773992028087, 9610.000000000002, 9610.000000000002, -38428.5346075769
2, 9610.000000000002, 9610.000000000002, -12.465392423087875, 9610.00000000
0002, 9610.000000000002, -38427.904239813724, 9610.000000000002, 9610.00000
0000002, -13.095760186282176, 9610.000000000002, 9610.000000000002, -38427.
47946916437, 9610.000000000002, 9610.000000000002, -13.520530835637695, 961
0.000000000002, 9610.000000000002, -38427.265737712194, 9610.000000000002, 
9610.000000000002, -13.734262287814724, 9610.000000000002, 9610.00000000000
2, -38427.265737712194, 9610.000000000002, 9610.000000000002, -13.734262287
814724, 9610.000000000002, 9610.000000000002, -38427.47946916437, 9610.0000
00000002, 9610.000000000002, -13.520530835637699, 9610.000000000002, 9610.0
00000000002, -38427.904239813724, 9610.000000000002, 9610.000000000002, -13
.095760186282176, 9610.000000000002, 9610.000000000002, -38428.53460757692,
 9610.000000000002, 9610.000000000002, -12.465392423087875, 9610.0000000000
02, 9610.000000000002, -38429.362260079724, 9610.000000000002, 9610.0000000
00002, -11.63773992028087, 9610.000000000002, 9610.000000000002, -38430.375
81958148, 9610.000000000002, 9610.000000000002, -10.624180418527155, 9610.0
00000000002, 9610.000000000002, -38431.56055256749, 9610.000000000002, 9610
.000000000002, -9.439447432518133, 9610.000000000002, 9610.000000000002, -3
8432.89794916606, 9610.000000000002, 9610.000000000002, -8.102050833952296,
 9610.000000000002, 9610.000000000002, -38434.36511336175, 9610.00000000000
2, 9610.000000000002, -6.634886638255415, 9610.000000000002, 9610.000000000
002, -38435.933860178724, 9610.000000000002, 9610.000000000002, -5.06613982
1284697, 9610.000000000002, 9610.000000000002, -38437.56932571907, 9610.000
000000002, 9610.000000000002, -3.43067428093732, 9610.000000000002, 9610.00
0000000002, -38439.227694345325, 9610.000000000002, 9610.000000000002, -1.7
723056546817468, 9610.000000000002, 9610.000000000002, -38440.85213180484, 
9610.000000000002, 9610.000000000002, -0.1478681951743419, 9610.00000000000
2, 9610.000000000002, -38442.36440892298, 9610.000000000002, 9610.000000000
002, 1.3644089229710312, 9610.000000000002, 9610.000000000002, -38443.64276
543411, 9610.000000000002, 9610.000000000002, 2.642765434103581, 9610.00000
0000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000
000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9
610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000
002, -38443.68226821267, 9610.000000000002, 9610.000000000002, 2.6822682126
64803, 9610.000000000002, 9610.000000000002, -38442.470599933244, 9610.0000
00000002, 9610.000000000002, 1.4705999332320157, 9610.000000000002, 9610.00
0000000002, -38441.03721402109, 9610.000000000002, 9610.000000000002, 0.037
21402107719518, 9610.000000000002, 9610.000000000002, -38439.49751885431, 9
610.000000000002, 9610.000000000002, -1.5024811456988805, 9610.000000000002
, 9610.000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000
002, -3.074337386537524, 9610.000000000002, 9610.000000000002, -38436.37551
466954, 9610.000000000002, 9610.000000000002, -4.6244853304696125, 9610.000
000000002, 9610.000000000002, -38434.888604922504, 9610.000000000002, 9610.
000000000002, -6.111395077501431, 9610.000000000002, 9610.000000000002, -38
433.49797851965, 9610.000000000002, 9610.000000000002, -7.502021480358708, 
9610.000000000002, 9610.000000000002, -38432.23035010661, 9610.000000000002
, 9610.000000000002, -8.769649893396544, 9610.000000000002, 9610.0000000000
02, -38431.10742127866, 9610.000000000002, 9610.000000000002, -9.8925787213
43803, 9610.000000000002, 9610.000000000002, -38430.14673630003, 9610.00000
0000002, 9610.000000000002, -10.853263699984586, 9610.000000000002, 9610.00
0000000002, -38429.362260079724, 9610.000000000002, 9610.000000000002, -11.
63773992028087, 9610.000000000002, 9610.000000000002, -38428.764776813994, 
9610.000000000002, 9610.000000000002, -12.235223186011535, 9610.00000000000
2, 9610.000000000002, -38428.362165244085, 9610.000000000002, 9610.00000000
0002, -12.637834755926546, 9610.000000000002, 9610.000000000002, -38428.159
58355528, 9610.000000000002, 9610.000000000002, -12.840416444731476, 9610.0
00000000002, 9610.000000000002, -38428.15958355528, 9610.000000000002, 9610
.000000000002, -12.840416444731476, 9610.000000000002, 9610.000000000002, -
38428.362165244085, 9610.000000000002, 9610.000000000002, -12.6378347559265
5, 9610.000000000002, 9610.000000000002, -38428.764776813994, 9610.00000000
0002, 9610.000000000002, -12.235223186011535, 9610.000000000002, 9610.00000
0000002, -38429.362260079724, 9610.000000000002, 9610.000000000002, -11.637
73992028087, 9610.000000000002, 9610.000000000002, -38430.14673630003, 9610
.000000000002, 9610.000000000002, -10.853263699984586, 9610.000000000002, 9
610.000000000002, -38431.10742127866, 9610.000000000002, 9610.000000000002,
 -9.892578721343803, 9610.000000000002, 9610.000000000002, -38432.230350106
61, 9610.000000000002, 9610.000000000002, -8.769649893396542, 9610.00000000
0002, 9610.000000000002, -38433.49797851965, 9610.000000000002, 9610.000000
000002, -7.502021480358708, 9610.000000000002, 9610.000000000002, -38434.88
8604922504, 9610.000000000002, 9610.000000000002, -6.111395077501433, 9610.
000000000002, 9610.000000000002, -38436.37551466954, 9610.000000000002, 961
0.000000000002, -4.624485330469614, 9610.000000000002, 9610.000000000002, -
38437.925662613474, 9610.000000000002, 9610.000000000002, -3.07433738653752
23, 9610.000000000002, 9610.000000000002, -38439.49751885431, 9610.00000000
0002, 9610.000000000002, -1.5024811456988805, 9610.000000000002, 9610.00000
0000002, -38441.03721402109, 9610.000000000002, 9610.000000000002, 0.037214
021077193404, 9610.000000000002, 9610.000000000002, -38442.470599933244, 96
10.000000000002, 9610.000000000002, 1.4705999332320137, 9610.000000000002, 
9610.000000000002, -38443.68226821267, 9610.000000000002, 9610.000000000002
, 2.6822682126648036, 9610.000000000002, 9610.000000000002, 9610.0000000000
02, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.4
0000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9
610.000000000002, 9610.000000000002, -38443.73064409074, 9610.000000000002,
 9610.000000000002, 2.7306440907303067, 9610.000000000002, 9610.00000000000
2, -38442.60064352894, 9610.000000000002, 9610.000000000002, 1.600643528926
8413, 9610.000000000002, 9610.000000000002, -38441.26386933627, 9610.000000
000002, 9610.000000000002, 0.2638693362621991, 9610.000000000002, 9610.0000
00000002, -38439.82795124467, 9610.000000000002, 9610.000000000002, -1.1720
487553441257, 9610.000000000002, 9610.000000000002, -38438.36203976924, 961
0.000000000002, 9610.000000000002, -2.637960230763317, 9610.000000000002, 9
610.000000000002, -38436.91637342867, 9610.000000000002, 9610.000000000002,
 -4.083626571341076, 9610.000000000002, 9610.000000000002, -38435.529682963
94, 9610.000000000002, 9610.000000000002, -5.4703170360642694, 9610.0000000
00002, 9610.000000000002, -38434.23278624463, 9610.000000000002, 9610.00000
0000002, -6.767213755373509, 9610.000000000002, 9610.000000000002, -38433.0
5059732115, 9610.000000000002, 9610.000000000002, -7.94940267886447, 9610.0
00000000002, 9610.000000000002, -38432.00335507838, 9610.000000000002, 9610
.000000000002, -8.996644921633852, 9610.000000000002, 9610.000000000002, -3
8431.10742127866, 9610.000000000002, 9610.000000000002, -9.892578721343801,
 9610.000000000002, 9610.000000000002, -38430.37581958148, 9610.00000000000
2, 9610.000000000002, -10.624180418527155, 9610.000000000002, 9610.00000000
0002, -38429.81860731687, 9610.000000000002, 9610.000000000002, -11.1813926
83141262, 9610.000000000002, 9610.000000000002, -38429.443132190616, 9610.0
00000000002, 9610.000000000002, -11.55686780939076, 9610.000000000002, 9610
.000000000002, -38429.254204720906, 9610.000000000002, 9610.000000000002, -
11.745795279100397, 9610.000000000002, 9610.000000000002, -38429.2542047209
06, 9610.000000000002, 9610.000000000002, -11.745795279100397, 9610.0000000
00002, 9610.000000000002, -38429.443132190616, 9610.000000000002, 9610.0000
00000002, -11.556867809390761, 9610.000000000002, 9610.000000000002, -38429
.81860731687, 9610.000000000002, 9610.000000000002, -11.181392683141262, 96
10.000000000002, 9610.000000000002, -38430.37581958148, 9610.000000000002, 
9610.000000000002, -10.624180418527155, 9610.000000000002, 9610.00000000000
2, -38431.10742127866, 9610.000000000002, 9610.000000000002, -9.89257872134
3801, 9610.000000000002, 9610.000000000002, -38432.00335507838, 9610.000000
000002, 9610.000000000002, -8.996644921633852, 9610.000000000002, 9610.0000
00000002, -38433.05059732115, 9610.000000000002, 9610.000000000002, -7.9494
02678864468, 9610.000000000002, 9610.000000000002, -38434.23278624463, 9610
.000000000002, 9610.000000000002, -6.767213755373509, 9610.000000000002, 96
10.000000000002, -38435.52968296394, 9610.000000000002, 9610.000000000002, 
-5.4703170360642694, 9610.000000000002, 9610.000000000002, -38436.916373428
67, 9610.000000000002, 9610.000000000002, -4.083626571341078, 9610.00000000
0002, 9610.000000000002, -38438.36203976924, 9610.000000000002, 9610.000000
000002, -2.6379602307633143, 9610.000000000002, 9610.000000000002, -38439.8
2795124467, 9610.000000000002, 9610.000000000002, -1.1720487553441257, 9610
.000000000002, 9610.000000000002, -38441.26386933627, 9610.000000000002, 96
10.000000000002, 0.26386933626219733, 9610.000000000002, 9610.000000000002,
 -38442.60064352894, 9610.000000000002, 9610.000000000002, 1.60064352892683
93, 9610.000000000002, 9610.000000000002, -38443.73064409074, 9610.00000000
0002, 9610.000000000002, 2.7306440907303076, 9610.000000000002, 9610.000000
000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610
.000000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 96
10.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -38443.78718985
68, 9610.000000000002, 9610.000000000002, 2.7871898567876396, 9610.00000000
0002, 9610.000000000002, -38442.75264934327, 9610.000000000002, 9610.000000
000002, 1.7526493432597003, 9610.000000000002, 9610.000000000002, -38441.52
8802996196, 9610.000000000002, 9610.000000000002, 0.5288029961897389, 9610.
000000000002, 9610.000000000002, -38440.21418821627, 9610.000000000002, 961
0.000000000002, -0.7858117837388678, 9610.000000000002, 9610.000000000002, 
-38438.872113829915, 9610.000000000002, 9610.000000000002, -2.1278861700969
46, 9610.000000000002, 9610.000000000002, -38437.548574312386, 9610.0000000
00002, 9610.000000000002, -3.4514256876210925, 9610.000000000002, 9610.0000
00000002, -38436.27902851394, 9610.000000000002, 9610.000000000002, -4.7209
71486066933, 9610.000000000002, 9610.000000000002, -38435.09169087595, 9610
.000000000002, 9610.000000000002, -5.90830912406382, 9610.000000000002, 961
0.000000000002, -38434.00937076272, 9610.000000000002, 9610.000000000002, -
6.990629237288765, 9610.000000000002, 9610.000000000002, -38433.05059732115
, 9610.000000000002, 9610.000000000002, -7.949402678864468, 9610.0000000000
02, 9610.000000000002, -38432.23035010661, 9610.000000000002, 9610.00000000
0002, -8.76964989339654, 9610.000000000002, 9610.000000000002, -38431.56055
256749, 9610.000000000002, 9610.000000000002, -9.439447432518133, 9610.0000
00000002, 9610.000000000002, -38431.05041241193, 9610.000000000002, 9610.00
0000000002, -9.949587588075156, 9610.000000000002, 9610.000000000002, -3843
0.70665662715, 9610.000000000002, 9610.000000000002, -10.293343372857397, 9
610.000000000002, 9610.000000000002, -38430.53368934896, 9610.000000000002,
 9610.000000000002, -10.466310651051492, 9610.000000000002, 9610.0000000000
02, -38430.53368934896, 9610.000000000002, 9610.000000000002, -10.466310651
051492, 9610.000000000002, 9610.000000000002, -38430.70665662715, 9610.0000
00000002, 9610.000000000002, -10.293343372857398, 9610.000000000002, 9610.0
00000000002, -38431.05041241193, 9610.000000000002, 9610.000000000002, -9.9
49587588075156, 9610.000000000002, 9610.000000000002, -38431.56055256749, 9
610.000000000002, 9610.000000000002, -9.439447432518133, 9610.000000000002,
 9610.000000000002, -38432.23035010661, 9610.000000000002, 9610.00000000000
2, -8.76964989339654, 9610.000000000002, 9610.000000000002, -38433.05059732
115, 9610.000000000002, 9610.000000000002, -7.949402678864468, 9610.0000000
00002, 9610.000000000002, -38434.00937076272, 9610.000000000002, 9610.00000
0000002, -6.990629237288763, 9610.000000000002, 9610.000000000002, -38435.0
9169087595, 9610.000000000002, 9610.000000000002, -5.90830912406382, 9610.0
00000000002, 9610.000000000002, -38436.27902851394, 9610.000000000002, 9610
.000000000002, -4.720971486066933, 9610.000000000002, 9610.000000000002, -3
8437.548574312386, 9610.000000000002, 9610.000000000002, -3.451425687621093
4, 9610.000000000002, 9610.000000000002, -38438.872113829915, 9610.00000000
0002, 9610.000000000002, -2.1278861700969443, 9610.000000000002, 9610.00000
0000002, -38440.21418821627, 9610.000000000002, 9610.000000000002, -0.78581
17837388678, 9610.000000000002, 9610.000000000002, -38441.528802996196, 961
0.000000000002, 9610.000000000002, 0.5288029961897376, 9610.000000000002, 9
610.000000000002, -38442.75264934327, 9610.000000000002, 9610.000000000002,
 1.7526493432596986, 9610.000000000002, 9610.000000000002, -38443.787189856
8, 9610.000000000002, 9610.000000000002, 2.7871898567876405, 9610.000000000
002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.0000000
00002, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.
000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002,
 -38443.85102205872, 9610.000000000002, 9610.000000000002, 2.85102205871115
5, 9610.000000000002, 9610.000000000002, -38442.92424248827, 9610.000000000
002, 9610.000000000002, 1.9242424882567188, 9610.000000000002, 9610.0000000
00002, -38441.82787576603, 9610.000000000002, 9610.000000000002, 0.82787576
60180562, 9610.000000000002, 9610.000000000002, -38440.65019533194, 9610.00
0000000002, 9610.000000000002, -0.34980466807195576, 9610.000000000002, 961
0.000000000002, -38439.44791556904, 9610.000000000002, 9610.000000000002, -
1.5520844309642476, 9610.000000000002, 9610.000000000002, -38438.2622400256
6, 9610.000000000002, 9610.000000000002, -2.737759974348729, 9610.000000000
002, 9610.000000000002, -38437.1249340484, 9610.000000000002, 9610.00000000
0002, -3.8750659516115458, 9610.000000000002, 9610.000000000002, -38436.061
27317505, 9610.000000000002, 9610.000000000002, -4.938726824953864, 9610.00
0000000002, 9610.000000000002, -38435.09169087595, 9610.000000000002, 9610.
000000000002, -5.908309124063818, 9610.000000000002, 9610.000000000002, -38
434.23278624463, 9610.000000000002, 9610.000000000002, -6.767213755373509, 
9610.000000000002, 9610.000000000002, -38433.49797851965, 9610.000000000002
, 9610.000000000002, -7.502021480358705, 9610.000000000002, 9610.0000000000
02, -38432.89794916606, 9610.000000000002, 9610.000000000002, -8.1020508339
52292, 9610.000000000002, 9610.000000000002, -38432.440946788614, 9610.0000
00000002, 9610.000000000002, -8.559053211390776, 9610.000000000002, 9610.00
0000000002, -38432.13299767073, 9610.000000000002, 9610.000000000002, -8.86
7002329280114, 9610.000000000002, 9610.000000000002, -38431.97804719978, 96
10.000000000002, 9610.000000000002, -9.02195280022952, 9610.000000000002, 9
610.000000000002, -38431.97804719978, 9610.000000000002, 9610.000000000002,
 -9.02195280022952, 9610.000000000002, 9610.000000000002, -38432.1329976707
3, 9610.000000000002, 9610.000000000002, -8.867002329280115, 9610.000000000
002, 9610.000000000002, -38432.440946788614, 9610.000000000002, 9610.000000
000002, -8.559053211390776, 9610.000000000002, 9610.000000000002, -38432.89
794916606, 9610.000000000002, 9610.000000000002, -8.102050833952292, 9610.0
00000000002, 9610.000000000002, -38433.49797851965, 9610.000000000002, 9610
.000000000002, -7.502021480358705, 9610.000000000002, 9610.000000000002, -3
8434.23278624463, 9610.000000000002, 9610.000000000002, -6.767213755373509,
 9610.000000000002, 9610.000000000002, -38435.09169087595, 9610.00000000000
2, 9610.000000000002, -5.908309124063818, 9610.000000000002, 9610.000000000
002, -38436.06127317505, 9610.000000000002, 9610.000000000002, -4.938726824
953864, 9610.000000000002, 9610.000000000002, -38437.1249340484, 9610.00000
0000002, 9610.000000000002, -3.8750659516115467, 9610.000000000002, 9610.00
0000000002, -38438.26224002566, 9610.000000000002, 9610.000000000002, -2.73
7759974348729, 9610.000000000002, 9610.000000000002, -38439.44791556904, 96
10.000000000002, 9610.000000000002, -1.5520844309642459, 9610.000000000002,
 9610.000000000002, -38440.65019533194, 9610.000000000002, 9610.00000000000
2, -0.34980466807195576, 9610.000000000002, 9610.000000000002, -38441.82787
576603, 9610.000000000002, 9610.000000000002, 0.8278757660180549, 9610.0000
00000002, 9610.000000000002, -38442.92424248827, 9610.000000000002, 9610.00
0000000002, 1.9242424882567173, 9610.000000000002, 9610.000000000002, -3844
3.85102205872, 9610.000000000002, 9610.000000000002, 2.8510220587111554, 96
10.000000000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 
9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 9610.0000000
00002, 9610.000000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.0
00000000002, -38443.92104789944, 9610.000000000002, 9610.000000000002, 2.92
1047899434171, 9610.000000000002, 9610.000000000002, -38443.112485316786, 9
610.000000000002, 9610.000000000002, 2.1124853167764104, 9610.000000000002,
 9610.000000000002, -38442.155967567865, 9610.000000000002, 9610.0000000000
02, 1.1559675678594372, 9610.000000000002, 9610.000000000002, -38441.128508
21972, 9610.000000000002, 9610.000000000002, 0.12850821971431392, 9610.0000
00000002, 9610.000000000002, -38440.07958735315, 9610.000000000002, 9610.00
0000000002, -0.9204126468566698, 9610.000000000002, 9610.000000000002, -384
39.04515272585, 9610.000000000002, 9610.000000000002, -1.9548472741575451, 
9610.000000000002, 9610.000000000002, -38438.05291780013, 9610.000000000002
, 9610.000000000002, -2.9470821998765317, 9610.000000000002, 9610.000000000
002, -38437.1249340484, 9610.000000000002, 9610.000000000002, -3.8750659516
115484, 9610.000000000002, 9610.000000000002, -38436.27902851394, 9610.0000
00000002, 9610.000000000002, -4.720971486066935, 9610.000000000002, 9610.00
0000000002, -38435.52968296394, 9610.000000000002, 9610.000000000002, -5.47
0317036064271, 9610.000000000002, 9610.000000000002, -38434.888604922504, 9
610.000000000002, 9610.000000000002, -6.111395077501433, 9610.000000000002,
 9610.000000000002, -38434.36511336175, 9610.000000000002, 9610.00000000000
2, -6.634886638255415, 9610.000000000002, 9610.000000000002, -38433.9664047
2125, 9610.000000000002, 9610.000000000002, -7.033595278759012, 9610.000000
000002, 9610.000000000002, -38433.69773659128, 9610.000000000002, 9610.0000
00000002, -7.302263408728354, 9610.000000000002, 9610.000000000002, -38433.
562551098454, 9610.000000000002, 9610.000000000002, -7.437448901556444, 961
0.000000000002, 9610.000000000002, -38433.562551098454, 9610.000000000002, 
9610.000000000002, -7.437448901556444, 9610.000000000002, 9610.000000000002
, -38433.69773659128, 9610.000000000002, 9610.000000000002, -7.302263408728
354, 9610.000000000002, 9610.000000000002, -38433.96640472125, 9610.0000000
00002, 9610.000000000002, -7.033595278759012, 9610.000000000002, 9610.00000
0000002, -38434.36511336175, 9610.000000000002, 9610.000000000002, -6.63488
6638255415, 9610.000000000002, 9610.000000000002, -38434.888604922504, 9610
.000000000002, 9610.000000000002, -6.111395077501433, 9610.000000000002, 96
10.000000000002, -38435.52968296394, 9610.000000000002, 9610.000000000002, 
-5.470317036064271, 9610.000000000002, 9610.000000000002, -38436.2790285139
4, 9610.000000000002, 9610.000000000002, -4.720971486066933, 9610.000000000
002, 9610.000000000002, -38437.1249340484, 9610.000000000002, 9610.00000000
0002, -3.8750659516115484, 9610.000000000002, 9610.000000000002, -38438.052
91780013, 9610.000000000002, 9610.000000000002, -2.9470821998765326, 9610.0
00000000002, 9610.000000000002, -38439.04515272585, 9610.000000000002, 9610
.000000000002, -1.954847274157546, 9610.000000000002, 9610.000000000002, -3
8440.07958735315, 9610.000000000002, 9610.000000000002, -0.9204126468566689
, 9610.000000000002, 9610.000000000002, -38441.12850821972, 9610.0000000000
02, 9610.000000000002, 0.12850821971431392, 9610.000000000002, 9610.0000000
00002, -38442.155967567865, 9610.000000000002, 9610.000000000002, 1.1559675
678594363, 9610.000000000002, 9610.000000000002, -38443.112485316786, 9610.
000000000002, 9610.000000000002, 2.112485316776409, 9610.000000000002, 9610
.000000000002, -38443.92104789944, 9610.000000000002, 9610.000000000002, 2.
9210478994341713, 9610.000000000002, 9610.000000000002, 9610.000000000002, 
-38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000
000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.
000000000002, 9610.000000000002, -38443.99592214669, 9610.000000000002, 961
0.000000000002, 2.9959221466807455, 9610.000000000002, 9610.000000000002, -
38443.31376158765, 9610.000000000002, 9610.000000000002, 2.3137615876423077
, 9610.000000000002, 9610.000000000002, -38442.50677559012, 9610.0000000000
02, 9610.000000000002, 1.5067755901124502, 9610.000000000002, 9610.00000000
0002, -38441.63993824401, 9610.000000000002, 9610.000000000002, 0.639938243
9962098, 9610.000000000002, 9610.000000000002, -38440.75499444365, 9610.000
000000002, 9610.000000000002, -0.24500555636513832, 9610.000000000002, 9610
.000000000002, -38439.88227225889, 9610.000000000002, 9610.000000000002, -1
.1177277411202087, 9610.000000000002, 9610.000000000002, -38439.04515272585
, 9610.000000000002, 9610.000000000002, -1.9548472741575451, 9610.000000000
002, 9610.000000000002, -38438.26224002566, 9610.000000000002, 9610.0000000
00002, -2.7377599743487306, 9610.000000000002, 9610.000000000002, -38437.54
8574312386, 9610.000000000002, 9610.000000000002, -3.4514256876210943, 9610
.000000000002, 9610.000000000002, -38436.91637342867, 9610.000000000002, 96
10.000000000002, -4.083626571341078, 9610.000000000002, 9610.000000000002, 
-38436.37551466954, 9610.000000000002, 9610.000000000002, -4.62448533046961
4, 9610.000000000002, 9610.000000000002, -38435.933860178724, 9610.00000000
0002, 9610.000000000002, -5.066139821284695, 9610.000000000002, 9610.000000
000002, -38435.59748138142, 9610.000000000002, 9610.000000000002, -5.402518
618588642, 9610.000000000002, 9610.000000000002, -38435.3708139525, 9610.00
0000000002, 9610.000000000002, -5.629186047513278, 9610.000000000002, 9610.
000000000002, -38435.25676191331, 9610.000000000002, 9610.000000000002, -5.
7432380866991295, 9610.000000000002, 9610.000000000002, -38435.25676191331,
 9610.000000000002, 9610.000000000002, -5.7432380866991295, 9610.0000000000
02, 9610.000000000002, -38435.3708139525, 9610.000000000002, 9610.000000000
002, -5.629186047513279, 9610.000000000002, 9610.000000000002, -38435.59748
138142, 9610.000000000002, 9610.000000000002, -5.402518618588642, 9610.0000
00000002, 9610.000000000002, -38435.933860178724, 9610.000000000002, 9610.0
00000000002, -5.066139821284695, 9610.000000000002, 9610.000000000002, -384
36.37551466954, 9610.000000000002, 9610.000000000002, -4.624485330469614, 9
610.000000000002, 9610.000000000002, -38436.91637342867, 9610.000000000002,
 9610.000000000002, -4.083626571341078, 9610.000000000002, 9610.00000000000
2, -38437.548574312386, 9610.000000000002, 9610.000000000002, -3.4514256876
210934, 9610.000000000002, 9610.000000000002, -38438.26224002566, 9610.0000
00000002, 9610.000000000002, -2.7377599743487306, 9610.000000000002, 9610.0
00000000002, -38439.04515272585, 9610.000000000002, 9610.000000000002, -1.9
548472741575451, 9610.000000000002, 9610.000000000002, -38439.88227225889, 
9610.000000000002, 9610.000000000002, -1.1177277411202096, 9610.00000000000
2, 9610.000000000002, -38440.75499444365, 9610.000000000002, 9610.000000000
002, -0.245005556365137, 9610.000000000002, 9610.000000000002, -38441.63993
824401, 9610.000000000002, 9610.000000000002, 0.6399382439962098, 9610.0000
00000002, 9610.000000000002, -38442.50677559012, 9610.000000000002, 9610.00
0000000002, 1.5067755901124493, 9610.000000000002, 9610.000000000002, -3844
3.31376158765, 9610.000000000002, 9610.000000000002, 2.3137615876423063, 96
10.000000000002, 9610.000000000002, -38443.99592214669, 9610.000000000002, 
9610.000000000002, 2.995922146680746, 9610.000000000002, 9610.000000000002,
 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000
000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, 3.4, 9610.000000000002, 9610.000000000002, -38444.073980791916, 9
610.000000000002, 9610.000000000002, 3.073980791904124, 9610.000000000002, 
9610.000000000002, -38443.52359812821, 9610.000000000002, 9610.000000000002
, 2.523598128204317, 9610.000000000002, 9610.000000000002, -38442.872503459
95, 9610.000000000002, 9610.000000000002, 1.8725034599417647, 9610.00000000
0002, 9610.000000000002, -38442.17311936155, 9610.000000000002, 9610.000000
000002, 1.1731193615376743, 9610.000000000002, 9610.000000000002, -38441.45
912656379, 9610.000000000002, 9610.000000000002, 0.4591265637804165, 9610.0
00000000002, 9610.000000000002, -38440.75499444365, 9610.000000000002, 9610
.000000000002, -0.24500555636513655, 9610.000000000002, 9610.000000000002, 
-38440.07958735315, 9610.000000000002, 9610.000000000002, -0.92041264685666
8, 9610.000000000002, 9610.000000000002, -38439.44791556904, 9610.000000000
002, 9610.000000000002, -1.5520844309642468, 9610.000000000002, 9610.000000
000002, -38438.872113829915, 9610.000000000002, 9610.000000000002, -2.12788
6170096945, 9610.000000000002, 9610.000000000002, -38438.36203976924, 9610.
000000000002, 9610.000000000002, -2.637960230763315, 9610.000000000002, 961
0.000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000002, 
-3.0743373865375214, 9610.000000000002, 9610.000000000002, -38437.569325719
07, 9610.000000000002, 9610.000000000002, -3.43067428093732, 9610.000000000
002, 9610.000000000002, -38437.29792765007, 9610.000000000002, 9610.0000000
00002, -3.7020723499392116, 9610.000000000002, 9610.000000000002, -38437.11
5047209896, 9610.000000000002, 9610.000000000002, -3.884952790112092, 9610.
000000000002, 9610.000000000002, -38437.02302742907, 9610.000000000002, 961
0.000000000002, -3.976972570933175, 9610.000000000002, 9610.000000000002, -
38437.02302742907, 9610.000000000002, 9610.000000000002, -3.976972570933175
, 9610.000000000002, 9610.000000000002, -38437.115047209896, 9610.000000000
002, 9610.000000000002, -3.884952790112093, 9610.000000000002, 9610.0000000
00002, -38437.29792765007, 9610.000000000002, 9610.000000000002, -3.7020723
499392116, 9610.000000000002, 9610.000000000002, -38437.56932571907, 9610.0
00000000002, 9610.000000000002, -3.43067428093732, 9610.000000000002, 9610.
000000000002, -38437.925662613474, 9610.000000000002, 9610.000000000002, -3
.0743373865375214, 9610.000000000002, 9610.000000000002, -38438.36203976924
, 9610.000000000002, 9610.000000000002, -2.637960230763315, 9610.0000000000
02, 9610.000000000002, -38438.872113829915, 9610.000000000002, 9610.0000000
00002, -2.1278861700969443, 9610.000000000002, 9610.000000000002, -38439.44
791556904, 9610.000000000002, 9610.000000000002, -1.5520844309642468, 9610.
000000000002, 9610.000000000002, -38440.07958735315, 9610.000000000002, 961
0.000000000002, -0.9204126468566689, 9610.000000000002, 9610.000000000002, 
-38440.75499444365, 9610.000000000002, 9610.000000000002, -0.24500555636513
7, 9610.000000000002, 9610.000000000002, -38441.45912656379, 9610.000000000
002, 9610.000000000002, 0.4591265637804174, 9610.000000000002, 9610.0000000
00002, -38442.17311936155, 9610.000000000002, 9610.000000000002, 1.17311936
15376743, 9610.000000000002, 9610.000000000002, -38442.87250345995, 9610.00
0000000002, 9610.000000000002, 1.872503459941764, 9610.000000000002, 9610.0
00000000002, -38443.52359812821, 9610.000000000002, 9610.000000000002, 2.52
35981282043163, 9610.000000000002, 9610.000000000002, -38444.073980791916, 
9610.000000000002, 9610.000000000002, 3.0739807919041247, 9610.000000000002
, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.0000000000
02, 3.4, 9610.000000000002, -38444.40000000001, 9610.000000000002, 9610.000
000000002, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, -3
8444.15313257167, 9610.000000000002, 9610.000000000002, 3.1531325716591234,
 9610.000000000002, 9610.000000000002, -38443.73637322309, 9610.00000000000
2, 9610.000000000002, 2.7363732230779934, 9610.000000000002, 9610.000000000
002, -38443.24335298879, 9610.000000000002, 9610.000000000002, 2.2433529887
819708, 9610.000000000002, 9610.000000000002, -38442.71376715609, 9610.0000
00000002, 9610.000000000002, 1.7137671560823644, 9610.000000000002, 9610.00
0000000002, -38442.17311936155, 9610.000000000002, 9610.000000000002, 1.173
1193615376734, 9610.000000000002, 9610.000000000002, -38441.63993824401, 96
10.000000000002, 9610.000000000002, 0.6399382439962102, 9610.000000000002, 
9610.000000000002, -38441.12850821972, 9610.000000000002, 9610.000000000002
, 0.12850821971431436, 9610.000000000002, 9610.000000000002, -38440.6501953
3194, 9610.000000000002, 9610.000000000002, -0.3498046680719562, 9610.00000
0000002, 9610.000000000002, -38440.21418821627, 9610.000000000002, 9610.000
000000002, -0.7858117837388687, 9610.000000000002, 9610.000000000002, -3843
9.82795124467, 9610.000000000002, 9610.000000000002, -1.1720487553441248, 9
610.000000000002, 9610.000000000002, -38439.49751885431, 9610.000000000002,
 9610.000000000002, -1.5024811456988805, 9610.000000000002, 9610.0000000000
02, -38439.227694345325, 9610.000000000002, 9610.000000000002, -1.772305654
6817468, 9610.000000000002, 9610.000000000002, -38439.0221869958, 9610.0000
00000002, 9610.000000000002, -1.9778130042102862, 9610.000000000002, 9610.0
00000000002, -38438.883706738074, 9610.000000000002, 9610.000000000002, -2.
1162932619319412, 9610.000000000002, 9610.000000000002, -38438.814027755725
, 9610.000000000002, 9610.000000000002, -2.1859722442853706, 9610.000000000
002, 9610.000000000002, -38438.814027755725, 9610.000000000002, 9610.000000
000002, -2.1859722442853706, 9610.000000000002, 9610.000000000002, -38438.8
83706738074, 9610.000000000002, 9610.000000000002, -2.1162932619319412, 961
0.000000000002, 9610.000000000002, -38439.0221869958, 9610.000000000002, 96
10.000000000002, -1.9778130042102862, 9610.000000000002, 9610.000000000002,
 -38439.227694345325, 9610.000000000002, 9610.000000000002, -1.772305654681
7468, 9610.000000000002, 9610.000000000002, -38439.49751885431, 9610.000000
000002, 9610.000000000002, -1.5024811456988805, 9610.000000000002, 9610.000
000000002, -38439.82795124467, 9610.000000000002, 9610.000000000002, -1.172
0487553441248, 9610.000000000002, 9610.000000000002, -38440.21418821627, 96
10.000000000002, 9610.000000000002, -0.7858117837388678, 9610.000000000002,
 9610.000000000002, -38440.65019533194, 9610.000000000002, 9610.00000000000
2, -0.3498046680719562, 9610.000000000002, 9610.000000000002, -38441.128508
21972, 9610.000000000002, 9610.000000000002, 0.12850821971431392, 9610.0000
00000002, 9610.000000000002, -38441.63993824401, 9610.000000000002, 9610.00
0000000002, 0.6399382439962098, 9610.000000000002, 9610.000000000002, -3844
2.17311936155, 9610.000000000002, 9610.000000000002, 1.1731193615376743, 96
10.000000000002, 9610.000000000002, -38442.71376715609, 9610.000000000002, 
9610.000000000002, 1.7137671560823644, 9610.000000000002, 9610.000000000002
, -38443.24335298879, 9610.000000000002, 9610.000000000002, 2.2433529887819
7, 9610.000000000002, 9610.000000000002, -38443.73637322309, 9610.000000000
002, 9610.000000000002, 2.736373223077993, 9610.000000000002, 9610.00000000
0002, -38444.15313257167, 9610.000000000002, 9610.000000000002, 3.153132571
659124, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.400
00000001, 9610.000000000002, 3.4, 9610.000000000002, -38444.40000000001, 96
10.000000000002, 9610.000000000002, 9610.000000000002, 3.4, 9610.0000000000
02, 9610.000000000002, -38444.23066486092, 9610.000000000002, 9610.00000000
0002, 3.2306648609131803, 9610.000000000002, 9610.000000000002, -38443.9447
94818315, 9610.000000000002, 9610.000000000002, 2.944794818307675, 9610.000
000000002, 9610.000000000002, -38443.606614725024, 9610.000000000002, 9610.
000000000002, 2.6066147250154335, 9610.000000000002, 9610.000000000002, -38
443.24335298879, 9610.000000000002, 9610.000000000002, 2.24335298878197, 96
10.000000000002, 9610.000000000002, -38442.87250345995, 9610.000000000002, 
9610.000000000002, 1.8725034599417636, 9610.000000000002, 9610.000000000002
, -38442.50677559012, 9610.000000000002, 9610.000000000002, 1.5067755901124
498, 9610.000000000002, 9610.000000000002, -38442.155967567865, 9610.000000
000002, 9610.000000000002, 1.1559675678594368, 9610.000000000002, 9610.0000
00000002, -38441.82787576603, 9610.000000000002, 9610.000000000002, 0.82787
57660180549, 9610.000000000002, 9610.000000000002, -38441.528802996196, 961
0.000000000002, 9610.000000000002, 0.5288029961897376, 9610.000000000002, 9
610.000000000002, -38441.26386933627, 9610.000000000002, 9610.000000000002,
 0.26386933626219733, 9610.000000000002, 9610.000000000002, -38441.03721402
109, 9610.000000000002, 9610.000000000002, 0.03721402107719385, 9610.000000
000002, 9610.000000000002, -38440.85213180484, 9610.000000000002, 9610.0000
00000002, -0.14786819517434147, 9610.000000000002, 9610.000000000002, -3844
0.71116701313, 9610.000000000002, 9610.000000000002, -0.2888329868832229, 9
610.000000000002, 9610.000000000002, -38440.61617848483, 9610.000000000002,
 9610.000000000002, -0.38382151517823804, 9610.000000000002, 9610.000000000
002, -38440.56838319548, 9610.000000000002, 9610.000000000002, -0.431616804
526632, 9610.000000000002, 9610.000000000002, -38440.56838319548, 9610.0000
00000002, 9610.000000000002, -0.431616804526632, 9610.000000000002, 9610.00
0000000002, -38440.61617848483, 9610.000000000002, 9610.000000000002, -0.38
38215151782385, 9610.000000000002, 9610.000000000002, -38440.71116701313, 9
610.000000000002, 9610.000000000002, -0.2888329868832229, 9610.000000000002
, 9610.000000000002, -38440.85213180484, 9610.000000000002, 9610.0000000000
02, -0.14786819517434147, 9610.000000000002, 9610.000000000002, -38441.0372
1402109, 9610.000000000002, 9610.000000000002, 0.03721402107719385, 9610.00
0000000002, 9610.000000000002, -38441.26386933627, 9610.000000000002, 9610.
000000000002, 0.26386933626219733, 9610.000000000002, 9610.000000000002, -3
8441.528802996196, 9610.000000000002, 9610.000000000002, 0.528802996189738,
 9610.000000000002, 9610.000000000002, -38441.82787576603, 9610.00000000000
2, 9610.000000000002, 0.8278757660180549, 9610.000000000002, 9610.000000000
002, -38442.155967567865, 9610.000000000002, 9610.000000000002, 1.155967567
8594363, 9610.000000000002, 9610.000000000002, -38442.50677559012, 9610.000
000000002, 9610.000000000002, 1.5067755901124495, 9610.000000000002, 9610.0
00000000002, -38442.87250345995, 9610.000000000002, 9610.000000000002, 1.87
25034599417643, 9610.000000000002, 9610.000000000002, -38443.24335298879, 9
610.000000000002, 9610.000000000002, 2.24335298878197, 9610.000000000002, 9
610.000000000002, -38443.606614725024, 9610.000000000002, 9610.000000000002
, 2.606614725015433, 9610.000000000002, 9610.000000000002, -38443.944794818
315, 9610.000000000002, 9610.000000000002, 2.9447948183076744, 9610.0000000
00002, 9610.000000000002, -38444.23066486092, 9610.000000000002, 9610.00000
0000002, 3.2306648609131807, 9610.000000000002, 9610.000000000002, 9610.000
000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -
38444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002,
 3.4, 9610.000000000002, 9610.000000000002, -38444.30284388282, 9610.000000
000002, 9610.000000000002, 3.30284388280787, 9610.000000000002, 9610.000000
000002, -38444.13882580889, 9610.000000000002, 9610.000000000002, 3.1388258
088813488, 9610.000000000002, 9610.000000000002, -38443.944794818315, 9610.
000000000002, 9610.000000000002, 2.944794818307675, 9610.000000000002, 9610
.000000000002, -38443.73637322309, 9610.000000000002, 9610.000000000002, 2.
736373223077993, 9610.000000000002, 9610.000000000002, -38443.52359812821, 
9610.000000000002, 9610.000000000002, 2.523598128204316, 9610.000000000002,
 9610.000000000002, -38443.31376158765, 9610.000000000002, 9610.00000000000
2, 2.3137615876423068, 9610.000000000002, 9610.000000000002, -38443.1124853
16786, 9610.000000000002, 9610.000000000002, 2.112485316776409, 9610.000000
000002, 9610.000000000002, -38442.92424248827, 9610.000000000002, 9610.0000
00000002, 1.924242488256717, 9610.000000000002, 9610.000000000002, -38442.7
5264934327, 9610.000000000002, 9610.000000000002, 1.7526493432596986, 9610.
000000000002, 9610.000000000002, -38442.60064352894, 9610.000000000002, 961
0.000000000002, 1.6006435289268393, 9610.000000000002, 9610.000000000002, -
38442.470599933244, 9610.000000000002, 9610.000000000002, 1.470599933232013
9, 9610.000000000002, 9610.000000000002, -38442.36440892298, 9610.000000000
002, 9610.000000000002, 1.3644089229710317, 9610.000000000002, 9610.0000000
00002, -38442.28353030619, 9610.000000000002, 9610.000000000002, 1.28353030
61813383, 9610.000000000002, 9610.000000000002, -38442.22903059256, 9610.00
0000000002, 9610.000000000002, 1.2290305925560006, 9610.000000000002, 9610.
000000000002, -38442.2016080224, 9610.000000000002, 9610.000000000002, 1.20
16080223900794, 9610.000000000002, 9610.000000000002, -38442.2016080224, 96
10.000000000002, 9610.000000000002, 1.2016080223900794, 9610.000000000002, 
9610.000000000002, -38442.22903059256, 9610.000000000002, 9610.000000000002
, 1.2290305925560001, 9610.000000000002, 9610.000000000002, -38442.28353030
619, 9610.000000000002, 9610.000000000002, 1.2835303061813383, 9610.0000000
00002, 9610.000000000002, -38442.36440892298, 9610.000000000002, 9610.00000
0000002, 1.3644089229710317, 9610.000000000002, 9610.000000000002, -38442.4
70599933244, 9610.000000000002, 9610.000000000002, 1.4705999332320139, 9610
.000000000002, 9610.000000000002, -38442.60064352894, 9610.000000000002, 96
10.000000000002, 1.6006435289268393, 9610.000000000002, 9610.000000000002, 
-38442.75264934327, 9610.000000000002, 9610.000000000002, 1.752649343259698
8, 9610.000000000002, 9610.000000000002, -38442.92424248827, 9610.000000000
002, 9610.000000000002, 1.924242488256717, 9610.000000000002, 9610.00000000
0002, -38443.112485316786, 9610.000000000002, 9610.000000000002, 2.11248531
6776409, 9610.000000000002, 9610.000000000002, -38443.31376158765, 9610.000
000000002, 9610.000000000002, 2.3137615876423068, 9610.000000000002, 9610.0
00000000002, -38443.52359812821, 9610.000000000002, 9610.000000000002, 2.52
35981282043163, 9610.000000000002, 9610.000000000002, -38443.73637322309, 9
610.000000000002, 9610.000000000002, 2.736373223077993, 9610.000000000002, 
9610.000000000002, -38443.944794818315, 9610.000000000002, 9610.00000000000
2, 2.9447948183076744, 9610.000000000002, 9610.000000000002, -38444.1388258
0889, 9610.000000000002, 9610.000000000002, 3.1388258088813488, 9610.000000
000002, 9610.000000000002, -38444.30284388282, 9610.000000000002, 9610.0000
00000002, 3.30284388280787, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, -3
8444.40000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 
3.4, 9610.000000000002, 9610.000000000002, -38444.36385817808, 9610.0000000
00002, 9610.000000000002, 3.3638581780710384, 9610.000000000002, 9610.00000
0000002, -38444.30284388282, 9610.000000000002, 9610.000000000002, 3.302843
88280787, 9610.000000000002, 9610.000000000002, -38444.23066486092, 9610.00
0000000002, 9610.000000000002, 3.2306648609131807, 9610.000000000002, 9610.
000000000002, -38444.15313257167, 9610.000000000002, 9610.000000000002, 3.1
53132571659124, 9610.000000000002, 9610.000000000002, -38444.073980791916, 
9610.000000000002, 9610.000000000002, 3.0739807919041247, 9610.000000000002
, 9610.000000000002, -38443.99592214669, 9610.000000000002, 9610.0000000000
02, 2.995922146680746, 9610.000000000002, 9610.000000000002, -38443.9210478
9944, 9610.000000000002, 9610.000000000002, 2.9210478994341718, 9610.000000
000002, 9610.000000000002, -38443.85102205872, 9610.000000000002, 9610.0000
00000002, 2.8510220587111554, 9610.000000000002, 9610.000000000002, -38443.
7871898568, 9610.000000000002, 9610.000000000002, 2.7871898567876405, 9610.
000000000002, 9610.000000000002, -38443.73064409074, 9610.000000000002, 961
0.000000000002, 2.7306440907303076, 9610.000000000002, 9610.000000000002, -
38443.68226821267, 9610.000000000002, 9610.000000000002, 2.682268212664804,
 9610.000000000002, 9610.000000000002, -38443.64276543411, 9610.00000000000
2, 9610.000000000002, 2.642765434103581, 9610.000000000002, 9610.0000000000
02, -38443.61267879984, 9610.000000000002, 9610.000000000002, 2.61267879982
51313, 9610.000000000002, 9610.000000000002, -38443.59240504866, 9610.00000
0000002, 9610.000000000002, 2.5924050486507078, 9610.000000000002, 9610.000
000000002, -38443.582203924154, 9610.000000000002, 9610.000000000002, 2.582
2039241470263, 9610.000000000002, 9610.000000000002, -38443.582203924154, 9
610.000000000002, 9610.000000000002, 2.5822039241470263, 9610.000000000002,
 9610.000000000002, -38443.59240504866, 9610.000000000002, 9610.00000000000
2, 2.5924050486507078, 9610.000000000002, 9610.000000000002, -38443.6126787
9984, 9610.000000000002, 9610.000000000002, 2.6126787998251313, 9610.000000
000002, 9610.000000000002, -38443.64276543411, 9610.000000000002, 9610.0000
00000002, 2.642765434103581, 9610.000000000002, 9610.000000000002, -38443.6
8226821267, 9610.000000000002, 9610.000000000002, 2.682268212664804, 9610.0
00000000002, 9610.000000000002, -38443.73064409074, 9610.000000000002, 9610
.000000000002, 2.7306440907303076, 9610.000000000002, 9610.000000000002, -3
8443.7871898568, 9610.000000000002, 9610.000000000002, 2.7871898567876405, 
9610.000000000002, 9610.000000000002, -38443.85102205872, 9610.000000000002
, 9610.000000000002, 2.8510220587111554, 9610.000000000002, 9610.0000000000
02, -38443.92104789944, 9610.000000000002, 9610.000000000002, 2.92104789943
41718, 9610.000000000002, 9610.000000000002, -38443.99592214669, 9610.00000
0000002, 9610.000000000002, 2.995922146680746, 9610.000000000002, 9610.0000
00000002, -38444.073980791916, 9610.000000000002, 9610.000000000002, 3.0739
80791904125, 9610.000000000002, 9610.000000000002, -38444.15313257167, 9610
.000000000002, 9610.000000000002, 3.153132571659124, 9610.000000000002, 961
0.000000000002, -38444.23066486092, 9610.000000000002, 9610.000000000002, 3
.2306648609131807, 9610.000000000002, 9610.000000000002, -38444.30284388282
, 9610.000000000002, 9610.000000000002, 3.30284388280787, 9610.000000000002
, 9610.000000000002, -38444.36385817808, 9610.000000000002, 9610.0000000000
02, 3.3638581780710384, 9610.000000000002, 9610.000000000002, 9610.00000000
0002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.0
00000000002, -38444.40000000001, 9610.000000000002, 9610.000000000002, 3.4,
 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.4000000000
1, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.00000
0000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 961
0.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3
.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.4000000
0001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.00
0000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 
9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002
, 3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.4000
0000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610
.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.00000000000
2, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000
002, 3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.4
0000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9
610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.00000000
0002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000
000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -3844
4.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002
, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.00000
0000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.000
000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -3
8444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000
002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.00
0000000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 9610.
000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002,
 -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000
000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9610
.000000000002, 9610.000000000002, 9610.000000000002, -38444.40000000001, 96
10.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.0000000000
02, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000
000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4, 9
610.000000000002, 9610.000000000002, 9610.000000000002, -38444.40000000001,
 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.0000000
00002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 9610.
000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 3.4
, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.400000000
01, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, -38444.40000000001, 9610.000000000002, 3.4, 9610.000000000002, 96
10.000000000002, 9610.000000000002, -38444.40000000001, 9610.000000000002, 
3.4, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.400000
00001, 9610.000000000002, 3.4, 9610.000000000002, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, -38444.40000000001, 3.4, 0.0, -38440.000000
00001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 9610.000000
000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9610
.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.0000000000
1, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.00000
0000002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.000
000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9
610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.0000000
0001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.00
0000000002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.
000000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002
, 9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.0000
0000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610
.000000000002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 96
10.000000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000
002, 9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.0
0000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9
610.000000000002, -38440.00000000001, 9610.000000000002, 9610.000000000002,
 9610.000000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000
000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -3844
0.00000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0
, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9610.0000000000
02, 9610.000000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000
000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -3
8440.00000000001, 9610.000000000002, 9610.000000000002, 9610.000000000002, 
0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9610.0000000
00002, 9610.000000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.
000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002,
 -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.00000000000
2, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9610.0000
00000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.00000000001, 96
10.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.0000000000
02, -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.00000000
0002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.00000000001,
 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.0000000
00002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.00000
0000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 961
0.000000000002, 9610.000000000002, 0.0, 9610.000000000002, -38440.000000000
01, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.0, 9610.0000
00000002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 9610.00
0000000002, 0.0, 9610.000000000002, -38440.00000000001, 9610.000000000002, 
9610.000000000002, 9610.000000000002, 0.0, 9610.000000000002, 9610.00000000
0002, -38440.00000000001, 9610.000000000002, 9610.000000000002, 0.014724445
971058452, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 96
10.000000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 
0.014724445971058452, 9610.000000000002, 9610.000000000002, -38440.01472444
598, 9610.000000000002, 9610.000000000002, 0.014724445971058452, 9610.00000
0000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.000
000000002, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -384
40.01472444598, 9610.000000000002, 9610.000000000002, 0.014724445971058452,
 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.00000000000
2, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 9610.0000000
00002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.01472444
5971058452, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.
000000000002, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 9
610.000000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002,
 0.014724445971058452, 9610.000000000002, 9610.000000000002, -38440.0147244
4598, 9610.000000000002, 9610.000000000002, 0.014724445971058452, 9610.0000
00000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.00
0000000002, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -38
440.01472444598, 9610.000000000002, 9610.000000000002, 0.014724445971058452
, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.0000000000
02, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 9610.000000
000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.0147244
45971058452, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610
.000000000002, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 
9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002
, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -38440.014724
44598, 9610.000000000002, 9610.000000000002, 0.014724445971058452, 9610.000
000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.0
00000000002, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -3
8440.01472444598, 9610.000000000002, 9610.000000000002, 0.01472444597105845
2, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.000000000
002, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 9610.00000
0000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.014724
445971058452, 9610.000000000002, 9610.000000000002, -38440.01472444598, 961
0.000000000002, 9610.000000000002, 0.014724445971058452, 9610.000000000002,
 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.00000000000
2, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -38440.01472
444598, 9610.000000000002, 9610.000000000002, 0.014724445971058452, 9610.00
0000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.
000000000002, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -
38440.01472444598, 9610.000000000002, 9610.000000000002, 0.0147244459710584
52, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.00000000
0002, 9610.000000000002, 0.014724445971058452, 9610.000000000002, 9610.0000
00000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.01472
4445971058452, 9610.000000000002, 9610.000000000002, -38440.01472444598, 96
10.000000000002, 9610.000000000002, 0.014724445971058452, 9610.000000000002
, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.0000000000
02, 0.014724445971058452, 9610.000000000002, 9610.000000000002, -38440.0147
2444598, 9610.000000000002, 9610.000000000002, 0.014724445971058452, 9610.0
00000000002, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610
.000000000002, 0.10640430008537624, 9610.000000000002, -38440.106404300095,
 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.106404300085376
24, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.0000000
00002, 9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.0000
00000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.1064
0430008537624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9
610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.000000000002
, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000
002, 0.10640430008537624, 9610.000000000002, 9610.000000000002, -38440.1064
04300095, 9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.0
00000000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 961
0.000000000002, 0.10640430008537624, 9610.000000000002, 9610.000000000002, 
-38440.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537
624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.000000
000002, 9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.000
000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.106
40430008537624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 
9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.00000000000
2, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.00000000
0002, 0.10640430008537624, 9610.000000000002, 9610.000000000002, -38440.106
404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.
000000000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 96
10.000000000002, 0.10640430008537624, 9610.000000000002, 9610.000000000002,
 -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.1064043000853
7624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.00000
0000002, 9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.00
0000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.10
640430008537624, 9610.000000000002, 9610.000000000002, -38440.106404300095,
 9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.0000000000
02, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.0000000
00002, 0.10640430008537624, 9610.000000000002, 9610.000000000002, -38440.10
6404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610
.000000000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9
610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.000000000002
, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.106404300085
37624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.0000
00000002, 9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.0
00000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.1
0640430008537624, 9610.000000000002, 9610.000000000002, -38440.106404300095
, 9610.000000000002, 9610.000000000002, 0.10640430008537624, 9610.000000000
002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000
000002, 0.10640430008537624, 9610.000000000002, 9610.000000000002, -38440.1
06404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537624, 961
0.000000000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 
9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.00000000000
2, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008
537624, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.000
000000002, 9610.000000000002, 0.10640430008537624, 9610.000000000002, 9610.
000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.
10640430008537624, 9610.000000000002, 9610.000000000002, 9610.000000000002,
 -38440.106404300095, 9610.000000000002, 0.3232310379566752, 9610.000000000
002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 9610.0000000
00002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38440.323
23103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.00
0000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.
000000000002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38
440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 
9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002
, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.0000000000
02, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.32323103795
66752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.00000
0000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.000
000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232
310379566752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 961
0.000000000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9
610.000000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002,
 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38440.323231037
96, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.00000000
0002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.000000
000002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38440.32
323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.0
00000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610
.000000000002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -3
8440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752,
 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.00000000000
2, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.000000000
002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379
566752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.0000
00000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.00
0000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.323
2310379566752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 96
10.000000000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 
9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002
, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38440.32323103
796, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.0000000
00002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.00000
0000002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -38440.3
2323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.
000000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 961
0.000000000002, 0.3232310379566752, 9610.000000000002, 9610.000000000002, -
38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566752
, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.0000000000
02, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.00000000
0002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.323231037
9566752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.000
000000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002, 9610.0
00000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.32
32310379566752, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9
610.000000000002, 9610.000000000002, 0.3232310379566752, 9610.000000000002,
 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.00000000000
2, 0.6869837512257033, 9610.000000000002, -38440.686983751235, 9610.0000000
00002, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000
0000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00
0000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -3844
0.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9
610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002
, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000000000
02, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512
257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000
000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0
00000000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6
869837512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235,
 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000000000
2, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000000
0002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869
83751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00
0000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610
.000000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -3
8440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033
, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000
002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000000
00002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837
512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.
000000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 961
0.000000000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 
0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869837512
35, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000000
0002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000
0000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6
86983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610
.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9
610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002,
 -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257
033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000
000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000
00000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869
837512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 96
10.000000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 
9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000000000
2, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869837
51235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000
0000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00
0000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -3844
0.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9
610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002
, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000000000
02, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512
257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000
000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0
00000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 1.1
981336221635341, 9610.000000000002, -38441.19813362217, 9610.000000000002, 
9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.000000000002
, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.0000000000
02, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -38441.198133
62217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.00000
0000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.000
000000002, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -38441
.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 961
0.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9
610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.000000000002,
 -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.19813362216353
41, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.00000000
0002, 9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.000000
000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336
221635341, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.0
00000000002, 9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610
.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.
1981336221635341, 9610.000000000002, 9610.000000000002, -38441.19813362217,
 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.00000000000
2, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000
002, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -38441.19813
362217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.0000
00000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.00
0000000002, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -3844
1.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 96
10.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 
9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.000000000002
, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635
341, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.0000000
00002, 9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.00000
0000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.198133
6221635341, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.
000000000002, 9610.000000000002, 1.1981336221635341, 9610.000000000002, 961
0.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1
.1981336221635341, 9610.000000000002, 9610.000000000002, -38441.19813362217
, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.0000000000
02, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.00000000
0002, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -38441.1981
3362217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9610.000
000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.0
00000000002, 1.1981336221635341, 9610.000000000002, 9610.000000000002, -384
41.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635341, 9
610.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002,
 9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.00000000000
2, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.198133622163
5341, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000
000002, 9610.000000000002, 1.1981336221635341, 9610.000000000002, 9610.0000
00000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.19813
36221635341, 9610.000000000002, 9610.000000000002, 9610.000000000002, -3844
1.19813362217, 9610.000000000002, 1.840555746382307, 9610.000000000002, -38
441.84055574639, 9610.000000000002, 9610.000000000002, 9610.000000000002, 1
.840555746382307, 9610.000000000002, 9610.000000000002, -38441.84055574639,
 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.000000000002
, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.0000000000
02, 1.840555746382307, 9610.000000000002, 9610.000000000002, -38441.8405557
4639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.0000000
00002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.00000
0000002, 1.840555746382307, 9610.000000000002, 9610.000000000002, -38441.84
055574639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.00
0000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.
000000000002, 1.840555746382307, 9610.000000000002, 9610.000000000002, -384
41.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 96
10.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 
9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.000000000002,
 -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.84055574638230
7, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000
002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.00000000
0002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746
382307, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.0000
00000002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.000
000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.8405
55746382307, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610
.000000000002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 961
0.000000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1
.840555746382307, 9610.000000000002, 9610.000000000002, -38441.84055574639,
 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.000000000002
, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.0000000000
02, 1.840555746382307, 9610.000000000002, 9610.000000000002, -38441.8405557
4639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.0000000
00002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.00000
0000002, 1.840555746382307, 9610.000000000002, 9610.000000000002, -38441.84
055574639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 9610.00
0000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.
000000000002, 1.840555746382307, 9610.000000000002, 9610.000000000002, -384
41.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746382307, 96
10.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 
9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.000000000002,
 -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.84055574638230
7, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000
002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.00000000
0002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746
382307, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.0000
00000002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 9610.000
000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.8405
55746382307, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610
.000000000002, 9610.000000000002, 1.840555746382307, 9610.000000000002, 961
0.000000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1
.840555746382307, 9610.000000000002, 9610.000000000002, 9610.000000000002, 
-38441.84055574639, 9610.000000000002, 2.585848303653401, 9610.000000000002
, -38442.58584830366, 9610.000000000002, 9610.000000000002, 9610.0000000000
02, 2.585848303653401, 9610.000000000002, 9610.000000000002, -38442.5858483
0366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.0000000
00002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.00000
0000002, 2.585848303653401, 9610.000000000002, 9610.000000000002, -38442.58
584830366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.00
0000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.
000000000002, 2.585848303653401, 9610.000000000002, 9610.000000000002, -384
42.58584830366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 96
10.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 
9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.000000000002,
 -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.58584830365340
1, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000
002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.00000000
0002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.585848303
653401, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.0000
00000002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.000
000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.5858
48303653401, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610
.000000000002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 961
0.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2
.585848303653401, 9610.000000000002, 9610.000000000002, -38442.58584830366,
 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.000000000002
, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.0000000000
02, 2.585848303653401, 9610.000000000002, 9610.000000000002, -38442.5858483
0366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.0000000
00002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.00000
0000002, 2.585848303653401, 9610.000000000002, 9610.000000000002, -38442.58
584830366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.00
0000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.
000000000002, 2.585848303653401, 9610.000000000002, 9610.000000000002, -384
42.58584830366, 9610.000000000002, 9610.000000000002, 2.585848303653401, 96
10.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 
9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.000000000002,
 -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.58584830365340
1, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000
002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.00000000
0002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.585848303
653401, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.0000
00000002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 9610.000
000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.5858
48303653401, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610
.000000000002, 9610.000000000002, 2.585848303653401, 9610.000000000002, 961
0.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2
.585848303653401, 9610.000000000002, 9610.000000000002, -38442.58584830366,
 9610.000000000002, 9610.000000000002, 2.585848303653401, 9610.000000000002
, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.0000000000
02, 2.585848303653401, 9610.000000000002, 9610.000000000002, 9610.000000000
002, -38442.58584830366, 9610.000000000002, 3.3972590768330555, 9610.000000
000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.
39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610
.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 96
10.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, 
-38443.39725907684, 9610.000000000002, 9610.000000000002, 3.397259076833055
5, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000
002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.0000000
00002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.39725907
68330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.00
0000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.
000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.3
972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 
9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002
, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.0000000000
02, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.397259
07684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.00000
0000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.000
000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443
.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 961
0.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9
610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002,
 -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.39725907683305
55, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.00000000
0002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000
000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590
768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.0
00000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610
.000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.
3972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684,
 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.00000000000
2, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000
002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725
907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.0000
00000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.00
0000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -3844
3.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 96
10.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 
9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002
, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330
555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.0000000
00002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.00000
0000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.397259
0768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.
000000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 961
0.000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3
.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684
, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.0000000000
02, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.00000000
0002, 4.233219318895422, 9610.000000000002, -38444.2332193189, 9610.0000000
00002, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000
000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00000
0000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.23
32193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000
000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00
0000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444
.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.
000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610
.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38
444.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 96
10.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9
610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, 
-38444.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422,
 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002
, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.00000000000
2, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.2332193188954
22, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000
002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.00000000
0002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.2332193188
95422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000
000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.00000
0000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.2332193
18895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000
000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.00
0000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.2332
19318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.
000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610
.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.2
33219318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 96
10.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9
610.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 
4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189,
 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002
, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00000000000
2, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.23321931
89, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000
002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00000000
0002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.23321
93189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000
000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00000
0000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.23
32193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000
000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.00
0000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444
.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.
000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610
.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, 961
0.000000000002, -38444.2332193189, 9610.000000000002, 5.050484968073051, 96
10.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 
9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002,
 -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.05048496807305
1, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000
002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.00000000
0002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968
073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.0000
00000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000
000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.0504
84968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610
.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 961
0.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5
.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808,
 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002
, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.0000000000
02, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.0504849
6808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.0000000
00002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.00000
0000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05
048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.00
0000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.
000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -384
45.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 96
10.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 
9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002,
 -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.05048496807305
1, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000
002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.00000000
0002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968
073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.0000
00000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000
000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.0504
84968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610
.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 961
0.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5
.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808,
 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002
, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.0000000000
02, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.0504849
6808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.0000000
00002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.00000
0000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05
048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.00
0000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.
000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -384
45.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 96
10.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 
9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002,
 9610.000000000002, -38445.05048496808, 9610.000000000002, 5.80688521110483
2, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000
002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.00000000
0002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211
104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.0000
00000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000
000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.8068
85211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610
.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 961
0.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5
.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111,
 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002
, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.0000000000
02, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.8068852
1111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.0000000
00002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.00000
0000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80
688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.00
0000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.
000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -384
45.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 96
10.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 
9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002,
 -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.80688521110483
2, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000
002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.00000000
0002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211
104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.0000
00000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000
000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.8068
85211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610
.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 961
0.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5
.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111,
 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002
, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.0000000000
02, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.8068852
1111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.0000000
00002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.00000
0000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80
688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.00
0000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.
000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -384
45.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 96
10.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 
9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002,
 -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.80688521110483
2, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000
002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.00000000
0002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 6.463678394
591357, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.00000
0000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000
00000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678
394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00
0000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0
00000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463
678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610
.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 961
0.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.
463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9
610.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 
9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002,
 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946
, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000000000
2, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000000000
02, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783
946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000000
0002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000000
00002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636
783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000
0000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000
00000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4
636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00
0000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0
00000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -3844
6.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610
.000000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 961
0.000000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -3
8446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9
610.000000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 
9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002,
 -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357
, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000000000
2, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000000000
02, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591
357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000000
0002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000000
00002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394
591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000
0000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000
00000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678
394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00
0000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0
00000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463
678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610
.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 961
0.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.
463678394591357, 9610.000000000002, 9610.000000000002, 9610.000000000002, -
38446.4636783946, 9610.000000000002, 6.987515284457731, 9610.000000000002, 
-38446.98751528446, 9610.000000000002, 9610.000000000002, 9610.000000000002
, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.987515284
46, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000
002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.0000000
00002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.9875
1528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.0000
00000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.00
0000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446
.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610
.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 96
10.000000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -
38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731,
 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.00000000000
2, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.0000000000
02, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.98751528445
7731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000
000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.00000
0000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515
284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.0
00000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.
000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.9
87515284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9
610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 
9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002
, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.987515284
46, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000
002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.0000000
00002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.9875
1528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.0000
00000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.00
0000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446
.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610
.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 96
10.000000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -
38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731,
 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.00000000000
2, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.0000000000
02, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.98751528445
7731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000
000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.00000
0000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515
284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.0
00000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.
000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.9
87515284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9
610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 
9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002
, 6.987515284457731, 9610.000000000002, 9610.000000000002, 9610.00000000000
2, -38446.98751528446, 9610.000000000002, 7.352009673523821, 9610.000000000
002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 9610.0000000
00002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.3520
0967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.0000
00000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.00
0000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447
.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610
.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 96
10.000000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -
38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821,
 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.00000000000
2, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.0000000000
02, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.35200967352
3821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000
000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.00000
0000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009
673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.0
00000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.
000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.3
52009673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9
610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 
9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002
, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.352009673
53, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000
002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.0000000
00002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.3520
0967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.0000
00000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.00
0000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447
.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610
.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 96
10.000000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -
38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821,
 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.00000000000
2, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.0000000000
02, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.35200967352
3821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000
000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.00000
0000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009
673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.0
00000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.
000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.3
52009673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9
610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 
9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002
, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.352009673
53, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000
002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.0000000
00002, 7.352009673523821, 9610.000000000002, 9610.000000000002, 9610.000000
000002, -38447.35200967353, 9610.000000000002, 7.538916337181928, 9610.0000
00000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 9610.00
0000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447
.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610
.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 96
10.000000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -
38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928,
 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.00000000000
2, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.0000000000
02, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.53891633718
1928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000
000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.00000
0000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916
337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.0
00000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.
000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.5
38916337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9
610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 
9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002
, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.538916337
19, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000
002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.0000000
00002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.5389
1633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.0000
00000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.00
0000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447
.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610
.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 96
10.000000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -
38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928,
 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.00000000000
2, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.0000000000
02, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.53891633718
1928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000
000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.00000
0000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916
337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.0
00000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.
000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.5
38916337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9
610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 
9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002
, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.538916337
19, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000
002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.0000000
00002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.5389
1633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.0000
00000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.00
0000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, 9610.0
00000000002, -38447.53891633719, 9610.000000000002, 7.538916337181928, 9610
.000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 96
10.000000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -
38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928,
 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.00000000000
2, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.0000000000
02, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.53891633718
1928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000
000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.00000
0000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916
337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.0
00000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.
000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.5
38916337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9
610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 
9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002
, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.538916337
19, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000
002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.0000000
00002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.5389
1633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.0000
00000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.00
0000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447
.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610
.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 96
10.000000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -
38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928,
 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.00000000000
2, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.0000000000
02, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.53891633718
1928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000
000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.00000
0000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.538916
337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9610.0
00000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 9610.
000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002, 7.5
38916337181928, 9610.000000000002, 9610.000000000002, -38447.53891633719, 9
610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000002, 
9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.000000000002
, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.538916337
19, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.000000000
002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.0000000
00002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447.5389
1633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610.0000
00000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 9610.00
0000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, -38447
.53891633719, 9610.000000000002, 9610.000000000002, 7.538916337181928, 9610
.000000000002, 9610.000000000002, -38447.53891633719, 9610.000000000002, 96
10.000000000002, 7.538916337181928, 9610.000000000002, 9610.000000000002, 9
610.000000000002, -38447.53891633719, 9610.000000000002, 7.352009673523821,
 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.00000000000
2, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.0000000000
02, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.35200967352
3821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000
000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.00000
0000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009
673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.0
00000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.
000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.3
52009673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9
610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 
9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002
, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.352009673
53, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000
002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.0000000
00002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.3520
0967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.0000
00000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.00
0000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447
.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610
.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 96
10.000000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -
38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821,
 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.00000000000
2, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.0000000000
02, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.35200967352
3821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000
000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.00000
0000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009
673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.0
00000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.
000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002, 7.3
52009673523821, 9610.000000000002, 9610.000000000002, -38447.35200967353, 9
610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000002, 
9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.000000000002
, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.352009673
53, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.000000000
002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.0000000
00002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447.3520
0967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610.0000
00000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 9610.00
0000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -38447
.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821, 9610
.000000000002, 9610.000000000002, -38447.35200967353, 9610.000000000002, 96
10.000000000002, 7.352009673523821, 9610.000000000002, 9610.000000000002, -
38447.35200967353, 9610.000000000002, 9610.000000000002, 7.352009673523821,
 9610.000000000002, 9610.000000000002, -38447.35200967353, 9610.00000000000
2, 9610.000000000002, 7.352009673523821, 9610.000000000002, 9610.0000000000
02, 9610.000000000002, -38447.35200967353, 9610.000000000002, 6.98751528445
7731, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.000000
000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.00000
0000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515
284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.0
00000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.
000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.9
87515284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9
610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 
9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002
, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.987515284
46, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000
002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.0000000
00002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.9875
1528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.0000
00000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.00
0000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446
.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610
.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 96
10.000000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -
38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731,
 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.00000000000
2, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.0000000000
02, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.98751528445
7731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000
000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.00000
0000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515
284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.0
00000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.
000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.9
87515284457731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9
610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 
9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.000000000002
, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.987515284
46, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.000000000
002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.0000000
00002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446.9875
1528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610.0000
00000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 9610.00
0000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -38446
.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731, 9610
.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 96
10.000000000002, 6.987515284457731, 9610.000000000002, 9610.000000000002, -
38446.98751528446, 9610.000000000002, 9610.000000000002, 6.987515284457731,
 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.00000000000
2, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.0000000000
02, -38446.98751528446, 9610.000000000002, 9610.000000000002, 6.98751528445
7731, 9610.000000000002, 9610.000000000002, -38446.98751528446, 9610.000000
000002, 9610.000000000002, 6.987515284457731, 9610.000000000002, 9610.00000
0000002, 9610.000000000002, -38446.98751528446, 9610.000000000002, 6.463678
394591357, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.00
0000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0
00000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463
678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610
.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 961
0.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.
463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9
610.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 
9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002,
 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946
, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000000000
2, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000000000
02, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783
946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000000
0002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000000
00002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636
783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00000
0000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0000
00000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -38446.4
636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610.00
0000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.0
00000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -3844
6.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9610
.000000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 961
0.000000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002, -3
8446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357, 9
610.000000000002, 9610.000000000002, -38446.4636783946, 9610.000000000002, 
9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.000000000002,
 -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591357
, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000000000
2, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000000000
02, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394591
357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000000
0002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000000
00002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678394
591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00000
0000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0000
00000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463678
394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610.00
0000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 9610.0
00000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.463
678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9610
.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 961
0.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002, 6.
463678394591357, 9610.000000000002, 9610.000000000002, -38446.4636783946, 9
610.000000000002, 9610.000000000002, 6.463678394591357, 9610.000000000002, 
9610.000000000002, -38446.4636783946, 9610.000000000002, 9610.000000000002,
 6.463678394591357, 9610.000000000002, 9610.000000000002, 9610.000000000002
, -38446.4636783946, 9610.000000000002, 5.806885211104832, 9610.00000000000
2, -38445.80688521111, 9610.000000000002, 9610.000000000002, 9610.000000000
002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.806885
21111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000
000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.0000
00000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.8
0688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.0
00000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610
.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38
445.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9
610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002,
 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002
, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.8068852111048
32, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.00000000
0002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.0000000
00002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.80688521
1104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000
000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.00
0000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.806
885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 961
0.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 96
10.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 
5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111
, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.00000000000
2, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000
002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.806885
21111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000
000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.0000
00000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.8
0688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.0
00000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610
.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002, -38
445.80688521111, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9
610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000000000002,
 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.000000000002
, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.8068852111048
32, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.00000000
0002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.0000000
00002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.80688521
1104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 9610.000
000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 9610.00
0000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 5.806
885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111, 961
0.000000000002, 9610.000000000002, 5.806885211104832, 9610.000000000002, 96
10.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000002, 
5.806885211104832, 9610.000000000002, 9610.000000000002, -38445.80688521111
, 9610.000000000002, 9610.000000000002, 5.806885211104832, 9610.00000000000
2, 9610.000000000002, -38445.80688521111, 9610.000000000002, 9610.000000000
002, 5.806885211104832, 9610.000000000002, 9610.000000000002, 9610.00000000
0002, -38445.80688521111, 9610.000000000002, 5.050484968073051, 9610.000000
000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 9610.0000
00000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.0
5048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.0
00000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610
.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38
445.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9
610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002,
 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002
, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.0504849680730
51, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.00000000
0002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.0000000
00002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.05048496
8073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000
000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.00
0000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.050
484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 961
0.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 96
10.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 
5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808
, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.00000000000
2, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000
002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.050484
96808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000
000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.0000
00000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.0
5048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.0
00000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610
.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38
445.05048496808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9
610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000000000002,
 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.000000000002
, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.0504849680730
51, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.00000000
0002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.0000000
00002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.05048496
8073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 9610.000
000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 9610.00
0000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 5.050
484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808, 961
0.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000000002, 96
10.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000002, 
5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.05048496808
, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.00000000000
2, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.000000000
002, 5.050484968073051, 9610.000000000002, 9610.000000000002, -38445.050484
96808, 9610.000000000002, 9610.000000000002, 5.050484968073051, 9610.000000
000002, 9610.000000000002, -38445.05048496808, 9610.000000000002, 9610.0000
00000002, 5.050484968073051, 9610.000000000002, 9610.000000000002, 9610.000
000000002, -38445.05048496808, 9610.000000000002, 4.233219318895422, 9610.0
00000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 9610.
000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -384
44.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 961
0.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 96
10.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -
38444.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 
9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002,
 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002
, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23321931889542
2, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.0000000000
02, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000
002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23321931889
5422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.0000000
00002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000
000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23321931
8895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.0000
00000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000
000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23321
9318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.0
00000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.
000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23
3219318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 961
0.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002, 96
10.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4
.233219318895422, 9610.000000000002, 9610.000000000002, -38444.2332193189, 
9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.000000000002,
 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000002
, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.233219318
9, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.0000000000
02, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000000
002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.233219
3189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.0000000
00002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.000000
000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.233
2193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.0000
00000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.000
000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -38444.
2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 9610.0
00000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 9610.
000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -384
44.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 961
0.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002, 96
10.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002, -
38444.2332193189, 9610.000000000002, 9610.000000000002, 4.233219318895422, 
9610.000000000002, 9610.000000000002, -38444.2332193189, 9610.000000000002,
 9610.000000000002, 4.233219318895422, 9610.000000000002, 9610.000000000002
, -38444.2332193189, 9610.000000000002, 9610.000000000002, 4.23321931889542
2, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38444.23321931
89, 9610.000000000002, 3.3972590768330555, 9610.000000000002, -38443.397259
07684, 9610.000000000002, 9610.000000000002, 9610.000000000002, 3.397259076
8330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.000
000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.0
00000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.39
72590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9
610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002,
 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.00000000000
2, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.3972590
7684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.000000
000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.0000
00000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.
39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610
.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 96
10.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, 
-38443.39725907684, 9610.000000000002, 9610.000000000002, 3.397259076833055
5, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000
002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.0000000
00002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.39725907
68330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.00
0000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.
000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.3
972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 
9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002
, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.0000000000
02, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.397259
07684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.00000
0000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.000
000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443
.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 961
0.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9
610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002,
 -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.39725907683305
55, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.00000000
0002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000
000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590
768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684, 9610.0
00000000002, 9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610
.000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000002, 3.
3972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725907684,
 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.00000000000
2, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.000000000
002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -38443.39725
907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 9610.0000
00000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 9610.00
0000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002, -3844
3.39725907684, 9610.000000000002, 9610.000000000002, 3.3972590768330555, 96
10.000000000002, 9610.000000000002, -38443.39725907684, 9610.000000000002, 
9610.000000000002, 3.3972590768330555, 9610.000000000002, 9610.000000000002
, 9610.000000000002, -38443.39725907684, 9610.000000000002, 2.5858483036534
023, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.0000000
00002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9610.00000
0000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.585848
3036534023, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.
000000000002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 961
0.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2
.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.58584830366
, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.0000000000
02, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.00000000
0002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.5858
4830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.000
000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.0
00000000002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -384
42.58584830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9
610.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002,
 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9610.00000000000
2, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.585848303653
4023, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000
000002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9610.0000
00000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.58584
83036534023, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610
.000000000002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 96
10.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 
2.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.5858483036
6, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.000000000
002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.0000000
00002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.585
84830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.00
0000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.
000000000002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -38
442.58584830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 
9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002
, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9610.0000000000
02, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.58584830365
34023, 9610.000000000002, 9610.000000000002, -38442.58584830366, 9610.00000
0000002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9610.000
000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002, 2.5858
483036534023, 9610.000000000002, 9610.000000000002, -38442.58584830366, 961
0.000000000002, 9610.000000000002, 2.5858483036534023, 9610.000000000002, 9
610.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000000002,
 2.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.585848303
66, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.00000000
0002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610.000000
000002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -38442.58
584830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023, 9610.0
00000000002, 9610.000000000002, -38442.58584830366, 9610.000000000002, 9610
.000000000002, 2.5858483036534023, 9610.000000000002, 9610.000000000002, -3
8442.58584830366, 9610.000000000002, 9610.000000000002, 2.5858483036534023,
 9610.000000000002, 9610.000000000002, 9610.000000000002, -38442.5858483036
6, 9610.000000000002, 1.8405557463823077, 9610.000000000002, -38441.8405557
4639, 9610.000000000002, 9610.000000000002, 9610.000000000002, 1.8405557463
823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.0000
00000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.00
0000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.840
5557463823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 96
10.000000000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 
9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002
, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441.84055574
639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.0000000
00002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.00000
0000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441.8
4055574639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.
000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 961
0.000000000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -
38441.84055574639, 9610.000000000002, 9610.000000000002, 1.8405557463823077
, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.0000000000
02, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.00000000
0002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746
3823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.000
000000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.0
00000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.84
05557463823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9
610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002,
 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.00000000000
2, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441.8405557
4639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.000000
000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.0000
00000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441.
84055574639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610
.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 96
10.000000000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002, 
-38441.84055574639, 9610.000000000002, 9610.000000000002, 1.840555746382307
7, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000
002, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.0000000
00002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.84055574
63823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 9610.00
0000000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.
000000000002, -38441.84055574639, 9610.000000000002, 9610.000000000002, 1.8
405557463823077, 9610.000000000002, 9610.000000000002, -38441.84055574639, 
9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.000000000002
, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.0000000000
02, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441.840555
74639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 9610.00000
0000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9610.000
000000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002, -38441
.84055574639, 9610.000000000002, 9610.000000000002, 1.8405557463823077, 961
0.000000000002, 9610.000000000002, -38441.84055574639, 9610.000000000002, 9
610.000000000002, 1.8405557463823077, 9610.000000000002, 9610.000000000002,
 9610.000000000002, -38441.84055574639, 9610.000000000002, 1.19813362216353
37, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.00000000
0002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610.000000
000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336
221635337, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.0
00000000002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610
.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.
1981336221635337, 9610.000000000002, 9610.000000000002, -38441.19813362217,
 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.00000000000
2, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000
002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -38441.19813
362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.0000
00000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.00
0000000002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -3844
1.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 96
10.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 
9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610.000000000002
, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635
337, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.0000000
00002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610.00000
0000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.198133
6221635337, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.
000000000002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 961
0.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1
.1981336221635337, 9610.000000000002, 9610.000000000002, -38441.19813362217
, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.0000000000
02, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.00000000
0002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -38441.1981
3362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.000
000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.0
00000000002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -384
41.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9
610.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002,
 9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610.00000000000
2, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.198133622163
5337, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610.000000
000002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 9610.0000
00000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 1.19813
36221635337, 9610.000000000002, 9610.000000000002, -38441.19813362217, 9610
.000000000002, 9610.000000000002, 1.1981336221635337, 9610.000000000002, 96
10.000000000002, -38441.19813362217, 9610.000000000002, 9610.000000000002, 
1.1981336221635337, 9610.000000000002, 9610.000000000002, -38441.1981336221
7, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.000000000
002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.0000000
00002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -38441.198
13362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 9610.00
0000000002, 9610.000000000002, -38441.19813362217, 9610.000000000002, 9610.
000000000002, 1.1981336221635337, 9610.000000000002, 9610.000000000002, -38
441.19813362217, 9610.000000000002, 9610.000000000002, 1.1981336221635337, 
9610.000000000002, 9610.000000000002, 9610.000000000002, -38441.19813362217
, 9610.000000000002, 0.6869837512257033, 9610.000000000002, -38440.68698375
1235, 9610.000000000002, 9610.000000000002, 9610.000000000002, 0.6869837512
257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000
000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0
00000000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6
869837512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235,
 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000000000
2, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000000
0002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869
83751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00
0000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610
.000000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -3
8440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033
, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000
002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000000
00002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837
512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.
000000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 961
0.000000000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 
0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869837512
35, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000000
0002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000
0000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6
86983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610
.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9
610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002,
 -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257
033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000
000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000
00000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869
837512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 96
10.000000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 
9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000000000
2, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869837
51235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000
0000002, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00
0000000002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -3844
0.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9
610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000000000002
, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0000000000
02, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6869837512
257033, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610.000
000000002, 9610.000000000002, 0.6869837512257033, 9610.000000000002, 9610.0
00000000002, -38440.686983751235, 9610.000000000002, 9610.000000000002, 0.6
869837512257033, 9610.000000000002, 9610.000000000002, -38440.686983751235,
 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00000000000
2, 9610.000000000002, -38440.686983751235, 9610.000000000002, 9610.00000000
0002, 0.6869837512257033, 9610.000000000002, 9610.000000000002, -38440.6869
83751235, 9610.000000000002, 9610.000000000002, 0.6869837512257033, 9610.00
0000000002, 9610.000000000002, 9610.000000000002, -38440.686983751235, 9610
.000000000002, 0.3232310379566754, 9610.000000000002, -38440.32323103796, 9
610.000000000002, 9610.000000000002, 9610.000000000002, 0.3232310379566754,
 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.00000000000
2, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.000000000
002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379
566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.0000
00000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.00
0000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.323
2310379566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 96
10.000000000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 
9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002
, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -38440.32323103
796, 9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.0000000
00002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.00000
0000002, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -38440.3
2323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.
000000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 961
0.000000000002, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -
38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566754
, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.0000000000
02, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.00000000
0002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.323231037
9566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.000
000000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.0
00000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.32
32310379566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9
610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002,
 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.00000000000
2, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -38440.3232310
3796, 9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.000000
000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.0000
00000002, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -38440.
32323103796, 9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610
.000000000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 96
10.000000000002, 0.3232310379566754, 9610.000000000002, 9610.000000000002, 
-38440.32323103796, 9610.000000000002, 9610.000000000002, 0.323231037956675
4, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.000000000
002, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.0000000
00002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.32323103
79566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 9610.00
0000000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002, 9610.
000000000002, -38440.32323103796, 9610.000000000002, 9610.000000000002, 0.3
232310379566754, 9610.000000000002, 9610.000000000002, -38440.32323103796, 
9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.000000000002
, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.0000000000
02, 0.3232310379566754, 9610.000000000002, 9610.000000000002, -38440.323231
03796, 9610.000000000002, 9610.000000000002, 0.3232310379566754, 9610.00000
0000002, 9610.000000000002, -38440.32323103796, 9610.000000000002, 9610.000
000000002, 0.3232310379566754, 9610.000000000002, 9610.000000000002, 9610.0
00000000002, -38440.32323103796, 9610.000000000002, 0.10640430008537645, 96
10.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002,
 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610.0000000000
02, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.1064043000
8537645, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.00
0000000002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610
.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0
.10640430008537645, 9610.000000000002, 9610.000000000002, -38440.1064043000
95, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 9610.0000000
00002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.0000
00000002, 0.10640430008537645, 9610.000000000002, 9610.000000000002, -38440
.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 9
610.000000000002, 9610.000000000002, -38440.106404300095, 9610.000000000002
, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610.000000000
002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.106404300
08537645, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.0
00000000002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 961
0.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 
0.10640430008537645, 9610.000000000002, 9610.000000000002, -38440.106404300
095, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 9610.000000
000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.000
000000002, 0.10640430008537645, 9610.000000000002, 9610.000000000002, -3844
0.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 
9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.00000000000
2, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610.00000000
0002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430
008537645, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.
000000000002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 96
10.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002,
 0.10640430008537645, 9610.000000000002, 9610.000000000002, -38440.10640430
0095, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 9610.00000
0000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.00
0000000002, 0.10640430008537645, 9610.000000000002, 9610.000000000002, -384
40.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537645,
 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.0000000000
02, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610.0000000
00002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.1064043
0008537645, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610
.000000000002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9
610.000000000002, -38440.106404300095, 9610.000000000002, 9610.000000000002
, 0.10640430008537645, 9610.000000000002, 9610.000000000002, -38440.1064043
00095, 9610.000000000002, 9610.000000000002, 0.10640430008537645, 9610.0000
00000002, 9610.000000000002, -38440.106404300095, 9610.000000000002, 9610.0
00000000002, 0.10640430008537645, 9610.000000000002, 9610.000000000002, -38
440.106404300095, 9610.000000000002, 9610.000000000002, 0.10640430008537645
, 9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.000000000
002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 9610.000000
000002, -38440.106404300095, 9610.000000000002, 9610.000000000002, 0.106404
30008537645, 9610.000000000002, 9610.000000000002, -38440.106404300095, 961
0.000000000002, 9610.000000000002, 0.10640430008537645, 9610.000000000002, 
9610.000000000002, 9610.000000000002, -38440.106404300095, 9610.00000000000
2, 0.014724445971058422, 9610.000000000002, -38440.01472444598, 9610.000000
000002, 9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.00
0000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.
000000000002, 0.014724445971058422, 9610.000000000002, 9610.000000000002, -
38440.01472444598, 9610.000000000002, 9610.000000000002, 0.0147244459710584
22, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.00000000
0002, 9610.000000000002, 0.014724445971058422, 9610.000000000002, 9610.0000
00000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.01472
4445971058422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 96
10.000000000002, 9610.000000000002, 0.014724445971058422, 9610.000000000002
, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.0000000000
02, 0.014724445971058422, 9610.000000000002, 9610.000000000002, -38440.0147
2444598, 9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.0
00000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610
.000000000002, 0.014724445971058422, 9610.000000000002, 9610.000000000002, 
-38440.01472444598, 9610.000000000002, 9610.000000000002, 0.014724445971058
422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.0000000
00002, 9610.000000000002, 0.014724445971058422, 9610.000000000002, 9610.000
000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.0147
24445971058422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9
610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.00000000000
2, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.000000000
002, 0.014724445971058422, 9610.000000000002, 9610.000000000002, -38440.014
72444598, 9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.
000000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 961
0.000000000002, 0.014724445971058422, 9610.000000000002, 9610.000000000002,
 -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.01472444597105
8422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.000000
000002, 9610.000000000002, 0.014724445971058422, 9610.000000000002, 9610.00
0000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.014
724445971058422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 
9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.0000000000
02, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.00000000
0002, 0.014724445971058422, 9610.000000000002, 9610.000000000002, -38440.01
472444598, 9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610
.000000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 96
10.000000000002, 0.014724445971058422, 9610.000000000002, 9610.000000000002
, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.0147244459710
58422, 9610.000000000002, 9610.000000000002, -38440.01472444598, 9610.00000
0000002, 9610.000000000002, 0.014724445971058422, 9610.000000000002, 9610.0
00000000002, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.01
4724445971058422, 9610.000000000002, 9610.000000000002, -38440.01472444598,
 9610.000000000002, 9610.000000000002, 0.014724445971058422, 9610.000000000
002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9610.0000000
00002, 0.014724445971058422, 9610.000000000002, 9610.000000000002, -38440.0
1472444598, 9610.000000000002, 9610.000000000002, 0.014724445971058422, 961
0.000000000002, 9610.000000000002, -38440.01472444598, 9610.000000000002, 9
610.000000000002, 0.014724445971058422, 9610.000000000002, 9610.00000000000
2, -38440.01472444598, 9610.000000000002, 9610.000000000002, 0.014724445971
058422, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.014
72444598, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, -38
440.00000000001, 9610.000000000002, 9610.000000000002, 0.0, 9610.0000000000
02, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.00000000
0002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.
00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 
9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.0000000
00002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.00000
0000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -384
40.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.00000000000
2, 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.0000
00000002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.00
0000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -
38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.00000000
0002, 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.0
00000000002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610
.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002
, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.00000
0000002, 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 961
0.000000000002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9
610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000
002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.00
0000000002, 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 
9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.00000000001
, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000
000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610
.000000000002, 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.
0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.00000000
001, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000
000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9
610.000000000002, 9610.000000000002, -38440.00000000001, 9610.000000000002,
 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.00000
000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.
000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000002
, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.0000000000
02, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440.00
000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002, 96
10.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000000
002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.0000000
00002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, -38440
.00000000001, 9610.000000000002, 0.0, 9610.000000000002, 9610.000000000002,
 9610.000000000002, -38440.00000000001, 9610.000000000002, 0.0, 9610.000000
000002, 9610.000000000002, 9610.000000000002, -38440.00000000001, 9610.0000
00000002, 0.0, 9610.000000000002, 9610.000000000002, 9610.000000000002, 961
0.000000000002, -38440.00000000001], 2048, 2048)), 0x0000000000007e09).
[Info] Solver RobustMultiNewton (GMRES) successfully solved the problem (no
rm = 1.1664069101655303e-8).
[Info] Solver Newton Raphson successfully solved the problem (norm = 2.6476
053818757766e-9).
[Info] Solver Newton Krylov successfully solved the problem (norm = 1.16640
69101655303e-8).
[Info] Solver Trust Region successfully solved the problem (norm = 2.647605
3818757766e-9).
[Info] Solver TR Krylov successfully solved the problem (norm = 1.166406910
1655303e-8).
[Info] Solver NR [NLsolve.jl] successfully solved the problem (norm = 2.629
767216137896e-9).
[Info] Solver TR [NLsolve.jl] successfully solved the problem (norm = 2.629
767216137896e-9).
[Info] Solver NR [Sundials] successfully solved the problem (norm = 5.94616
2345262808e-6).
[Warn] Solver Newton Krylov [Sundials] returned retcode Failure with an res
idual norm = 34.06162309153361.
[Info] Solver Mod. Powell [MINPACK] successfully solved the problem (norm =
 1.962972484440553e-6).
```





Plotting the Work-Precision Diagram.

```julia
fig = begin
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:seaborn_bright, length(successful_solvers); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    with_theme(plot_theme) do 
        fig = Figure(; size = (WIDTH, HEIGHT))
        # `textbf` doesn't work
        ax = Axis(fig[1, 1], ylabel = L"Time $\mathbf{(s)}$",
            xlabelsize = 22, ylabelsize = 22,
            xlabel = L"Error: $\mathbf{||f(u^\ast)||_\infty}$",
            xscale = log10, yscale = log10, xtickwidth = STROKEWIDTH,
            ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
            xticklabelsize = 20, yticklabelsize = 20)

        idxs = sortperm(median.(getfield.(wp_set.wps, :times)))

        ls, scs = [], []

        for (i, (wp, solver)) in enumerate(zip(wp_set.wps[idxs], successful_solvers[idxs]))
            (; name, times, errors) = wp
            errors = [err.l∞ for err in errors]
            l = lines!(ax, errors, times; linestyle = LINESTYLES[solver.pkg], label = name,
                linewidth = 5, color = colors[i])
            sc = scatter!(ax, errors, times; label = name, markersize = 16, strokewidth = 2,
                color = colors[i])
            push!(ls, l)
            push!(scs, sc)
        end

        xlims!(ax; high=1)
        ylims!(ax; low=5e-3)

        axislegend(ax, [[l, sc] for (l, sc) in zip(ls, scs)],
            [solver.name for solver in successful_solvers[idxs]], "Successful Solvers";
            framevisible=true, framewidth = STROKEWIDTH, position = :rb,
            titlesize = 20, labelsize = 16, patchsize = (40.0f0, 20.0f0))

        fig[0, :] = Label(fig, "Brusselator Steady State PDE: Work Precision Diagram",
            fontsize = 24, tellwidth = false, font = :bold)

        fig
    end
end
```

![](figures/bruss_17_1.png)

```julia
save("brusselator_wpd.svg", fig)
```

```
CairoMakie.Screen{SVG}
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/NonlinearProblem","bruss.jmd")
```

Computer Information:

```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
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
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Project.toml`
⌃ [2169fc97] AlgebraicMultigrid v0.6.0
⌃ [6e4b80f9] BenchmarkTools v1.5.0
⌃ [13f3f980] CairoMakie v0.12.16
⌃ [2b5f629d] DiffEqBase v6.158.3
⌃ [f3b72e0c] DiffEqDevTools v2.45.1
⌃ [a0c0ee7d] DifferentiationInterface v0.6.22
⌃ [7da242da] Enzyme v0.13.14
  [40713840] IncompleteLU v0.2.1
  [b964fa9f] LaTeXStrings v1.4.0
  [d3d80556] LineSearches v7.3.0
⌅ [7ed4a6bd] LinearSolve v2.36.2
  [4854310b] MINPACK v1.3.0
  [2774e3e8] NLsolve v4.5.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
⌃ [8913a72c] NonlinearSolve v4.1.0
  [ace2c81b] PETSc v0.3.1
  [98d1487c] PolyesterForwardDiff v0.1.2
  [08abe8d2] PrettyTables v2.4.0
  [f2c3362d] RecursiveFactorization v0.2.23
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [efcf1570] Setfield v1.1.1
⌃ [727e6d20] SimpleNonlinearSolve v2.0.0
⌃ [9f842d2f] SparseConnectivityTracer v0.6.8
⌃ [47a9eef4] SparseDiffTools v2.23.0
  [f1835b91] SpeedMapping v0.3.0
  [860ef19b] StableRNGs v1.0.2
⌃ [90137ffa] StaticArrays v1.9.8
⌃ [c3572dad] Sundials v4.26.1
⌃ [0c5d862f] Symbolics v6.18.3
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/NonlinearProblem/Manifest.toml`
⌃ [47edcb42] ADTypes v1.9.0
  [a4c015fc] ANSIColoredPrinters v0.0.1
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
⌃ [7d9f7c33] Accessors v0.1.38
  [22286c92] AccurateArithmetic v0.3.8
⌃ [79e6a3ab] Adapt v4.1.1
  [35492f91] AdaptivePredicates v1.2.0
⌃ [2169fc97] AlgebraicMultigrid v0.6.0
  [66dad0bd] AliasTables v1.1.3
⌃ [27a7e980] Animations v0.4.1
⌃ [4c88cf16] Aqua v0.8.9
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.17.0
⌃ [4c555306] ArrayLayouts v1.10.4
  [67c07d97] Automa v1.1.0
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.7
⌃ [6e4b80f9] BenchmarkTools v1.5.0
  [e2ed5e7c] Bijections v0.1.9
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [8e7c35d0] BlockArrays v1.1.1
⌃ [70df07ce] BracketingNonlinearSolve v1.1.0
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.6
⌃ [159f3aea] Cairo v1.1.0
⌃ [13f3f980] CairoMakie v0.12.16
  [7057c7e9] Cassette v0.3.14
⌃ [d360d2e6] ChainRulesCore v1.25.0
  [fb6a15b2] CloseOpenIntervals v0.1.13
⌃ [944b1d66] CodecZlib v0.7.6
⌃ [a2cac450] ColorBrewer v0.4.0
⌃ [35d6a980] ColorSchemes v3.27.1
⌅ [3da002f7] ColorTypes v0.11.5
⌅ [c3611d14] ColorVectorSpace v0.10.0
⌅ [5ae59095] Colors v0.12.11
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
  [34da2185] Compat v4.16.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
⌃ [f0e56b4a] ConcurrentUtilities v2.4.2
  [8f4d0f93] Conda v1.10.2
  [187b0558] ConstructionBase v1.5.8
  [d38c429a] Contour v0.6.3
  [a2441757] Coverage v1.6.1
  [c36e975a] CoverageTools v1.3.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
⌃ [864edb3b] DataStructures v0.18.20
  [e2d170a0] DataValueInterfaces v1.0.0
⌃ [927a84f5] DelaunayTriangulation v1.6.1
⌃ [2b5f629d] DiffEqBase v6.158.3
⌃ [f3b72e0c] DiffEqDevTools v2.45.1
⌃ [77a26b50] DiffEqNoiseProcess v5.23.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌃ [a0c0ee7d] DifferentiationInterface v0.6.22
  [b4f34e82] Distances v0.10.12
⌃ [31c24e10] Distributions v0.25.113
  [ffbed154] DocStringExtensions v0.9.3
⌃ [e30172f5] Documenter v1.7.0
  [35a29f4d] DocumenterTools v0.1.20
⌃ [5b8099bc] DomainSets v0.7.14
⌃ [7c1d4256] DynamicPolynomials v0.6.0
  [4e289a0a] EnumX v1.0.4
⌃ [7da242da] Enzyme v0.13.14
⌃ [f151be2c] EnzymeCore v0.8.5
  [429591f6] ExactPredicates v2.2.8
⌃ [460bff9d] ExceptionUnwrapping v0.1.10
  [e2ba6199] ExprTools v0.1.10
⌅ [6b7a57c9] Expronicon v0.8.5
⌃ [411431e0] Extents v0.1.4
⌃ [7a1cc6ca] FFTW v1.8.0
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.4
⌃ [5789e2e9] FileIO v1.16.4
⌅ [8fc22ac5] FilePaths v0.8.3
⌃ [48062228] FilePathsBase v0.9.22
  [1a297f60] FillArrays v1.13.0
⌃ [6a86dc24] FiniteDiff v2.26.0
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.38
  [b38be410] FreeType v4.1.1
⌃ [663a7486] FreeTypeAbstraction v0.10.4
  [f62d2435] FunctionProperties v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
⌅ [46192b85] GPUArraysCore v0.1.6
⌃ [61eb1bfa] GPUCompiler v1.0.1
⌃ [68eda718] GeoFormatTypes v0.4.2
⌃ [cf35fbd7] GeoInterface v1.3.8
⌅ [5c1252a2] GeometryBasics v0.4.11
  [d7ba0133] Git v1.3.1
  [a2bd30eb] Graphics v1.1.3
  [86223c79] Graphs v1.12.0
⌃ [3955a311] GridLayoutBase v0.11.0
  [42e2da0e] Grisu v1.0.2
⌃ [708ec375] Gumbo v0.8.2
⌃ [cd3eb016] HTTP v1.10.10
  [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.17
⌃ [34004b35] HypergeometricFunctions v0.3.25
  [7073ff75] IJulia v1.26.0
  [b5f81e59] IOCapture v0.2.5
  [615f187c] IfElse v0.1.1
  [2803e5a7] ImageAxes v0.6.12
  [c817782e] ImageBase v0.1.7
⌃ [a09fc81d] ImageCore v0.10.4
  [82e4d734] ImageIO v0.6.9
  [bc367c6b] ImageMetadata v0.9.10
  [40713840] IncompleteLU v0.2.1
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.5
  [18e54dd8] IntegerMathUtils v0.1.2
  [a98d9a8b] Interpolations v0.15.1
⌃ [d1acc4aa] IntervalArithmetic v0.22.19
  [8197267c] IntervalSets v0.7.10
  [3587e190] InverseFunctions v0.1.17
⌃ [92d709cd] IrrationalConstants v0.2.2
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
⌃ [692b3bcd] JLLWrappers v1.6.1
  [682c06a0] JSON v0.21.4
⌃ [b835a17e] JpegTurbo v0.1.5
  [ef3ab10e] KLU v0.6.0
  [5ab0869b] KernelDensity v0.6.9
⌃ [ba0b0d4f] Krylov v0.9.8
⌃ [929cbde3] LLVM v9.1.3
  [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.5
  [10f19ff3] LayoutPointers v0.1.17
  [0e77f7df] LazilyInitializedFields v1.3.0
⌃ [5078a376] LazyArrays v2.2.1
  [8cdb02fc] LazyModules v0.3.1
  [87fe0de2] LineSearch v0.1.4
  [d3d80556] LineSearches v7.3.0
⌅ [7ed4a6bd] LinearSolve v2.36.2
⌃ [2ab3a3ac] LogExpFunctions v0.3.28
  [e6f89c97] LoggingExtras v1.1.0
  [bdcacae8] LoopVectorization v0.12.171
  [4854310b] MINPACK v1.3.0
  [d8e11817] MLStyle v0.4.17
  [da04e1cc] MPI v0.20.22
  [3da0fdf6] MPIPreferences v0.1.11
⌃ [1914dd2f] MacroTools v0.5.13
⌅ [ee78f7c6] Makie v0.21.16
⌅ [20f20a25] MakieCore v0.8.10
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
  [d0879d2d] MarkdownAST v0.1.2
  [0a4f8689] MathTeXEngine v0.6.2
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.7
  [ffc61752] Mustache v1.0.20
⌃ [d8a4904e] MutableArithmetics v1.5.2
⌃ [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
⌃ [77ba4419] NaNMath v1.0.2
  [f09324ee] Netpbm v1.1.1
  [b7050fa9] NonlinearProblemLibrary v0.1.2
⌃ [8913a72c] NonlinearSolve v4.1.0
⌃ [be0214bd] NonlinearSolveBase v1.3.1
⌃ [5959db7a] NonlinearSolveFirstOrder v1.0.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.0.0
⌃ [26075421] NonlinearSolveSpectralMethods v1.0.0
⌃ [d8793406] ObjectFile v0.4.2
  [510215fc] Observables v0.5.5
⌃ [6fe1bfb0] OffsetArrays v1.14.1
  [52e1d378] OpenEXR v0.3.3
  [4d8831e6] OpenSSL v1.4.3
⌃ [429524aa] Optim v1.9.4
⌃ [bac558e1] OrderedCollections v1.6.3
⌃ [90014a1f] PDMats v0.11.31
  [ace2c81b] PETSc v0.3.1
⌃ [f57f5aa1] PNGFiles v0.4.3
  [65ce6f38] PackageExtensionCompat v1.0.2
⌃ [19eb6ba3] Packing v0.5.0
  [5432bcbf] PaddedViews v0.5.12
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [eebad327] PkgVersion v0.3.3
  [995b91a9] PlotUtils v1.4.3
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.16
  [98d1487c] PolyesterForwardDiff v0.1.2
  [1d0040c9] PolyesterWeave v0.2.2
  [647866c9] PolygonOps v0.1.2
  [85a6dd25] PositiveFactorizations v0.2.4
⌃ [d236fae5] PreallocationTools v0.4.24
  [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.4.0
⌃ [27ebfcd6] Primes v0.5.6
  [92933f4c] ProgressMeter v1.10.2
⌃ [43287f4e] PtrArrays v1.2.1
  [4b34888f] QOI v1.0.1
⌃ [1fd47b50] QuadGK v2.11.1
  [74087812] Random123 v1.7.0
  [e6cf234a] RandomNumbers v1.6.0
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
⌃ [731186ca] RecursiveArrayTools v3.27.3
  [f2c3362d] RecursiveFactorization v0.2.23
  [189a3867] Reexport v1.2.2
  [2792f1a3] RegistryInstances v0.1.0
  [05181044] RelocatableFolders v1.0.1
⌃ [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.8.0
  [47965b36] RootedTrees v2.23.1
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.13
⌃ [fdea26ae] SIMD v3.7.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.43
  [322a6be2] Sass v0.2.0
⌃ [0bca4576] SciMLBase v2.59.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.1
⌃ [c0aeaf25] SciMLOperators v0.3.12
⌃ [53ae85a6] SciMLStructures v1.5.0
  [6c6a2e73] Scratch v1.2.1
⌃ [efcf1570] Setfield v1.1.1
⌅ [65257c39] ShaderAbstractions v0.4.1
  [992d4aef] Showoff v1.0.3
  [73760f76] SignedDistanceFields v0.4.0
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.0.0
  [699a6c99] SimpleTraits v0.9.4
  [45858cf5] Sixel v0.1.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
⌃ [9f842d2f] SparseConnectivityTracer v0.6.8
⌃ [47a9eef4] SparseDiffTools v2.23.0
⌃ [0a514795] SparseMatrixColorings v0.4.9
  [e56a9233] Sparspak v0.3.9
⌃ [276daf66] SpecialFunctions v2.4.0
  [f1835b91] SpeedMapping v0.3.0
  [860ef19b] StableRNGs v1.0.2
  [cae243ae] StackViews v0.1.1
⌃ [aedffcd0] Static v1.1.1
  [0d7ed370] StaticArrayInterface v1.8.0
⌃ [90137ffa] StaticArrays v1.9.8
  [1e83bf80] StaticArraysCore v1.4.3
  [82ae8749] StatsAPI v1.7.0
⌃ [2913bbd2] StatsBase v0.34.3
  [4c63d2b9] StatsFuns v1.3.2
  [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
⌃ [892a3eda] StringManipulation v0.4.0
⌅ [09ab397b] StructArrays v0.6.18
  [53d494c1] StructIO v0.3.1
⌃ [c3572dad] Sundials v4.26.1
⌃ [2efcf032] SymbolicIndexingInterface v0.3.34
  [19f23fe9] SymbolicLimits v0.2.2
⌃ [d1185830] SymbolicUtils v3.7.2
⌃ [0c5d862f] Symbolics v6.18.3
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.0
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [8290d209] ThreadingUtilities v0.5.2
⌃ [731e570b] TiffImages v0.11.1
⌃ [a759f4b9] TimerOutputs v0.5.25
  [3bb67fe8] TranscodingStreams v0.11.3
  [d5829a12] TriangularSolve v0.2.1
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌃ [1986cc42] Unitful v1.21.0
  [a7c27f48] Unityper v0.1.6
  [3d5dd08c] VectorizationBase v0.21.71
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [e3aaa7dc] WebP v0.1.3
  [efce3f68] WoodburyMatrices v1.0.0
⌃ [ddb6d928] YAML v0.4.12
⌃ [c2297ded] ZMQ v1.3.0
⌃ [6e34b625] Bzip2_jll v1.0.8+2
  [4e9b3aee] CRlibm_jll v1.0.1+0
⌃ [83423d85] Cairo_jll v1.18.2+1
  [5ae413db] EarCut_jll v2.2.4+0
⌅ [7cc45869] Enzyme_jll v0.0.163+0
⌃ [2e619515] Expat_jll v2.6.2+0
⌅ [b22a6f82] FFMPEG_jll v6.1.2+0
⌃ [f5851436] FFTW_jll v3.3.10+1
⌃ [a3f928ae] Fontconfig_jll v2.13.96+0
⌃ [d7e528f0] FreeType2_jll v2.13.2+0
⌃ [559328eb] FriBidi_jll v1.0.14+0
  [78b55507] Gettext_jll v0.21.0+0
⌃ [59f7168a] Giflib_jll v5.2.2+0
⌃ [f8c6e375] Git_jll v2.46.2+0
⌃ [7746bdde] Glib_jll v2.80.5+0
⌃ [3b182d85] Graphite2_jll v1.3.14+0
  [528830af] Gumbo_jll v0.10.2+0
⌃ [2e76f6c2] HarfBuzz_jll v8.3.1+0
⌃ [e33a78d0] Hwloc_jll v2.11.2+1
  [905a6f67] Imath_jll v3.1.11+0
⌅ [1d5cc7b8] IntelOpenMP_jll v2024.2.1+0
⌃ [aacddb02] JpegTurbo_jll v3.0.4+0
  [c1c5ebd0] LAME_jll v3.100.2+0
⌃ [88015f11] LERC_jll v4.0.0+0
⌅ [dad2f222] LLVMExtra_jll v0.0.34+0
  [1d63c593] LLVMOpenMP_jll v18.1.7+0
⌃ [dd4b983a] LZO_jll v2.10.2+1
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.11.0+0
⌃ [7e76a0d4] Libglvnd_jll v1.6.0+0
⌃ [7add5ba3] Libgpg_error_jll v1.50.0+0
⌃ [94ce4f54] Libiconv_jll v1.17.0+1
⌃ [4b2f31a3] Libmount_jll v2.40.1+0
⌃ [89763e89] Libtiff_jll v4.7.0+0
⌃ [38a345b3] Libuuid_jll v2.40.1+0
⌅ [856f044c] MKL_jll v2024.2.0+0
⌃ [7cb0a576] MPICH_jll v4.2.3+0
⌃ [f1f71cc9] MPItrampoline_jll v5.5.1+0
⌃ [9237b28f] MicrosoftMPI_jll v10.1.4+2
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [656ef2d0] OpenBLAS32_jll v0.3.24+0
  [18a262bb] OpenEXR_jll v3.2.4+0
⌃ [fe0851c0] OpenMPI_jll v5.0.5+0
⌃ [9bd350c2] OpenSSH_jll v9.9.1+1
⌃ [458c3c95] OpenSSL_jll v3.0.15+1
⌃ [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.3+0
  [8fa3689e] PETSc_jll v3.22.0+0
⌃ [36c8627f] Pango_jll v1.54.1+0
⌅ [30392449] Pixman_jll v0.43.4+0
  [f50d1b31] Rmath_jll v0.5.1+0
  [aabda75e] SCALAPACK32_jll v2.2.1+1
⌅ [fb77eaff] Sundials_jll v5.2.2+0
⌃ [02c8fc9c] XML2_jll v2.13.4+0
⌃ [aed1982a] XSLT_jll v1.1.41+0
⌃ [ffd25f8a] XZ_jll v5.6.3+0
⌃ [4f6342f7] Xorg_libX11_jll v1.8.6+0
⌃ [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
⌃ [a3789734] Xorg_libXdmcp_jll v1.1.4+0
⌃ [1082639a] Xorg_libXext_jll v1.3.6+0
⌃ [ea2f1a96] Xorg_libXrender_jll v0.9.11+0
⌃ [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
⌃ [c7cfdc94] Xorg_libxcb_jll v1.17.0+0
⌃ [c5fb5394] Xorg_xtrans_jll v1.5.0+0
⌃ [8f1865be] ZeroMQ_jll v4.3.5+1
⌃ [3161d3a3] Zstd_jll v1.5.6+1
  [b792d7bf] cminpack_jll v1.3.8+0
  [9a68df92] isoband_jll v0.2.3+0
⌃ [a4ae2306] libaom_jll v3.9.0+0
  [0ac62f75] libass_jll v0.15.2+0
  [f638f0a6] libfdk_aac_jll v2.0.3+0
⌃ [b53b4c65] libpng_jll v1.6.44+0
  [47bcb7c8] libsass_jll v3.6.6+0
⌃ [075b6546] libsixel_jll v1.10.3+1
⌃ [a9144af2] libsodium_jll v1.0.20+1
  [f27f6e37] libvorbis_jll v1.3.7+2
⌃ [c5f90fcd] libwebp_jll v1.4.0+0
⌃ [1317d2d5] oneTBB_jll v2021.12.0+0
⌃ [1270edf5] x264_jll v10164.0.0+0
⌅ [dfaa095f] x265_jll v3.6.0+0
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
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

