
using NonlinearSolve, LinearAlgebra, SparseArrays, DiffEqDevTools,
    CairoMakie, Symbolics, BenchmarkTools, PolyesterForwardDiff, LinearSolve, Sundials,
    Enzyme, SparseConnectivityTracer, DifferentiationInterface, SparseMatrixColorings
import NLsolve, MINPACK, PETSc, RecursiveFactorization

const RUS = RadiusUpdateSchemes;
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.2;


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


test_problem = generate_brusselator_problem(4)
bruss_f!, u0 = (du, u) -> test_problem.f(du, u, test_problem.p), test_problem.u0
y = similar(u0)

J = Float64.(ADTypes.jacobian_sparsity(bruss_f!, y, u0, TracerSparsityDetector()))
colors = fast_coloring(J, ColoringProblem(), GreedyColoringAlgorithm())

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

Ns = [2^i for i in 1:8];

adtypes = [
    (
        AutoSparse(
            AutoFiniteDiff();
            sparsity_detector=TracerSparsityDetector(),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:finitediff, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoPolyesterForwardDiff();
            sparsity_detector=TracerSparsityDetector(),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:polyester, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoEnzyme(; mode=Enzyme.Forward);
            sparsity_detector=TracerSparsityDetector(),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:enzyme, :exact_sparse]
    ),
    (
        AutoSparse(
            AutoFiniteDiff();
            sparsity_detector=DenseSparsityDetector(AutoFiniteDiff(); atol=1e-5),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:finitediff, :approx_sparse]
    ),
    (
        AutoSparse(
            AutoPolyesterForwardDiff();
            sparsity_detector=DenseSparsityDetector(
                AutoPolyesterForwardDiff(); atol=1e-5
            ),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:polyester, :approx_sparse]
    ),
    (
        AutoSparse(
            AutoEnzyme(; mode=Enzyme.Forward);
            sparsity_detector=DenseSparsityDetector(
                AutoEnzyme(; mode=Enzyme.Forward); atol=1e-5
            ),
            coloring_algorithm=GreedyColoringAlgorithm(LargestFirst())
        ),
        [:enzyme, :approx_sparse]
    ),
    (
        AutoPolyesterForwardDiff(),
        [:polyester, :none]
    ),
];

times = Matrix{Float64}(undef, length(Ns), length(adtypes));

for (i, N) in enumerate(Ns)
    str = "$(lpad(N, 10)) "
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
        str = str * "$(lpad(times[i, j], 16))"
    end
    println(str)
end
nothing


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
            colormap=:tableau_20)

        ax = Axis(fig[1, 2]; title="Scaling of Sparse Jacobian Computation",
            titlesize=22, titlegap=10, xscale=log2, yscale=log2,
            xticksize=20, yticksize=20, xticklabelsize=20, yticklabelsize=20,
            xtickwidth=2.5, ytickwidth=2.5, spinewidth=2.5,
            xlabel=L"Input Dimension ($\mathbf{N}$)",
            ylabel=L"Time $\mathbf{(s)}$", xlabelsize=22,
            ylabelsize=22, yaxisposition=:right)

        colors = cgrad(:tableau_20, length(adtypes); categorical=true)

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
            ["Exact Sparsity", "Approx. Local Sparsity", "Dense"];
            position=:lt, framevisible=true, framewidth=2.5, titlesize=18,
            labelsize=16, patchsize=(40.0f0, 20.0f0)
        )

        fig
    end
end


save("brusselator_sparse_jacobian_scaling.svg", fig)


Ns = vcat(collect(2 .^ (2:7)), [150, 175, 200])

solvers_scaling = [
    (; pkg = :nonlinearsolve, sparsity = :none,   name = "NR (No Sparsity)",                    alg = NewtonRaphson()),
    (; pkg = :nonlinearsolve, sparsity = :exact,  name = "NR (Exact Sparsity)",                 alg = NewtonRaphson()),
    (; pkg = :wrapper,        sparsity = :none,   name = "NR [NLsolve.jl]",                     alg = NLsolveJL(; method = :newton, autodiff = :forward)),
    (; pkg = :wrapper,        sparsity = :none,   name = "NR [Sundials]",                       alg = KINSOL(; linear_solver = :LapackDense, maxsetupcalls=1)),
    (; pkg = :wrapper,        sparsity = :none,   name = "NR [PETSc] (No Sparsity)",            alg = PETScSNES(; snes_type = "newtonls", snes_linesearch_type = "basic", autodiff = missing)),
    (; pkg = :wrapper,        sparsity = :exact,  name = "NR [PETSc] (Exact Sparsity)",         alg = PETScSNES(; snes_type = "newtonls", snes_linesearch_type = "basic")),

    (; pkg = :nonlinearsolve, sparsity = :none,   name = "TR (No Sparsity)",                    alg = TrustRegion(; radius_update_scheme = RUS.NLsolve)),
    (; pkg = :nonlinearsolve, sparsity = :exact,  name = "TR (Exact Sparsity)",                 alg = TrustRegion(; radius_update_scheme = RUS.NLsolve)),
    (; pkg = :wrapper,        sparsity = :none,   name = "TR [NLsolve.jl]",                     alg = NLsolveJL(; autodiff = :forward)),
    (; pkg = :wrapper,        sparsity = :none,   name = "TR [PETSc] (No Sparsity)",            alg = PETScSNES(; snes_type = "newtontr", autodiff = missing)),
    (; pkg = :wrapper,        sparsity = :exact,  name = "TR [PETSc] (Exact Sparsity)",         alg = PETScSNES(; snes_type = "newtontr")),

    (; pkg = :wrapper,        sparsity = :none,   name = "Mod. Powell [MINPACK]",               alg = CMINPACK()),
]

GC.enable(false) # for PETSc

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
            (alg isa PETScSNES && N > 64)
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


fig = begin
    ASPECT_RATIO = 0.7
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    cycle = Cycle([:marker], covary = true)
    colors = cgrad(:tableau_20, length(solvers_scaling); categorical = true)
    theme = Theme(Lines = (cycle = cycle,), Scatter = (cycle = cycle,))
    LINESTYLES = Dict(
        (:nonlinearsolve, :none) => :solid,
        (:nonlinearsolve, :exact) => :dashdot,
        # (:simplenonlinearsolve, :none) => :solid,
        (:wrapper, :exact) => :dash,
        (:wrapper, :none) => :dot,
    )

    Ns_ = Ns .^ 2 .* 2

    with_theme(theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        ax = Axis(fig[1, 1:3], ylabel = L"Time ($s$)", xlabel = L"Problem Size ($N$)",
            xscale = log2, yscale = log2, xlabelsize = 22, ylabelsize = 22,
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
            "Successful Solvers";
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


save("brusselator_scaling.svg", fig)


solvers_all = [
    (; pkg = :nonlinearsolve, name = "Default PolyAlg",           solver = Dict(:alg => FastShortcutNonlinearPolyalg())),
    (; pkg = :nonlinearsolve, name = "RobustMultiNewton (GMRES)", solver = Dict(:alg => RobustMultiNewton(; linsolve = KrylovJL_GMRES()))),

    (; pkg = :nonlinearsolve, name = "Newton Raphson",            solver = Dict(:alg => NewtonRaphson(; linsolve = nothing))),
    (; pkg = :nonlinearsolve, name = "Newton Krylov",             solver = Dict(:alg => NewtonRaphson(; linsolve = KrylovJL_GMRES()))),
    (; pkg = :nonlinearsolve, name = "Trust Region",              solver = Dict(:alg => TrustRegion())),
    (; pkg = :nonlinearsolve, name = "TR Krylov",                 solver = Dict(:alg => TrustRegion(; linsolve = KrylovJL_GMRES()))),

    (; pkg = :wrapper,        name = "NR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL(; method = :newton, autodiff = :forward))),
    (; pkg = :wrapper,        name = "TR [NLsolve.jl]",           solver = Dict(:alg => NLsolveJL(; autodiff = :forward))),

    (; pkg = :wrapper,        name = "NR [Sundials]",             solver = Dict(:alg => KINSOL(; linear_solver = :LapackDense, maxsetupcalls=1))),
    (; pkg = :wrapper,        name = "Newton Krylov [Sundials]",  solver = Dict(:alg => KINSOL(; linear_solver = :GMRES, maxsetupcalls=1, krylov_dim = 1000))),

    (; pkg = :wrapper,        name = "Mod. Powell [MINPACK]",     solver = Dict(:alg => CMINPACK())),

    (; pkg = :wrapper,        name = "NR [PETSc]",  solver = Dict(:alg => PETScSNES(; snes_type = "newtonls", snes_linesearch_type = "basic", autodiff = missing))),
    (; pkg = :wrapper,        name = "TR [PETSc]",  solver = Dict(:alg => PETScSNES(; snes_type = "newtontr", autodiff = missing))),
    (; pkg = :wrapper,        name = "Newton Krylov [PETSc]",  solver = Dict(:alg => PETScSNES(; snes_type = "newtonls", snes_linesearch_type = "basic", ksp_type = "gmres", autodiff = missing, snes_mf = true, ksp_gmres_restart = 1000))),
];


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


wp_set, successful_solvers = generate_wpset(prob_wpd, solvers_all);


fig = begin
    LINESTYLES = Dict(:nonlinearsolve => :solid, :simplenonlinearsolve => :dash,
        :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1200
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:tableau_20, length(successful_solvers); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    with_theme(plot_theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))
        # `textbf` doesn't work
        ax = Axis(fig[1, 1], ylabel = L"Time $\mathbf{(s)}$",
            xlabelsize = 22, ylabelsize = 22,
            xlabel = L"Error: $\mathbf{||f(u^\ast)||_\infty}$",
            xscale = log2, yscale = log2, xtickwidth = STROKEWIDTH,
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


save("brusselator_wpd.svg", fig)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

