
using BoundaryValueDiffEq, SimpleBoundaryValueDiffEq, OrdinaryDiffEq, ODEInterface, DiffEqDevTools, BenchmarkTools,
      BVProblemLibrary, CairoMakie, NonlinearSolveFirstOrder


solvers_all = [
    (; pkg = :boundaryvaluediffeq,          type = :mirk,         name = "MIRK4",                solver = Dict(:alg => MIRK4(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :mirk,         name = "MIRK5",                solver = Dict(:alg => MIRK5(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :mirk,         name = "MIRK6",                solver = Dict(:alg => MIRK6(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "RadauIIa3",            solver = Dict(:alg => RadauIIa3(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "RadauIIa5",            solver = Dict(:alg => RadauIIa5(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "RadauIIa7",            solver = Dict(:alg => RadauIIa7(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIa4",         solver = Dict(:alg => LobattoIIIa4(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIa5",         solver = Dict(:alg => LobattoIIIa5(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIb4",         solver = Dict(:alg => LobattoIIIb4(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIb5",         solver = Dict(:alg => LobattoIIIb5(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIc4",         solver = Dict(:alg => LobattoIIIc4(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,         name = "LobattoIIIc5",         solver = Dict(:alg => LobattoIIIc5(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :boundaryvaluediffeq,          type = :shooting,     name = "Single Shooting",      solver = Dict(:alg => Shooting(Tsit5(), NewtonRaphson()))),
    (; pkg = :boundaryvaluediffeq,          type = :shooting,     name = "Multiple Shooting",    solver = Dict(:alg => MultipleShooting(10, Tsit5()))),
    (; pkg = :wrapper,                      type = :general,      name = "BVPM2",                solver = Dict(:alg => BVPM2(), :dts=>1.0 ./ 5.0 .^ (1:4))),
    (; pkg = :wrapper,                      type = :general,      name = "COLNEW",               solver = Dict(:alg => COLNEW(), :dts=>1.0 ./ 5.0 .^ (1:4))),
];

solver_tracker = [];
wp_general_tracker = [];


abstols = 1.0 ./ 10.0 .^ (1:4)
reltols = 1.0 ./ 10.0 .^ (1:4);


function benchmark(prob)
    sol = solve(prob, MIRK6(), dt = 0.01, abstol = 1e-6)
    testsol = TestSolution(sol)
    wps = WorkPrecisionSet(prob, abstols, reltols, getfield.(solvers_all, :solver); names = getfield.(solvers_all, :name), appxsol = testsol, maxiters=Int(1e4))
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
                xlabel = L"Error: $\mathbf{||f(u^\ast)||_2}$",
                xscale = log10, yscale = log10, xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                xticklabelsize = 20, yticklabelsize = 20)

            idxs = sortperm(median.(getfield.(wp_set.wps, :times)))

            ls, scs = [], []

            for (i, (wp, solver)) in enumerate(zip(wp_set.wps[idxs], solvers_all[idxs]))
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
            ylims!(ax; low=5e-7)

            Legend(fig[1,2], [[l, sc] for (l, sc) in zip(ls, scs)],
                [solver.name for solver in solvers_all[idxs]], "BVP Solvers";
                framevisible=true, framewidth = STROKEWIDTH, position = :rb,
                titlesize = 20, labelsize = 16, patchsize = (40.0f0, 20.0f0))

            fig[0, :] = Label(fig, "Nonlinear BVP Benchmark",
                fontsize = 24, tellwidth = false, font = :bold)
            fig
        end
    end
end


prob_1 = BVProblemLibrary.prob_bvp_nonlinear_1
wps = benchmark(prob_1)
plot_wpd(wps)


prob_2 = BVProblemLibrary.prob_bvp_nonlinear_2
wps = benchmark(prob_2)
plot_wpd(wps)


prob_3 = BVProblemLibrary.prob_bvp_nonlinear_3
wps = benchmark(prob_3)
plot_wpd(wps)


prob_4 = BVProblemLibrary.prob_bvp_nonlinear_4
wps = benchmark(prob_4)
plot_wpd(wps)


prob_5 = BVProblemLibrary.prob_bvp_nonlinear_5
wps = benchmark(prob_5)
plot_wpd(wps)


prob_6 = BVProblemLibrary.prob_bvp_nonlinear_6
wps = benchmark(prob_6)
plot_wpd(wps)


prob_7 = BVProblemLibrary.prob_bvp_nonlinear_7
wps = benchmark(prob_7)
plot_wpd(wps)


prob_8 = BVProblemLibrary.prob_bvp_nonlinear_8
wps = benchmark(prob_8)
plot_wpd(wps)


prob_9 = BVProblemLibrary.prob_bvp_nonlinear_9
wps = benchmark(prob_9)
plot_wpd(wps)


prob_10 = BVProblemLibrary.prob_bvp_nonlinear_10
wps = benchmark(prob_10)
plot_wpd(wps)


prob_11 = BVProblemLibrary.prob_bvp_nonlinear_11
wps = benchmark(prob_11)
plot_wpd(wps)


prob_12 = BVProblemLibrary.prob_bvp_nonlinear_12
wps = benchmark(prob_12)
plot_wpd(wps)


prob_13 = BVProblemLibrary.prob_bvp_nonlinear_13
wps = benchmark(prob_13)
plot_wpd(wps)


prob_14 = BVProblemLibrary.prob_bvp_nonlinear_14
wps = benchmark(prob_14)
plot_wpd(wps)


prob_15 = BVProblemLibrary.prob_bvp_nonlinear_15
wps = benchmark(prob_15)
plot_wpd(wps)


fig = begin
    LINESTYLES = Dict(:boundaryvaluediffeq => :solid, :wrapper => :dot)
    ASPECT_RATIO = 0.7
    WIDTH = 1800
    HEIGHT = round(Int, WIDTH * ASPECT_RATIO)
    STROKEWIDTH = 2.5

    colors = cgrad(:seaborn_bright, length(solvers_all); categorical = true)
    cycle = Cycle([:marker], covary = true)
    plot_theme = Theme(Lines = (; cycle), Scatter = (; cycle))

    with_theme(plot_theme) do
        fig = Figure(; size = (WIDTH, HEIGHT))

        ls = []
        scs = []
        labels = []
        solver_times = []

        for i in 1:3, j in 1:5
            idx = 5 * (i - 1) + j

            idx > length(wp_general_tracker) && break

            wp = wp_general_tracker[idx]

            ax = Axis(fig[i, j],
                xscale = log10, yscale = log10,
                xtickwidth = STROKEWIDTH,
                ytickwidth = STROKEWIDTH, spinewidth = STROKEWIDTH,
                title = "No. $(idx) Nonlinear BVP benchmarking", titlegap = 10,
                xticklabelsize = 16, yticklabelsize = 16)

            for wpᵢ in wp.wps
                idx = findfirst(s -> s.name == wpᵢ.name, solvers_all)
                errs = getindex.(wpᵢ.errors, :l∞)
                times = wpᵢ.times

                l = lines!(ax, errs, times; color = colors[idx], linewidth = 5,
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

        fig[0, :] = Label(fig, "Work-Precision Diagram for 15 Nonlinear Test Problems",
            fontsize = 24, tellwidth = false, font = :bold)

        fig[:, 0] = Label(fig, "Time (s)", fontsize = 20, tellheight = false, font = :bold,
            rotation = π / 2)
        fig[end + 1, :] = Label(fig,
            L"Error: $\mathbf{||f(u^\ast)||_2}$",
            fontsize = 20, tellwidth = false, font = :bold)

        Legend(fig[:, 6], [[l, sc] for (l, sc) in zip(ls, scs)],
            labels, "BVP Solvers";
            framevisible=true, framewidth = STROKEWIDTH, orientation = :vertical,
            titlesize = 20, nbanks = 1, labelsize = 20, halign = :center,
            tellheight = false, tellwidth = false, patchsize = (40.0f0, 20.0f0))

        return fig
    end
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

