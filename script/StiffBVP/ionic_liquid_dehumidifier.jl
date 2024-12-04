
using BoundaryValueDiffEq, BracketingNonlinearSolve, ODEInterface, DiffEqDevTools, BenchmarkTools,
      Interpolations, StaticArrays, CairoMakie


solvers_all = [
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "RadauIIa3",            solver = Dict(:alg => RadauIIa3(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "RadauIIa5",            solver = Dict(:alg => RadauIIa5(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "RadauIIa7",            solver = Dict(:alg => RadauIIa7(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIa3",         solver = Dict(:alg => LobattoIIIa3(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIa4",         solver = Dict(:alg => LobattoIIIa4(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIa5",         solver = Dict(:alg => LobattoIIIa5(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb3",         solver = Dict(:alg => LobattoIIIb3(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb4",         solver = Dict(:alg => LobattoIIIb4(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb5",         solver = Dict(:alg => LobattoIIIb5(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb3",         solver = Dict(:alg => LobattoIIIc3(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb4",         solver = Dict(:alg => LobattoIIIc4(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :boundaryvaluediffeq,          type = :firk,              name = "LobattoIIIb5",         solver = Dict(:alg => LobattoIIIc5(; nested_nlsolve = true), :dts=>1.0 ./ 10.0 .^ (2:4))),
    (; pkg = :wrapper,                      type = :general,           name = "COLNEW",               solver = Dict(:alg => COLNEW(), :dts=>1.0 ./ 10.0 .^ (2:4))),
];


abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3);


iᵥ_ₛₐₜ(T) = 10^(6.697227966814859 - 273.8702703951898 /
									(T + 642.1729733423742))

begin
	"Properties Interpolations and Extrapolations"

	const Tⁿᵒᵈᵉˢ = @SVector[x + 273.15 for x in [25.0, 35.0, 60.0, 80.0]]
	const ξⁿᵒᵈᵉˢ_2 = @SVector[x * 0.01 for x in [0.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0]]
	const nodes = (Tⁿᵒᵈᵉˢ, ξⁿᵒᵈᵉˢ_2)

	const Δh_data = @SMatrix[
		0.0  -58000.0  -75000.0  -74000.0  -68000.0  -55000.0  -34000.0
		0.0  -57000.0  -72000.0  -72000.0  -67000.0  -54000.0  -33000.0
		0.0  -52000.0  -67000.0  -67000.0  -62000.0  -51000.0  -31000.0
		0.0  -48000.0  -62000.0  -64000.0  -59000.0  -49000.0  -30000.0
	]
	# # ================== Interpolation and Extrapolation P_ν ==================
	const a0_p = 12.10
	const a1_p = -28.01
	const a2_p = 50.34
	const a3_p = -24.63
	const b0_p = 1212.67
	const b1_p = 772.37
	const b2_p = 614.59
	const b3_p = 493.33

	@inline function _Pᵥₐₚₒᵣ_ₛₒₗ(T, ξ)
		A = a0_p + a1_p * ξ + a2_p * ξ^2 + a3_p * ξ^3
		B = b0_p + b1_p * ξ + b2_p * ξ^2 + b3_p * ξ^3
		return 10^(A - B / T) * 100.0
	end
	# ================== Interpolation and Extrapolation cp ====================  
	function _cpₛₒₗ(T, ξ)
		return ((0.00476 * T - 4.01) * ξ + 4.21) * 1e3
	end

	@inline function _Δh(T, ξ)
		Δh_interpolated = interpolate(nodes, Δh_data, Gridded(Linear()))
		Δh_extrapolated = extrapolate(Δh_interpolated, Line())
		return Δh_extrapolated(T, ξ)
	end

	@inline function _iₛₒₗ(T, ξ)
		Δh = _Δh(T, ξ)
		i = _cpₛₒₗ(T, ξ) * (T - 273.15) + Δh
		return i
	end

	# ================== Find T given i_sol and ξ ====================
	# Function to find the root, given i_sol and ξ
	@inline function calculate_T_sol(iᵛₛₒₗ, ξ; T_lower = -150.0 + 273.15, T_upper = 95.0 + 273.15)
		f(T, p) = _iₛₒₗ(T, p[2]) - p[1]
		p = @SVector[iᵛₛₒₗ, ξ]
		T_span = (T_lower, T_upper)
		prob = IntervalNonlinearProblem{false}(f, T_span, p)
		result = solve(prob, BracketingNonlinearSolve.ITP())
		return result.u
	end
end

function ionic_liquid_coil_ode!(du, u, p, t)
	# ωₐᵢᵣ, iₐᵢᵣ, ṁₛₒₗ,ξₛₒₗ, iₛₒₗ = u
	# ========================================
	Le = p[1]
	∂Qᵣ = p[2]
	ṁₐᵢᵣ = p[3]
	NTUᴰₐᵢᵣ = p[4]
	σ = p[5]
	ṁₛₒₗ_ᵢₙ = p[6]
	ξₛₒₗ_ᵢₙ = p[7]
	iₛₒₗ_ᵢₙ = p[8]
	ωₐ_ᵢₙ = p[9]
	iₐ_ᵢₙ = p[10]
	MR = ṁₛₒₗ_ᵢₙ / ṁₐᵢᵣ
	ER = iₛₒₗ_ᵢₙ / iₐ_ᵢₙ
	# ========================================
	Tₛₒₗ = calculate_T_sol(u[5] * iₛₒₗ_ᵢₙ, u[4] * ξₛₒₗ_ᵢₙ)
	Pᵥₐₚₒᵣ_ₛₒₗ = _Pᵥₐₚₒᵣ_ₛₒₗ(Tₛₒₗ, u[4] * ξₛₒₗ_ᵢₙ)
	ωₑ = 0.622 * Pᵥₐₚₒᵣ_ₛₒₗ / (101325.0 - Pᵥₐₚₒᵣ_ₛₒₗ) / ωₐ_ᵢₙ
	iₑ = (1.005 * (Tₛₒₗ - 273.15) + ωₑ * ωₐ_ᵢₙ * (2500.9 + 1.82 * (Tₛₒₗ - 273.15))) / iₐ_ᵢₙ
	iₑ *= 1000
	iᵥₐₚₒᵣ_ₜₛ = iᵥ_ₛₐₜ(Tₛₒₗ) / iₐ_ᵢₙ

	du[1] = σ * NTUᴰₐᵢᵣ * (u[1] - ωₑ)
	du[2] = σ * NTUᴰₐᵢᵣ * Le * ((u[2] - iₑ) + (ωₐ_ᵢₙ * iᵥₐₚₒᵣ_ₜₛ * (1 / Le - 1) * (u[1] - ωₑ)))
	du[3] = σ * ωₐ_ᵢₙ * du[1] / MR
	du[4] = (-u[4] / u[3]) * du[3]
	du[5] = (1 / u[3]) * (σ * (1.0 / (MR * ER)) * du[2] - u[5] * du[3] - ∂Qᵣ / (ṁₛₒₗ_ᵢₙ * iₛₒₗ_ᵢₙ))
	nothing
end

function bca!(res_a, u_a, p)
	res_a[1] = u_a[3] - 1.0
	res_a[2] = u_a[4] - 1.0
	res_a[3] = u_a[5] - 1.0
	nothing
end

function bcb!(res_b, u_b, p)
	res_b[1] = u_b[1] - 1.0
	res_b[2] = u_b[2] - 1.0
	nothing
end


dt = 0.05
tspan = (0.0, 1.0)

Le = 0.85
σ = 1.0
ṁₛₒₗ_ᵢₙ = 7.466666666666666e-5
ξₛₒₗ_ᵢₙ = 0.8
iₛₒₗ_ᵢₙ = -30235.4128
ωₐ_ᵢₙ = 0.022800264832054707
iₐ_ᵢₙ = 88436.57753410653
∂Qᵣ = -12.416666666666666
ṁₐᵢᵣ_ᵢₙ = 0.0003733333333333333
NTUᴰₐᵢᵣ = 4.678477517542263

p = @SVector[Le, ∂Qᵣ, ṁₐᵢᵣ_ᵢₙ, NTUᴰₐᵢᵣ, σ, ṁₛₒₗ_ᵢₙ, ξₛₒₗ_ᵢₙ, iₛₒₗ_ᵢₙ, ωₐ_ᵢₙ, iₐ_ᵢₙ]

u0 = [0.1, 0.1, 1.0001, 0.9, 1.01]

bvp_fun = BVPFunction(
	ionic_liquid_coil_ode!, (bca!, bcb!);
	bcresid_prototype = (zeros(3), zeros(2)), twopoint = Val(true),
)

prob = TwoPointBVProblem(bvp_fun, u0, tspan, p)
sol = solve(prob, RadauIIa7(nested_nlsolve = true, nest_tol = 1e-3), dt = dt, abstol = 1e-5)
testsol = TestSolution(sol)
wp_set = WorkPrecisionSet(prob, abstols, reltols, getfield.(solvers_all, :solver); names = getfield.(solvers_all, :name), appxsol = testsol, maxiters=Int(1e4))


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
            l = lines!(ax, errors, times; linestyle = LINESTYLES[solver.pkg], label = name,
                linewidth = 5, color = colors[i])
            sc = scatter!(ax, errors, times; label = name, markersize = 16, strokewidth = 2,
                color = colors[i])
            push!(ls, l)
            push!(scs, sc)
        end

        xlims!(ax; high=1)
        ylims!(ax; low=1e-4)

        axislegend(ax, [[l, sc] for (l, sc) in zip(ls, scs)],
            [solver.name for solver in solvers_all[idxs]], "BVP Solvers";
            framevisible=true, framewidth = STROKEWIDTH, position = :rb,
            titlesize = 20, labelsize = 16, patchsize = (40.0f0, 20.0f0))

        fig[0, :] = Label(fig, "Ionic Liquid Dehumidifier Benchmark",
            fontsize = 24, tellwidth = false, font = :bold)
        fig
    end
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

