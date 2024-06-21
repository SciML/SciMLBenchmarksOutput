
using Pkg
# Rev fixes precompilation https://github.com/hzgzh/XSteam.jl/pull/2
Pkg.add(Pkg.PackageSpec(;name="XSteam", rev="f2a1c589054cfd6bba307985a3a534b6f5a1863b"))

using ModelingToolkit, JuliaSimCompiler, Symbolics, XSteam, Polynomials, BenchmarkTools, CairoMakie, OrdinaryDiffEq
using OMJulia


#          o  o  o  o  o  o  o < heat capacitors
#          |  |  |  |  |  |  | < heat conductors
#          o  o  o  o  o  o  o
#          |  |  |  |  |  |  |
#Source -> o--o--o--o--o--o--o -> Sink
#       advection diff source PDE

@variables t
D = Differential(t)
m_flow_source(t) = 2.75
T_source(t) = (t > 12 * 3600) * 56.0 + 12.0
@register_symbolic m_flow_source(t)
@register_symbolic T_source(t)

#build polynomial liquid-water property only dependent on Temperature
p_l = 5 #bar
T_vec = collect(1:1:150);
@generated kin_visc_T(t) = :(Base.evalpoly(t, $(fit(T_vec, my_pT.(p_l, T_vec) ./ rho_pT.(p_l, T_vec), 5).coeffs...,)))
@generated lambda_T(t) = :(Base.evalpoly(t, $(fit(T_vec, tc_pT.(p_l, T_vec), 3).coeffs...,)))
@generated Pr_T(t) = :(Base.evalpoly(t, $(fit(T_vec, 1e3 * Cp_pT.(p_l, T_vec) .* my_pT.(p_l, T_vec) ./ tc_pT.(p_l, T_vec), 5).coeffs...,)))
@generated rho_T(t) = :(Base.evalpoly(t, $(fit(T_vec, rho_pT.(p_l, T_vec), 4).coeffs...,)))
@generated rhocp_T(t) = :(Base.evalpoly(t, $(fit(T_vec, 1000 * rho_pT.(p_l, T_vec) .* Cp_pT.(p_l, T_vec), 5).coeffs...,)))
@register_symbolic kin_visc_T(t)
@register_symbolic lambda_T(t)
@register_symbolic Pr_T(t)
@register_symbolic rho_T(t)
@register_symbolic rhocp_T(t)

@connector function FluidPort(; name, p=101325.0, m=0.0, T=0.0)
  sts = @variables p(t) = p m(t) = m [connect = Flow] T(t) = T [connect = Stream]
  ODESystem(Equation[], t, sts, []; name=name)
end

@connector function VectorHeatPort(; name, N=100, T0=0.0, Q0=0.0)
  sts = @variables (T(t))[1:N] = T0 (Q(t))[1:N] = Q0 [connect = Flow]
  ODESystem(Equation[], t, [T; Q], []; name=name)
end

@register_symbolic Dxx_coeff(u, d, T)
#Taylor-aris dispersion model
function Dxx_coeff(u, d, T)
  Re = abs(u) * d / kin_visc_T(T) + 0.1
  if Re < 1000.0
    (d^2 / 4) * u^2 / 48 / 0.14e-6
  else
    d * u * (1.17e9 * Re^(-2.5) + 0.41)
  end
end

@register_symbolic Nusselt(Re, Pr, f)
#Nusselt number model
function Nusselt(Re, Pr, f)
  if Re <= 2300.0
    3.66
  elseif Re <= 3100.0
    3.5239 * (Re / 1000)^4 - 45.158 * (Re / 1000)^3 + 212.13 * (Re / 1000)^2 - 427.45 * (Re / 1000) + 316.08
  else
    f / 8 * ((Re - 1000) * Pr) / (1 + 12.7 * (f / 8)^(1 / 2) * (Pr^(2 / 3) - 1))
  end
end

@register_symbolic Churchill_f(Re, epsilon, d)
#Darcy weisbach friction factor
function Churchill_f(Re, epsilon, d)
  theta_1 = (-2.457 * log(((7 / Re)^0.9) + (0.27 * (epsilon / d))))^16
  theta_2 = (37530 / Re)^16
  8 * ((((8 / Re)^12) + (1 / ((theta_1 + theta_2)^1.5)))^(1 / 12))
end

function FluidRegion(; name, L=1.0, dn=0.05, N=100, T0=0.0,
  lumped_T=50, diffusion=true, e=1e-4)
  @named inlet = FluidPort()
  @named outlet = FluidPort()
  @named heatport = VectorHeatPort(N=N)

  dx = L / N
  c = [-1 / 8, -3 / 8, -3 / 8] # advection stencil coefficients
  A = pi * dn^2 / 4

  p = @parameters C_shift = 0.0 Rw = 0.0 # stuff for latter
  @variables begin
    (T(t))[1:N] = fill(T0, N)
    Twall(t)[1:N] = fill(T0, N)
    (S(t))[1:N] = fill(T0, N)
    (C(t))[1:N] = fill(1.0, N)
    u(t) = 1e-6
    Re(t) = 1000.0
    Dxx(t) = 0.0
    Pr(t) = 1.0
    alpha(t) = 1.0
    f(t) = 1.0
  end

  sts = vcat(T, Twall, S, C, Num[u], Num[Re], Num[Dxx], Num[Pr], Num[alpha], Num[f])

  eqs = Equation[
    Re ~ 0.1 + dn * abs(u) / kin_visc_T(lumped_T)
    Pr ~ Pr_T(lumped_T)
    f ~ Churchill_f(Re, e, dn) #Darcy-weisbach
    alpha ~ Nusselt(Re, Pr, f) * lambda_T(lumped_T) / dn
    Dxx ~ diffusion * Dxx_coeff(u, dn, lumped_T)
    inlet.m ~ -outlet.m
    inlet.p ~ outlet.p
    inlet.T ~ instream(inlet.T)
    outlet.T ~ T[N]
    u ~ inlet.m / rho_T(inlet.T) / A
    [C[i] ~ dx * A * rhocp_T(T[i]) for i in 1:N]
    [S[i] ~ heatport.Q[i] for i in 1:N]
    [Twall[i] ~ heatport.T[i] for i in 1:N]

    #source term
    [S[i] ~ (1 / (1 / (alpha * dn * pi * dx) + abs(Rw / 1000))) * (Twall[i] - T[i]) for i in 1:N]

    #second order upwind + diffusion + source
    D(T[1]) ~ u / dx * (inlet.T - T[1]) + Dxx * (T[2] - T[1]) / dx^2 + S[1] / (C[1] - C_shift)
    D(T[2]) ~ u / dx * (c[1] * inlet.T - sum(c) * T[1] + c[2] * T[2] + c[3] * T[3]) + Dxx * (T[1] - 2 * T[2] + T[3]) / dx^2 + S[2] / (C[2] - C_shift)
    [D(T[i]) ~ u / dx * (c[1] * T[i-2] - sum(c) * T[i-1] + c[2] * T[i] + c[3] * T[i+1]) + Dxx * (T[i-1] - 2 * T[i] + T[i+1]) / dx^2 + S[i] / (C[i] - C_shift) for i in 3:N-1]
    D(T[N]) ~ u / dx * (T[N-1] - T[N]) + Dxx * (T[N-1] - T[N]) / dx^2 + S[N] / (C[N] - C_shift)
  ]

  ODESystem(eqs, t, sts, p; systems=[inlet, outlet, heatport], name=name)
end

@register_symbolic Cn_circular_wall_inner(d, D, cp, ρ)
function Cn_circular_wall_inner(d, D, cp, ρ)
  C = pi / 4 * (D^2 - d^2) * cp * ρ
  return C / 2
end

@register_symbolic Cn_circular_wall_outer(d, D, cp, ρ)
function Cn_circular_wall_outer(d, D, cp, ρ)
  C = pi / 4 * (D^2 - d^2) * cp * ρ
  return C / 2
end

@register_symbolic Ke_circular_wall(d, D, λ)
function Ke_circular_wall(d, D, λ)
  2 * pi * λ / log(D / d)
end

function CircularWallFEM(; name, L=100, N=10, d=0.05, t_layer=[0.002],
  λ=[50], cp=[500], ρ=[7850], T0=0.0)
  @named inner_heatport = VectorHeatPort(N=N)
  @named outer_heatport = VectorHeatPort(N=N)
  dx = L / N
  Ne = length(t_layer)
  Nn = Ne + 1
  dn = vcat(d, d .+ 2.0 .* cumsum(t_layer))
  Cn = zeros(Nn)
  Cn[1:Ne] += Cn_circular_wall_inner.(dn[1:Ne], dn[2:Nn], cp, ρ) .* dx
  Cn[2:Nn] += Cn_circular_wall_outer.(dn[1:Ne], dn[2:Nn], cp, ρ) .* dx
  p = @parameters C_shift = 0.0
  Ke = Ke_circular_wall.(dn[1:Ne], dn[2:Nn], λ) .* dx
  @variables begin
    (Tn(t))[1:N, 1:Nn] = fill(T0, N, Nn)
    (Qe(t))[1:N, 1:Ne] = fill(T0, N, Ne)
  end
  sts = [vec(Tn); vec(Qe)]
  e0 = Equation[inner_heatport.T[i] ~ Tn[i, 1] for i in 1:N]
  e1 = Equation[outer_heatport.T[i] ~ Tn[i, Nn] for i in 1:N]
  e2 = Equation[Qe[i, j] ~ Ke[j] * (-Tn[i, j+1] + Tn[i, j]) for i in 1:N for j in 1:Ne]
  e3 = Equation[D(Tn[i, 1]) * (Cn[1] + C_shift) ~ inner_heatport.Q[i] - Qe[i, 1] for i in 1:N]
  e4 = Equation[D(Tn[i, j]) * Cn[j] ~ Qe[i, j-1] - Qe[i, j] for i in 1:N for j in 2:Nn-1]
  e5 = Equation[D(Tn[i, Nn]) * Cn[Nn] ~ Qe[i, Ne] + outer_heatport.Q[i] for i in 1:N]
  eqs = vcat(e0, e1, e2, e3, e4, e5)
  ODESystem(eqs, t, sts, p; systems=[inner_heatport, outer_heatport], name=name)
end

function CylindricalSurfaceConvection(; name, L=100, N=100, d=1.0, α=5.0)
  dx = L / N
  S = pi * d * dx
  @named heatport = VectorHeatPort(N=N)
  sts = @variables Tenv(t) = 0.0
  eqs = [
    Tenv ~ 18.0
    [heatport.Q[i] ~ α * S * (heatport.T[i] - Tenv) for i in 1:N]
  ]

  ODESystem(eqs, t, sts, []; systems=[heatport], name=name)
end

function PreinsulatedPipe(; name, L=100.0, N=100.0, dn=0.05, T0=0.0, t_layer=[0.004, 0.013],
  λ=[50, 0.04], cp=[500, 1200], ρ=[7800, 40], α=5.0,
  e=1e-4, lumped_T=50, diffusion=true)
  @named inlet = FluidPort()
  @named outlet = FluidPort()
  @named fluid_region = FluidRegion(L=L, N=N, dn=dn, e=e, lumped_T=lumped_T, diffusion=diffusion)
  @named shell = CircularWallFEM(L=L, N=N, d=dn, t_layer=t_layer, λ=λ, cp=cp, ρ=ρ)
  @named surfconv = CylindricalSurfaceConvection(L=L, N=N, d=dn + 2.0 * sum(t_layer), α=α)
  systems = [fluid_region, shell, inlet, outlet, surfconv]
  eqs = [
    connect(fluid_region.inlet, inlet)
    connect(fluid_region.outlet, outlet)
    connect(fluid_region.heatport, shell.inner_heatport)
    connect(shell.outer_heatport, surfconv.heatport)
  ]
  ODESystem(eqs, t, [], []; systems=systems, name=name)
end

function Source(; name, p_feed=100000)
  @named outlet = FluidPort()
  sts = @variables m_flow(t) = 1e-6
  eqs = [
    m_flow ~ m_flow_source(t)
    outlet.m ~ -m_flow
    outlet.p ~ p_feed
    outlet.T ~ T_source(t)
  ]
  compose(ODESystem(eqs, t, sts, []; name=name), [outlet])
end

function Sink(; name)
  @named inlet = FluidPort()
  eqs = [
    inlet.T ~ instream(inlet.T)
  ]
  compose(ODESystem(eqs, t, [], []; name=name), [inlet])
end

function TestBenchPreinsulated(; name, L=1.0, dn=0.05, t_layer=[0.0056, 0.013], N=100, diffusion=true, lumped_T=20)
  @named pipe = PreinsulatedPipe(L=L, dn=dn, N=N, diffusion=diffusion, t_layer=t_layer, lumped_T=lumped_T)
  @named source = Source()
  @named sink = Sink()
  subs = [source, pipe, sink]
  eqs = [
    connect(source.outlet, pipe.inlet)
    connect(pipe.outlet, sink.inlet)
  ]
  compose(ODESystem(eqs, t, [], []; name=name), subs)
end

function build_system(fsys, N)
  N >= 4 || throw("Problem sizes smaller than 4 not supported; received $N.")
  @named testbench = TestBenchPreinsulated(; L=470, N, dn=0.3127, t_layer=[0.0056, 0.058])
  t0 = time()
  sys = structural_simplify(fsys(testbench))
  return time() - t0, sys
end

function compile_run_problem(sys; target=JuliaSimCompiler.JuliaTarget(), solver=FBDF(;autodiff = target===JuliaSimCompiler.JuliaTarget()), duref=nothing)
  tspan = (0.0, 19 * 3600.0)
  t0 = time()
  prob = if target === JuliaSimCompiler.JuliaTarget()
    ODEProblem(sys, [], tspan; sparse = true)
  else
    ODEProblem(sys, target, [], tspan; sparse = true)
  end
  (; f, u0, p) = prob
  fill!(u0, 12.0)
  ff = f.f
  du = similar(u0)
  ff(du, u0, p, 0.0)
  t_fode = time() - t0
  duref === nothing || @assert duref ≈ du
  t_run = @belapsed $ff($du, $u0, $p, 0.0)
  t_solve = @elapsed sol = solve(prob, solver; reltol=1e-6, abstol=1e-6, saveat=100)
  @assert SciMLBase.successful_retcode(sol)
  (t_fode, t_run, t_solve), du
end

const C = JuliaSimCompiler.CTarget();
const LLVM = JuliaSimCompiler.llvm.LLVMTarget();

function run_and_time_julia!(ss_times, times, max_sizes, i, N)
  @named testbench = TestBenchPreinsulated(L=470, N=N, dn=0.3127, t_layer=[0.0056, 0.058])
  if N <= max_sizes[1]
    ss_times[i, 1] = @elapsed sys_mtk = structural_simplify(testbench)
    times[i, 1], _ = compile_run_problem(sys_mtk)
  end
  ss_times[i, 2] = @elapsed sys_jsir_scalar = structural_simplify(IRSystem(testbench), loop=false)
  oderef = daeref = nothing
  N <= max_sizes[2] && ((times[i, 2], oderef) = compile_run_problem(sys_jsir_scalar; duref = oderef))
  N <= max_sizes[3] && ((times[i, 3], oderef) = compile_run_problem(sys_jsir_scalar; target=C, duref = oderef))
  N <= max_sizes[4] && ((times[i, 4], oderef) = compile_run_problem(sys_jsir_scalar; target=LLVM, duref = oderef))
  for j = 1:4
    ss_time = ss_times[i, 1 + (j>1)] 
    t_fode, t_run, t_solve = times[i, j]
    total_times[i, j] = ss_time + t_fode + t_solve
  end
end


N = [5, 10, 20, 40, 60, 80, 160, 320, 480, 640, 800, 960, 1280];
N_states = 4 .* N; # x-axis for plots
# max size we test per method
max_sizes = [480, last(N), last(N), last(N), last(N)];
# NaN-initialize so Makie will ignore incomplete
ss_times = fill(NaN, length(N), 2);
times = fill((NaN,NaN,NaN), length(N), length(max_sizes) - 1);
total_times = fill(NaN, length(N), length(max_sizes));


@time run_and_time_julia!(ss_times, times, max_sizes, 1, 4); # precompile
for (i, n) in enumerate(N)
  @time run_and_time_julia!(ss_times, times, max_sizes, i, n)
end


using OMJulia, CSV, DataFrames
mod = OMJulia.OMCSession();
OMJulia.sendExpression(mod, "getVersion()")
OMJulia.sendExpression(mod, "installPackage(Modelica)")
modelicafile = "../../benchmarks/ModelingToolkit/DhnControl.mo"
resultfile = "modelica_res.csv"

@show "Start OpenModelica Timings"

for i in 1:length(N)
    N = N[i]
    N > max_sizes[end] && break
    @show N
    totaltime = @elapsed res = begin
        @sync ModelicaSystem(mod, modelicafile, "DhnControl.Test.test_preinsulated_470_$N")
        sendExpression(mod, "simulate(DhnControl.Test.test_preinsulated_470_$N)")
    end
    #runtime = res["timeTotal"]
    @assert res["messages"][1:11] == "LOG_SUCCESS"
    total_times[i, 5] = totaltime
end

OMJulia.quit(mod)
total_times[:, 5]


translation_and_total_times = [1.802 1.921
                               1.78 1.846
                               1.84 1.877
                               2.028 2.075
                               2.221 2.283
                               2.409 2.496
                               3.189 3.427
                               4.758 5.577
                               6.39 8.128
                               8.052 11.026
                               9.707 14.393
                               11.411 17.752
                               15.094 27.268]
total_times[:, 6] = translation_and_total_times[1:length(N),2]


f = Figure(size=(800,1200));
ss_names = ["MTK", "JSIR-Scalar", "JSIR-Loop"];
let ax = Axis(f[1, 1]; yscale = log10, xscale = log10, title="Structural Simplify Time")
  _lines = map(eachcol(ss_times)) do ts
    lines!(N, ts)
  end
  Legend(f[1,2], _lines, ss_names)
end
method_names = ["MTK", "JSIR - Scalar - Julia", "JSIR - Scalar - C", "JSIR - Scalar - LLVM", "JSIR - Loop - Julia", "JSIR - Loop - C", "JSIR - Loop - LLVM"];
for (i, timecat) in enumerate(("ODEProblem + f!", "Run", "Solve"))
  title = timecat * " Time"
  ax = Axis(f[i+1, 1]; yscale = log10, xscale = log10, title)
  _lines = map(eachcol(times)) do ts
    lines!(N, getindex.(ts, i))
  end
  Legend(f[i+1, 2], _lines, method_names)
end
let method_names_m = vcat(method_names, "OpenModelica");
  ax = Axis(f[5, 1]; yscale = log10, xscale = log10, title = "Total Time")
  _lines = map(Base.Fix1(lines!, N), eachcol(total_times))
  Legend(f[5, 2], _lines, method_names_m)
end
f


f2 = Figure(size = (800, 400));
title = "Total Time: Thermal Fluid Benchmark"
ax = Axis(f2[1, 1]; yscale = log10, xscale = log10, title)
names = ["MTK", "JSIR - Julia", "JSIR - C", "JSIR - LLVM", "OpenModelica", "Dymola"]
_lines = map(enumerate(names)) do (j, label)
    ts = @view(total_times[:, j])
    lines!(N_states, ts)
end
Legend(f2[1,2], _lines, names)
f2


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

