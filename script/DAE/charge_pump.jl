
using OrdinaryDiffEq, DiffEqDevTools, Sundials,
      Plots, DASSL, DASKR
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
import ModelingToolkit: Symbolics, ForwardDiff


const VT0 = 0.20
const GAMMA_MOS = 0.035
const PHI = 1.01
const COX = 4.0e-12
const CAPD = 0.40e-12
const CAPS = 1.60e-12
const VHIGH = 20.0
const DELTAT_PULSE = 120.0e-9
const T1_PULSE = 50.0e-9
const T2_PULSE = 60.0e-9
const T3_PULSE = 110.0e-9

function qgate(vgb::T, vgs::T, vgd::T) where T <: Real
    if (vgs - vgd) <= 0
        ugs = vgd; ugd = vgs
    else
        ugs = vgs; ugd = vgd
    end
    ugb = vgb
    ubs = ugs - ugb
    vfb = VT0 - GAMMA_MOS * sqrt(PHI) - PHI
    phi_ubs = max(PHI - ubs, zero(T))
    vte = VT0 + GAMMA_MOS * (sqrt(phi_ubs) - sqrt(PHI))

    if ugb <= vfb
        # Accumulation region
        return COX * (ugb - vfb)
    elseif ugb > vfb && ugs <= vte
        # Depletion region
        return COX * GAMMA_MOS * (sqrt(max((GAMMA_MOS / 2)^2 + ugb - vfb, zero(T))) - GAMMA_MOS / 2)
    else
        # Inversion region
        ugst = ugs - vte
        ugdt = ugd > vte ? ugd - vte : zero(T)
        denom = ugdt + ugst
        denom = abs(denom) < 1e-30 ? T(1e-30) : denom
        return COX * ((2 / 3) * (ugdt + ugst - (ugdt * ugst) / denom) +
                       GAMMA_MOS * sqrt(phi_ubs))
    end
end
qgate(a, b, c) = qgate(promote(float(a), float(b), float(c))...)

function qsrc(vgb::T, vgs::T, vgd::T) where T <: Real
    if (vgs - vgd) <= 0
        ugs = vgd; ugd = vgs
    else
        ugs = vgs; ugd = vgd
    end
    ugb = vgb
    ubs = ugs - ugb
    vfb = VT0 - GAMMA_MOS * sqrt(PHI) - PHI
    phi_ubs = max(PHI - ubs, zero(T))
    vte = VT0 + GAMMA_MOS * (sqrt(phi_ubs) - sqrt(PHI))

    if ugb <= vfb || (ugb > vfb && ugs <= vte)
        return zero(T)
    else
        ugst = ugs - vte
        ugdt = ugd >= vte ? ugd - vte : zero(T)
        denom = ugdt + ugst
        denom = abs(denom) < 1e-30 ? T(1e-30) : denom
        return -COX * (1 / 3) * (ugdt + ugst - (ugdt * ugst) / denom)
    end
end
qsrc(a, b, c) = qsrc(promote(float(a), float(b), float(c))...)

function qdrain(vgb::T, vgs::T, vgd::T) where T <: Real
    if (vgs - vgd) <= 0
        ugs = vgd; ugd = vgs
    else
        ugs = vgs; ugd = vgd
    end
    ugb = vgb
    ubs = ugs - ugb
    vfb = VT0 - GAMMA_MOS * sqrt(PHI) - PHI
    phi_ubs = max(PHI - ubs, zero(T))
    vte = VT0 + GAMMA_MOS * (sqrt(phi_ubs) - sqrt(PHI))

    if ugb <= vfb || (ugb > vfb && ugs <= vte)
        return zero(T)
    else
        ugst = ugs - vte
        ugdt = ugd >= vte ? ugd - vte : zero(T)
        denom = ugdt + ugst
        denom = abs(denom) < 1e-30 ? T(1e-30) : denom
        return -COX * (1 / 3) * (ugdt + ugst - (ugdt * ugst) / denom)
    end
end
qdrain(a, b, c) = qdrain(promote(float(a), float(b), float(c))...)


function vin(t)
    dummy = mod(t, DELTAT_PULSE)
    if dummy < T1_PULSE
        return 0.0
    elseif dummy < T2_PULSE
        return (dummy - T1_PULSE) * 0.10e9 * VHIGH
    elseif dummy < T3_PULSE
        return VHIGH
    else
        return (DELTAT_PULSE - dummy) * 0.10e9 * VHIGH
    end
end


tspan = (0.0, 1200.0e-9)

disc_times = Float64[]
base_disc = [50.0e-9, 60.0e-9, 110.0e-9, 120.0e-9]
for k in 0:9
    for td in base_disc
        push!(disc_times, td + k * 120.0e-9)
    end
end
disc_times = sort(unique(filter(t -> 0.0 < t < tspan[2], disc_times)))


function dvin(t_val)
    dummy = mod(t_val, DELTAT_PULSE)
    dummy < T1_PULSE ? 0.0 : dummy < T2_PULSE ? 0.10e9 * VHIGH :
    dummy < T3_PULSE ? 0.0 : -0.10e9 * VHIGH
end

@register_symbolic qgate(vgb, vgs, vgd)
@register_symbolic qsrc(vgb, vgs, vgd)
@register_symbolic qdrain(vgb, vgs, vgd)
@register_symbolic vin(t_val)
@register_symbolic dvin(t_val)

# ForwardDiff-based partial derivatives for Pantelides index reduction
for (fn, dfn_prefix) in [(qgate, :dqgate), (qsrc, :dqsrc), (qdrain, :dqdrain)]
    for i in 1:3
        dfn_name = Symbol(dfn_prefix, "_", i)
        @eval begin
            $dfn_name(vgb, vgs, vgd) = ForwardDiff.derivative(
                x -> $fn(ntuple(j -> j == $i ? x : [vgb, vgs, vgd][j], 3)...),
                Float64([vgb, vgs, vgd][$i]))
            @register_symbolic $dfn_name(vgb, vgs, vgd)
            Symbolics.derivative(::typeof($fn), args::NTuple{3, Any}, ::Val{$i}) =
                $dfn_name(args...)
        end
    end
end
Symbolics.derivative(::typeof(vin), args::NTuple{1, Any}, ::Val{1}) = dvin(args...)


@variables begin
    YT1(t) = qgate(0.0, 0.0, 0.0)
    YS(t)  = 0.0
    YT2(t) = qsrc(0.0, 0.0, 0.0)
    YD(t)  = 0.0
    YT3(t) = qdrain(0.0, 0.0, 0.0)
    U1(t)  = 0.0
    U2(t)  = 0.0
    U3(t)  = 0.0
    II(t)  = 0.0
end

eqs = [
    D(YT1) ~ -II,                                              # row 1 (differential)
    D(YS) + D(YT2) ~ 0,                                       # row 2 (differential)
    D(YD) + D(YT3) ~ 0,                                       # row 3 (differential)
    0 ~ -U1 + vin(t),                                          # row 4 (algebraic)
    0 ~ YT1 - qgate(U1, U1 - U2, U1 - U3),                   # row 5 (algebraic)
    0 ~ YS  - CAPS * U2,                                       # row 6 (algebraic)
    0 ~ YT2 - qsrc(U1, U1 - U2, U1 - U3),                    # row 7 (algebraic)
    0 ~ YD  - CAPD * U3,                                       # row 8 (algebraic)
    0 ~ YT3 - qdrain(U1, U1 - U2, U1 - U3),                  # row 9 (algebraic)
]

@mtkbuild sys = ODESystem(eqs, t)


println("MTK index reduction: $(9) original → $(length(unknowns(sys))) unknowns")
println("States: ", unknowns(sys))


mtkprob = ODEProblem(sys, [], tspan)
mtk_test = solve(mtkprob, Rodas5P(autodiff = false), abstol = 1e-4, reltol = 1e-4,
                 tstops = disc_times, maxiters = Int(1e6), dt = 1e-15)
println("Rodas5P on MTK-reduced system: retcode = $(mtk_test.retcode), ",
        "steps = $(length(mtk_test.t)), final t = $(mtk_test.t[end])")


function charge_pump_rhs!(du, u, p, t)
    y1, y2, y3, y4, y5, y6, y7, y8, y9 = u
    v = vin(t)

    du[1] = -y9
    du[2] = 0.0
    du[3] = 0.0
    du[4] = -y6 + v
    du[5] = y1 - qgate(y6, y6 - y7, y6 - y8)
    du[6] = y2 - CAPS * y7
    du[7] = y3 - qsrc(y6, y6 - y7, y6 - y8)
    du[8] = y4 - CAPD * y8
    du[9] = y5 - qdrain(y6, y6 - y7, y6 - y8)
    nothing
end

M = zeros(9, 9)
M[1, 1] = 1.0
M[2, 2] = 1.0
M[2, 3] = 1.0
M[3, 4] = 1.0
M[3, 5] = 1.0

# Consistent initial conditions from Fortran init subroutine
y0 = zeros(9)
y0[1] = qgate(0.0, 0.0, 0.0)
y0[3] = qsrc(0.0, 0.0, 0.0)
y0[5] = qdrain(0.0, 0.0, 0.0)

mmf = ODEFunction(charge_pump_rhs!, mass_matrix = M)
mmprob = ODEProblem(mmf, y0, tspan)


function charge_pump_dae!(out, du, u, p, t)
    y1, y2, y3, y4, y5, y6, y7, y8, y9 = u
    dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9 = du
    v = vin(t)

    # Differential equations (rows 1–3)
    out[1] = dy1 + y9                                          # Y'_T1 = -I
    out[2] = dy2 + dy3                                         # Y'_S + Y'_T2 = 0
    out[3] = dy4 + dy5                                         # Y'_D + Y'_T3 = 0

    # Algebraic constraints (rows 4–9)
    out[4] = y6 - v                                            # U_1 = V_in(t)
    out[5] = -(y1 - qgate(y6, y6 - y7, y6 - y8))             # Y_T1 = Q_G(U_1, U_1-U_2, U_1-U_3)
    out[6] = -(y2 - CAPS * y7)                                 # Y_S = C_S · U_2
    out[7] = -(y3 - qsrc(y6, y6 - y7, y6 - y8))              # Y_T2 = Q_S(U_1, U_1-U_2, U_1-U_3)
    out[8] = -(y4 - CAPD * y8)                                 # Y_D = C_D · U_3
    out[9] = -(y5 - qdrain(y6, y6 - y7, y6 - y8))            # Y_T3 = Q_D(U_1, U_1-U_2, U_1-U_3)
    nothing
end

du0 = zeros(9)
differential_vars = [true, true, true, true, true, false, false, false, false]
daeprob = DAEProblem(charge_pump_dae!, du0, y0, tspan,
                     differential_vars = differential_vars)


ref_sol = solve(daeprob, IDA(), abstol = 5e-4, reltol = 5e-4,
                dt = 1e-15, tstops = disc_times, maxiters = Int(1e7), dense = true)
@assert ref_sol.retcode == ReturnCode.Success "Reference solve failed: $(ref_sol.retcode)"

probs = [daeprob]
refs  = [ref_sol]


y_ref = zeros(9)
y_ref[1] = 0.126280042987675933170893e-12
y_ref[9] = 0.152255686815577679043511e-3

println("=== Reference Solution Verification (IDA, tol = 5e-4) ===")
for i in [1, 9]
    computed = ref_sol.u[end][i]
    ref_val  = y_ref[i]
    rel_err  = abs(ref_val) > 0 ? abs(computed - ref_val) / abs(ref_val) : abs(computed)
    println("  y[$i]: computed = $computed,  GAMD ref = $ref_val,  rel_error = $rel_err")
end


p1 = plot(ref_sol, idxs = [6], title = "U₁ (node 1 potential)",
          xlabel = "t [s]", ylabel = "V", legend = false)
p2 = plot(ref_sol, idxs = [7], title = "U₂ (node 2 potential)",
          xlabel = "t [s]", ylabel = "V", legend = false)
p3 = plot(ref_sol, idxs = [8], title = "U₃ (node 3 potential)",
          xlabel = "t [s]", ylabel = "V", legend = false)
p4 = plot(ref_sol, idxs = [9], title = "I (current)",
          xlabel = "t [s]", ylabel = "A", legend = false)
plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600),
     plot_title = "Charge Pump — DAE Reference Solution")


plot(ref_sol, idxs = [1],
     title = "Y_T1 (gate charge)",
     xlabel = "t [s]", ylabel = "C", legend = false)


abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3)

setups = [
    Dict(:prob_choice => 1, :alg => IDA()),
    Dict(:prob_choice => 1, :alg => DASKR.daskr()),
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
    save_everystep = false, appxsol = refs, maxiters = Int(1e5), numruns = 1,
    names = ["IDA", "DASKR"], dt = 1e-15, tstops = disc_times)
plot(wp, title = "Charge Pump DAE — Loose Tolerances (Final Value)")


abstols = 1.0 ./ 10.0 .^ (1:3)
reltols = 1.0 ./ 10.0 .^ (1:3)

setups = [
    Dict(:prob_choice => 1, :alg => IDA()),
    Dict(:prob_choice => 1, :alg => DASKR.daskr()),
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups; error_estimate = :l2,
    save_everystep = false, appxsol = refs, maxiters = Int(1e5), numruns = 1,
    names = ["IDA", "DASKR"], dt = 1e-15, tstops = disc_times)
plot(wp, title = "Charge Pump DAE — Loose Tolerances (L₂ Timeseries)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

