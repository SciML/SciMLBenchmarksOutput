
using OrdinaryDiffEq, DiffEqDevTools, ModelingToolkit, ODEInterfaceDiffEq,
      Plots, Sundials, DASSL, DASKR
using LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D


const RGS = 4.0
const RGD = 4.0
const RBS = 10.0
const RBD = 10.0
const CGS = 6e-5
const CGD = 6e-5
const CBD = 2.4e-5
const CBS = 2.4e-5
const C9 = 5e-5
const DELTA = 0.02
const CURIS = 1e-14
const VTH = 25.85
const VDD = 5.0
const VBB = -2.5
const VT0_DEPL = -2.43
const CGAMMA_DEPL = 0.2
const PHI_DEPL = 1.28
const BETA_DEPL = 5.35e-4
const VT0_ENH = 0.2
const CGAMMA_ENH = 0.035
const PHI_ENH = 1.01
const BETA_ENH = 1.748e-3


function pulse(t, t_start, v_low, t_rise, v_high, t_high, t_fall, t_period)
    t_mod = mod(t, t_period)
    
    if t_mod < t_start
        return v_low
    elseif t_mod < t_start + t_rise
        return v_low + (v_high - v_low) * (t_mod - t_start) / t_rise
    elseif t_mod < t_start + t_rise + t_high
        return v_high
    elseif t_mod < t_start + t_rise + t_high + t_fall
        return v_high - (v_high - v_low) * (t_mod - t_start - t_rise - t_high) / t_fall
    else
        return v_low
    end
end

V1(t) = pulse(t, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 20.0)
V2(t) = pulse(t, 0.0, 0.0, 15.0, 5.0, 15.0, 5.0, 40.0)

function V1_derivative(t)
    t_mod = mod(t, 20.0)
    if 0.0 < t_mod < 5.0
        return 1.0
    elseif 10.0 < t_mod < 15.0
        return -1.0
    else
        return 0.0
    end
end
function V2_derivative(t)
    t_mod = mod(t, 40.0)
    if 0.0 < t_mod < 15.0
        return 1.0/15.0
    elseif 20.0 < t_mod < 35.0
        return -1.0/15.0
    else
        return 0.0
    end
end


function gdsp(ned, vds, vgs, vbs)
    if ned == 1
        vt0, cgamma, phi, beta = VT0_DEPL, CGAMMA_DEPL, PHI_DEPL, BETA_DEPL
    else
        vt0, cgamma, phi, beta = VT0_ENH, CGAMMA_ENH, PHI_ENH, BETA_ENH
    end
    phi_vbs = max(phi - vbs, 1e-12)
    phi_safe = max(phi, 1e-12)
    vte = vt0 + cgamma * (sqrt(phi_vbs) - sqrt(phi_safe))
    if vgs - vte <= 0.0
        return 0.0
    elseif 0.0 < vgs - vte <= vds
        return -beta * (vgs - vte)^2 * (1.0 + DELTA * vds)
    elseif 0.0 < vds < vgs - vte
        return -beta * vds * (2.0 * (vgs - vte) - vds) * (1.0 + DELTA * vds)
    else
        return 0.0
    end
end

function gdsm(ned, vds, vgd, vbd)
    if ned == 1
        vt0, cgamma, phi, beta = VT0_DEPL, CGAMMA_DEPL, PHI_DEPL, BETA_DEPL
    else
        vt0, cgamma, phi, beta = VT0_ENH, CGAMMA_ENH, PHI_ENH, BETA_ENH
    end
    phi_vbd = max(phi - vbd, 1e-12)
    phi_safe = max(phi, 1e-12)
    vte = vt0 + cgamma * (sqrt(phi_vbd) - sqrt(phi_safe))
    if vgd - vte <= 0.0
        return 0.0
    elseif 0.0 < vgd - vte <= -vds
        return beta * (vgd - vte)^2 * (1.0 - DELTA * vds)
    elseif 0.0 < -vds < vgd - vte
        return -beta * vds * (2.0 * (vgd - vte) + vds) * (1.0 - DELTA * vds)
    else
        return 0.0
    end
end

function ids(ned, vds, vgs, vbs, vgd, vbd)
    if vds > 0.0
        return gdsp(ned, vds, vgs, vbs)
    elseif vds == 0.0
        return 0.0
    else
        return gdsm(ned, vds, vgd, vbd)
    end
end

function ibs(vbs)
    if vbs <= 0.0
        return -CURIS * (exp(vbs / VTH) - 1.0)
    else
        return 0.0
    end
end

function ibd(vbd)
    if vbd <= 0.0
        return -CURIS * (exp(vbd / VTH) - 1.0)
    else
        return 0.0
    end
end


function nand_rhs!(f, y, p, t)
    v1 = V1(t)
    v2 = V2(t)
    v1d = V1_derivative(t)
    v2d = V2_derivative(t)
    
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14 = y
    
    f[1] = -(y1 - y5) / RGS - ids(1, y2 - y1, y5 - y1, y3 - y5, y5 - y2, y4 - VDD)
    f[2] = -(y2 - VDD) / RGD + ids(1, y2 - y1, y5 - y1, y3 - y5, y5 - y2, y4 - VDD)
    f[3] = -(y3 - VBB) / RBS + ibs(y3 - y5)
    f[4] = -(y4 - VBB) / RBD + ibd(y4 - VDD)
    f[5] = -(y5 - y1) / RGS - ibs(y3 - y5) - (y5 - y7) / RGD - ibd(y9 - y5)
    
    f[6] = CGS * v1d - (y6 - y10) / RGS - ids(2, y7 - y6, v1 - y6, y8 - y10, v1 - y7, y9 - y5)
    f[7] = CGD * v1d - (y7 - y5) / RGD + ids(2, y7 - y6, v1 - y6, y8 - y10, v1 - y7, y9 - y5)
    f[8] = -(y8 - VBB) / RBS + ibs(y8 - y10)
    f[9] = -(y9 - VBB) / RBD + ibd(y9 - y5)
    f[10] = -(y10 - y6) / RGS - ibs(y8 - y10) - (y10 - y12) / RGD - ibd(y14 - y10)
    
    f[11] = CGS * v2d - y11 / RGS - ids(2, y12 - y11, v2 - y11, y13, v2 - y12, y14 - y10)
    f[12] = CGD * v2d - (y12 - y10) / RGD + ids(2, y12 - y11, v2 - y11, y13, v2 - y12, y14 - y10)
    f[13] = -(y13 - VBB) / RBS + ibs(y13)
    f[14] = -(y14 - VBB) / RBD + ibd(y14 - y10)
    
    return nothing
end

# Mass matrix (singular is fine!)
dirMassMatrix = [
    CGS 0   0   0   0   0   0   0   0   0   0   0   0   0
    0   CGD 0   0   0   0   0   0   0   0   0   0   0   0
    0   0   CBS 0   0   0   0   0   0   0   0   0   0   0
    0   0   0   CBD 0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   CGS 0   0   0   0   0   0   0   0
    0   0   0   0   0   0   CGD 0   0   0   0   0   0   0
    0   0   0   0   0   0   0   CBS 0   0   0   0   0   0
    0   0   0   0   0   0   0   0   CBD 0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   CGS 0   0   0
    0   0   0   0   0   0   0   0   0   0   0   CGD 0   0
    0   0   0   0   0   0   0   0   0   0   0   0   CBS 0
    0   0   0   0   0   0   0   0   0   0   0   0   0   CBD
]

# Initial conditions
y0 = [5.0, 5.0, VBB, VBB, 5.0, 3.62385, 5.0, VBB, VBB, 3.62385, 0.0, 3.62385, VBB, VBB]
tspan = (0.0, 80.0)

# Mass matrix problem (original approach)
mmf = ODEFunction(nand_rhs!, mass_matrix=dirMassMatrix)
mmprob = ODEProblem(mmf, y0, tspan)

# DAEProblem version using direct DAE formulation
function nand_dae!(out, du, u, p, t)
    v1 = V1(t)
    v2 = V2(t)
    v1d = V1_derivative(t)
    v2d = V2_derivative(t)
    
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14 = u
    dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13, dy14 = du
    
    # Differential equations: M*dy/dt - f = 0
    # Convert from mass matrix form: M*dy/dt = f  =>  M*dy/dt - f = 0
    out[1] = CGS * dy1 - (-(y1 - y5) / RGS - ids(1, y2 - y1, y5 - y1, y3 - y5, y5 - y2, y4 - VDD))
    out[2] = CGD * dy2 - (-(y2 - VDD) / RGD + ids(1, y2 - y1, y5 - y1, y3 - y5, y5 - y2, y4 - VDD))
    out[3] = CBS * dy3 - (-(y3 - VBB) / RBS + ibs(y3 - y5))
    out[4] = CBD * dy4 - (-(y4 - VBB) / RBD + ibd(y4 - VDD))
    
    # Algebraic equations: g(y) = 0
    out[5] = -(y5 - y1) / RGS - ibs(y3 - y5) - (y5 - y7) / RGD - ibd(y9 - y5)
    
    out[6] = CGS * dy6 - (CGS * v1d - (y6 - y10) / RGS - ids(2, y7 - y6, v1 - y6, y8 - y10, v1 - y7, y9 - y5))
    out[7] = CGD * dy7 - (CGD * v1d - (y7 - y5) / RGD + ids(2, y7 - y6, v1 - y6, y8 - y10, v1 - y7, y9 - y5))
    out[8] = CBS * dy8 - (-(y8 - VBB) / RBS + ibs(y8 - y10))
    out[9] = CBD * dy9 - (-(y9 - VBB) / RBD + ibd(y9 - y5))
    
    # Algebraic equation: g(y) = 0
    out[10] = -(y10 - y6) / RGS - ibs(y8 - y10) - (y10 - y12) / RGD - ibd(y14 - y10)
    
    out[11] = CGS * dy11 - (CGS * v2d - y11 / RGS - ids(2, y12 - y11, v2 - y11, y13, v2 - y12, y14 - y10))
    out[12] = CGD * dy12 - (CGD * v2d - (y12 - y10) / RGD + ids(2, y12 - y11, v2 - y11, y13, v2 - y12, y14 - y10))
    out[13] = CBS * dy13 - (-(y13 - VBB) / RBS + ibs(y13))
    out[14] = CBD * dy14 - (-(y14 - VBB) / RBD + ibd(y14 - y10))
    
    return nothing
end

# Create DAE problem with automatic initialization
# Let IDA determine consistent initial derivatives automatically
du0_dae = zeros(14)
daeprob = DAEProblem(nand_dae!, du0_dae, y0, tspan)

# Generate reference solutions
ref_sol = solve(mmprob, Rodas5P(), abstol=1e-12, reltol=1e-12, tstops=0.0:5.0:80.0)
dae_ref_sol = solve(daeprob, DASKR.daskr(), abstol=1e-10, reltol=1e-10)

probs = [mmprob, daeprob]
refs = [ref_sol, dae_ref_sol]


plot(ref_sol, title="NAND Gate Circuit - Node Potentials (Mass Matrix)", 
     xlabel="Time", ylabel="Voltage (V)", legend=:outertopright)


plot(dae_ref_sol, title="NAND Gate Circuit - Node Potentials (DAE)", 
     xlabel="Time", ylabel="Voltage (V)", legend=:outertopright)


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 1, :alg=>radau()),
    Dict(:prob_choice => 1, :alg=>RadauIIA5()),
    Dict(:prob_choice => 2, :alg=>IDA()),
    Dict(:prob_choice => 2, :alg=>DASKR.daskr())
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Work-Precision (High Tolerances)")


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (2:4)

setups = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 2, :alg=>IDA()),
    Dict(:prob_choice => 2, :alg=>DASKR.daskr())
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Work-Precision (Medium Tolerances)")


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 1, :alg=>radau()),
    Dict(:prob_choice => 1, :alg=>RadauIIA5()),
    Dict(:prob_choice => 2, :alg=>IDA())
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups; error_estimate=:l2,
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Timeseries Errors (High Tolerances)")


abstols = 1.0 ./ 10.0 .^ (6:8)
reltols = 1.0 ./ 10.0 .^ (2:4)

setups = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 2, :alg=>IDA()),
    Dict(:prob_choice => 2, :alg=>DASKR.daskr())
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups; error_estimate=:l2,
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Timeseries Errors (Medium Tolerances)")


abstols = 1.0 ./ 10.0 .^ (7:12)
reltols = 1.0 ./ 10.0 .^ (4:9)

setups = [
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 1, :alg=>radau()),
    Dict(:prob_choice => 1, :alg=>RadauIIA5()),
    Dict(:prob_choice => 2, :alg=>IDA()),
    Dict(:prob_choice => 2, :alg=>DASKR.daskr())
]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Work-Precision (Low Tolerances)")


wp = WorkPrecisionSet(probs, abstols, reltols, setups; error_estimate=:l2,
                      save_everystep=false, appxsol=refs, 
                      maxiters=Int(1e5), numruns=10,
                      tstops=0.0:5.0:80.0)
plot(wp, title="NAND Gate DAE - Timeseries Errors (Low Tolerances)")


# Original 14-variable system: y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14
key_nodes = [1, 5, 6, 10, 11, 12]  # Representative nodes from different parts of the circuit
node_names = ["Node 1", "Node 5", "Node 6", "Node 10", "Node 11", "Node 12"]

p_nodes = plot()
for (i, node) in enumerate(key_nodes)
    plot!(ref_sol.t, [u[node] for u in ref_sol.u], 
          label=node_names[i], linewidth=2)
end
plot!(p_nodes, title="NAND Gate - Key Node Potentials", 
      xlabel="Time (s)", ylabel="Voltage (V)", legend=:outertopright)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

