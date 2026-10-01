
using OrdinaryDiffEq, DiffEqParamEstim, Optimization, ForwardDiff
using OptimizationBBO, OptimizationNLopt, Plots, BenchmarkTools, DataFrames
using ParallelParticleSwarms
gr(fmt = :png)


function chassagnole!(du, u, p, t)
    cdhap, ce4p, cf6p, cfdp, cg1p, cg6p, cgap, cpep, cpg, cpg2, cpg3, cpgp,
    cpyr, crib5p, cribu5p, csed7p, cxyl5p, cglcex = u

    kALDOdhap, kALDOeq, kALDOfdp, kALDOgap, kALDOgapinh, KDAHPSe4p, KDAHPSpep,
    KENOeq, KENOpep, KENOpg2, KG1PATatp, KG1PATfdp, KG1PATg1p, KG3PDHdhap,
    KG6PDHg6p, KG6PDHnadp, KG6PDHnadphg6pinh, KG6PDHnadphnadpinh, KGAPDHeq,
    KGAPDHgap, KGAPDHnad, KGAPDHnadh, KGAPDHpgp, KPDHpyr, KpepCxylasefdp,
    KpepCxylasepep, KPFKadpa, KPFKadpb, KPFKadpc, KPFKampa, KPFKampb, KPFKatps,
    KPFKf6ps, KPFKpep, KPGDHatpinh, KPGDHnadp, KPGDHnadphinh, KPGDHpg, KPGIeq,
    KPGIf6p, KPGIf6ppginh, KPGIg6p, KPGIg6ppginh, KPGKadp, KPGKatp, KPGKeq,
    KPGKpg3, KPGKpgp, KPGluMueq, KPGluMupg2, KPGluMupg3, KPGMeq, KPGMg1p,
    KPGMg6p, KPKadp, KPKamp, KPKatp, KPKfdp, KPKpep, KPTSa1, KPTSa2, KPTSa3,
    KPTSg6p, KR5PIeq, KRPPKrib5p, KRu5Peq, KSerSynthpg3, KSynth1pep,
    KSynth2pyr, KTAeq, kTISdhap, kTISeq, kTISgap, KTKaeq, KTKbeq, LPFK, LPK,
    nDAHPSe4p, nDAHPSpep, nG1PATfdp, nPDH, npepCxylasefdp, nPFK, nPK,
    nPTSg6p, rmaxALDO, rmaxDAHPS, rmaxENO, rmaxG1PAT, rmaxG3PDH, rmaxG6PDH,
    rmaxGAPDH, rmaxMetSynth, rmaxMurSynth, rmaxPDH, rmaxpepCxylase, rmaxPFK,
    rmaxPGDH, rmaxPGI, rmaxPGK, rmaxPGluMu, rmaxPGM, rmaxPK, rmaxPTS,
    rmaxR5PI, rmaxRPPK, rmaxRu5P, rmaxSerSynth, rmaxSynth1, rmaxSynth2,
    rmaxTA, rmaxTIS, rmaxTKa, rmaxTKb, rmaxTrpSynth, VALDOblf = p

    # known (fixed, not estimated) parameters
    cfeed = 110.96
    Dil = 2.78e-05
    mu = 2.78e-05
    cytosol = 1.0
    extracellular = 1.0

    # empirical time-varying cofactor concentrations
    cadp = 0.582 + 1.73 * 2.731^(-0.15 * t) * (0.12 * t + 0.000214 * t^3)
    camp = 0.123 + 7.25 * (t / (7.25 + 1.47 * t + 0.17 * t^2)) + 1.073 / (1.29 + 8.05 * t)
    catp = 4.27 - 4.163 * (t / (0.657 + 1.43 * t + 0.0364 * t^2))
    cnad = 1.314 + 1.314 * 2.73^(-0.0435 * t - 0.342) -
           (t + 7.871) * (2.73^(-0.0218 * t - 0.171) / (8.481 + t))
    cnadh = 0.0934 + 0.00111 * 2.371^(-0.123 * t) * (0.844 * t + 0.104 * t^3)
    cnadp = 0.159 - 0.00554 * (t / (2.8 - 0.271 * t + 0.01 * t^2)) + 0.182 / (4.82 + 0.526 * t)
    cnadph = 0.062 +
             0.332 * 2.718^(-0.464 * t) *
             (0.0166 * t^1.58 + 0.000166 * t^4.73 + 0.1312e-9 * t^7.89 +
              0.1362e-12 * t^11 + 0.1233e-15 * t^14.2)

    vALDO = cytosol * rmaxALDO * (cfdp - cgap * cdhap / kALDOeq) /
            (kALDOfdp + cfdp + kALDOgap * cdhap / (kALDOeq * VALDOblf) +
             kALDOdhap * cgap / (kALDOeq * VALDOblf) + cfdp * cgap / kALDOgapinh +
             cgap * cdhap / (VALDOblf * kALDOeq))
    vDAHPS = cytosol * rmaxDAHPS * ce4p^nDAHPSe4p * cpep^nDAHPSpep /
             ((KDAHPSe4p + ce4p^nDAHPSe4p) * (KDAHPSpep + cpep^nDAHPSpep))
    vDHAP = cytosol * mu * cdhap
    vE4P = cytosol * mu * ce4p
    vENO = cytosol * rmaxENO * (cpg2 - cpep / KENOeq) / (KENOpg2 * (1 + cpep / KENOpep) + cpg2)
    vEXTER = extracellular * Dil * (cfeed - cglcex)
    vG1PAT = cytosol * rmaxG1PAT * cg1p * catp * (1 + (cfdp / KG1PATfdp)^nG1PATfdp) /
             ((KG1PATatp + catp) * (KG1PATg1p + cg1p))
    vG3PDH = cytosol * rmaxG3PDH * cdhap / (KG3PDHdhap + cdhap)
    vG6P = cytosol * mu * cg6p
    vG6PDH = cytosol * rmaxG6PDH * cg6p * cnadp /
             ((cg6p + KG6PDHg6p) * (1 + cnadph / KG6PDHnadphg6pinh) *
              (KG6PDHnadp * (1 + cnadph / KG6PDHnadphnadpinh) + cnadp))
    vGAP = cytosol * mu * cgap
    vGAPDH = cytosol * rmaxGAPDH * (cgap * cnad - cpgp * cnadh / KGAPDHeq) /
             ((KGAPDHgap * (1 + cpgp / KGAPDHpgp) + cgap) *
              (KGAPDHnad * (1 + cnadh / KGAPDHnadh) + cnad))
    vGLP = cytosol * mu * cg1p
    vMURSyNTH = cytosol * rmaxMurSynth
    vMethSynth = cytosol * rmaxMetSynth
    vPDH = cytosol * rmaxPDH * cpyr^nPDH / (KPDHpyr + cpyr^nPDH)
    vPEP = cytosol * mu * cpep
    vPFK = cytosol * rmaxPFK * catp * cf6p /
           ((catp + KPFKatps * (1 + cadp / KPFKadpc)) *
            (cf6p + KPFKf6ps * (1 + cpep / KPFKpep + cadp / KPFKadpb + camp / KPFKampb) /
                    (1 + cadp / KPFKadpa + camp / KPFKampa)) *
            (1 + LPFK / (1 + cf6p * (1 + cadp / KPFKadpa + camp / KPFKampa) /
                              (KPFKf6ps * (1 + cpep / KPFKpep + cadp / KPFKadpb + camp / KPFKampb)))^nPFK))
    vPG = cytosol * mu * cpg
    vPG3 = cytosol * mu * cpg3
    vPGDH = cytosol * rmaxPGDH * cpg * cnadp /
            ((cpg + KPGDHpg) * (cnadp + KPGDHnadp * (1 + cnadph / KPGDHnadphinh) * (1 + catp / KPGDHatpinh)))
    vPGI = cytosol * rmaxPGI * (cg6p - cf6p / KPGIeq) /
           (KPGIg6p * (1 + cf6p / (KPGIf6p * (1 + cpg / KPGIf6ppginh)) + cpg / KPGIg6ppginh) + cg6p)
    vPGK = cytosol * rmaxPGK * (cadp * cpgp - catp * cpg3 / KPGKeq) /
           ((KPGKadp * (1 + catp / KPGKatp) + cadp) * (KPGKpgp * (1 + cpg3 / KPGKpg3) + cpgp))
    vPGM = cytosol * rmaxPGM * (cg6p - cg1p / KPGMeq) / (KPGMg6p * (1 + cg1p / KPGMg1p) + cg6p)
    vPGP = cytosol * mu * cpgp
    vPK = cytosol * rmaxPK * cpep * (cpep / KPKpep + 1)^(nPK - 1) * cadp /
          (KPKpep * (LPK * ((1 + catp / KPKatp) / (cfdp / KPKfdp + camp / KPKamp + 1))^nPK +
                     (cpep / KPKpep + 1)^nPK) * (cadp + KPKadp))
    vPPK = cytosol * rmaxRPPK * crib5p / (KRPPKrib5p + crib5p)
    vPTS = extracellular * rmaxPTS * cglcex * (cpep / cpyr) /
           ((KPTSa1 + KPTSa2 * (cpep / cpyr) + KPTSa3 * cglcex + cglcex * (cpep / cpyr)) *
            (1 + cg6p^nPTSg6p / KPTSg6p))
    vR5PI = cytosol * rmaxR5PI * (cribu5p - crib5p / KR5PIeq)
    vRIB5P = cytosol * mu * crib5p
    vRibu5p = cytosol * mu * cribu5p
    vRu5P = cytosol * rmaxRu5P * (cribu5p - cxyl5p / KRu5Peq)
    vSED7P = cytosol * mu * csed7p
    vSynth1 = cytosol * rmaxSynth1 * cpep / (KSynth1pep + cpep)
    vSynth2 = cytosol * rmaxSynth2 * cpyr / (KSynth2pyr + cpyr)
    vTA = cytosol * rmaxTA * (cgap * csed7p - ce4p * cf6p / KTAeq)
    vTIS = cytosol * rmaxTIS * (cdhap - cgap / kTISeq) / (kTISdhap * (1 + cgap / kTISgap) + cdhap)
    vTKA = cytosol * rmaxTKa * (crib5p * cxyl5p - csed7p * cgap / KTKaeq)
    vTKB = cytosol * rmaxTKb * (cxyl5p * ce4p - cf6p * cgap / KTKbeq)
    vTRPSYNTH = cytosol * rmaxTrpSynth
    vXYL5P = cytosol * mu * cxyl5p
    vf6P = cytosol * mu * cf6p
    vfdP = cytosol * mu * cfdp
    vpepCxylase = cytosol * rmaxpepCxylase * cpep * (1 + (cfdp / KpepCxylasefdp)^npepCxylasefdp) /
                  (KpepCxylasepep + cpep)
    vpg2 = cytosol * mu * cpg2
    vpyr = cytosol * mu * cpyr
    vrpGluMu = cytosol * rmaxPGluMu * (cpg3 - cpg2 / KPGluMueq) /
               (KPGluMupg3 * (1 + cpg2 / KPGluMupg2) + cpg3)
    vsersynth = cytosol * rmaxSerSynth * cpg3 / (KSerSynthpg3 + cpg3)

    du[1] = (vALDO - vDHAP - vG3PDH - vTIS) / cytosol
    du[2] = (-vDAHPS - vE4P + vTA - vTKB) / cytosol
    du[3] = (-2.0 * vMURSyNTH - vPFK + vPGI + vTA + vTKB - vf6P) / cytosol
    du[4] = (-vALDO + vPFK - vfdP) / cytosol
    du[5] = (-vG1PAT - vGLP + vPGM) / cytosol
    du[6] = (-vG6P - vG6PDH - vPGI - vPGM + 65.0 * vPTS) / cytosol
    du[7] = (vALDO - vGAP - vGAPDH - vTA + vTIS + vTKA + vTKB + vTRPSYNTH) / cytosol
    du[8] = (-vDAHPS + vENO - vPEP - vPK - 65.0 * vPTS - vSynth1 - vpepCxylase) / cytosol
    du[9] = (vG6PDH - vPG - vPGDH) / cytosol
    du[10] = (-vENO - vpg2 + vrpGluMu) / cytosol
    du[11] = (-vPG3 + vPGK - vrpGluMu - vsersynth) / cytosol
    du[12] = (vGAPDH - vPGK - vPGP) / cytosol
    du[13] = (vMethSynth - vPDH + vPK + 65.0 * vPTS - vSynth2 + vTRPSYNTH - vpyr) / cytosol
    du[14] = (-vPPK + vR5PI - vRIB5P - vTKA) / cytosol
    du[15] = (vPGDH - vR5PI - vRibu5p - vRu5P) / cytosol
    du[16] = (-vSED7P - vTA + vTKA) / cytosol
    du[17] = (vRu5P - vTKA - vTKB - vXYL5P) / cytosol
    du[18] = (vEXTER - vPTS) / extracellular
    nothing
end


p_nom = [
    0.088, 0.144, 1.75, 0.088, 0.6, 0.035, 0.0053, 6.73, 0.135, 0.1, 4.42,
    0.119, 3.2, 1.0, 14.4, 0.0246, 6.43, 0.01, 0.63, 0.683, 0.252, 1.09,
    1.04e-5, 1159.0, 0.7, 4.07, 128.0, 3.89, 4.14, 19.1, 3.2, 0.123, 0.325,
    3.26, 208.0, 0.0506, 0.0138, 37.5, 0.1725, 0.266, 0.2, 2.9, 0.2, 0.185,
    0.653, 1934.4, 0.473, 0.0468, 0.188, 0.369, 0.2, 0.196, 0.0136, 1.038,
    0.26, 0.2, 22.5, 0.19, 0.31, 3082.3, 0.01, 245.3, 2.15, 4.0, 0.1, 1.4,
    1.0, 1.0, 1.0, 1.05, 2.8, 1.39, 0.3, 1.2, 10.0, 5.62907e6, 1000.0, 2.6,
    2.2, 1.2, 3.68, 4.21, 11.1, 4.0, 3.66, 17.4146, 0.107953, 330.448,
    0.00752546, 0.0116204, 1.3802, 921.594, 0.0022627, 0.00043711, 6.05953,
    0.107021, 1840.58, 16.2324, 650.988, 3021.77, 89.0497, 0.839824,
    0.0611315, 7829.78, 4.83841, 0.0129005, 6.73903, 0.0257121, 0.019539,
    0.0736186, 10.8716, 68.6747, 9.47338, 86.5586, 0.001037, 2.0]

# lower/upper bounds: an order of magnitude below/above nominal, except for
# the eight Hill-type exponents (n***, indices 78-85), bounded to [1, 12]
p_lower = p_nom ./ 10
p_upper = p_nom .* 10
for i in 78:85
    p_lower[i] = 1.0
    p_upper[i] = 12.0
end

u0 = [0.167, 0.098, 0.6, 0.272, 0.653, 3.48, 0.218, 2.67, 0.808, 0.399, 2.13,
    0.008, 2.67, 0.398, 0.111, 0.276, 0.138, 2.0]

tspan = (0.0, 301.0)
prob = ODEProblem(chassagnole!, u0, tspan, p_nom)


exp1_times = [0.15, 0.3, 0.45, 0.6, 0.8, 5.5, 12.0, 21.5, 31.5, 61.0, 90.0, 120.5, 180.5, 300.5]
exp1_states = [8, 6, 13, 3]  # pep, g6p, pyr, f6p
exp1_data = [
    1.99 4.39 4.07 0.62
    2.10 4.76 3.71 0.66
    2.09 4.86 3.19 0.74
    1.84 4.65 3.57 0.62
    2.31 4.75 3.14 0.75
    2.76 5.52 2.38 0.92
    3.05 5.86 3.71 1.15
    2.42 4.39 3.19 0.57
    2.23 3.60 5.24 0.46
    2.52 3.83 4.47 0.57
    2.81 4.30 3.62 0.57
    2.71 4.05 3.62 0.69
    2.71 3.27 2.86 0.46
    2.70 3.38 2.40 0.46]

exp2_times = [5.5, 13.5, 31.0, 61.0, 91.0, 151.0, 181.0, 212.5, 241.0, 270.5, 301.0]
exp2_states = [18]  # glcex
exp2_data = reshape(
    [1.255555556, 1.311111111, 1.283333333, 0.8611111111, 0.5972222223,
        0.09611111112, 0.04333333334, 0.05055555556, 0.04777777778,
        0.04777777778, 0.06], :, 1)

exp3_times = [2.0, 16.0, 19.0, 31.0, 57.0, 91.5, 150.5, 299.0]
exp3_states = [5]  # g1p
exp3_data = reshape([1.35, 0.83, 0.83, 0.78, 0.84, 0.64, 0.74, 0.70], :, 1)

exp4_times = [3.5, 4.0, 12.0, 12.25, 21.0, 25.75, 30.0, 32.25, 58.5, 59.0,
    119.75, 124.0, 178.0, 180.0, 209.0]
exp4_states = [9]  # 6pg
exp4_data = reshape(
    [1.01, 0.92, 1.15, 1.19, 1.06, 1.10, 1.05, 1.08, 0.97, 1.01, 0.92, 0.89,
        0.74, 0.88, 0.80], :, 1)

exp5_times = [4.5, 11.0, 20.0, 30.0, 60.0, 90.0, 119.5, 180.0, 239.5, 300.0]
exp5_states = [4, 7]  # fdp, gap
exp5_data = [
    0.19 0.28
    0.56 0.32
    1.00 0.31
    2.83 0.24
    1.50 0.30
    2.26 0.18
    2.40 0.22
    1.25 0.21
    0.07 0.22
    0.02 0.20]

experiments = [
    (times = exp1_times, states = exp1_states, data = exp1_data),
    (times = exp2_times, states = exp2_states, data = exp2_data),
    (times = exp3_times, states = exp3_states, data = exp3_data),
    (times = exp4_times, states = exp4_states, data = exp4_data),
    (times = exp5_times, states = exp5_states, data = exp5_data)]


function biopredyn_b2_cost(p)
    # Away from the nominal parameters, global optimizers routinely probe
    # combinations that drive a state negative, which raised to one of the
    # model's non-integer Hill exponents throws a DomainError. The original
    # AMIGO2/MATLAB objective handles this by checking `isreal`/`isnan` after
    # integration and substituting a large penalty (`f = 1e20`); we do the
    # equivalent with a try/catch here, since Julia raises instead of
    # returning a complex number.
    try
        sol = solve(prob, Rodas5P(), p = p, reltol = 1e-6, abstol = 1e-8)
        sol.retcode == ReturnCode.Success || return 1e20

        cost = 0.0
        for e in experiments
            simvals = Array(sol(e.times))[e.states, :]'
            err = max.(0.15 .* abs.(e.data), 1e-6)
            cost += sum(((simvals .- e.data) ./ err) .^ 2)
        end
        return cost
    catch e
        e isa DomainError || rethrow()
        return 1e20
    end
end


@time cost_nominal = biopredyn_b2_cost(p_nom)


sol_nom = solve(prob, Rodas5P(), p = p_nom, reltol = 1e-6, abstol = 1e-8)
plot(sol_nom, idxs = [8, 6, 13, 3], label = ["pep" "g6p" "pyr" "f6p"],
    xlabel = "time (min)", title = "B2 dynamics at nominal parameters")


optf = OptimizationFunction((p, _) -> biopredyn_b2_cost(p))
optprob = OptimizationProblem(optf, p_nom, lb = p_lower, ub = p_upper)


losses_bbo = Float64[]
times_bbo = Float64[]
t0_bbo = time()
cb_bbo = (state, l) -> (push!(losses_bbo, l); push!(times_bbo, time() - t0_bbo); false)
@time res_bbo = solve(
    optprob, BBO_adaptive_de_rand_1_bin(), maxiters = 20000, callback = cb_bbo)
res_bbo.objective


losses_nlopt = Float64[]
times_nlopt = Float64[]
t0_nlopt = time()
cb_nlopt = (state, l) -> (push!(losses_nlopt, l); push!(times_nlopt, time() - t0_nlopt); false)
opt = Opt(:GN_CRS2_LM, length(p_nom))
@time res_nlopt = solve(optprob, opt, maxiters = 20000, callback = cb_nlopt)
res_nlopt.objective


n_particles = 40
@time res_pso = solve(optprob, ParallelPSOArray(n_particles), maxiters = 12000 ÷ n_particles)
res_pso.objective


optf_grad = OptimizationFunction((p, _) -> biopredyn_b2_cost(p), Optimization.AutoForwardDiff())

function polish(start_u, label)
    prob = OptimizationProblem(optf, start_u, lb = p_lower, ub = p_upper)
    losses = Float64[]
    times = Float64[]
    t0 = time()
    cb = (state, l) -> (push!(losses, l); push!(times, time() - t0); false)
    t = @elapsed res = solve(
        prob, Opt(:LN_BOBYQA, length(p_nom)), maxiters = 5000, callback = cb)
    println(label, ": ", res.objective, " (", t, "s)")
    return res, losses, times
end

function polish_grad(start_u, label)
    prob = OptimizationProblem(optf_grad, start_u, lb = p_lower, ub = p_upper)
    losses = Float64[]
    times = Float64[]
    t0 = time()
    cb = (state, l) -> (push!(losses, l); push!(times, time() - t0); false)
    t = @elapsed res = solve(
        prob, Opt(:LD_LBFGS, length(p_nom)), maxiters = 50, callback = cb)
    println(label, ": ", res.objective, " (", t, "s)")
    return res, losses, times
end

res_bbo_polish, losses_bbo_polish, times_bbo_polish = polish(res_bbo.u, "BBO -> LN_BOBYQA")
res_nlopt_polish, losses_nlopt_polish, times_nlopt_polish = polish(
    res_nlopt.u, "GN_CRS2_LM -> LN_BOBYQA")
res_pso_polish, losses_pso_polish, times_pso_polish = polish(
    res_pso.u, "ParallelPSOArray -> LN_BOBYQA")

res_bbo_lbfgs, losses_bbo_lbfgs, times_bbo_lbfgs = polish_grad(res_bbo.u, "BBO -> LD_LBFGS")
res_nlopt_lbfgs, losses_nlopt_lbfgs, times_nlopt_lbfgs = polish_grad(
    res_nlopt.u, "GN_CRS2_LM -> LD_LBFGS")
res_pso_lbfgs, losses_pso_lbfgs, times_pso_lbfgs = polish_grad(
    res_pso.u, "ParallelPSOArray -> LD_LBFGS")


df = DataFrame(
    method = ["Nominal (no fit)", "BBO_adaptive_de_rand_1_bin", "GN_CRS2_LM",
        "ParallelPSOArray",
        "LN_BOBYQA polish (from BBO)", "LD_LBFGS polish (from BBO)",
        "LN_BOBYQA polish (from GN_CRS2_LM)", "LD_LBFGS polish (from GN_CRS2_LM)",
        "LN_BOBYQA polish (from ParallelPSOArray)",
        "LD_LBFGS polish (from ParallelPSOArray)"],
    cost = [cost_nominal, res_bbo.objective, res_nlopt.objective, res_pso.objective,
        res_bbo_polish.objective, res_bbo_lbfgs.objective,
        res_nlopt_polish.objective, res_nlopt_lbfgs.objective,
        res_pso_polish.objective, res_pso_lbfgs.objective])


bestcost_bbo = accumulate(min, losses_bbo)
bestcost_nlopt = accumulate(min, losses_nlopt)
bestcost_bbo_polish = accumulate(min, losses_bbo_polish)
bestcost_nlopt_polish = accumulate(min, losses_nlopt_polish)
bestcost_pso_polish = accumulate(min, losses_pso_polish)
bestcost_bbo_lbfgs = accumulate(min, losses_bbo_lbfgs)
bestcost_nlopt_lbfgs = accumulate(min, losses_nlopt_lbfgs)
bestcost_pso_lbfgs = accumulate(min, losses_pso_lbfgs)

bbo_polish_iters = length(losses_bbo) .+ (1:length(losses_bbo_polish))
nlopt_polish_iters = length(losses_bbo) .+ (1:length(losses_nlopt_polish))
pso_polish_iters = length(losses_bbo) .+ (1:length(losses_pso_polish))
bbo_lbfgs_iters = length(losses_bbo) .+ (1:length(losses_bbo_lbfgs))
nlopt_lbfgs_iters = length(losses_bbo) .+ (1:length(losses_nlopt_lbfgs))
pso_lbfgs_iters = length(losses_bbo) .+ (1:length(losses_pso_lbfgs))

plot(bestcost_bbo, label = "BBO_adaptive_de_rand_1_bin", yscale = :log10,
    xlabel = "iteration", ylabel = "best cost so far (log scale)",
    title = "B2 parameter estimation convergence", legend = :outertopright,
    size = (900, 500))
plot!(bestcost_nlopt, label = "GN_CRS2_LM")
plot!(bbo_polish_iters, bestcost_bbo_polish, label = "LN_BOBYQA polish (from BBO)")
plot!(bbo_lbfgs_iters, bestcost_bbo_lbfgs, label = "LD_LBFGS polish (from BBO)")
plot!(nlopt_polish_iters, bestcost_nlopt_polish, label = "LN_BOBYQA polish (from GN_CRS2_LM)")
plot!(nlopt_lbfgs_iters, bestcost_nlopt_lbfgs, label = "LD_LBFGS polish (from GN_CRS2_LM)")
plot!(pso_polish_iters, bestcost_pso_polish, label = "LN_BOBYQA polish (from PSO)")
plot!(pso_lbfgs_iters, bestcost_pso_lbfgs, label = "LD_LBFGS polish (from PSO)")
hline!([234.2], label = "paper reference (eSS, ~10^5 evals)", linestyle = :dash)


plot(times_bbo, bestcost_bbo, label = "BBO_adaptive_de_rand_1_bin", yscale = :log10,
    xlabel = "wall-clock time (s)", ylabel = "best cost so far (log scale)",
    title = "B2 parameter estimation convergence (wall time)", legend = :outertopright,
    size = (900, 500))
plot!(times_nlopt, bestcost_nlopt, label = "GN_CRS2_LM")
plot!(times_bbo_polish, bestcost_bbo_polish, label = "LN_BOBYQA polish (from BBO)")
plot!(times_bbo_lbfgs, bestcost_bbo_lbfgs, label = "LD_LBFGS polish (from BBO)")
plot!(times_nlopt_polish, bestcost_nlopt_polish, label = "LN_BOBYQA polish (from GN_CRS2_LM)")
plot!(times_nlopt_lbfgs, bestcost_nlopt_lbfgs, label = "LD_LBFGS polish (from GN_CRS2_LM)")
plot!(times_pso_polish, bestcost_pso_polish, label = "LN_BOBYQA polish (from PSO)")
plot!(times_pso_lbfgs, bestcost_pso_lbfgs, label = "LD_LBFGS polish (from PSO)")
hline!([234.2], label = "paper reference (eSS, ~10^5 evals)", linestyle = :dash)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

