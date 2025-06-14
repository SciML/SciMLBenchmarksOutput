---
author: "Guilherme Zagatti"
title: "Synapse model"
---
```julia
using PiecewiseDeterministicMarkovProcesses, JumpProcesses, OrdinaryDiffEq
using Catalyst, Parameters, LazySets, StaticArrays, Distributions, LinearAlgebra, SparseArrays, RecursiveArrayTools
using Plots
using BenchmarkTools
const PDMP = PiecewiseDeterministicMarkovProcesses
const fmt = :png
```

```
:png
```





This benchmark implements the stochastic model of hippocampal synaptic plasticity with geometrical readount of enzyme dinamics from Rodrigues et al. [1]. The source code for the model was obtained from the Github repository [SynapseElife](https://github.com/rveltz/SynapseElife/) that accompanies the paper. The original source code is licensed with the MIT license. We have added comments on the parts of the code that were directly borrowed from the repository.

Initial idea for benchmarking this model comes from a [Discourse discussion](https://discourse.julialang.org/t/help-me-beat-lsoda/88236).

# Model and example solutions

## Parameters and initial conditions

### Presynaptic parameters

```julia
# adapted from  SynapseElife/src/ParamsSynapse.jl
@with_kw struct PreSynapseParams
    "recovery constant of pre calcium decay function."
    τ_rec::Float64 = 20000
    "fraction of decay constant of pre calcium decay f."
    δ_ca::Float64 = 0.0004
    "decay time constant of pre calcium."
    τ_pre::Float64 = 20
    "decay time constant for AP induced by EPSP."
    τ_V::Float64 = 40
    "delay to EPSPs onset and evoked AP."
    δ_delay_AP::Float64 = 15.0
    "initial conditions ready releaseble pool."
    D_0::Int64 = 25
    "initial conditions recovery pool."
    R_0::Int64 = 30
    "rate for `D -> R`."
    τ_R::Float64 = 5000
    "rate for `R -> D`."
    τ_D::Float64 = 45000
    "rate for `infinite reservoir -> R`."
    τ_R_ref::Float64 = 40000
    "sigmoid parameter for release probability."
    s::Float64 = 2.0
    "sigmoid parameter for release probability."
    h::Float64 = 0.7 # this value changes given Ca external concentration
    "sampling rate for plotting / printing."
    sampling_rate::Float64 = 1.0
end
```

```
Main.var"##WeaveSandBox#225".PreSynapseParams
```





### Postsynpatic parameters


```julia
# adapted from  SynapseElife/src/ParamsSynapse.jl
@with_kw struct SynapseParams{Tp}
    "polygonal threshold."
    LTP_region::Tp = VPolygon([[6.35, 1.4], [10, 1.4], [6.35, 29.5], [10, 29.5]]) # VPolygon([SVector(6.35,1.4), SVector(10,1.4),SVector(6.35,29.5), SVector(10,29.5)])
    "polygonal threshold."
    LTD_region::Tp = VPolygon([
        [6.35, 1.4],
        [6.35, 23.25],
        [6.35, 29.5],
        [1.85, 11.327205882352938],
        [1.85, 23.25],
        [3.7650354609929075, 1.4],
        [5.650675675675676, 29.5],
    ]) # VPolygon([SVector(6.35,1.4),SVector(6.35,23.25),SVector(6.35,29.5),SVector(1.85,11.327205882352938),SVector(1.85,23.25),SVector(3.7650354609929075,1.4),SVector(5.650675675675676,29.5)])
    "activation rates for thresholds."
    a_D::Float64 = 0.1
    "activation rates for thresholds."
    b_D::Float64 = 0.00002
    "activation rates for thresholds."
    a_P::Float64 = 0.2
    "activation rates for thresholds."
    b_P::Float64 = 0.0001
    "activation rates for thresholds."
    t_D::Float64 = 18000
    "activation rates for thresholds."
    t_P::Float64 = 13000
    "sigmoids controlling the rate of plasticity change."
    K_D::Float64 = 80000.0
    "sigmoids controlling the rate of plasticity change."
    K_P::Float64 = 13000.0
    "plasticity states."
    rest_plstcty::Int64 = 100
    "simulation."
    t_end::Float64 = 100
    "simulation."
    sampling_rate::Float64 = 10
    "biophysical and GHK parameters."
    temp_rates::Float64 = 35.0
    "biophysical and GHK parameters."
    age::Float64 = 60.0
    "biophysical and GHK parameters."
    faraday::Float64 = 96485e-6 * 1e-3
    "biophysical and GHK parameters."
    Ca_ext::Float64 = 2.5e3
    "biophysical and GHK parameters."
    Ca_infty::Float64 = 50e-3
    "biophysical and GHK parameters."
    tau_ca::Float64 = 10.0
    "biophysical and GHK parameters."
    D_Ca::Float64 = 0.3338
    "biophysical and GHK parameters."
    f_Ca::Float64 = 0.1
    "biophysical and GHK parameters."
    perm::Float64 = -0.04583333333333333
    "biophysical and GHK parameters."
    z::Float64 = 2.0
    "biophysical and GHK parameters."
    gas::Float64 = 8.314e-6
    "biophysical and GHK parameters."
    p_release::NTuple{4,Float64} =
        (0.004225803293622208, 1708.4124496514878, 1.3499793762587964, 0.6540248201173222)
    "backpropagation attenuation."
    trec::Float64 = 2000
    "backpropagation attenuation."
    trec_soma::Float64 = 500
    "backpropagation attenuation."
    delta_decay::Float64 = 1.7279e-5
    "backpropagation attenuation."
    p_age_decay_bap::NTuple{3,Float64} =
        (0.13525468256031167, 16.482800452454164, 5.564691354645679)
    "backpropagation attenuation."
    delta_soma::Float64 =
        2.5e-5 *
        (p_age_decay_bap[3] / (1 + exp(p_age_decay_bap[1] * (age - p_age_decay_bap[2]))))
    "backpropagation attenuation."
    delta_aux::Float64 = 2.304e-5
    "backpropagation attenuation."
    injbap::Float64 = 2.0
    "backpropagation attenuation."
    soma_dist::Float64 = 200.0
    "backpropagation attenuation."
    p_dist::NTuple{4,Float64} =
        (0.019719018173341547, 230.3206470553394, 1.4313810030893268, 0.10406540965358434)
    "backpropagation attenuation."
    ϕ_dist::Float64 =
        (p_dist[4] + p_dist[3] / (1 + exp(p_dist[1] * (soma_dist - p_dist[2]))))
    "backpropagation attenuation."
    I_clamp::Float64 = 0.0
    "Na and K."
    gamma_Na::Float64 = 8e2
    "Na and K."
    Erev_Na::Float64 = 50.0
    "Na and K."
    gamma_K::Float64 = 4e1
    "Na and K."
    Erev_K::Float64 = -90.0
    "NMDAr temperature modification."
    p_nmda_frwd::NTuple{4,Float64} =
        (-0.09991802053299291, -37.63132907014948, 1239.0673283348326, -1230.6805720050966)
    "NMDAr temperature modification."
    frwd_T_chng_NMDA::Float64 = (
        p_nmda_frwd[4] +
        p_nmda_frwd[3] / (1 + exp(p_nmda_frwd[1] * (temp_rates - p_nmda_frwd[2])))
    )
    "NMDAr temperature modification."
    p_nmda_bcwd::NTuple{4,Float64} =
        (-0.10605060141396823, 98.99939433046647, 1621.6168608608068, 3.0368551011554143)
    "NMDAr temperature modification."
    bcwd_T_chng_NMDA::Float64 = (
        p_nmda_bcwd[4] +
        p_nmda_bcwd[3] / (1 + exp(p_nmda_bcwd[1] * (temp_rates - p_nmda_bcwd[2])))
    ) # 0.16031*temp_rates - 0.80775
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_ka::Float64 = frwd_T_chng_NMDA * 34.0 * 1e-3
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_kb::Float64 = frwd_T_chng_NMDA * 17.0 * 1e-3
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_kc::Float64 = frwd_T_chng_NMDA * 127.0 * 1e-3
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_kd::Float64 = frwd_T_chng_NMDA * 580.0 * 1e-3
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_ke::Float64 = frwd_T_chng_NMDA * 2508.0 * 1e-3
    "NMDAr kinetics (GluN2A type), uM-1ms-1."
    NMDA_N2A_kf::Float64 = frwd_T_chng_NMDA * 3449.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_f::Float64 = bcwd_T_chng_NMDA * 662.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_e::Float64 = bcwd_T_chng_NMDA * 2167.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_d::Float64 = bcwd_T_chng_NMDA * 2610.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_c::Float64 = bcwd_T_chng_NMDA * 161.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_b::Float64 = bcwd_T_chng_NMDA * 120.0 * 1e-3
    "NMDAr kinetics (GluN2A type), ms-1."
    NMDA_N2A_k_a::Float64 = bcwd_T_chng_NMDA * 60.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_sa::Float64 = frwd_T_chng_NMDA * 0.25 * 34.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_sb::Float64 = frwd_T_chng_NMDA * 0.25 * 17.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_sc::Float64 = frwd_T_chng_NMDA * 0.25 * 127.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_sd::Float64 = frwd_T_chng_NMDA * 0.25 * 580.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_se::Float64 = frwd_T_chng_NMDA * 0.25 * 2508.0 * 1e-3
    "NMDAr kinetics (GluN2B type), uM-1ms-1."
    NMDA_N2B_sf::Float64 = frwd_T_chng_NMDA * 0.25 * 3449.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_f::Float64 = bcwd_T_chng_NMDA * 0.23 * 662.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_e::Float64 = bcwd_T_chng_NMDA * 0.23 * 2167.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_d::Float64 = bcwd_T_chng_NMDA * 0.23 * 2610.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_c::Float64 = bcwd_T_chng_NMDA * 0.23 * 161.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_b::Float64 = bcwd_T_chng_NMDA * 0.23 * 120.0 * 1e-3
    "NMDAr kinetics (GluN2B type), ms-1."
    NMDA_N2B_s_a::Float64 = bcwd_T_chng_NMDA * 0.23 * 60.0 * 1e-3
    "NMDA details."
    p_nmda::NTuple{4,Float64} =
        (0.004477162852447629, 2701.3929349701334, 58.38819453272428, 33.949463268365555)
    "NMDA details."
    gamma_nmda::Float64 =
        (p_nmda[4] + p_nmda[3] / (1 + exp(p_nmda[1] * (Ca_ext - p_nmda[2])))) * 1e-3
    "NMDA details."
    p_age::NTuple{4,Float64} =
        (0.09993657672916968, 25.102347872464193, 0.9642137892004939, 0.5075183905839776)
    "ratio N2B/N2A."
    r_NMDA_age::Float64 =
        rand(Normal(0, 0.05)) + p_age[4] + p_age[3] / (1 + exp(p_age[1] * (age - p_age[2]))) # 0.5+1.6*exp(-age/16.66) + rand(Normal(0,.05))
    "ratio N2B/N2A."
    N_NMDA::Int64 = 15
    "ratio N2B/N2A."
    N_N2B::Int64 = round(N_NMDA * r_NMDA_age / (r_NMDA_age + 1))
    "ratio N2B/N2A, using Sinclair ratio."
    N_N2A::Int64 = round(N_NMDA / (r_NMDA_age + 1))
    "other NMDAr parameters."
    Erev_nmda::Float64 = 0.0
    "other NMDAr parameters."
    Mg::Float64 = 1.0
    "AMPAr temperature modification."
    p_ampa_frwd::NTuple{3,Float64} =
        (-0.4737773089201679, 31.7248285571622, 10.273135485873242)
    "AMPAr temperature modification."
    frwd_T_chng_AMPA::Float64 =
        (p_ampa_frwd[3] / (1 + exp(p_ampa_frwd[1] * (temp_rates - p_ampa_frwd[2])))) # temp_rates*0.78-18.7
    "AMPAr temperature modification."
    p_ampa_bcwd::NTuple{3,Float64} =
        (-0.36705555170278986, 28.976662403966674, 5.134547217640794)
    "AMPAr temperature modification."
    bcwd_T_chng_AMPA::Float64 =
        (p_ampa_bcwd[3] / (1 + exp(p_ampa_bcwd[1] * (temp_rates - p_ampa_bcwd[2])))) # temp_rates*0.37-8.25
    "AMPAr kinetics, uM-1ms-1."
    AMPA_k1::Float64 = frwd_T_chng_AMPA * 1.6 * 1e7 * 1e-6 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_k_1::Float64 = bcwd_T_chng_AMPA * 7400 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_k_2::Float64 = bcwd_T_chng_AMPA * 0.41 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_alpha::Float64 = 2600 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_beta::Float64 = 9600 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_delta_1::Float64 = 1500 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_gamma_1::Float64 = 9.1 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_delta_2::Float64 = 170 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_gamma_2::Float64 = 42 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_delta_0::Float64 = 0.003 * 1e-3
    "AMPAr kinetics, ms-1."
    AMPA_gamma_0::Float64 = 0.83 * 1e-3
    "AMPAr conductances, nS."
    gamma_ampa1::Float64 = 0.5 * 31e-3
    "AMPAr conductances, nS."
    gamma_ampa2::Float64 = 0.5 * 52e-3
    "AMPAr conductances, nS."
    gamma_ampa3::Float64 = 0.5 * 73e-3
    "AMPAr conductances, AMPAr number."
    N_ampa::Int64 = 120
    "AMPAr conductances, AMPAR reversal potential, mV."
    Erev_ampa::Float64 = 0.0
    "GABAr."
    N_GABA::Int64 = 34
    "GABAr."
    p_Cl::NTuple{4,Float64} =
        (0.09151696057098718, 0.6919298240788684, 243.5159017060495, -92.6496083089155)
    "GABAr, Cl reversal potential."
    Erev_Cl::Float64 = (p_Cl[4] + p_Cl[3] / (1 + exp(p_Cl[1] * (age - p_Cl[2]))))
    "GABAr, Cl reversal potential."
    gamma_GABA::Float64 = 35e-3
    "GABAr, Cl reversal potential."
    GABA_r_b1::Float64 = 1e6 * 1e-6 * 1e-3 * 20
    "GABAr, Cl reversal potential."
    GABA_r_u1::Float64 = 1e3 * 4.6e-3
    "GABAr, Cl reversal potential."
    GABA_r_b2::Float64 = 1e6 * 1e-6 * 1e-3 * 10
    "GABAr, Cl reversal potential."
    GABA_r_u2::Float64 = 1e3 * 9.2e-3
    "GABAr, Cl reversal potential."
    GABA_r_ro1::Float64 = 1e3 * 3.3e-3
    "GABAr, Cl reversal potential."
    GABA_r_ro2::Float64 = 1e3 * 10.6e-3
    "GABAr, Cl reversal potential."
    p_GABA::NTuple{4,Float64} =
        (0.19127068198185954, 32.16771140618756, -1.2798050197287802, 1.470692263981145)
    "GABAr, Cl reversal potential."
    GABA_r_c1::Float64 =
        (p_GABA[4] + p_GABA[3] / (1 + exp(p_GABA[1] * (temp_rates - p_GABA[2])))) *
        1e3 *
        9.8e-3
    "GABAr, Cl reversal potential."
    GABA_r_c2::Float64 =
        (p_GABA[4] + p_GABA[3] / (1 + exp(p_GABA[1] * (temp_rates - p_GABA[2])))) * 400e-3
    "passive electrical properties."
    E_leak::Float64 = -70.0
    "passive electrical properties."
    g_leak::Float64 = 4e-6
    "passive electrical properties."
    Cm::Float64 = 0.6e-2
    "passive electrical properties."
    R_a::Float64 = 1e-2
    "morphology, Dendritic properties, dendrite diameter, um."
    D_dend::Float64 = 2.0
    "morphology, Dendritic properties, dendrite length, chosen to tune attenuation, but not modified in BaP adaptation for simplicity sake, um."
    L_dend::Float64 = 1400
    "morphology, Dendritic properties, dendrite surface area, 500 gives dendrite input resistance of 200MOhm, um^2."
    A_dend::Float64 = 2 * pi * (D_dend / 2) * L_dend
    "morphology, Dendritic properties, dendrite volume, um^3."
    Vol_dend::Float64 = pi * ((D_dend / 2)^2) * L_dend
    "morphology, Dendritic properties, dendritic membrane capacitance."
    Cdend::Float64 = Cm * A_dend
    "morphology, Dendritic properties, dendrite cross-sectional area, um^2."
    CS_dend::Float64 = pi * (D_dend / 2) .^ 2
    "morphology, Dendritic properties, nS."
    g_leakdend::Float64 = g_leak * A_dend
    "morphology, Soma properties, soma diameter, um."
    D_soma::Float64 = 30
    "morphology, Soma properties, soma surface area, 500 gives dendrite input resistance of 200MOhm, um^2."
    A_soma::Float64 = pi * D_soma^2
    "morphology, Soma properties, soma membrane capacitance."
    Csoma::Float64 = Cm * A_soma
    "morphology, Soma properties, soma cross-sectional area, um^2."
    CS_soma::Float64 = pi * (D_soma / 2) .^ 2
    "morphology, Soma properties, nS."
    g_leaksoma::Float64 = 15.0
    "morphology, Soma properties, value subject to modifications due to BaP adaptation implementation."
    g_diff::Float64 = D_dend / (4R_a)
    "spine properties, spine head volume [bartol2015], um^3."
    Vol_sp::Float64 = 0.03
    "spine properties, spine head surface area."
    A_sp::Float64 = 4 * pi * ((3 * Vol_sp) / (4 * pi))^(2.0 / 3.0)
    "spine properties, spine membrane capacitance."
    Csp::Float64 = Cm * A_sp
    "spine properties, spine head leak conductance."
    g_leaksp::Float64 = g_leak * A_sp
    "neck properties, spine neck diameter [bartol2015], um."
    D_neck::Float64 = 0.1
    "neck properties, neck length [bartol2015], um."
    L_neck::Float64 = 0.2
    "neck properties, neck cross sectional area, um^2."
    CS_neck::Float64 = pi * (D_neck / 2) .^ 2
    "neck properties."
    g_neck::Float64 = CS_neck / (L_neck * R_a)
    "neck properties."
    tau_diff::Float64 = ((Vol_sp / (2 * D_Ca * D_neck)) + (L_neck^2 / (2 * D_Ca)))
    "synpatic glutamate transient parameters, arbitrary, ms."
    glu_width::Float64 = 1.0 # 0.1 ms for synapse
    "synpatic glutamate transient parameters, arbitrary, mM."
    glu_amp::Float64 = 1e+3
    "synpatic glutamate transient parameters [liu1999]."
    glu_cv::Float64 = 0.5
    "SK channels, number of SK channels."
    N_SK::Int64 = 15
    "SK channels [maylie2004], ns."
    SK_gamma::Float64 = 10e-3
    "SK channels [mellor2016annex], mv."
    SK_Erev::Float64 = -90
    "SK channels [mellor2016annex], uM."
    SK_gating_half::Float64 = 0.33
    "SK channels [mellor2016annex], ms."
    SK_time::Float64 = 6.3
    "SK channels [mellor2016annex], ms."
    SK_hill::Float64 = 6
    "SK channels."
    p_SK_bcwd::NTuple{4,Float64} =
        (0.09391588258147192, 98.85165844770867, -147.61669527876904, 149.37767054612135)
    "SK channels."
    bcwd_SK::Float64 = (
        p_SK_bcwd[4] + p_SK_bcwd[3] / (1 + exp(p_SK_bcwd[1] * (temp_rates - p_SK_bcwd[2])))
    )
    "SK channels."
    p_SK_frwd::NTuple{4,Float64} =
        (-0.334167923607112, 25.590920461511878, 2.2052151559841193, 0.005904170174699533)
    "SK channels."
    frwd_SK::Float64 = (
        p_SK_frwd[4] + p_SK_frwd[3] / (1 + exp(p_SK_frwd[1] * (temp_rates - p_SK_frwd[2])))
    )
    "CaM, CaMKII and CaN concentrations."
    CaM_con::Float64 = 30.0
    "CaM, CaMKII and CaN concentrations, renamed [feng2011], 100um."
    mKCaM_con::Float64 = 70.0
    "CaM, CaMKII and CaN concentrations [lisman?], uM."
    mCaN_con::Float64 = 20.0
    "Chang Pepke model - CaM reactions I."
    kon_1C::Float64 = 5e-3
    "Chang Pepke model - CaM reactions I."
    koff_1C::Float64 = 50e-3
    "Chang Pepke model - CaM reactions I."
    kon_2C::Float64 = 10e-3
    "Chang Pepke model - CaM reactions I."
    koff_2C::Float64 = 10e-3
    "Chang Pepke model - CaM reactions I."
    kon_1N::Float64 = 100e-3
    "Chang Pepke model - CaM reactions I."
    koff_1N::Float64 = 2000e-3
    "Chang Pepke model - CaM reactions I."
    kon_2N::Float64 = 200e-3
    "Chang Pepke model - CaM reactions I."
    koff_2N::Float64 = 500e-3
    "Chang Pepke model - CaM reactions II."
    kf_CaM0::Float64 = 3.8e-6
    "Chang Pepke model - CaM reactions II."
    kb_CaM0::Float64 = 5.5e-3
    "Chang Pepke model - CaM reactions II."
    kf_CaM2C::Float64 = 0.92e-3
    "Chang Pepke model - CaM reactions II."
    kb_CaM2C::Float64 = 6.8e-3
    "Chang Pepke model - CaM reactions II."
    kf_CaM2N::Float64 = 0.12e-3
    "Chang Pepke model - CaM reactions II."
    kb_CaM2N::Float64 = 1.7e-3
    "Chang Pepke model - CaM reactions II."
    kf_CaM4::Float64 = 30e-3
    "Chang Pepke model - CaM reactions II."
    kb_CaM4::Float64 = 1.5e-3
    "Chang Pepke model - CaMKII reactions."
    kon_K1C::Float64 = 44e-3
    "Chang Pepke model - CaMKII reactions."
    koff_K1C::Float64 = 33e-3
    "Chang Pepke model - CaMKII reactions."
    kon_K2C::Float64 = 44e-3
    "Chang Pepke model - CaMKII reactions."
    koff_K2C::Float64 = 0.8e-3
    "Chang Pepke model - CaMKII reactions."
    kon_K1N::Float64 = 76e-3
    "Chang Pepke model - CaMKII reactions."
    koff_K1N::Float64 = 300e-3
    "Chang Pepke model - CaMKII reactions."
    kon_K2N::Float64 = 76e-3
    "Chang Pepke model - CaMKII reactions."
    koff_K2N::Float64 = 20e-3
    "Chang Pepke model - autophosphorilation."
    p_camkii_q10::NTuple{4,Float64} =
        (0.5118207068695309, 45.47503600542303, -161.42634157226917, 162.1718925882677)
    "Chang Pepke model - autophosphorilation."
    q10::Float64 =
        p_camkii_q10[4] +
        p_camkii_q10[3] / (1 + exp(p_camkii_q10[1] * (temp_rates - p_camkii_q10[2]))) # change of temp to fit chang 35C
    "Chang Pepke model - autophosphorilation."
    k1::Float64 = 12.6e-3
    "Chang Pepke model - autophosphorilation."
    k2::Float64 = q10 * 0.33e-3 # q10 * 0.33e-3
    "Chang Pepke model - autophosphorilation."
    k3::Float64 = 4 * q10 * 0.17e-3 # q10 * 0.17e-3
    "Chang Pepke model - autophosphorilation."
    k4::Float64 = 4 * 0.041e-3
    "Chang Pepke model - autophosphorilation."
    k5::Float64 = 4 * q10 * 2 * 0.017e-3 # q10 * 2* 0.017e-3
    "CaM-CaN reactions."
    p_CaN_frwd::NTuple{4,Float64} =
        (-0.29481489145354556, 29.999999999999968, 0.15940019940354327, 0.870299900298228)
    "CaM-CaN reactions, 22C - 4.6e-2 [quintana2005]."
    kcanf::Float64 =
        (
            p_CaN_frwd[4] +
            p_CaN_frwd[3] / (1 + exp(p_CaN_frwd[1] * (temp_rates - p_CaN_frwd[2])))
        ) * 1.75e-2
    "CaM-CaN reactions."
    p_CaN_bcwd::NTuple{4,Float64} =
        (-0.6833299932488973, 26.277500129849113, 0.7114524682690591, 0.29037766196937326)
    "CaM-CaN reactions, 22C - 1.2e-6 [quintana2005]."
    kcanb::Float64 =
        (
            p_CaN_bcwd[4] +
            p_CaN_bcwd[3] / (1 + exp(p_CaN_bcwd[1] * (temp_rates - p_CaN_bcwd[2])))
        ) * 2e-5
    "VGCCs."
    p_frwd_VGCC::NTuple{4,Float64} =
        (1.0485098341579628, 30.66869198447378, -0.3040010721391852, 2.5032059559264357)
    "VGCCs."
    frwd_VGCC::Float64 = (
        p_frwd_VGCC[4] +
        p_frwd_VGCC[3] / (1 + exp(p_frwd_VGCC[1] * (temp_rates - p_frwd_VGCC[2])))
    )
    "VGCCs."
    p_bcwd_VGCC::NTuple{4,Float64} =
        (-0.3302682317933842, 36.279019647221226, 3.2259761593440155, 0.7298285671937866)
    "VGCCs."
    bcwd_VGCC::Float64 = (
        p_bcwd_VGCC[4] +
        p_bcwd_VGCC[3] / (1 + exp(p_bcwd_VGCC[1] * (temp_rates - p_bcwd_VGCC[2])))
    )
    "VGCCs, calcium channel reversal potential, mV."
    Erev_CaT::Float64 = 10.0
    "VGCCs, calcium channel reversal potential, mV."
    Erev_CaR::Float64 = 10.0
    "VGCCs, calcium channel reversal potential, mV."
    Erev_CaL::Float64 = 10.0
    "VGCCs [magee1995], nS."
    gamma_CaT::Float64 = 12e-3
    "VGCCs [magee1995], nS."
    gamma_CaR::Float64 = 17e-3
    "VGCCs [magee1995], nS."
    gamma_CaL::Float64 = 27e-3
    "VGCCs."
    N_caT::Int64 = 3
    "VGCCs."
    N_caR::Int64 = 3
    "VGCCs."
    N_caL::Int64 = 3
    "calcium dye and buffers [zenisek2003,naraghi1997], uMms-1."
    EGTA_kf::Float64 = 2.7e-3
    "calcium dye and buffers, assuming Kd of 0.18uM [naraghi1997] ms-1."
    EGTA_kb::Float64 = 0.18 * EGTA_kf
    "calcium dye and buffers, 0.2 for imaging, 200 for elecrophysiology [tigaret2016] uM."
    EGTA_con::Float64 = 0.0
    "calcium dye and buffers [zenisek2003,naraghi1997], uM-1ms-1."
    BAPTA_kf::Float64 = 0.45
    "calcium dye and buffers, assuming Kd of 0.176uM [naraghi1997], ms-1."
    BAPTA_kb::Float64 = 0.176 * BAPTA_kf
    "calcium dye and buffers, uM."
    BAPTA_con::Float64 = 0.0
    "calcium dye and buffers [bartol2015], uM-1ms-1."
    Imbuf_k_on::Float64 = 0.247
    "calcium dye and buffers [bartol2015], ms-1."
    Imbuf_k_off::Float64 = 0.524
    "calcium dye and buffers."
    K_buff_diss::Float64 = Imbuf_k_off / Imbuf_k_on
    "calcium dye and buffers, 76.7 [bartol2015], uM."
    Imbuf_con::Float64 = 62
    "calcium dye and buffers."
    Imbuf_con_dend::Float64 = Imbuf_con * 4
    "calcium fluorescent dyes, assuming a [Ca] = 1um [bartol2015], ms-1."
    ogb1_kf::Float64 = 0.8
    "calcium fluorescent dyes [bartol2015], ms-1."
    ogb1_kb::Float64 = 0.16
    "calcium fluorescent dyes, assuming a [Ca] = 1um [bartol2015], ms-1."
    fluo4_kf::Float64 = 0.8
    "calcium fluorescent dyes [bartol2015], ms-1."
    fluo4_kb::Float64 = 0.24
    "calcium fluorescent dyes."
    dye::Float64 = 0.0
    "calcium fluorescent dyes [zenisek2003,naraghi1997], uMms-1."
    fluo5f_kf::Float64 = dye * 0.01
    "calcium fluorescent dyes assuming [Kd] = 1.3uM [yasuda2004]."
    fluo5f_kb::Float64 = dye * 26 * fluo5f_kf
    "calcium fluorescent dyes uM [tigaret2016]."
    fluo5f_con::Float64 = dye * 200.0
end
```

```
Main.var"##WeaveSandBox#225".SynapseParams
```





### Transition matrices

We define some utilities to help us construct the transition matrices.

```julia
# adapted from SynapseElife/src/JumpMatrices.jl
get_stoichmatrix(model) = transpose(Catalyst.netstoichmat(model)) |> Matrix

# adapted from SynapseElife/src/JumpMatrices.jl
jump_matrix(matrix_list) = cat(matrix_list..., dims = (1, 2))
```

```
jump_matrix (generic function with 1 method)
```





Now we define each matrix one-by-one.

```julia
# adapted from SynapseElife/src/JumpMatrices.jl
const ampa_model = @reaction_network begin
    #1line
    #2line-GO
    1, C0 → C1
    1, C1 → C2
    1, C2 → C3
    1, C3 → C4
    #2line-BAC
    1, C4 → C3
    1, C3 → C2
    1, C2 → C1
    1, C1 → C0
    #3line-GO
    1, D0 → D1
    1, D1 → D2
    1, D2 → D3
    1, D3 → D4
    #3line-BACK
    1, D4 → D3
    1, D3 → D2
    1, D2 → D1
    1, D1 → D0
    #4line-GO
    1, D22 → D23
    1, D23 → D24
    #4line-BACK
    1, D24 → D23
    1, D23 → D22
    #1column-GO-BACK
    1, C0 → D0
    1, D0 → C0
    #2column-GO-BACK
    1, C1 → D1
    1, D1 → C1
    #3column-GO
    1, O2 → C2
    1, C2 → D2
    1, D2 → D22
    #3column-BACK
    1, D22 → D2
    1, D2 → C2
    1, C2 → O2
    #4column-GO
    1, O3 → C3
    1, C3 → D3
    1, D3 → D23
    #4column-BACK
    1, D23 → D3
    1, D3 → C3
    1, C3 → O3
    #5column-GO
    1, O4 → C4
    1, C4 → D4
    1, D4 → D24
    #5column-BACK
    1, D24 → D4
    1, D4 → C4
    1, C4 → O4
end

# adapted from SynapseElife/src/JumpMatrices.jl
AMPA_matrix() = get_stoichmatrix(ampa_model)

# adapted from SynapseElife/src/JumpMatrices.jl
const nmda_model_v2 = @reaction_network begin #same structure for N2A and N2B
    #1line-GO
    1, A0 → A1
    1, A1 → A2
    1, A2 → A3
    1, A3 → A4
    1, A4 → AO1
    1, AO1 → AO2
    #2line-BACK
    1, AO2 → AO1
    1, AO1 → A4
    1, A4 → A3
    1, A3 → A2
    1, A2 → A1
    1, A1 → A0
end

# adapted from SynapseElife/src/JumpMatrices.jl
NMDA_matrix() = get_stoichmatrix(nmda_model_v2)

# adapted from SynapseElife/src/JumpMatrices.jl
R_channel_matrix() = [
    [-1 1 0 0] # CaR m0h0 -> m1h0
    [1 -1 0 0] # CaR m1h0 -> m0h0
    [-1 0 1 0] # CaR m0h0 -> m0h1
    [1 0 -1 0] # CaR m0h1 -> m0h0
    [0 -1 0 1] # CaR m1h0 -> O
    [0 1 0 -1] # CaR    O -> m1h0
    [0 0 -1 1] # CaR m0h1 -> O
    [0 0 1 -1] # CaR    O -> m0h1
]

# adapted from SynapseElife/src/JumpMatrices.jl
T_channel_matrix() = [
    [-1 1 0 0] # CaT m0h0 -> m1h0
    [1 -1 0 0] # CaT m1h0 -> m0h0
    [-1 0 1 0] # CaT m0h0 -> m0h1
    [1 0 -1 0] # CaT m0h1 -> m0h0
    [0 -1 0 1] # CaT m1h0 -> O
    [0 1 0 -1] # CaT    O -> m1h0
    [0 0 -1 1] # CaT m0h1 -> O
    [0 0 1 -1] # CaT    O -> m0h
]

# adapted from SynapseElife/src/JumpMatrices.jl
L_channel_matrix() = [
    [-1 1 0] # CaL C -> O1
    [1 -1 0] # CaL O1 -> C
    [-1 0 1] # CaL C -> O2
    [1 0 -1] # CaL O2 -> C
]

# adapted from SynapseElife/src/JumpMatrices.jl
const plasticity = @reaction_network begin
    1, NC → LTD
    1, LTD → NC
    1, NC → LTP
    1, LTP → NC
end

# adapted from SynapseElife/src/JumpMatrices.jl
LTP_LTD_matrix() = get_stoichmatrix(plasticity)

# adapted from SynapseElife/src/JumpMatrices.jl
const gaba_destexhe = @reaction_network begin
    #1line-GO
    (1, 1), CO ↔ C1
    (1, 1), C1 ↔ C2
    (1, 1), C1 ↔ O1
    (1, 1), C2 ↔ O2
end

# adapted from SynapseElife/src/JumpMatrices.jl
GABA_matrix() = get_stoichmatrix(gaba_destexhe)
```

```
GABA_matrix (generic function with 1 method)
```





Finally, we define a wrapper function to build the main transition matrix.

```julia
# adapted from SynapseElife/src/SynapseModel.jl
function buildTransitionMatrix()
    matrix_list = [AMPA_matrix()]
    push!(matrix_list, NMDA_matrix()) # for GluN2A
    push!(matrix_list, Matrix{Int64}(I, 1, 1)) # print from Poisson Rate
    push!(matrix_list, R_channel_matrix())
    push!(matrix_list, T_channel_matrix())
    push!(matrix_list, L_channel_matrix())
    push!(matrix_list, LTP_LTD_matrix())
    push!(matrix_list, NMDA_matrix()) # for GluN2B
    push!(matrix_list, GABA_matrix())
    return sparse(jump_matrix(matrix_list))
end
```

```
buildTransitionMatrix (generic function with 1 method)
```





### Assembling it all

Parameters.

```julia
const p_synapse = SynapseParams(t_end = 1000.0);
const glu = 0.0;
const events_sorted_times = [500.0];
const is_pre_or_post_event = [true];
const events_bap = events_sorted_times[is_pre_or_post_event.==false];
const bap_by_epsp = Float64[];
const nu = buildTransitionMatrix();
```




Initial conditions.

```julia
# adapted from SynapseElife/src/UtilsData.jl
function initial_conditions_continuous_temp(param_synapse)
    @unpack_SynapseParams param_synapse
    if temp_rates <= 25
        return vec(
            [
                -70.10245699808998
                -70.02736715107497
                -70.01992573979436
                1.0
                5.251484030952095
                0.17942311304488254
                0.0
                18.422417385628144
                0.061182835845181506
                0.007491230194287401
                2.5601235159798556e-5
                18.60768149870139
                1.3923185018623478
                47.943467158481965
                0.33484901412981444
                0.00873078540237604
                0.26160543666353603
                0.006945031752908706
                5.28369636945184
                4.008117080305045
                0.11294478061086997
                0.09967594670655765
                4.656494469862573
                7.283473926518337
                0.0
                0.0
                0.0
                0.0
                0.012283139643655655
                0.9999998289470913
                0.00010811866849049202
                0.09878906052566663
                1.0
                1.0
            ],
        )
    end
    if 25 < temp_rates <= 30.0
        return vec(
            [
                -70.0140727673961
                -70.00177103943689
                -70.00007589726667
                1.0
                3.48177628683147
                0.11706193391209137
                0.0
                19.902059159951726
                0.04813075789246737
                0.0033156783870759324
                1.3058864642411135e-5
                19.473469645955362
                0.5265303542539336
                49.658388018645105
                0.39979810740476907
                0.005987535794514187
                0.22114161049072126
                0.003443459770906218
                5.486877056572574
                3.2429635679748126
                0.0967411298498974
                0.06299852278885718
                4.315666798657558
                6.505994192089851
                0.0
                0.0
                0.0
                0.0
                0.0123167984282587
                0.9999998268859568
                0.00010833059621284
                0.016112012029112062
                1.0
                1.0
            ],
        )
    end
    if temp_rates > 30
        return vec(
            [
                -70.02953996060384
                -70.00364683510847
                -70.0013995913228
                1.0
                3.3989773357494646
                0.11430174346528181
                0.0
                25.14694054430742
                0.048232611580821316
                0.004340152415081933
                6.8123069384392556e-6
                19.865575695641734
                0.13442430463415694
                62.32357049746406
                0.8459878113145832
                0.009234347120412881
                0.3607751119415231
                0.003937688645035126
                2.3440067259757305
                1.0593952833042108
                0.02913354160885602
                0.013585064852268125
                1.6677561415335218
                1.3426177862569586
                0.0
                0.0
                0.0
                0.0
                0.012314741437834478
                0.999999826986425
                0.00010831999489540837
                0.03393795777123425
                1.0
                1.0
            ],
        )
    end
end

# adapted from SynapseElife/src/UtilsData.jl
function initial_conditions_discrete(param_synapse)
    @unpack_SynapseParams param_synapse

    return vec([
        N_ampa, # AMPA 1-16
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        N_N2A,  # NMDA 17-23
        0,
        0,
        0,
        0,
        0,
        0,
        0, # Print 24
        0, # R-type 25-28
        0,
        N_caR,
        0,
        0, # T-type 29-32
        0,
        N_caT,
        0,
        N_caL, # L-type 33-35
        0,
        0,
        rest_plstcty, # GLUN2B
        0,
        0,
        N_N2B,
        0,
        0,
        0,
        0,
        0,
        0,
        N_GABA, # GABA
        0,
        0,
        0,
        0,
    ])  
end
```

```
initial_conditions_discrete (generic function with 1 method)
```



```julia
const xc0 = initial_conditions_continuous_temp(p_synapse);
const xd0 = initial_conditions_discrete(p_synapse);
```




## Model

Rodrigues et al. [1] synapse model that replaces NMDA model with fully state-based one from Jahr and Stevens, plus three types of VGCCs (R-type, T-type and L-type), from Magee and Johhston [2].

### Continuous evolution

Helper functions.

```julia
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline alpha_m(V) = 0.4 * (V + 30) / (1 - exp(-(V + 30) / 7.2))
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline beta_m(V) = 0.124 * (V + 30) / (exp((V + 30) / 7.2) - 1)
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline alpha_h(V) = 0.01 * (V + 45) / (exp((V + 45) / 1.5) - 1)
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline beta_h(V) = 0.03 * (V + 45) / (1 - exp(-(V + 45) / 1.5))
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline alpha_n(V, nspeedfactor = 1) = nspeedfactor * exp(-0.11 * (V - 13))
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline beta_n(V, nspeedfactor = 1) = nspeedfactor * exp(-0.08 * (V - 13))
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function ghk(V, Ca_int, Ca_ext, p_synapse)
    # Kelvin physiological temp.
    @unpack_SynapseParams p_synapse
    x = z * V * faraday / (gas * (temp_rates + 273.15))
    v = z * x * faraday * ((Ca_int) - (Ca_ext) * exp(-x)) / (1 - exp(-x))
    return v
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline SK_chnnl(Ca) = Ca^6 / (Ca^6 + 0.333^6) # 0.333
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function mollifier(t, duration::T; pw = 20) where {T}
    if abs(t/duration) > 10
        return zero(T)
    else
        return one(T) / (one(T) + (t/duration)^pw)
    end
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function inputBaP(t, bapTimes::Vector, duration::T, amp::T) where {T}
    if isempty(bapTimes)
        return zero(T)
    end
    res = zero(T)
    Δ = duration / 2
    for ts in bapTimes
        res += mollifier(t - (ts + Δ), Δ)
    end
    return res * amp
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline rates_adapt(a, b, c, d, Ca) = a * b / (c + d * Ca)
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline B(v, Mg) = 1 / (1 + exp(-0.062 * v) * (Mg / 3.57)) #Jahr Stevens
```

```
B (generic function with 1 method)
```





The continuous evolution.

```julia
# adapted from SynapseElife/src/SynapseModel.jl
function F_synapse(dxc, xc, xd, p_synapse::SynapseParams, t, events_bap, bap_by_epsp)
    @unpack_SynapseParams p_synapse

    # Stochastic channels/receptors
    n1_ampa = xd[14] # ampa subconductance 1
    n2_ampa = xd[15] # ampa subconductance 2
    n3_ampa = xd[16] # ampa subconductance 3
    n1_nmda_A = xd[22] # nmda subconductance 1
    n2_nmda_A = xd[23] # nmda subconductance 2
    n1_nmda_B = xd[44] # nmda subconductance 1
    n2_nmda_B = xd[45] # nmda subconductance 2
    n_car = xd[28] # vgcc-R opened state
    n_cat = xd[32] # vgcc-T opened state
    n_cal = xd[34] + xd[35] # vgcc-L opened states
    n_gaba1 = xd[49] # GABA opened state
    n_gaba2 = xd[50] # GABA opened state

    # Continuous variables
    Vsp,
    Vdend,
    Vsoma,
    λ,
    ImbufCa,
    Ca,
    Dye,
    CaM0,
    CaM2C,
    CaM2N,
    CaM4,
    mCaN,
    CaN4,
    mKCaM,
    KCaM0,
    KCaM2N,
    KCaM2C,
    KCaM4,
    PCaM0,
    PCaM2C,
    PCaM2N,
    PCaM4,
    P,
    P2,
    LTD,
    LTP,
    act_D,
    act_P,
    m,
    h,
    n,
    SK,
    λ_age,
    λ_aux = xc

    # Plasticity prediction regions
    CaMKII = KCaM0 + KCaM2C + KCaM2N + KCaM4 + PCaM0 + PCaM2C + PCaM2N + PCaM4 + P + P2
    CaN = CaN4

    # Activation when it is inside the region
    # this following line allocates 74.779 ns (3 allocations: 144 bytes). The 2 lines count for 25% of the performance
    ∂LTD = SVector(CaN, CaMKII) ∈ LTD_region
    ∂LTP = SVector(CaN, CaMKII) ∈ LTP_region

    ∂act_D = a_D * ∂LTD - b_D * act_D * (1 - ∂LTD)
    ∂act_P = a_P * ∂LTP - b_P * act_P * (1 - ∂LTP)

    # Na channel
    m_inf = alpha_m(Vsoma) / (alpha_m(Vsoma) + beta_m(Vsoma))
    m_tau = 1 / (alpha_m(Vsoma) + beta_m(Vsoma))
    ∂m = (m_inf - m) / m_tau
    ∂h = alpha_h(Vsoma) * (1 - h) - beta_h(Vsoma) * h
    I_Na = gamma_Na * (m^3) * h * (Erev_Na - Vsoma)


    # K channel
    n_inf = 1 / (1 + alpha_n(Vsoma))
    n_tau = max(50 * beta_n(Vsoma) / (1 + alpha_n(Vsoma)), 2.0)
    ∂n = (n_inf - n) / n_tau
    I_K = gamma_K * n * (Erev_K - Vsoma)

    # NMDA
    NMDA = (n1_nmda_A + n2_nmda_A + n1_nmda_B + n2_nmda_B) * B(Vsp, Mg) * gamma_nmda
    Inmda = (Erev_nmda - Vsp) * NMDA # current nmda

    # AMPA
    Iampa =
        (Erev_ampa - Vsp) *
        (gamma_ampa1 * n1_ampa + gamma_ampa2 * n2_ampa + gamma_ampa3 * n3_ampa) # current ampa

    # GABA
    Igaba = (n_gaba1 + n_gaba2) * (Erev_Cl - Vdend) * gamma_GABA

    # Calcium sources (VGCCs currents, and NMDA calcium contribution)
    ΦCa = perm * ghk(Vsp, Ca, Ca_ext, p_synapse) #GHK factor
    Ica_nmda = f_Ca * ΦCa * NMDA
    Icar = gamma_CaR * n_car * ΦCa
    Icat = gamma_CaT * n_cat * ΦCa
    Ical = gamma_CaL * n_cal * ΦCa

    # SK channel (not stochastic)
    ∂SK = (SK_chnnl(Ca) * frwd_SK - SK) / (SK_time * bcwd_SK) #SK spine
    Isk = SK_gamma * (SK_Erev - Vsp) * SK * N_SK

    # Backpropgation
    # Post input - for experimentally induced BaPs and those induced by EPSPs
    I_BaP =
        inputBaP(t, bap_by_epsp, injbap, I_clamp) + inputBaP(t, events_bap, injbap, I_clamp)
    # Bap decay/attenuation - two component for adaptation in the Bap
    ∂λ = (1 - λ) / trec - delta_decay * (1 / λ_aux) * λ * I_BaP
    ∂λ_aux = (1 - λ_aux) / trec - delta_aux * λ_aux * I_BaP
    gadapt = λ * g_diff * ϕ_dist

    # Bap decay/attenuation - age dependent modification factor
    ∂λ_age = (1 - λ_age) / trec_soma - delta_soma * λ_age * I_BaP

    # Voltage
    # Spine
    ∂Vsp =
        (
            Isk +
            Inmda +
            Iampa +
            Icat +
            Icar +
            Ical +
            g_neck * (Vdend - Vsp) +
            g_leak * (E_leak - Vsp)
        ) / (Csp)
    # Dendrite
    ∂Vdend =
        (
            g_neck * (Vsp - Vdend) +
            Igaba +
            g_leakdend * (E_leak - Vdend) +
            gadapt * (Vsoma - Vdend)
        ) / Cdend
    # Soma
    ∂Vsoma =
        (
            (I_BaP + I_Na) * λ_age +
            I_K +
            g_leaksoma * (E_leak - Vsoma) +
            gadapt * (Vdend - Vsoma)
        ) / Csoma


    # Buffer and dye (spine only - no neck diffusion)
    ∂ImbufCa = Imbuf_k_on * (Imbuf_con - ImbufCa) * Ca - Imbuf_k_off * ImbufCa
    ∂Dye = 4 * fluo5f_kf * (fluo5f_con - Dye) * Ca - 8 * fluo5f_kb * Dye

    # Ca Downstream
    # CaM-KCaM-rates (coarsed model) from Pepke adapted by
    kf_2C = rates_adapt(kon_1C, kon_2C, koff_1C, kon_2C, Ca)
    kb_2C = rates_adapt(koff_1C, koff_2C, koff_1C, kon_2C, Ca)
    kf_2N = rates_adapt(kon_1N, kon_2N, koff_1N, kon_2N, Ca)
    kb_2N = rates_adapt(koff_1N, koff_2N, koff_1N, kon_2N, Ca)
    kf_K2C = rates_adapt(kon_K1C, kon_K2C, koff_K1C, kon_K2C, Ca)
    kb_K2C = rates_adapt(koff_K1C, koff_K2C, koff_K1C, kon_K2C, Ca)
    kf_K2N = rates_adapt(kon_K1N, kon_K2N, koff_K1N, kon_K2N, Ca)
    kb_K2N = rates_adapt(koff_K1N, koff_K2N, koff_K1N, kon_K2N, Ca)
    F = CaMKII / mKCaM_con

    ∂CaM0 =
        k2 * PCaM0 +
        kb_2C * CaM2C +
        kb_2N * CaM2N +
        kb_CaM0 * KCaM0 +
        -(1 // 2) * kf_2C * (Ca^2) * CaM0 - (1 // 2) * kf_2N * (Ca^2) * CaM0 +
        -kf_CaM0 * CaM0 * mKCaM

    ∂CaM2C =
        kb_2N * CaM4 + kb_CaM2C * KCaM2C + k2 * PCaM2C + +(1 // 2) * kf_2C * (Ca^2) * CaM0 -
        kb_2C * CaM2C - (1 // 2) * kf_2N * (Ca^2) * CaM2C + -kf_CaM2C * CaM2C * mKCaM

    ∂CaM2N =
        kb_2C * CaM4 + kb_CaM2N * KCaM2N + k2 * PCaM2N + +(1 // 2) * kf_2N * (Ca^2) * CaM0 -
        kb_2N * CaM2N - (1 // 2) * kf_2C * (Ca^2) * CaM2N + -kf_CaM2N * CaM2N * mKCaM

    ∂CaM4 =
        k2 * PCaM4 +
        kcanb * CaN4 +
        kb_CaM4 * KCaM4 +
        +(1 // 2) * kf_2N * (Ca^2) * CaM2C +
        (1 // 2) * kf_2C * (Ca^2) * CaM2N - kb_2C * CaM4 + -kb_2N * CaM4 -
        kcanf * CaM4 * mCaN - kf_CaM4 * CaM4 * mKCaM

    ∂mCaN = kcanb * CaN4 - kcanf * CaM4 * mCaN

    ∂CaN4 = kcanf * CaM4 * mCaN - kcanb * CaN4

    ∂mKCaM =
        kb_CaM0 * KCaM0 +
        k3 * P +
        kb_CaM2C * KCaM2C +
        kb_CaM2N * KCaM2N +
        +kb_CaM4 * KCaM4 - kf_CaM0 * CaM0 * mKCaM - kf_CaM2C * CaM2C * mKCaM +
        -kf_CaM2N * CaM2N * mKCaM - kf_CaM4 * CaM4 * mKCaM

    ∂KCaM0 =
        kb_K2C * KCaM2C + kb_K2N * KCaM2N + kf_CaM0 * CaM0 * mKCaM + -kb_CaM0 * KCaM0 -
        (1 // 2) * kf_K2C * (Ca^2) * KCaM0 - F * k1 * KCaM0 +
        -(1 // 2) * kf_K2N * (Ca^2) * KCaM0

    ∂KCaM2N =
        kb_K2C * KCaM4 + kf_CaM2N * CaM2N * mKCaM + +(1 // 2) * kf_K2N * (Ca^2) * KCaM0 -
        kb_CaM2N * KCaM2N - kb_K2N * KCaM2N + -(1 // 2) * kf_K2C * (Ca^2) * KCaM2N -
        F * k1 * KCaM2N

    ∂KCaM2C =
        kb_K2N * KCaM4 + kf_CaM2C * CaM2C * mKCaM + +(1 // 2) * kf_K2C * (Ca^2) * KCaM0 -
        kb_CaM2C * KCaM2C - kb_K2C * KCaM2C + -F * k1 * KCaM2C -
        (1 // 2) * kf_K2N * (Ca^2) * KCaM2C

    ∂KCaM4 =
        kf_CaM4 * CaM4 * mKCaM +
        (1 // 2) * kf_K2C * (Ca^2) * KCaM2N +
        +(1 // 2) * kf_K2N * (Ca^2) * KCaM2C - kb_CaM4 * KCaM4 - kb_K2C * KCaM4 +
        -kb_K2N * KCaM4 - F * k1 * KCaM4

    ∂PCaM0 = F * k1 * KCaM0 - k2 * PCaM0

    ∂PCaM2N = F * k1 * KCaM2N - k2 * PCaM2N

    ∂PCaM2C = F * k1 * KCaM2C - k2 * PCaM2C

    ∂PCaM4 = F * k1 * KCaM4 - k2 * PCaM4

    ∂P = k2 * PCaM0 + k5 * P2 + k2 * PCaM2C + k2 * PCaM2N + k2 * PCaM4 - k3 * P - k4 * P

    ∂P2 = k4 * P - k5 * P2

    # Postsynaptic Ca
    ∂Ca =
        (Ca_infty - Ca) / tau_ca +
        +(Ica_nmda + Icar + Ical + Icat) / (2 * faraday * A_sp) +
        +(max(Ca_infty, Ca / 3) - Ca) / tau_diff +
        -∂ImbufCa +
        -∂Dye +
        +2kb_2C * CaM2C +
        2kb_2C * CaM4 +
        2kb_2N * CaM2N +
        2kb_2N * CaM4 +
        +2kb_K2C * KCaM2C +
        2kb_K2N * KCaM2N +
        2kb_K2C * KCaM4 +
        2kb_K2N * KCaM4 +
        -kf_2C * (Ca^2) * CaM0 - kf_2N * (Ca^2) * CaM0 - kf_2N * (Ca^2) * CaM2C +
        -kf_2C * (Ca^2) * CaM2N - kf_K2C * (Ca^2) * KCaM0 - kf_K2C * (Ca^2) * KCaM2N +
        -kf_K2N * (Ca^2) * KCaM0 - kf_K2N * (Ca^2) * KCaM2C

    # dxc update
    dxc[1] = ∂Vsp
    dxc[2] = ∂Vdend
    dxc[3] = ∂Vsoma
    dxc[4] = ∂λ
    dxc[5] = ∂ImbufCa
    dxc[6] = ∂Ca
    dxc[7] = ∂Dye
    dxc[8] = ∂CaM0
    dxc[9] = ∂CaM2C
    dxc[10] = ∂CaM2N
    dxc[11] = ∂CaM4
    dxc[12] = ∂mCaN
    dxc[13] = ∂CaN4
    dxc[14] = ∂mKCaM
    dxc[15] = ∂KCaM0
    dxc[16] = ∂KCaM2N
    dxc[17] = ∂KCaM2C
    dxc[18] = ∂KCaM4
    dxc[19] = ∂PCaM0
    dxc[20] = ∂PCaM2C
    dxc[21] = ∂PCaM2N
    dxc[22] = ∂PCaM4
    dxc[23] = ∂P
    dxc[24] = ∂P2
    dxc[25] = ∂LTD
    dxc[26] = ∂LTP
    dxc[27] = ∂act_D
    dxc[28] = ∂act_P
    dxc[29] = ∂m
    dxc[30] = ∂h
    dxc[31] = ∂n
    dxc[32] = ∂SK
    dxc[33] = ∂λ_age
    dxc[34] = ∂λ_aux
end
```

```
F_synapse (generic function with 1 method)
```





### Discrete jumps. 

For the PDMP package we need to define the rates in which the jumps occur.

Helper functions.

```julia
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function rates_m_r(Vsp)
    beta_m_r_star = 1 / (4e-1) # /ms
    minf_m_r_star = 1 / (1 + exp((3 - 10) / 8))
    alpha_m_r_star = beta_m_r_star * minf_m_r_star / (1 - minf_m_r_star)
    tau_m_r = 1 / (alpha_m_r_star + beta_m_r_star)
    minf_r = 1 / (1 + exp((3 - Vsp) / 8))
    alpha_m_r = minf_r / tau_m_r
    beta_m_r = (1 - minf_r) / tau_m_r
    return alpha_m_r, beta_m_r
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function rates_h_r(Vsp)
    tau_h_r = 100 # ms
    hinf_r = 1 / (1 + exp((Vsp + 39) / 9.2))
    alpha_h_r = hinf_r / tau_h_r
    beta_h_r = (1 - hinf_r) / tau_h_r
    return alpha_h_r, beta_h_r
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function rates_m_t(Vsp)
    beta_m_t_star = 1 # /ms
    minf_m_t_star = 1 / (1 + exp((-32 + 20) / 7))
    alpha_m_t_star = beta_m_t_star * minf_m_t_star / (1 - minf_m_t_star)
    tau_m_t = 1 / (alpha_m_t_star + beta_m_t_star)
    minf_t = 1 / (1 + exp((-32 - Vsp) / 7))
    alpha_m_t = minf_t / tau_m_t
    beta_m_t = (1 - minf_t) / tau_m_t
    return alpha_m_t, beta_m_t
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function rates_h_t(Vsp)
    tau_h_t = 50 # ms
    hinf_t = 1 / (1 + exp((Vsp + 70) / 6.5))
    alpha_h_t = hinf_t / tau_h_t
    beta_h_t = (1 - hinf_t) / tau_h_t
    return alpha_h_t, beta_h_t
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function rates_l(Vsp)
    return 0.83 / (1 + exp((13.7 - Vsp) / 6.1)),
    0.53 / (1 + exp((Vsp - 11.5) / 6.4)),
    1.86 / (1 + exp((Vsp - 18.8) / 6.17))
end
# adapted from SynapseElife/src/UtilsDynamics.jl
@inline function plasticityRate(p, nhill, K)
    Pmax = 1
    r = p^nhill
    return Pmax * r / (r + K^nhill)
end
```

```
plasticityRate (generic function with 1 method)
```





The rate function.

```julia
# adapted from SynapseElife/src/SynapseModel.jl
function R_synapse(rate, xc, xd, p_synapse::SynapseParams, t, sum_rate, glu = 0)

    @unpack_SynapseParams p_synapse

    # Voltage
    Vsp = xc[1]

    # Glutamate & GABA
    Glu = glu_amp * glu

    # AMPA
    #2line-GO
    rate[1] = 4 * AMPA_k1 * Glu * xd[1]
    rate[2] = 3 * AMPA_k1 * Glu * xd[2]
    rate[3] = 2 * AMPA_k1 * Glu * xd[3]
    rate[4] = 1 * AMPA_k1 * Glu * xd[4]
    #2line-BACK
    rate[5] = 4 * AMPA_k_1 * xd[5]
    rate[6] = 3 * AMPA_k_1 * xd[4]
    rate[7] = 2 * AMPA_k_1 * xd[3]
    rate[8] = 1 * AMPA_k_1 * xd[2]
    #3line-GO
    rate[9] = 3 * AMPA_k1 * Glu * xd[6]
    rate[10] = 3 * AMPA_k1 * Glu * xd[7]
    rate[11] = 2 * AMPA_k1 * Glu * xd[8]
    rate[12] = 1 * AMPA_k1 * Glu * xd[9]
    #3line-BACK
    rate[13] = 3 * AMPA_k_1 * xd[10]
    rate[14] = 2 * AMPA_k_1 * xd[9]
    rate[15] = 1 * AMPA_k_1 * xd[8]
    rate[16] = 1 * AMPA_k_2 * xd[7]
    #4line-GO
    rate[17] = 2 * AMPA_k1 * Glu * xd[11]
    rate[18] = 1 * AMPA_k1 * Glu * xd[12]
    #4line-BACK
    rate[19] = 2 * AMPA_k_1 * xd[13]
    rate[20] = 1 * AMPA_k_1 * xd[12]
    #1column-GO-BACK
    rate[21] = 4 * AMPA_delta_0 * xd[1]
    rate[22] = 1 * AMPA_gamma_0 * xd[6]
    #2column-GO-BACK
    rate[23] = 1 * AMPA_delta_1 * xd[2]
    rate[24] = 1 * AMPA_gamma_1 * xd[7]
    #3column-GO
    rate[25] = 1 * AMPA_alpha * xd[14]
    rate[26] = 2 * AMPA_delta_1 * xd[3]
    rate[27] = 1 * AMPA_delta_2 * xd[8]
    #3column-BACK
    rate[28] = 1 * AMPA_gamma_2 * xd[11]
    rate[29] = 1 * AMPA_gamma_1 * xd[8]
    rate[30] = 2 * AMPA_beta * xd[3]
    #4column-GO
    rate[31] = 1 * AMPA_alpha * xd[15]
    rate[32] = 3 * AMPA_delta_1 * xd[4]
    rate[33] = 2 * AMPA_delta_2 * xd[9]
    #4column-BACK
    rate[34] = 1 * AMPA_gamma_2 * xd[12]
    rate[35] = 1 * AMPA_gamma_1 * xd[9]
    rate[36] = 2 * AMPA_beta * xd[4]
    #5column-GO
    rate[37] = 1 * AMPA_alpha * xd[16]
    rate[38] = 4 * AMPA_delta_1 * xd[5]
    rate[39] = 3 * AMPA_delta_2 * xd[10]
    #5column-BACK
    rate[40] = 1 * AMPA_gamma_2 * xd[13]
    rate[41] = 1 * AMPA_gamma_1 * xd[10]
    rate[42] = 4 * AMPA_beta * xd[5]

    # NMDA
    #1line-GO
    rate[43] = NMDA_N2A_ka * xd[17] * Glu
    rate[44] = NMDA_N2A_kb * xd[18] * Glu
    rate[45] = NMDA_N2A_kc * xd[19]
    rate[46] = NMDA_N2A_kd * xd[20]
    rate[47] = NMDA_N2A_ke * xd[21]
    rate[48] = NMDA_N2A_kf * xd[22]
    #1line-BACK
    rate[49] = NMDA_N2A_k_f * xd[23]
    rate[50] = NMDA_N2A_k_e * xd[22]
    rate[51] = NMDA_N2A_k_d * xd[21]
    rate[52] = NMDA_N2A_k_c * xd[20]
    rate[53] = NMDA_N2A_k_b * xd[19]
    rate[54] = NMDA_N2A_k_a * xd[18]

    # Sampling
    rate[55] = sampling_rate

    # R-type VGCC
    alpha_m_r, beta_m_r = rates_m_r(Vsp)
    alpha_h_r, beta_h_r = rates_h_r(Vsp)
    rate[56] = xd[25] * alpha_m_r * frwd_VGCC
    rate[57] = xd[26] * beta_m_r * bcwd_VGCC
    rate[58] = xd[25] * alpha_h_r * frwd_VGCC
    rate[59] = xd[27] * beta_h_r * bcwd_VGCC
    rate[60] = xd[26] * alpha_h_r * frwd_VGCC
    rate[61] = xd[28] * beta_h_r * bcwd_VGCC
    rate[62] = xd[27] * alpha_m_r * frwd_VGCC
    rate[63] = xd[28] * beta_m_r * bcwd_VGCC

    # T-type VGCC
    alpha_m_t, beta_m_t = rates_m_t(Vsp)
    alpha_h_t, beta_h_t = rates_h_t(Vsp)
    rate[64] = xd[29] * alpha_m_t * frwd_VGCC
    rate[65] = xd[30] * beta_m_t * bcwd_VGCC # this one can have a high rate
    rate[66] = xd[29] * alpha_h_t * frwd_VGCC
    rate[67] = xd[31] * beta_h_t * bcwd_VGCC
    rate[68] = xd[30] * alpha_h_t * frwd_VGCC
    rate[69] = xd[32] * beta_h_t * bcwd_VGCC
    rate[70] = xd[31] * alpha_m_t * frwd_VGCC
    rate[71] = xd[32] * beta_m_t * bcwd_VGCC # this one can have a high rate

    # L-type VGCC
    alpha_l, beta_1_l, beta_2_l = rates_l(Vsp)
    rate[72] = xd[33] * alpha_l * frwd_VGCC
    rate[73] = xd[34] * beta_1_l * bcwd_VGCC
    rate[74] = xd[33] * alpha_l * frwd_VGCC
    rate[75] = xd[35] * beta_2_l * bcwd_VGCC

    # LTD/LTP
    #the 6 lines take 50ns on 200ns, 1/4 of computations are here!!
    D_rate = plasticityRate(xc[27], 2, K_D) / t_D
    P_rate = plasticityRate(xc[28], 2, K_P) / t_P
    rate[76] = xd[36] * D_rate
    rate[77] = xd[37] * P_rate
    rate[78] = xd[36] * P_rate
    rate[79] = xd[38] * D_rate

    # NMDA GLUN2B
    #1line-GO
    rate[80] = NMDA_N2B_sa * xd[39] * Glu
    rate[81] = NMDA_N2B_sb * xd[40] * Glu
    rate[82] = NMDA_N2B_sc * xd[41]
    rate[83] = NMDA_N2B_sd * xd[42]
    rate[84] = NMDA_N2B_se * xd[43]
    rate[85] = NMDA_N2B_sf * xd[44]

    #1line-BACK
    rate[86] = NMDA_N2B_s_f * xd[45]
    rate[87] = NMDA_N2B_s_e * xd[44]
    rate[88] = NMDA_N2B_s_d * xd[43]
    rate[89] = NMDA_N2B_s_c * xd[42]
    rate[90] = NMDA_N2B_s_b * xd[41]
    rate[91] = NMDA_N2B_s_a * xd[40]

    # GABA
    rate[92] = GABA_r_b1 * xd[46] * Glu #to simplify, we use the same amount at the same time
    rate[93] = GABA_r_u1 * xd[47]
    rate[94] = GABA_r_b2 * xd[47] * Glu
    rate[95] = GABA_r_u2 * xd[48]
    rate[96] = GABA_r_ro1 * xd[47]
    rate[97] = GABA_r_c1 * xd[49]
    rate[98] = GABA_r_ro2 * xd[48]
    rate[99] = GABA_r_c2 * xd[50]

    bound = 0.0
    if sum_rate == false
        return 0.0, bound
    else
        return sum(rate), bound
    end
end
```

```
R_synapse (generic function with 2 methods)
```





Alternatively, for the JumpProcesses packages we need to define the jumps. First we define a macro to aid us in the definition of the jumps.

```julia
"""
Macro to help with defining the Synapse problem for the JumpProcesses package, used in the `J_synapse` function body.

# Arguments
- i : Jump index
- p_synapse : Synapse parameters
- rate_ex : Rate as Julia expression
- urate_ex : Rate upper bound as a Julia expression 
- rateinterval_ex : Rate interval as a Julia expression
"""
macro j_jump(i, p_synapse, nu, rate_ex, urate_ex = nothing, rateinterval_ex = nothing)

    assignments = Expr[]

    alpha_beta_regex = r"(alpha|beta)_(m_r|h_r|m_t|h_t|l|1_l|2_l)"
    alpha_beta_matches = Set([m.match for m in eachmatch(alpha_beta_regex, "$rate_ex")])

    if length(alpha_beta_matches) > 0

        for m in ("alpha_1_l", "alpha_2_l", "beta_l")
            if m in alpha_beta_matches
                throw(DomainError(m, "this variable does not exist in the model."))
            end
        end

        push!(assignments, :(Vsp = u[1]))

        if "alpha_m_r" in alpha_beta_matches || "beta_m_r" in alpha_beta_matches
            push!(assignments, :((alpha_m_r, beta_m_r) = rates_m_r(Vsp)))
        end

        if "alpha_h_r" in alpha_beta_matches || "beta_h_r" in alpha_beta_matches
            push!(assignments, :((alpha_h_r, beta_h_r) = rates_h_r(Vsp)))
        end

        if "alpha_m_t" in alpha_beta_matches || "beta_m_t" in alpha_beta_matches
            push!(assignments, :((alpha_m_t, beta_m_t) = rates_m_t(Vsp)))
        end

        if "alpha_h_t" in alpha_beta_matches || "beta_h_t" in alpha_beta_matches
            push!(assignments, :((alpha_h_t, beta_h_t) = rates_h_t(Vsp)))
        end

        if "alpha_l" in alpha_beta_matches ||
           "beta_1_l" in alpha_beta_matches ||
           "beta_2_l" in alpha_beta_matches
            push!(assignments, :((alpha_l, beta_1_l, beta_2_l) = rates_l(Vsp)))
        end

    end

    if occursin("D_rate", "$rate_ex")
        push!(assignments, :(D_rate = plasticityRate(u[27], 2, K_D) / t_D))
    end

    if occursin("P_rate", "$rate_ex")
        push!(assignments, :(P_rate = plasticityRate(u[28], 2, K_D) / t_P))
    end

    ex = Expr[]

    push!(
        ex,
        quote
            @unpack_SynapseParams $(esc(p_synapse))
            @inline @inbounds function rate(u, p, t)
                $(assignments...)
                return $rate_ex
            end
            @inline @inbounds function affect!(integrator)
                for (j, a) in zip(findnz($(esc(nu))[$(esc(i)), :])...)
                    integrator.p.xd[j] += a
                end
            end
        end,
    )

    if !isnothing(urate_ex)
        push!(ex, quote
            max_m_r = rates_m_r(1_000.0)[1]
            max_h_r = rates_h_r(1_000.0)[2]
            max_m_t = rates_m_t(1_000.0)[1]
            max_h_t = rates_h_t(1_000.0)[2]
            max_alpha_l = rates_l(1_000)[1]
            max_beta_1_l, max_beta_2_l = rates_l(-1_000)[2:3]
            @inline @inbounds function urate(u, p, t)
                return $urate_ex
            end
            @inline @inbounds function rateinterval(u, p, t)
                return $rateinterval_ex
            end
        end)
    end

    if isnothing(urate_ex)
        push!(ex, :(ConstantRateJump(rate, affect!)))
    else
        push!(
            ex,
            :(VariableRateJump(rate, affect!; urate = urate, rateinterval = rateinterval)),
        )
    end

    quote
        $(ex...)
    end

end
```

```
Main.var"##WeaveSandBox#225".@j_jump
```





Next, we define the jumps themselves.

```julia
function J_synapse(p_synapse::SynapseParams, nu)

    # we order the jumps in their order they appear in the dependency graph
    jumps = JumpSet(;
        constant_jumps = [
            # AMPA
            #2line-GO
            @j_jump(1, p_synapse, nu, 4 * AMPA_k1 * p.Glu * p.xd[1]), # 1
            @j_jump(2, p_synapse, nu, 3 * AMPA_k1 * p.Glu * p.xd[2]), # 2
            @j_jump(3, p_synapse, nu, 2 * AMPA_k1 * p.Glu * p.xd[3]), # 3
            @j_jump(4, p_synapse, nu, 1 * AMPA_k1 * p.Glu * p.xd[4]), # 4
            #2line-BACK
            @j_jump(5, p_synapse, nu, 4 * AMPA_k_1 * p.xd[5]), # 5
            @j_jump(6, p_synapse, nu, 3 * AMPA_k_1 * p.xd[4]), # 6
            @j_jump(7, p_synapse, nu, 2 * AMPA_k_1 * p.xd[3]), # 7
            @j_jump(8, p_synapse, nu, 1 * AMPA_k_1 * p.xd[2]), # 8
            #3line-GO
            @j_jump(9, p_synapse, nu, 3 * AMPA_k1 * p.Glu * p.xd[6]), # 9
            @j_jump(10, p_synapse, nu, 3 * AMPA_k1 * p.Glu * p.xd[7]), # 10
            @j_jump(11, p_synapse, nu, 2 * AMPA_k1 * p.Glu * p.xd[8]), # 11
            @j_jump(12, p_synapse, nu, 1 * AMPA_k1 * p.Glu * p.xd[9]), # 12
            #3line-BACK
            @j_jump(13, p_synapse, nu, 3 * AMPA_k_1 * p.xd[10]), # 13
            @j_jump(14, p_synapse, nu, 2 * AMPA_k_1 * p.xd[9]), # 14
            @j_jump(15, p_synapse, nu, 1 * AMPA_k_1 * p.xd[8]), # 15
            @j_jump(16, p_synapse, nu, 1 * AMPA_k_2 * p.xd[7]), # 16
            #4line-GO
            @j_jump(17, p_synapse, nu, 2 * AMPA_k1 * p.Glu * p.xd[11]), # 17
            @j_jump(18, p_synapse, nu, 1 * AMPA_k1 * p.Glu * p.xd[12]), # 18
            #4line-BACK
            @j_jump(19, p_synapse, nu, 2 * AMPA_k_1 * p.xd[13]), # 19
            @j_jump(20, p_synapse, nu, 1 * AMPA_k_1 * p.xd[12]), # 20
            #1column-GO-BACK
            @j_jump(21, p_synapse, nu, 4 * AMPA_delta_0 * p.xd[1]), # 21
            @j_jump(22, p_synapse, nu, 1 * AMPA_gamma_0 * p.xd[6]), # 22
            #2column-GO-BACK
            @j_jump(23, p_synapse, nu, 1 * AMPA_delta_1 * p.xd[2]), # 23
            @j_jump(24, p_synapse, nu, 1 * AMPA_gamma_1 * p.xd[7]), # 24
            #3column-GO
            @j_jump(25, p_synapse, nu, 1 * AMPA_alpha * p.xd[14]), # 25
            @j_jump(26, p_synapse, nu, 2 * AMPA_delta_1 * p.xd[3]), # 26
            @j_jump(27, p_synapse, nu, 1 * AMPA_delta_2 * p.xd[8]), # 27
            #3column-BACK
            @j_jump(28, p_synapse, nu, 1 * AMPA_gamma_2 * p.xd[11]), # 28
            @j_jump(29, p_synapse, nu, 1 * AMPA_gamma_1 * p.xd[8]), # 29
            @j_jump(30, p_synapse, nu, 2 * AMPA_beta * p.xd[3]), # 30
            #4column-GO
            @j_jump(31, p_synapse, nu, 1 * AMPA_alpha * p.xd[15]), # 31
            @j_jump(32, p_synapse, nu, 3 * AMPA_delta_1 * p.xd[4]), # 32
            @j_jump(33, p_synapse, nu, 2 * AMPA_delta_2 * p.xd[9]), # 33
            #4column-BACK
            @j_jump(34, p_synapse, nu, 1 * AMPA_gamma_2 * p.xd[12]), # 34
            @j_jump(35, p_synapse, nu, 1 * AMPA_gamma_1 * p.xd[9]), # 35
            @j_jump(36, p_synapse, nu, 2 * AMPA_beta * p.xd[4]), # 36
            #5column-GO
            @j_jump(37, p_synapse, nu, 1 * AMPA_alpha * p.xd[16]), # 37
            @j_jump(38, p_synapse, nu, 4 * AMPA_delta_1 * p.xd[5]), # 38
            @j_jump(39, p_synapse, nu, 3 * AMPA_delta_2 * p.xd[10]), # 39
            #5column-BACK
            @j_jump(40, p_synapse, nu, 1 * AMPA_gamma_2 * p.xd[13]), # 40
            @j_jump(41, p_synapse, nu, 1 * AMPA_gamma_1 * p.xd[10]), # 41
            @j_jump(42, p_synapse, nu, 4 * AMPA_beta * p.xd[5]), # 42

            # NMDA
            #1line-GO
            @j_jump(43, p_synapse, nu, NMDA_N2A_ka * p.xd[17] * p.Glu), # 43
            @j_jump(44, p_synapse, nu, NMDA_N2A_kb * p.xd[18] * p.Glu), # 44
            @j_jump(45, p_synapse, nu, NMDA_N2A_kc * p.xd[19]), # 45
            @j_jump(46, p_synapse, nu, NMDA_N2A_kd * p.xd[20]), # 46
            @j_jump(47, p_synapse, nu, NMDA_N2A_ke * p.xd[21]), # 47
            @j_jump(48, p_synapse, nu, NMDA_N2A_kf * p.xd[22]), # 48
            #1line-BACK
            @j_jump(49, p_synapse, nu, NMDA_N2A_k_f * p.xd[23]), # 49
            @j_jump(50, p_synapse, nu, NMDA_N2A_k_e * p.xd[22]), # 50
            @j_jump(51, p_synapse, nu, NMDA_N2A_k_d * p.xd[21]), # 51
            @j_jump(52, p_synapse, nu, NMDA_N2A_k_c * p.xd[20]), # 52
            @j_jump(53, p_synapse, nu, NMDA_N2A_k_b * p.xd[19]), # 53
            @j_jump(54, p_synapse, nu, NMDA_N2A_k_a * p.xd[18]), # 54

            # NMDA GLUN2B
            #1line-GO
            @j_jump(80, p_synapse, nu, NMDA_N2B_sa * p.xd[39] * p.Glu), # 80
            @j_jump(81, p_synapse, nu, NMDA_N2B_sb * p.xd[40] * p.Glu), # 81
            @j_jump(82, p_synapse, nu, NMDA_N2B_sc * p.xd[41]), # 82
            @j_jump(83, p_synapse, nu, NMDA_N2B_sd * p.xd[42]), # 83
            @j_jump(84, p_synapse, nu, NMDA_N2B_se * p.xd[43]), # 84
            @j_jump(85, p_synapse, nu, NMDA_N2B_sf * p.xd[44]), # 85
            #1line-BACK
            @j_jump(86, p_synapse, nu, NMDA_N2B_s_f * p.xd[45]), # 86
            @j_jump(87, p_synapse, nu, NMDA_N2B_s_e * p.xd[44]), # 87
            @j_jump(88, p_synapse, nu, NMDA_N2B_s_d * p.xd[43]), # 88
            @j_jump(89, p_synapse, nu, NMDA_N2B_s_c * p.xd[42]), # 89
            @j_jump(90, p_synapse, nu, NMDA_N2B_s_b * p.xd[41]), # 90
            @j_jump(91, p_synapse, nu, NMDA_N2B_s_a * p.xd[40]), # 91

            # GABA
            @j_jump(92, p_synapse, nu, GABA_r_b1 * p.xd[46] * p.Glu), # 92 to simplify, we use the same amount at the same time)
            @j_jump(93, p_synapse, nu, GABA_r_u1 * p.xd[47]), # 93
            @j_jump(94, p_synapse, nu, GABA_r_b2 * p.xd[47] * p.Glu), # 94
            @j_jump(95, p_synapse, nu, GABA_r_u2 * p.xd[48]), # 95
            @j_jump(96, p_synapse, nu, GABA_r_ro1 * p.xd[47]), # 96
            @j_jump(97, p_synapse, nu, GABA_r_c1 * p.xd[49]), # 97
            @j_jump(98, p_synapse, nu, GABA_r_ro2 * p.xd[48]), # 98
            @j_jump(99, p_synapse, nu, GABA_r_c2 * p.xd[50]), # 99
        ],
        variable_jumps = [
            # R-type VGCC
            @j_jump(
                56,
                p_synapse,
                nu,
                p.xd[25] * alpha_m_r * frwd_VGCC,
                p.xd[25] * max_m_r * frwd_VGCC,
                typemax(Float64)
            ), # 56
            @j_jump(
                57,
                p_synapse,
                nu,
                p.xd[26] * beta_m_r * bcwd_VGCC,
                p.xd[26] * max_m_r * frwd_VGCC,
                typemax(Float64)
            ), # 57
            @j_jump(
                58,
                p_synapse,
                nu,
                p.xd[25] * alpha_h_r * frwd_VGCC,
                p.xd[25] * max_h_r * frwd_VGCC,
                typemax(Float64)
            ), # 58
            @j_jump(
                59,
                p_synapse,
                nu,
                p.xd[27] * beta_h_r * bcwd_VGCC,
                p.xd[27] * max_h_r * bcwd_VGCC,
                typemax(Float64)
            ), # 59
            @j_jump(
                60,
                p_synapse,
                nu,
                p.xd[26] * alpha_h_r * frwd_VGCC,
                p.xd[26] * max_h_r * frwd_VGCC,
                typemax(Float64)
            ), # 60
            @j_jump(
                61,
                p_synapse,
                nu,
                p.xd[28] * beta_h_r * bcwd_VGCC,
                p.xd[28] * max_h_r * bcwd_VGCC,
                typemax(Float64)
            ), # 61
            @j_jump(
                62,
                p_synapse,
                nu,
                p.xd[27] * alpha_m_r * frwd_VGCC,
                p.xd[27] * max_m_r * frwd_VGCC,
                typemax(Float64)
            ), # 62
            @j_jump(
                63,
                p_synapse,
                nu,
                p.xd[28] * beta_m_r * bcwd_VGCC,
                p.xd[28] * max_m_r * bcwd_VGCC,
                typemax(Float64)
            ), # 63

            # T-type VGCC
            @j_jump(
                64,
                p_synapse,
                nu,
                p.xd[29] * alpha_m_t * frwd_VGCC,
                p.xd[29] * max_m_t * frwd_VGCC,
                typemax(Float64)
            ), # 64
            @j_jump(
                65,
                p_synapse,
                nu,
                p.xd[30] * beta_m_t * bcwd_VGCC,
                p.xd[30] * max_m_t * bcwd_VGCC,
                typemax(Float64)
            ), # 65 this one can have a high rate
            @j_jump(
                66,
                p_synapse,
                nu,
                p.xd[29] * alpha_h_t * frwd_VGCC,
                p.xd[29] * max_h_t * frwd_VGCC,
                typemax(Float64)
            ), # 66
            @j_jump(
                67,
                p_synapse,
                nu,
                p.xd[31] * beta_h_t * bcwd_VGCC,
                p.xd[31] * max_h_t * bcwd_VGCC,
                typemax(Float64)
            ), # 67
            @j_jump(
                68,
                p_synapse,
                nu,
                p.xd[30] * alpha_h_t * frwd_VGCC,
                p.xd[30] * max_h_t * frwd_VGCC,
                typemax(Float64)
            ), # 68
            @j_jump(
                69,
                p_synapse,
                nu,
                p.xd[32] * beta_h_t * bcwd_VGCC,
                p.xd[32] * max_h_t * bcwd_VGCC,
                typemax(Float64)
            ), # 69
            @j_jump(
                70,
                p_synapse,
                nu,
                p.xd[31] * alpha_m_t * frwd_VGCC,
                p.xd[31] * max_m_t * frwd_VGCC,
                typemax(Float64)
            ), # 70
            @j_jump(
                71,
                p_synapse,
                nu,
                p.xd[32] * beta_m_t * bcwd_VGCC,
                p.xd[32] * max_m_t * bcwd_VGCC,
                typemax(Float64)
            ), # 71, this one can have a high rate

            # L-type VGCC
            @j_jump(
                72,
                p_synapse,
                nu,
                p.xd[33] * alpha_l * frwd_VGCC,
                p.xd[33] * max_alpha_l * frwd_VGCC,
                typemax(Float64)
            ), # 72
            @j_jump(
                73,
                p_synapse,
                nu,
                p.xd[34] * beta_1_l * bcwd_VGCC,
                p.xd[34] * max_beta_1_l * bcwd_VGCC,
                typemax(Float64)
            ), # 73
            @j_jump(
                74,
                p_synapse,
                nu,
                p.xd[33] * alpha_l * frwd_VGCC,
                p.xd[33] * max_alpha_l * frwd_VGCC,
                typemax(Float64)
            ), # 74
            @j_jump(
                75,
                p_synapse,
                nu,
                p.xd[35] * beta_2_l * bcwd_VGCC,
                p.xd[35] * max_beta_2_l * bcwd_VGCC,
                typemax(Float64)
            ), # 75

            # LTD/LTP
            @j_jump(76, p_synapse, nu, p.xd[36] * D_rate, 1, typemax(Float64)), # 76
            @j_jump(77, p_synapse, nu, p.xd[37] * P_rate, 1, typemax(Float64)), # 77
            @j_jump(78, p_synapse, nu, p.xd[36] * P_rate, 1, typemax(Float64)), # 78
            @j_jump(79, p_synapse, nu, p.xd[38] * D_rate, 1, typemax(Float64)), # 79
        ],
    )

    return jumps
end
```

```
J_synapse (generic function with 1 method)
```





`R_synapse` and `J_synapse` define the exact same discrete problem. The comments at the end of the line in `J_synapse` corresponds to the rate index defined in `R_synapse`. For example, in `R_synapse` we have `rate[1]`

```
rate[1] = 4 * AMPA_k1 * Glu * xd[1]
```

Equivalently, in `J_synapse` we have the equivalent formulation in terms of a jump using our macro.

```
@j_jump(1, p_synapse, nu, 4 * AMPA_k1 * p.Glu * p.xd[1]), # 1
```

Also, note that we modify the order of jumps in `J_synapse` compared to `R_synapse` because we bundle `ConstantRateJump` and `VariableRateJump` separately. Algorithms in `JumpProcesses` will approach each of this jumps in a different way.

We also remove the sampling rate from `J_synapse` (`rate[55]` in `R_synapse`) as it is not needed for `JumpProcesses`. The package can take a snapshot of the evolution at any point desired.

### Problem wrappers

We define wrappers for setting up and solving problems involving synapses.

First, we define the problem for the `PDMP` package.

```julia
function SynapseProblem(
    xc,
    xd,
    t1,
    t2,
    events_bap,
    bap_by_epsp,
    glu,
    p_synapse,
    nu,
    algo::T,
    agg = nothing;
    saveat = [],
    save_everystep = isempty(saveat),
    kwargs...,
) where {T<:CHV}
    problem = PDMP.PDMPProblem(
        (dxc, xc, xd, p, t) -> F_synapse(dxc, xc, xd, p, t, events_bap, bap_by_epsp),
        (rate, xc, xd, p, t, sum_rate) -> R_synapse(rate, xc, xd, p, t, sum_rate, glu),
        nu,
        xc,
        xd,
        p_synapse,
        (t1, t2);
        Ncache = 12, # this option is for AD in PreallocationTools
    )
    sol = solve(problem, algo; kwargs...)
    return sol
end
```

```
SynapseProblem (generic function with 2 methods)
```





Second, we define the problem for the `JumpProcesses` package.

We define a custom saving callback for the `JumpProcesses` problem to bring the saving behaviour as close as possible to the `PDMP` problem. Our callback will save jumps whenever the integrator steps and `save_modified == true`.

```julia
using DiffEqCallbacks: SavedValues, SavingAffect
import DataStructures

# adapted from DiffEqCallbacks.jl/src/saving.jl
function (affect!::SavingAffect)(integrator, force_save = false)
    just_saved = false
    # see OrdinaryDiffEq.jl -> integrator_utils.jl, function savevalues!
    while !isempty(affect!.saveat) &&
        integrator.tdir * first(affect!.saveat) <= integrator.tdir * integrator.t # Perform saveat
        affect!.saveiter += 1
        curt = pop!(affect!.saveat) # current time
        if curt != integrator.t # If <t, interpolate
            if integrator isa SciMLBase.AbstractODEIntegrator
                # Expand lazy dense for interpolation
                DiffEqBase.addsteps!(integrator)
            end
            if !DiffEqBase.isinplace(integrator.sol.prob)
                curu = integrator(curt)
            else
                curu = first(get_tmp_cache(integrator))
                integrator(curu, curt) # inplace since save_func allocates
            end
            copyat_or_push!(affect!.saved_values.t, affect!.saveiter, curt)
            copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                affect!.save_func(curu, curt, integrator), Val{false})
        else # ==t, just save
            just_saved = true
            copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
            copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                affect!.save_func(integrator.u, integrator.t, integrator),
                Val{false})
        end
    end
    if !just_saved && affect!.save_everystep ||
       force_save ||
       (
           affect!.save_end &&
           # ensures we don't save twice; this the only difference from the original source
           affect!.saved_values.t[affect!.saveiter] != integrator.t &&
           integrator.t == integrator.sol.prob.tspan[end]
       )
        affect!.saveiter += 1
        copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
        copyat_or_push!(
            affect!.saved_values.saveval,
            affect!.saveiter,
            affect!.save_func(integrator.u, integrator.t, integrator),
            Val{false},
        )
    end
    u_modified!(integrator, false)
end

# adapted from DiffEqCallbacks.jl/src/saving.jl
function saving_initialize!(cb, u, t, integrator)
    integrator.p.xd .= integrator.p.xd0
    cb.affect!.saveat = deepcopy(integrator.opts.saveat)
    cb.affect!.save_everystep = integrator.opts.save_everystep
    cb.affect!.save_start = integrator.opts.save_start
    cb.affect!.save_end = integrator.opts.save_end
    cb.affect!.saveiter = 0
    cb.affect!.save_start && cb.affect!(integrator, true)
end

# adapted from DiffEqCallbacks.jl/src/saving.jl
function SavingCallback(save_func, saved_values::SavedValues; save_modified = true)
    saveat_internal =
        DataStructures.BinaryHeap{eltype(saved_values.t)}(DataStructures.FasterForward())
    affect! = SavingAffect(
        save_func,
        saved_values,
        saveat_internal,
        nothing,
        false,
        false,
        false,
        0,
    )
    # only save when save_modified is true; SavingCallback from DiffEqCallbacks
    # saves every step regardless of save_modified
    condition = if save_modified
        function (u, t, integrator)
            if integrator.u_modified
                push!(affect!.saveat, t)
            end

            return true
        end
    else
        function (u, t, integrator)
            return true
        end
    end
    DiscreteCallback(
        condition,
        affect!;
        initialize = saving_initialize!,
        save_positions = (false, false),
    )
end
```

```
SavingCallback (generic function with 1 method)
```





Next, we define our problem wrapper that uses our custom saving callback.

```julia
function buildRxDependencyGraph(nu)
    numrxs, _ = size(nu)
    dep_graph = [Vector{Int}() for n = 1:(numrxs-1)]
    for rx = 1:numrxs
        if rx == 55  # no need to track the Poisson process
            continue
        end
        rx_ix = rx
        if 56 <= rx < 80
            rx_ix += 19
        elseif rx >= 80
            rx_ix -= 25
        end
        for (spec, _) in zip(findnz(nu[rx, :])...)
            # we need to reorder the indices according to the order
            # they apper in the problem
            for (dependent_rx, _) in zip(findnz(nu[:, spec])...)
                # we need to reorder the indices according to the order
                # they apper in the problem
                dependent_rx_ix = dependent_rx
                if 56 <= dependent_rx < 80
                    dependent_rx_ix += 19
                elseif dependent_rx >= 80
                    dependent_rx_ix -= 25
                end
                push!(dep_graph[rx_ix], dependent_rx_ix)
            end
        end
    end
    return dep_graph
end

function SynapseProblem(
    xc,
    xd,
    t1,
    t2,
    events_bap,
    bap_by_epsp,
    glu,
    p_synapse,
    nu,
    algo,
    agg;
    jumps = nothing,
    save_positions = (false, true),
    saveat = [],
    save_everystep = isempty(saveat),
    kwargs...,
)
    p = (
        xd0 = copy(xd),
        xd = copy(xd),
        Glu = p_synapse.glu_amp * glu,
        p_synapse = p_synapse,
    )
    oprob = ODEProblem(
        (dxc, xc, p, t) ->
            F_synapse(dxc, xc, p.xd, p.p_synapse, t, events_bap, bap_by_epsp),
        xc,
        (t1, t2),
        p,
    )
    xdsol = SavedValues(typeof(t1), typeof(xd))
    dep_graph = buildRxDependencyGraph(nu)
    callback = SavingCallback(
        (u, t, integrator) -> copy(integrator.p.xd),
        xdsol;
        save_modified = typeof(save_positions) <: Bool ? save_positions : save_positions[2],
    )
    jprob = JumpProblem(
        oprob,
        agg,
        jumps;
        dep_graph,
        save_positions,
        saveat,
        save_everystep,
        callback,
    )
    sol = (xcsol = solve(jprob, algo; saveat, save_everystep, kwargs...), xdsol = xdsol)
    return sol
end
```

```
SynapseProblem (generic function with 3 methods)
```





### Assembling it all

We define functions to run the evolution of the whole synapse. The evolution includes a period before glutamate is released and after.

```julia
# adapted from SynapseElife/src/SynapseModel.jl
function evolveSynapse(
    xc0::Vector{T},
    xd0,
    p_synapse::SynapseParams,
    events_sorted_times,
    is_pre_or_post_event,
    bap_by_epsp,
    is_glu_released,
    nu,
    algos,
    agg = nothing;
    progress = false,
    abstol = 1e-8,
    reltol = 1e-7,
    save_positions = (false, true),
    saveat = [],
    kwargs...,
) where {T}

    tt, XC, XD = evolveSynapse_noformat(
        xc0,
        xd0,
        p_synapse,
        events_sorted_times,
        is_pre_or_post_event,
        bap_by_epsp,
        is_glu_released,
        nu,
        algos,
        agg;
        progress,
        abstol,
        reltol,
        save_positions,
        saveat,
        kwargs...,
    )

    out = formatSynapseResult(tt, XC, XD)
end

# adapted from SynapseElife/src/SynapseModel.jl
function evolveSynapse_noformat(
    xc0::Vector{T},
    xd0,
    p_synapse::SynapseParams,
    events_sorted_times,
    is_pre_or_post_event,
    bap_by_epsp,
    is_glu_released,
    nu,
    algos,
    agg = nothing;
    progress = false,
    abstol = 1e-8,
    reltol = 1e-7,
    save_positions = (false, true),
    saveat = [],
    kwargs...,
) where {T}

    if save_positions isa Tuple{Bool,Bool}
        save_positionsON = save_positions
        save_positionsOFF = save_positions
    else
        save_positionsON = save_positions[1]
        save_positionsOFF = save_positions[2]
    end

    @assert eltype(is_pre_or_post_event) == Bool "Provide booleans for glutamate releases."
    @assert eltype(is_glu_released) == Bool "Provide booleans for glutamate indices."

    XC = VectorOfArray([xc0]) # vector to hold continuous variables
    if isnothing(agg)
        XD = VectorOfArray([xd0]) # vector to hold discrete variables
        jumps = nothing
    else
        XD = VectorOfArray([xd0])
        jumps = J_synapse(p_synapse, nu)
    end
    tt = [0.0] # vector of times

    # we collect which external events correspond to BaPs
    events_bap = events_sorted_times[is_pre_or_post_event.==false]

    # function to simulate the synapse when Glutamate is ON
    SimGluON =
        (xc, xd, t1, t2, glu) -> SynapseProblem(
            xc,
            xd,
            t1,
            t2,
            events_bap,
            bap_by_epsp,
            glu,
            p_synapse,
            nu,
            algos[1],
            agg;
            jumps,
            reltol,
            abstol,
            saveat,
            save_positions = save_positionsON,
            kwargs...,
        )

    # function to simulate the synapse when Glutamate is OFF
    SimGluOFF =
        (xc, xd, t1, t2) -> SynapseProblem(
            xc,
            xd,
            t1,
            t2,
            events_bap,
            bap_by_epsp,
            zero(T),
            p_synapse,
            nu,
            algos[2],
            agg;
            jumps,
            reltol,
            abstol,
            saveat,
            save_positions = save_positionsOFF,
            kwargs...,
        )

    # random variable for Glutamate concentration
    gluDist = Gamma(1 / p_synapse.glu_cv^2, p_synapse.glu_cv^2)

    # we loop over the external events, simulate them and append to res
    for (eveindex, eve) in enumerate(events_sorted_times)
        if is_pre_or_post_event[eveindex] == true # it is a pre-synaptic event
            # we simulate the synapse with Glutamate OFF until event time
            # then we put  Glutamate ON for dt = p_synapse.glu_width with variable amplitude (concentration)

            # simulate the event with Glutamate OFF
            res = SimGluOFF(XC[:, end], XD[:, end], tt[end], eve)
            formatSimResult!(res, XC, XD, tt)
            gluamp = rand(gluDist)

            # simulate the event with Glutamate ON
            # variability here
            res = SimGluON(
                XC[:, end],
                XD[:, end],
                eve,
                eve + p_synapse.glu_width,
                ifelse(is_glu_released[eveindex], gluamp, zero(T)),
            )
            formatSimResult!(res, XC, XD, tt)
        end
    end

    # reaching tend: we simulate the synapse with Glutamate OFF until simulation end time required
    # by the user. In  most protocol, this is taking most of the time.
    res = SimGluOFF(XC[:, end], XD[:, end], tt[end], p_synapse.t_end)
    formatSimResult!(res, XC, XD, tt)
    if isnothing(agg)
        @info "last bit" agg length(res.time) tt[end] p_synapse.t_end
    else
        @info "last bit" agg length(res.xcsol.t) tt[end] p_synapse.t_end
    end

    if tt[end] != p_synapse.t_end
        @warn "The simulation did not reach requested simulated time."
    end

    return (t = tt, XC = XC, XD = XD)
end

function formatSimResult!(res::PDMP.PDMPResult, XC, XD, tt)
    append!(XC, res.xc)
    append!(XD, res.xd)
    append!(tt, res.time)
    nothing
end

function formatSimResult!(res::NamedTuple, XC, XD, tt)
    append!(XC, VectorOfArray(res.xcsol.u))
    append!(XD, VectorOfArray(res.xdsol.saveval))
    append!(tt, res.xcsol.t)
    nothing
end

# adapted from SynapseElife/src/SynapseModel.jl
function formatSynapseResult(tt, XC, XD)
    namesC = (
        :Vsp,
        :Vdend,
        :Vsoma,
        :λ,
        :ImbufCa,
        :Ca,
        :Dye,
        :CaM0,
        :CaM2C,
        :CaM2N,
        :CaM4,
        :mCaN,
        :CaN4,
        :mKCaM,
        :KCaM0,
        :KCaM2N,
        :KCaM2C,
        :KCaM4,
        :PCaM0,
        :PCaM2C,
        :PCaM2N,
        :PCaM4,
        :P,
        :P2,
        :LTD,
        :LTP,
        :act_D,
        :act_P,
        :m,
        :h,
        :n,
        :SK,
        :λ_age,
        :λ_aux,
    )
    values = (XC[i, :] for i = 1:length(namesC))
    return (t = tt, XD = XD, XC = XC, zip(namesC, values)...)
end
```

```
formatSynapseResult (generic function with 1 method)
```





## Algorithms to benchmark

`CoevolveSynced` allow us to save at regular intervals. Thus, rather than saving when a jump occurs, we save at the same average frequency as obtained with `PDMP`.

```julia
const solver = AutoTsit5(Rosenbrock23());
const algorithms = (
    (
        label = "PDMP",
        agg = nothing,
        solver = (CHV(solver), CHV(solver)),
        saveat = [],
    ),
    (
        label = "Coevolve",
        agg = Coevolve(),
        solver = (solver, solver),
        saveat = 1 / p_synapse.sampling_rate,
    ),
);
```




## Example solutions

```julia
results = []

for algo in algorithms
    push!(
        results,
        evolveSynapse(
            xc0,
            xd0,
            p_synapse,
            events_sorted_times,
            is_pre_or_post_event,
            bap_by_epsp,
            [true],
            nu,
            algo.solver,
            algo.agg;
            save_positions = (false, true),
            saveat = algo.saveat,
            save_everystep = false,
        ),
    )
end
```


```julia
fig = plot(xlabel = "Voltage", ylabel = "Time");
for (i, algo) in enumerate(algorithms)
    res = results[i]
    plot!(res.t, res.Vsp, label = algo.label)
end
title!("Vsp")
```

![](figures/Synapse_22_1.png)

```julia
fig = plot(xlabel = "N", ylabel = "Time");
for (i, algo) in enumerate(algorithms)
    res = results[i]
    plot!(res.t, res.XD[1, :], label = algo.label)
end
title!("2line-Go, AMPA")
```

```
Error: BoundsError: attempt to access 10004-element Vector{Float64} at inde
x [1:22761]
```





# Benchmarking performance

```julia
bs = Vector{BenchmarkTools.Trial}()

for algo in algorithms
    push!(
        bs,
        @benchmark(
            evolveSynapse(
                xc0,
                xd0,
                p_synapse,
                events_sorted_times,
                is_pre_or_post_event,
                bap_by_epsp,
                [true],
                nu,
                $(algo).solver,
                $(algo).agg;
                save_positions = (false, true),
                saveat = $(algo).saveat,
                save_everystep = false,
            ),
            samples = 50,
            evals = 1,
            seconds = 500,
        )
    )
end
```


```julia
labels = [a.label for a in algorithms]
medtimes = [text(string(round(median(b).time/1e9, digits=3),"s"), :center, 12) for b in bs]
relmedtimes = [median(b).time for b in bs]
relmedtimes ./= relmedtimes[1]
bar(labels, relmedtimes, markeralpha=0, series_annotation=medtimes, fmt=fmt)
title!("evolveSynapse (Median time)")
```

![](figures/Synapse_25_1.png)

```julia
medmem = [text(string(round(median(b).memory/1e6, digits=3),"Mb"), :center, 12) for b in bs]
relmedmem = Float64[median(b).memory for b in bs]
relmedmem ./= relmedmem[1]
bar(labels, relmedmem, markeralpha=0, series_annotation=medmem, fmt=fmt)
title!("evolveSynapse (Median memory)")
```

![](figures/Synapse_26_1.png)



# References

[1] Y. E. Rodrigues, C. M. Tigaret, H. Marie, C. O’Donnell, and R. Veltz, "A stochastic model of hippocampal synaptic plasticity with geometrical readout of enzyme dynamics." bioRxiv, p. 2021.03.30.437703, Mar. 30, 2021. doi: 10.1101/2021.03.30.437703.

[2] Magee, J.C., Johnston, D., 1995. Characterization of single voltage-gated na+ and ca2+ channels in apical dendrites of rat ca1 pyramidal neurons. The Journal of physiology 487, 67–90.
