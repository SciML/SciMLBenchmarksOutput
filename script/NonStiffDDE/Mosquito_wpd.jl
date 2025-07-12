
using DelayDiffEq, DiffEqDevTools, Plots
using LabelledArrays, StaticArrays
gr()

# Predation parameters
pupae_pars = (a=1, h=0.002, r=0.001, V=200)
p0 = pupae_pars.r / pupae_pars.h
p1 = pupae_pars.V / (pupae_pars.a * pupae_pars.h)

# Parameter vector definition
parvec = @SLVector (
    # temp
    :phi, # PHASE
    :lambda, # A
    :mu, # M
    :gamma, # POWER
    # photoperiod
    :L, # latitude
    # oviposition
    :max_egg, # max egg raft size, R
    # gonotrophic cycle
    :q1, # KG
    :q2, # QG
    :q3, # BG
    # egg death
    :nu_0E,  # U3
    :nu_1E,  # U4
    :nu_2E, # U5
    # larvae death
    :nu_0L,  # U3
    :nu_1L,  # U4
    :nu_2L, # U5
    # pupae death
    :nu_0P,  # U3
    :nu_1P,  # U4
    :nu_2P, # U5
    # adult death
    :alpha_A, # ALPHA
    :beta_A, # BETA
    # egg maturation
    :alpha_E, # ALPHA
    :beta_E, # BETA
    # larvae maturation
    :alpha_L, # ALPHA
    :beta_L, # BETA
    # pupae maturation
    :alpha_P, # ALPHA
    :beta_P, # BETA
    # predation on pupae
    :p0,
    :p1
)

# Parameter values
parameters = parvec(
    1.4, # phi
    6.3, # lambda
    10.3, # mu
    1.21, # gamma
    51, # L
    200, # max_egg
    # gonotrophic cycle
    0.2024, # (q1) KG
    74.48, # (q2) QG
    0.2456, # (q3) BG
    # egg death
    0.0157,  # (nu_0E) U3
    20.5,  # (nu_1E) U4
    7,  # (nu_2E) U5
    # larvae death
    0.0157,  # (nu_0L) U3
    20.5,  # (nu_1L) U4
    7,  # (nu_2L) U5
    # pupae death
    0.0157,  # (nu_0P) U3
    20.5,  # (nu_1P) U4
    7,  # (nu_2P) U5
    # adult death
    2.166e-8, # (alpha_A) ALPHA
    4.483, # (beta_A) BETA
    # egg maturation
    0.0022, # (alpha_E) ALPHA
    1.77, # (beta_E) BETA
    # larvae maturation
    0.00315, # (alpha_L) ALPHA
    1.12, # (beta_L) BETA
    # pupae maturation
    0.0007109, # (alpha_P) ALPHA
    1.8865648, # (beta_P) BETA
    # predation on pupae
    p0, # p0
    p1 # p1
)

# Maximum allowed values for rates (same as FORTRAN)
death_max = 1.0 # 1.0
death_min_a = 0.01 # 0.01
gon_min = 0.0333 # 0.0333
maturation_min = 0.016667 # 0.016667

# Temperature as modified cosine function
function temperature(t, pars)
    phi = pars.phi # PHASE
    lambda = pars.lambda # A
    mu = pars.mu # M
    gamma = pars.gamma # POWER
    
    temp = 0.0
    
    if t < 0.0
        temp = (mu - lambda) + lambda * 2.0 * (0.5 * (1.0 + cos(2.0 * pi * (0.0 - phi) / 365.0)))^gamma
    else
        temp = (mu - lambda) + lambda * 2.0 * (0.5 * (1.0 + cos(2.0 * pi * (t - phi) / 365.0)))^gamma
    end
    
    return temp
end

# Photoperiod
function daylight(t, pars)
    L = pars.L # latitude (51 in thesis)
    
    # define photoperiod values
    EPS = asin(0.39795 * cos(0.2163108 + 2 * atan(0.9671396 * tan(0.00860 * (t - 3.5)))))
    NUM = sin(0.8333 * pi/ 180.0) + (sin(L * pi / 180.0) * sin(EPS))
    DEN = cos(L * pi / 180.0) * cos(EPS)
    DAYLIGHT = 24.0 - (24.0 / pi) * acos(NUM / DEN)
    
    return DAYLIGHT
end

# Diapause functions
# pp: photoperiod
function diapause_spring(pp)
    1.0 / (1.0 + exp(5.0 * (14.0 - pp)))
end

function diapause_autumn(pp)
    1.0 / (1.0 + exp(5.0 * (13.0 - pp)))
end

# Per-capita oviposition rate
# d: diapause
# G: duration of gonotrophic cycle
# pars: parameters
function oviposition(d, G, pars)
    max_egg = pars.max_egg
    
    egg_raft = d * max_egg * 0.5
    ovi = egg_raft / G
    
    return ovi
end

# Egg mortality
function death_egg_rate(temp, pars)
    nu_0E = pars.nu_0E # U3
    nu_1E = pars.nu_1E # U4
    nu_2E = pars.nu_2E # U5
    
    # calculate egg death rate
    egg_d = nu_0E * exp(((temp - nu_1E) / nu_2E)^2)
    
    if egg_d > death_max
        egg_d = death_max
    end
    
    return egg_d
end

# Larvae mortality
function death_larvae_rate(temp, pars)
    nu_0L = pars.nu_0L # U3
    nu_1L = pars.nu_1L # U4
    nu_2L = pars.nu_2L # U5
    
    # calculate larvae death rate
    larvae_d = nu_0L * exp(((temp - nu_1L) / nu_2L)^2)
    
    if larvae_d > death_max
        larvae_d = death_max
    end
    
    return larvae_d
end

# Pupal mortality
function death_pupae_rate(temp, pars)
    nu_0P = pars.nu_0P # U3
    nu_1P = pars.nu_1P # U4
    nu_2P = pars.nu_2P # U5
    
    # calculate pupae death rate
    pupal_d = nu_0P * exp(((temp - nu_1P)/nu_2P)^2)
    
    if pupal_d > death_max
        pupal_d = death_max
    end
    
    return pupal_d
end

# Adult mortality
function death_adult_rate(temp, pars)
    alpha_A = pars.alpha_A # ALPHA
    beta_A = pars.beta_A # BETA
    
    # calculate adult death rate
    adult_d = alpha_A * (temp^beta_A)
    
    if adult_d < death_min_a
        adult_d = death_min_a
    end
    return adult_d
end

# G: duration of gonotrophic cycle
function gonotrophic(temp, pars)
    q1 = pars.q1 # KG
    q2 = pars.q2 # QG
    q3 = pars.q3 # BG
    
    # calculate gonotrophic cycle length
    if temp < 0.0
        grate = 0.0333
    else
        grate = q1 / (1 + q2*exp(-q3*temp))
    end
    
    if grate < gon_min
        grate = gon_min
    end
    
    return 1.0 / grate
end

# g_E
function egg_maturation_rate(temp, pars)
    alpha_E = pars.alpha_E # ALPHA
    beta_E = pars.beta_E # BETA
    
    # calculate egg development rate
    if temp < 0.0
        egg_maturation = 0.016667
    else
        egg_maturation = alpha_E * (temp^beta_E)
    end
    
    if egg_maturation < maturation_min
        egg_maturation = maturation_min
    end
    return egg_maturation
end

# g_L
function larvae_maturation_rate(temp, pars)
    alpha_L = pars.alpha_L # ALPHA
    beta_L = pars.beta_L # BETA
    
    # calculate larvae development rate
    if temp < 0.0
        larvae_maturation = 0.016667
    else
        larvae_maturation = alpha_L * (temp^beta_L)
    end
    
    if larvae_maturation < maturation_min
        larvae_maturation = maturation_min
    end
    return larvae_maturation
end

# g_P
function pupae_maturation_rate(temp, pars)
    alpha_P = pars.alpha_P # ALPHA
    beta_P = pars.beta_P # BETA
    
    # calculate pupae development rate
    if temp < 0.0
        pupae_maturation = 0.016667
    else
        pupae_maturation = alpha_P * (temp^beta_P)
    end
    
    if pupae_maturation < maturation_min
        pupae_maturation = maturation_min
    end
    return pupae_maturation
end

# State variable history
function h(p, t; idxs = nothing)
    temp = temperature(t, p)
    
    # history vector
    Y = zeros(13)
    
    Y[8] = 1.0 / egg_maturation_rate(temp, p) # tau_E
    Y[9] = 1.0 / larvae_maturation_rate(temp, p) # tau_L
    Y[10] = 1.0 / pupae_maturation_rate(temp, p) # tau_P
    
    Y[5] = exp(-death_egg_rate(temp, p) * Y[8]) # S_E
    Y[6] = exp(-death_larvae_rate(temp, p) * Y[9]) # S_L
    Y[7] = exp(-death_pupae_rate(temp, p) * Y[10]) #S_P
    
    temp_L = temperature(t - Y[9], p)
    temp_P = temperature(t - Y[10], p)
    
    Y[11] = 1.0 / egg_maturation_rate(temp_L, p) # tau_E(t - tau_L(t))
    Y[12] = 1.0 / larvae_maturation_rate(temp_P, p) # tau_L(t - tau_P(t))
    
    temp_LP = temperature(t - Y[10] - Y[12], p)
    
    Y[13] = 1.0 / egg_maturation_rate(temp_LP, p) # tau_E(t - tau_P(t) - tau_L(t - tau_P(t)))
    
    idxs === nothing ? Y : Y[idxs]
end

# Initial condition calculation
# A0: A(0)
# t0: temp(0); assumed constant for t<0
function calculate_IC(A0, t0, pars)
    u0 = zeros(13)
    u0[4] = A0
    
    # calculate initial lags first
    u0[8] = 1.0 / egg_maturation_rate(t0, pars) # tau_E
    u0[9] = 1.0 / larvae_maturation_rate(t0, pars) # tau_L
    u0[10] = 1.0 / pupae_maturation_rate(t0, pars) # tau_P
    
    u0[11] = u0[8] # tau_E(t - tau_L(t))
    u0[12] = u0[9] # tau_L(t - tau_P(t))
    u0[13] = u0[8] # tau_E(t - tau_P(t) - tau_L(t - tau_P(t)))
    
    # survival probabilities
    u0[5] = exp(-u0[8] * death_egg_rate(t0, pars)) # S_E
    u0[6] = exp(-u0[9] * death_larvae_rate(t0, pars)) # S_L
    u0[7] = exp(-u0[10] * death_pupae_rate(t0, pars)) # S_P
    
    return u0
end

# System of DDEs
# 13 equations
# 6 delays
# DDEs
function ewing_dde(du, u, h, p, t)
    
    # state variables
    E = u[1]
    LAR = u[2]
    PUP = u[3]
    ADU = u[4]
    SE = u[5]
    SL = u[6]
    SP = u[7]
    DE = u[8] # tau_E(t)
    DL = u[9] # tau_L(t)
    DP = u[10] # tau_P(t)
    DEL = u[11] # tau_E(t - tau_L(t))
    DLP = u[12] # tau_L(t - tau_P(t))
    DELP = u[13] # tau_E(t - tau_P(t) - tau_L(t - taup_P(t)))
    
    # larval predation parameters
    p0 = p.p0
    p1 = p.p1
    
    # Z: state variables at each of the 6 lagged times (lags follow same order as Z/BETA in DDE_SOLVER)
    
    # Z
    Z1 = h(p, t - DE; idxs = 4) # Z(x,1): t - tau_E(t)
    Z2 = h(p, t - DL - DEL; idxs = 4) # Z(x,2): t - tau_L(t) - tau_E(t - tau_L(t))
    Z3 = h(p, t - DP - DLP - DELP; idxs = 4) # Z(x,3): t - tau_P(t) - tau_L(t - tau_P(t)) - tau_E(t - tau_P(t) - tau_L(t - tau_P(t)))
    Z4 = h(p, t - DL) # Z(x,4): t - tau_L(t)
    Z5 = h(p, t - DP - DLP) # Z(x,5): t - tau_P(t) - tau_L(t - tau_P(t))
    Z6 = h(p, t - DP) # Z(x,6): t - tau_P(t)
    
    # (lagged) temperature
    temp = temperature(t, p)
    temp_E = temperature(t - DE, p)
    temp_L = temperature(t - DL, p)
    temp_P = temperature(t - DP, p)
    temp_EL = temperature(t - DL - Z4[8], p)
    temp_ELP = temperature(t - DP - Z6[9] - Z5[8], p)
    temp_LP = temperature(t - DP - Z6[9], p)
    
    
    # (lagged) photoperiod
    pp = daylight(t, p)
    pp_1 = daylight(t - 1, p)
    pp_E = daylight(t - DE, p)
    pp_EL = daylight(t - DL - Z4[8], p)
    pp_ELP = daylight(t - DP - Z6[9] - Z5[8], p)
    
    # (lagged) gonotrophic cycle
    gon = gonotrophic(temp, p)
    gon_E = gonotrophic(temp_E, p)
    gon_EL = gonotrophic(temp_EL, p)
    gon_ELP = gonotrophic(temp_ELP, p)
    
    # diapause and birth
    if pp > pp_1
        dia = diapause_spring(pp)
        dia_E = diapause_spring(pp_E)
        dia_EL = diapause_spring(pp_EL)
        dia_ELP = diapause_spring(pp_ELP)
    else
        dia = diapause_autumn(pp)
        dia_E = diapause_autumn(pp_E)
        dia_EL = diapause_autumn(pp_EL)
        dia_ELP = diapause_autumn(pp_ELP)
    end
    
    birth = oviposition(dia, gon, p)
    birth_E = oviposition(dia_E, gon_E, p)
    birth_EL = oviposition(dia_EL, gon_EL, p)
    birth_ELP = oviposition(dia_ELP, gon_ELP, p)
    
    # (lagged) death
    death_egg = death_egg_rate(temp, p)
    death_egg_E = death_egg_rate(temp_E, p)
    
    death_larvae = death_larvae_rate(temp, p)
    death_larvae_L = death_larvae_rate(temp_L, p)
    
    death_pupae = death_pupae_rate(temp, p)
    death_pupae_P = death_pupae_rate(temp_P, p)
    
    death_adult = death_adult_rate(temp, p)
    
    # (lagged) development
    larvae_maturation = larvae_maturation_rate(temp, p)
    larvae_maturation_L = larvae_maturation_rate(temp_L, p)
    larvae_maturation_P = larvae_maturation_rate(temp_P, p)
    larvae_maturation_LP = larvae_maturation_rate(temp_LP, p)
    
    egg_maturation = egg_maturation_rate(temp, p)
    egg_maturation_E = egg_maturation_rate(temp_E, p)
    egg_maturation_L = egg_maturation_rate(temp_L, p)
    egg_maturation_EL = egg_maturation_rate(temp_EL, p)
    egg_maturation_LP = egg_maturation_rate(temp_LP, p)
    egg_maturation_ELP = egg_maturation_rate(temp_ELP, p)
    
    pupae_maturation = pupae_maturation_rate(temp, p)
    pupae_maturation_P = pupae_maturation_rate(temp_P, p)
    
    # DDEs describing change in state duration
    dDEdt = 1 - egg_maturation/egg_maturation_E
    dDLdt = 1 - larvae_maturation/larvae_maturation_L
    dDPdt = 1 - pupae_maturation/pupae_maturation_P
    dDELdt = (1 - dDLdt) * (1 - egg_maturation_L/egg_maturation_EL)
    dDLPdt = (1 - dDPdt) * (1 - larvae_maturation_P/larvae_maturation_LP)
    dDELPdt = (1 - dDPdt - dDLPdt) * (1 - egg_maturation_LP/egg_maturation_ELP)
    
    # stage recruitment
    R_E = birth * ADU
    R_L = birth_E * Z1 * SE * egg_maturation/egg_maturation_E
    R_P = birth_EL * Z2 * Z4[5] * SL * larvae_maturation/larvae_maturation_L * (1 - dDELdt)
    R_A = birth_ELP * Z3 * Z5[5] * Z6[6] * SP * pupae_maturation/pupae_maturation_P * (1 - dDLPdt) * (1 - dDELPdt)
    
    # maturation rates
    M_E = R_L
    M_L = R_P
    M_P = R_A
    
    # death rates
    D_E = death_egg * E
    D_L = ((p0*LAR/(p1+LAR)) + death_larvae) * LAR
    D_P = death_pupae * PUP
    D_A = death_adult * ADU
    
    # DDE system
    du_1 = R_E - M_E - D_E  # E
    du_2 = R_L - M_L - D_L  # L
    du_3 = R_P - M_P - D_P  # P
    du_4 = R_A - D_A        # A
    
    du_5 = SE * ((egg_maturation * death_egg_E / egg_maturation_E) - death_egg)
    du_6 = SL * (((p0*Z4[2] / (p1+Z4[2])) + death_larvae_L) * (1-dDLdt) - (p0*LAR / (p1+LAR)) - death_larvae)
    du_7 = SP * ((pupae_maturation * death_pupae_P / pupae_maturation_P) - death_pupae)
    
    du_8 = dDEdt # tau_E(t)
    du_9 = dDLdt # tau_L(t)
    du_10 = dDPdt # tau_P(t)
    du_11 = dDELdt # tau_E(t - tau_L(t))
    du_12 = dDLPdt # tau_L(t - tau_P(t))
    du_13 = dDELPdt # tau_E(t - tau_P(t) - tau_L(t - tau_P(t)))
    
    du[1] = du_1
    du[2] = du_2
    du[3] = du_3
    du[4] = du_4
    du[5] = du_5
    du[6] = du_6
    du[7] = du_7
    du[8] = du_8
    du[9] = du_9
    du[10] = du_10
    du[11] = du_11
    du[12] = du_12
    du[13] = du_13
end

# Dependent lag functions
deplag_1(u, p, t) = u[8]  # t - tau_E(t)
deplag_2(u, p, t) = u[9] + u[11]  # t - tau_L(t) - tau_E(t - tau_L(t))
deplag_3(u, p, t) = u[10] + u[12] + u[13]  # t - tau_P(t) - tau_L(t - tau_P(t)) - tau_E(t - tau_P(t) - tau_L(t - tau_P(t)))
deplag_4(u, p, t) = u[9]  # t - tau_L(t)
deplag_5(u, p, t) = u[10] + u[12]  # t - tau_P(t) - tau_L(t - tau_P(t))
deplag_6(u, p, t) = u[10]  # t - tau_P(t)

# Simulation setup
A0 = 12000.0
t0 = 0.0

temp0 = temperature(t0, parameters)
u0 = calculate_IC(A0, temp0, parameters)

t0 = 0.0
times = (t0, t0 + 365.0 * 5)  # 2 year simulation for benchmarking

prob = DDEProblem{true}(ewing_dde, u0, h, times, parameters; 
                        dependent_lags = (deplag_1, deplag_2, deplag_3, deplag_4, deplag_5, deplag_6))

# Reference solution
sol = solve(prob, MethodOfSteps(Vern9()); 
            reltol=1e-12, abstol=1e-12, maxiters=Int(1e7))
test_sol = TestSolution(sol)

# Plot the reference solution - life stages
plot(sol, vars = [1,2,3,4], title="Life Stages", 
     legend=:topleft, labels=["E" "L" "P" "A"])


abstols = 1.0 ./ 10.0 .^ (8:10)
reltols = 1.0 ./ 10.0 .^ (5:7)

setups = [Dict(:alg=>MethodOfSteps(BS3())),
          Dict(:alg=>MethodOfSteps(RK4())),
          Dict(:alg=>MethodOfSteps(Tsit5())),
          Dict(:alg=>MethodOfSteps(DP5())),
          Dict(:alg=>MethodOfSteps(OwrenZen4())),
          Dict(:alg=>MethodOfSteps(OwrenZen5())),
          Dict(:alg=>MethodOfSteps(Vern6())),
          Dict(:alg=>MethodOfSteps(Vern7()))]

wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      appxsol=test_sol, maxiters=Int(1e5), error_estimate=:final)
plot(wp)


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      appxsol=test_sol, maxiters=Int(1e5), error_estimate=:L2)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (10:13)
reltols = 1.0 ./ 10.0 .^ (7:10)

setups = [Dict(:alg=>MethodOfSteps(DP5())),
          Dict(:alg=>MethodOfSteps(OwrenZen5())),
          Dict(:alg=>MethodOfSteps(Vern7())),
          Dict(:alg=>MethodOfSteps(Vern8())),
          Dict(:alg=>MethodOfSteps(Vern9()))]

wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      appxsol=test_sol, maxiters=Int(1e6), error_estimate=:final)
plot(wp)


wp = WorkPrecisionSet(prob, abstols, reltols, setups;
                      appxsol=test_sol, maxiters=Int(1e6), error_estimate=:L2)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

