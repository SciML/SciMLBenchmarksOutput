
using OrdinaryDiffEq, DiffEqDevTools, Sundials, ModelingToolkit, ODEInterfaceDiffEq,
      Plots, DASSL, DASKR, StaticArrays
using LinearAlgebra, SparseArrays
using ModelingToolkit: t_nounits as t, D_nounits as D
const SA = StaticArrays.SA

# Common tolerances for reference solutions
abstol_ref = 1e-12
reltol_ref = 1e-12


# RLC Circuit Parameters
L_rlc, C_rlc, R_rlc = 1e-3, 1e-6, 1e3

# System matrices from DAEV repository
E_rlc = [L_rlc  0    0   0
         0      0    1   0  
         0      0    0   0
         0      0    0   0]

A_rlc = [0      1    0   0
         1/C_rlc 0   0   0
         -R_rlc  0   0   1
         0       1   1   1]

B_rlc = [0; 0; 0; -1]
C_rlc = [1 0 0 0; 0 0 1 0]

# ModelingToolkit formulation using E*Dx = A*x + B*u
@variables i_L(t)=0.0 v_C(t)=0.0 i_R(t)=0.0 i_C(t)=0.0

# State vector x and its derivative Dx
x = [i_L, v_C, i_R, i_C]
Dx = D.(x)

# Input function: constant voltage source
u(t) = 1.0

# E*Dx = A*x + B*u
rlc_eqs = E_rlc * Dx ~ A_rlc * x + B_rlc .* u(t)

@mtkbuild rlc_sys = ODESystem(rlc_eqs, t)

# Problems using constant voltage input
rlc_prob = ODEProblem(rlc_sys, [i_R => 0.0, v_C => 0.0], (0.0, 1e-3))
rlc_static_prob = ODEProblem{false}(rlc_sys, SA[i_R => 0.0, v_C => 0.0], (0.0, 1e-3))


# Two Masses Parameters  
J1_masses, J2_masses = 1.0, 1.0

# System matrices from DAEV repository
E_masses = [J1_masses  0   0  0
            0    J2_masses  0  0
            0          0   0  0
            0          0   0  0]

A_masses = [0   0   1   0
            0   0   0   1  
            0   0  -1  -1
           -1   1   0   0]

B_masses = [1 0; 0 1; 0 0; 0 0]
C_masses = [1 0 0 0; 0 0 1 0]

# ModelingToolkit formulation using E*Dx = A*x + B*u
@variables θ1(t)=0.0 θ2(t)=0.0 ω1(t)=0.0 ω2(t)=0.0

# State vector x and its derivative Dx
x = [θ1, θ2, ω1, ω2]
Dx = D.(x)

# Input functions: torque on first mass, sine wave on second
u1(t) = 1.0  # Constant torque on first mass
u2(t) = 0.5*sin(2π*t)  # Sinusoidal torque on second mass
u(t) = [u1(t), u2(t)]

# E*Dx = A*x + B*u
masses_eqs = E_masses * Dx ~ A_masses * x + B_masses * u(t)

@mtkbuild masses_sys = ODESystem(masses_eqs, t)

# Problems using torque inputs
masses_prob = ODEProblem(masses_sys, [], (0.0, 1.0))
masses_static_prob = ODEProblem{false}(masses_sys, SA[], (0.0, 1.0))


# RL Network Parameters
R_rl, L_rl = 1.0, 1.0

# System matrices from DAEV repository  
E_rl = [0  0  0
        0  0  0
        0  0  L_rl]

A_rl = [-R_rl   R_rl   0
         R_rl  -R_rl  -1
         0      1     0]

B_rl = [1; 0; 0]
C_rl = [1 0 0]

# ModelingToolkit formulation using E*Dx = A*x + B*u
@variables i1(t)=0.0 i2(t)=0.0 v_L(t)=0.0

# State vector x and its derivative Dx
x = [i1, i2, v_L]
Dx = D.(x)

# Input function: step current source
u(t) = 1.0

# E*Dx = A*x + B*u
rl_eqs = E_rl * Dx ~ A_rl * x + B_rl .* u(t)

@mtkbuild rl_sys = ODESystem(rl_eqs, t)

# Problems using current source input
rl_prob = ODEProblem(rl_sys, [v_L => 1.0], (0.0, 1.0))
rl_static_prob = ODEProblem{false}(rl_sys, SA[v_L => 1.0], (0.0, 1.0))


# Cart Pendulum Parameters
m1_cart, m2_cart, L_cart, g_cart = 1.0, 1.0, 1.0, 9.81

# System matrices (linearized around equilibrium)
E_cart = [1.0  0   0   0   0   0   0
          0   1.0  0   0   0   0   0  
          0    0  1.0  0   0   0   0
          0    0   0  m1_cart  0   0   0
          0    0   0   0  m2_cart  0   0
          0    0   0   0   0   0   0
          0    0   0   0   0   0   0]

A_cart = [0  0  0   1   0   0   0
          0  0  0   0   1   0   0
          0  0  0   0   0   1   0  
          0  0  0   0   0   0   1
          0  0  0   0   0   0   1
          0  0 -g_cart/L_cart  0   0   0   0
          1  0 -L_cart  0   0   0   0]

B_cart = [0; 0; 0; 1; 0; 0; 0]
C_cart = [1 0 0 0 0 0 0; 0 0 1 0 0 0 0]

# ModelingToolkit formulation using E*Dx = A*x + B*u  
@variables x_cart(t)=0.0 y_cart(t)=0.0 φ_cart(t)=0.1 
@variables dx_cart(t)=0.0 dy_cart(t)=0.0 dφ_cart(t)=0.0 λ_cart(t)=0.0

# State vector x and its derivative Dx
x = [x_cart, y_cart, φ_cart, dx_cart, dy_cart, dφ_cart, λ_cart]
Dx = D.(x)

# Input function: step force input to stabilize the cart
u(t) = 1.0 * exp(-t)  # Decaying force input

# E*Dx = A*x + B*u
cart_eqs = E_cart * Dx ~ A_cart * x + B_cart .* u(t)

@mtkbuild cart_sys = ODESystem(cart_eqs, t)

# Problems using force input
cart_prob = ODEProblem(cart_sys, [dy_cart => 0.0, y_cart => 0.0], (0.0, 1.0))
cart_static_prob = ODEProblem{false}(cart_sys, SA[dy_cart => 0.0, y_cart => 0.0], (0.0, 1.0))


# Electric Generator Parameters
J_gen, L_gen, R1_gen, R2_gen, k_gen = 1.0, 1.0, 1.0, 1.0, 1.0

# System matrices (simplified 4x4 version)
E_gen = [J_gen  0   0   0
         0      0   0   0
         0      0   0   0  
         0      0   0   0]

A_gen = [0   0   0   0
         0   0   0   1
         0   0   0  -R2_gen
         0  -k_gen  1   0]

B_gen = [1; 0; 0; 0]
C_gen = [1 0 0 0; 0 0 0 1]

# ModelingToolkit formulation using E*Dx = A*x + B*u
@variables ω_gen(t)=1.0 i_gen(t)=0.0 v1_gen(t)=0.0 v2_gen(t)=0.0

# State vector x and its derivative Dx
x = [ω_gen, i_gen, v1_gen, v2_gen]
Dx = D.(x)

# Input function: variable torque input
u(t) = 1.0 + 0.5*cos(2π*t)  # Oscillating torque

# E*Dx = A*x + B*u
gen_eqs = E_gen * Dx ~ A_gen * x + B_gen .* u(t)

@mtkbuild gen_sys = ODESystem(gen_eqs, t)

# Problems using torque input
gen_prob = ODEProblem(gen_sys, [ω_gen => 1.0], (0.0, 1.0))
gen_static_prob = ODEProblem{false}(gen_sys, SA[ω_gen => 1.0], (0.0, 1.0))


# Mass-Spring Parameters (3 masses)
m_spring, k_spring, d_spring = 100.0, 2.0, 5.0

# System matrices
E_spring = [m_spring    0         0     0  0  0  0
            0       m_spring     0     0  0  0  0
            0           0    m_spring  0  0  0  0
            0           0         0    0  0  0  0
            0           0         0    0  0  0  0
            0           0         0    0  0  0  0
            0           0         0    0  0  0  0]

A_spring = [0   0   0   1    0    0   0
            0   0   0   0    1    0   0
            0   0   0   0    0    1   0
           -k_spring  k_spring   0  -d_spring  d_spring   0   1
            k_spring -2*k_spring k_spring  d_spring -2*d_spring d_spring  0
            0   k_spring  -k_spring   0   d_spring  -d_spring  1
            1   0   -1   0    0    0   0]

B_spring = [0; 0; 0; 1; 0; 0; 0]
C_spring = [1 0 0 0 0 0 0; 0 0 1 0 0 0 0]

# Simplified 5x5 system matrices (2 masses + constraint)
E_spring_5 = [1.0  0   0   0   0
              0   1.0  0   0   0
              0    0  m_spring  0   0
              0    0   0  m_spring  0
              0    0   0   0   0]

A_spring_5 = [0   0   1   0   0
              0   0   0   1   0
             -k_spring  0  -d_spring  0   1
              k_spring  -k_spring  d_spring  -d_spring  -1
              1   -1   0   0   0]

B_spring_5 = [0; 0; 1; 0; 0]
C_spring_5 = [1 0 0 0 0; 0 1 0 0 0]

# ModelingToolkit formulation using E*Dx = A*x + B*u
@variables x1_spring(t)=0.0 x2_spring(t)=0.0 v1_spring(t)=0.0 v2_spring(t)=0.0 λ_spring(t)=0.0

# State vector x and its derivative Dx
x = [x1_spring, x2_spring, v1_spring, v2_spring, λ_spring]
Dx = D.(x)

# Input function: impulse force followed by decay
u(t) = ifelse((t < 0.1), 10.0, 0.1*exp(-5*t))  # Initial impulse then decay

# E*Dx = A*x + B*u
spring_eqs = E_spring_5 * Dx ~ A_spring_5 * x + B_spring_5 .* u(t)

@mtkbuild spring_sys = ODESystem(spring_eqs, t)

# Problems using force input
spring_prob = ODEProblem(spring_sys, [λ_spring => 0.0, v1_spring => 1.0], (0.0, 20.0))
spring_static_prob = ODEProblem{false}(spring_sys, SA[λ_spring => 0.0, v1_spring => 1.0], (0.0, 20.0))


# Generate reference solutions for all systems using robust methods
rlc_ref = solve(rlc_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
rlc_static_ref = solve(rlc_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

masses_ref = solve(masses_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
masses_static_ref = solve(masses_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

rl_ref = solve(rl_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
rl_static_ref = solve(rl_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

cart_ref = solve(cart_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
cart_static_ref = solve(cart_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

gen_ref = solve(gen_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
gen_static_ref = solve(gen_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

spring_ref = solve(spring_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)
spring_static_ref = solve(spring_static_prob, Rodas5P(), abstol=abstol_ref, reltol=reltol_ref)

# Problem and reference solution arrays
all_probs = [
    # Index-1: RLC Circuit
    [rlc_prob, rlc_static_prob],
    # Index-2: Two Masses and RL Network
    [masses_prob, masses_static_prob],
    [rl_prob, rl_static_prob],
    # Index-3: Cart, Generator, Spring
    [cart_prob, cart_static_prob],
    [gen_prob, gen_static_prob],
    [spring_prob, spring_static_prob]
]

all_refs = [
    [rlc_ref, rlc_static_ref],
    [masses_ref, masses_static_ref],
    [rl_ref, rl_static_ref],
    [cart_ref, cart_static_ref],
    [gen_ref, gen_static_ref],
    [spring_ref, spring_static_ref]
]

system_names = ["RLC Circuit (Index-1)", "Two Masses (Index-2)", "RL Network (Index-2)",
                "Cart Pendulum (Index-3)", "Electric Generator (Index-3)", "Mass-Spring (Index-3)"]


# Plot solutions for each system
p1 = plot(rlc_ref, title="RLC Circuit", legend=:topright)
p2 = plot(masses_ref, title="Two Masses", legend=:topright)
p3 = plot(rl_ref, title="RL Network", legend=:topright)
plot(p1, p2, p3, layout=(1,3), size=(1200,400))


p4 = plot(cart_ref, title="Cart Pendulum", legend=:topright)
p5 = plot(gen_ref, title="Electric Generator", legend=:topright)
p6 = plot(spring_ref, title="Mass-Spring", legend=:topright)
plot(p4, p5, p6, layout=(1,3), size=(1200,400))


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4)

# RLC Circuit Work-Precision
setups_rlc = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_rlc = WorkPrecisionSet(all_probs[1], abstols, reltols, setups_rlc;
                         save_everystep=false, appxsol=all_refs[1], maxiters=Int(1e5), numruns=10)
plot(wp_rlc, title="RLC Circuit (Index-1) Work-Precision")


setups_masses = [
   #Dict(:prob_choice => 2, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_masses = WorkPrecisionSet(all_probs[2], abstols, reltols, setups_masses;
                            save_everystep=false, appxsol=all_refs[2], maxiters=Int(1e5), numruns=10)
plot(wp_masses, title="Two Masses (Index-2) Work-Precision")


#=
setups_rl = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_rl = WorkPrecisionSet(all_probs[3], abstols, reltols, setups_rl;
                        save_everystep=false, appxsol=all_refs[3], maxiters=Int(1e5), numruns=10)
plot(wp_rl, title="RL Network (Index-2) Work-Precision")
=#


setups_cart = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_cart = WorkPrecisionSet(all_probs[4], abstols, reltols, setups_cart;
                          save_everystep=false, appxsol=all_refs[4], maxiters=Int(1e5), numruns=10)
plot(wp_cart, title="Cart Pendulum (Index-3) Work-Precision")


setups_gen = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_gen = WorkPrecisionSet(all_probs[5], abstols, reltols, setups_gen;
                         save_everystep=false, appxsol=all_refs[5], maxiters=Int(1e5), numruns=10)
plot(wp_gen, title="Electric Generator (Index-3) Work-Precision")


setups_spring = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas4()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

wp_spring = WorkPrecisionSet(all_probs[6], abstols, reltols, setups_spring;
                            save_everystep=false, appxsol=all_refs[6], maxiters=Int(1e5), numruns=10)
plot(wp_spring, title="Mass-Spring (Index-3) Work-Precision")


abstols_low = 1.0 ./ 10.0 .^ (7:12)
reltols_low = 1.0 ./ 10.0 .^ (4:9)

all_setups = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas4()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

# Generate work-precision plots for all systems at low tolerances
for (i, (probs, refs, name)) in enumerate(zip(all_probs, all_refs, system_names))
    wp = WorkPrecisionSet(probs, abstols_low, reltols_low, all_setups;
                         save_everystep=false, appxsol=refs, maxiters=Int(1e5), numruns=10)
    p = plot(wp, title="$name - Low Tolerances")
    display(p)
end


abstols_high = 1.0 ./ 10.0 .^ (3:6)
reltols_high = 1.0 ./ 10.0 .^ (1:4)

# High tolerance setups - focus on speed
high_setups = [
    Dict(:prob_choice => 1, :alg=>Rosenbrock23()),
    Dict(:prob_choice => 1, :alg=>Rodas5P()),
    #Dict(:prob_choice => 1, :alg=>CVODE_BDF()),
    Dict(:prob_choice => 1, :alg=>FBDF()),
    Dict(:prob_choice => 1, :alg=>QNDF()),
    Dict(:prob_choice => 2, :alg=>Rodas5P()),
]

for (i, (probs, refs, name)) in enumerate(zip(all_probs, all_refs, system_names))
    wp = WorkPrecisionSet(probs, abstols_high, reltols_high, high_setups;
                         save_everystep=false, appxsol=refs, maxiters=Int(1e5), numruns=10)
    p = plot(wp, title="$name - High Tolerances")
    display(p)
end


# Create summary comparison of all DAE types
plot_array = []
for (i, (probs, refs, name)) in enumerate(zip(all_probs, all_refs, system_names))
    wp = WorkPrecisionSet(probs, abstols, reltols, all_setups;
                         save_everystep=false, appxsol=refs, maxiters=Int(1e5), numruns=10)
    p = plot(wp, title=name, legend=false, titlefont=font(10))
    push!(plot_array, p)
end

plot(plot_array..., layout=(2,3), size=(1500,800))


# Analyze L2 timeseries errors
abstols_ts = 1.0 ./ 10.0 .^ (5:8)
reltols_ts = 1.0 ./ 10.0 .^ (2:5)

for (i, (probs, refs, name)) in enumerate(zip(all_probs, all_refs, system_names))
    wp = WorkPrecisionSet(probs, abstols_ts, reltols_ts, all_setups;
                         error_estimate=:l2, save_everystep=false, appxsol=refs, 
                         maxiters=Int(1e5), numruns=10)
    p = plot(wp, title="$name - L2 Timeseries Error")
    display(p)
end


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

