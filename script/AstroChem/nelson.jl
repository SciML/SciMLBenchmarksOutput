
using OrdinaryDiffEq
using DiffEqDevTools, Plots
using Sundials, LSODA
using ODEInterface, ODEInterfaceDiffEq
using RecursiveFactorization


function Nelson!(du,u,p,t)
    T, Av, Go, n_H, shield = p

    # 1: H2
    du[1] = -1.2e-17 * u[1] + 
            n_H * (1.9e-6 * u[2] * u[3]) / (T^0.54) - 
            n_H * 4e-16 * u[1] * u[12] - 
            n_H * 7e-15 * u[1] * u[5] + 
            n_H * 1.7e-9 * u[10] * u[2] + 
            n_H * 2e-9 * u[2] * u[6] + 
            n_H * 2e-9 * u[2] * u[14] + 
            n_H * 8e-10 * u[2] * u[8] 

    # 2: H3+
    du[2] = 1.2e-17 * u[1] + 
            n_H * (-1.9e-6 * u[3] * u[2]) / (T^0.54) - 
            n_H * 1.7e-9 * u[10] * u[2] - 
            n_H * 2e-9 * u[2] * u[6] - 
            n_H * 2e-9 * u[2] * u[14] - 
            n_H * 8e-10 * u[2] * u[8]

    # 3: e
    du[3] = n_H * (-1.4e-10 * u[3] * u[12]) / (T^0.61) - 
            n_H * (3.8e-10 * u[13] * u[3]) / (T^0.65) - 
            n_H * (3.3e-5 * u[11] * u[3]) / T + 
            1.2e-17 * u[1] - 
            n_H * (1.9e-6 * u[3] * u[2]) / (T^0.54) + 
            6.8e-18 * u[4] - 
            n_H * (9e-11 * u[3] * u[5]) / (T^0.64) + 
            3e-10 * Go * exp(-3 * Av) * u[6] +
            n_H * 2e-9 * u[2] * u[13]
            + 2.0e-10 * Go * exp(-1.9 * Av) * u[14]

    # 4: He
    du[4] = n_H * (9e-11 * u[3] * u[5]) / (T^0.64) - 
            6.8e-18 * u[4] + 
            n_H * 7e-15 * u[1] * u[5] + 
            n_H * 1.6e-9 * u[10] * u[5]

    # 5: He+
    du[5] = 6.8e-18 * u[4] - 
            n_H * (9e-11 * u[3] * u[5]) / (T^0.64) - 
            n_H * 7e-15 * u[1] * u[5] - 
            n_H * 1.6e-9 * u[10] * u[5]

    # 6: C
    du[6] = n_H * (1.4e-10 * u[3] * u[12]) / (T^0.61) - 
            n_H * 2e-9 * u[2] * u[6] - 
            n_H * 5.8e-12 * (T^0.5) * u[9] * u[6] + 
            1e-9 * Go * exp(-1.5 * Av) * u[7] - 
            3e-10 * Go * exp(-3 * Av) * u[6] + 
            1e-10 * Go * exp(-3 * Av) * u[10] * shield

    # 7: CHx
    du[7] = n_H * (-2e-10) * u[7] * u[8] + 
            n_H * 4e-16 * u[1] * u[12] + 
            n_H * 2e-9 * u[2] * u[6] - 
            1e-9 * Go * u[7] * exp(-1.5 * Av)

    # 8: O
    du[8] = n_H * (-2e-10) * u[7] * u[8] + 
            n_H * 1.6e-9 * u[10] * u[5] - 
            n_H * 8e-10 * u[2] * u[8] + 
            5e-10 * Go * exp(-1.7 * Av) * u[9] + 
            1e-10 * Go * exp(-3 * Av) * u[10] * shield

    # 9: OHx
    du[9] = n_H * (-1e-9) * u[9] * u[12] + 
            n_H * 8e-10 * u[2] * u[8] - 
            n_H * 5.8e-12 * (T^0.5) * u[9] * u[6] - 
            5e-10 * Go * exp(-1.7 * Av) * u[9]

    # 10: CO
    du[10] = n_H * (3.3e-5 * u[11] * u[3]) / T + 
             n_H * 2e-10 * u[7] * u[8] - 
             n_H * 1.7e-9 * u[10] * u[2] - 
             n_H * 1.6e-9 * u[10] * u[5] + 
             n_H * 5.8e-12 * (T^0.5) * u[9] * u[6] - 
             1e-10 * Go * exp(-3 * Av) * u[10] + 
             1.5e-10 * Go * exp(-2.5 * Av) * u[11] * shield

    # 11: HCO+
    du[11] = n_H * (-3.3e-5 * u[11] * u[3]) / T + 
             n_H * 1e-9 * u[9] * u[12] + 
             n_H * 1.7e-9 * u[10] * u[2] - 
             1.5e-10 * Go * exp(-2.5 * Av) * u[11]

    # 12: C+
    du[12] = n_H * (-1.4e-10 * u[3] * u[12]) / (T^0.61) - 
             n_H * 4e-16 * u[1] * u[12] - 
             n_H * 1e-9 * u[9] * u[12] + 
             n_H * 1.6e-9 * u[10] * u[5] + 
             3e-10 * Go * exp(-3 * Av) * u[6]

    # 13: M+
    du[13] = n_H * (-3.8e-10 * u[13] * u[3]) / (T^0.65) + 
             n_H * 2e-9 * u[2] * u[14] +
             2.0e-10 * Go * exp(-1.9 * Av) * u[14]

    # 14: M
    du[14] = n_H * (3.8e-10 * u[13] * u[3]) / (T^0.65) - 
             n_H * 2e-9 * u[2] * u[14] -
             2.0e-10 * Go * exp(-1.9 * Av) * u[14]

end

# Set the Timespan, Parameters, and Initial Conditions
seconds_per_year = 3600 * 24 * 365
tspan = (0.0, 30000 * seconds_per_year) # ~30 thousand yrs

params = (10,  # T
          2,   # Av
          1.7, # Go
          611, # n_H
          1)   # shield

u0 = [0.5,      # 1:  H2
      9.059e-9, # 2:  H3+
      2.0e-4,   # 3:  e
      0.1,      # 4:  He
      7.866e-7, # 5:  He+
      0.0,      # 6:  C
      0.0,      # 7:  CHx
      0.0004,   # 8:  O
      0.0,      # 9:  OHx
      0.0,      # 10: CO
      0.0,      # 11: HCO+
      0.0002,   # 12: C+
      2.0e-7,   # 13: M+
      2.0e-7]   # 14: M

prob = ODEProblem(Nelson!, u0, tspan, params)
refsol = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14)
sol1 = solve(prob, Rodas5P())
sol2 = solve(prob, FBDF())
sol3 = solve(prob, lsoda())
sol4 = solve(prob, lsoda(), saveat = 1e10)


using Plots
colors = palette(:acton, 5)
p1 = plot(sol1, vars = (0,11), lc=colors[1], legend = false, titlefontsize = 12, lw = 3, title = "Rodas5")
p2 = plot(sol2, vars = (0,11), lc=colors[2], legend = false, titlefontsize = 12, lw = 3, title = "FBDF")
p3 = plot(sol3, vars = (0,11), lc=colors[3], legend = false, titlefontsize = 12, lw = 3, title = "lsoda")
p4 = plot(sol4, vars = (0,11), lc=colors[4], legend = false, titlefontsize = 12, lw = 3, title = "lsoda with saveat")

combined_plot = plot(p1, p2, p3, p4, layout=(4, 1), dpi = 600, pallete=:acton)


abstols = 1.0 ./ 10.0 .^ (8:10)
reltols = 1.0 ./ 10.0 .^ (8:10)

setups = [
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
          #Dict(:alg=>Rodas4P()),
          Dict(:alg=>CVODE_BDF()),
          #Dict(:alg=>ddebdf()),
          #Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>KenCarp47()),
          Dict(:alg=>RadauIIA9()),
		  Dict(:alg=>lsoda()),
          #Dict(:alg=>rodas()),
          #Dict(:alg=>radau()),
          #Dict(:alg=>lsoda()),
          #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEqCore.PolyesterThreads())),
          #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = false)),
          #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEqCore.PolyesterThreads())),
          #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false)),
          ]

wp = WorkPrecisionSet(prob,abstols,reltols,setups;appxsol=refsol,save_everystep=false, print_names = true)
plot(wp)

