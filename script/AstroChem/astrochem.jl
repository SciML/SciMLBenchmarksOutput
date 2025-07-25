
using Catalyst
using OrdinaryDiffEq
using Plots
using Symbolics
using DiffEqDevTools
using Sundials, ODEInterface, ODEInterfaceDiffEq, LSODA
using RecursiveFactorization


# Some basic astrochemistry constants:
# u_vec = [H2 O C O⁺ OH⁺ H H2O⁺ H3O⁺ E H2O OH C⁺ CO CO⁺ H⁺ HCO⁺ T]
# println(u_vec)
# @species 
kboltzmann = 1.38064852e-16  # erg / K
pmass = 1.6726219e-24  # g
# dust2gas = 1e-2 # ratio
mu = 2.34
seconds_per_year = 3600 * 24 * 365
gamma_ad = 1.4
gnot = 1e1
# Simulation parameters:
number_density = 1e5
# dust2gas = 0.01
minimum_fractional_density = 1e-30 * number_density

# @register_symbolic get_heating(H, H2, E, tgas, ntot, dust2gas)
function get_heating(H, H2, E, tgas, ntot, dust2gas)
    """
       get_heating(x, tgas, cr_rate, gnot)

    Calculate the total heating rate based on various processes.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
        - `"E"`: Abundance of electrons
        - `"dust2gas"`: Dust-to-gas ratio
    - `tgas`: Float64 — Gas temperature
    - `cr_rate`: Float64 — Cosmic ray ionization rate
    - `gnot`: Float64 — Scaling factor for cosmic ray ionization rate

    ## Returns
    - Float64 — Total heating rate considering cosmic ray ionization and photoelectric heating processes.
    """

    rate_H2 = 5.68e-11 * gnot
    heats = [
        cosmic_ionisation_rate * (5.5e-12 * H + 2.5e-11 * H2),
        get_photoelectric_heating(H, E, tgas, gnot, ntot, dust2gas),
        6.4e-13 * rate_H2 * H2,
    ]

    return sum(heats)
end

# @register_symbolic get_photoelectric_heating(H, E, tgas, gnot, ntot, dust2gas)
function get_photoelectric_heating(H, E, tgas, gnot, ntot, dust2gas)
    """
       get_photoelectric_heating(x, tgas, gnot)

    Calculate the photoelectric heating rate due to dust grains.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
        - `"E"`: Abundance of electrons
    - `tgas`: Float64 — Gas temperature
    - `gnot`: Float64 — Scaling factor for cosmic ray ionization rate

    ## Returns
    - Float64 — Photoelectric heating rate based on dust recombination and ionization processes.
    """
    # ntot = sum(x)
    bet = 0.735 * tgas^(-0.068)
    psi = (E>0) * gnot * sqrt(tgas) / E

    # grains recombination cooling
    recomb_cool = 4.65e-30 * tgas^0.94 * psi^bet * E * H

    eps = 4.9e-2 / (1 + 4e-3 * psi^0.73) + 3.7e-2 * (tgas * 1e-4)^0.7 / (1 + 2e-4 * psi)

    # net photoelectric heating
    return (1.3e-24 * eps * gnot * ntot - recomb_cool) * dust2gas
end

# @register_symbolic get_cooling(H, H2, O, E, tgas)
function get_cooling(H, H2, O, E, tgas)
    """
       get_cooling(x, tgas)

    Calculate the total cooling rate based on various processes.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"E"`: Abundance of electrons
        - `"O"`: Abundance of oxygen
        - `"H2"`: Abundance of molecular hydrogen
    - `tgas`: Float64 — Gas temperature

    ## Returns
    - Float64 — Total cooling rate considering Lyman-alpha, OI 630nm, and H2 cooling processes.
    """

    cool = 7.3e-19 * H * E * exp(-118400.0 / tgas)  # Ly-alpha
    cool += 1.8e-24 * O * E * exp(-22800 / tgas)  # OI 630nm
    cool += cooling_H2(H, H2, tgas) # H2 cooling by dissacoiation and recombination
    return cool
end

@register_symbolic cooling_H2(H, H2, temp)
function cooling_H2(H, H2, temp)
    """
       cooling_H2(x, temp)

    Calculate the cooling rate for molecular hydrogen (H2) at a given temperature.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
    - `temp`: Float64 — Gas temperature

    ## Returns
    - Float64 — Cooling rate due to molecular hydrogen (H2) dissociation and recombination processes.
    """
    t3 = temp * 1e-3  # (T/1000)
    logt3 = log10(t3)

    logt32 = logt3 * logt3
    logt33 = logt32 * logt3
    logt34 = logt33 * logt3
    logt35 = logt34 * logt3
    logt36 = logt35 * logt3
    logt37 = logt36 * logt3
    logt38 = logt37 * logt3

    if temp < 2e3
        HDLR = (9.5e-22 * t3^3.76) / (1.0 + 0.12 * t3^2.1) * exp(-((0.13 / t3)^3)) + 3.0e-24 * exp(-0.51 / t3)
        HDLV = 6.7e-19 * exp(-5.86 / t3) + 1.6e-18 * exp(-11.7 / t3)
        HDL = HDLR + HDLV
    elseif 2e3 <= temp <= 1e4
        HDL = 1e1^(
            -2.0584225e1
            +
            5.0194035 * logt3
            -
            1.5738805 * logt32
            -
            4.7155769 * logt33
            + 2.4714161 * logt34
            + 5.4710750 * logt35
            -
            3.9467356 * logt36
            -
            2.2148338 * logt37
            +
            1.8161874 * logt38
        )
    else
        HDL = 5.531333679406485e-19
    end

    if temp <= 1e2
        f = 1e1^(
            -16.818342e0
            + 3.7383713e1 * logt3
            + 5.8145166e1 * logt32
            + 4.8656103e1 * logt33
            + 2.0159831e1 * logt34
            + 3.8479610e0 * logt35
        )
    elseif 1e2 < temp <= 1e3
        f = 1e1^(
            -2.4311209e1
            +
            3.5692468e0 * logt3
            -
            1.1332860e1 * logt32
            -
            2.7850082e1 * logt33
            -
            2.1328264e1 * logt34
            -
            4.2519023e0 * logt35
        )
    elseif 1e3 < temp <= 6e3
        f = 1e1^(
            -2.4311209e1
            +
            4.6450521e0 * logt3
            -
            3.7209846e0 * logt32
            +
            5.9369081e0 * logt33
            -
            5.5108049e0 * logt34
            +
            1.5538288e0 * logt35
        )
    else
        f = 1.862314467912518e-22
    end

    LDL = f * H

    if LDL * HDL == 0.0
        return 0.0
    end

    cool = H2 / (1.0 / HDL + 1.0 / LDL)

    return cool
end

function get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas)
    ntot = get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return (gamma_ad - 1e0) * (get_heating(H, H2, E, T, ntot, dust2gas) - get_cooling(H, H2, O, E, T)) / kboltzmann / ntot
end

function get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return sum([H2 O C O⁺ OH⁺ H H2O⁺ H3O⁺ E H2O OH C⁺ CO CO⁺ H⁺ HCO⁺])
end

ka_reaction(Tgas, α=1.0, β=1.0, γ=0.0) = α*(Tgas/300)^β*exp(−γ / Tgas)


# CONTINUE HERE
# Try this: https://docs.sciml.ai/Catalyst/stable/catalyst_functionality/constraint_equations/#Coupling-ODE-constraints-via-directly-building-a-ReactionSystem


@variables t T(t) = 100.0 # Define the variables before the species!
@species H2(t) O(t) C(t) O⁺(t) OH⁺(t) H(t) H2O⁺(t) H3O⁺(t) E(t) H2O(t) OH(t) C⁺(t) CO(t) CO⁺(t) H⁺(t) HCO⁺(t)
@parameters cosmic_ionisation_rate radiation_field dust2gas

D = Differential(t)
reaction_equations = [
	(@reaction 1.6e-9, $O⁺ + $H2 --> $OH⁺ + $H),
	(@reaction 1e-9, $OH⁺ + $H2 --> $H2O⁺ + $H),
	(@reaction 6.1e-10, $H2O⁺ + $H2 --> $H3O⁺ + $H),
	(@reaction ka_reaction(T, 1.1e-7, -1/2), $H3O⁺ + $E --> $H2O + $H),
	(@reaction ka_reaction(T, 8.6e-8, -1/2), $H2O⁺ + $E --> $OH + $H),
	(@reaction ka_reaction(T, 3.9e-8, -1/2), $H2O⁺ + $E --> $O + $H2),
	(@reaction ka_reaction(T, 6.3e-9, -0.48), $OH⁺ + $E --> $O + $H),
	(@reaction ka_reaction(T, 3.4e-12, -0.63), $O⁺ + $E --> $O),
	(@reaction 2.8 * cosmic_ionisation_rate, $O --> $O⁺ + $E),
	(@reaction 2.62 * cosmic_ionisation_rate, $C --> $C⁺ + $E),
	(@reaction 5.0 * cosmic_ionisation_rate, $CO --> $C + $O),
	(@reaction ka_reaction(T, 4.4e-12, -0.61), $C⁺ + $E --> $C),
	(@reaction ka_reaction(T, 1.15e-10, -0.339), $C⁺ + $OH --> CO + $H),
	(@reaction 9.15e-10 * (0.62 + 0.4767 * 5.5 * sqrt(300 / T)), $C⁺ + $OH --> $CO⁺ + $H),
	(@reaction 4e-10, $CO⁺ + $H --> $CO + $H⁺),
	(@reaction 7.28e-10, $CO⁺ + $H2 --> $HCO⁺ + $H),
	(@reaction ka_reaction(T, 2.8e-7, -0.69), $HCO⁺ + $E --> $CO + $H),
	(@reaction ka_reaction(T, 3.5e-12, -0.7), $H⁺ + $E --> $H),
	(@reaction 2.121e-17 * dust2gas / 1e-2, $H + $H --> $H2),
    (@reaction 1e-1 * cosmic_ionisation_rate, $H2 --> $H + $H),
	(@reaction 3.39e-10 * radiation_field, $C --> $C⁺ + $E),
	(@reaction 2.43e-10 * radiation_field, $CO --> $C + $O),
	(@reaction 7.72e-10 * radiation_field, $H2O --> $OH + $H),
    # (D(T) ~ get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas)) 
]

@named system = ReactionSystem(reaction_equations, t)

u0 = [:H2 => number_density, :O => number_density*2e-4, :C => number_density*1e-4, :O⁺=>minimum_fractional_density, :OH⁺=>minimum_fractional_density, :H=> minimum_fractional_density, :H2O⁺=> minimum_fractional_density, :H3O⁺=>minimum_fractional_density, :E=>minimum_fractional_density, :H2O=>minimum_fractional_density, :OH=>minimum_fractional_density, :C⁺=>minimum_fractional_density, :CO=>minimum_fractional_density, :CO⁺=>minimum_fractional_density, :H⁺=>minimum_fractional_density, :HCO⁺=> minimum_fractional_density, :T=> 100.0]

odesys = convert(ODESystem, complete(system))

setdefaults!(system, u0)

tspan = (0.0, 1e6*seconds_per_year)

params = [dust2gas => 0.01, radiation_field => 1e-1, cosmic_ionisation_rate => 1e-17]

println("Lets try to solve the ODE:")

sys = convert(ODESystem, complete(system))
# oprob = ODEProblemExpr(sys, [], tspan, params)

ssys = structural_simplify(sys)


oprob = ODEProblem(ssys, [], tspan, params)
println("Created the ODEproblem.")
sol = solve(oprob, Rodas5()) # Rodas5()) # Tsit5()

# Generate a solution using high precision arithmetic
bigprob = remake(oprob, u0 = big.(oprob.u0), tspan = big.(oprob.tspan))
refsol = solve(bigprob, Rodas5P(), abstol=1e-18, reltol=1e-18)


abstols = 1.0 ./ 10.0 .^ (7:13)
reltols = 1.0 ./ 10.0 .^ (4:10)

setups = [
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>Rodas4P()),
          Dict(:alg=>CVODE_BDF()),
          #Dict(:alg=>ddebdf()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P()),
          #Dict(:alg=>rodas()),
          #Dict(:alg=>radau()),
          Dict(:alg=>lsoda()),
          #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEqCore.PolyesterThreads())),
          Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = false)),
          #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEqCore.PolyesterThreads())),
          Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false)),
          ]
wp = WorkPrecisionSet(oprob,abstols,reltols,setups;verbose=false,
                      save_everystep=false,appxsol=refsol,maxiters=Int(1e5),numruns=10)
plot(wp)


reaction_equations = [
	(@reaction 1.6e-9, $O⁺ + $H2 --> $OH⁺ + $H),
	(@reaction 1e-9, $OH⁺ + $H2 --> $H2O⁺ + $H),
	(@reaction 6.1e-10, $H2O⁺ + $H2 --> $H3O⁺ + $H),
	(@reaction ka_reaction(T, 1.1e-7, -1/2), $H3O⁺ + $E --> $H2O + $H),
	(@reaction ka_reaction(T, 8.6e-8, -1/2), $H2O⁺ + $E --> $OH + $H),
	(@reaction ka_reaction(T, 3.9e-8, -1/2), $H2O⁺ + $E --> $O + $H2),
	(@reaction ka_reaction(T, 6.3e-9, -0.48), $OH⁺ + $E --> $O + $H),
	(@reaction ka_reaction(T, 3.4e-12, -0.63), $O⁺ + $E --> $O),
	(@reaction 2.8 * cosmic_ionisation_rate, $O --> $O⁺ + $E),
	(@reaction 2.62 * cosmic_ionisation_rate, $C --> $C⁺ + $E),
	(@reaction 5.0 * cosmic_ionisation_rate, $CO --> $C + $O),
	(@reaction ka_reaction(T, 4.4e-12, -0.61), $C⁺ + $E --> $C),
	(@reaction ka_reaction(T, 1.15e-10, -0.339), $C⁺ + $OH --> CO + $H),
	(@reaction 9.15e-10 * (0.62 + 0.4767 * 5.5 * sqrt(300 / T)), $C⁺ + $OH --> $CO⁺ + $H),
	(@reaction 4e-10, $CO⁺ + $H --> $CO + $H⁺),
	(@reaction 7.28e-10, $CO⁺ + $H2 --> $HCO⁺ + $H),
	(@reaction ka_reaction(T, 2.8e-7, -0.69), $HCO⁺ + $E --> $CO + $H),
	(@reaction ka_reaction(T, 3.5e-12, -0.7), $H⁺ + $E --> $H),
	(@reaction 2.121e-17 * dust2gas / 1e-2, $H + $H --> $H2),
    (@reaction 1e-1 * cosmic_ionisation_rate, $H2 --> $H + $H),
	(@reaction 3.39e-10 * radiation_field, $C --> $C⁺ + $E),
	(@reaction 2.43e-10 * radiation_field, $CO --> $C + $O),
	(@reaction 7.72e-10 * radiation_field, $H2O --> $OH + $H),
    (D(T) ~ get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas)) 
]

@named system = ReactionSystem(reaction_equations, t)

u0 = [:H2 => number_density, :O => number_density*2e-4, :C => number_density*1e-4, :O⁺=>minimum_fractional_density, :OH⁺=>minimum_fractional_density, :H=> minimum_fractional_density, :H2O⁺=> minimum_fractional_density, :H3O⁺=>minimum_fractional_density, :E=>minimum_fractional_density, :H2O=>minimum_fractional_density, :OH=>minimum_fractional_density, :C⁺=>minimum_fractional_density, :CO=>minimum_fractional_density, :CO⁺=>minimum_fractional_density, :H⁺=>minimum_fractional_density, :HCO⁺=> minimum_fractional_density, :T=> 100.0]

odesys = convert(ODESystem, complete(system))

setdefaults!(system, u0)

tspan = (0.0, 1e6*seconds_per_year)

params = [dust2gas => 0.01, radiation_field => 1e-1, cosmic_ionisation_rate => 1e-17]

println("Lets try to solve the ODE:")

sys = convert(ODESystem, complete(system))
# oprob = ODEProblemExpr(sys, [], tspan, params)

ssys = structural_simplify(sys)

oprob = ODEProblem(ssys, [], tspan, params)
println("Created the ODEproblem.")
refsol = solve(oprob, Rodas5P(), abstol=1e-14, reltol=1e-14)


refsol = solve(oprob, Rodas5P(), abstol=1e-13, reltol=1e-13)

# Run Benchmark

abstols = 1.0 ./ 10.0 .^ (9:10)
reltols = 1.0 ./ 10.0 .^ (9:10)

setups = [
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>CVODE_BDF()),
          #Dict(:alg=>ddebdf()),
          Dict(:alg=>Rodas5P()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>KenCarp47()),
          #Dict(:alg=>RadauIIA9()),
          #Dict(:alg=>rodas()),
          #Dict(:alg=>radau()),
          Dict(:alg=>lsoda()),
          #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEqCore.PolyesterThreads())),
          #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = false)),
          #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEqCore.PolyesterThreads())),
          #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false)),
          ]
wp = WorkPrecisionSet(oprob,abstols,reltols,setups;verbose=false,
                      save_everystep=false,appxsol=refsol,maxiters=Int(1e5),numruns=10,
                      print_names = true)
plot(wp)

