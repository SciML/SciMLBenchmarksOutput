---
author: "Gijs Vermariën and Chris Rackauckas"
title: "AstroChem Work-Precision Diagrams"
---
```julia
using Catalyst
using SciMLLogging
using OrdinaryDiffEq
using OrdinaryDiffEqBDF, OrdinaryDiffEqExtrapolation, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSDIRK
using Plots
using Symbolics
using DiffEqDevTools
using Sundials, ODEInterface, ODEInterfaceDiffEq, LSODA
using RecursiveFactorization
```




## Without Temperature Dynamics

```julia
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
        6.4e-13 * rate_H2 * H2
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
        HDLR = (9.5e-22 * t3^3.76) / (1.0 + 0.12 * t3^2.1) * exp(-((0.13 / t3)^3)) +
               3.0e-24 * exp(-0.51 / t3)
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

function get_heating_cooling(
        T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas)
    ntot = get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return (gamma_ad - 1e0) *
           (get_heating(H, H2, E, T, ntot, dust2gas) - get_cooling(H, H2, O, E, T)) /
           kboltzmann / ntot
end

function get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return sum([H2 O C O⁺ OH⁺ H H2O⁺ H3O⁺ E H2O OH C⁺ CO CO⁺ H⁺ HCO⁺])
end

ka_reaction(Tgas, α = 1.0, β = 1.0, γ = 0.0) = α*(Tgas/300)^β*exp(−γ / Tgas)

# CONTINUE HERE
# Try this: https://docs.sciml.ai/Catalyst/stable/catalyst_functionality/constraint_equations/#Coupling-ODE-constraints-via-directly-building-a-ReactionSystem

@independent_variables t
@variables T(t) = 100.0 # Define the variables before the species!
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
    (@reaction 7.72e-10 * radiation_field, $H2O --> $OH + $H)    # (D(T) ~ get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas)) 
]

@named system = ReactionSystem(reaction_equations, t)

u0 = [:H2 => number_density, :O => number_density*2e-4, :C => number_density*1e-4,
    :O⁺=>minimum_fractional_density, :OH⁺=>minimum_fractional_density,
    :H => minimum_fractional_density, :H2O⁺ => minimum_fractional_density,
    :H3O⁺=>minimum_fractional_density, :E=>minimum_fractional_density,
    :H2O=>minimum_fractional_density, :OH=>minimum_fractional_density,
    :C⁺=>minimum_fractional_density, :CO=>minimum_fractional_density,
    :CO⁺=>minimum_fractional_density, :H⁺=>minimum_fractional_density,
    :HCO⁺ => minimum_fractional_density, :T => 100.0]

tspan = (0.0, 1e6*seconds_per_year)

params = [dust2gas => 0.01, radiation_field => 1e-1, cosmic_ionisation_rate => 1e-17]

println("Attempting to solve the ODE...")

sys = ode_model(complete(system))
# oprob = ODEProblemExpr(sys, [], tspan, params)

ssys = structural_simplify(sys)
```

```
Attempting to solve the ODE...
Model system:
Equations (16):
  16 standard: see equations(system)
Unknowns (16): see unknowns(system)
  O⁺(t)
  H2(t)
  OH⁺(t)
  H(t)
  ⋮
Parameters (4): see parameters(system)
  T
  cosmic_ionisation_rate
  dust2gas
  radiation_field
```



```julia
oprob = ODEProblem(ssys, merge(Dict{Any, Any}(u0), Dict{Any, Any}(params)), tspan)
println("ODEProblem created successfully.")
sol = solve(oprob, Rodas5()) # Rodas5()) # Tsit5()

# Generate a tight-tolerance reference solution
refsol = solve(oprob, Rodas5P(), abstol = 1e-14, reltol = 1e-14)
```

```
ODEProblem created successfully.
retcode: Success
Interpolation: specialized 4th (Rodas6P = 5th) order "free" stiffness-aware
 interpolation
t: 978-element Vector{Float64}:
     0.0
     0.06082607907396814
     1.229171955447668
    12.912630719184667
   122.6468167297148
   898.9157153896226
  3214.104466100127
  6507.217634603039
 10641.300715959518
 16036.900868926028
     ⋮
     2.110409174495309e13
     2.2494837364569926e13
     2.3913851109084848e13
     2.5365421444391902e13
     2.6837601620852344e13
     2.833774045110104e13
     2.986430466869871e13
     3.141627187867463e13
     3.1536e13
u: 978-element Vector{Vector{Float64}}:
 [1.0e-25, 100000.0, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 
1.0e-25, 20.0, 10.0, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25]
 [3.4062438630386e-17, 100000.0, 1.658506995116355e-22, 1.2165215980645034e
-14, 1.0033630285797331e-25, 1.0000037135125647e-25, 2.0620090805091148e-11
, 9.999999999953043e-26, 1.0000000000046958e-25, 20.0, 9.99999999997938, 2.
062005674248678e-11, 9.99999999998522e-26, 9.999955718712477e-26, 1.0e-25, 
1.0000044281287524e-25]
 [6.882686128941345e-16, 100000.0, 6.767958348646247e-20, 2.458344587746652
e-13, 2.8730174327549624e-24, 1.0005947892198404e-25, 4.166903032674292e-10
, 9.999999999051079e-26, 1.0000000000948927e-25, 19.999999999999996, 9.9999
99999583311, 4.1668961493113254e-10, 9.999999999701312e-26, 9.9991052028519
19e-26, 1.0e-25, 1.0000894797148081e-25]
 [7.223608570186277e-15, 100000.0, 7.461420306318696e-18, 2.582533611682799
e-12, 3.2119181733692928e-21, 7.326734499751496e-25, 4.3773924270280184e-9,
 9.999999990031619e-26, 1.0000000010509517e-25, 19.999999999999993, 9.99999
9995622616, 4.377385195954844e-9, 9.999999996862394e-26, 9.99060402182374e-
26, 1.0e-25, 1.0009395978176092e-25]
 [6.801271144253272e-14, 99999.99999999999, 6.667727575960034e-16, 2.453003
559015065e-11, 2.7280563011545485e-18, 5.112610641877976e-21, 4.15773716006
23365e-8, 9.999999990253721e-26, 1.000004159759221e-25, 19.999999999999936,
 9.999999958422698, 4.157730291840595e-8, 9.999999970212093e-26, 9.91111054
108735e-26, 1.0e-25, 1.0088889458897444e-25]
 [4.688670696664225e-13, 99999.99999999991, 3.3502198071062406e-14, 1.79818
7063811648e-10, 1.0094934752286112e-15, 1.40394058351936e-17, 3.04733161782
702e-7, 1.0012518578000845e-25, 1.0832837691084396e-25, 19.9999999999995, 9
.999999695267343, 3.0473265838990135e-7, 9.999999782416538e-26, 9.366542477
130789e-26, 9.999999999999992e-26, 1.0633457522022484e-25]
 [1.407192931108865e-12, 99999.99999999967, 3.5227708315862376e-13, 6.43256
0108292992e-10, 3.844493412975846e-14, 1.9835526093574404e-15, 1.0895839966
423412e-6, 3.277646647465669e-25, 4.281481849481192e-24, 19.9999999999982, 
9.999998910417805, 1.0895821967438376e-6, 9.999999230964617e-26, 7.91372257
9957852e-26, 9.999999999999875e-26, 1.2086277409014347e-25]
 [2.264336329389599e-12, 99999.99999999933, 1.1048442517437004e-12, 1.30312
5335386832e-9, 2.4761966816846963e-13, 2.7241625847968868e-14, 2.2059518837
52828e-6, 1.325549574842794e-23, 1.1542392349565964e-22, 19.999999999996355
, 9.999997794051762, 2.2059482397109504e-6, 9.999998477135235e-26, 6.226782
004051789e-26, 9.999999999999473e-26, 1.3773218057370973e-25]
 [2.862272911676639e-12, 99999.99999999891, 2.148189240342992e-12, 2.132458
3134898403e-9, 7.96017688625694e-13, 1.526485583067051e-13, 3.6074090391910
9e-6, 2.0413399117497236e-22, 1.045075957056738e-21, 19.99999999999404, 9.9
99996392596923, 3.607403080062675e-6, 9.999997677622162e-26, 4.608484123748
2916e-26, 9.99999999999858e-26, 1.5391518650700624e-25]
 [3.2310257916806207e-12, 99999.99999999836, 3.313381370672394e-12, 3.21614
73182812564e-9, 1.8550088455135524e-12, 5.812484673113851e-13, 5.4365210991
13368e-6, 1.841212859746178e-21, 5.908062730539153e-21, 19.999999999991015,
 9.999994563487885, 5.43651211844888e-6, 9.999997887239649e-26, 3.111514449
375462e-26, 9.99999999999676e-26, 1.6888524420428603e-25]
 ⋮
 [3.500069496772314e-12, 99997.84446203698, 5.597356731185568e-12, 4.311061
628605185, 9.028274113939597e-12, 6.270265804991508e-10, 4.609840997728512,
 7.133586143904758e-6, 2.4710727829652507e-8, 19.999969922702828, 5.3905370
2480375, 4.609440056840723, 2.2918152522165948e-5, 7.388755328498843e-12, 3
.654987602185756e-10, 1.9527006026211283e-10]
 [3.5000744615996185e-12, 99997.70261580904, 5.597364656837386e-12, 4.59475
4084433942, 9.028286149861342e-12, 6.27024190391514e-10, 4.609858176154698,
 7.133585534993957e-6, 2.4710776773754933e-8, 19.999969922704036, 5.3905462
8408275, 4.609430797562398, 2.2918151878488494e-5, 7.388754084431795e-12, 3
.8958201984709413e-10, 1.9526902273094803e-10]
 [3.500079525961235e-12, 99997.55792627811, 5.597372741378864e-12, 4.884133
146273262, 9.028298426918215e-12, 6.270217517218217e-10, 4.6098757037719, 7
.133584913751338e-6, 2.471082671280785e-8, 19.99996992270527, 5.39055573153
9568, 4.609421350106304, 2.291815122177496e-5, 7.388752815384788e-12, 4.141
478424888236e-10, 1.952679642039994e-10]
 [3.5000847049817304e-12, 99997.40996135969, 5.5973810089538665e-12, 5.1800
62983055499, 9.028310981754676e-12, 6.270192571215692e-10, 4.60989363355267
5, 7.133584278305407e-6, 2.4710877797745704e-8, 19.999969922706526, 5.39056
5395724075, 4.609411685922468, 2.2918150550047035e-5, 7.388751517562551e-12
, 4.392695767585692e-10, 1.9526688149389007e-10]
 [3.500089955856012e-12, 99997.25994402144, 5.597389391229728e-12, 5.480097
659570318, 9.02832371058324e-12, 6.270167271232742e-10, 4.609911817932322, 
7.133583633892028e-6, 2.4710929608138583e-8, 19.999969922707802, 5.39057519
7096452, 4.609401884550778, 2.2918149868839462e-5, 7.388750201686396e-12, 4
.6473957448550204e-10, 1.9526578352314975e-10]
 [3.5000953046107117e-12, 99997.10713070499, 5.597397929753889e-12, 5.78572
42924656695, 9.028336676474965e-12, 6.270141490990692e-10, 4.60993034768474
3, 7.133582977300569e-6, 2.471098240262395e-8, 19.999969922709102, 5.390585
184581664, 4.609391897066163, 2.291814917475828e-5, 7.388748861228561e-12, 
4.906840673603892e-10, 1.952646648227387e-10]
 [3.5001007455718995e-12, 99996.95168353878, 5.597406615468294e-12, 6.09661
8624834236, 9.028349865652111e-12, 6.270115256854294e-10, 4.609949203872547
, 7.1335823092090095e-6, 2.4711036127252923e-8, 19.999969922710424, 5.39059
5347970304, 4.609381733678304, 2.2918148468520227e-5, 7.388747497604886e-12
, 5.170755180897057e-10, 1.9526352654929902e-10]
 [3.5001062748807087e-12, 99996.79371279391, 5.597415442212365e-12, 6.41256
0114566047, 9.028363268736126e-12, 6.270088586408815e-10, 4.609968373868464
, 7.133581630071954e-6, 2.4711090746056976e-8, 19.99996992271177, 5.3906056
804538975, 4.609371401195377, 2.2918147750605794e-5, 7.38874611177613e-12, 
5.438951941800562e-10, 1.9526236947888078e-10]
 [3.5001067013498338e-12, 99996.78152871574, 5.597416123008617e-12, 6.43692
8270871343, 9.028364302489385e-12, 6.270086528899915e-10, 4.609969852758919
, 7.133581577682343e-6, 2.4711094959688232e-8, 19.999969922711873, 5.390606
477562813, 4.609370604086524, 2.2918147695224838e-5, 7.38874600488603e-12, 
5.459637516310662e-10, 1.9526228022174094e-10]
```



```julia
abstols = 1.0 ./ 10.0 .^ (7:13)
reltols = 1.0 ./ 10.0 .^ (4:10)

setups = [
    Dict(:alg=>FBDF()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>NordsieckBDF()),
    Dict(:alg=>Rodas4P()),
    Dict(:alg=>CVODE_BDF()),
    #Dict(:alg=>ddebdf()),
    Dict(:alg=>Rodas4()),
    Dict(:alg=>Rodas5P()),
    #Dict(:alg=>rodas()),
    #Dict(:alg=>radau()),
    Dict(:alg=>lsoda()),
    #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEqCore.PolyesterThreads())),
    Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3, threading = false)),
    #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEqCore.PolyesterThreads())),
    Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false))
]
wp = WorkPrecisionSet(oprob, abstols, reltols, setups; verbose = SciMLLogging.None(),
    save_everystep = false, appxsol = refsol, maxiters = Int(1e5), numruns = 10)
plot(wp)
```

![](figures/astrochem_4_1.png)



## With Temperature Dynamics

```julia
@variables T(t) = 100.0
D = Differential(t)
# Interpolate T-dependent rates so @reaction does not also treat T as a parameter.
k_H3O_E = ka_reaction(T, 1.1e-7, -1/2)
k_H2O_E_OH = ka_reaction(T, 8.6e-8, -1/2)
k_H2O_E_O = ka_reaction(T, 3.9e-8, -1/2)
k_OH_E = ka_reaction(T, 6.3e-9, -0.48)
k_O_E = ka_reaction(T, 3.4e-12, -0.63)
k_Cp_E = ka_reaction(T, 4.4e-12, -0.61)
k_Cp_OH_CO = ka_reaction(T, 1.15e-10, -0.339)
k_Cp_OH_COp = 9.15e-10 * (0.62 + 0.4767 * 5.5 * sqrt(300 / T))
k_HCO_E = ka_reaction(T, 2.8e-7, -0.69)
k_Hp_E = ka_reaction(T, 3.5e-12, -0.7)
reaction_equations = [
    (@reaction 1.6e-9, $O⁺ + $H2 --> $OH⁺ + $H),
    (@reaction 1e-9, $OH⁺ + $H2 --> $H2O⁺ + $H),
    (@reaction 6.1e-10, $H2O⁺ + $H2 --> $H3O⁺ + $H),
    (@reaction $k_H3O_E, $H3O⁺ + $E --> $H2O + $H),
    (@reaction $k_H2O_E_OH, $H2O⁺ + $E --> $OH + $H),
    (@reaction $k_H2O_E_O, $H2O⁺ + $E --> $O + $H2),
    (@reaction $k_OH_E, $OH⁺ + $E --> $O + $H),
    (@reaction $k_O_E, $O⁺ + $E --> $O),
    (@reaction 2.8 * cosmic_ionisation_rate, $O --> $O⁺ + $E),
    (@reaction 2.62 * cosmic_ionisation_rate, $C --> $C⁺ + $E),
    (@reaction 5.0 * cosmic_ionisation_rate, $CO --> $C + $O),
    (@reaction $k_Cp_E, $C⁺ + $E --> $C),
    (@reaction $k_Cp_OH_CO, $C⁺ + $OH --> CO + $H),
    (@reaction $k_Cp_OH_COp, $C⁺ + $OH --> $CO⁺ + $H),
    (@reaction 4e-10, $CO⁺ + $H --> $CO + $H⁺),
    (@reaction 7.28e-10, $CO⁺ + $H2 --> $HCO⁺ + $H),
    (@reaction $k_HCO_E, $HCO⁺ + $E --> $CO + $H),
    (@reaction $k_Hp_E, $H⁺ + $E --> $H),
    (@reaction 2.121e-17 * dust2gas / 1e-2, $H + $H --> $H2),
    (@reaction 1e-1 * cosmic_ionisation_rate, $H2 --> $H + $H),
    (@reaction 3.39e-10 * radiation_field, $C --> $C⁺ + $E),
    (@reaction 2.43e-10 * radiation_field, $CO --> $C + $O),
    (@reaction 7.72e-10 * radiation_field, $H2O --> $OH + $H),
    (D(T) ~ get_heating_cooling(
        T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺, dust2gas))
]

@named system = ReactionSystem(reaction_equations, t)

u0 = [:H2 => number_density, :O => number_density*2e-4, :C => number_density*1e-4,
    :O⁺=>minimum_fractional_density, :OH⁺=>minimum_fractional_density,
    :H => minimum_fractional_density, :H2O⁺ => minimum_fractional_density,
    :H3O⁺=>minimum_fractional_density, :E=>minimum_fractional_density,
    :H2O=>minimum_fractional_density, :OH=>minimum_fractional_density,
    :C⁺=>minimum_fractional_density, :CO=>minimum_fractional_density,
    :CO⁺=>minimum_fractional_density, :H⁺=>minimum_fractional_density,
    :HCO⁺ => minimum_fractional_density, :T => 100.0]

tspan = (0.0, 1e6*seconds_per_year)

params = [dust2gas => 0.01, radiation_field => 1e-1, cosmic_ionisation_rate => 1e-17]

println("Attempting to solve the ODE...")

sys = ode_model(complete(system))
# oprob = ODEProblemExpr(sys, [], tspan, params)

ssys = structural_simplify(sys)

oprob = ODEProblem(ssys, merge(Dict{Any, Any}(u0), Dict{Any, Any}(params)), tspan)
println("ODEProblem created successfully.")
refsol = solve(oprob, Rodas5P(), abstol = 1e-14, reltol = 1e-14)
```

```
Attempting to solve the ODE...
ODEProblem created successfully.
retcode: Success
Interpolation: specialized 4th (Rodas6P = 5th) order "free" stiffness-aware
 interpolation
t: 11491-element Vector{Float64}:
    0.0
    0.03306297961107439
    0.12617696833110145
    1.057316855531372
    6.261690323857861
   27.19826262604805
   93.06543891029484
  263.96956434356827
  643.1511578903481
 1389.9907038126917
    ⋮
    3.150547601467421e13
    3.1509596734903207e13
    3.1513717455132203e13
    3.15178381753612e13
    3.1521958895590195e13
    3.152607961581919e13
    3.1530200336048188e13
    3.1534321056277184e13
    3.1536e13
u: 11491-element Vector{Vector{Float64}}:
 [1.0e-25, 100000.0, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 
1.0e-25, 20.0, 10.0, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 1.0e-25, 100.0]
 [1.851521970869178e-17, 100000.0, 4.907345567696503e-23, 6.61259597128843e
-15, 1.000541025692682e-25, 1.0000020171151974e-25, 1.120837726591731e-11, 
9.999999999974476e-26, 1.0000000000025525e-25, 20.0, 9.999999999988791, 1.1
208358750648673e-11, 9.999999999991965e-26, 9.999975930179811e-26, 1.0e-25,
 1.000002406982019e-25, 100.00000003481105]
 [7.065838912611004e-17, 100000.0, 7.133363059488448e-22, 2.523539437956275
4e-14, 1.0300029434488718e-25, 1.0000077545362807e-25, 4.277409598162033e-1
1, 9.999999999902593e-26, 1.0000000000097408e-25, 20.0, 9.999999999957225, 
4.277402532251791e-11, 9.999999999969339e-26, 9.999908143588938e-26, 1.0e-2
5, 1.0000091856411062e-25, 100.00000013284807]
 [5.920473592534142e-16, 100000.0, 5.007827915430374e-20, 2.114634211880853
e-13, 1.86496619450405e-24, 1.0003490864197849e-25, 3.584312831331701e-10, 
9.999999999183754e-26, 1.0000000000816249e-25, 19.999999999999996, 9.999999
999641568, 3.584306910357288e-10, 9.999999999743072e-26, 9.999230302952371e
-26, 1.0e-25, 1.0000769697047627e-25, 100.00000111321835]
 [3.5047906152222807e-15, 100000.0, 1.7555998542952994e-18, 1.2523398211044
454e-12, 3.665488112520461e-22, 1.350341974093558e-25, 2.1227181666719792e-
9, 9.999999995165989e-26, 1.0000000004848544e-25, 19.999999999999993, 9.999
999997877287, 2.122714660125385e-9, 9.999999998478446e-26, 9.99544252828729
7e-26, 1.0e-25, 1.0004557471712661e-25, 100.0000065927527]
 [1.519793449337562e-14, 99999.99999999999, 3.306258491474883e-17, 5.439685
647791985e-12, 2.9980156108095737e-20, 1.2540528089642273e-23, 9.2202333829
51568e-9, 9.999999979013191e-26, 1.0000000043402723e-25, 19.999999999999982
, 9.999999990779786, 9.220218151924504e-9, 9.999999993391559e-26, 9.9802192
54540353e-26, 1.0e-25, 1.0019780745458901e-25, 100.00002863626433]
 [5.1730543897239773e-14, 99999.99999999997, 3.849054369511803e-16, 1.86134
750821057e-11, 1.1947576821417641e-18, 1.698292106131739e-21, 3.15492602406
1316e-8, 9.999999944099834e-26, 1.0000010535182336e-25, 19.999999999999943,
 9.999999968450796, 3.154920812396739e-8, 9.999999977393868e-26, 9.93247735
7250873e-26, 1.0e-25, 1.0067522642740382e-25, 100.00009798590979]
 [1.4474478290231673e-13, 99999.99999999994, 3.0511720691794615e-15, 5.2797
018151620906e-11, 2.689234568218402e-17, 1.0871561925050839e-19, 8.94858988
950671e-8, 1.0000007998731001e-25, 1.0001899229610249e-25, 19.9999999999998
47, 9.999999910514253, 8.948575107211103e-8, 9.999999935926758e-26, 9.80966
4848376226e-26, 1.0e-25, 1.0190335151552857e-25, 100.00027792592209]
 [3.422530486136397e-13, 99999.99999999991, 1.7530512755434428e-14, 1.28648
5280043315e-10, 3.7734764366143763e-16, 3.739406248269907e-18, 2.1802876881
826152e-7, 1.0001677394059735e-25, 1.0158983328278126e-25, 19.9999999999996
38, 9.999999781971594, 2.180284086536134e-7, 9.999999844145873e-26, 9.54257
8087195534e-26, 9.999999999999996e-26, 1.0457421912375984e-25, 100.00067715
525888]
 [6.979131143109874e-13, 99999.99999999985, 7.681438941499765e-14, 2.780823
6753282137e-10, 3.5894903611241124e-15, 7.780004814324868e-17, 4.7120798006
3051e-7, 1.0164009379515017e-25, 1.7134261429398353e-25, 19.999999999999222
, 9.999999528792802, 4.7120720166825675e-7, 9.999999664316237e-26, 9.037601
06036272e-26, 9.999999999999977e-26, 1.0962398937571805e-25, 100.0014634810
2491]
 ⋮
 [3.5001071379535685e-12, 99996.78407319635, 5.599316448336524e-12, 6.43183
9125940741, 9.136069157949107e-12, 2.1987443323093008e-9, 8.154098543873568
, 7.21868471256555e-6, 5.299005818920598e-8, 19.99996971685321, 1.846321391
6315204, 8.153655599114575, 2.3007899092028474e-5, 7.390102945798092e-12, 3
.888387229236178e-9, 1.347944235834925e-9, 3757.141001727155]
 [3.500107152629995e-12, 99996.78365382088, 5.599316457093529e-12, 6.432677
876864353, 9.13606839442745e-12, 2.1987043172200764e-9, 8.15406392880594, 7
.218684078715501e-6, 5.298963125043e-8, 19.99996971685502, 1.84635609361338
87, 8.153620897133415, 2.300789841977878e-5, 7.390101894485517e-12, 3.88879
1626568236e-9, 1.3479081742012812e-9, 3756.9730118180114]
 [3.5001071673064066e-12, 99996.78323444587, 5.599316465852536e-12, 6.43351
6626840199, 9.136067631015363e-12, 2.1986643090440347e-9, 8.154029318578923
, 7.2186834449521486e-6, 5.298920437678353e-8, 19.99996971685683, 1.8463907
907548922, 8.153586199992619, 2.3007897747621405e-5, 7.390100843360567e-12,
 3.889196014633697e-9, 1.3478721190715666e-9, 3756.805054791114]
 [3.5001071819828016e-12, 99996.78281507135, 5.599316474613539e-12, 6.43435
5375868158, 9.136066867712819e-12, 2.198624307778992e-9, 8.153994713191162,
 7.21868281127547e-6, 5.298877756824755e-8, 19.999969716858637, 1.846425483
0573874, 8.153551507690828, 2.300789707555633e-5, 7.390099792423182e-12, 3.
889600393433864e-9, 1.3478360704436772e-9, 3756.6371306353003]
 [3.5001071966591796e-12, 99996.78239569732, 5.599316483376541e-12, 6.43519
4123948108, 9.13606610451978e-12, 2.1985843134227684e-9, 8.153960112641304,
 7.218682177685441e-6, 5.298835082480308e-8, 19.999969716860445, 1.84646017
05222283, 8.153516820226695, 2.300789640358352e-5, 7.390098741673322e-12, 3
.890004762970039e-9, 1.3478000283155105e-9, 3756.469239339413]
 [3.5001072113355415e-12, 99996.78197632375, 5.59931649214154e-12, 6.436032
871079926, 9.136065341436224e-12, 2.1985443259731704e-9, 8.153925516927991,
 7.218681544182038e-6, 5.2987924146430774e-8, 19.999969716862253, 1.8464948
531507694, 8.153482137598864, 2.3007895731702957e-5, 7.390097691110912e-12,
 3.890409123243523e-9, 1.3477639926849496e-9, 3756.3013808922515]
 [3.5001072260118884e-12, 99996.78155695065, 5.599316500908541e-12, 6.43687
1617263492, 9.13606457846212e-12, 2.198504345428025e-9, 8.153890926049872, 
7.2186809107652346e-6, 5.298749753311176e-8, 19.99996971686406, 1.846529530
9443641, 8.153447459805976, 2.3007895059914608e-5, 7.390096640735891e-12, 3
.890813474255619e-9, 1.3477279635498967e-9, 3756.1335552826877]
 [3.5001072406882176e-12, 99996.78113757803, 5.5993165096775325e-12, 6.4377
10362498684, 9.136063815597427e-12, 2.198464371785126e-9, 8.153856340005595
, 7.218680277435009e-6, 5.298707098482671e-8, 19.999969716865866, 1.8465642
03904365, 8.153412786846683, 2.300789438821846e-5, 7.390095590548213e-12, 3
.891217816007627e-9, 1.3476919409082301e-9, 3755.9657624994907]
 [3.5001072466679284e-12, 99996.78096670925, 5.59931651325094e-12, 6.438052
100070949, 9.136063504807675e-12, 2.1984480869271125e-9, 8.15384224967459, 
7.2186800194161495e-6, 5.298689721089741e-8, 19.999969716866605, 1.84657832
96483311, 8.153398661103006, 2.300789411456942e-5, 7.390095162714089e-12, 3
.891382558108296e-9, 1.347677265726522e-9, 3755.897406520325]
```



```julia
refsol = solve(oprob, Rodas5P(), abstol = 1e-13, reltol = 1e-13)

# Run Benchmark

abstols = 1.0 ./ 10.0 .^ (9:10)
reltols = 1.0 ./ 10.0 .^ (9:10)

setups = [
    Dict(:alg=>FBDF()),
    Dict(:alg=>QNDF()),
    Dict(:alg=>NordsieckBDF()),
    Dict(:alg=>CVODE_BDF()),
    #Dict(:alg=>ddebdf()),
    Dict(:alg=>Rodas5P()),
    Dict(:alg=>KenCarp4()),
    Dict(:alg=>KenCarp47()),
    #Dict(:alg=>RadauIIA9()),
    #Dict(:alg=>rodas()),
    #Dict(:alg=>radau()),
    Dict(:alg=>lsoda())
    #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEqCore.PolyesterThreads())),
    #Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = false)),
    #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEqCore.PolyesterThreads())),
    #Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false)),
]
wp = WorkPrecisionSet(oprob, abstols, reltols, setups; verbose = SciMLLogging.None(),
    save_everystep = false, appxsol = refsol, maxiters = Int(1e5), numruns = 10,
    print_names = true)
plot(wp)
```

```
FBDF
QNDF
NordsieckBDF
CVODE_BDF
Rodas5P
KenCarp4
KenCarp47
lsoda
```


![](figures/astrochem_6_1.png)
