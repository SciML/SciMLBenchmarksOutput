---
author: "Stella Offner and Chris Rackauckas"
title: "Nelson Work-Precision Diagrams"
---
```julia
using OrdinaryDiffEq
using DiffEqDevTools, Plots
using Sundials, LSODA
using ODEInterface, ODEInterfaceDiffEq
```


```julia
T = 10
Av = 2 # This is what depotic uses for visual extinction
Go = 1.7 # I think despotic uses 1.7
n_H = 611
shield = 1

function Nelson_ODE(du,u,p,t)
	# 1: H2
	#= du[1] = -1.2e-17 * u[1] + 
			n_H * (1.9e-6 * u[2] * u[3]) / (T^0.54) - 
			n_H * 4e-16 * u[1] * u[12] - 
			n_H * 7e-15 * u[1] * u[5] + 
			n_H * 1.7e-9 * u[10] * u[2] + 
			n_H * 2e-9 * u[2] * u[6] + 
			n_H * 2e-9 * u[2] * u[14] + 
			n_H * 8e-10 * u[2] * u[8] =#
	du[1] = 0

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
			n_H * 2e-9 * u[2] * u[13] ## CHECK I added this extra term from a CR ionization reaction
			+ 2.0e-10 * Go * exp(-1.9 * Av) * u[14] # this term was added as part of the skipped photoreaction
	
	
	# 4: He
	du[4] = n_H * (9e-11 * u[3] * u[5]) / (T^0.64) - 
			6.8e-18 * u[4] + 
			n_H * 7e-15 * u[1] * u[5] + 
			n_H * 1.6e-9 * u[10] * u[5]
	#du[4] = 0
	
	# 5: He+   6.8e-18 or 1.1
	du[5] = 6.8e-18 * u[4] - 
			n_H * (9e-11 * u[3] * u[5]) / (T^0.64) - 
			n_H * 7e-15 * u[1] * u[5] - 
			n_H * 1.6e-9 * u[10] * u[5]
	#u[5] = 0
	
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
			n_H * 2e-9 * u[2] * u[14] 
			+ 2.0e-10 * Go * exp(-1.9 * Av) * u[14] # this term was added as part of the skipped photoreaction
	
	# 14: M
	du[14] = n_H * (3.8e-10 * u[13] * u[3]) / (T^0.65) - 
			n_H * 2e-9 * u[2] * u[14] 
			- 2.0e-10 * Go * exp(-1.9 * Av) * u[14] # this term was added as part of the skipped photoreaction

end


u0 = [0.5 ;    # 1:  H2   yep?
	9.059e-9 ; # 2:  H3+  yep
	2.0e-4 ;   # 3:  e    yep
	0.1 ;              # 4:  He  SEE lines 535 NL99
	7.866e-7 ; # 5:  He+  yep? should be 2.622e-5
	0.0 ;      # 6:  C    yep
	0.0 ;      # 7:  CHx  yep
	0.0004 ;   # 8:  O    yep
	0.0 ;      # 9:  OHx  yep
	0.0 ;      # 10: CO   yep
	0.0 ;      # 11: HCO+ yep
	0.0002 ;   # 12: C+   yep
	2.0e-7 ;   # 13: M+   yep
	2.0e-7 ]   # 14: M    yep


tspan = (0.0, 30 * 3.16e10) # ~30 thousand yrs

prob = ODEProblem(Nelson_ODE, u0, tspan)
refsol = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14)

using Plots
plot(sol; yscale=:log10, idxs = (0,5))
```

```
Error: UndefVarError: `sol` not defined
```





## Run Benchmark

```julia
abstols = 1.0 ./ 10.0 .^ (8:10)
reltols = 1.0 ./ 10.0 .^ (8:10)

sol = solve(prob, CVODE_BDF(), abstol=1e-14, reltol=1e-14)

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
```

```
FBDF
QNDF
CVODE_BDF
Rodas5P
KenCarp4
KenCarp47
RadauIIA9
lsoda
```


![](figures/nelson_3_1.png)
