---
author: "Vaibhav Dixit, Chris Rackauckas"
title: "Lotka-Volterra Bayesian Parameter Estimation Benchmarks"
---


## Parameter Estimation of Lotka-Volterra Equation using DiffEqBayes.jl

```julia
using DiffEqBayes, StanSample, DynamicHMC, Turing
```


```julia
using Distributions, BenchmarkTools, StaticArrays
using OrdinaryDiffEq, RecursiveArrayTools, ParameterizedFunctions
using Plots, LinearAlgebra
```


```julia
gr(fmt=:png)
```

```
Plots.GRBackend()
```





#### Initializing the problem

```julia
f = @ode_def LotkaVolterraTest begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d
```

```
(::Main.var"##WeaveSandBox#225".LotkaVolterraTest{Main.var"##WeaveSandBox#2
25".var"###ParameterizedDiffEqFunction#227", Main.var"##WeaveSandBox#225".v
ar"###ParameterizedTGradFunction#228", Main.var"##WeaveSandBox#225".var"###
ParameterizedJacobianFunction#229", Nothing, Nothing, ModelingToolkit.ODESy
stem}) (generic function with 1 method)
```



```julia
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1,0]
```

```
5-element Vector{Float64}:
 1.5
 1.0
 3.0
 1.0
 0.0
```



```julia
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob,Tsit5())
```

```
retcode: Success
Interpolation: specialized 4th order "free" interpolation
t: 34-element Vector{Float64}:
  0.0
  0.0776084743154256
  0.23264513699277584
  0.4291185174543143
  0.6790821987497083
  0.9444046158046306
  1.2674601546021105
  1.6192913303893046
  1.9869754428624007
  2.2640902393538296
  ⋮
  7.584863345264154
  7.978068981329682
  8.48316543760351
  8.719248247740158
  8.949206788834692
  9.200185054623292
  9.438029017301554
  9.711808134779586
 10.0
u: 34-element Vector{Vector{Float64}}:
 [1.0, 1.0]
 [1.0454942346944578, 0.8576684823217128]
 [1.1758715885138271, 0.6394595703175443]
 [1.419680960717083, 0.4569962601282089]
 [1.8767193950080012, 0.3247334292791134]
 [2.588250064553348, 0.26336255535952197]
 [3.860708909220769, 0.2794458098285261]
 [5.750812667710401, 0.522007253793458]
 [6.8149789991301635, 1.9177826328390826]
 [4.392999292571394, 4.1946707928506015]
 ⋮
 [2.6142539677883248, 0.26416945387526314]
 [4.24107612719179, 0.3051236762922018]
 [6.791123785297775, 1.1345287797146668]
 [6.26537067576476, 2.741693507540315]
 [3.780765111887945, 4.431165685863443]
 [1.816420140681737, 4.064056625315896]
 [1.1465021407690763, 2.791170661621642]
 [0.9557986135403417, 1.623562295185047]
 [1.0337581256020802, 0.9063703842885995]
```



```julia
su0 = SA[1.0,1.0]
sp = SA[1.5,1.0,3.0,1,0]
sprob = ODEProblem{false,SciMLBase.FullSpecialize}(f, su0, tspan, sp)
sol = solve(sprob,Tsit5())
```

```
retcode: Success
Interpolation: specialized 4th order "free" interpolation
t: 34-element Vector{Float64}:
  0.0
  0.0776084743154256
  0.23264513699277584
  0.4291185174543143
  0.6790821987497083
  0.9444046158046306
  1.2674601546021105
  1.6192913303893046
  1.9869754428624007
  2.2640902393538296
  ⋮
  7.584863345264154
  7.978068981329682
  8.48316543760351
  8.719248247740158
  8.949206788834692
  9.200185054623292
  9.438029017301554
  9.711808134779586
 10.0
u: 34-element Vector{StaticArraysCore.SVector{2, Float64}}:
 [1.0, 1.0]
 [1.0454942346944578, 0.8576684823217128]
 [1.1758715885138271, 0.6394595703175443]
 [1.419680960717083, 0.4569962601282089]
 [1.8767193950080012, 0.3247334292791134]
 [2.588250064553348, 0.26336255535952197]
 [3.860708909220769, 0.2794458098285261]
 [5.750812667710401, 0.522007253793458]
 [6.8149789991301635, 1.9177826328390826]
 [4.392999292571394, 4.1946707928506015]
 ⋮
 [2.6142539677883248, 0.26416945387526314]
 [4.241076127191789, 0.30512367629220183]
 [6.791123785297779, 1.1345287797146653]
 [6.265370675764766, 2.7416935075403135]
 [3.7807651118879293, 4.431165685863457]
 [1.8164201406817235, 4.064056625315901]
 [1.146502140769069, 2.791170661621637]
 [0.9557986135403385, 1.6235622951850437]
 [1.033758125602079, 0.9063703842885992]
```





#### We take the solution data obtained and add noise to it to obtain data for using in the Bayesian Inference of the parameters

```julia
t = collect(range(1,stop=10,length=10))
sig = 0.49
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(2)) for i in 1:length(t)]))
```

```
2×10 Matrix{Float64}:
 3.18084   6.72495  1.47715   1.89996   …  4.20071  3.44035  0.862223
 0.937477  1.77722  1.14996  -0.101926     1.13171  4.23348  0.929488
```





#### Plots of the actual data and generated data

```julia
scatter(t, data[1,:], lab="#prey (data)")
scatter!(t, data[2,:], lab="#predator (data)")
plot!(sol)
```

![](figures/DiffEqBayesLotkaVolterra_9_1.png)

```julia
priors = [truncated(Normal(1.5,0.5),0.5,2.5),truncated(Normal(1.2,0.5),0,2),truncated(Normal(3.0,0.5),1,4),truncated(Normal(1.0,0.5),0,2)]
```

```
4-element Vector{Distributions.Truncated{Distributions.Normal{Float64}, Dis
tributions.Continuous, Float64, Float64, Float64}}:
 Truncated(Distributions.Normal{Float64}(μ=1.5, σ=0.5); lower=0.5, upper=2.
5)
 Truncated(Distributions.Normal{Float64}(μ=1.2, σ=0.5); lower=0.0, upper=2.
0)
 Truncated(Distributions.Normal{Float64}(μ=3.0, σ=0.5); lower=1.0, upper=4.
0)
 Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.5); lower=0.0, upper=2.
0)
```





### Stan.jl backend

The solution converges for tolerance values lower than 1e-3, lower tolerance leads to better accuracy in result but is accompanied by longer warmup and sampling time, truncated normal priors are used for preventing Stan from stepping into negative values.

```julia
@btime bayesian_result_stan = stan_inference(prob,t,data,priors,num_samples=10_000,print_summary=false,delta = 0.65, vars = (DiffEqBayes.StanODEData(), InverseGamma(2, 3)))
```

```
37.269094 seconds (2.14 M allocations: 146.043 MiB, 0.12% gc time, 4.36% c
ompilation time)
 38.159040 seconds (674 allocations: 56.625 KiB)
 30.567834 seconds (674 allocations: 56.625 KiB)
 32.539313 seconds (674 allocations: 56.625 KiB)
  51.473 s (260872 allocations: 31.96 MiB)
Chains MCMC chain (10000×6×1 Array{Float64, 3}):

Iterations        = 1:1:10000
Number of chains  = 1
Samples per chain = 10000
parameters        = sigma1.1, sigma1.2, theta_1, theta_2, theta_3, theta_4
internals         = 

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rha
t   ⋯
      Symbol   Float64   Float64   Float64     Float64     Float64   Float6
4   ⋯

    sigma1.1    0.5012    0.1526    0.0026   2906.8051   1702.2976    1.000
4   ⋯
    sigma1.2    0.7358    0.1993    0.0037   3395.0329   2966.8413    1.000
6   ⋯
     theta_1    1.5339    0.1091    0.0024   1969.9347   2373.7754    0.999
9   ⋯
     theta_2    1.0615    0.1343    0.0028   2700.0659   2616.4125    1.000
4   ⋯
     theta_3    2.9182    0.2926    0.0066   2002.3218   2277.4273    1.000
0   ⋯
     theta_4    0.9775    0.1050    0.0023   2019.2970   2571.7774    0.999
9   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

    sigma1.1    0.2938    0.3917    0.4723    0.5777    0.8721
    sigma1.2    0.4626    0.5989    0.6990    0.8298    1.2335
     theta_1    1.3419    1.4597    1.5253    1.6026    1.7630
     theta_2    0.8467    0.9701    1.0466    1.1342    1.3662
     theta_3    2.3936    2.7152    2.9064    3.0986    3.5281
     theta_4    0.7866    0.9051    0.9737    1.0421    1.1968
```





### Direct Turing.jl

```julia
@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(1.2, 0.5), 0, 2)
    γ ~ truncated(Normal(3.0, 0.5), 1, 4)
    δ ~ truncated(Normal(1.0, 0.5), 0, 2)

    # Simulate Lotka-Volterra model. 
    p = SA[α, β, γ, δ]
    _prob = remake(prob, p = p)
    predicted = solve(_prob, Tsit5(); saveat=t)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(data, sprob)

@time chain = sample(model, Turing.NUTS(0.65), 10000; progress=false)
```

```
47.397708 seconds (144.00 M allocations: 23.410 GiB, 7.01% gc time, 45.57%
 compilation time)
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 39.44 seconds
Compute duration  = 39.44 seconds
parameters        = σ, α, β, γ, δ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rha
t   ⋯
      Symbol   Float64   Float64   Float64     Float64     Float64   Float6
4   ⋯

           σ    0.5493    0.1061    0.0020   2677.0180   2511.9880    1.000
6   ⋯
           α    1.5230    0.0997    0.0022   2030.6911   2654.5861    1.000
5   ⋯
           β    1.0405    0.1054    0.0020   2966.9190   3112.9225    0.999
9   ⋯
           γ    2.9346    0.2661    0.0058   2130.7072   2787.3330    1.000
1   ⋯
           δ    0.9856    0.0990    0.0022   2045.7468   2845.2720    1.000
2   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    0.3836    0.4747    0.5348    0.6065    0.8004
           α    1.3493    1.4539    1.5147    1.5826    1.7397
           β    0.8650    0.9677    1.0312    1.1016    1.2737
           γ    2.4279    2.7571    2.9310    3.1072    3.4824
           δ    0.7973    0.9190    0.9823    1.0502    1.1892
```





### Turing.jl backend

```julia
@btime bayesian_result_turing = turing_inference(prob, Tsit5(), t, data, priors, num_samples=10_000)
```

```
19.781 s (112829776 allocations: 18.20 GiB)
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 19.71 seconds
Compute duration  = 19.71 seconds
parameters        = theta[1], theta[2], theta[3], theta[4], σ[1]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, h
amiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, 
tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rha
t   ⋯
      Symbol   Float64   Float64   Float64     Float64     Float64   Float6
4   ⋯

    theta[1]    1.5229    0.0994    0.0022   1965.0563   2831.2972    1.000
3   ⋯
    theta[2]    1.0392    0.1039    0.0020   2721.8731   3510.8975    1.001
9   ⋯
    theta[3]    2.9354    0.2667    0.0059   2071.7566   2633.9494    1.000
2   ⋯
    theta[4]    0.9856    0.1004    0.0023   2025.2440   2813.9824    1.000
2   ⋯
        σ[1]    0.5540    0.1068    0.0019   3063.7138   3763.6247    1.001
5   ⋯
                                                                1 column om
itted

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

    theta[1]    1.3465    1.4543    1.5158    1.5844    1.7337
    theta[2]    0.8653    0.9695    1.0283    1.0984    1.2695
    theta[3]    2.4385    2.7566    2.9279    3.1048    3.4962
    theta[4]    0.8027    0.9182    0.9820    1.0490    1.1961
        σ[1]    0.3908    0.4785    0.5379    0.6118    0.8011
```





### DynamicHMC.jl backend

```julia
@btime bayesian_result_dynamichmc = dynamichmc_inference(prob,Tsit5(),t,data,priors,num_samples=10_000)
```

```
30.969 s (261383358 allocations: 19.33 GiB)
(posterior = @NamedTuple{parameters::Vector{Float64}, σ::Vector{Float64}}[(
parameters = [1.3222578549539439, 0.9251465117615063, 3.530848595308553, 1.
1911775078852416], σ = [0.42463764407357013, 0.6497097498548102]), (paramet
ers = [1.5966881409820233, 1.0761608434690144, 2.7659684034914207, 0.904737
8893685866], σ = [0.28741256640072377, 0.5415211251806117]), (parameters = 
[1.5267631123553367, 0.998973316508563, 2.852759646683009, 0.97494581601418
5], σ = [0.2738405368321744, 0.3894380261298722]), (parameters = [1.6086267
367792269, 1.0683381604321225, 2.678577599677015, 0.9117976103649813], σ = 
[0.25623261151085447, 0.49162523869800956]), (parameters = [1.6218643195356
621, 1.0513888812431291, 2.6743531396570304, 0.8834236913394804], σ = [0.28
38363591038976, 0.6641989289723392]), (parameters = [1.5852623782525286, 0.
9964331064541775, 2.7729762553938007, 0.904012682024255], σ = [0.1870111699
57959, 0.6624770223497735]), (parameters = [1.3522273781151966, 0.864562885
6216209, 3.5159024904237537, 1.1498098176160536], σ = [0.4723687132027934, 
1.2751931042424078]), (parameters = [1.3565148520061214, 0.8276867738964817
, 3.3569466159970207, 1.178897632344254], σ = [0.3797778520563567, 1.284349
7099794297]), (parameters = [1.4576821942889808, 1.016374779965666, 3.04032
6021596035, 1.0645748245540418], σ = [0.5377907320508352, 0.367595573978023
17]), (parameters = [1.4493162601414449, 0.9201893983066607, 3.080861224754
194, 1.0399963448922047], σ = [0.46906772710926087, 0.38878752615229123])  
…  (parameters = [1.554818230931119, 1.0818159083635377, 2.722419764298772,
 0.9561997456330118], σ = [0.7175899333914487, 0.6408747368287047]), (param
eters = [1.5595825042032552, 1.1847464304262263, 2.963845191541017, 0.92708
85625585458], σ = [0.6233976544301695, 0.533223123599574]), (parameters = [
1.4459879335021069, 0.8688640220318478, 3.062874071472865, 1.05913128189687
01], σ = [0.5545025070022185, 0.5224107406781667]), (parameters = [1.397101
6867280315, 0.9977073039188887, 3.2991756315485583, 1.118608348517123], σ =
 [0.32604306786049514, 0.7257923190294405]), (parameters = [1.4232811932108
87, 0.9720791015473629, 3.1762118740371075, 1.1056047010958414], σ = [0.350
68375513341493, 0.6999171472688328]), (parameters = [1.4545906836751918, 0.
8382384558903104, 3.111677634298614, 1.047260443294073], σ = [0.31581182179
82505, 0.8858784898387406]), (parameters = [1.4551968752612483, 0.902735070
5272859, 3.1262658401889922, 1.034105591163242], σ = [0.39554220927201494, 
0.6465101702933306]), (parameters = [1.690719259646092, 1.179307672999047, 
2.5247317163721252, 0.8368718139618843], σ = [0.2342228010022602, 0.8176197
634998806]), (parameters = [1.6197435091094246, 1.171632016602915, 2.709060
0246296517, 0.8785780013063488], σ = [0.29152522149515253, 0.79119740611525
99]), (parameters = [1.726477996773983, 1.1407042872473159, 2.4880294880820
6, 0.8040843315924617], σ = [0.3054494157039274, 0.6835413700658756])], pos
terior_matrix = [0.2793407715429747 0.4679315721310914 … 0.482267808998906 
0.5460834933554528; -0.07780316291873636 0.07339993334604998 … 0.1583976628
453404 0.13164586748010051; … ; -0.8565190759095201 -1.2468365818646918 … -
1.232628753795041 -1.1859710927907037; -0.4312295545054496 -0.6133732009648
201 … -0.23420777709753762 -0.380468097852621], tree_statistics = DynamicHM
C.TreeStatisticsNUTS[DynamicHMC.TreeStatisticsNUTS(-22.538228215771195, 3, 
turning at positions -2:-9, 0.9870704272138824, 15, DynamicHMC.Directions(0
xa939da66)), DynamicHMC.TreeStatisticsNUTS(-21.674421348553736, 6, turning 
at positions -13:50, 0.9954429776600223, 63, DynamicHMC.Directions(0xd66097
b2)), DynamicHMC.TreeStatisticsNUTS(-18.47899450445755, 5, turning at posit
ions -1:30, 0.9996197634903131, 31, DynamicHMC.Directions(0xa82be7fe)), Dyn
amicHMC.TreeStatisticsNUTS(-18.73036073428713, 5, turning at positions -54:
-61, 0.9963253285213837, 63, DynamicHMC.Directions(0x8bb63742)), DynamicHMC
.TreeStatisticsNUTS(-17.749653072915812, 5, turning at positions -8:-39, 0.
9861189322769188, 63, DynamicHMC.Directions(0x9aed1cd8)), DynamicHMC.TreeSt
atisticsNUTS(-21.788363313172628, 5, turning at positions -13:-44, 0.949617
0628976832, 63, DynamicHMC.Directions(0xfc124f13)), DynamicHMC.TreeStatisti
csNUTS(-27.251629633969245, 6, turning at positions -63:-126, 0.95846352965
61347, 127, DynamicHMC.Directions(0xecc71381)), DynamicHMC.TreeStatisticsNU
TS(-24.314660110108694, 6, turning at positions -36:27, 0.988700417944279, 
63, DynamicHMC.Directions(0x28a22cdb)), DynamicHMC.TreeStatisticsNUTS(-26.4
35695988258118, 6, turning at positions -9:54, 0.7729370492648747, 63, Dyna
micHMC.Directions(0x76e02436)), DynamicHMC.TreeStatisticsNUTS(-23.752519834
84638, 5, turning at positions -5:26, 0.9892064412432646, 31, DynamicHMC.Di
rections(0xf464123a))  …  DynamicHMC.TreeStatisticsNUTS(-26.15166573453191,
 5, turning at positions -10:-41, 0.6651840816551673, 63, DynamicHMC.Direct
ions(0xc5b8d896)), DynamicHMC.TreeStatisticsNUTS(-23.06691505556487, 5, tur
ning at positions 28:59, 0.9123343951538908, 63, DynamicHMC.Directions(0x7b
fcdffb)), DynamicHMC.TreeStatisticsNUTS(-25.25032233897643, 6, turning at p
ositions 37:40, 0.9815649553671265, 83, DynamicHMC.Directions(0x13ba2fd4)),
 DynamicHMC.TreeStatisticsNUTS(-21.425626836184648, 5, turning at positions
 28:59, 0.9711058940690602, 63, DynamicHMC.Directions(0x3b38fffb)), Dynamic
HMC.TreeStatisticsNUTS(-22.790295956836765, 5, turning at positions -13:18,
 0.7463131360877131, 31, DynamicHMC.Directions(0xf8714032)), DynamicHMC.Tre
eStatisticsNUTS(-21.29793971532498, 5, turning at positions 8:39, 0.9864656
496132187, 63, DynamicHMC.Directions(0x9aa30567)), DynamicHMC.TreeStatistic
sNUTS(-20.013803235785254, 5, turning at positions 28:59, 0.991976611545151
9, 63, DynamicHMC.Directions(0x7ab9f17b)), DynamicHMC.TreeStatisticsNUTS(-2
0.83778069899103, 6, turning at positions 55:86, 0.9612983417086235, 95, Dy
namicHMC.Directions(0x23fd45f6)), DynamicHMC.TreeStatisticsNUTS(-19.1953155
74562887, 6, turning at positions -32:31, 0.939434424544655, 63, DynamicHMC
.Directions(0x29091b5f)), DynamicHMC.TreeStatisticsNUTS(-20.551238667390088
, 6, turning at positions -7:56, 0.7336248798249541, 63, DynamicHMC.Directi
ons(0x5a645e38))], κ = Gaussian kinetic energy (Diagonal), √diag(M⁻¹): [0.0
6756663495222748, 0.12064729178457598, 0.09190133503193172, 0.0985857049994
9916, 0.2814517830972898, 0.2642359285268731], ϵ = 0.05661549801185918)
```






## Conclusion

Lotka-Volterra Equation is a "predator-prey" model, it models population of two species in which one is the predator (wolf) and the other is the prey (rabbit).
It depicts a cyclic behaviour, which is also seen in its Uncertainty Quantification Plots. This behaviour makes it easy to estimate even at very high tolerance values (1e-3).




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/BayesianInference","DiffEqBayesLotkaVolterra.jmd")
```

Computer Information:

```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/BayesianInference/Project.toml`
⌃ [6e4b80f9] BenchmarkTools v1.3.2
⌃ [ebbdde9d] DiffEqBayes v3.6.0
⌅ [459566f4] DiffEqCallbacks v2.29.1
⌃ [31c24e10] Distributions v0.25.100
⌃ [bbc10e6e] DynamicHMC v3.4.6
⌃ [1dea7af3] OrdinaryDiffEq v6.55.0
⌃ [65888b18] ParameterizedFunctions v5.15.0
⌃ [91a5bcdd] Plots v1.39.0
⌅ [731186ca] RecursiveArrayTools v2.38.7
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [c1514b29] StanSample v7.4.2
⌃ [90137ffa] StaticArrays v1.6.2
⌅ [fce5fe82] Turing v0.28.3
  [37e2e46d] LinearAlgebra
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/BayesianInference/Manifest.toml`
⌅ [47edcb42] ADTypes v0.2.1
  [a4c015fc] ANSIColoredPrinters v0.0.1
⌅ [c3fe647b] AbstractAlgebra v0.31.1
  [621f4979] AbstractFFTs v1.5.0
⌅ [80f14c24] AbstractMCMC v4.4.2
⌅ [7a57a42e] AbstractPPL v0.6.2
⌃ [1520ce14] AbstractTrees v0.4.4
⌅ [79e6a3ab] Adapt v3.6.2
⌅ [0bf59076] AdvancedHMC v0.5.4
⌅ [5b7e9947] AdvancedMH v0.7.5
⌅ [576499cb] AdvancedPS v0.4.3
⌅ [b5ca4192] AdvancedVI v0.2.4
⌃ [dce04be8] ArgCheck v2.3.0
⌅ [ec485272] ArnoldiMethod v0.2.0
⌃ [4fba245c] ArrayInterface v7.4.11
  [30b0a656] ArrayInterfaceCore v0.1.29
⌅ [a9b6321e] Atomix v0.1.0
⌃ [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.7
⌅ [198e06fe] BangBang v0.3.39
  [9718e550] Baselet v0.1.1
⌃ [6e4b80f9] BenchmarkTools v1.3.2
⌅ [e2ed5e7c] Bijections v0.1.4
⌅ [76274a88] Bijectors v0.13.6
⌃ [d1d4a3ce] BitFlags v0.1.7
⌃ [62783981] BitTwiddlingConvenienceFunctions v0.1.5
⌅ [fa961155] CEnum v0.4.2
⌃ [2a0fbf3d] CPUSummary v0.2.3
⌃ [00ebfdb7] CSTParser v3.3.6
⌃ [336ed68f] CSV v0.10.11
⌃ [49dc2e85] Calculus v0.5.1
⌃ [082447d4] ChainRules v1.53.0
⌃ [d360d2e6] ChainRulesCore v1.16.0
⌃ [9e997f8a] ChangesOfVariables v0.1.8
⌃ [fb6a15b2] CloseOpenIntervals v0.1.12
⌃ [944b1d66] CodecZlib v0.7.2
⌃ [35d6a980] ColorSchemes v3.23.0
⌅ [3da002f7] ColorTypes v0.11.4
⌅ [c3611d14] ColorVectorSpace v0.10.0
⌅ [5ae59095] Colors v0.12.10
⌃ [861a8166] Combinatorics v1.0.2
⌅ [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
⌃ [bbf7d656] CommonSubexpressions v0.3.0
⌃ [34da2185] Compat v4.9.0
⌃ [5224ae11] CompatHelperLocal v0.1.25
⌃ [b152e2b5] CompositeTypes v0.1.3
  [a33af91c] CompositionsBase v0.1.2
⌃ [f0e56b4a] ConcurrentUtilities v2.2.1
⌃ [8f4d0f93] Conda v1.9.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
⌅ [187b0558] ConstructionBase v1.5.3
⌃ [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
⌃ [9a962f9c] DataAPI v1.15.0
⌃ [a93c6f00] DataFrames v1.6.1
⌃ [864edb3b] DataStructures v0.18.15
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
⌃ [2b5f629d] DiffEqBase v6.128.2
⌃ [ebbdde9d] DiffEqBayes v3.6.0
⌅ [459566f4] DiffEqCallbacks v2.29.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌃ [b4f34e82] Distances v0.10.9
⌃ [31c24e10] Distributions v0.25.100
⌃ [ced4e74d] DistributionsAD v0.6.52
⌃ [ffbed154] DocStringExtensions v0.9.3
⌅ [e30172f5] Documenter v0.27.25
⌅ [5b8099bc] DomainSets v0.6.7
⌃ [fa6b7ba4] DualNumbers v0.6.8
⌃ [bbc10e6e] DynamicHMC v3.4.6
⌅ [366bfd00] DynamicPPL v0.23.14
⌅ [7c1d4256] DynamicPolynomials v0.5.2
⌅ [cad2338a] EllipticalSliceSampling v1.1.0
⌃ [4e289a0a] EnumX v1.0.4
⌃ [460bff9d] ExceptionUnwrapping v0.1.9
⌃ [d4d017d3] ExponentialUtilities v1.24.0
  [e2ba6199] ExprTools v0.1.10
⌃ [c87230d0] FFMPEG v0.4.1
⌃ [7a1cc6ca] FFTW v1.7.1
⌅ [7034ab61] FastBroadcast v0.2.6
  [9aa1b823] FastClosures v0.3.2
⌃ [29a986be] FastLapackInterface v2.0.0
⌃ [48062228] FilePathsBase v0.9.20
⌃ [1a297f60] FillArrays v1.6.1
⌃ [6a86dc24] FiniteDiff v2.21.1
⌃ [53c48c17] FixedPointNumbers v0.8.4
⌃ [59287772] Formatting v0.4.2
⌅ [f6369f11] ForwardDiff v0.10.36
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
⌅ [d9f16b24] Functors v0.4.5
⌅ [46192b85] GPUArraysCore v0.1.5
⌅ [28b8d3ca] GR v0.72.9
⌃ [c145ed77] GenericSchur v0.5.3
⌃ [d7ba0133] Git v1.3.0
  [c27321d9] Glob v1.3.1
⌃ [86223c79] Graphs v1.8.0
  [42e2da0e] Grisu v1.0.2
⌅ [0b43b601] Groebner v0.4.2
⌅ [d5909c97] GroupsCore v0.4.0
⌃ [cd3eb016] HTTP v1.9.14
⌃ [eafb193a] Highlights v0.5.2
⌃ [3e5b6fbb] HostCPUFeatures v0.1.16
⌃ [34004b35] HypergeometricFunctions v0.3.23
⌃ [7073ff75] IJulia v1.24.2
⌃ [b5f81e59] IOCapture v0.2.3
  [615f187c] IfElse v0.1.1
⌃ [d25df0c9] Inflate v0.1.3
  [22cec73e] InitialValues v0.3.1
⌃ [842dd82b] InlineStrings v1.4.0
  [505f98c9] InplaceOps v0.3.0
  [18e54dd8] IntegerMathUtils v0.1.2
⌅ [a98d9a8b] Interpolations v0.14.7
⌃ [8197267c] IntervalSets v0.7.7
⌃ [3587e190] InverseFunctions v0.1.12
⌃ [41ab1584] InvertedIndices v1.3.0
⌃ [92d709cd] IrrationalConstants v0.2.2
⌃ [c8e1da08] IterTools v1.8.0
  [82899510] IteratorInterfaceExtensions v1.0.0
⌃ [1019f520] JLFzf v0.1.5
⌃ [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
⌅ [98e50ef6] JuliaFormatter v1.0.35
⌃ [ccbc3e58] JumpProcesses v9.7.2
⌅ [ef3ab10e] KLU v0.4.0
⌃ [63c18a36] KernelAbstractions v0.9.8
⌃ [5ab0869b] KernelDensity v0.6.7
⌅ [ba0b0d4f] Krylov v0.9.3
⌅ [929cbde3] LLVM v6.1.0
⌃ [8ac3fa9e] LRUCache v1.4.1
⌃ [b964fa9f] LaTeXStrings v1.3.0
⌃ [2ee39098] LabelledArrays v1.14.0
⌅ [984bce1d] LambertW v0.4.6
⌅ [23fbe1c1] Latexify v0.15.21
⌃ [10f19ff3] LayoutPointers v0.1.14
  [50d2b5c4] Lazy v0.15.1
⌃ [1fad7336] LazyStack v0.1.1
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
⌅ [6f1fad26] Libtask v0.8.6
⌃ [d3d80556] LineSearches v7.2.0
⌅ [7ed4a6bd] LinearSolve v2.5.1
⌃ [6fdf6af0] LogDensityProblems v2.1.1
⌃ [996a588d] LogDensityProblemsAD v1.5.0
⌃ [2ab3a3ac] LogExpFunctions v0.3.26
⌃ [e6f89c97] LoggingExtras v1.0.1
⌃ [bdcacae8] LoopVectorization v0.12.165
⌅ [c7f686f2] MCMCChains v6.0.3
⌃ [be115224] MCMCDiagnosticTools v0.3.5
⌃ [e80e1ace] MLJModelInterface v1.9.2
  [d8e11817] MLStyle v0.4.17
⌃ [1914dd2f] MacroTools v0.5.11
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
⌃ [739be429] MbedTLS v1.1.7
  [442fdcdd] Measures v0.3.2
⌅ [128add7d] MicroCollections v0.1.4
⌃ [e1d29d7a] Missings v1.1.0
⌅ [961ee093] ModelingToolkit v8.65.0
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [102ac46a] MultivariatePolynomials v0.5.1
⌃ [ffc61752] Mustache v1.0.17
⌃ [d8a4904e] MutableArithmetics v1.3.1
⌃ [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
⌃ [872c559c] NNlib v0.9.4
⌃ [77ba4419] NaNMath v1.0.2
⌃ [86f7a689] NamedArrays v0.10.0
  [d9ec5142] NamedTupleTools v0.14.3
  [c020b1a1] NaturalSort v1.0.0
⌅ [8913a72c] NonlinearSolve v1.10.0
⌃ [6fe1bfb0] OffsetArrays v1.12.10
⌃ [4d8831e6] OpenSSL v1.4.1
⌃ [429524aa] Optim v1.7.7
⌅ [3bd65402] Optimisers v0.2.20
⌃ [bac558e1] OrderedCollections v1.6.2
⌃ [1dea7af3] OrdinaryDiffEq v6.55.0
⌃ [90014a1f] PDMats v0.11.17
⌃ [65ce6f38] PackageExtensionCompat v1.0.1
⌃ [65888b18] ParameterizedFunctions v5.15.0
  [d96e819e] Parameters v0.12.3
⌃ [69de0a69] Parsers v2.7.2
  [b98c9c47] Pipe v1.3.0
⌃ [ccf2f8ad] PlotThemes v3.1.0
⌃ [995b91a9] PlotUtils v1.3.5
⌃ [91a5bcdd] Plots v1.39.0
  [e409e4f3] PoissonRandom v0.4.4
⌃ [f517fe37] Polyester v0.7.5
⌃ [1d0040c9] PolyesterWeave v0.2.1
⌃ [2dfb63ee] PooledArrays v1.4.2
  [85a6dd25] PositiveFactorizations v0.2.4
⌃ [d236fae5] PreallocationTools v0.4.12
⌅ [aea7be01] PrecompileTools v1.2.0
⌃ [21216c6a] Preferences v1.4.0
⌃ [08abe8d2] PrettyTables v2.2.7
⌃ [27ebfcd6] Primes v0.5.4
  [33c8b6b6] ProgressLogging v0.1.4
⌃ [92933f4c] ProgressMeter v1.8.0
⌃ [1fd47b50] QuadGK v2.8.2
⌃ [74087812] Random123 v1.6.1
⌃ [fb686558] RandomExtensions v0.4.3
⌃ [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌅ [731186ca] RecursiveArrayTools v2.38.7
⌃ [f2c3362d] RecursiveFactorization v0.2.20
  [189a3867] Reexport v1.2.2
⌃ [05181044] RelocatableFolders v1.0.0
⌃ [ae029012] Requires v1.3.0
⌅ [79098fc4] Rmath v0.7.1
⌃ [f2b01f46] Roots v2.0.19
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.12
⌃ [fdea26ae] SIMD v3.4.5
  [94e857df] SIMDTypes v0.1.0
⌃ [476501e8] SLEEFPirates v0.6.39
⌅ [0bca4576] SciMLBase v1.95.0
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [e9a6253c] SciMLNLSolve v0.1.8
⌅ [c0aeaf25] SciMLOperators v0.3.6
  [30f210dd] ScientificTypesBase v3.0.0
⌃ [6c6a2e73] Scratch v1.2.0
⌃ [91c51154] SentinelArrays v1.4.0
⌃ [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
⌃ [777ac1f9] SimpleBufferStream v1.1.0
⌅ [727e6d20] SimpleNonlinearSolve v0.1.19
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [66db9d55] SnoopPrecompile v1.0.3
  [b85f4697] SoftGlobalScope v1.1.0
⌃ [a2af1166] SortingAlgorithms v1.1.1
⌃ [47a9eef4] SparseDiffTools v2.5.0
⌃ [e56a9233] Sparspak v0.3.9
⌃ [276daf66] SpecialFunctions v2.3.1
  [171d559e] SplittablesBase v0.1.15
⌃ [d0ee94f6] StanBase v4.8.1
⌃ [c1514b29] StanSample v7.4.2
⌅ [aedffcd0] Static v0.8.8
⌃ [0d7ed370] StaticArrayInterface v1.4.1
⌃ [90137ffa] StaticArrays v1.6.2
⌃ [1e83bf80] StaticArraysCore v1.4.2
⌃ [64bff920] StatisticalTraits v3.2.0
⌃ [82ae8749] StatsAPI v1.6.0
⌃ [2913bbd2] StatsBase v0.34.0
⌃ [4c63d2b9] StatsFuns v1.3.0
⌅ [7792a7ef] StrideArraysCore v0.4.17
⌅ [5e0ebb24] Strided v1.2.3
  [69024149] StringEncodings v0.3.7
⌅ [892a3eda] StringManipulation v0.3.0
⌅ [09ab397b] StructArrays v0.6.15
⌅ [2efcf032] SymbolicIndexingInterface v0.2.2
⌅ [d1185830] SymbolicUtils v1.2.0
⌅ [0c5d862f] Symbolics v5.5.1
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.10.1
⌃ [02d47bb6] TensorCast v0.4.6
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
⌃ [8290d209] ThreadingUtilities v0.5.2
⌃ [a759f4b9] TimerOutputs v0.5.23
⌃ [0796e94c] Tokenize v0.5.25
⌃ [9f7883ad] Tracker v0.2.26
⌅ [3bb67fe8] TranscodingStreams v0.9.13
⌃ [28d57a85] Transducers v0.4.78
⌃ [84d833dd] TransformVariables v0.8.7
⌃ [f9bc47f6] TransformedLogDensities v1.0.3
⌃ [24ddb15e] TransmuteDims v0.1.15
  [a2a6695c] TreeViews v0.3.0
⌅ [d5829a12] TriangularSolve v0.1.19
⌃ [410a4b4d] Tricks v0.1.7
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [9d95972d] TupleTools v1.3.0
⌅ [fce5fe82] Turing v0.28.3
⌃ [5c2747f8] URIs v1.5.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌃ [1986cc42] Unitful v1.17.0
⌃ [45397f5d] UnitfulLatexify v1.6.3
⌃ [a7c27f48] Unityper v0.1.5
⌅ [013be700] UnsafeAtomics v0.2.1
⌅ [d80eeb9a] UnsafeAtomicsLLVM v0.1.3
  [41fe7b60] Unzip v0.2.0
⌃ [3d5dd08c] VectorizationBase v0.21.64
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [ea10d353] WeakRefStrings v1.4.2
  [44d3d7a6] Weave v0.10.12
⌅ [efce3f68] WoodburyMatrices v0.5.5
  [76eceee3] WorkerUtilities v1.6.1
⌃ [ddb6d928] YAML v0.4.9
⌃ [c2297ded] ZMQ v1.2.2
⌃ [700de1a5] ZygoteRules v0.2.3
⌃ [6e34b625] Bzip2_jll v1.0.8+0
⌃ [83423d85] Cairo_jll v1.16.1+1
⌃ [2e619515] Expat_jll v2.5.0+0
⌅ [b22a6f82] FFMPEG_jll v4.4.2+2
⌃ [f5851436] FFTW_jll v3.3.10+0
⌃ [a3f928ae] Fontconfig_jll v2.13.93+0
⌃ [d7e528f0] FreeType2_jll v2.13.1+0
⌃ [559328eb] FriBidi_jll v1.0.10+0
⌃ [0656b61e] GLFW_jll v3.3.8+0
⌅ [d2c73de3] GR_jll v0.72.9+1
  [78b55507] Gettext_jll v0.21.0+0
⌃ [f8c6e375] Git_jll v2.36.1+2
⌃ [7746bdde] Glib_jll v2.74.0+2
⌃ [3b182d85] Graphite2_jll v1.3.14+0
⌅ [2e76f6c2] HarfBuzz_jll v2.8.1+1
⌅ [1d5cc7b8] IntelOpenMP_jll v2023.2.0+0
⌃ [aacddb02] JpegTurbo_jll v2.1.91+0
⌃ [c1c5ebd0] LAME_jll v3.100.1+0
⌅ [88015f11] LERC_jll v3.0.0+1
⌅ [dad2f222] LLVMExtra_jll v0.0.23+0
⌃ [1d63c593] LLVMOpenMP_jll v15.0.4+0
⌃ [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
⌃ [d4300ac3] Libgcrypt_jll v1.8.7+0
⌃ [7e76a0d4] Libglvnd_jll v1.6.0+0
⌃ [7add5ba3] Libgpg_error_jll v1.42.0+0
⌃ [94ce4f54] Libiconv_jll v1.16.1+2
⌃ [4b2f31a3] Libmount_jll v2.35.0+0
⌅ [89763e89] Libtiff_jll v4.5.1+1
⌃ [38a345b3] Libuuid_jll v2.36.0+0
⌅ [856f044c] MKL_jll v2023.2.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [458c3c95] OpenSSL_jll v1.1.22+0
⌃ [efe28fd5] OpenSpecFun_jll v0.5.5+0
⌃ [91d4177d] Opus_jll v1.3.2+0
⌃ [30392449] Pixman_jll v0.42.2+0
⌅ [c0090381] Qt6Base_jll v6.4.2+3
⌅ [f50d1b31] Rmath_jll v0.4.0+0
⌃ [a2964d1f] Wayland_jll v1.21.0+0
⌃ [2381bf8a] Wayland_protocols_jll v1.25.0+0
⌅ [02c8fc9c] XML2_jll v2.10.3+0
⌃ [aed1982a] XSLT_jll v1.1.34+0
⌃ [ffd25f8a] XZ_jll v5.4.4+0
⌃ [4f6342f7] Xorg_libX11_jll v1.8.6+0
⌃ [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
⌃ [935fb764] Xorg_libXcursor_jll v1.2.0+4
⌃ [a3789734] Xorg_libXdmcp_jll v1.1.4+0
⌃ [1082639a] Xorg_libXext_jll v1.3.4+4
⌃ [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
⌃ [a51aa0fd] Xorg_libXi_jll v1.7.10+4
⌃ [d1454406] Xorg_libXinerama_jll v1.1.4+4
⌃ [ec84b674] Xorg_libXrandr_jll v1.5.2+4
⌃ [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
⌃ [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
⌃ [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
⌃ [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
⌃ [35661453] Xorg_xkbcomp_jll v1.4.6+0
⌃ [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
⌃ [c5fb5394] Xorg_xtrans_jll v1.5.0+0
⌃ [8f1865be] ZeroMQ_jll v4.3.4+0
⌃ [3161d3a3] Zstd_jll v1.5.5+0
⌅ [214eeab7] fzf_jll v0.29.0+0
⌃ [a4ae2306] libaom_jll v3.4.0+0
⌃ [0ac62f75] libass_jll v0.15.1+0
⌃ [f638f0a6] libfdk_aac_jll v2.0.2+0
⌃ [b53b4c65] libpng_jll v1.6.38+0
⌃ [a9144af2] libsodium_jll v1.0.20+0
⌃ [f27f6e37] libvorbis_jll v1.3.7+1
⌅ [1270edf5] x264_jll v2021.5.5+0
⌅ [dfaa095f] x265_jll v3.5.0+0
⌃ [d8fb68d0] xkbcommon_jll v1.4.1+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.9.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.2+0
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.2+0
  [14a3606d] MozillaCACerts_jll v2022.10.11
  [4536629a] OpenBLAS_jll v0.3.21+4
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.42.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+6
  [83775a58] Zlib_jll v1.2.13+0
  [8e850b90] libblastrampoline_jll v5.8.0+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

