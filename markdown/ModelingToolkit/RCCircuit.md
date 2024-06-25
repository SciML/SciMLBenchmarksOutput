---
author: "Avinash Subramanian, Yingbo Ma, Chris Elrod"
title: "RC Circuit"
---


When a model is defined using repeated components, JuliaSimCompiler is able to take advantage of this
to scale efficiently by rerolling equations into loops. This option can be disabled by setting `loop=false`.
Here, we build an RC circuit model with variable numbers of components to show scaling of compile and
runtimes of MTK vs JuliaSimCompiler's three backends with and without loop rerolling.

## Setup Model Code

```julia
using JuliaSimCompiler, ModelingToolkit, OrdinaryDiffEq, BenchmarkTools, ModelingToolkitStandardLibrary, OMJulia, CairoMakie
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkitStandardLibrary.Electrical
```


```julia
# ModelingToolkit and JuliaSimCompiler
const t = Blocks.t

function build_system(n)
  systems = @named begin
    sine = Sine(frequency = 10)
    source = Voltage()
    resistors[1:n] = Resistor()
    capacitors[1:n] = Capacitor()
    ground = Ground()
  end
  systems = reduce(vcat, systems)
  eqs = [connect(sine.output, source.V)
       connect(source.p, resistors[1].p)
       [connect(resistors[i].n, resistors[i + 1].p, capacitors[i].p)
        for i in 1:(n - 1)]
       connect(resistors[end].n, capacitors[end].p)
       [connect(capacitors[i].n, source.n) for i in 1:n]
       connect(source.n, ground.g)]
  @named sys = ODESystem(eqs, t; systems)
  u0 = [capacitors[i].v => float(i) for i in 1:n];
  ps = [[resistors[i].R => 1 / i for i in 1:n];
        [capacitors[i].C => 1 / i^2 for i in 1:n]]
  return sys, u0, ps
end

function compile_run_problem(sys, u0, ps; target=JuliaSimCompiler.JuliaTarget(), duref=nothing)
  tspan = (0.0, 10.0)
  t0 = time()
  prob = if target === JuliaSimCompiler.JuliaTarget()
    ODEProblem(sys, u0, tspan, ps; sparse = true)
  else
    ODEProblem(sys, target, u0, tspan, ps; sparse = true)
  end
  (; f, u0, p) = prob
  ff = f.f
  du = similar(u0)
  ff(du, u0, p, 0.0)
  t_fode = time() - t0
  duref === nothing || @assert duref ≈ du
  t_run = @belapsed $ff($du, $u0, $p, 0.0)
  t_solve = @elapsed sol = solve(prob, Rodas5(autodiff = false))
  @assert SciMLBase.successful_retcode(sol)
  (t_fode, t_run, t_solve), du
end

const C = JuliaSimCompiler.CTarget();
const LLVM = JuliaSimCompiler.llvm.LLVMTarget();

function run_and_time_julia!(ss_times, times, max_sizes, i, n)
  sys, u0, ps = build_system(n);
  if n <= max_sizes[1]
    ss_times[i, 1] = @elapsed sys_mtk = structural_simplify(sys)
    times[i, 1], _ = compile_run_problem(sys_mtk, u0, ps)
  end
  ss_times[i, 2] = @elapsed sys_jsir_scalar = structural_simplify(IRSystem(sys), loop=false)
  ss_times[i, 3] = @elapsed sys_jsir_loop = structural_simplify(JuliaSimCompiler.compressed_connection_expansion(sys))
  oderef = daeref = nothing
  n <= max_sizes[2] && ((times[i, 2], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; duref = oderef))
  n <= max_sizes[3] && ((times[i, 3], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; target=C, duref = oderef))
  n <= max_sizes[4] && ((times[i, 4], oderef) = compile_run_problem(sys_jsir_scalar, u0, ps; target=LLVM, duref = oderef))
  n <= max_sizes[5] && ((times[i, 5], deeref) = compile_run_problem(sys_jsir_loop, u0, ps; duref = daeref))
  n <= max_sizes[6] && ((times[i, 6], daeref) = compile_run_problem(sys_jsir_loop, u0, ps; target=C, duref = daeref))
  n <= max_sizes[7] && ((times[i, 7], daeref) = compile_run_problem(sys_jsir_loop, u0, ps; target=LLVM, duref = daeref))
  for j = 1:7
    ss_time = j == 1 ? ss_times[i,1] : ss_times[i, 2 + (j >= 5)]
    t_fode, t_run, t_solve = times[i,j]
    total_times[i, j] = ss_time + t_fode + t_solve
  end
end
```

```
run_and_time_julia! (generic function with 1 method)
```



```julia
N = [5, 10, 20, 40, 60, 80, 160, 320, 480, 640, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000];

# max size we test per method
max_sizes = [4_000, 8_000, 20_000, 20_000, 20_000, 20_000, 20_000, 9000, 20_000];

# NaN-initialize so Makie will ignore incomplete
ss_times = fill(NaN, length(N), 3);
times = fill((NaN,NaN,NaN), length(N), length(max_sizes) - 1);
total_times = fill(NaN, length(N), length(max_sizes));
```




## Julia Timings

```julia
@time run_and_time_julia!(ss_times, times, max_sizes, 1, 4); # precompile
for (i, n) in enumerate(N)
  @time run_and_time_julia!(ss_times, times, max_sizes, i, n)
end
```

```
97.014383 seconds (77.58 M allocations: 5.156 GiB, 29.12% gc time, 72.07% 
compilation time: 17% of which was recompilation)
 34.401686 seconds (8.05 M allocations: 573.827 MiB, 75.97% gc time, 20.18%
 compilation time)
 36.277297 seconds (8.95 M allocations: 638.857 MiB, 72.44% gc time, 21.87%
 compilation time)
 39.614930 seconds (10.22 M allocations: 739.024 MiB, 66.70% gc time, 24.00
% compilation time)
 43.904312 seconds (11.89 M allocations: 878.470 MiB, 60.60% gc time, 25.14
% compilation time)
 43.000246 seconds (13.95 M allocations: 1.022 GiB, 62.55% gc time, 19.83% 
compilation time)
 44.082612 seconds (16.41 M allocations: 1.216 GiB, 61.33% gc time, 20.51% 
compilation time)
 52.079956 seconds (27.85 M allocations: 2.110 GiB, 54.07% gc time, 21.91% 
compilation time)
 64.945852 seconds (58.94 M allocations: 4.868 GiB, 45.93% gc time, 29.93% 
compilation time)
 86.844785 seconds (102.80 M allocations: 9.292 GiB, 36.62% gc time, 34.98%
 compilation time)
115.458789 seconds (157.19 M allocations: 15.094 GiB, 28.27% gc time, 38.76
% compilation time)
148.348772 seconds (227.45 M allocations: 24.855 GiB, 24.17% gc time, 40.25
% compilation time)
197.740471 seconds (337.69 M allocations: 44.832 GiB, 19.63% gc time, 42.68
% compilation time)
567.323611 seconds (1.21 G allocations: 231.227 GiB, 11.29% gc time, 41.60%
 compilation time)
1229.020464 seconds (2.68 G allocations: 582.993 GiB, 10.80% gc time, 41.21
% compilation time)
2188.262917 seconds (4.79 G allocations: 1.370 TiB, 9.29% gc time, 39.47% c
ompilation time)
475.959889 seconds (306.34 M allocations: 51.822 GiB, 10.62% gc time, 36.29
% compilation time)
660.967122 seconds (388.36 M allocations: 69.539 GiB, 8.63% gc time, 37.50%
 compilation time)
864.599253 seconds (480.66 M allocations: 90.071 GiB, 7.70% gc time, 37.81%
 compilation time)
1118.681578 seconds (584.19 M allocations: 112.767 GiB, 6.58% gc time, 38.4
8% compilation time)
800.754684 seconds (670.40 M allocations: 131.841 GiB, 8.50% gc time, 0.62%
 compilation time)
1030.620868 seconds (784.96 M allocations: 157.064 GiB, 7.52% gc time, 0.48
% compilation time)
4585.040831 seconds (2.38 G allocations: 550.116 GiB, 9.32% gc time, 0.11% 
compilation time)
```





## OpenModelica Timings

```julia
# OMJ
const omod = OMJulia.OMCSession();
OMJulia.sendExpression(omod, "getVersion()")
OMJulia.sendExpression(omod, "installPackage(Modelica)")
const modelicafile = joinpath(@__DIR__, "RC_Circuit.mo")

function time_open_modelica(n::Int)
  totaltime = @elapsed res = begin
    @sync ModelicaSystem(omod, modelicafile, "RC_Circuit.Test.RC_Circuit_MTK_test_$n")
    sendExpression(omod, "simulate(RC_Circuit.Test.RC_Circuit_MTK_test_$n)")
  end
  @assert res["messages"][1:11] == "LOG_SUCCESS"
  return totaltime
end

function run_and_time_om!(ss_times, times, max_sizes, i, n)
  run_and_time_julia!(ss_times, times, max_sizes, i, n)
  if n <= max_sizes[8]
    total_times[i, end] = time_open_modelica(n)
  end
  @views println("n = $(n)\nstructural_simplify_times = $(ss_times[i,:])\ncomponent times = $(times[i, :])\ntotal times = $(total_times[i, :])")
end

for (i, n) in enumerate(N)
  @time run_and_time_om!(ss_times, times, max_sizes, i, n)
end

OMJulia.quit(omod)
```

```
n = 5
structural_simplify_times = [0.032106053, 0.02316419, 0.020802171]
component times = [(0.005957126617431641, 1.494889779559118e-8, 0.004071726
), (0.0009021759033203125, 1.4958917835671342e-8, 0.003735377), (0.08677387
237548828, 9.57957957957958e-9, 0.003680069), (0.008772134780883789, 9.7587
58758758759e-9, 0.003637489), (0.0014009475708007812, 1.5269539078156315e-8
, 0.008941123), (0.09356808662414551, 9.218218218218218e-9, 0.008224079), (
0.011886119842529297, 8.108108108108109e-9, 0.00818961), (NaN, NaN, NaN)]
total times = [0.04213490561743164, 0.02780174290332031, 0.1136181313754882
8, 0.03557381378088379, 0.031144241570800785, 0.12259433662414551, 0.040877
9008425293, NaN, 2.938572435]
 42.301241 seconds (2.12 M allocations: 177.583 MiB, 88.09% gc time, 2.25% 
compilation time)
n = 10
structural_simplify_times = [0.058598256, 0.039296641, 0.029474007]
component times = [(0.009829998016357422, 2.9878391959799e-8, 0.005291735),
 (0.001394033432006836, 2.9878391959799e-8, 0.004888508), (0.08634710311889
648, 1.494889779559118e-8, 0.004326683), (0.011254072189331055, 1.294488977
9559117e-8, 0.004190974), (0.001519918441772461, 3.439979859013091e-8, 0.01
2650671), (0.1017599105834961, 1.198098098098098e-8, 0.011109724), (0.01497
0064163208008, 1.4357715430861724e-8, 0.010867827), (NaN, NaN, NaN)]
total times = [0.07371998901635743, 0.045579182432006836, 0.129970427118896
48, 0.054741687189331055, 0.04364459644177246, 0.14234364158349608, 0.05531
1898163208, NaN, 2.30106808]
 41.275427 seconds (1.72 M allocations: 155.112 MiB, 88.90% gc time, 0.41% 
compilation time)
n = 20
structural_simplify_times = [0.139157533, 0.071258717, 0.047066915]
component times = [(0.017901897430419922, 5.972403258655804e-8, 0.00704488)
, (0.002368927001953125, 7.226002055498458e-8, 0.006216676), (0.10202503204
345703, 2.3904618473895583e-8, 0.005118195), (0.013647079467773438, 1.78446
89378757514e-8, 0.005134526), (0.0018029212951660156, 5.918431771894094e-8,
 0.019573462), (0.1026461124420166, 2.674949698189135e-8, 0.016457858), (0.
01654815673828125, 2.5407035175879395e-8, 0.016528298), (NaN, NaN, NaN)]
total times = [0.16410431043041993, 0.07984432000195313, 0.1784019440434570
2, 0.09004032246777344, 0.06844329829516602, 0.16617088544201658, 0.0801433
6973828125, NaN, 2.494170567]
 43.045011 seconds (2.41 M allocations: 219.960 MiB, 85.18% gc time, 0.39% 
compilation time)
n = 40
structural_simplify_times = [0.27117317, 0.138322412, 0.081774977]
component times = [(0.03338003158569336, 1.1943784378437845e-7, 0.013040418
), (0.004479169845581055, 1.4929575757575755e-7, 0.007358327), (0.116672992
70629883, 3.8829637096774194e-8, 0.00694547), (0.019498109817504883, 3.3323
262839879156e-8, 0.007188449), (0.002513885498046875, 1.0900967741935484e-7
, 0.042825473), (0.09730005264282227, 4.15438950554995e-8, 0.02793929), (0.
01988697052001953, 4.614372469635627e-8, 0.028455166), (NaN, NaN, NaN)]
total times = [0.3175936195856934, 0.15015990884558106, 0.26194087470629884
, 0.1650089708175049, 0.12711433549804688, 0.20701431964282227, 0.130117113
52001953, NaN, 3.836733096]
 46.977755 seconds (3.92 M allocations: 355.702 MiB, 78.06% gc time, 0.35% 
compilation time)
n = 60
structural_simplify_times = [0.424010089, 0.205862333, 0.11648424]
component times = [(0.04914212226867676, 1.8214306569343066e-7, 0.018553721
), (0.006654024124145508, 2.292945054945055e-7, 0.010154493), (0.1357309818
2678223, 5.375532994923858e-8, 0.008909963), (0.02703714370727539, 4.659150
657229524e-8, 0.009011153), (0.0030930042266845703, 1.5443494423791822e-7, 
0.049500825), (0.10268592834472656, 6.123241590214067e-8, 0.05714181), (0.0
28352975845336914, 6.452400408580183e-8, 0.040053206), (NaN, NaN, NaN)]
total times = [0.49170593226867676, 0.2226708501241455, 0.35050327782678226
, 0.2419106297072754, 0.16907806922668456, 0.2763119783447266, 0.1848904218
4533694, NaN, 4.845633074]
 49.504296 seconds (5.60 M allocations: 508.329 MiB, 74.44% gc time, 0.34% 
compilation time)
n = 80
structural_simplify_times = [0.622953365, 0.279091195, 0.159193985]
component times = [(0.06895995140075684, 2.465518134715026e-7, 0.034713153)
, (0.009284019470214844, 3.069595141700405e-7, 0.014641004), (0.15483188629
15039, 6.867725409836065e-8, 0.01169365), (0.034219980239868164, 6.15178389
3985729e-8, 0.011558241), (0.0038299560546875, 2.028017094017094e-7, 0.0610
93027), (0.10114598274230957, 8.155175983436852e-8, 0.051851065), (0.020927
19078063965, 8.482726326742975e-8, 0.051403079), (NaN, NaN, NaN)]
total times = [0.7266264694007568, 0.30301621847021487, 0.4456167312915039,
 0.3248694162398682, 0.2241169680546875, 0.31219103274230964, 0.23152425478
063965, NaN, 5.985800178]
 51.441492 seconds (7.48 M allocations: 678.289 MiB, 72.13% gc time, 0.32% 
compilation time)
n = 160
structural_simplify_times = [1.645558482, 0.579215695, 0.353505559]
component times = [(0.1386260986328125, 5.810441988950277e-7, 0.087558749),
 (0.021032094955444336, 6.172196531791908e-7, 0.021458376), (0.285426139831
54297, 1.2837725225225223e-7, 0.020367446), (0.06587100028991699, 1.2093812
154696132e-7, 0.020361646), (0.0065572261810302734, 4.06445e-7, 0.168117309
), (0.12074995040893555, 1.4473063170441e-7, 0.153689433), (0.0347099304199
21875, 1.5709056122448979e-7, 0.152340645), (NaN, NaN, NaN)]
total times = [1.8717433296328125, 0.6217061659554444, 0.885009280831543, 0
.665448341289917, 0.5281800941810303, 0.6279449424089356, 0.540556134419921
9, NaN, 10.350352173]
 60.429362 seconds (16.56 M allocations: 1.447 GiB, 61.51% gc time, 0.27% c
ompilation time)
n = 320
structural_simplify_times = [5.262541128, 1.281626532, 0.677313788]
component times = [(0.2887840270996094, 1.2649000000000002e-6, 0.237242798)
, (0.056318044662475586, 1.2429e-6, 0.039019125), (0.4879570007324219, 3.70
2864077669903e-7, 0.032762919), (0.13637280464172363, 2.404214463840399e-7,
 0.032221664), (0.01129913330078125, 8.054835164835165e-7, 0.326193017), (0
.12922406196594238, 2.7877700348432057e-7, 0.309324821), (0.034296035766601
56, 2.942528735632184e-7, 0.30709359), (NaN, NaN, NaN)]
total times = [5.788567953099609, 1.3769637016624756, 1.802346451732422, 1.
4502210006417235, 1.0148059383007813, 1.1158626709659423, 1.018703413766601
6, NaN, 20.425667121]
 74.475460 seconds (43.15 M allocations: 3.993 GiB, 50.75% gc time, 0.22% c
ompilation time)
n = 480
structural_simplify_times = [11.029625193, 2.098151194, 1.03058929]
component times = [(0.44681596755981445, 2.033333333333333e-6, 0.589250007)
, (0.11295700073242188, 1.872e-6, 0.076226788), (0.7363312244415283, 6.9127
7027027027e-7, 0.064666596), (0.2132251262664795, 3.5985238095238096e-7, 0.
063657286), (0.01693105697631836, 1.208e-6, 0.470788742), (0.14366507530212
402, 4.2205527638190955e-7, 0.442811462), (0.040576934814453125, 4.36358585
85858587e-7, 0.44767569), (NaN, NaN, NaN)]
total times = [12.065691167559814, 2.287334982732422, 2.8991490144415284, 2
.37503360626648, 1.5183090889763182, 1.617065827302124, 1.5188419148144532,
 NaN, 31.842700328]
 95.729609 seconds (82.21 M allocations: 8.192 GiB, 40.35% gc time, 0.17% c
ompilation time)
n = 640
structural_simplify_times = [19.371871772, 2.958918661, 1.542945338]
component times = [(0.5792241096496582, 2.6611111111111112e-6, 0.964530534)
, (0.18884515762329102, 2.4943333333333334e-6, 0.074370964), (1.02197122573
85254, 1.0989000000000002e-6, 0.065862407), (0.2990989685058594, 4.79276923
0769231e-7, 0.064867856), (0.022360801696777344, 1.6269e-6, 1.063680478), (
0.1675271987915039, 6.147371428571429e-7, 1.029996636), (0.0458569526672363
3, 6.045706214689266e-7, 1.020856515), (NaN, NaN, NaN)]
total times = [20.91562641564966, 3.2221347826232907, 4.046752293738525, 3.
322885485505859, 2.6289866176967776, 2.740469172791504, 2.6096588056672365,
 NaN, 42.449332088]
120.533204 seconds (133.22 M allocations: 13.831 GiB, 33.00% gc time, 0.14%
 compilation time)
n = 800
structural_simplify_times = [30.404667644, 3.981232973, 1.979362217]
component times = [(0.7390320301055908, 3.5425e-6, 1.488098623), (0.2711768
15032959, 3.13875e-6, 0.079097352), (1.3907959461212158, 1.4899000000000001
e-6, 0.073038132), (0.3784470558166504, 5.987022471910112e-7, 0.082208166),
 (0.02747511863708496, 2.0643333333333333e-6, 1.003498881), (0.182975053787
23145, 8.11561797752809e-7, 0.983018187), (0.05050992965698242, 8.537727272
727273e-7, 1.040513483), (NaN, NaN, NaN)]
total times = [32.63179829710559, 4.331507140032959, 5.445067051121216, 4.4
4188819481665, 3.0103362166370853, 3.1453554577872316, 3.0703856296569825, 
NaN, 51.759442736]
145.492311 seconds (198.93 M allocations: 23.396 GiB, 28.60% gc time, 0.12%
 compilation time)
n = 1000
structural_simplify_times = [49.318859741, 5.295833579, 2.556631444]
component times = [(1.0388479232788086, 4.744142857142857e-6, 2.160400181),
 (0.41202521324157715, 4.027142857142857e-6, 0.128941558), (1.9124100208282
47, 2.034333333333333e-6, 0.110147472), (0.49768519401550293, 7.51658333333
3333e-7, 0.106849209), (0.03443479537963867, 2.572222222222222e-6, 1.309585
163), (0.2017049789428711, 9.89e-7, 1.250705647), (0.05644989013671875, 9.8
14285714285714e-7, 1.270168313), (NaN, NaN, NaN)]
total times = [52.518107845278806, 5.836800350241577, 7.318391071828247, 5.
900367982015503, 3.900651402379639, 4.009042069942871, 3.883249647136719, N
aN, 66.754372624]
185.527538 seconds (300.93 M allocations: 42.986 GiB, 23.48% gc time, 0.09%
 compilation time)
n = 2000
structural_simplify_times = [219.632867083, 15.204454583, 6.349053766]
component times = [(3.610811948776245, 9.459e-6, 7.617560099), (1.525599002
8381348, 8.729666666666665e-6, 0.228641889), (5.885847091674805, 4.61428571
4285715e-6, 0.200414433), (1.1955931186676025, 1.7639000000000002e-6, 0.180
744146), (0.07515215873718262, 5.143166666666667e-6, 3.265363315), (0.35354
018211364746, 2.006666666666667e-6, 3.220973479), (0.09638690948486328, 2.0
222222222222223e-6, 3.369066861), (NaN, NaN, NaN)]
total times = [230.86123913077626, 16.958695474838134, 21.290716107674804, 
16.5807918476676, 9.689569239737182, 9.923567427113648, 9.814507536484863, 
NaN, 161.985607774]
496.246830 seconds (1.14 G allocations: 228.186 GiB, 13.25% gc time, 0.04% 
compilation time)
n = 3000
structural_simplify_times = [526.3751029, 27.002297649, 11.175252471]
component times = [(6.44140100479126, 1.575e-5, 18.912827277), (3.365432977
6763916, 1.33e-5, 0.357039146), (12.430819988250732, 7.6675e-6, 0.317655248
), (2.1014981269836426, 2.2621111111111113e-6, 0.314297555), (0.10690999031
066895, 7.7475e-6, 6.852044897), (0.5370528697967529, 3.14875e-6, 6.8596729
65), (0.13244390487670898, 2.9925e-6, 7.345048602), (NaN, NaN, NaN)]
total times = [551.7293311817913, 30.72476977267639, 39.75077288525073, 29.
41809333098364, 18.134207358310668, 18.571978305796755, 18.652744977876708,
 NaN, 272.924188229]
975.109808 seconds (2.59 G allocations: 578.526 GiB, 9.76% gc time, 0.02% c
ompilation time)
n = 4000
structural_simplify_times = [1108.544903965, 42.971304364, 17.531591514]
component times = [(8.515460968017578, 2.1129e-5, 30.53605693), (5.65490388
8702393, 1.798e-5, 0.461959335), (23.213011026382446, 9.49e-6, 0.413080539)
, (3.2466790676116943, 3.402375e-6, 0.409538817), (0.1461038589477539, 1.03
8e-5, 17.216343878), (0.7672529220581055, 4.291428571428571e-6, 16.12285926
), (0.20978021621704102, 4.051428571428572e-6, 17.613286652), (NaN, NaN, Na
N)]
total times = [1147.5964218630177, 49.088167587702394, 66.59739592938244, 4
6.62752224861169, 34.894039250947756, 34.4217036960581, 35.35465838221704, 
NaN, 393.147793104]
1766.773599 seconds (4.66 G allocations: 1.364 TiB, 12.81% gc time, 0.01% c
ompilation time)
n = 5000
structural_simplify_times = [NaN, 60.301987035, 27.625975968]
component times = [(NaN, NaN, NaN), (8.821807146072388, 2.394e-5, 0.5207199
98), (39.19996094703674, 1.23e-5, 0.455738279), (4.667370796203613, 3.75237
5e-6, 0.441551456), (0.17795920372009277, 1.296e-5, 29.482758217), (0.83599
30515289307, 5.33e-6, 30.167920146), (0.25308799743652344, 5.06666666666666
7e-6, 29.311528035), (NaN, NaN, NaN)]
total times = [NaN, 69.64451417907239, 99.95768626103674, 65.41090928720361
, 57.286693388720096, 58.62988916552893, 57.190592000436524, NaN, 535.44155
1719]
839.839418 seconds (294.49 M allocations: 51.163 GiB, 6.80% gc time, 0.02% 
compilation time)
n = 6000
structural_simplify_times = [NaN, 82.814120598, 33.834056135]
component times = [(NaN, NaN, NaN), (12.16453218460083, 3.0629e-5, 0.613011
735), (66.13105297088623, 1.488e-5, 0.573787069), (6.520599126815796, 4.735
714285714285e-6, 0.543310806), (0.21951913833618164, 1.565e-5, 45.190678123
), (0.9844307899475098, 6.258e-6, 43.691043116), (0.2943720817565918, 6.227
8e-6, 44.751567964), (NaN, NaN, NaN)]
total times = [NaN, 95.59166451760083, 149.5189606378862, 89.87803053081579
, 79.24425339633618, 78.50953004094751, 78.87999618075659, NaN, 702.2684411
35]
1118.143909 seconds (375.39 M allocations: 68.828 GiB, 5.69% gc time, 0.01%
 compilation time)
n = 7000
structural_simplify_times = [NaN, 108.136254188, 50.775350867]
component times = [(NaN, NaN, NaN), (19.209516048431396, 3.206e-5, 0.793212
619), (109.70531296730042, 1.809e-5, 0.699992419), (9.100503206253052, 5.25
98333333333334e-6, 0.697057023), (0.24931597709655762, 1.837e-5, 56.5954246
46), (1.0557761192321777, 7.6075e-6, 57.473600824), (0.2662050724029541, 7.
195e-6, 61.035518775), (NaN, NaN, NaN)]
total times = [NaN, 128.13898285543138, 218.54155957430044, 117.93381441725
305, 107.62009149009656, 109.30472781023218, 112.07707471440295, NaN, 953.8
10578412]
1512.161202 seconds (466.66 M allocations: 89.316 GiB, 5.56% gc time, 0.01%
 compilation time)
n = 8000
structural_simplify_times = [NaN, 139.969268804, 77.367622674]
component times = [(NaN, NaN, NaN), (25.959579944610596, 3.947e-5, 0.908324
395), (165.40007996559143, 2.093e-5, 0.825176681), (12.626968145370483, 6.0
04833333333333e-6, 0.845123756), (0.38523006439208984, 2.105e-5, 64.2292980
48), (1.2717208862304688, 8.473333333333334e-6, 64.993131344), (0.342423915
8630371, 8.41e-6, 67.659863567), (NaN, NaN, NaN)]
total times = [NaN, 166.83717314361058, 306.1945254505914, 153.441360705370
47, 141.98215078639208, 143.6324749042305, 145.36991015686306, NaN, 1274.96
9517146]
1987.672828 seconds (568.98 M allocations: 111.955 GiB, 5.63% gc time, 0.01
% compilation time)
n = 9000
structural_simplify_times = [NaN, 184.053613657, 121.553944984]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (234.07095885276794, 2
.285e-5, 0.85528937), (16.591476917266846, 6.7678e-6, 1.006506213), (0.3618
8507080078125, 2.36e-5, 72.352178968), (1.3656179904937744, 9.68e-6, 80.931
131662), (0.36010193824768066, 9.59e-6, 76.563143005), (NaN, NaN, NaN)]
total times = [NaN, NaN, 418.97986187976795, 201.65159678726684, 194.268009
02280078, 203.85069463649376, 198.47718992724768, NaN, 1582.974844839]
2452.645719 seconds (669.07 M allocations: 131.743 GiB, 6.20% gc time, 0.01
% compilation time)
n = 10000
structural_simplify_times = [NaN, 255.925064077, 164.737787866]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (307.4785759449005, 2.
5799e-5, 1.448975481), (22.682386875152588, 7.82725e-6, 1.370073096), (2.08
0104112625122, 2.628e-5, 78.793033824), (1.5097839832305908, 1.089e-5, 86.3
32137831), (0.40529704093933105, 1.083e-5, 93.763963229), (NaN, NaN, NaN)]
total times = [NaN, NaN, 564.8526155029006, 279.9775240481526, 245.61092580
26251, 252.57970968023056, 258.9070481359393, NaN, NaN]
1101.657126 seconds (780.66 M allocations: 156.771 GiB, 19.47% gc time, 0.0
1% compilation time)
n = 20000
structural_simplify_times = [NaN, 1525.693016063, 950.807505351]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (1609.1566741466522, 5
.428e-5, 9.081319966), (313.1420331001282, 2.591e-5, 5.699913447), (2.88182
80696868896, 5.343e-5, 243.918930482), (6.112215042114258, 2.255e-5, 276.20
1790656), (3.0803449153900146, 2.242e-5, 255.007454319), (NaN, NaN, NaN)]
total times = [NaN, NaN, 3143.9310101756523, 1844.534962610128, 1197.608263
902687, 1233.1215110491144, 1208.89530458539, NaN, NaN]
5304.528601 seconds (2.38 G allocations: 549.829 GiB, 17.84% gc time, 0.00%
 compilation time)
```





## Dymola Timings

Dymola requires a license server and thus cannot be hosted. This was run locally for the
following times:

```julia
translation_and_total_times = [
  5 2.428 2.458
  10 2.727 2.757
  20 1.764 1.797
  40 1.849 1.885
  60 1.953 1.995
  80 2.041 2.089
  160 2.422 2.485
  320 3.157 3.258
  480 3.943 4.092
  640 4.718 4.912
  800 5.531 5.773
  1000 6.526 6.826
  2000 11.467 12.056
  3000 16.8 17.831
  4000 22.355 24.043
  5000 27.768 30.083
  6000 33.561 36.758
  7000 39.197 43.154
  8000 45.194 52.153
  9000 50.689 57.187
  10000 NaN NaN
  20000 NaN NaN
]

total_times[:, 9] = translation_and_total_times[:,3]
```

```
22-element Vector{Float64}:
   2.458
   2.757
   1.797
   1.885
   1.995
   2.089
   2.485
   3.258
   4.092
   4.912
   ⋮
  17.831
  24.043
  30.083
  36.758
  43.154
  52.153
  57.187
 NaN
 NaN
```





## Results

```julia
f = Figure(size=(800,1200));
ss_names = ["MTK", "JSIR-Scalar", "JSIR-Loop"];
let ax = Axis(f[1, 1]; yscale = log10, xscale = log10, title="Structural Simplify Time")
  _lines = map(eachcol(ss_times)) do ts
    lines!(N, ts)
  end
  Legend(f[1,2], _lines, ss_names)
end
method_names = ["MTK", "JSIR - Scalar - Julia", "JSIR - Scalar - C", "JSIR - Scalar - LLVM", "JSIR - Loop - Julia", "JSIR - Loop - C", "JSIR - Loop - LLVM"];
for (i, timecat) in enumerate(("ODEProblem + f!", "Run", "Solve"))
  title = timecat * " Time"
  ax = Axis(f[i+1, 1]; yscale = log10, xscale = log10, title)
  _lines = map(eachcol(times)) do ts
    lines!(N, getindex.(ts, i))
  end
  Legend(f[i+1, 2], _lines, method_names)
end
let method_names_m = vcat(method_names, "OpenModelica");
  ax = Axis(f[5, 1]; yscale = log10, xscale = log10, title = "Total Time")
  _lines = map(Base.Fix1(lines!, N), eachcol(total_times))
  Legend(f[5, 2], _lines, method_names_m)
end
f
```

```
Error: Number of elements not equal: 8 content elements and 7 labels.
```



```julia
f2 = Figure(size = (800, 400));
title = "Total Time: RC Circuit Benchmark"
ax = Axis(f2[1, 1]; yscale = log10, xscale = log10, title)
names = ["MTK", "JSIR - Scalar - Julia", "JSIR - Scalar - C", "JSIR - Scalar - LLVM", "JSIR - Loop - Julia", "JSIR - Loop - C", "JSIR - Loop - LLVM", "OpenModelica", "Dymola"]
_lines = map(enumerate(names)) do (j, label)
    ts = @view(total_times[:, j])
    lines!(N, ts)
end
Legend(f2[1,2], _lines, names)
f2
```

![](figures/RCCircuit_8_1.png)




All three backends compiled more quickly with loops, but the C and LLVM backends are so much quicker to compile than the Julia backend that this made much less difference for them.
The impact on runtime was more varied.

## Appendix


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/ModelingToolkit","RCCircuit.jmd")
```

Computer Information:

```
Julia Version 1.10.4
Commit 48d4fd48430 (2024-06-04 10:41 UTC)
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
  JULIA_PKG_SERVER = juliahub.com

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Project.toml`
  [6e4b80f9] BenchmarkTools v1.5.0
  [336ed68f] CSV v0.10.14
⌅ [13f3f980] CairoMakie v0.11.11
  [a93c6f00] DataFrames v1.6.1
  [8391cb6b] JuliaSimCompiler v0.1.12
  [9cbdfd5a] JuliaSimCompilerRuntime v1.0.2
  [7ed4a6bd] LinearSolve v2.30.1
  [961ee093] ModelingToolkit v9.19.0
  [16a59e39] ModelingToolkitStandardLibrary v2.7.2
  [0f4fe800] OMJulia v0.3.2
  [1dea7af3] OrdinaryDiffEq v6.84.0
  [f27b6e38] Polynomials v4.0.11
  [ba661fbb] PreferenceTools v0.1.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [0c5d862f] Symbolics v5.30.4
  [de0858da] Printf
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Manifest.toml`
  [47edcb42] ADTypes v1.5.0
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.36
  [79e6a3ab] Adapt v4.0.4
  [66dad0bd] AliasTables v1.1.3
  [27a7e980] Animations v0.4.1
  [ec485272] ArnoldiMethod v0.4.0
  [4fba245c] ArrayInterface v7.11.0
  [4c555306] ArrayLayouts v1.9.4
  [67c07d97] Automa v1.0.3
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.7
  [6e4b80f9] BenchmarkTools v1.5.0
  [e2ed5e7c] Bijections v0.1.6
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.5
  [00ebfdb7] CSTParser v3.4.3
  [336ed68f] CSV v0.10.14
  [159f3aea] Cairo v1.0.5
⌅ [13f3f980] CairoMakie v0.11.11
  [49dc2e85] Calculus v0.5.1
  [082447d4] ChainRules v1.69.0
  [d360d2e6] ChainRulesCore v1.24.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [944b1d66] CodecZlib v0.7.4
  [a2cac450] ColorBrewer v0.4.0
  [35d6a980] ColorSchemes v3.25.0
  [3da002f7] ColorTypes v0.11.5
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.11
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.15.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.5
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.6.1
  [864edb3b] DataStructures v0.18.20
  [e2d170a0] DataValueInterfaces v1.0.0
  [927a84f5] DelaunayTriangulation v1.0.3
  [2b5f629d] DiffEqBase v6.151.4
  [459566f4] DiffEqCallbacks v3.6.2
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [a0c0ee7d] DifferentiationInterface v0.5.5
  [31c24e10] Distributions v0.25.109
  [ffbed154] DocStringExtensions v0.9.3
  [5b8099bc] DomainSets v0.7.14
  [fa6b7ba4] DualNumbers v0.6.8
⌅ [7c1d4256] DynamicPolynomials v0.5.7
⌅ [06fc5a27] DynamicQuantities v0.13.2
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.7.6
  [429591f6] ExactPredicates v2.2.8
  [d4d017d3] ExponentialUtilities v1.26.1
  [e2ba6199] ExprTools v0.1.10
⌅ [6b7a57c9] Expronicon v0.8.5
  [411431e0] Extents v0.1.3
  [7a1cc6ca] FFTW v1.8.0
  [7034ab61] FastBroadcast v0.3.3
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.4
  [5789e2e9] FileIO v1.16.3
  [8fc22ac5] FilePaths v0.8.3
  [48062228] FilePathsBase v0.9.21
  [1a297f60] FillArrays v1.11.0
  [64ca27bc] FindFirstFunctions v1.2.0
  [6a86dc24] FiniteDiff v2.23.1
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.36
  [b38be410] FreeType v4.1.1
  [663a7486] FreeTypeAbstraction v0.10.3
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.4.11
  [46192b85] GPUArraysCore v0.1.6
  [61eb1bfa] GPUCompiler v0.26.5
  [c145ed77] GenericSchur v0.5.4
  [cf35fbd7] GeoInterface v1.3.4
  [5c1252a2] GeometryBasics v0.4.11
  [d7ba0133] Git v1.3.1
  [c27321d9] Glob v1.3.1
  [a2bd30eb] Graphics v1.1.2
  [86223c79] Graphs v1.11.1
⌅ [3955a311] GridLayoutBase v0.10.2
  [42e2da0e] Grisu v1.0.2
  [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [2803e5a7] ImageAxes v0.6.11
  [c817782e] ImageBase v0.1.7
  [a09fc81d] ImageCore v0.10.2
  [82e4d734] ImageIO v0.6.8
  [bc367c6b] ImageMetadata v0.9.9
  [9b13fd28] IndirectArrays v1.0.0
  [d25df0c9] Inflate v0.1.5
  [842dd82b] InlineStrings v1.4.1
  [a98d9a8b] Interpolations v0.15.1
  [d1acc4aa] IntervalArithmetic v0.22.14
  [8197267c] IntervalSets v0.7.10
  [3587e190] InverseFunctions v0.1.14
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [f1662d9f] Isoband v0.1.1
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
  [b835a17e] JpegTurbo v0.1.5
  [98e50ef6] JuliaFormatter v1.0.56
  [9c9cc66b] JuliaSimBase v0.1.0
  [8391cb6b] JuliaSimCompiler v0.1.12
  [9cbdfd5a] JuliaSimCompilerRuntime v1.0.2
  [ccbc3e58] JumpProcesses v9.11.1
  [ef3ab10e] KLU v0.6.0
  [5ab0869b] KernelDensity v0.6.9
  [ba0b0d4f] Krylov v0.9.6
  [929cbde3] LLVM v7.2.1
  [b964fa9f] LaTeXStrings v1.3.1
  [2ee39098] LabelledArrays v1.16.0
  [984bce1d] LambertW v0.4.6
  [23fbe1c1] Latexify v0.16.3
  [10f19ff3] LayoutPointers v0.1.15
  [5078a376] LazyArrays v2.0.5
  [8cdb02fc] LazyModules v0.3.1
  [9c8b4983] LightXML v0.9.1
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.30.1
  [2ab3a3ac] LogExpFunctions v0.3.28
  [bdcacae8] LoopVectorization v0.12.170
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.13
⌅ [ee78f7c6] Makie v0.20.10
⌅ [20f20a25] MakieCore v0.7.3
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.2
⌅ [0a4f8689] MathTeXEngine v0.5.7
  [bb5d69b7] MaybeInplace v0.1.3
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
  [961ee093] ModelingToolkit v9.19.0
  [16a59e39] ModelingToolkitStandardLibrary v2.7.2
  [e94cdb99] MosaicViews v0.3.4
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.6
  [ffc61752] Mustache v1.0.19
  [d8a4904e] MutableArithmetics v1.4.5
  [d41bc354] NLSolversBase v7.8.3
  [77ba4419] NaNMath v1.0.2
  [f09324ee] Netpbm v1.1.1
  [8913a72c] NonlinearSolve v3.13.0
  [0f4fe800] OMJulia v0.3.2
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.14.0
  [52e1d378] OpenEXR v0.3.2
  [bac558e1] OrderedCollections v1.6.3
  [1dea7af3] OrdinaryDiffEq v6.84.0
  [90014a1f] PDMats v0.11.31
  [f57f5aa1] PNGFiles v0.4.3
  [65ce6f38] PackageExtensionCompat v1.0.2
  [19eb6ba3] Packing v0.5.0
  [5432bcbf] PaddedViews v0.5.12
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [eebad327] PkgVersion v0.3.3
  [995b91a9] PlotUtils v1.4.1
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.14
  [1d0040c9] PolyesterWeave v0.2.1
  [647866c9] PolygonOps v0.1.2
  [f27b6e38] Polynomials v4.0.11
  [2dfb63ee] PooledArrays v1.4.3
  [d236fae5] PreallocationTools v0.4.22
  [aea7be01] PrecompileTools v1.2.1
  [ba661fbb] PreferenceTools v0.1.2
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.3.2
  [92933f4c] ProgressMeter v1.10.0
  [43287f4e] PtrArrays v1.2.0
  [4b34888f] QOI v1.0.0
  [1fd47b50] QuadGK v2.9.4
  [e6cf234a] RandomNumbers v1.5.3
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
⌃ [731186ca] RecursiveArrayTools v3.23.1
  [f2c3362d] RecursiveFactorization v0.2.23
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [5eaf0fd0] RoundingEmulator v0.2.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.13
  [fdea26ae] SIMD v3.5.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
  [0bca4576] SciMLBase v2.41.3
  [31c91b34] SciMLBenchmarks v0.1.3
  [c0aeaf25] SciMLOperators v0.3.8
  [53ae85a6] SciMLStructures v1.3.0
  [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.3
  [efcf1570] Setfield v1.1.1
  [65257c39] ShaderAbstractions v0.4.1
  [992d4aef] Showoff v1.0.3
  [73760f76] SignedDistanceFields v0.4.0
  [727e6d20] SimpleNonlinearSolve v1.10.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [45858cf5] Sixel v0.1.3
  [66db9d55] SnoopPrecompile v1.0.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.19.0
  [dc90abb0] SparseInverseSubset v0.1.2
  [0a514795] SparseMatrixColorings v0.3.3
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.4.0
  [cae243ae] StackViews v0.1.1
  [aedffcd0] Static v0.8.10
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.5
  [1e83bf80] StaticArraysCore v1.4.3
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.3
  [4c63d2b9] StatsFuns v1.3.1
  [7792a7ef] StrideArraysCore v0.5.6
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [09ab397b] StructArrays v0.6.18
  [2efcf032] SymbolicIndexingInterface v0.3.22
  [19f23fe9] SymbolicLimits v0.2.1
  [d1185830] SymbolicUtils v2.0.2
  [0c5d862f] Symbolics v5.30.4
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [62fd8b95] TensorCore v0.1.1
⌅ [8ea1fca8] TermInterface v0.4.1
  [8290d209] ThreadingUtilities v0.5.2
  [731e570b] TiffImages v0.10.0
  [a759f4b9] TimerOutputs v0.5.24
  [0796e94c] Tokenize v0.5.29
  [3bb67fe8] TranscodingStreams v0.10.10
  [d5829a12] TriangularSolve v0.2.0
  [410a4b4d] Tricks v0.1.8
  [981d1d27] TriplotBase v0.1.0
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.20.0
  [a7c27f48] Unityper v0.1.6
  [3d5dd08c] VectorizationBase v0.21.68
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [ea10d353] WeakRefStrings v1.4.2
  [44d3d7a6] Weave v0.10.12
  [efce3f68] WoodburyMatrices v1.0.0
  [76eceee3] WorkerUtilities v1.6.1
  [ddb6d928] YAML v0.4.11
  [c2297ded] ZMQ v1.2.6
  [6e34b625] Bzip2_jll v1.0.8+1
  [4e9b3aee] CRlibm_jll v1.0.1+0
  [83423d85] Cairo_jll v1.18.0+2
  [5ae413db] EarCut_jll v2.2.4+0
  [2e619515] Expat_jll v2.6.2+0
  [b22a6f82] FFMPEG_jll v6.1.1+0
  [f5851436] FFTW_jll v3.3.10+0
  [a3f928ae] Fontconfig_jll v2.13.96+0
  [d7e528f0] FreeType2_jll v2.13.2+0
  [559328eb] FriBidi_jll v1.0.14+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.44.0+2
  [7746bdde] Glib_jll v2.80.2+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [905a6f67] Imath_jll v3.1.11+0
  [1d5cc7b8] IntelOpenMP_jll v2024.1.0+0
  [aacddb02] JpegTurbo_jll v3.0.3+0
  [c1c5ebd0] LAME_jll v3.100.2+0
⌅ [dad2f222] LLVMExtra_jll v0.0.29+0
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [dd4b983a] LZO_jll v2.10.2+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.11+0
  [7add5ba3] Libgpg_error_jll v1.49.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.40.1+0
  [38a345b3] Libuuid_jll v2.40.1+0
  [856f044c] MKL_jll v2024.1.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [18a262bb] OpenEXR_jll v3.2.4+0
  [458c3c95] OpenSSL_jll v3.0.14+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [36c8627f] Pango_jll v1.52.2+0
  [30392449] Pixman_jll v0.43.4+0
  [f50d1b31] Rmath_jll v0.4.2+0
  [02c8fc9c] XML2_jll v2.12.7+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.11+0
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.5+0
  [9a68df92] isoband_jll v0.2.3+0
  [a4ae2306] libaom_jll v3.9.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.43+1
  [075b6546] libsixel_jll v1.10.3+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1317d2d5] oneTBB_jll v2021.12.0+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [8bf52ea8] CRC32c
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.10.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.1+2
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.8.0+1
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

