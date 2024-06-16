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
  t_solve = @elapsed solve(prob, Rodas5(autodiff = false))
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
max_sizes = [4_000, 8_000, 20_000, 20_000, 20_000, 20_000, 20_000, 9000];

# NaN-initialize so Makie will ignore incomplete
ss_times = fill(NaN, length(N), 3);
times = fill((NaN,NaN,NaN), length(N), length(max_sizes) - 1);
total_times = fill(NaN, length(N), length(max_sizes));
```




## Julia Timings

```julia
@time run_and_time_julia!(ss_times, times, max_sizes, 1, 4); # precompile
```

```
96.020502 seconds (77.81 M allocations: 5.171 GiB, 30.01% gc time, 69.65% 
compilation time: 17% of which was recompilation)
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
    total_times[i, 8] = time_open_modelica(n)
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
structural_simplify_times = [0.055962216, 0.02311075, 0.041149984]
component times = [(0.13594698905944824, 1.494889779559118e-8, 0.286749747)
, (0.07150983810424805, 1.494889779559118e-8, 2.492945955), (0.101182937622
07031, 9.70970970970971e-9, 0.003752588), (0.007174968719482422, 9.87887887
887888e-9, 0.003747647), (0.15171098709106445, 1.5270541082164328e-8, 3.476
601664), (0.0997309684753418, 9.149149149149149e-9, 0.009821115), (0.011605
024337768555, 8.098098098098099e-9, 0.008205019)]
total times = [0.47865895205944825, 2.587566543104248, 0.1280462756220703, 
0.03403336571948242, 3.6694626350910644, 0.1507020674753418, 0.060960027337
768556, 2.977973614]
 39.730579 seconds (8.78 M allocations: 624.391 MiB, 70.60% gc time, 18.94%
 compilation time)
n = 10
structural_simplify_times = [0.217498868, 0.038834604, 0.049849268]
component times = [(0.17409300804138184, 2.9878391959799e-8, 0.332379503), 
(0.08066916465759277, 2.9878391959799e-8, 2.483733586), (0.0996239185333252
, 1.494889779559118e-8, 0.003991365), (0.008913040161132812, 1.317635270541
0822e-8, 0.003897086), (0.15877199172973633, 3.3795568982880166e-8, 4.24615
9622), (0.10629701614379883, 1.5541082164328656e-8, 0.010520989), (0.012482
166290283203, 1.199099099099099e-8, 0.01043231)]
total times = [0.7239713790413819, 2.6032373546575927, 0.1424498875333252, 
0.051644730161132815, 4.454780881729737, 0.16666727314379884, 0.07276374429
02832, 1.998771288]
 40.021114 seconds (8.95 M allocations: 638.233 MiB, 70.27% gc time, 19.43%
 compilation time)
n = 20
structural_simplify_times = [1.04034289, 0.069928445, 0.066253837]
component times = [(0.1775529384613037, 5.972403258655804e-8, 0.421359875),
 (0.10079503059387207, 7.226002055498458e-8, 2.458908222), (0.1128020286560
0586, 2.3904618473895583e-8, 0.004822269), (0.011995077133178711, 1.7912738
21464393e-8, 0.004859198), (0.1637101173400879, 5.9479633401222e-8, 4.13609
3023), (0.1066739559173584, 3.405141129032258e-8, 0.718687935), (0.01440286
636352539, 2.458835341365462e-8, 0.015680194)]
total times = [1.6392557034613038, 2.629631697593872, 0.18755274265600586, 
0.08678272013317871, 4.3660569773400875, 0.8916157279173584, 0.096336897363
52538, 2.660943393]
 43.894563 seconds (10.22 M allocations: 739.064 MiB, 64.39% gc time, 20.98
% compilation time)
n = 40
structural_simplify_times = [3.156993985, 0.13993237, 0.101908179]
component times = [(0.2424180507659912, 1.1943784378437845e-7, 0.638181403)
, (0.14669513702392578, 1.4928242424242425e-7, 2.449286658), (0.12163305282
592773, 3.8829637096774194e-8, 0.006849561), (0.017750978469848633, 3.22726
3581488933e-8, 0.00706396), (0.1633758544921875, 1.0693683083511778e-7, 4.1
75262925), (0.11295199394226074, 4.1534343434343434e-8, 0.024734727), (0.01
7606019973754883, 4.615283400809717e-8, 0.025382571)]
total times = [4.037593438765992, 2.735914165023926, 0.26841498382592777, 0
.16474730846984864, 4.440546958492187, 0.23959489994226074, 0.1448967699737
549, 3.936791752]
 49.514251 seconds (11.90 M allocations: 879.270 MiB, 57.56% gc time, 21.83
% compilation time)
n = 60
structural_simplify_times = [0.418165567, 0.199927583, 0.135006803]
component times = [(0.3094191551208496, 1.8214306569343066e-7, 0.895276244)
, (0.19982099533081055, 2.292945054945055e-7, 2.457098232), (0.143754959106
4453, 5.375532994923858e-8, 0.008806704), (0.02452993392944336, 4.660161779
5753284e-8, 0.008823713), (0.16492414474487305, 1.5858333333333335e-7, 4.13
1905991), (0.11380219459533691, 6.138673469387755e-8, 0.034465893), (0.0195
77980041503906, 6.450357507660878e-8, 0.034301674)]
total times = [1.6228609661208497, 2.8568468103308104, 0.35248924610644533,
 0.23328122992944336, 4.431836938744873, 0.28327489059533695, 0.18888645704
150392, 4.851144145]
 49.346454 seconds (13.97 M allocations: 1.023 GiB, 58.14% gc time, 16.65% 
compilation time)
n = 80
structural_simplify_times = [0.593369476, 0.28248948, 0.174742172]
component times = [(0.40892601013183594, 2.465518134715026e-7, 1.247508328)
, (0.26174402236938477, 3.069595141700405e-7, 2.44826818), (0.1591110229492
1875, 6.867725409836065e-8, 0.010314641), (0.03115105628967285, 6.137512742
099898e-8, 0.010166962), (0.1643989086151123, 2.02817094017094e-7, 4.133892
034), (0.11824607849121094, 8.155336787564766e-8, 0.045220499), (0.01858806
610107422, 8.006101344364012e-8, 0.045027361)]
total times = [2.2498038141318357, 2.992501682369385, 0.4519151439492188, 0
.32380749828967287, 4.473033114615112, 0.33820874949121094, 0.2383575991010
7424, 6.078085331]
 51.617568 seconds (16.44 M allocations: 1.218 GiB, 55.81% gc time, 16.81% 
compilation time)
n = 160
structural_simplify_times = [1.634548133, 0.561448604, 0.33373785]
component times = [(0.8750650882720947, 5.813756906077348e-7, 3.014677525),
 (0.5449450016021729, 6.171618497109827e-7, 2.475878296), (0.29151701927185
06, 1.2837725225225223e-7, 0.017808446), (0.0624089241027832, 1.21383167220
37653e-7, 0.018003535), (0.16976189613342285, 4.10445e-7, 4.194220426), (0.
13185405731201172, 1.4487365911799763e-7, 0.088505996), (0.0322220325469970
7, 1.6200262123197903e-7, 0.088194079)]
total times = [5.524290746272095, 3.582271901602173, 0.8707740692718505, 0.
6418610631027831, 4.697720172133423, 0.5540979033120117, 0.4541539615469970
5, 10.588681953]
 63.593598 seconds (27.91 M allocations: 2.115 GiB, 46.16% gc time, 17.45% 
compilation time)
n = 320
structural_simplify_times = [5.943898508, 1.251816827, 1.416206719]
component times = [(2.2271511554718018, 1.268e-6, 8.789504261), (1.42588901
5197754, 1.2429e-6, 2.48110632), (0.4993860721588135, 3.702864077669903e-7,
 0.033351433), (0.13523101806640625, 2.408312958435208e-7, 0.030060661), (0
.17807602882385254, 7.947422680412372e-7, 4.47250908), (0.14578819274902344
, 2.790655172413793e-7, 0.205422429), (0.031210899353027344, 2.945135135135
135e-7, 0.203463677)]
total times = [16.960553924471803, 5.158812162197753, 1.7845543321588135, 1
.4171085060664064, 6.066791827823852, 1.7674173407490235, 1.650881295353027
4, 21.24477477]
 87.671262 seconds (59.05 M allocations: 4.876 GiB, 36.24% gc time, 21.57% 
compilation time)
n = 480
structural_simplify_times = [11.985246599, 2.112687605, 1.079447973]
component times = [(4.596576929092407, 2.0254444444444445e-6, 17.286343735)
, (2.636476993560791, 1.8649000000000001e-6, 2.519024658), (0.7520720958709
717, 6.931689189189189e-7, 0.05687624), (0.21306395530700684, 3.59852380952
38096e-7, 0.056300025), (0.18342804908752441, 1.199e-6, 4.533588078), (0.16
07210636138916, 4.2507035175879396e-7, 0.349732589), (0.037407875061035156,
 4.3221717171717173e-7, 0.351435054)]
total times = [33.8681672630924, 7.268189256560791, 2.9216359408709716, 2.3
82051585307007, 5.796464100087524, 1.5899016256138916, 1.468290902061035, 3
4.237785521]
121.862775 seconds (102.96 M allocations: 9.302 GiB, 27.39% gc time, 24.33%
 compilation time)
n = 640
structural_simplify_times = [20.858320906, 3.127551273, 1.423317469]
component times = [(5.977561950683594, 2.6621111111111113e-6, 27.414821076)
, (4.497936964035034, 2.493222222222222e-6, 2.986384131), (1.03083896636962
9, 1.0969e-6, 0.061079045), (0.2985720634460449, 4.792769230769231e-7, 0.05
9647138), (0.18740606307983398, 1.625e-6, 5.307199825), (0.1930170059204101
6, 6.161085714285714e-7, 1.164040584), (0.04276704788208008, 6.288176470588
236e-7, 1.170139892)]
total times = [54.25070393268359, 10.611872368035034, 4.219469284369628, 3.
4857704744460447, 6.917923357079833, 2.78037505892041, 2.6362244088820797, 
47.691071023]
164.386261 seconds (157.40 M allocations: 15.108 GiB, 21.30% gc time, 26.56
% compilation time)
n = 800
structural_simplify_times = [32.103341631, 4.085944433, 1.919213185]
component times = [(8.481768131256104, 3.54125e-6, 39.231240687), (6.350100
040435791, 3.164875e-6, 2.676963693), (1.3979871273040771, 1.489e-6, 0.0693
84805), (0.3920409679412842, 5.989325842696629e-7, 0.075449953), (0.1939790
2488708496, 2.054444444444444e-6, 5.329468125), (0.21952509880065918, 8.485
507246376811e-7, 1.097676721), (0.048033952713012695, 7.873168316831683e-7,
 1.090282775)]
total times = [79.8163504492561, 13.11300816643579, 5.553316365304077, 4.55
3435353941284, 7.442660334887085, 3.2364150048006595, 3.0575299127130124, 5
6.668048584]
204.451047 seconds (227.73 M allocations: 24.875 GiB, 18.10% gc time, 28.69
% compilation time)
n = 1000
structural_simplify_times = [50.998983242, 5.329089782, 2.566782702]
component times = [(12.696861982345581, 5.105e-6, 58.706836564), (9.0867679
1191101, 4.02e-6, 2.592123831), (1.9237749576568604, 2.2054444444444445e-6,
 0.104304844), (0.5080690383911133, 7.493305785123966e-7, 0.100461096), (0.
2019050121307373, 2.5787777777777778e-6, 6.370961228), (0.24386906623840332
, 9.927272727272727e-7, 1.403054546), (0.052926063537597656, 9.96e-7, 1.386
691131)]
total times = [122.40268178834557, 17.00798152491101, 7.35716958365686, 5.9
37619916391113, 9.139648942130737, 4.213706314238403, 4.006399896537598, 76
.576270411]
276.377404 seconds (338.02 M allocations: 44.853 GiB, 15.12% gc time, 30.10
% compilation time)
n = 2000
structural_simplify_times = [221.849713755, 15.288125823, 7.484429316]
component times = [(34.949114084243774, 9.48e-6, 176.300842073), (30.361908
91265869, 8.663e-6, 2.771215973), (5.9065961837768555, 4.631285714285715e-6
, 0.199837098), (1.2971129417419434, 1.511e-6, 0.175259703), (0.24333214759
82666, 5.263333333333333e-6, 7.99654846), (0.3919498920440674, 2.0688888888
88889e-6, 3.529939779), (0.0937809944152832, 1.9899e-6, 3.522440151)]
total times = [433.0996699122438, 48.42125070865869, 21.394559104776857, 16
.760498467741943, 15.724309923598266, 11.406318987044067, 11.10065046141528
3, 172.981590769]
743.670086 seconds (1.21 G allocations: 231.272 GiB, 9.14% gc time, 31.80% 
compilation time)
n = 3000
structural_simplify_times = [543.334348026, 27.18149096, 13.474527301]
component times = [(70.06858587265015, 1.546e-5, 385.069128283), (62.314460
99281311, 1.381e-5, 2.972553877), (12.282616138458252, 6.89e-6, 0.327576154
), (2.2706358432769775, 2.391e-6, 0.330052513), (0.2789771556854248, 7.7397
5e-6, 11.202565713), (0.5636188983917236, 3.099875e-6, 6.52094887), (0.1431
0503005981445, 2.99e-6, 6.780246863)]
total times = [998.4720621816502, 92.46850582981311, 39.791683252458256, 29
.782179316276977, 24.956070169685425, 20.559095069391724, 20.39787919405981
5, 273.229783573]
1487.406310 seconds (2.68 G allocations: 583.065 GiB, 9.16% gc time, 33.32%
 compilation time)
n = 4000
structural_simplify_times = [1038.422110883, 42.864754326, 20.414720375]
component times = [(123.07310009002686, 2.095e-5, 656.912869343), (108.5870
3899383545, 1.917e-5, 3.058824964), (23.0722758769989, 9.52e-6, 0.417834512
), (3.5720999240875244, 3.0075e-6, 0.429670087), (0.3170359134674072, 1.042
e-5, 21.844827063), (0.7697608470916748, 4.028571428571429e-6, 17.096597799
), (0.16802716255187988, 4.035714285714286e-6, 15.839356565)]
total times = [1818.4080803160268, 154.51061828383544, 66.3548647149989, 46
.866524337087526, 42.576583351467406, 38.28107902109168, 36.422104102551884
, 381.648024701]
2538.000712 seconds (4.79 G allocations: 1.370 TiB, 8.31% gc time, 33.49% c
ompilation time)
n = 5000
structural_simplify_times = [NaN, 63.451551988, 25.086043132]
component times = [(NaN, NaN, NaN), (162.31976985931396, 2.274e-5, 3.012876
081), (39.54512596130371, 1.224e-5, 0.460517135), (5.091353178024292, 3.749
875e-6, 0.45483112), (0.35204601287841797, 1.3059e-5, 33.624782322), (0.965
7459259033203, 5.251666666666667e-6, 31.287642724), (0.24301815032958984, 5
.048333333333333e-6, 31.631704887)]
total times = [NaN, 228.78419792831397, 103.45719508430372, 68.997736286024
29, 59.06287146687842, 57.33943178190332, 56.96076616932959, 509.591553363]
979.512005 seconds (307.98 M allocations: 51.932 GiB, 5.93% gc time, 16.41%
 compilation time)
n = 6000
structural_simplify_times = [NaN, 89.234660114, 35.616282902]
component times = [(NaN, NaN, NaN), (239.25767397880554, 2.7449e-5, 3.11340
0712), (67.16731309890747, 1.532e-5, 0.553430192), (7.5815510749816895, 4.5
e-6, 0.550862136), (0.3829820156097412, 1.5639e-5, 48.774392363), (0.977191
9250488281, 6.16e-6, 47.590877348), (0.28621816635131836, 6.132e-6, 45.8027
31325)]
total times = [NaN, 331.60573480480554, 156.95540340490749, 97.367073324981
69, 84.77365728060974, 84.18435217504883, 81.70523239335132, 721.393761949]
1386.428831 seconds (390.34 M allocations: 69.668 GiB, 4.81% gc time, 16.87
% compilation time)
n = 7000
structural_simplify_times = [NaN, 111.416990762, 57.564236093]
component times = [(NaN, NaN, NaN), (319.21988892555237, 3.346e-5, 3.448434
594), (117.12885093688965, 1.818e-5, 0.719597933), (10.723907947540283, 5.2
598333333333334e-6, 0.712474208), (0.4635601043701172, 1.823e-5, 62.6165734
29), (1.0939080715179443, 7.3175e-6, 57.315784043), (0.26888608932495117, 7
.3125e-6, 57.166019857)]
total times = [NaN, 434.08531428155237, 229.26543963188965, 122.85337291754
028, 120.64436962637012, 115.97392820751794, 114.99914203932495, 969.754289
256]
1854.252737 seconds (483.06 M allocations: 90.232 GiB, 4.20% gc time, 16.66
% compilation time)
n = 8000
structural_simplify_times = [NaN, 136.898405362, 66.169324506]
component times = [(NaN, NaN, NaN), (426.4493818283081, 3.759e-5, 3.5152085
84), (168.4442298412323, 2.237e-5, 0.842497969), (15.379122972488403, 6.104
e-6, 0.902032875), (0.5140330791473389, 2.099e-5, 70.70353037), (1.33994984
62677002, 9.013333333333334e-6, 73.466124735), (0.3989100456237793, 8.40666
6666666665e-6, 71.484508292)]
total times = [NaN, 566.862995774308, 306.1851331722323, 153.1795612094884,
 137.38688795514733, 140.9753990872677, 138.05274284362378, 1241.375228496]
2368.544197 seconds (586.83 M allocations: 112.939 GiB, 4.42% gc time, 17.1
4% compilation time)
n = 9000
structural_simplify_times = [NaN, 170.064762205, 93.419819197]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (232.5521821975708, 2.
484e-5, 0.916540139), (16.620287895202637, 7.40725e-6, 1.075436606), (0.519
8929309844971, 2.375e-5, 75.787152295), (1.3906209468841553, 9.68e-6, 78.05
5502188), (0.4027838706970215, 9.659e-6, 76.965132001)]
total times = [NaN, NaN, 403.5334845415708, 187.76048670620264, 169.7268644
2298448, 172.86594233188416, 170.78773506869703, 1604.811459372]
2432.355862 seconds (673.37 M allocations: 132.035 GiB, 4.70% gc time, 0.19
% compilation time)
n = 10000
structural_simplify_times = [NaN, 258.412801334, 161.402181187]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (312.9138810634613, 2.
72e-5, 1.010260955), (20.753078937530518, 7.50975e-6, 0.904594523), (0.6475
708484649658, 2.637e-5, 90.605738335), (1.6930899620056152, 1.086e-5, 92.32
5829724), (0.4694480895996094, 1.096e-5, 92.623814117)]
total times = [NaN, NaN, 572.3369433524614, 280.0704747945305, 252.65549037
046497, 255.42110087300563, 254.4954433935996, NaN]
1119.419219 seconds (784.95 M allocations: 157.058 GiB, 17.67% gc time, 0.4
3% compilation time)
n = 20000
structural_simplify_times = [NaN, 1376.049222709, 993.362175622]
component times = [(NaN, NaN, NaN), (NaN, NaN, NaN), (1625.7704689502716, 5
.4749e-5, 8.713745898), (293.159215927124, 2.3009e-5, 5.725976445), (3.0953
900814056396, 5.357e-5, 256.072742091), (4.385202884674072, 2.274e-5, 253.1
1184266), (1.258882999420166, 2.28e-5, 260.146564586)]
total times = [NaN, NaN, 3010.5334375572716, 1674.9344150811241, 1252.53030
77944056, 1250.8592211666742, 1254.7676232074202, NaN]
5183.725517 seconds (2.38 G allocations: 550.116 GiB, 14.11% gc time, 0.09%
 compilation time)
```





## Dymola Timings

Dymola requires a license server and thus cannot be hosted. This was run locally for the
following times:

```julia
translation_and_total_times = [
7.027, 7.237
11.295, 11.798
16.681, 17.646
22.125, 23.839
27.529, 29.82
33.282, 36.622
39.007, 43.088
44.825, 51.601
50.281, 56.676
] # TODO: I will add other times once the Dymola license server is back up.
#total_times[:, 6] = translation_and_total_times[1:length(N_x),2]
```

```
Error: ParseError:
# Error @ /cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/
benchmarks/ModelingToolkit/RCCircuit.jmd:4:1
7.027, 7.237
┌─────────────
11.295, 11.798
16.681, 17.646
⋮
44.825, 51.601
50.281, 56.676
#────────────┘ ── Expected `]`
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

![](figures/RCCircuit_7_1.png)



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
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Project.toml`
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
  [1dea7af3] OrdinaryDiffEq v6.83.2
  [f27b6e38] Polynomials v4.0.11
  [ba661fbb] PreferenceTools v0.1.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [0c5d862f] Symbolics v5.30.3
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/ModelingToolkit/Manifest.toml`
  [47edcb42] ADTypes v1.3.0
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
  [7c1d4256] DynamicPolynomials v0.5.7
⌅ [06fc5a27] DynamicQuantities v0.13.2
  [4e289a0a] EnumX v1.0.4
  [f151be2c] EnzymeCore v0.7.5
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
  [842dd82b] InlineStrings v1.4.0
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
  [5078a376] LazyArrays v2.0.4
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
  [1dea7af3] OrdinaryDiffEq v6.83.2
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
  [731186ca] RecursiveArrayTools v3.24.0
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
  [2efcf032] SymbolicIndexingInterface v0.3.23
  [19f23fe9] SymbolicLimits v0.2.1
  [d1185830] SymbolicUtils v2.0.2
  [0c5d862f] Symbolics v5.30.3
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [62fd8b95] TensorCore v0.1.1
⌅ [8ea1fca8] TermInterface v0.4.1
  [8290d209] ThreadingUtilities v0.5.2
  [731e570b] TiffImages v0.10.0
  [a759f4b9] TimerOutputs v0.5.24
  [0796e94c] Tokenize v0.5.29
  [3bb67fe8] TranscodingStreams v0.10.9
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
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`
```

