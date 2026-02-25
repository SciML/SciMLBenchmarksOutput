---
author: "Chris Rackauckas and Yingbo Ma"
title: "Bruss Scaling PDE Differentaition Benchmarks"
---


From the paper [A Comparison of Automatic Differentiation and Continuous Sensitivity Analysis for Derivatives of Differential Equation Solutions](https://ieeexplore.ieee.org/abstract/document/9622796)

```julia
using OrdinaryDiffEq, ReverseDiff, ForwardDiff, FiniteDiff, SciMLSensitivity
using LinearAlgebra, Tracker, Mooncake, Plots
```


```julia
function makebrusselator(N = 8)
    xyd_brusselator = range(0, stop = 1, length = N)
    function limit(a, N)
        if a == N+1
            return 1
        elseif a == 0
            return N
        else
            return a
        end
    end
    brusselator_f(x, y, t) = ifelse(
        (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
        (t >= 1.1), 5.0, 0.0)
    brusselator_2d_loop = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
        function brusselator_2d_loop(du, u, p, t)
            @inbounds begin
                ii1 = N^2
                ii2 = ii1+N^2
                ii3 = ii2+2(N^2)
                A = @view p[1:ii1]
                B = @view p[(ii1 + 1):ii2]
                α = @view p[(ii2 + 1):ii3]
                II = LinearIndices((N, N, 2))
                for I in CartesianIndices((N, N))
                    x = xyd[I[1]]
                    y = xyd[I[2]]
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N);
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N);
                    jm1 = limit(j-1, N)
                    du[II[i, j, 1]] = α[II[
                                          i, j, 1]]*(u[II[im1, j, 1]] + u[II[ip1, j, 1]] +
                                                     u[II[i, jp1, 1]] + u[II[i, jm1, 1]] -
                                                     4u[II[i, j, 1]])/dx^2 +
                                      B[II[i, j, 1]] + u[II[i, j, 1]]^2*u[II[i, j, 2]] -
                                      (A[II[i, j, 1]] + 1)*u[II[i, j, 1]] +
                                      brusselator_f(x, y, t)
                end
                for I in CartesianIndices((N, N))
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N)
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N)
                    jm1 = limit(j-1, N)
                    du[II[i, j, 2]] = α[II[
                        i, j, 2]]*(u[II[im1, j, 2]] + u[II[ip1, j, 2]] + u[II[i, jp1, 2]] +
                                   u[II[i, jm1, 2]] - 4u[II[i, j, 2]])/dx^2 +
                                      A[II[i, j, 1]]*u[II[i, j, 1]] -
                                      u[II[i, j, 1]]^2*u[II[i, j, 2]]
                end
                return nothing
            end
        end
    end
    function init_brusselator_2d(xyd)
        N = length(xyd)
        u = zeros(N, N, 2)
        for I in CartesianIndices((N, N))
            x = xyd[I[1]]
            y = xyd[I[2]]
            u[I, 1] = 22*(y*(1-y))^(3/2)
            u[I, 2] = 27*(x*(1-x))^(3/2)
        end
        vec(u)
    end
    dx = step(xyd_brusselator)
    e1 = ones(N-1)
    off = N-1
    e4 = ones(N-off)
    T = diagm(0=>-2ones(N), -1=>e1, 1=>e1, off=>e4, -off=>e4) ./ dx^2
    Ie = Matrix{Float64}(I, N, N)
    # A + df/du
    Op = kron(Ie, T) + kron(T, Ie)
    brusselator_jac = let N=N
        (J, a, p, t) -> begin
            ii1 = N^2
            ii2 = ii1+N^2
            ii3 = ii2+2(N^2)
            A = @view p[1:ii1]
            B = @view p[(ii1 + 1):ii2]
            α = @view p[(ii2 + 1):ii3]
            u = @view a[1:(end ÷ 2)]
            v = @view a[(end ÷ 2 + 1):end]
            N2 = length(a)÷2
            α1 = @view α[1:(end ÷ 2)]
            α2 = @view α[(end ÷ 2 + 1):end]
            fill!(J, 0)

            J[1:N2, 1:N2] .= α1 .* Op
            J[(N2 + 1):end, (N2 + 1):end] .= α2 .* Op

            J1 = @view J[1:N2, 1:N2]
            J2 = @view J[(N2 + 1):end, 1:N2]
            J3 = @view J[1:N2, (N2 + 1):end]
            J4 = @view J[(N2 + 1):end, (N2 + 1):end]
            J1[diagind(J1)] .+= @. 2u*v-(A+1)
            J2[diagind(J2)] .= @. A-2u*v
            J3[diagind(J3)] .= @. u^2
            J4[diagind(J4)] .+= @. -u^2
            nothing
        end
    end
    Jmat = zeros(2N*N, 2N*N)
    dp = zeros(2N*N, 4N*N)
    brusselator_comp = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator), Jmat=Jmat,
        dp=dp, brusselator_jac=brusselator_jac

        function brusselator_comp(dus, us, p, t)
            @inbounds begin
                ii1 = N^2
                ii2 = ii1+N^2
                ii3 = ii2+2(N^2)
                @views u, s = us[1:ii2], us[(ii2 + 1):end]
                du = @view dus[1:ii2]
                ds = @view dus[(ii2 + 1):end]
                fill!(dp, 0)
                A = @view p[1:ii1]
                B = @view p[(ii1 + 1):ii2]
                α = @view p[(ii2 + 1):ii3]
                dfdα = @view dp[:, (ii2 + 1):ii3]
                diagind(dfdα)
                for i in 1:ii1
                    dp[i, ii1 + i] = 1
                end
                II = LinearIndices((N, N, 2))
                uu = @view u[1:(end ÷ 2)]
                for i in eachindex(uu)
                    dp[i, i] = -uu[i]
                    dp[i + ii1, i] = uu[i]
                end
                for I in CartesianIndices((N, N))
                    x = xyd[I[1]]
                    y = xyd[I[2]]
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N);
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N);
                    jm1 = limit(j-1, N)
                    au = dfdα[II[i, j, 1], II[i, j, 1]] = (u[II[im1, j, 1]] +
                                                           u[II[ip1, j, 1]] +
                                                           u[II[i, jp1, 1]] +
                                                           u[II[i, jm1, 1]] -
                                                           4u[II[i, j, 1]])/dx^2
                    du[II[i, j, 1]] = α[II[i, j, 1]]*(au) + B[II[i, j, 1]] +
                                      u[II[i, j, 1]]^2*u[II[i, j, 2]] -
                                      (A[II[i, j, 1]] + 1)*u[II[i, j, 1]] +
                                      brusselator_f(x, y, t)
                end
                for I in CartesianIndices((N, N))
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N)
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N)
                    jm1 = limit(j-1, N)
                    av = dfdα[II[i, j, 2], II[i, j, 2]] = (u[II[im1, j, 2]] +
                                                           u[II[ip1, j, 2]] +
                                                           u[II[i, jp1, 2]] +
                                                           u[II[i, jm1, 2]] -
                                                           4u[II[i, j, 2]])/dx^2
                    du[II[i, j, 2]] = α[II[i, j, 2]]*(av) + A[II[i, j, 1]]*u[II[i, j, 1]] -
                                      u[II[i, j, 1]]^2*u[II[i, j, 2]]
                end
                brusselator_jac(Jmat, u, p, t)
                BLAS.gemm!('N', 'N', 1.0, Jmat, reshape(s, 2N*N, 4N*N), 1.0, dp)
                copyto!(ds, vec(dp))
                return nothing
            end
        end
    end
    u0 = init_brusselator_2d(xyd_brusselator)
    p = [fill(3.4, N^2); fill(1.0, N^2); fill(10.0, 2*N^2)]
    brusselator_2d_loop, u0,
    p,
    brusselator_jac,
    ODEProblem(brusselator_comp, copy([u0; zeros((N^2*2)*(N^2*4))]), (0.0, 10.0), p)
end

Base.eps(::Type{Tracker.TrackedReal{T}}) where {T} = eps(T)
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v') # bad bad hack
```




## Setup AutoDiff

```julia
bt = 0:0.1:1
tspan = (0.0, 1.0)
forwarddiffn = vcat(2:10, 12, 15)
reversediffn = 2:10
numdiffn = vcat(2:10, 12)
csan = vcat(2:10, 12, 15, 17)
#csaseedn = 2:10
tols = (abstol = 1e-5, reltol = 1e-7)

@isdefined(PROBS) || (const PROBS = Dict{Int, Any}())
makebrusselator!(dict, n) = get!(()->makebrusselator(n), dict, n)

_adjoint_methods_iq = ntuple(2) do ii
    Alg = (InterpolatingAdjoint, QuadratureAdjoint)[ii]
    (
        user = Alg(autodiff = false, autojacvec = false), # user Jacobian
        adjc = Alg(autodiff = true, autojacvec = false), # AD Jacobian
        advj = Alg(autodiff = true, autojacvec = EnzymeVJP()) # AD vJ
    )
end |> NamedTuple{(:interp, :quad)}
# GaussAdjoint/GaussKronrodAdjoint do not support user-provided Jacobians (autodiff=false)
_adjoint_methods_g = ntuple(2) do ii
    Alg = (GaussAdjoint, GaussKronrodAdjoint)[ii]
    (
        adjc = Alg(autodiff = true, autojacvec = false), # AD Jacobian
        advj = Alg(autodiff = true, autojacvec = EnzymeVJP()) # AD vJ
    )
end |> NamedTuple{(:gauss, :gausskronrod)}
@isdefined(ADJOINT_METHODS_IQ) ||
    (const ADJOINT_METHODS_IQ = mapreduce(collect, vcat, _adjoint_methods_iq))
@isdefined(ADJOINT_METHODS_G) ||
    (const ADJOINT_METHODS_G = mapreduce(collect, vcat, _adjoint_methods_g))

function auto_sen_l2(
        f, u0, tspan, p, t, alg = Tsit5(); diffalg = ReverseDiff.gradient, kwargs...)
    test_f(p) = begin
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, convert.(eltype(p), u0), tspan, p)
        sol = solve(prob, alg, saveat = t; kwargs...)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    diffalg(test_f, p)
end
@inline function diffeq_sen_l2(df, u0, tspan, p, t, alg = Tsit5();
        abstol = 1e-5, reltol = 1e-7, iabstol = abstol, ireltol = reltol,
        sensalg = SensitivityAlg(), kwargs...)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(df, u0, tspan, p)
    saveat = tspan[1] != t[1] && tspan[end] != t[end] ? vcat(tspan[1], t, tspan[end]) : t
    sol = solve(prob, alg, abstol = abstol, reltol = reltol, saveat = saveat; kwargs...)
    dg(out, u, p, t, i) = (out.=u .- 1.0)
    adjoint_sensitivities(sol, alg; t, abstol = abstol, dgdu_discrete = dg,
        reltol = reltol, sensealg = sensalg)
end
```

```
diffeq_sen_l2 (generic function with 2 methods)
```





## AD Choice Benchmarks

```julia
forwarddiff = map(forwarddiffn) do n
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    @elapsed auto_sen_l2(
        bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg = (ForwardDiff.gradient), tols...)
    t = @elapsed auto_sen_l2(
        bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg = (ForwardDiff.gradient), tols...)
    @show n, t
    t
end
```

```
(n, t) = (2, 0.002040158)
(n, t) = (3, 0.024217215)
(n, t) = (4, 0.10356527)
(n, t) = (5, 0.405362752)
(n, t) = (6, 1.124729214)
(n, t) = (7, 2.907330858)
(n, t) = (8, 10.529452678)
(n, t) = (9, 13.91474206)
(n, t) = (10, 27.970482694)
(n, t) = (12, 111.441132354)
(n, t) = (15, 1108.833921963)
11-element Vector{Float64}:
    0.002040158
    0.024217215
    0.10356527
    0.405362752
    1.124729214
    2.907330858
   10.529452678
   13.91474206
   27.970482694
  111.441132354
 1108.833921963
```



```julia
#=
reversediff = map(reversediffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...)
  @show n,t
  t
end
=#
```


```julia
numdiff = map(numdiffn) do n
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5());
        diffalg = (FiniteDiff.finite_difference_gradient), tols...)
    t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5());
        diffalg = (FiniteDiff.finite_difference_gradient), tols...)
    @show n, t
    t
end
```

```
(n, t) = (2, 0.004432713)
(n, t) = (3, 0.043896629)
(n, t) = (4, 0.153959612)
(n, t) = (5, 0.447314984)
(n, t) = (6, 1.116653605)
(n, t) = (7, 2.422400463)
(n, t) = (8, 23.733686468)
(n, t) = (9, 46.048168635)
(n, t) = (10, 83.647805051)
(n, t) = (12, 239.815758293)
10-element Vector{Float64}:
   0.004432713
   0.043896629
   0.153959612
   0.447314984
   1.116653605
   2.422400463
  23.733686468
  46.048168635
  83.647805051
 239.815758293
```





Warmup: run each adjoint method once at the smallest size to ensure all compilation
is complete before we start timing.

```julia
let n = first(csan)
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    solver = Rodas5(autodiff = false)
    for alg in ADJOINT_METHODS_IQ
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction(bfun, jac = brusselator_jac)
        diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
    end
    for alg in ADJOINT_METHODS_G
        diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
    end
end
```


```julia
csa_iq = map(csan) do n
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    @time ts = map(ADJOINT_METHODS_IQ) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction(bfun, jac = brusselator_jac)
        solver = Rodas5(autodiff = false)
        @time diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        t = @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        return t
    end
    @show n, ts
    ts
end
```

```
0.006236 seconds (12.90 k allocations: 950.062 KiB)
  0.003674 seconds (10.97 k allocations: 838.234 KiB)
  0.003285 seconds (6.28 k allocations: 976.500 KiB)
  0.002205 seconds (5.36 k allocations: 389.234 KiB)
  0.001749 seconds (4.07 k allocations: 294.672 KiB)
  0.002087 seconds (6.78 k allocations: 668.500 KiB)
  0.780130 seconds (979.30 k allocations: 74.035 MiB, 94.31% compilation ti
me)
(n, ts) = (2, [0.006062114, 0.00321655, 0.002879233, 0.001819519, 0.0013643
32, 0.00168936])
  0.026048 seconds (18.72 k allocations: 2.246 MiB)
 19.208661 seconds (4.80 M allocations: 325.827 MiB, 0.72% gc time, 99.90% 
compilation time)
  0.006242 seconds (8.44 k allocations: 2.089 MiB)
  0.004810 seconds (6.74 k allocations: 832.625 KiB)
  7.562890 seconds (3.78 M allocations: 261.082 MiB, 0.97% gc time, 99.87% 
compilation time)
  0.003518 seconds (7.87 k allocations: 1.062 MiB)
 26.874217 seconds (8.68 M allocations: 602.166 MiB, 0.79% gc time, 99.51% 
compilation time)
(n, ts) = (3, [0.025765926, 0.013539899, 0.005924224, 0.004561682, 0.004197
145, 0.002898243])
  0.113978 seconds (30.31 k allocations: 5.617 MiB)
 20.484387 seconds (4.81 M allocations: 328.686 MiB, 0.66% gc time, 99.69% 
compilation time)
  0.013594 seconds (13.09 k allocations: 4.943 MiB)
  0.010861 seconds (9.03 k allocations: 1.721 MiB)
  9.885418 seconds (3.78 M allocations: 261.719 MiB, 0.61% gc time, 99.82% 
compilation time)
  0.005478 seconds (9.71 k allocations: 1.804 MiB)
 30.732428 seconds (8.75 M allocations: 624.831 MiB, 0.64% gc time, 98.56% 
compilation time)
(n, ts) = (4, [0.114000107, 0.057350536, 0.013126302, 0.010782696, 0.012738
873, 0.005307468])
  0.789384 seconds (44.47 k allocations: 11.983 MiB, 7.35% gc time)
 19.793292 seconds (4.19 M allocations: 293.634 MiB, 0.32% gc time, 98.26% 
compilation time)
  0.070438 seconds (18.81 k allocations: 10.368 MiB)
  0.026241 seconds (12.13 k allocations: 3.424 MiB)
  9.045043 seconds (3.78 M allocations: 262.839 MiB, 0.89% gc time, 99.49% 
compilation time)
  0.009736 seconds (12.20 k allocations: 3.155 MiB)
 30.875796 seconds (8.19 M allocations: 627.175 MiB, 0.65% gc time, 92.13% 
compilation time)
(n, ts) = (5, [0.685932454, 0.302709228, 0.068225081, 0.025323479, 0.040104
37, 0.009233825])
  1.305657 seconds (61.65 k allocations: 23.606 MiB)
 21.877337 seconds (4.23 M allocations: 305.804 MiB, 97.11% compilation tim
e)
  0.119597 seconds (25.70 k allocations: 19.800 MiB)
  0.053114 seconds (15.54 k allocations: 6.511 MiB)
 11.415101 seconds (3.78 M allocations: 264.929 MiB, 0.43% gc time, 99.02% 
compilation time)
  0.014758 seconds (14.93 k allocations: 5.270 MiB)
 37.483984 seconds (8.31 M allocations: 705.720 MiB, 0.74% gc time, 86.83% 
compilation time)
(n, ts) = (6, [1.570017079, 0.619854408, 0.326902613, 0.051631621, 0.106445
352, 0.014204914])
  3.019686 seconds (85.16 k allocations: 43.958 MiB)
 23.791409 seconds (3.65 M allocations: 285.018 MiB, 0.86% gc time, 93.19% 
compilation time)
  0.296999 seconds (35.11 k allocations: 37.172 MiB)
  0.138696 seconds (19.57 k allocations: 10.920 MiB)
  2.122874 seconds (297.39 k allocations: 26.693 MiB, 6.54% gc time, 83.70%
 compilation time)
  0.023440 seconds (18.16 k allocations: 8.466 MiB)
 35.261770 seconds (4.34 M allocations: 558.115 MiB, 0.97% gc time, 67.92% 
compilation time)
(n, ts) = (7, [3.52981972, 1.686694748, 0.298176064, 0.11024221, 0.20550403
8, 0.023104851])
  7.260600 seconds (109.48 k allocations: 73.645 MiB)
  3.562679 seconds (87.69 k allocations: 62.066 MiB, 3.43% gc time)
  0.426785 seconds (44.84 k allocations: 61.664 MiB)
  0.246452 seconds (24.97 k allocations: 18.519 MiB)
  0.600926 seconds (15.90 k allocations: 13.675 MiB)
  0.095499 seconds (22.48 k allocations: 13.599 MiB)
 23.250362 seconds (612.15 k allocations: 486.577 MiB, 0.66% gc time)
(n, ts) = (8, [6.208048704, 3.444487957, 0.452262221, 0.24708268, 0.6027212
79, 0.091282784])
 12.805686 seconds (142.00 k allocations: 119.495 MiB, 0.39% gc time)
  7.664899 seconds (121.67 k allocations: 104.006 MiB, 0.51% gc time)
  0.671166 seconds (61.84 k allocations: 103.098 MiB, 5.20% gc time)
  0.436471 seconds (30.42 k allocations: 29.385 MiB)
  1.204360 seconds (19.17 k allocations: 21.647 MiB, 0.77% gc time)
  0.128228 seconds (26.84 k allocations: 20.817 MiB)
 45.496688 seconds (805.30 k allocations: 797.139 MiB, 0.29% gc time)
(n, ts) = (9, [12.78630962, 7.520486273, 0.533291945, 0.439394607, 1.169663
412, 0.124640163])
 23.530229 seconds (173.85 k allocations: 177.456 MiB, 0.16% gc time)
 13.106965 seconds (139.19 k allocations: 148.371 MiB)
  0.814591 seconds (70.60 k allocations: 146.135 MiB, 1.99% gc time)
  0.746394 seconds (36.50 k allocations: 43.825 MiB)
  2.121546 seconds (22.82 k allocations: 32.313 MiB)
  0.190976 seconds (31.70 k allocations: 30.308 MiB)
 81.030976 seconds (950.73 k allocations: 1.130 GiB, 0.16% gc time)
(n, ts) = (10, [23.52359096, 13.146022325, 0.737904317, 0.785133595, 2.1224
61659, 0.193375041])
 70.311492 seconds (256.54 k allocations: 370.194 MiB, 0.06% gc time)
 41.339047 seconds (205.34 k allocations: 309.939 MiB, 0.07% gc time)
  1.407581 seconds (103.68 k allocations: 303.808 MiB, 0.58% gc time)
  1.905055 seconds (50.59 k allocations: 88.514 MiB, 0.22% gc time)
  6.361798 seconds (31.27 k allocations: 65.751 MiB)
  0.379570 seconds (42.97 k allocations: 59.668 MiB)
243.204191 seconds (1.38 M allocations: 2.340 GiB, 0.09% gc time)
(n, ts) = (12, [70.270186137, 41.123283225, 1.363252743, 1.914674074, 6.401
447358, 0.414731382])
264.593260 seconds (397.51 k allocations: 909.305 MiB, 0.84% gc time)
174.097586 seconds (350.79 k allocations: 799.431 MiB, 0.04% gc time)
  3.219311 seconds (170.98 k allocations: 767.125 MiB, 1.09% gc time)
  6.178622 seconds (76.52 k allocations: 218.971 MiB)
 23.578083 seconds (46.83 k allocations: 161.755 MiB)
  0.906791 seconds (63.71 k allocations: 143.222 MiB)
943.068709 seconds (2.21 M allocations: 5.859 GiB, 0.28% gc time)
(n, ts) = (15, [262.353773639, 174.265508816, 3.112992218, 6.213941025, 23.
603260127, 0.931583577])
571.671359 seconds (526.33 k allocations: 1.481 GiB, 0.21% gc time)
323.272178 seconds (393.25 k allocations: 1.192 GiB, 0.35% gc time)
  4.996087 seconds (197.65 k allocations: 1.160 GiB, 0.23% gc time)
 12.401940 seconds (97.00 k allocations: 357.023 MiB, 0.41% gc time)
 49.144526 seconds (59.12 k allocations: 267.234 MiB, 0.02% gc time)
  1.383570 seconds (80.10 k allocations: 234.334 MiB)
1922.930223 seconds (2.71 M allocations: 9.345 GiB, 0.15% gc time)
(n, ts) = (17, [571.605597027, 320.993306257, 4.61190021, 12.308186149, 49.
09628573, 1.428670235])
12-element Vector{Vector{Float64}}:
 [0.006062114, 0.00321655, 0.002879233, 0.001819519, 0.001364332, 0.0016893
6]
 [0.025765926, 0.013539899, 0.005924224, 0.004561682, 0.004197145, 0.002898
243]
 [0.114000107, 0.057350536, 0.013126302, 0.010782696, 0.012738873, 0.005307
468]
 [0.685932454, 0.302709228, 0.068225081, 0.025323479, 0.04010437, 0.0092338
25]
 [1.570017079, 0.619854408, 0.326902613, 0.051631621, 0.106445352, 0.014204
914]
 [3.52981972, 1.686694748, 0.298176064, 0.11024221, 0.205504038, 0.02310485
1]
 [6.208048704, 3.444487957, 0.452262221, 0.24708268, 0.602721279, 0.0912827
84]
 [12.78630962, 7.520486273, 0.533291945, 0.439394607, 1.169663412, 0.124640
163]
 [23.52359096, 13.146022325, 0.737904317, 0.785133595, 2.122461659, 0.19337
5041]
 [70.270186137, 41.123283225, 1.363252743, 1.914674074, 6.401447358, 0.4147
31382]
 [262.353773639, 174.265508816, 3.112992218, 6.213941025, 23.603260127, 0.9
31583577]
 [571.605597027, 320.993306257, 4.61190021, 12.308186149, 49.09628573, 1.42
8670235]
```



```julia
csa_g = map(csan) do n
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    @time ts = map(ADJOINT_METHODS_G) do alg
        @info "Running $alg"
        solver = Rodas5(autodiff = false)
        @time diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        return t
    end
    @show n, ts
    ts
end
```

```
0.002542 seconds (4.94 k allocations: 356.656 KiB)
  0.002667 seconds (5.38 k allocations: 621.141 KiB)
  0.002557 seconds (7.46 k allocations: 441.141 KiB)
  0.002851 seconds (7.51 k allocations: 738.625 KiB)
  0.472304 seconds (523.63 k allocations: 39.362 MiB, 95.01% compilation ti
me)
(n, ts) = (2, [0.00155588, 0.002052917, 0.001893388, 0.002240766])
  8.747381 seconds (4.79 M allocations: 328.840 MiB, 0.98% gc time, 99.85% 
compilation time)
  0.003625 seconds (6.14 k allocations: 1.099 MiB)
  8.170195 seconds (4.32 M allocations: 300.180 MiB, 99.85% compilation tim
e)
  0.004634 seconds (11.92 k allocations: 1.430 MiB)
 16.948448 seconds (9.16 M allocations: 636.147 MiB, 0.50% gc time, 99.67% 
compilation time)
(n, ts) = (3, [0.004682731, 0.003433098, 0.00650831, 0.004363293])
 11.685243 seconds (4.79 M allocations: 329.798 MiB, 1.59% gc time, 99.84% 
compilation time)
  0.006673 seconds (7.82 k allocations: 2.119 MiB)
 11.047912 seconds (4.36 M allocations: 303.577 MiB, 99.76% compilation tim
e)
  0.008220 seconds (16.74 k allocations: 2.660 MiB)
 22.797334 seconds (9.23 M allocations: 647.323 MiB, 0.81% gc time, 99.52% 
compilation time)
(n, ts) = (4, [0.012290934, 0.006391111, 0.01943052, 0.007879672])
  9.429176 seconds (3.61 M allocations: 252.252 MiB, 1.33% gc time, 99.53% 
compilation time)
  0.011352 seconds (9.95 k allocations: 4.073 MiB)
  9.414494 seconds (3.90 M allocations: 272.177 MiB, 0.77% gc time, 99.37% 
compilation time)
  0.012612 seconds (18.46 k allocations: 4.638 MiB)
 18.987890 seconds (7.60 M allocations: 550.233 MiB, 1.05% gc time, 98.70% 
compilation time)
(n, ts) = (5, [0.038742992, 0.011170402, 0.053924388, 0.012446813])
 12.096487 seconds (3.61 M allocations: 255.941 MiB, 0.56% gc time, 99.16% 
compilation time)
  0.019530 seconds (12.35 k allocations: 7.463 MiB)
 12.216798 seconds (3.88 M allocations: 274.202 MiB, 98.84% compilation tim
e)
  0.020788 seconds (23.93 k allocations: 8.291 MiB)
 24.630393 seconds (7.62 M allocations: 577.687 MiB, 0.27% gc time, 97.73% 
compilation time)
(n, ts) = (6, [0.097874428, 0.018615286, 0.135844663, 0.020572523])
 10.485550 seconds (3.54 M allocations: 256.884 MiB, 0.52% gc time, 98.27% 
compilation time)
  0.031963 seconds (15.13 k allocations: 12.913 MiB)
 10.737038 seconds (3.80 M allocations: 274.070 MiB, 0.64% gc time, 97.37% 
compilation time)
  0.034428 seconds (29.79 k allocations: 14.044 MiB)
 21.813655 seconds (7.51 M allocations: 613.111 MiB, 0.57% gc time, 95.16% 
compilation time)
(n, ts) = (7, [0.176610734, 0.031404756, 0.277786201, 0.034598287])
  0.742689 seconds (17.93 k allocations: 21.923 MiB)
  0.159322 seconds (18.38 k allocations: 21.279 MiB)
  0.924576 seconds (83.23 k allocations: 25.142 MiB)
  0.129110 seconds (36.93 k allocations: 22.856 MiB)
  3.737359 seconds (313.84 k allocations: 182.562 MiB)
(n, ts) = (8, [0.744598349, 0.159918056, 0.745764141, 0.126261053])
  1.095668 seconds (21.51 k allocations: 34.909 MiB, 4.40% gc time)
  0.187652 seconds (22.06 k allocations: 33.402 MiB)
  1.436953 seconds (117.99 k allocations: 39.784 MiB)
  0.196002 seconds (42.87 k allocations: 35.347 MiB)
  5.812384 seconds (409.73 k allocations: 287.046 MiB, 0.83% gc time)
(n, ts) = (9, [1.054427462, 0.186995559, 1.442442664, 0.207392354])
  1.962139 seconds (26.60 k allocations: 53.342 MiB)
  0.394659 seconds (27.94 k allocations: 51.018 MiB, 23.92% gc time)
  2.585577 seconds (141.07 k allocations: 59.220 MiB)
  0.291054 seconds (49.70 k allocations: 53.246 MiB)
 10.333375 seconds (491.51 k allocations: 433.815 MiB, 0.91% gc time)
(n, ts) = (10, [1.968526828, 0.297287381, 2.536755872, 0.29093112])
  5.900019 seconds (37.58 k allocations: 110.775 MiB, 0.12% gc time)
  0.631928 seconds (36.92 k allocations: 103.043 MiB)
  7.755832 seconds (246.95 k allocations: 121.791 MiB, 0.52% gc time)
  0.790551 seconds (62.10 k allocations: 106.140 MiB)
 30.338125 seconds (768.00 k allocations: 883.660 MiB, 0.32% gc time)
(n, ts) = (12, [5.932253892, 0.628329804, 7.83494398, 0.854342215])
 25.767299 seconds (65.76 k allocations: 284.738 MiB)
  1.518842 seconds (56.98 k allocations: 256.956 MiB, 2.51% gc time)
 30.275388 seconds (418.12 k allocations: 303.745 MiB, 0.01% gc time)
  1.427760 seconds (89.60 k allocations: 262.210 MiB)
118.448212 seconds (1.26 M allocations: 2.164 GiB, 0.25% gc time)
(n, ts) = (15, [25.274117708, 1.457432407, 31.234333421, 1.481579811])
 42.622180 seconds (66.59 k allocations: 452.073 MiB, 0.15% gc time)
  3.003821 seconds (71.83 k allocations: 422.756 MiB, 1.58% gc time)
 52.370282 seconds (574.65 k allocations: 479.595 MiB, 0.08% gc time)
  3.059337 seconds (107.12 k allocations: 429.493 MiB, 1.55% gc time)
202.515248 seconds (1.64 M allocations: 3.484 GiB, 0.17% gc time)
(n, ts) = (17, [43.086275282, 3.050466152, 52.338010853, 2.963527885])
12-element Vector{Vector{Float64}}:
 [0.00155588, 0.002052917, 0.001893388, 0.002240766]
 [0.004682731, 0.003433098, 0.00650831, 0.004363293]
 [0.012290934, 0.006391111, 0.01943052, 0.007879672]
 [0.038742992, 0.011170402, 0.053924388, 0.012446813]
 [0.097874428, 0.018615286, 0.135844663, 0.020572523]
 [0.176610734, 0.031404756, 0.277786201, 0.034598287]
 [0.744598349, 0.159918056, 0.745764141, 0.126261053]
 [1.054427462, 0.186995559, 1.442442664, 0.207392354]
 [1.968526828, 0.297287381, 2.536755872, 0.29093112]
 [5.932253892, 0.628329804, 7.83494398, 0.854342215]
 [25.274117708, 1.457432407, 31.234333421, 1.481579811]
 [43.086275282, 3.050466152, 52.338010853, 2.963527885]
```



```julia
n_to_param(n) = 4n^2

lw = 2
ms = 0.5
plt1 = plot(title = "Sensitivity Scaling on Brusselator");
plot!(plt1, n_to_param.(forwarddiffn), forwarddiff, lab = "Forward-Mode DSAAD",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
#plot!(plt1, n_to_param.(reversediffn), reversediff, lab="Reverse-Mode DSAAD", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
csadata_iq = [[csa_iq[j][i] for j in eachindex(csa_iq)] for i in eachindex(csa_iq[1])]
csadata_g = [[csa_g[j][i] for j in eachindex(csa_g)] for i in eachindex(csa_g[1])]
plot!(plt1, n_to_param.(csan), csadata_iq[1], lab = "Interpolating CASA user-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(csan), csadata_iq[2], lab = "Interpolating CASA AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(
    plt1, n_to_param.(csan), csadata_iq[3], lab = raw"Interpolating CASA AD-$v^{T}J$ seeding",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(csan), csadata_iq[1 + 3], lab = "Quadrature CASA user-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(csan), csadata_iq[2 + 3], lab = "Quadrature CASA AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(
    plt1, n_to_param.(csan), csadata_iq[3 + 3], lab = raw"Quadrature CASA AD-$v^{T}J$ seeding",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(csan), csadata_g[1], lab = "Gauss CASA AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(
    plt1, n_to_param.(csan), csadata_g[2], lab = raw"Gauss CASA AD-$v^{T}J$ seeding",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(csan), csadata_g[1 + 2], lab = "GaussKronrod CASA AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(
    plt1, n_to_param.(csan), csadata_g[2 + 2], lab = raw"GaussKronrod CASA AD-$v^{T}J$ seeding",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt1, n_to_param.(numdiffn), numdiff, lab = "Numerical Differentiation",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
xaxis!(plt1, "Number of Parameters", :log10);
yaxis!(plt1, "Runtime (s)", :log10);
plot!(plt1, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_10_1.png)



## VJP Choice Benchmarks

```julia
bt = 0:0.1:1
tspan = (0.0, 1.0)
csan = vcat(2:10, 12, 15, 17)
tols = (abstol = 1e-5, reltol = 1e-7)

_adjoint_methods = ntuple(4) do ii
    Alg = (InterpolatingAdjoint, QuadratureAdjoint, GaussAdjoint, GaussKronrodAdjoint)[ii]
    (
        advj1 = Alg(autodiff = true, autojacvec = EnzymeVJP()), # AD vJ (Enzyme)
        advj2 = Alg(autodiff = true, autojacvec = ReverseDiffVJP(false)), # AD vJ (ReverseDiff)
        advj3 = Alg(autodiff = true, autojacvec = ReverseDiffVJP(true)), # AD vJ (Compiled ReverseDiff)
        advj4 = Alg(autodiff = true, autojacvec = SciMLSensitivity.MooncakeVJP()) # AD vJ (Mooncake)
    )
end |> NamedTuple{(:interp, :quad, :gauss, :gausskronrod)}
adjoint_methods = mapreduce(collect, vcat, _adjoint_methods)
```

```
16-element Vector{SciMLBase.AbstractAdjointSensitivityAlgorithm{0, true, Va
l{:central}}}:
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensiti
vity.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIAB
I, false, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false,
 false, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{
false, false, false, EnzymeCore.FFIABI, false, false}()), false, false)
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensiti
vity.ReverseDiffVJP{false}}(SciMLSensitivity.ReverseDiffVJP{false}(), false
, false)
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensiti
vity.ReverseDiffVJP{true}}(SciMLSensitivity.ReverseDiffVJP{true}(), false, 
false)
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensiti
vity.MooncakeVJP}(SciMLSensitivity.MooncakeVJP(), false, false)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, SciMLSensitivit
y.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIABI, 
false, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false, fa
lse, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{fal
se, false, false, EnzymeCore.FFIABI, false, false}()), 1.0e-6, 0.001)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, SciMLSensitivit
y.ReverseDiffVJP{false}}(SciMLSensitivity.ReverseDiffVJP{false}(), 1.0e-6, 
0.001)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, SciMLSensitivit
y.ReverseDiffVJP{true}}(SciMLSensitivity.ReverseDiffVJP{true}(), 1.0e-6, 0.
001)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, SciMLSensitivit
y.MooncakeVJP}(SciMLSensitivity.MooncakeVJP(), 1.0e-6, 0.001)
 SciMLSensitivity.GaussAdjoint{0, true, Val{:central}, SciMLSensitivity.Enz
ymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIABI, false
, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false, false, 
false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{false, f
alse, false, EnzymeCore.FFIABI, false, false}()), false)
 SciMLSensitivity.GaussAdjoint{0, true, Val{:central}, SciMLSensitivity.Rev
erseDiffVJP{false}}(SciMLSensitivity.ReverseDiffVJP{false}(), false)
 SciMLSensitivity.GaussAdjoint{0, true, Val{:central}, SciMLSensitivity.Rev
erseDiffVJP{true}}(SciMLSensitivity.ReverseDiffVJP{true}(), false)
 SciMLSensitivity.GaussAdjoint{0, true, Val{:central}, SciMLSensitivity.Moo
ncakeVJP}(SciMLSensitivity.MooncakeVJP(), false)
 SciMLSensitivity.GaussKronrodAdjoint{0, true, Val{:central}, SciMLSensitiv
ity.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIABI
, false, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false, 
false, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{f
alse, false, false, EnzymeCore.FFIABI, false, false}()), false)
 SciMLSensitivity.GaussKronrodAdjoint{0, true, Val{:central}, SciMLSensitiv
ity.ReverseDiffVJP{false}}(SciMLSensitivity.ReverseDiffVJP{false}(), false)
 SciMLSensitivity.GaussKronrodAdjoint{0, true, Val{:central}, SciMLSensitiv
ity.ReverseDiffVJP{true}}(SciMLSensitivity.ReverseDiffVJP{true}(), false)
 SciMLSensitivity.GaussKronrodAdjoint{0, true, Val{:central}, SciMLSensitiv
ity.MooncakeVJP}(SciMLSensitivity.MooncakeVJP(), false)
```





Warmup: compile all VJP backends before benchmarking.

```julia
let n = first(csan)
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    solver = Rodas5(autodiff = false)
    for alg in adjoint_methods
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction(bfun, jac = brusselator_jac)
        diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
    end
end
```


```julia
csavjp = map(csan) do n
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
    @time ts = map(adjoint_methods) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction(bfun, jac = brusselator_jac)
        solver = Rodas5(autodiff = false)
        @time diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        t = @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...)
        return t
    end
    @show n, ts
    ts
end
```

```
0.003556 seconds (6.28 k allocations: 976.500 KiB)
  0.070375 seconds (721.33 k allocations: 31.414 MiB)
  0.008850 seconds (3.84 k allocations: 530.109 KiB)
  1.149218 seconds (1.80 M allocations: 103.504 MiB, 5.80% gc time, 72.68% 
compilation time)
  0.002042 seconds (6.78 k allocations: 668.500 KiB)
  0.036964 seconds (361.35 k allocations: 15.712 MiB)
  0.005662 seconds (4.61 k allocations: 363.984 KiB)
  0.002693 seconds (7.32 k allocations: 687.641 KiB)
  0.002209 seconds (5.38 k allocations: 621.141 KiB)
  0.036899 seconds (362.05 k allocations: 15.747 MiB)
  0.005588 seconds (5.32 k allocations: 401.547 KiB)
  0.002910 seconds (7.68 k allocations: 703.016 KiB)
  0.002405 seconds (7.51 k allocations: 738.625 KiB)
  0.039941 seconds (363.78 k allocations: 15.808 MiB)
  0.006333 seconds (7.05 k allocations: 463.344 KiB)
  0.003171 seconds (10.21 k allocations: 830.812 KiB)
  3.252416 seconds (7.11 M allocations: 390.069 MiB, 3.48% gc time, 75.65% 
compilation time)
(n, ts) = (2, [0.002956111, 0.07000238, 0.008435328, 0.003911405, 0.0017003
8, 0.039095329, 0.005115709, 0.002225376, 0.001877869, 0.03596072, 0.005129
168, 0.002434825, 0.002094097, 0.039323489, 0.005926833, 0.002760703])
  0.006244 seconds (8.44 k allocations: 2.089 MiB)
  0.186882 seconds (2.10 M allocations: 96.852 MiB)
  0.023132 seconds (5.88 k allocations: 1.508 MiB)
  0.009339 seconds (13.55 k allocations: 2.263 MiB)
  0.003010 seconds (7.87 k allocations: 1.062 MiB)
  0.106586 seconds (1.10 M allocations: 50.511 MiB)
  0.013706 seconds (7.78 k allocations: 816.922 KiB)
  0.005060 seconds (9.21 k allocations: 1.164 MiB)
  0.003772 seconds (6.14 k allocations: 1.099 MiB)
  0.131436 seconds (1.03 M allocations: 47.528 MiB, 24.67% gc time)
  0.013493 seconds (8.38 k allocations: 971.312 KiB)
  0.005515 seconds (9.00 k allocations: 1.256 MiB)
  0.004240 seconds (11.92 k allocations: 1.430 MiB)
  0.121096 seconds (1.04 M allocations: 47.701 MiB)
  0.017942 seconds (13.00 k allocations: 1.122 MiB)
  0.007113 seconds (15.94 k allocations: 1.617 MiB)
  1.342051 seconds (10.76 M allocations: 518.553 MiB, 5.92% gc time)
(n, ts) = (3, [0.005914443, 0.237452112, 0.023001139, 0.009124734, 0.002860
372, 0.106559426, 0.013443357, 0.00482963, 0.003567178, 0.096629166, 0.0132
20059, 0.005273697, 0.004101885, 0.120853318, 0.017582652, 0.006892718])
  0.012486 seconds (13.09 k allocations: 4.943 MiB)
  0.604335 seconds (6.24 M allocations: 271.949 MiB, 7.46% gc time)
  0.065022 seconds (9.02 k allocations: 3.971 MiB)
  0.023290 seconds (21.66 k allocations: 5.205 MiB)
  0.005142 seconds (9.71 k allocations: 1.804 MiB)
  0.296443 seconds (2.94 M allocations: 127.393 MiB, 7.29% gc time)
  0.033160 seconds (12.18 k allocations: 1.524 MiB)
  0.011167 seconds (12.42 k allocations: 1.942 MiB)
  0.006568 seconds (7.82 k allocations: 2.119 MiB)
  0.268390 seconds (2.71 M allocations: 117.900 MiB, 8.29% gc time)
  0.032265 seconds (12.76 k allocations: 1.972 MiB)
  0.011608 seconds (11.91 k allocations: 2.308 MiB)
  0.007629 seconds (16.74 k allocations: 2.660 MiB)
  0.312403 seconds (2.72 M allocations: 118.195 MiB)
  0.044086 seconds (19.88 k allocations: 2.267 MiB)
  0.015456 seconds (22.63 k allocations: 2.896 MiB)
  3.487499 seconds (29.56 M allocations: 1.307 GiB, 4.48% gc time)
(n, ts) = (4, [0.012320564, 0.576476821, 0.064738643, 0.023459845, 0.004985
609, 0.273030534, 0.033402915, 0.010837103, 0.00638885, 0.274554985, 0.0319
82343, 0.011554788, 0.007495564, 0.336360204, 0.043700492, 0.015139956])
  0.055829 seconds (18.81 k allocations: 10.368 MiB)
  1.387053 seconds (14.53 M allocations: 674.712 MiB, 4.98% gc time)
  0.178067 seconds (13.06 k allocations: 8.941 MiB)
  0.078768 seconds (31.60 k allocations: 10.738 MiB)
  0.010476 seconds (12.20 k allocations: 3.155 MiB)
  0.642252 seconds (6.64 M allocations: 306.415 MiB, 3.80% gc time)
  0.072242 seconds (17.93 k allocations: 2.876 MiB)
  0.021975 seconds (16.71 k allocations: 3.340 MiB)
  0.012242 seconds (9.95 k allocations: 4.073 MiB)
  0.557549 seconds (5.91 M allocations: 274.181 MiB, 3.92% gc time)
  0.066677 seconds (18.47 k allocations: 3.973 MiB)
  0.022776 seconds (15.53 k allocations: 4.303 MiB)
  0.012757 seconds (18.46 k allocations: 4.638 MiB)
  0.655606 seconds (5.92 M allocations: 274.511 MiB, 3.34% gc time)
  0.085714 seconds (25.26 k allocations: 4.303 MiB)
  0.027681 seconds (25.74 k allocations: 4.911 MiB)
  7.937286 seconds (66.42 M allocations: 3.117 GiB, 5.14% gc time)
(n, ts) = (5, [0.048721391, 1.492374997, 0.175694902, 0.079556382, 0.009580
301, 0.657643143, 0.070964374, 0.021578838, 0.011566109, 0.576995007, 0.066
065715, 0.022049684, 0.012645242, 0.674699808, 0.084494322, 0.027074694])
  0.096882 seconds (25.70 k allocations: 19.800 MiB)
  2.876938 seconds (29.31 M allocations: 1.275 GiB, 8.46% gc time)
  0.423263 seconds (17.89 k allocations: 17.802 MiB, 11.80% gc time)
  0.159500 seconds (43.63 k allocations: 20.309 MiB)
  0.015257 seconds (14.93 k allocations: 5.270 MiB)
  1.265786 seconds (12.89 M allocations: 570.523 MiB, 6.46% gc time)
  0.136315 seconds (24.79 k allocations: 4.982 MiB)
  0.064993 seconds (21.49 k allocations: 5.524 MiB, 37.96% gc time)
  0.019110 seconds (12.35 k allocations: 7.463 MiB)
  1.102746 seconds (11.41 M allocations: 507.990 MiB, 6.00% gc time)
  0.127434 seconds (25.26 k allocations: 7.389 MiB)
  0.040865 seconds (19.70 k allocations: 7.754 MiB)
  0.022088 seconds (23.93 k allocations: 8.291 MiB)
  1.260701 seconds (11.42 M allocations: 508.497 MiB, 3.84% gc time)
  0.162743 seconds (34.50 k allocations: 7.896 MiB)
  0.050359 seconds (33.63 k allocations: 8.642 MiB)
 15.665363 seconds (130.67 M allocations: 5.887 GiB, 6.55% gc time)
(n, ts) = (6, [0.094890797, 2.906632916, 0.382524752, 0.156066552, 0.014904
678, 1.229216254, 0.134205526, 0.038270785, 0.018529446, 1.115578571, 0.128
66745, 0.040648401, 0.022126304, 1.329708196, 0.160537525, 0.049736285])
  0.189070 seconds (35.11 k allocations: 37.172 MiB)
  5.872621 seconds (55.56 M allocations: 2.356 GiB, 9.59% gc time)
  1.015217 seconds (23.91 k allocations: 34.366 MiB)
  0.468196 seconds (60.12 k allocations: 37.980 MiB, 12.88% gc time)
  0.038032 seconds (18.16 k allocations: 8.466 MiB)
  2.290018 seconds (22.91 M allocations: 987.645 MiB, 7.52% gc time)
  0.238363 seconds (32.90 k allocations: 8.166 MiB)
  0.067110 seconds (27.18 k allocations: 9.023 MiB)
  0.033346 seconds (15.13 k allocations: 12.913 MiB)
  1.958492 seconds (20.15 M allocations: 874.307 MiB, 7.23% gc time)
  0.223959 seconds (33.27 k allocations: 12.873 MiB)
  0.069738 seconds (24.62 k allocations: 13.498 MiB)
  0.036954 seconds (29.79 k allocations: 14.044 MiB)
  2.278073 seconds (20.16 M allocations: 875.020 MiB, 6.35% gc time)
  0.279849 seconds (44.79 k allocations: 13.586 MiB)
  0.084721 seconds (41.51 k allocations: 14.655 MiB)
 31.027769 seconds (238.34 M allocations: 10.481 GiB, 7.27% gc time)
(n, ts) = (7, [0.194302017, 6.530890629, 1.029414881, 0.402313001, 0.039004
59, 2.303255482, 0.235427405, 0.066875619, 0.035158954, 2.014178696, 0.2244
75751, 0.069409865, 0.038598963, 2.310818126, 0.278766139, 0.084014725])
  0.450539 seconds (44.84 k allocations: 61.664 MiB, 5.27% gc time)
 10.488421 seconds (93.72 M allocations: 4.233 GiB, 8.73% gc time)
  1.898518 seconds (30.57 k allocations: 58.072 MiB, 5.97% gc time)
  0.718186 seconds (77.14 k allocations: 62.659 MiB)
  0.114486 seconds (22.48 k allocations: 13.599 MiB)
  4.918451 seconds (39.30 M allocations: 1.763 GiB, 14.32% gc time)
  0.460522 seconds (42.43 k allocations: 13.287 MiB)
  0.157002 seconds (34.74 k allocations: 14.241 MiB)
  0.137048 seconds (18.38 k allocations: 21.279 MiB)
  3.569475 seconds (33.37 M allocations: 1.507 GiB, 11.89% gc time)
  0.443306 seconds (42.52 k allocations: 21.338 MiB)
  0.198594 seconds (30.30 k allocations: 21.930 MiB)
  0.135457 seconds (36.93 k allocations: 22.856 MiB)
  4.062776 seconds (33.39 M allocations: 1.508 GiB, 9.76% gc time)
  0.549605 seconds (56.49 k allocations: 22.342 MiB)
  0.226071 seconds (51.88 k allocations: 23.548 MiB)
 56.204667 seconds (400.54 M allocations: 18.719 GiB, 8.65% gc time)
(n, ts) = (8, [0.442940531, 10.364179507, 1.844122702, 0.718998257, 0.11469
6686, 4.045573901, 0.458894453, 0.15630949, 0.137756065, 3.717160348, 0.459
721728, 0.196764933, 0.144539193, 4.055479353, 0.562052531, 0.220804435])
  0.613117 seconds (61.84 k allocations: 103.098 MiB, 2.94% gc time)
 16.810451 seconds (159.93 M allocations: 7.025 GiB, 11.34% gc time)
  2.093023 seconds (39.11 k allocations: 96.129 MiB)
  0.870845 seconds (103.39 k allocations: 102.339 MiB)
  0.135099 seconds (26.84 k allocations: 20.817 MiB)
  6.612715 seconds (61.72 M allocations: 2.693 GiB, 12.14% gc time)
  0.708318 seconds (53.08 k allocations: 20.475 MiB)
  0.253756 seconds (42.36 k allocations: 21.546 MiB)
  0.208615 seconds (22.06 k allocations: 33.402 MiB)
  5.789740 seconds (52.34 M allocations: 2.299 GiB, 13.91% gc time)
  1.032176 seconds (53.01 k allocations: 33.501 MiB)
  0.331804 seconds (36.74 k allocations: 34.126 MiB)
  0.242194 seconds (42.87 k allocations: 35.347 MiB)
  6.641221 seconds (52.36 M allocations: 2.301 GiB, 11.88% gc time)
  1.183581 seconds (69.75 k allocations: 34.877 MiB)
  0.431582 seconds (62.02 k allocations: 36.198 MiB, 19.87% gc time)
 88.186390 seconds (653.95 M allocations: 29.754 GiB, 10.14% gc time)
(n, ts) = (9, [0.702004852, 17.023501063, 2.155600474, 0.873388461, 0.13904
2126, 6.548932309, 0.708274465, 0.249968236, 0.226448061, 6.07625648, 0.986
542989, 0.322438436, 0.225610961, 6.72464141, 0.889063379, 0.342259853])
  0.782425 seconds (70.60 k allocations: 146.135 MiB)
 25.309821 seconds (233.97 M allocations: 10.076 GiB, 10.24% gc time)
  3.760084 seconds (47.14 k allocations: 140.334 MiB, 7.24% gc time)
  1.308290 seconds (122.26 k allocations: 150.395 MiB, 1.97% gc time)
  0.193342 seconds (31.70 k allocations: 30.308 MiB)
 10.982931 seconds (92.73 M allocations: 3.966 GiB, 16.60% gc time)
  1.079120 seconds (64.97 k allocations: 29.933 MiB)
  0.372597 seconds (50.87 k allocations: 31.166 MiB)
  0.346107 seconds (27.94 k allocations: 51.018 MiB)
  9.214981 seconds (84.33 M allocations: 3.630 GiB, 10.46% gc time)
  1.157941 seconds (65.20 k allocations: 50.985 MiB)
  0.469027 seconds (46.98 k allocations: 51.889 MiB)
  0.315824 seconds (49.70 k allocations: 53.246 MiB)
 10.585530 seconds (84.34 M allocations: 3.631 GiB, 12.01% gc time)
  1.351643 seconds (83.35 k allocations: 52.678 MiB)
  0.512011 seconds (72.92 k allocations: 54.209 MiB)
136.115149 seconds (992.22 M allocations: 44.253 GiB, 10.37% gc time)
(n, ts) = (10, [0.795453563, 26.248263111, 3.351016396, 1.252118846, 0.1896
89507, 10.385685975, 1.181795646, 0.383199109, 0.331345953, 9.477168513, 1.
176044653, 0.467583073, 0.347081735, 10.896700161, 1.314518131, 0.528111216
])
  1.419366 seconds (103.68 k allocations: 303.808 MiB, 0.73% gc time)
 54.907321 seconds (498.86 M allocations: 22.545 GiB, 15.16% gc time)
  6.374570 seconds (67.78 k allocations: 295.208 MiB)
  2.520734 seconds (180.09 k allocations: 305.946 MiB, 1.70% gc time)
  0.395138 seconds (42.97 k allocations: 59.668 MiB)
 22.035393 seconds (188.69 M allocations: 8.473 GiB, 17.20% gc time)
  2.162533 seconds (92.52 k allocations: 59.350 MiB)
  1.071773 seconds (70.59 k allocations: 60.750 MiB, 9.42% gc time)
  0.830815 seconds (36.92 k allocations: 103.043 MiB)
 19.880572 seconds (165.54 M allocations: 7.484 GiB, 17.30% gc time)
  3.166942 seconds (92.17 k allocations: 103.300 MiB)
  0.977259 seconds (62.72 k allocations: 104.094 MiB)
  0.675315 seconds (62.10 k allocations: 106.140 MiB, 3.46% gc time)
 21.490447 seconds (165.57 M allocations: 7.486 GiB, 15.48% gc time)
  2.678490 seconds (114.83 k allocations: 105.987 MiB)
  1.106238 seconds (95.24 k allocations: 107.548 MiB)
284.263153 seconds (2.04 G allocations: 95.326 GiB, 13.29% gc time)
(n, ts) = (12, [1.392831413, 54.004006595, 6.384133275, 2.503062412, 0.3825
43352, 22.138029133, 2.190069557, 0.999599149, 0.891765023, 21.312989516, 2
.861038942, 0.983513775, 0.634455388, 22.047298924, 2.642204739, 1.15949795
4])
  3.902715 seconds (170.98 k allocations: 767.125 MiB, 16.12% gc time)
127.432643 seconds (1.25 G allocations: 53.932 GiB, 8.44% gc time)
 16.340440 seconds (106.03 k allocations: 738.594 MiB, 0.61% gc time)
  6.526728 seconds (307.39 k allocations: 785.726 MiB, 0.71% gc time)
  0.897508 seconds (63.71 k allocations: 143.222 MiB)
 53.438565 seconds (453.45 M allocations: 19.400 GiB, 17.36% gc time)
  6.293774 seconds (143.24 k allocations: 142.763 MiB, 0.34% gc time)
  2.286873 seconds (106.94 k allocations: 145.738 MiB, 2.28% gc time)
  1.469863 seconds (56.98 k allocations: 256.956 MiB, 0.52% gc time)
 43.360483 seconds (411.58 M allocations: 17.733 GiB, 11.80% gc time)
  5.501628 seconds (142.75 k allocations: 257.148 MiB, 2.30% gc time)
  2.549490 seconds (137.10 k allocations: 279.527 MiB, 0.34% gc time)
  1.354371 seconds (89.60 k allocations: 262.210 MiB, 0.39% gc time)
 50.457117 seconds (411.61 M allocations: 17.737 GiB, 13.96% gc time)
  6.069405 seconds (170.19 k allocations: 261.721 MiB, 0.34% gc time)
  2.701406 seconds (176.05 k allocations: 284.995 MiB, 0.16% gc time)
657.693555 seconds (5.06 G allocations: 226.054 GiB, 9.78% gc time)
(n, ts) = (15, [4.075047553, 132.642669972, 15.853916687, 5.989026574, 1.01
2260249, 44.882153874, 6.50579429, 2.122425575, 1.52080364, 45.467125804, 5
.401891868, 2.577413314, 1.394801024, 48.880328133, 6.063153998, 2.68021807
7])
  4.644983 seconds (197.65 k allocations: 1.160 GiB, 0.42% gc time)
200.311331 seconds (1.92 G allocations: 81.273 GiB, 8.68% gc time)
 24.892640 seconds (132.04 k allocations: 1.144 GiB, 4.65% gc time)
  8.418943 seconds (344.54 k allocations: 1.164 GiB, 0.28% gc time)
  1.352895 seconds (80.10 k allocations: 234.334 MiB)
 80.903004 seconds (743.41 M allocations: 31.181 GiB, 12.44% gc time)
  8.415267 seconds (183.31 k allocations: 233.763 MiB, 0.38% gc time)
  2.694162 seconds (135.62 k allocations: 237.176 MiB, 0.17% gc time)
  2.162823 seconds (71.83 k allocations: 422.756 MiB, 0.34% gc time)
 71.108967 seconds (674.54 M allocations: 28.497 GiB, 10.29% gc time)
  8.682781 seconds (182.43 k allocations: 422.995 MiB, 0.24% gc time)
  3.341924 seconds (123.82 k allocations: 425.525 MiB, 0.33% gc time)
  2.116381 seconds (107.12 k allocations: 429.493 MiB, 0.19% gc time)
 77.794764 seconds (674.57 M allocations: 28.503 GiB, 11.99% gc time)
  9.815087 seconds (210.85 k allocations: 428.807 MiB, 1.69% gc time)
  3.521824 seconds (166.79 k allocations: 432.522 MiB, 0.22% gc time)
1024.307609 seconds (8.04 G allocations: 352.226 GiB, 8.93% gc time)
(n, ts) = (17, [4.627666106, 201.834228399, 25.735837836, 8.592395924, 1.38
9558886, 75.464626005, 8.659579775, 2.763920353, 2.020713371, 74.943977418,
 8.73213247, 3.355430934, 2.067915925, 80.713957283, 9.56107125, 3.63245942
8])
12-element Vector{Vector{Float64}}:
 [0.002956111, 0.07000238, 0.008435328, 0.003911405, 0.00170038, 0.03909532
9, 0.005115709, 0.002225376, 0.001877869, 0.03596072, 0.005129168, 0.002434
825, 0.002094097, 0.039323489, 0.005926833, 0.002760703]
 [0.005914443, 0.237452112, 0.023001139, 0.009124734, 0.002860372, 0.106559
426, 0.013443357, 0.00482963, 0.003567178, 0.096629166, 0.013220059, 0.0052
73697, 0.004101885, 0.120853318, 0.017582652, 0.006892718]
 [0.012320564, 0.576476821, 0.064738643, 0.023459845, 0.004985609, 0.273030
534, 0.033402915, 0.010837103, 0.00638885, 0.274554985, 0.031982343, 0.0115
54788, 0.007495564, 0.336360204, 0.043700492, 0.015139956]
 [0.048721391, 1.492374997, 0.175694902, 0.079556382, 0.009580301, 0.657643
143, 0.070964374, 0.021578838, 0.011566109, 0.576995007, 0.066065715, 0.022
049684, 0.012645242, 0.674699808, 0.084494322, 0.027074694]
 [0.094890797, 2.906632916, 0.382524752, 0.156066552, 0.014904678, 1.229216
254, 0.134205526, 0.038270785, 0.018529446, 1.115578571, 0.12866745, 0.0406
48401, 0.022126304, 1.329708196, 0.160537525, 0.049736285]
 [0.194302017, 6.530890629, 1.029414881, 0.402313001, 0.03900459, 2.3032554
82, 0.235427405, 0.066875619, 0.035158954, 2.014178696, 0.224475751, 0.0694
09865, 0.038598963, 2.310818126, 0.278766139, 0.084014725]
 [0.442940531, 10.364179507, 1.844122702, 0.718998257, 0.114696686, 4.04557
3901, 0.458894453, 0.15630949, 0.137756065, 3.717160348, 0.459721728, 0.196
764933, 0.144539193, 4.055479353, 0.562052531, 0.220804435]
 [0.702004852, 17.023501063, 2.155600474, 0.873388461, 0.139042126, 6.54893
2309, 0.708274465, 0.249968236, 0.226448061, 6.07625648, 0.986542989, 0.322
438436, 0.225610961, 6.72464141, 0.889063379, 0.342259853]
 [0.795453563, 26.248263111, 3.351016396, 1.252118846, 0.189689507, 10.3856
85975, 1.181795646, 0.383199109, 0.331345953, 9.477168513, 1.176044653, 0.4
67583073, 0.347081735, 10.896700161, 1.314518131, 0.528111216]
 [1.392831413, 54.004006595, 6.384133275, 2.503062412, 0.382543352, 22.1380
29133, 2.190069557, 0.999599149, 0.891765023, 21.312989516, 2.861038942, 0.
983513775, 0.634455388, 22.047298924, 2.642204739, 1.159497954]
 [4.075047553, 132.642669972, 15.853916687, 5.989026574, 1.012260249, 44.88
2153874, 6.50579429, 2.122425575, 1.52080364, 45.467125804, 5.401891868, 2.
577413314, 1.394801024, 48.880328133, 6.063153998, 2.680218077]
 [4.627666106, 201.834228399, 25.735837836, 8.592395924, 1.389558886, 75.46
4626005, 8.659579775, 2.763920353, 2.020713371, 74.943977418, 8.73213247, 3
.355430934, 2.067915925, 80.713957283, 9.56107125, 3.632459428]
```



```julia
csacompare = [[csavjp[j][i] for j in eachindex(csavjp)] for i in eachindex(csavjp[1])]

plt_interp = plot(title = "Brusselator interpolating adjoint VJP scaling");
plot!(plt_interp, n_to_param.(csan), csadata_iq[2], lab = "AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_interp, n_to_param.(csan), csacompare[1], lab = raw"EnzymeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_interp, n_to_param.(csan), csacompare[2], lab = raw"ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_interp, n_to_param.(csan), csacompare[3], lab = raw"Compiled ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_interp, n_to_param.(csan), csacompare[4], lab = raw"MooncakeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
xaxis!(plt_interp, "Number of Parameters", :log10);
yaxis!(plt_interp, "Runtime (s)", :log10);
plot!(plt_interp, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_14_1.png)

```julia
plt2 = plot(title = "Brusselator quadrature adjoint VJP scaling");
plot!(plt2, n_to_param.(csan), csadata_iq[2 + 3], lab = "AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt2, n_to_param.(csan), csacompare[1 + 4], lab = raw"EnzymeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt2, n_to_param.(csan), csacompare[2 + 4], lab = raw"ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt2, n_to_param.(csan), csacompare[3 + 4], lab = raw"Compiled ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt2, n_to_param.(csan), csacompare[4 + 4], lab = raw"MooncakeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
xaxis!(plt2, "Number of Parameters", :log10);
yaxis!(plt2, "Runtime (s)", :log10);
plot!(plt2, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_15_1.png)

```julia
plt_gauss = plot(title = "Brusselator Gauss adjoint VJP scaling");
plot!(plt_gauss, n_to_param.(csan), csadata_g[1], lab = "AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gauss, n_to_param.(csan), csacompare[1 + 8], lab = raw"EnzymeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gauss, n_to_param.(csan), csacompare[2 + 8], lab = raw"ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gauss, n_to_param.(csan), csacompare[3 + 8], lab = raw"Compiled ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gauss, n_to_param.(csan), csacompare[4 + 8], lab = raw"MooncakeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
xaxis!(plt_gauss, "Number of Parameters", :log10);
yaxis!(plt_gauss, "Runtime (s)", :log10);
plot!(plt_gauss, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_16_1.png)

```julia
plt_gk = plot(title = "Brusselator GaussKronrod adjoint VJP scaling");
plot!(plt_gk, n_to_param.(csan), csadata_g[1 + 2], lab = "AD-Jacobian",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gk, n_to_param.(csan), csacompare[1 + 12], lab = raw"EnzymeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gk, n_to_param.(csan), csacompare[2 + 12], lab = raw"ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gk, n_to_param.(csan), csacompare[3 + 12], lab = raw"Compiled ReverseDiffVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
plot!(plt_gk, n_to_param.(csan), csacompare[4 + 12], lab = raw"MooncakeVJP",
    lw = lw, marksize = ms, linestyle = :auto, marker = :auto);
xaxis!(plt_gk, "Number of Parameters", :log10);
yaxis!(plt_gk, "Runtime (s)", :log10);
plot!(plt_gk, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_17_1.png)



## Peak Memory Benchmarks

Measures the memory consumed by each sensitivity computation. Each configuration runs
in a separate subprocess. We measure current RSS (from `/proc/self/statm`) before and
after the computation to isolate the memory used by the sensitivity solve from the
large fixed cost of package loading.

```julia
const CHILD_PREAMBLE = raw"""
using OrdinaryDiffEq, ReverseDiff, ForwardDiff, FiniteDiff, SciMLSensitivity
using LinearAlgebra, Mooncake

function get_rss_mib()
    statm = read("/proc/self/statm", String)
    resident_pages = parse(Int, split(statm)[2])
    return resident_pages * 4096 / (1024^2)
end

function makebrusselator(N = 8)
    xyd_brusselator = range(0, stop = 1, length = N)
    function limit(a, N)
        if a == N+1
            return 1
        elseif a == 0
            return N
        else
            return a
        end
    end
    brusselator_f(x, y, t) = ifelse(
        (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
        (t >= 1.1), 5.0, 0.0)
    brusselator_2d_loop = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
        function brusselator_2d_loop(du, u, p, t)
            @inbounds begin
                ii1 = N^2
                ii2 = ii1+N^2
                ii3 = ii2+2(N^2)
                A = @view p[1:ii1]
                B = @view p[(ii1 + 1):ii2]
                α = @view p[(ii2 + 1):ii3]
                II = LinearIndices((N, N, 2))
                for I in CartesianIndices((N, N))
                    x = xyd[I[1]]
                    y = xyd[I[2]]
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N);
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N);
                    jm1 = limit(j-1, N)
                    du[II[i, j, 1]] = α[II[
                                          i, j, 1]]*(u[II[im1, j, 1]] + u[II[ip1, j, 1]] +
                                                     u[II[i, jp1, 1]] + u[II[i, jm1, 1]] -
                                                     4u[II[i, j, 1]])/dx^2 +
                                      B[II[i, j, 1]] + u[II[i, j, 1]]^2*u[II[i, j, 2]] -
                                      (A[II[i, j, 1]] + 1)*u[II[i, j, 1]] +
                                      brusselator_f(x, y, t)
                end
                for I in CartesianIndices((N, N))
                    i = I[1]
                    j = I[2]
                    ip1 = limit(i+1, N)
                    im1 = limit(i-1, N)
                    jp1 = limit(j+1, N)
                    jm1 = limit(j-1, N)
                    du[II[i, j, 2]] = α[II[
                        i, j, 2]]*(u[II[im1, j, 2]] + u[II[ip1, j, 2]] + u[II[i, jp1, 2]] +
                                   u[II[i, jm1, 2]] - 4u[II[i, j, 2]])/dx^2 +
                                      A[II[i, j, 1]]*u[II[i, j, 1]] -
                                      u[II[i, j, 1]]^2*u[II[i, j, 2]]
                end
                return nothing
            end
        end
    end
    function init_brusselator_2d(xyd)
        N = length(xyd)
        u = zeros(N, N, 2)
        for I in CartesianIndices((N, N))
            x = xyd[I[1]]
            y = xyd[I[2]]
            u[I, 1] = 22*(y*(1-y))^(3/2)
            u[I, 2] = 27*(x*(1-x))^(3/2)
        end
        vec(u)
    end
    dx = step(xyd_brusselator)
    e1 = ones(N-1)
    off = N-1
    e4 = ones(N-off)
    T = diagm(0=>-2ones(N), -1=>e1, 1=>e1, off=>e4, -off=>e4) ./ dx^2
    Ie = Matrix{Float64}(I, N, N)
    Op = kron(Ie, T) + kron(T, Ie)
    brusselator_jac = let N=N
        (J, a, p, t) -> begin
            ii1 = N^2
            ii2 = ii1+N^2
            ii3 = ii2+2(N^2)
            A = @view p[1:ii1]
            B = @view p[(ii1 + 1):ii2]
            α = @view p[(ii2 + 1):ii3]
            u = @view a[1:(end ÷ 2)]
            v = @view a[(end ÷ 2 + 1):end]
            N2 = length(a)÷2
            α1 = @view α[1:(end ÷ 2)]
            α2 = @view α[(end ÷ 2 + 1):end]
            fill!(J, 0)
            J[1:N2, 1:N2] .= α1 .* Op
            J[(N2 + 1):end, (N2 + 1):end] .= α2 .* Op
            J1 = @view J[1:N2, 1:N2]
            J2 = @view J[(N2 + 1):end, 1:N2]
            J3 = @view J[1:N2, (N2 + 1):end]
            J4 = @view J[(N2 + 1):end, (N2 + 1):end]
            J1[diagind(J1)] .+= @. 2u*v-(A+1)
            J2[diagind(J2)] .= @. A-2u*v
            J3[diagind(J3)] .= @. u^2
            J4[diagind(J4)] .+= @. -u^2
            nothing
        end
    end
    u0 = init_brusselator_2d(xyd_brusselator)
    p = [fill(3.4, N^2); fill(1.0, N^2); fill(10.0, 2*N^2)]
    brusselator_2d_loop, u0, p, brusselator_jac
end

Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v')

bt = 0:0.1:1
tspan = (0.0, 1.0)
tols = (abstol = 1e-5, reltol = 1e-7)

function auto_sen_l2(
        f, u0, tspan, p, t, alg = Tsit5(); diffalg = ReverseDiff.gradient, kwargs...)
    test_f(p) = begin
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, convert.(eltype(p), u0), tspan, p)
        sol = solve(prob, alg, saveat = t; kwargs...)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    diffalg(test_f, p)
end

@inline function diffeq_sen_l2(df, u0, tspan, p, t, alg = Tsit5();
        abstol = 1e-5, reltol = 1e-7, iabstol = abstol, ireltol = reltol,
        sensalg = SensitivityAlg(), kwargs...)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(df, u0, tspan, p)
    saveat = tspan[1] != t[1] && tspan[end] != t[end] ? vcat(tspan[1], t, tspan[end]) : t
    sol = solve(prob, alg, abstol = abstol, reltol = reltol, saveat = saveat; kwargs...)
    dg(out, u, p, t, i) = (out.=u .- 1.0)
    adjoint_sensitivities(sol, alg; t, abstol = abstol, dgdu_discrete = dg,
        reltol = reltol, sensealg = sensalg)
end
"""

const PROJECT_DIR = @__DIR__

function run_memory_benchmark(n::Int, method_setup::String)
    child_script = CHILD_PREAMBLE * """

    n = $(n)
    bfun, b_u0, b_p, brusselator_jac = makebrusselator(n)

    GC.gc(); GC.gc()
    rss_before = get_rss_mib()

    """ * method_setup * """

    GC.gc(); GC.gc()
    rss_after = get_rss_mib()

    println("BRUSSMEM_TIMING:", t)
    println("BRUSSMEM_RSS_BEFORE:", rss_before)
    println("BRUSSMEM_RSS_AFTER:", rss_after)
    """

    try
        output = read(
            `$(Base.julia_cmd()) --project=$(PROJECT_DIR) -e $(child_script)`, String)
        time_m = match(r"BRUSSMEM_TIMING:([\d.eE+-]+)", output)
        before_m = match(r"BRUSSMEM_RSS_BEFORE:([\d.eE+-]+)", output)
        after_m = match(r"BRUSSMEM_RSS_AFTER:([\d.eE+-]+)", output)
        if time_m === nothing || before_m === nothing || after_m === nothing
            @warn "Failed to parse subprocess output" n output
            return (; rss_before = NaN, rss_after = NaN, delta_mib = NaN, timing = NaN)
        end
        timing = parse(Float64, time_m.captures[1])
        rss_before = parse(Float64, before_m.captures[1])
        rss_after = parse(Float64, after_m.captures[1])
        delta_mib = rss_after - rss_before
        return (; rss_before, rss_after, delta_mib, timing)
    catch e
        @warn "Subprocess failed" n exception = (e, catch_backtrace())
        return (; rss_before = NaN, rss_after = NaN, delta_mib = NaN, timing = NaN)
    end
end

mem_sizes = [2, 4, 6, 8, 10, 12]
```

```
6-element Vector{Int64}:
  2
  4
  6
  8
 10
 12
```



```julia
forwarddiff_mem = map(mem_sizes) do n
    result = run_memory_benchmark(n, """
    auto_sen_l2(bfun, b_u0, tspan, b_p, bt, Rodas5();
        diffalg = ForwardDiff.gradient, tols...)
    t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, Rodas5();
        diffalg = ForwardDiff.gradient, tols...)
    """)
    @show n, result
    result
end
```

```
(n, result) = (2, (rss_before = 962.01171875, rss_after = 1045.23828125, de
lta_mib = 83.2265625, timing = 0.001930138))
(n, result) = (4, (rss_before = 957.57421875, rss_after = 1005.6484375, del
ta_mib = 48.07421875, timing = 0.10433573))
(n, result) = (6, (rss_before = 1053.41796875, rss_after = 1080.11328125, d
elta_mib = 26.6953125, timing = 1.141785002))
(n, result) = (8, (rss_before = 963.19140625, rss_after = 1037.0390625, del
ta_mib = 73.84765625, timing = 11.166201725))
(n, result) = (10, (rss_before = 1053.9453125, rss_after = 1085.046875, del
ta_mib = 31.1015625, timing = 29.51626992))
(n, result) = (12, (rss_before = 957.15234375, rss_after = 1002.76953125, d
elta_mib = 45.6171875, timing = 103.037990436))
6-element Vector{@NamedTuple{rss_before::Float64, rss_after::Float64, delta
_mib::Float64, timing::Float64}}:
 (rss_before = 962.01171875, rss_after = 1045.23828125, delta_mib = 83.2265
625, timing = 0.001930138)
 (rss_before = 957.57421875, rss_after = 1005.6484375, delta_mib = 48.07421
875, timing = 0.10433573)
 (rss_before = 1053.41796875, rss_after = 1080.11328125, delta_mib = 26.695
3125, timing = 1.141785002)
 (rss_before = 963.19140625, rss_after = 1037.0390625, delta_mib = 73.84765
625, timing = 11.166201725)
 (rss_before = 1053.9453125, rss_after = 1085.046875, delta_mib = 31.101562
5, timing = 29.51626992)
 (rss_before = 957.15234375, rss_after = 1002.76953125, delta_mib = 45.6171
875, timing = 103.037990436)
```



```julia
numdiff_mem = map(mem_sizes) do n
    result = run_memory_benchmark(n, """
    auto_sen_l2(bfun, b_u0, tspan, b_p, bt, Rodas5();
        diffalg = FiniteDiff.finite_difference_gradient, tols...)
    t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, Rodas5();
        diffalg = FiniteDiff.finite_difference_gradient, tols...)
    """)
    @show n, result
    result
end
```

```
(n, result) = (2, (rss_before = 1003.13671875, rss_after = 1009.1171875, de
lta_mib = 5.98046875, timing = 0.004432052))
(n, result) = (4, (rss_before = 954.3515625, rss_after = 961.0234375, delta
_mib = 6.671875, timing = 0.150151551))
(n, result) = (6, (rss_before = 1005.01171875, rss_after = 1035.86328125, d
elta_mib = 30.8515625, timing = 1.112846577))
(n, result) = (8, (rss_before = 999.11328125, rss_after = 1005.1875, delta_
mib = 6.07421875, timing = 21.81582065))
(n, result) = (10, (rss_before = 1003.14453125, rss_after = 1008.8046875, d
elta_mib = 5.66015625, timing = 88.270882306))
(n, result) = (12, (rss_before = 997.59375, rss_after = 1005.8125, delta_mi
b = 8.21875, timing = 251.962559874))
6-element Vector{@NamedTuple{rss_before::Float64, rss_after::Float64, delta
_mib::Float64, timing::Float64}}:
 (rss_before = 1003.13671875, rss_after = 1009.1171875, delta_mib = 5.98046
875, timing = 0.004432052)
 (rss_before = 954.3515625, rss_after = 961.0234375, delta_mib = 6.671875, 
timing = 0.150151551)
 (rss_before = 1005.01171875, rss_after = 1035.86328125, delta_mib = 30.851
5625, timing = 1.112846577)
 (rss_before = 999.11328125, rss_after = 1005.1875, delta_mib = 6.07421875,
 timing = 21.81582065)
 (rss_before = 1003.14453125, rss_after = 1008.8046875, delta_mib = 5.66015
625, timing = 88.270882306)
 (rss_before = 997.59375, rss_after = 1005.8125, delta_mib = 8.21875, timin
g = 251.962559874)
```



```julia
adjoint_ad_configs = [
    ("Interp user-Jacobian",
     "InterpolatingAdjoint(autodiff = false, autojacvec = false)", true),
    ("Interp AD-Jacobian",
     "InterpolatingAdjoint(autodiff = true, autojacvec = false)", false),
    ("Quad user-Jacobian",
     "QuadratureAdjoint(autodiff = false, autojacvec = false)", true),
    ("Quad AD-Jacobian",
     "QuadratureAdjoint(autodiff = true, autojacvec = false)", false),
    ("Gauss AD-Jacobian",
     "GaussAdjoint(autodiff = true, autojacvec = false)", false),
    ("GaussKronrod AD-Jacobian",
     "GaussKronrodAdjoint(autodiff = true, autojacvec = false)", false),
]

adjoint_ad_mem = map(adjoint_ad_configs) do (name, sensalg_str, needs_jac)
    results = map(mem_sizes) do n
        f_expr = needs_jac ? "ODEFunction(bfun, jac = brusselator_jac)" : "bfun"
        result = run_memory_benchmark(n, """
        sensalg = $(sensalg_str)
        f = $(f_expr)
        solver = Rodas5(autodiff = false)
        diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = sensalg, tols...)
        t = @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver;
            sensalg = sensalg, tols...)
        """)
        @show name, n, result
        result
    end
    (name = name, results = results)
end
```

```
(name, n, result) = ("Interp user-Jacobian", 2, (rss_before = 982.83203125,
 rss_after = 1026.7890625, delta_mib = 43.95703125, timing = 0.006102091))
(name, n, result) = ("Interp user-Jacobian", 4, (rss_before = 1004.57421875
, rss_after = 1063.515625, delta_mib = 58.94140625, timing = 0.116232686))
(name, n, result) = ("Interp user-Jacobian", 6, (rss_before = 990.10546875,
 rss_after = 1040.24609375, delta_mib = 50.140625, timing = 1.838037656))
(name, n, result) = ("Interp user-Jacobian", 8, (rss_before = 963.20703125,
 rss_after = 1047.953125, delta_mib = 84.74609375, timing = 7.762733207))
(name, n, result) = ("Interp user-Jacobian", 10, (rss_before = 983.37109375
, rss_after = 1090.78125, delta_mib = 107.41015625, timing = 24.884803903))
(name, n, result) = ("Interp user-Jacobian", 12, (rss_before = 982.68359375
, rss_after = 1103.41015625, delta_mib = 120.7265625, timing = 72.31769302)
)
(name, n, result) = ("Interp AD-Jacobian", 2, (rss_before = 1003.33203125, 
rss_after = 1049.81640625, delta_mib = 46.484375, timing = 0.003599298))
(name, n, result) = ("Interp AD-Jacobian", 4, (rss_before = 963.328125, rss
_after = 999.3046875, delta_mib = 35.9765625, timing = 0.052426451))
(name, n, result) = ("Interp AD-Jacobian", 6, (rss_before = 955.21484375, r
ss_after = 1032.515625, delta_mib = 77.30078125, timing = 0.930814869))
(name, n, result) = ("Interp AD-Jacobian", 8, (rss_before = 963.3984375, rs
s_after = 1014.140625, delta_mib = 50.7421875, timing = 4.375893011))
(name, n, result) = ("Interp AD-Jacobian", 10, (rss_before = 962.75390625, 
rss_after = 1024.625, delta_mib = 61.87109375, timing = 13.870388276))
(name, n, result) = ("Interp AD-Jacobian", 12, (rss_before = 962.4375, rss_
after = 1142.87890625, delta_mib = 180.44140625, timing = 41.715235378))
(name, n, result) = ("Quad user-Jacobian", 2, (rss_before = 960.01171875, r
ss_after = 1007.4453125, delta_mib = 47.43359375, timing = 0.002169866))
(name, n, result) = ("Quad user-Jacobian", 4, (rss_before = 963.265625, rss
_after = 1033.28515625, delta_mib = 70.01953125, timing = 0.01129018))
(name, n, result) = ("Quad user-Jacobian", 6, (rss_before = 992.24609375, r
ss_after = 1097.734375, delta_mib = 105.48828125, timing = 0.05117379))
(name, n, result) = ("Quad user-Jacobian", 8, (rss_before = 1001.1640625, r
ss_after = 1040.55859375, delta_mib = 39.39453125, timing = 0.392806672))
(name, n, result) = ("Quad user-Jacobian", 10, (rss_before = 969.625, rss_a
fter = 1035.28515625, delta_mib = 65.66015625, timing = 0.91595626))
(name, n, result) = ("Quad user-Jacobian", 12, (rss_before = 960.76171875, 
rss_after = 1056.31640625, delta_mib = 95.5546875, timing = 2.271826671))
(name, n, result) = ("Quad AD-Jacobian", 2, (rss_before = 961.3515625, rss_
after = 1061.40625, delta_mib = 100.0546875, timing = 0.00162047))
(name, n, result) = ("Quad AD-Jacobian", 4, (rss_before = 963.078125, rss_a
fter = 1003.5859375, delta_mib = 40.5078125, timing = 0.012236963))
(name, n, result) = ("Quad AD-Jacobian", 6, (rss_before = 962.9140625, rss_
after = 1045.94140625, delta_mib = 83.02734375, timing = 0.105513332))
(name, n, result) = ("Quad AD-Jacobian", 8, (rss_before = 988.9609375, rss_
after = 1041.796875, delta_mib = 52.8359375, timing = 0.870751881))
(name, n, result) = ("Quad AD-Jacobian", 10, (rss_before = 984.75, rss_afte
r = 1061.81640625, delta_mib = 77.06640625, timing = 2.982462034))
(name, n, result) = ("Quad AD-Jacobian", 12, (rss_before = 963.4375, rss_af
ter = 1005.89453125, delta_mib = 42.45703125, timing = 7.249959207))
(name, n, result) = ("Gauss AD-Jacobian", 2, (rss_before = 1016.671875, rss
_after = 1068.94921875, delta_mib = 52.27734375, timing = 0.001788619))
(name, n, result) = ("Gauss AD-Jacobian", 4, (rss_before = 973.41015625, rs
s_after = 998.078125, delta_mib = 24.66796875, timing = 0.012504442))
(name, n, result) = ("Gauss AD-Jacobian", 6, (rss_before = 980.79296875, rs
s_after = 1007.4375, delta_mib = 26.64453125, timing = 0.097911))
(name, n, result) = ("Gauss AD-Jacobian", 8, (rss_before = 963.5, rss_after
 = 1002.9296875, delta_mib = 39.4296875, timing = 0.844736191))
(name, n, result) = ("Gauss AD-Jacobian", 10, (rss_before = 962.7578125, rs
s_after = 1042.546875, delta_mib = 79.7890625, timing = 2.749811013))
(name, n, result) = ("Gauss AD-Jacobian", 12, (rss_before = 983.16015625, r
ss_after = 996.6640625, delta_mib = 13.50390625, timing = 7.08979248))
(name, n, result) = ("GaussKronrod AD-Jacobian", 2, (rss_before = 1017.7929
6875, rss_after = 1042.859375, delta_mib = 25.06640625, timing = 0.00205132
7))
(name, n, result) = ("GaussKronrod AD-Jacobian", 4, (rss_before = 954.47265
625, rss_after = 1003.15234375, delta_mib = 48.6796875, timing = 0.01914752
1))
(name, n, result) = ("GaussKronrod AD-Jacobian", 6, (rss_before = 963.15625
, rss_after = 984.41796875, delta_mib = 21.26171875, timing = 0.136595926))
(name, n, result) = ("GaussKronrod AD-Jacobian", 8, (rss_before = 961.94531
25, rss_after = 1000.62109375, delta_mib = 38.67578125, timing = 1.10808572
3))
(name, n, result) = ("GaussKronrod AD-Jacobian", 10, (rss_before = 997.0859
375, rss_after = 1052.26171875, delta_mib = 55.17578125, timing = 3.4569514
41))
(name, n, result) = ("GaussKronrod AD-Jacobian", 12, (rss_before = 963.2851
5625, rss_after = 1035.359375, delta_mib = 72.07421875, timing = 8.70635495
3))
6-element Vector{@NamedTuple{name::String, results::Vector{@NamedTuple{rss_
before::Float64, rss_after::Float64, delta_mib::Float64, timing::Float64}}}
}:
 (name = "Interp user-Jacobian", results = [(rss_before = 982.83203125, rss
_after = 1026.7890625, delta_mib = 43.95703125, timing = 0.006102091), (rss
_before = 1004.57421875, rss_after = 1063.515625, delta_mib = 58.94140625, 
timing = 0.116232686), (rss_before = 990.10546875, rss_after = 1040.2460937
5, delta_mib = 50.140625, timing = 1.838037656), (rss_before = 963.20703125
, rss_after = 1047.953125, delta_mib = 84.74609375, timing = 7.762733207), 
(rss_before = 983.37109375, rss_after = 1090.78125, delta_mib = 107.4101562
5, timing = 24.884803903), (rss_before = 982.68359375, rss_after = 1103.410
15625, delta_mib = 120.7265625, timing = 72.31769302)])
 (name = "Interp AD-Jacobian", results = [(rss_before = 1003.33203125, rss_
after = 1049.81640625, delta_mib = 46.484375, timing = 0.003599298), (rss_b
efore = 963.328125, rss_after = 999.3046875, delta_mib = 35.9765625, timing
 = 0.052426451), (rss_before = 955.21484375, rss_after = 1032.515625, delta
_mib = 77.30078125, timing = 0.930814869), (rss_before = 963.3984375, rss_a
fter = 1014.140625, delta_mib = 50.7421875, timing = 4.375893011), (rss_bef
ore = 962.75390625, rss_after = 1024.625, delta_mib = 61.87109375, timing =
 13.870388276), (rss_before = 962.4375, rss_after = 1142.87890625, delta_mi
b = 180.44140625, timing = 41.715235378)])
 (name = "Quad user-Jacobian", results = [(rss_before = 960.01171875, rss_a
fter = 1007.4453125, delta_mib = 47.43359375, timing = 0.002169866), (rss_b
efore = 963.265625, rss_after = 1033.28515625, delta_mib = 70.01953125, tim
ing = 0.01129018), (rss_before = 992.24609375, rss_after = 1097.734375, del
ta_mib = 105.48828125, timing = 0.05117379), (rss_before = 1001.1640625, rs
s_after = 1040.55859375, delta_mib = 39.39453125, timing = 0.392806672), (r
ss_before = 969.625, rss_after = 1035.28515625, delta_mib = 65.66015625, ti
ming = 0.91595626), (rss_before = 960.76171875, rss_after = 1056.31640625, 
delta_mib = 95.5546875, timing = 2.271826671)])
 (name = "Quad AD-Jacobian", results = [(rss_before = 961.3515625, rss_afte
r = 1061.40625, delta_mib = 100.0546875, timing = 0.00162047), (rss_before 
= 963.078125, rss_after = 1003.5859375, delta_mib = 40.5078125, timing = 0.
012236963), (rss_before = 962.9140625, rss_after = 1045.94140625, delta_mib
 = 83.02734375, timing = 0.105513332), (rss_before = 988.9609375, rss_after
 = 1041.796875, delta_mib = 52.8359375, timing = 0.870751881), (rss_before 
= 984.75, rss_after = 1061.81640625, delta_mib = 77.06640625, timing = 2.98
2462034), (rss_before = 963.4375, rss_after = 1005.89453125, delta_mib = 42
.45703125, timing = 7.249959207)])
 (name = "Gauss AD-Jacobian", results = [(rss_before = 1016.671875, rss_aft
er = 1068.94921875, delta_mib = 52.27734375, timing = 0.001788619), (rss_be
fore = 973.41015625, rss_after = 998.078125, delta_mib = 24.66796875, timin
g = 0.012504442), (rss_before = 980.79296875, rss_after = 1007.4375, delta_
mib = 26.64453125, timing = 0.097911), (rss_before = 963.5, rss_after = 100
2.9296875, delta_mib = 39.4296875, timing = 0.844736191), (rss_before = 962
.7578125, rss_after = 1042.546875, delta_mib = 79.7890625, timing = 2.74981
1013), (rss_before = 983.16015625, rss_after = 996.6640625, delta_mib = 13.
50390625, timing = 7.08979248)])
 (name = "GaussKronrod AD-Jacobian", results = [(rss_before = 1017.79296875
, rss_after = 1042.859375, delta_mib = 25.06640625, timing = 0.002051327), 
(rss_before = 954.47265625, rss_after = 1003.15234375, delta_mib = 48.67968
75, timing = 0.019147521), (rss_before = 963.15625, rss_after = 984.4179687
5, delta_mib = 21.26171875, timing = 0.136595926), (rss_before = 961.945312
5, rss_after = 1000.62109375, delta_mib = 38.67578125, timing = 1.108085723
), (rss_before = 997.0859375, rss_after = 1052.26171875, delta_mib = 55.175
78125, timing = 3.456951441), (rss_before = 963.28515625, rss_after = 1035.
359375, delta_mib = 72.07421875, timing = 8.706354953)])
```



```julia
mem_params = n_to_param.(mem_sizes)

plt_mem1 = plot(title = "Brusselator Sensitivity Memory Scaling");
plot!(plt_mem1, mem_params, [r.delta_mib for r in forwarddiff_mem],
    lab = "Forward-Mode DSAAD", lw = lw, marksize = ms,
    linestyle = :auto, marker = :auto);
plot!(plt_mem1, mem_params, [r.delta_mib for r in numdiff_mem],
    lab = "Numerical Differentiation", lw = lw, marksize = ms,
    linestyle = :auto, marker = :auto);
for entry in adjoint_ad_mem
    plot!(plt_mem1, mem_params, [r.delta_mib for r in entry.results],
        lab = entry.name, lw = lw, marksize = ms,
        linestyle = :auto, marker = :auto)
end
xaxis!(plt_mem1, "Number of Parameters", :log10);
yaxis!(plt_mem1, "Memory (MiB)");
plot!(plt_mem1, legend = :outertopleft, size = (1200, 600))
```

![](figures/BrussScaling_22_1.png)



## Appendix


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AutomaticDifferentiation","BrussScaling.jmd")
```

Computer Information:

```
Julia Version 1.10.10
Commit 95f30e51f41 (2025-06-27 09:51 UTC)
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
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953:

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Project.toml`
  [6e4b80f9] BenchmarkTools v1.6.3
  [a93c6f00] DataFrames v1.8.1
  [1313f7d8] DataFramesMeta v0.15.6
  [a0c0ee7d] DifferentiationInterface v0.7.16
  [a82114a7] DifferentiationInterfaceTest v0.11.0
  [7da242da] Enzyme v0.13.129
  [6a86dc24] FiniteDiff v2.29.0
  [f6369f11] ForwardDiff v1.3.2
  [da2b9cff] Mooncake v0.5.8
  [1dea7af3] OrdinaryDiffEq v6.108.0
  [65888b18] ParameterizedFunctions v5.22.0
  [91a5bcdd] Plots v1.41.6
⌅ [08abe8d2] PrettyTables v2.4.0
  [37e2e3b7] ReverseDiff v1.16.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [1ed8b502] SciMLSensitivity v7.96.0
  [90137ffa] StaticArrays v1.9.17
  [9f7883ad] Tracker v0.2.38
  [e88e6eb3] Zygote v0.7.10
  [37e2e46d] LinearAlgebra
  [d6f4376e] Markdown
  [de0858da] Printf
  [8dfed614] Test
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Manifest.toml`
  [47edcb42] ADTypes v1.21.0
  [621f4979] AbstractFFTs v1.5.0
  [6e696c72] AbstractPlutoDingetjes v1.3.2
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.43
  [79e6a3ab] Adapt v4.4.0
  [66dad0bd] AliasTables v1.1.3
  [9b6a8646] AllocCheck v0.2.3
  [ec485272] ArnoldiMethod v0.4.0
  [4fba245c] ArrayInterface v7.22.0
  [4c555306] ArrayLayouts v1.12.2
  [a9b6321e] Atomix v1.1.2
  [6e4b80f9] BenchmarkTools v1.6.3
  [e2ed5e7c] Bijections v0.2.2
  [caf10ac8] BipartiteGraphs v0.1.7
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
  [8e7c35d0] BlockArrays v1.9.3
  [70df07ce] BracketingNonlinearSolve v1.10.0
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.7
  [7057c7e9] Cassette v0.3.14
  [8be319e6] Chain v1.0.0
  [082447d4] ChainRules v1.73.0
  [d360d2e6] ChainRulesCore v1.26.0
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [944b1d66] CodecZlib v0.7.8
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌅ [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.6
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
  [34da2185] Compat v4.18.1
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.5.1
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.8.1
  [1313f7d8] DataFramesMeta v0.15.6
  [864edb3b] DataStructures v0.19.3
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v6.210.0
  [459566f4] DiffEqCallbacks v4.12.0
  [77a26b50] DiffEqNoiseProcess v5.27.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [a0c0ee7d] DifferentiationInterface v0.7.16
  [a82114a7] DifferentiationInterfaceTest v0.11.0
  [8d63f2c5] DispatchDoctor v0.4.28
  [31c24e10] Distributions v0.25.123
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.7.16
  [7c1d4256] DynamicPolynomials v0.6.4
  [4e289a0a] EnumX v1.0.6
  [7da242da] Enzyme v0.13.129
  [f151be2c] EnzymeCore v0.8.18
  [460bff9d] ExceptionUnwrapping v0.1.11
  [d4d017d3] ExponentialUtilities v1.30.0
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
  [c87230d0] FFMPEG v0.4.5
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
  [442a2c76] FastGaussQuadrature v1.1.0
  [a4df4552] FastPower v1.3.1
  [1a297f60] FillArrays v1.16.0
  [64ca27bc] FindFirstFunctions v1.8.0
  [6a86dc24] FiniteDiff v2.29.0
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.3.2
  [f62d2435] FunctionProperties v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.5.2
  [46192b85] GPUArraysCore v0.2.0
  [61eb1bfa] GPUCompiler v1.8.2
  [28b8d3ca] GR v0.73.22
  [c145ed77] GenericSchur v0.5.6
  [d7ba0133] Git v1.5.0
  [86223c79] Graphs v1.13.4
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.10.19
  [076d061b] HashArrayMappedTries v0.2.0
⌅ [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.28
  [7073ff75] IJulia v1.34.3
  [7869d1d1] IRTools v0.4.15
  [615f187c] IfElse v0.1.1
  [3263718b] ImplicitDiscreteSolve v1.7.0
  [d25df0c9] Inflate v0.1.5
  [842dd82b] InlineStrings v1.4.5
  [18e54dd8] IntegerMathUtils v0.1.3
  [8197267c] IntervalSets v0.7.13
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.7.1
⌅ [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
⌃ [ccbc3e58] JumpProcesses v9.22.1
  [63c18a36] KernelAbstractions v0.9.40
  [ba0b0d4f] Krylov v0.10.5
  [929cbde3] LLVM v9.4.6
  [b964fa9f] LaTeXStrings v1.4.0
  [23fbe1c1] Latexify v0.16.10
  [10f19ff3] LayoutPointers v0.1.17
  [87fe0de2] LineSearch v0.1.6
  [d3d80556] LineSearches v7.6.0
  [7ed4a6bd] LinearSolve v3.59.1
  [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.10
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
  [961ee093] ModelingToolkit v11.11.1
⌃ [7771a370] ModelingToolkitBase v1.14.0
  [6bb917b9] ModelingToolkitTearing v1.4.0
  [da2b9cff] Mooncake v0.5.8
  [2e0e35c7] Moshi v0.3.7
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.13
  [ffc61752] Mustache v1.0.21
  [d8a4904e] MutableArithmetics v1.6.7
  [d41bc354] NLSolversBase v8.0.0
  [872c559c] NNlib v0.9.33
  [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.15.0
  [be0214bd] NonlinearSolveBase v2.14.0
⌅ [5959db7a] NonlinearSolveFirstOrder v1.11.1
  [9a2c21bd] NonlinearSolveQuasiNewton v1.12.0
  [26075421] NonlinearSolveSpectralMethods v1.6.0
  [d8793406] ObjectFile v0.5.0
  [6fe1bfb0] OffsetArrays v1.17.0
  [4d8831e6] OpenSSL v1.6.1
  [3bd65402] Optimisers v0.4.7
  [bac558e1] OrderedCollections v1.8.1
  [1dea7af3] OrdinaryDiffEq v6.108.0
  [89bda076] OrdinaryDiffEqAdamsBashforthMoulton v1.9.0
  [6ad6398a] OrdinaryDiffEqBDF v1.21.0
  [bbf590c4] OrdinaryDiffEqCore v3.9.0
  [50262376] OrdinaryDiffEqDefault v1.12.0
  [4302a76b] OrdinaryDiffEqDifferentiation v2.1.0
  [9286f039] OrdinaryDiffEqExplicitRK v1.9.0
  [e0540318] OrdinaryDiffEqExponentialRK v1.13.0
  [becaefa8] OrdinaryDiffEqExtrapolation v1.15.0
  [5960d6e9] OrdinaryDiffEqFIRK v1.23.0
  [101fe9f7] OrdinaryDiffEqFeagin v1.8.0
  [d3585ca7] OrdinaryDiffEqFunctionMap v1.9.0
  [d28bc4f8] OrdinaryDiffEqHighOrderRK v1.9.0
  [9f002381] OrdinaryDiffEqIMEXMultistep v1.12.0
  [521117fe] OrdinaryDiffEqLinear v1.10.0
  [1344f307] OrdinaryDiffEqLowOrderRK v1.10.0
  [b0944070] OrdinaryDiffEqLowStorageRK v1.12.0
  [127b3ac7] OrdinaryDiffEqNonlinearSolve v1.23.0
  [c9986a66] OrdinaryDiffEqNordsieck v1.9.0
  [5dd0a6cf] OrdinaryDiffEqPDIRK v1.11.0
  [5b33eab2] OrdinaryDiffEqPRK v1.8.0
  [04162be5] OrdinaryDiffEqQPRK v1.8.0
  [af6ede74] OrdinaryDiffEqRKN v1.10.0
  [43230ef6] OrdinaryDiffEqRosenbrock v1.25.0
  [2d112036] OrdinaryDiffEqSDIRK v1.12.0
  [669c94d9] OrdinaryDiffEqSSPRK v1.11.0
  [e3e12d00] OrdinaryDiffEqStabilizedIRK v1.11.0
  [358294b1] OrdinaryDiffEqStabilizedRK v1.8.0
  [fa646aed] OrdinaryDiffEqSymplecticRK v1.11.0
  [b1df2697] OrdinaryDiffEqTsit5 v1.9.0
  [79d7bb75] OrdinaryDiffEqVerner v1.11.0
  [90014a1f] PDMats v0.11.37
  [65888b18] ParameterizedFunctions v5.22.0
  [69de0a69] Parsers v2.8.3
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.6
  [e409e4f3] PoissonRandom v0.4.7
  [f517fe37] Polyester v0.7.19
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
  [d236fae5] PreallocationTools v1.1.2
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.1
⌅ [08abe8d2] PrettyTables v2.4.0
  [27ebfcd6] Primes v0.5.7
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [1fd47b50] QuadGK v2.11.2
  [e6cf234a] RandomNumbers v1.6.0
  [988b38a3] ReadOnlyArrays v0.2.0
  [795d4caa] ReadOnlyDicts v1.0.1
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v3.48.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.2.0
  [37e2e3b7] ReverseDiff v1.16.2
  [79098fc4] Rmath v0.9.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.17
  [9dfe8606] SCCNonlinearSolve v1.11.0
  [94e857df] SIMDTypes v0.1.0
  [0bca4576] SciMLBase v2.144.0
  [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.12
  [a6db7da4] SciMLLogging v1.9.1
  [c0aeaf25] SciMLOperators v1.15.1
  [431bcebd] SciMLPublic v1.0.1
  [1ed8b502] SciMLSensitivity v7.96.0
  [53ae85a6] SciMLStructures v1.10.0
  [7e506255] ScopedValues v1.5.0
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.9
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
  [727e6d20] SimpleNonlinearSolve v2.11.0
  [699a6c99] SimpleTraits v0.9.5
  [a2af1166] SortingAlgorithms v1.2.2
  [dc90abb0] SparseInverseSubset v0.1.2
  [0a514795] SparseMatrixColorings v0.4.23
  [276daf66] SpecialFunctions v2.7.1
  [860ef19b] StableRNGs v1.0.4
  [64909d44] StateSelection v1.3.0
  [aedffcd0] Static v1.3.1
  [0d7ed370] StaticArrayInterface v1.9.0
  [90137ffa] StaticArrays v1.9.17
  [1e83bf80] StaticArraysCore v1.4.4
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.10
  [4c63d2b9] StatsFuns v1.5.2
  [7792a7ef] StrideArraysCore v0.5.8
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.4.2
  [09ab397b] StructArrays v0.7.2
  [53d494c1] StructIO v0.3.1
  [3384d301] SymbolicCompilerPasses v0.1.2
⌃ [2efcf032] SymbolicIndexingInterface v0.3.44
  [19f23fe9] SymbolicLimits v1.1.0
  [d1185830] SymbolicUtils v4.18.5
⌃ [0c5d862f] Symbolics v7.15.1
  [9ce81f87] TableMetadataTools v0.1.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.1
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [8290d209] ThreadingUtilities v0.5.5
  [a759f4b9] TimerOutputs v0.5.29
  [9f7883ad] Tracker v0.2.38
  [e689c965] Tracy v0.1.6
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.6.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.28.0
  [013be700] UnsafeAtomics v0.3.0
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [d30d5f5c] WeakCacheSets v0.1.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [e88e6eb3] Zygote v0.7.10
  [700de1a5] ZygoteRules v0.2.7
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.5+1
  [ee1fde0b] Dbus_jll v1.16.2+0
  [7cc45869] Enzyme_jll v0.0.249+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.7.3+0
  [b22a6f82] FFMPEG_jll v8.0.1+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.13.4+0
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.4.1+0
  [d2c73de3] GR_jll v0.73.22+0
  [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [020c3dae] Git_LFS_jll v3.7.0+0
  [f8c6e375] Git_jll v2.53.0+0
  [7746bdde] Glib_jll v2.86.3+0
  [3b182d85] Graphite2_jll v1.3.15+0
  [2e76f6c2] HarfBuzz_jll v8.5.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.1.4+0
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.0.1+0
  [dad2f222] LLVMExtra_jll v0.0.38+0
  [1d63c593] LLVMOpenMP_jll v18.1.8+0
  [dd4b983a] LZO_jll v2.10.3+0
  [ad6e5548] LibTracyClient_jll v0.13.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.41.3+0
  [89763e89] Libtiff_jll v4.7.2+0
  [38a345b3] Libuuid_jll v2.41.3+0
  [856f044c] MKL_jll v2025.2.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [9bd350c2] OpenSSH_jll v10.2.1+0
  [458c3c95] OpenSSL_jll v3.5.5+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.57.0+0
⌅ [30392449] Pixman_jll v0.44.2+0
⌅ [c0090381] Qt6Base_jll v6.8.2+2
⌅ [629bc702] Qt6Declarative_jll v6.8.2+1
⌅ [ce943373] Qt6ShaderTools_jll v6.8.2+1
⌃ [e99dba38] Qt6Wayland_jll v6.8.2+2
  [f50d1b31] Rmath_jll v0.5.1+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
  [ffd25f8a] XZ_jll v5.8.2+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.13+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.8+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.2+0
  [a51aa0fd] Xorg_libXi_jll v1.8.3+0
  [d1454406] Xorg_libXinerama_jll v1.1.7+0
  [ec84b674] Xorg_libXrandr_jll v1.5.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.2.0+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.6+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.44.0+0
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
  [214eeab7] fzf_jll v0.61.1+0
  [a4ae2306] libaom_jll v3.13.1+0
  [0ac62f75] libass_jll v0.17.4+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.55+0
  [a9144af2] libsodium_jll v1.0.21+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.0.0+1
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
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
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

