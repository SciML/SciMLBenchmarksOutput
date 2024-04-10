---
author: "Chris Rackauckas and Yingbo Ma"
title: "Bruss Scaling PDE Differentaition Benchmarks"
---


From the paper [A Comparison of Automatic Differentiation and Continuous Sensitivity Analysis for Derivatives of Differential Equation Solutions](https://ieeexplore.ieee.org/abstract/document/9622796)

```julia
using OrdinaryDiffEq, ReverseDiff, ForwardDiff, FiniteDiff, SciMLSensitivity
using LinearAlgebra, Tracker, Plots
```


```julia
function makebrusselator(N=8)
    xyd_brusselator = range(0,stop=1,length=N)
    function limit(a, N)
      if a == N+1
        return 1
      elseif a == 0
        return N
      else
        return a
      end
    end
    brusselator_f(x, y, t) = ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) &&
                                    (t >= 1.1), 5., 0.)
    brusselator_2d_loop = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator)
      function brusselator_2d_loop(du, u, p, t)
          @inbounds begin
              ii1 = N^2
              ii2 = ii1+N^2
              ii3 = ii2+2(N^2)
              A = @view p[1:ii1]
              B = @view p[ii1+1:ii2]
              α = @view p[ii2+1:ii3]
              II = LinearIndices((N, N, 2))
              for I in CartesianIndices((N, N))
                  x = xyd[I[1]]
                  y = xyd[I[2]]
                  i = I[1]
                  j = I[2]
                  ip1 = limit(i+1, N); im1 = limit(i-1, N)
                  jp1 = limit(j+1, N); jm1 = limit(j-1, N)
                  du[II[i,j,1]] = α[II[i,j,1]]*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2 +
                      B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
              end
              for I in CartesianIndices((N, N))
                i = I[1]
                j = I[2]
                ip1 = limit(i+1, N)
                im1 = limit(i-1, N)
                jp1 = limit(j+1, N)
                jm1 = limit(j-1, N)
                du[II[i,j,2]] = α[II[i,j,2]]*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2 +
                    A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
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
            u[I,1] = 22*(y*(1-y))^(3/2)
            u[I,2] = 27*(x*(1-x))^(3/2)
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
      (J,a,p,t) -> begin
        ii1 = N^2
        ii2 = ii1+N^2
        ii3 = ii2+2(N^2)
        A = @view p[1:ii1]
        B = @view p[ii1+1:ii2]
        α = @view p[ii2+1:ii3]
        u = @view a[1:end÷2]
        v = @view a[end÷2+1:end]
        N2 = length(a)÷2
        α1 = @view α[1:end÷2]
        α2 = @view α[end÷2+1:end]
        fill!(J, 0)

        J[1:N2, 1:N2] .= α1.*Op
        J[N2+1:end, N2+1:end] .= α2.*Op

        J1 = @view J[1:N2,     1:N2]
        J2 = @view J[N2+1:end, 1:N2]
        J3 = @view J[1:N2,     N2+1:end]
        J4 = @view J[N2+1:end, N2+1:end]
        J1[diagind(J1)] .+= @. 2u*v-(A+1)
        J2[diagind(J2)] .= @. A-2u*v
        J3[diagind(J3)] .= @. u^2
        J4[diagind(J4)] .+= @. -u^2
        nothing
      end
    end
    Jmat = zeros(2N*N, 2N*N)
    dp = zeros(2N*N, 4N*N)
    brusselator_comp = let N=N, xyd=xyd_brusselator, dx=step(xyd_brusselator), Jmat=Jmat, dp=dp, brusselator_jac=brusselator_jac
      function brusselator_comp(dus, us, p, t)
          @inbounds begin
              ii1 = N^2
              ii2 = ii1+N^2
              ii3 = ii2+2(N^2)
              @views u, s = us[1:ii2], us[ii2+1:end]
              du = @view dus[1:ii2]
              ds = @view dus[ii2+1:end]
              fill!(dp, 0)
              A = @view p[1:ii1]
              B = @view p[ii1+1:ii2]
              α = @view p[ii2+1:ii3]
              dfdα = @view dp[:, ii2+1:ii3]
              diagind(dfdα)
              for i in 1:ii1
                dp[i, ii1+i] = 1
              end
              II = LinearIndices((N, N, 2))
              uu = @view u[1:end÷2]
              for i in eachindex(uu)
                dp[i, i] = -uu[i]
                dp[i+ii1, i] = uu[i]
              end
              for I in CartesianIndices((N, N))
                  x = xyd[I[1]]
                  y = xyd[I[2]]
                  i = I[1]
                  j = I[2]
                  ip1 = limit(i+1, N); im1 = limit(i-1, N)
                  jp1 = limit(j+1, N); jm1 = limit(j-1, N)
                  au = dfdα[II[i,j,1],II[i,j,1]] = (u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]])/dx^2
                  du[II[i,j,1]] = α[II[i,j,1]]*(au) + B[II[i,j,1]] + u[II[i,j,1]]^2*u[II[i,j,2]] - (A[II[i,j,1]] + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
              end
              for I in CartesianIndices((N, N))
                  i = I[1]
                  j = I[2]
                  ip1 = limit(i+1, N)
                  im1 = limit(i-1, N)
                  jp1 = limit(j+1, N)
                  jm1 = limit(j-1, N)
                  av = dfdα[II[i,j,2],II[i,j,2]] = (u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]])/dx^2
                  du[II[i,j,2]] = α[II[i,j,2]]*(av) + A[II[i,j,1]]*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
              end
              brusselator_jac(Jmat,u,p,t)
              BLAS.gemm!('N', 'N', 1., Jmat, reshape(s, 2N*N, 4N*N), 1., dp)
              copyto!(ds, vec(dp))
              return nothing
          end
      end
    end
    u0 = init_brusselator_2d(xyd_brusselator)
    p = [fill(3.4,N^2); fill(1.,N^2); fill(10.,2*N^2)]
    brusselator_2d_loop, u0, p, brusselator_jac, ODEProblem(brusselator_comp, copy([u0;zeros((N^2*2)*(N^2*4))]), (0.,10.), p)
end

Base.eps(::Type{Tracker.TrackedReal{T}}) where T = eps(T)
Base.vec(v::Adjoint{<:Real, <:AbstractVector}) = vec(v') # bad bad hack
```




## Setup AutoDiff

```julia
bt = 0:0.1:1
tspan = (0.0, 1.0)
forwarddiffn = vcat(2:10,12,15)
reversediffn = 2:10
numdiffn = vcat(2:10,12)
csan = vcat(2:10,12,15,17)
#csaseedn = 2:10
tols = (abstol=1e-5, reltol=1e-7)

@isdefined(PROBS) || (const PROBS = Dict{Int,Any}())
makebrusselator!(dict, n) = get!(()->makebrusselator(n), dict, n)

_adjoint_methods = ntuple(3) do ii
  Alg = (InterpolatingAdjoint, QuadratureAdjoint, BacksolveAdjoint)[ii]
  (
    user = Alg(autodiff=false,autojacvec=false), # user Jacobian
    adjc = Alg(autodiff=true,autojacvec=false), # AD Jacobian
    advj = Alg(autodiff=true,autojacvec=EnzymeVJP()), # AD vJ
  )
end |> NamedTuple{(:interp, :quad, :backsol)}
@isdefined(ADJOINT_METHODS) || (const ADJOINT_METHODS = mapreduce(collect, vcat, _adjoint_methods))

function auto_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); diffalg=ReverseDiff.gradient, kwargs...)
  test_f(p) = begin
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,convert.(eltype(p),u0),tspan,p)
    sol = solve(prob,alg,saveat=t; kwargs...)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  diffalg(test_f, p)
end
@inline function diffeq_sen_l2(df, u0, tspan, p, t, alg=Tsit5();
                       abstol=1e-5, reltol=1e-7, iabstol=abstol, ireltol=reltol,
                       sensalg=SensitivityAlg(), kwargs...)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(df,u0,tspan,p)
    saveat = tspan[1] != t[1] && tspan[end] != t[end] ? vcat(tspan[1],t,tspan[end]) : t
    sol = solve(prob, alg, abstol=abstol, reltol=reltol, saveat=saveat; kwargs...)
    dg(out,u,p,t,i) = (out.=u.-1.0)
    adjoint_sensitivities(sol,alg;t,abstol=abstol,dgdu_discrete = dg,
                            reltol=reltol,sensealg=sensalg)
end
```

```
diffeq_sen_l2 (generic function with 2 methods)
```





## AD Choice Benchmarks

```julia
forwarddiff = map(forwarddiffn) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...)
  @show n,t
  t
end
```

```
(n, t) = (2, 0.000756015)
(n, t) = (3, 0.019032022)
(n, t) = (4, 0.0434409)
(n, t) = (5, 0.636910733)
(n, t) = (6, 2.355942132)
(n, t) = (7, 7.45100535)
(n, t) = (8, 20.342086053)
(n, t) = (9, 48.778052187)
(n, t) = (10, 111.928815992)
(n, t) = (12, 462.868346248)
(n, t) = (15, 2809.63038436)
11-element Vector{Float64}:
    0.000756015
    0.019032022
    0.0434409
    0.636910733
    2.355942132
    7.45100535
   20.342086053
   48.778052187
  111.928815992
  462.868346248
 2809.63038436
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
  @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(FiniteDiff.finite_difference_gradient), tols...)
  t = @elapsed auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(FiniteDiff.finite_difference_gradient), tols...)
  @show n,t
  t
end
```

```
(n, t) = (2, 0.003164519)
(n, t) = (3, 0.028563767)
(n, t) = (4, 0.096642999)
(n, t) = (5, 0.29421005)
(n, t) = (6, 0.811913953)
(n, t) = (7, 1.750564522)
(n, t) = (8, 4.12755468)
(n, t) = (9, 9.006533995)
(n, t) = (10, 16.398053475)
(n, t) = (12, 59.336468408)
10-element Vector{Float64}:
  0.003164519
  0.028563767
  0.096642999
  0.29421005
  0.811913953
  1.750564522
  4.12755468
  9.006533995
 16.398053475
 59.336468408
```



```julia
csa = map(csan) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
  @time ts = map(ADJOINT_METHODS[1:2end÷3]) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? bfun : ODEFunction(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    @time diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    return t
  end
  @show n,ts
  ts
end
```

```
14.241675 seconds (21.81 M allocations: 1.426 GiB, 4.64% gc time, 99.91% c
ompilation time)
  9.803122 seconds (8.68 M allocations: 590.099 MiB, 3.25% gc time, 99.93% 
compilation time)
 45.022359 seconds (30.85 M allocations: 2.078 GiB, 2.80% gc time, 99.99% c
ompilation time)
 13.325827 seconds (16.17 M allocations: 1.042 GiB, 2.36% gc time, 99.95% c
ompilation time)
  7.103375 seconds (5.53 M allocations: 387.971 MiB, 2.72% gc time, 99.93% 
compilation time)
  7.241399 seconds (7.77 M allocations: 535.592 MiB, 2.62% gc time, 99.93% 
compilation time)
 97.493285 seconds (91.39 M allocations: 6.063 GiB, 3.01% gc time, 99.93% c
ompilation time)
(n, ts) = (2, [0.007631601, 0.003497588, 0.002133777, 0.00325365, 0.0017622
99, 0.001389081])
  0.121318 seconds (71.65 k allocations: 6.203 MiB)
 13.904428 seconds (4.38 M allocations: 293.260 MiB, 0.50% gc time, 99.87% 
compilation time)
  0.004034 seconds (1.34 k allocations: 334.328 KiB)
  0.020257 seconds (10.46 k allocations: 527.250 KiB)
  6.731243 seconds (2.65 M allocations: 178.657 MiB, 2.38% gc time, 99.89% 
compilation time)
  0.002184 seconds (2.00 k allocations: 321.094 KiB)
 20.953451 seconds (7.21 M allocations: 488.233 MiB, 1.09% gc time, 98.36% 
compilation time)
(n, ts) = (3, [0.120028459, 0.014442597, 0.003771436, 0.020123211, 0.004212
653, 0.001939078])
  0.745265 seconds (155.81 k allocations: 13.591 MiB)
 15.188999 seconds (4.28 M allocations: 286.437 MiB, 7.36% gc time, 99.62% 
compilation time)
  0.009277 seconds (1.35 k allocations: 642.984 KiB)
  0.141803 seconds (25.52 k allocations: 1.131 MiB)
  7.901840 seconds (2.65 M allocations: 178.949 MiB, 0.87% gc time, 99.81% 
compilation time)
  0.003919 seconds (2.00 k allocations: 472.828 KiB)
 24.961806 seconds (7.32 M allocations: 499.637 MiB, 4.76% gc time, 92.21% 
compilation time)
(n, ts) = (4, [0.745960262, 0.05464729, 0.008847733, 0.141364253, 0.0119309
04, 0.003622907])
  3.195000 seconds (285.04 k allocations: 25.051 MiB)
 14.045504 seconds (3.42 M allocations: 231.008 MiB, 0.55% gc time, 98.60% 
compilation time)
  0.017125 seconds (1.38 k allocations: 1.186 MiB)
  0.499563 seconds (38.30 k allocations: 1.827 MiB)
  7.359902 seconds (2.65 M allocations: 179.319 MiB, 1.13% gc time, 99.42% 
compilation time)
  0.006134 seconds (2.01 k allocations: 709.859 KiB)
 29.088418 seconds (6.76 M allocations: 472.023 MiB, 0.55% gc time, 72.76% 
compilation time)
(n, ts) = (5, [3.204982379, 0.193746426, 0.016889382, 0.50010746, 0.0391584
79, 0.005814323])
 10.169269 seconds (448.67 k allocations: 39.973 MiB)
 17.320746 seconds (3.36 M allocations: 227.915 MiB, 0.47% gc time, 96.66% 
compilation time)
  0.031963 seconds (1.39 k allocations: 2.124 MiB)
  1.608720 seconds (60.51 k allocations: 3.136 MiB)
  8.900155 seconds (2.65 M allocations: 179.843 MiB, 0.59% gc time, 98.79% 
compilation time)
  0.010174 seconds (2.01 k allocations: 1002.266 KiB)
 50.547736 seconds (7.10 M allocations: 506.933 MiB, 0.36% gc time, 50.52% 
compilation time)
(n, ts) = (6, [10.207400638, 0.534722699, 0.031400218, 1.613913233, 0.10406
8202, 0.009865837])
 30.961218 seconds (736.49 k allocations: 65.896 MiB, 0.20% gc time)
 15.811095 seconds (2.29 M allocations: 159.208 MiB, 0.77% gc time, 91.92% 
compilation time)
  0.065299 seconds (1.39 k allocations: 3.628 MiB)
  4.275289 seconds (86.10 k allocations: 4.643 MiB)
  1.777314 seconds (467.41 k allocations: 31.838 MiB, 89.36% compilation ti
me)
  0.016558 seconds (2.01 k allocations: 1.344 MiB)
 89.668034 seconds (4.49 M allocations: 352.375 MiB, 0.21% gc time, 17.98% 
compilation time)
(n, ts) = (7, [30.937640879, 1.273757227, 0.064941974, 4.27231622, 0.188822
449, 0.016302055])
 71.136668 seconds (1.01 M allocations: 91.394 MiB, 0.13% gc time)
  3.347130 seconds (96.60 k allocations: 11.623 MiB)
  0.113274 seconds (1.39 k allocations: 5.839 MiB)
 10.998154 seconds (130.45 k allocations: 7.147 MiB)
  0.549867 seconds (15.30 k allocations: 4.190 MiB)
  0.027175 seconds (2.02 k allocations: 1.850 MiB)
172.524694 seconds (2.50 M allocations: 244.300 MiB, 0.11% gc time)
(n, ts) = (8, [71.323703129, 3.337935774, 0.113410122, 10.999207476, 0.5442
6899, 0.02638745])
170.642957 seconds (1.51 M allocations: 137.390 MiB, 0.02% gc time)
  6.256108 seconds (113.13 k allocations: 16.670 MiB, 0.63% gc time)
  0.256347 seconds (1.40 k allocations: 9.037 MiB)
 22.205952 seconds (161.98 k allocations: 9.805 MiB)
  1.081618 seconds (18.57 k allocations: 6.090 MiB)
  0.044947 seconds (2.02 k allocations: 2.481 MiB)
401.163320 seconds (3.61 M allocations: 363.160 MiB, 0.03% gc time)
(n, ts) = (9, [170.983698607, 6.2457719, 0.252826934, 22.060105529, 1.07941
2601, 0.044113449])
351.686442 seconds (2.03 M allocations: 187.363 MiB, 0.02% gc time)
 11.912558 seconds (138.61 k allocations: 23.825 MiB)
  0.364934 seconds (1.40 k allocations: 13.524 MiB)
 49.423872 seconds (238.50 k allocations: 14.728 MiB)
  1.980246 seconds (22.22 k allocations: 8.636 MiB)
  0.061254 seconds (2.02 k allocations: 3.277 MiB)
830.984357 seconds (4.87 M allocations: 502.919 MiB, 0.02% gc time)
(n, ts) = (10, [351.798544088, 11.903739057, 0.409931713, 49.387049711, 1.9
81465502, 0.060394713])
945.298169 seconds (2.68 M allocations: 259.927 MiB, 0.01% gc time)
 37.752769 seconds (204.75 k allocations: 45.190 MiB)
  0.787056 seconds (1.41 k allocations: 27.084 MiB)
165.522162 seconds (387.70 k allocations: 25.952 MiB)
  5.955136 seconds (30.67 k allocations: 16.154 MiB)
  0.140532 seconds (2.02 k allocations: 5.384 MiB)
2310.208877 seconds (6.61 M allocations: 759.592 MiB, 0.01% gc time)
(n, ts) = (12, [944.895829177, 37.65082896, 0.791223912, 165.289455409, 5.9
73566711, 0.139461706])
3769.746243 seconds (4.42 M allocations: 456.847 MiB, 0.00% gc time)
149.269875 seconds (328.40 k allocations: 102.016 MiB)
  1.821735 seconds (1.42 k allocations: 64.527 MiB)
639.721609 seconds (617.28 k allocations: 51.291 MiB, 0.01% gc time)
 22.470967 seconds (46.22 k allocations: 36.236 MiB)
  0.395890 seconds (2.02 k allocations: 10.565 MiB)
9169.553585 seconds (10.83 M allocations: 1.409 GiB, 0.01% gc time)
(n, ts) = (15, [3770.75989337, 150.551946526, 1.857404498, 640.032119327, 2
2.516853904, 0.394835175])
10461.943087 seconds (7.45 M allocations: 766.085 MiB, 0.00% gc time)
294.593109 seconds (392.64 k allocations: 161.463 MiB, 0.02% gc time)
  2.852221 seconds (1.42 k allocations: 105.403 MiB, 1.01% gc time)
1491.114220 seconds (872.69 k allocations: 80.013 MiB, 0.00% gc time)
 47.099557 seconds (58.51 k allocations: 57.747 MiB)
  0.961198 seconds (2.02 k allocations: 15.771 MiB)
24609.075508 seconds (17.56 M allocations: 2.318 GiB, 0.01% gc time)
(n, ts) = (17, [10474.043633816, 294.011395082, 2.80905904, 1491.148559767,
 47.471459197, 1.013831069])
12-element Vector{Vector{Float64}}:
 [0.007631601, 0.003497588, 0.002133777, 0.00325365, 0.001762299, 0.0013890
81]
 [0.120028459, 0.014442597, 0.003771436, 0.020123211, 0.004212653, 0.001939
078]
 [0.745960262, 0.05464729, 0.008847733, 0.141364253, 0.011930904, 0.0036229
07]
 [3.204982379, 0.193746426, 0.016889382, 0.50010746, 0.039158479, 0.0058143
23]
 [10.207400638, 0.534722699, 0.031400218, 1.613913233, 0.104068202, 0.00986
5837]
 [30.937640879, 1.273757227, 0.064941974, 4.27231622, 0.188822449, 0.016302
055]
 [71.323703129, 3.337935774, 0.113410122, 10.999207476, 0.54426899, 0.02638
745]
 [170.983698607, 6.2457719, 0.252826934, 22.060105529, 1.079412601, 0.04411
3449]
 [351.798544088, 11.903739057, 0.409931713, 49.387049711, 1.981465502, 0.06
0394713]
 [944.895829177, 37.65082896, 0.791223912, 165.289455409, 5.973566711, 0.13
9461706]
 [3770.75989337, 150.551946526, 1.857404498, 640.032119327, 22.516853904, 0
.394835175]
 [10474.043633816, 294.011395082, 2.80905904, 1491.148559767, 47.471459197,
 1.013831069]
```



```julia
n_to_param(n) = 4n^2

lw = 2
ms = 0.5
plt1 = plot(title="Sensitivity Scaling on Brusselator");
plot!(plt1, n_to_param.(forwarddiffn), forwarddiff, lab="Forward-Mode DSAAD", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
#plot!(plt1, n_to_param.(reversediffn), reversediff, lab="Reverse-Mode DSAAD", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
csadata = [[csa[j][i] for j in eachindex(csa)] for i in eachindex(csa[1])]
plot!(plt1, n_to_param.(csan), csadata[1], lab="Interpolating CASA user-Jacobian", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(csan), csadata[2], lab="Interpolating CASA AD-Jacobian", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(csan), csadata[3], lab=raw"Interpolating CASA AD-$v^{T}J$ seeding", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(csan), csadata[1+3], lab="Quadrature CASA user-Jacobian", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(csan), csadata[2+3], lab="Quadrature CASA AD-Jacobian", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(csan), csadata[3+3], lab=raw"Quadrature CASA AD-$v^{T}J$ seeding", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt1, n_to_param.(numdiffn), numdiff, lab="Numerical Differentiation", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
xaxis!(plt1, "Number of Parameters", :log10);
yaxis!(plt1, "Runtime (s)", :log10);
plot!(plt1, legend=:outertopleft, size=(1200, 600));
```




## VJP Choice Benchmarks

```julia
bt = 0:0.1:1
tspan = (0.0, 1.0)
csan = vcat(2:10,12,15,17)
tols = (abstol=1e-5, reltol=1e-7)

_adjoint_methods = ntuple(2) do ii
  Alg = (InterpolatingAdjoint, QuadratureAdjoint)[ii]
  (
    advj1 = Alg(autodiff=true,autojacvec=EnzymeVJP()), # AD vJ
    advj2 = Alg(autodiff=true,autojacvec=ReverseDiffVJP(false)), # AD vJ
    advj3 = Alg(autodiff=true,autojacvec=ReverseDiffVJP(true)), # AD vJ
  )
end |> NamedTuple{(:interp, :quad)}
adjoint_methods = mapreduce(collect, vcat, _adjoint_methods)

csavjp = map(csan) do n
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator!(PROBS, n)
  @time ts = map(adjoint_methods) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? bfun : ODEFunction(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    @time diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    t = @elapsed diffeq_sen_l2(bfun, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...)
    return t
  end
  @show n,ts
  ts
end
```

```
0.002239 seconds (1.29 k allocations: 189.812 KiB)
  6.868772 seconds (8.48 M allocations: 563.408 MiB, 2.05% gc time, 99.02% 
compilation time)
  5.823953 seconds (6.14 M allocations: 427.778 MiB, 1.50% gc time, 99.79% 
compilation time)
  0.001393 seconds (2.02 k allocations: 213.359 KiB)
  5.806270 seconds (6.29 M allocations: 428.748 MiB, 2.38% gc time, 99.33% 
compilation time)
  5.851399 seconds (5.93 M allocations: 412.440 MiB, 1.81% gc time, 99.85% 
compilation time)
 25.002686 seconds (28.10 M allocations: 1.850 GiB, 2.06% gc time, 98.82% c
ompilation time)
(n, ts) = (2, [0.00185554, 0.102926832, 0.007727035, 0.001099094, 0.0377393
71, 0.00533119])
  0.004166 seconds (1.34 k allocations: 334.328 KiB)
  0.225626 seconds (2.09 M allocations: 95.601 MiB, 18.01% gc time)
  0.022411 seconds (5.43 k allocations: 549.156 KiB)
  0.002349 seconds (2.00 k allocations: 321.094 KiB)
  0.170420 seconds (1.10 M allocations: 50.153 MiB, 29.56% gc time)
  0.013498 seconds (7.26 k allocations: 600.281 KiB)
  0.790782 seconds (6.41 M allocations: 295.251 MiB, 11.51% gc time)
(n, ts) = (3, [0.003786099, 0.200668513, 0.022075832, 0.001927859, 0.106087
764, 0.013250203])
  0.008682 seconds (1.35 k allocations: 642.984 KiB)
  0.573309 seconds (6.23 M allocations: 268.499 MiB, 3.06% gc time)
  0.063700 seconds (8.56 k allocations: 1018.078 KiB)
  0.003672 seconds (2.00 k allocations: 472.828 KiB)
  0.300938 seconds (2.94 M allocations: 126.580 MiB, 5.86% gc time)
  0.032599 seconds (11.66 k allocations: 956.938 KiB)
  1.971533 seconds (18.39 M allocations: 796.413 MiB, 3.59% gc time)
(n, ts) = (4, [0.008435851, 0.601456554, 0.062646546, 0.00340313, 0.2706990
36, 0.038154998])
  0.016913 seconds (1.38 k allocations: 1.186 MiB)
  1.365009 seconds (14.51 M allocations: 666.823 MiB, 4.88% gc time)
  0.147420 seconds (12.52 k allocations: 1.763 MiB)
  0.006057 seconds (2.01 k allocations: 709.859 KiB)
  0.654548 seconds (6.63 M allocations: 304.666 MiB, 5.72% gc time)
  0.069948 seconds (17.33 k allocations: 1.452 MiB)
  4.491306 seconds (42.35 M allocations: 1.908 GiB, 4.15% gc time)
(n, ts) = (5, [0.016704112, 1.364531867, 0.147684651, 0.005787706, 0.622993
908, 0.069696985])
  0.031521 seconds (1.39 k allocations: 2.124 MiB)
  2.779248 seconds (29.30 M allocations: 1.260 GiB, 4.97% gc time)
  0.304677 seconds (17.34 k allocations: 2.941 MiB)
  0.010027 seconds (2.01 k allocations: 1002.266 KiB)
  1.235171 seconds (12.88 M allocations: 567.153 MiB, 4.21% gc time)
  0.132986 seconds (24.20 k allocations: 2.051 MiB)
  8.965141 seconds (84.44 M allocations: 3.643 GiB, 4.09% gc time)
(n, ts) = (6, [0.031317818, 2.759838844, 0.295756441, 0.009758683, 1.238334
21, 0.132289261])
  0.064897 seconds (1.39 k allocations: 3.628 MiB)
  5.273729 seconds (55.53 M allocations: 2.326 GiB, 4.84% gc time)
  0.577553 seconds (23.34 k allocations: 4.747 MiB)
  0.016275 seconds (2.01 k allocations: 1.344 MiB)
  2.185222 seconds (22.90 M allocations: 981.690 MiB, 4.19% gc time)
  0.239308 seconds (32.31 k allocations: 2.785 MiB)
 16.738583 seconds (156.98 M allocations: 6.593 GiB, 4.18% gc time)
(n, ts) = (7, [0.064706554, 5.243979316, 0.575246715, 0.015937258, 2.225434
407, 0.250245015])
  0.112258 seconds (1.39 k allocations: 5.839 MiB)
  8.846701 seconds (93.69 M allocations: 4.182 GiB, 4.51% gc time)
  0.979864 seconds (30.00 k allocations: 7.323 MiB)
  0.025942 seconds (2.02 k allocations: 1.850 MiB)
  3.770054 seconds (39.29 M allocations: 1.753 GiB, 4.28% gc time)
  0.422606 seconds (41.82 k allocations: 3.789 MiB)
 28.353224 seconds (266.11 M allocations: 11.907 GiB, 4.05% gc time)
(n, ts) = (8, [0.111361593, 8.866607183, 0.991224433, 0.026817154, 3.779157
543, 0.415644082])
  0.250918 seconds (1.40 k allocations: 9.037 MiB)
 15.747758 seconds (165.38 M allocations: 7.177 GiB, 4.66% gc time)
  1.785217 seconds (39.02 k allocations: 10.988 MiB)
  0.042075 seconds (2.02 k allocations: 2.481 MiB)
  5.949078 seconds (61.70 M allocations: 2.677 GiB, 4.61% gc time)
  0.635980 seconds (52.47 k allocations: 4.905 MiB)
 48.801124 seconds (454.35 M allocations: 19.763 GiB, 4.13% gc time)
(n, ts) = (9, [0.2527349, 15.761194694, 1.76407861, 0.042236154, 5.93792499
9, 0.624730872])
  0.351439 seconds (1.40 k allocations: 13.524 MiB)
 23.050221 seconds (233.92 M allocations: 9.952 GiB, 6.85% gc time)
  2.557184 seconds (46.55 k allocations: 15.842 MiB)
  0.062974 seconds (2.02 k allocations: 3.277 MiB)
  9.295514 seconds (92.71 M allocations: 3.942 GiB, 6.83% gc time)
  0.936804 seconds (64.36 k allocations: 6.243 MiB)
 72.686962 seconds (653.49 M allocations: 27.864 GiB, 6.25% gc time)
(n, ts) = (10, [0.398119832, 23.092759014, 2.628896452, 0.086013079, 9.2752
62324, 0.941363233])
  0.691788 seconds (1.41 k allocations: 27.084 MiB)
 49.033433 seconds (498.79 M allocations: 22.283 GiB, 7.00% gc time)
  5.282475 seconds (67.19 k allocations: 30.514 MiB)
  0.143591 seconds (2.02 k allocations: 5.384 MiB)
 19.733352 seconds (188.66 M allocations: 8.423 GiB, 9.58% gc time)
  1.944549 seconds (91.91 k allocations: 9.739 MiB)
153.100265 seconds (1.38 G allocations: 61.554 GiB, 6.64% gc time)
(n, ts) = (12, [0.722123031, 49.290484892, 5.308535142, 0.140035932, 18.847
159483, 1.950582867])
  2.006034 seconds (1.42 k allocations: 64.527 MiB)
136.798009 seconds (1.25 G allocations: 53.273 GiB, 14.45% gc time)
 13.800515 seconds (105.42 k allocations: 69.894 MiB)
  0.396526 seconds (2.02 k allocations: 10.565 MiB)
 41.035880 seconds (453.40 M allocations: 19.275 GiB, 4.60% gc time)
  4.728351 seconds (142.62 k allocations: 17.231 MiB)
390.988182 seconds (3.41 G allocations: 145.412 GiB, 10.17% gc time)
(n, ts) = (15, [2.132332125, 130.355471093, 13.743580252, 0.394838035, 40.9
14303991, 4.668504922])
  3.612675 seconds (1.42 k allocations: 105.403 MiB, 0.42% gc time)
178.854078 seconds (1.92 G allocations: 80.231 GiB, 5.17% gc time)
 21.522343 seconds (131.43 k allocations: 112.047 MiB)
  0.977217 seconds (2.02 k allocations: 15.771 MiB)
 69.217325 seconds (743.36 M allocations: 30.973 GiB, 7.21% gc time)
  8.689379 seconds (182.68 k allocations: 24.262 MiB)
571.949259 seconds (5.34 G allocations: 222.910 GiB, 5.23% gc time)
(n, ts) = (17, [3.585309959, 187.423109633, 21.682107222, 1.016363891, 67.2
16849018, 8.137595789])
12-element Vector{Vector{Float64}}:
 [0.00185554, 0.102926832, 0.007727035, 0.001099094, 0.037739371, 0.0053311
9]
 [0.003786099, 0.200668513, 0.022075832, 0.001927859, 0.106087764, 0.013250
203]
 [0.008435851, 0.601456554, 0.062646546, 0.00340313, 0.270699036, 0.0381549
98]
 [0.016704112, 1.364531867, 0.147684651, 0.005787706, 0.622993908, 0.069696
985]
 [0.031317818, 2.759838844, 0.295756441, 0.009758683, 1.23833421, 0.1322892
61]
 [0.064706554, 5.243979316, 0.575246715, 0.015937258, 2.225434407, 0.250245
015]
 [0.111361593, 8.866607183, 0.991224433, 0.026817154, 3.779157543, 0.415644
082]
 [0.2527349, 15.761194694, 1.76407861, 0.042236154, 5.937924999, 0.62473087
2]
 [0.398119832, 23.092759014, 2.628896452, 0.086013079, 9.275262324, 0.94136
3233]
 [0.722123031, 49.290484892, 5.308535142, 0.140035932, 18.847159483, 1.9505
82867]
 [2.132332125, 130.355471093, 13.743580252, 0.394838035, 40.914303991, 4.66
8504922]
 [3.585309959, 187.423109633, 21.682107222, 1.016363891, 67.216849018, 8.13
7595789]
```



```julia
plt2 = plot(title="Brusselator quadrature adjoint scaling");
csacompare = [[csavjp[j][i] for j in eachindex(csavjp)] for i in eachindex(csavjp[1])]
plot!(plt2, n_to_param.(csan), csadata[2+3], lab="AD-Jacobian", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt2, n_to_param.(csan), csacompare[1+3], lab=raw"EnzymeVJP", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt2, n_to_param.(csan), csacompare[2+3], lab=raw"ReverseDiffVJP", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
plot!(plt2, n_to_param.(csan), csacompare[3+3], lab=raw"Compiled ReverseDiffVJP", lw=lw, marksize=ms, linestyle=:auto, marker=:auto);
xaxis!(plt2, "Number of Parameters", :log10);
yaxis!(plt2, "Runtime (s)", :log10);
plot!(plt2, legend=:outertopleft, size=(1200, 600));
savefig(plt2, "../figure3.pdf")
```

```
"/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/markdown/
figure3.pdf"
```





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
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
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
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Project.toml`
  [6e4b80f9] BenchmarkTools v1.5.0
  [a93c6f00] DataFrames v1.6.1
  [1313f7d8] DataFramesMeta v0.15.2
  [a0c0ee7d] DifferentiationInterface v0.1.0 `https://github.com/gdalle/DifferentiationInterface.jl:DifferentiationInterface#main`
  [a82114a7] DifferentiationInterfaceTest v0.1.0 `https://github.com/gdalle/DifferentiationInterface.jl:DifferentiationInterfaceTest#main`
  [7da242da] Enzyme v0.11.20
  [6a86dc24] FiniteDiff v2.23.0
  [f6369f11] ForwardDiff v0.10.36
  [1dea7af3] OrdinaryDiffEq v6.74.1
  [65888b18] ParameterizedFunctions v5.17.0
  [91a5bcdd] Plots v1.40.3
  [08abe8d2] PrettyTables v2.3.1
  [37e2e3b7] ReverseDiff v1.15.1
  [31c91b34] SciMLBenchmarks v0.1.3
  [1ed8b502] SciMLSensitivity v7.56.2
  [90137ffa] StaticArrays v1.9.3
  [07d77754] Tapir v0.1.2
  [9f7883ad] Tracker v0.2.33
  [e88e6eb3] Zygote v0.6.69
  [37e2e46d] LinearAlgebra
  [d6f4376e] Markdown
  [de0858da] Printf
  [8dfed614] Test
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Manifest.toml`
  [47edcb42] ADTypes v0.2.7
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.36
  [79e6a3ab] Adapt v4.0.4
  [ec485272] ArnoldiMethod v0.4.0
  [4fba245c] ArrayInterface v7.9.0
  [4c555306] ArrayLayouts v1.9.1
  [a9b6321e] Atomix v0.1.0
  [6e4b80f9] BenchmarkTools v1.5.0
  [e2ed5e7c] Bijections v0.1.6
  [d1d4a3ce] BitFlags v0.1.8
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.4
  [00ebfdb7] CSTParser v3.4.2
  [49dc2e85] Calculus v0.5.1
  [7057c7e9] Cassette v0.3.13
  [8be319e6] Chain v0.6.0
  [082447d4] ChainRules v1.63.0
  [d360d2e6] ChainRulesCore v1.23.0
  [0ca39b1e] Chairmarks v1.2.1
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [da1fd8a2] CodeTracking v1.3.5
  [944b1d66] CodecZlib v0.7.4
  [35d6a980] ColorSchemes v3.24.0
  [3da002f7] ColorTypes v0.11.5
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.14.0
  [b0b7db55] ComponentArrays v0.15.11
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.4.1
  [8f4d0f93] Conda v1.10.0
  [187b0558] ConstructionBase v1.5.5
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.6.1
  [1313f7d8] DataFramesMeta v0.15.2
  [864edb3b] DataStructures v0.18.18
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v6.149.0
  [459566f4] DiffEqCallbacks v3.5.0
  [77a26b50] DiffEqNoiseProcess v5.21.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [de460e47] DiffTests v0.1.2
  [a0c0ee7d] DifferentiationInterface v0.1.0 `https://github.com/gdalle/DifferentiationInterface.jl:DifferentiationInterface#main`
  [a82114a7] DifferentiationInterfaceTest v0.1.0 `https://github.com/gdalle/DifferentiationInterface.jl:DifferentiationInterfaceTest#main`
  [b4f34e82] Distances v0.10.11
  [31c24e10] Distributions v0.25.107
  [ffbed154] DocStringExtensions v0.9.3
  [5b8099bc] DomainSets v0.7.11
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.5.5
  [06fc5a27] DynamicQuantities v0.13.2
  [da5c29d0] EllipsisNotation v1.8.0
  [4e289a0a] EnumX v1.0.4
  [7da242da] Enzyme v0.11.20
⌅ [f151be2c] EnzymeCore v0.6.6
  [460bff9d] ExceptionUnwrapping v0.1.10
  [d4d017d3] ExponentialUtilities v1.26.1
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [7034ab61] FastBroadcast v0.2.8
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.2
  [1a297f60] FillArrays v1.10.0
  [64ca27bc] FindFirstFunctions v1.2.0
  [6a86dc24] FiniteDiff v2.23.0
  [53c48c17] FixedPointNumbers v0.8.4
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v0.10.36
  [f62d2435] FunctionProperties v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.4.10
  [0c68f7d7] GPUArrays v10.1.0
  [46192b85] GPUArraysCore v0.1.6
⌅ [61eb1bfa] GPUCompiler v0.25.0
  [28b8d3ca] GR v0.73.3
  [c145ed77] GenericSchur v0.5.4
  [d7ba0133] Git v1.3.1
  [c27321d9] Glob v1.3.1
  [86223c79] Graphs v1.10.0
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v1.10.5
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.16
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [7869d1d1] IRTools v0.4.12
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.4
  [842dd82b] InlineStrings v1.4.0
  [8197267c] IntervalSets v0.7.10
  [3587e190] InverseFunctions v0.1.13
  [41ab1584] InvertedIndices v1.3.0
  [92d709cd] IrrationalConstants v0.2.2
  [82899510] IteratorInterfaceExtensions v1.0.0
⌃ [c3a54625] JET v0.8.22
  [27aeb0d3] JLArrays v0.1.4
  [1019f520] JLFzf v0.1.7
  [692b3bcd] JLLWrappers v1.5.0
  [682c06a0] JSON v0.21.4
⌃ [98e50ef6] JuliaFormatter v1.0.55
  [aa1ae85d] JuliaInterpreter v0.9.31
  [ccbc3e58] JumpProcesses v9.11.1
  [ef3ab10e] KLU v0.6.0
  [63c18a36] KernelAbstractions v0.9.18
  [ba0b0d4f] Krylov v0.9.5
  [929cbde3] LLVM v6.6.3
  [b964fa9f] LaTeXStrings v1.3.1
  [2ee39098] LabelledArrays v1.15.1
  [984bce1d] LambertW v0.4.6
  [23fbe1c1] Latexify v0.16.2
  [10f19ff3] LayoutPointers v0.1.15
  [5078a376] LazyArrays v1.9.1
  [2d8b4e74] LevyArea v1.0.0
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.28.0
  [2ab3a3ac] LogExpFunctions v0.3.27
  [e6f89c97] LoggingExtras v1.0.3
  [bdcacae8] LoopVectorization v0.12.169
  [6f1432cf] LoweredCodeUtils v2.4.5
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.13
  [d125e4d3] ManualMemory v0.1.8
⌃ [a3b82374] MatrixFactorizations v2.1.1
  [bb5d69b7] MaybeInplace v0.1.2
  [739be429] MbedTLS v1.1.9
  [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.2.0
  [961ee093] ModelingToolkit v9.9.0
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.4
  [ffc61752] Mustache v1.0.19
  [d8a4904e] MutableArithmetics v1.4.2
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [872c559c] NNlib v0.9.13
  [77ba4419] NaNMath v1.0.2
  [8913a72c] NonlinearSolve v3.9.1
  [d8793406] ObjectFile v0.4.1
  [6fe1bfb0] OffsetArrays v1.13.0
  [4d8831e6] OpenSSL v1.4.2
  [429524aa] Optim v1.9.4
⌃ [3bd65402] Optimisers v0.3.2
  [bac558e1] OrderedCollections v1.6.3
  [1dea7af3] OrdinaryDiffEq v6.74.1
  [90014a1f] PDMats v0.11.31
  [65ce6f38] PackageExtensionCompat v1.0.2
  [65888b18] ParameterizedFunctions v5.17.0
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.1
  [b98c9c47] Pipe v1.3.0
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.4.1
  [91a5bcdd] Plots v1.40.3
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.12
  [1d0040c9] PolyesterWeave v0.2.1
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.20
  [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.3.1
  [92933f4c] ProgressMeter v1.10.0
  [1fd47b50] QuadGK v2.9.4
  [74087812] Random123 v1.7.0
  [e6cf234a] RandomNumbers v1.5.3
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v3.13.0
  [f2c3362d] RecursiveFactorization v0.2.21
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [37e2e3b7] ReverseDiff v1.15.1
  [295af30f] Revise v3.5.14
  [79098fc4] Rmath v0.7.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.12
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.42
  [0bca4576] SciMLBase v2.31.0
  [31c91b34] SciMLBenchmarks v0.1.3
  [c0aeaf25] SciMLOperators v0.3.8
  [1ed8b502] SciMLSensitivity v7.56.2
  [53ae85a6] SciMLStructures v1.1.0
  [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.1
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [727e6d20] SimpleNonlinearSolve v1.7.0
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
  [47a9eef4] SparseDiffTools v2.17.0
  [dc90abb0] SparseInverseSubset v0.1.2
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.1
  [aedffcd0] Static v0.8.10
  [0d7ed370] StaticArrayInterface v1.5.0
  [90137ffa] StaticArrays v1.9.3
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.7.0
  [2913bbd2] StatsBase v0.34.3
  [4c63d2b9] StatsFuns v1.3.1
  [789caeaf] StochasticDiffEq v6.65.1
  [7792a7ef] StrideArraysCore v0.5.2
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.3.4
  [09ab397b] StructArrays v0.6.18
  [53d494c1] StructIO v0.3.0
  [2efcf032] SymbolicIndexingInterface v0.3.15
  [19f23fe9] SymbolicLimits v0.2.0
  [d1185830] SymbolicUtils v1.5.1
  [0c5d862f] Symbolics v5.27.1
  [9ce81f87] TableMetadataTools v0.1.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.11.1
  [07d77754] Tapir v0.1.2
  [62fd8b95] TensorCore v0.1.1
  [8290d209] ThreadingUtilities v0.5.2
  [a759f4b9] TimerOutputs v0.5.23
  [0796e94c] Tokenize v0.5.28
  [9f7883ad] Tracker v0.2.33
  [3bb67fe8] TranscodingStreams v0.10.7
  [d5829a12] TriangularSolve v0.1.21
  [410a4b4d] Tricks v0.1.8
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.5.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.19.0
  [45397f5d] UnitfulLatexify v1.6.3
  [a7c27f48] Unityper v0.1.6
  [013be700] UnsafeAtomics v0.2.1
  [d80eeb9a] UnsafeAtomicsLLVM v0.1.3
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.65
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [e88e6eb3] Zygote v0.6.69
  [700de1a5] ZygoteRules v0.2.5
  [6e34b625] Bzip2_jll v1.0.8+1
  [83423d85] Cairo_jll v1.18.0+1
⌅ [7cc45869] Enzyme_jll v0.0.102+0
  [2702e6a9] EpollShim_jll v0.0.20230411+0
  [2e619515] Expat_jll v2.5.0+0
⌅ [b22a6f82] FFMPEG_jll v4.4.4+1
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.9+0
  [d2c73de3] GR_jll v0.73.3+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.44.0+2
  [7746bdde] Glib_jll v2.80.0+0
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2024.0.2+0
  [aacddb02] JpegTurbo_jll v3.0.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [dad2f222] LLVMExtra_jll v0.0.29+0
  [1d63c593] LLVMOpenMP_jll v15.0.7+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.17.0+0
  [4b2f31a3] Libmount_jll v2.39.3+0
⌅ [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.39.3+1
  [856f044c] MKL_jll v2024.0.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
  [458c3c95] OpenSSL_jll v3.0.13+1
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.5.3+1
  [f50d1b31] Rmath_jll v0.4.0+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.21.0+1
  [2381bf8a] Wayland_protocols_jll v1.31.0+0
  [02c8fc9c] XML2_jll v2.12.6+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.4.6+0
  [f67eecfb] Xorg_libICE_jll v1.0.10+1
  [c834827a] Xorg_libSM_jll v1.2.3+0
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.4+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.5+0
  [3161d3a3] Zstd_jll v1.5.6+0
  [35ca27e7] eudev_jll v3.2.9+0
⌅ [214eeab7] fzf_jll v0.43.0+0
  [1a1c6b14] gperf_jll v3.1.1+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [2db6ffa8] libevdev_jll v1.11.0+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [36db933b] libinput_jll v1.18.0+0
  [b53b4c65] libpng_jll v1.6.43+1
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [009596ad] mtdev_jll v1.1.6+0
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+1
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
  [e66e0078] CompilerSupportLibraries_jll v1.1.0+0
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
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

