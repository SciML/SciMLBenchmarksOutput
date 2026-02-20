---
author: "Chris Rackauckas and Yingbo Ma"
title: "Differentiation of Simple ODE Benchmarks"
---


From the paper [A Comparison of Automatic Differentiation and Continuous Sensitivity Analysis for Derivatives of Differential Equation Solutions](https://ieeexplore.ieee.org/abstract/document/9622796)

```julia
using ParameterizedFunctions, OrdinaryDiffEq, LinearAlgebra, StaticArrays
using SciMLSensitivity, ForwardDiff, FiniteDiff, ReverseDiff, BenchmarkTools, Test
using DataFrames, PrettyTables, Markdown
tols = (abstol = 1e-5, reltol = 1e-7)
```

```
(abstol = 1.0e-5, reltol = 1.0e-7)
```





## Define the Test ODEs

```julia
function lvdf(du, u, p, t)
    a, b, c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

function lvcom_df(du, u, p, t)
    a, b, c = p
    x, y, s1, s2, s3, s4, s5, s6 = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    #####################
    #     [a-by -bx]
    # J = [        ]
    #     [y    x-c]
    #####################
    J = @SMatrix [a-b*y -b*x
                  y x-c]
    JS = J*@SMatrix[s1 s3 s5
                    s2 s4 s6]
    G = @SMatrix [x -x*y 0
                  0 0 -y]
    du[3:end] .= vec(JS+G)
    nothing
end

lvdf_with_jacobian = ODEFunction{true, SciMLBase.FullSpecialize}(lvdf, jac = (
    J, u, p, t)->begin
    a, b, c = p
    x, y = u
    J[1] = a-b*y
    J[2] = y
    J[3] = -b*x
    J[4] = x-c
    nothing
end)

u0 = [1.0, 1.0];
tspan = (0.0, 10.0);
p = [1.5, 1.0, 3.0];
lvcom_u0 = [u0...; zeros(6)]
lvprob = ODEProblem{true, SciMLBase.FullSpecialize}(lvcom_df, lvcom_u0, tspan, p)
```

```
ODEProblem with uType Vector{Float64} and tType Float64. In-place: true
Non-trivial mass matrix: false
timespan: (0.0, 10.0)
u0: 8-element Vector{Float64}:
 1.0
 1.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
```



```julia
pkpdf = @ode_def begin
    dEv = -Ka1*Ev
    dCent = Ka1*Ev - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc) + Q*(Periph/Vp) - Q2*(Cent/Vc) +
            Q2*(Periph2/Vp2)
    dPeriph = Q*(Cent/Vc) - Q*(Periph/Vp)
    dPeriph2 = Q2*(Cent/Vc) - Q2*(Periph2/Vp2)
    dResp = Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ))) - Kout*Resp
end Ka1 CL Vc Q Vp Kin Kout IC50 IMAX γ Vmax Km Q2 Vp2

pkpdp = [
    1, # Ka1  Absorption rate constant 1 (1/time)
    1, # CL   Clearance (volume/time)
    20, # Vc   Central volume (volume)
    2, # Q    Inter-compartmental clearance (volume/time)
    10, # Vp   Peripheral volume of distribution (volume)
    10, # Kin  Response in rate constant (1/time)
    2, # Kout Response out rate constant (1/time)
    2, # IC50 Concentration for 50% of max inhibition (mass/volume)
    1, # IMAX Maximum inhibition
    1, # γ    Emax model sigmoidicity
    0, # Vmax Maximum reaction velocity (mass/time)
    2,  # Km   Michaelis constant (mass/volume)
    0.5, # Q2    Inter-compartmental clearance2 (volume/time)
    100 # Vp2   Peripheral2 volume of distribution (volume)
];

pkpdu0 = [100, eps(), eps(), eps(), 5.0] # exact zero in the initial condition triggers NaN in Jacobian
#pkpdu0 = ones(5)
pkpdcondition = function (u, t, integrator)
    t in 0:24:240
end
pkpdaffect! = function (integrator)
    integrator.u[1] += 100
end
pkpdcb = DiscreteCallback(pkpdcondition, pkpdaffect!, save_positions = (false, true))
pkpdtspan = (0.0, 240.0)
pkpdprob = ODEProblem{true, SciMLBase.FullSpecialize}(pkpdf.f, pkpdu0, pkpdtspan, pkpdp)

pkpdfcomp = let pkpdf=pkpdf, J=zeros(5, 5), JP=zeros(5, 14), tmpdu=zeros(5, 14)
    function (du, u, p, t)
        pkpdf.f(@view(du[:, 1]), u, p, t)
        pkpdf.jac(J, u, p, t)
        pkpdf.paramjac(JP, u, p, t)
        mul!(tmpdu, J, @view(u[:, 2:end]))
        du[:, 2:end] .= tmpdu .+ JP
        nothing
    end
end
pkpdcompprob = ODEProblem{true, SciMLBase.FullSpecialize}(
    pkpdfcomp, hcat(pkpdprob.u0, zeros(5, 14)), pkpdprob.tspan, pkpdprob.p)
```

```
ODEProblem with uType Matrix{Float64} and tType Float64. In-place: true
Non-trivial mass matrix: false
timespan: (0.0, 240.0)
u0: 5×15 Matrix{Float64}:
 100.0          0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0
.0
   2.22045e-16  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0
.0
   2.22045e-16  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0
.0
   2.22045e-16  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0
.0
   5.0          0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0
.0
```



```julia
pollution = @ode_def begin
    dy1 = -k1 * y1-k10*y11*y1-k14*y1*y6-k23*y1*y4-k24*y19*y1+
          k2 * y2 * y4+k3 * y5 * y2+k9 * y11 * y2+k11*y13+k12*y10*y2+k22*y19+k25*y20
    dy2 = -k2 * y2 * y4-k3 * y5 * y2-k9 * y11 * y2-k12*y10*y2+k1 * y1+k21*y19
    dy3 = -k15*y3+k1 * y1+k17*y4+k19*y16+k22*y19
    dy4 = -k2 * y2 * y4-k16*y4-k17*y4-k23*y1*y4+k15*y3
    dy5 = -k3 * y5 * y2+k4 * y7+k4 * y7+k6 * y7 * y6+k7 * y9+k13*y14+k20*y17*y6
    dy6 = -k6 * y7 * y6-k8 * y9 * y6-k14*y1*y6-k20*y17*y6+k3 * y5 * y2+k18*y16+k18*y16
    dy7 = -k4 * y7-k5 * y7-k6 * y7 * y6+k13*y14
    dy8 = k4 * y7+k5 * y7+k6 * y7 * y6+k7 * y9
    dy9 = -k7 * y9-k8 * y9 * y6
    dy10 = -k12*y10*y2+k7 * y9+k9 * y11 * y2
    dy11 = -k9 * y11 * y2-k10*y11*y1+k8 * y9 * y6+k11*y13
    dy12 = k9 * y11 * y2
    dy13 = -k11*y13+k10*y11*y1
    dy14 = -k13*y14+k12*y10*y2
    dy15 = k14*y1*y6
    dy16 = -k18*y16-k19*y16+k16*y4
    dy17 = -k20*y17*y6
    dy18 = k20*y17*y6
    dy19 = -k21*y19-k22*y19-k24*y19*y1+k23*y1*y4+k25*y20
    dy20 = -k25*y20+k24*y19*y1
end k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19 k20 k21 k22 k23 k24 k25

function make_pollution()
    comp = let pollution = pollution, J = zeros(20, 20), JP = zeros(20, 25),
        tmpdu = zeros(20, 25), tmpu = zeros(20, 25)

        function comp(du, u, p, t)
            tmpu .= @view(u[:, 2:26])
            pollution(@view(du[:, 1]), u, p, t)
            pollution.jac(J, u, p, t)
            pollution.paramjac(JP, u, p, t)
            mul!(tmpdu, J, tmpu)
            du[:, 2:26] .= tmpdu .+ JP
            nothing
        end
    end

    u0 = zeros(20)
    p = [.35e0, .266e2, .123e5, .86e-3, .82e-3, .15e5, .13e-3, .24e5, .165e5,
        .9e4, .22e-1, .12e5, .188e1, .163e5, .48e7, .35e-3, .175e-1,
        .1e9, .444e12, .124e4, .21e1, .578e1, .474e-1, .178e4, .312e1]
    u0[2] = 0.2
    u0[4] = 0.04
    u0[7] = 0.1
    u0[8] = 0.3
    u0[9] = 0.01
    u0[17] = 0.007
    compu0 = zeros(20, 26)
    compu0[1:20] .= u0
    comp, u0, p, compu0
end
```

```
make_pollution (generic function with 1 method)
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
    ODEProblem{true, SciMLBase.FullSpecialize}(
        brusselator_comp, copy([u0; zeros((N^2*2)*(N^2*4))]), (0.0, 10.0), p)
end
```

```
makebrusselator (generic function with 2 methods)
```





## Differentiation Setups

```julia
function diffeq_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
    diffeq_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end
function auto_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
    auto_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end

function diffeq_sen(
        f, u0, tspan, p, alg = Tsit5(); sensalg = ForwardSensitivity(), kwargs...)
    prob = ODEForwardSensitivityProblem(f, u0, tspan, p, sensalg)
    sol = solve(prob, alg; save_everystep = false, kwargs...)
    extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, u0, tspan, p, alg = Tsit5(); kwargs...)
    test_f(p) = begin
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, eltype(p).(u0), tspan, p)
        solve(prob, alg; save_everystep = false, kwargs...)[end]
    end
    ForwardDiff.jacobian(test_f, p)
end

function numerical_sen(f, u0, tspan, p, alg = Tsit5(); kwargs...)
    test_f(out, p) = begin
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, eltype(p).(u0), tspan, p)
        copyto!(out, solve(prob, alg; kwargs...)[end])
    end
    J = Matrix{Float64}(undef, length(u0), length(p))
    FiniteDiff.finite_difference_jacobian!(
        J, test_f, p, FiniteDiff.JacobianCache(p, Array{Float64}(undef, length(u0))))
    return J
end

function diffeq_sen_l2(df, u0, tspan, p, t, alg = Tsit5();
        abstol = 1e-5, reltol = 1e-7,
        sensalg = InterpolatingAdjoint(), kwargs...)
    prob = ODEProblem(df, u0, tspan, p)
    sol = solve(prob, alg, sensealg = DiffEqBase.SensitivityADPassThrough(),
        abstol = abstol, reltol = reltol; kwargs...)
    dg(out, u, p, t, i) = (out.=u .- 1.0)
    adjoint_sensitivities(sol, alg; t, abstol = abstol, dgdu_discrete = dg,
        reltol = reltol, sensealg = sensalg)[2]
end

function auto_sen_l2(
        f, u0, tspan, p, t, alg = Tsit5(); diffalg = ReverseDiff.gradient, kwargs...)
    test_f(p) = begin
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, eltype(p).(u0), tspan, p)
        sol = solve(prob, alg; sensealg = DiffEqBase.SensitivityADPassThrough(), kwargs...)(t)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    diffalg(test_f, p)
end

function numerical_sen_l2(f, u0, tspan, p, t, alg = Tsit5(); kwargs...)
    test_f(p) = begin
        prob = ODEProblem(f, eltype(p).(u0), tspan, p)
        sol = solve(prob, alg; kwargs...)(t)
        sum(sol.u) do x
            sum(z->(1-z)^2/2, x)
        end
    end
    FiniteDiff.finite_difference_gradient(test_f, p, Val{:central})
end
```

```
Error: UndefVarError: `DiffEqBase` not defined
```



```julia
_adjoint_methods = ntuple(3) do ii
    Alg = (InterpolatingAdjoint, QuadratureAdjoint, BacksolveAdjoint)[ii]
    (
        user = Alg(autodiff = false, autojacvec = false), # user Jacobian
        adjc = Alg(autodiff = true, autojacvec = false), # AD Jacobian
        advj = Alg(autodiff = true, autojacvec = EnzymeVJP()) # AD vJ
    )
end |> NamedTuple{(:interp, :quad, :backsol)}
@isdefined(ADJOINT_METHODS) ||
    (const ADJOINT_METHODS = mapreduce(collect, vcat, _adjoint_methods))
```

```
9-element Vector{SciMLBase.AbstractAdjointSensitivityAlgorithm{0, AD, Val{:
central}} where AD}:
 SciMLSensitivity.InterpolatingAdjoint{0, false, Val{:central}, Bool}(false
, false, false)
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, Bool}(false,
 false, false)
 SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensiti
vity.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIAB
I, false, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false,
 false, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{
false, false, false, EnzymeCore.FFIABI, false, false}()), false, false)
 SciMLSensitivity.QuadratureAdjoint{0, false, Val{:central}, Bool}(false, 1
.0e-6, 0.001)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, Bool}(false, 1.
0e-6, 0.001)
 SciMLSensitivity.QuadratureAdjoint{0, true, Val{:central}, SciMLSensitivit
y.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIABI, 
false, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false, fa
lse, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{fal
se, false, false, EnzymeCore.FFIABI, false, false}()), 1.0e-6, 0.001)
 SciMLSensitivity.BacksolveAdjoint{0, false, Val{:central}, Bool}(false, tr
ue, false)
 SciMLSensitivity.BacksolveAdjoint{0, true, Val{:central}, Bool}(false, tru
e, false)
 SciMLSensitivity.BacksolveAdjoint{0, true, Val{:central}, SciMLSensitivity
.EnzymeVJP{EnzymeCore.ReverseMode{false, false, false, EnzymeCore.FFIABI, f
alse, false}}}(SciMLSensitivity.EnzymeVJP{EnzymeCore.ReverseMode{false, fal
se, false, EnzymeCore.FFIABI, false, false}}(0, EnzymeCore.ReverseMode{fals
e, false, false, EnzymeCore.FFIABI, false, false}()), true, false)
```





## Run Forward Mode Benchmarks

These are testing for the construction of the full Jacobian.

```julia
forward_lv = let
    @info "Running the Lotka-Volterra model:"
    @info "  Running compile-time CSA"
    t1 = @belapsed solve($lvprob, $(Tsit5()); $tols...)
    @info "  Running DSA"
    t2 = @belapsed auto_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); $tols...)
    @info "  Running CSA user-Jacobian"
    t3 = @belapsed diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Tsit5());
        sensalg = ForwardSensitivity(autodiff = false, autojacvec = false), $tols...)
    @info "  Running AD-Jacobian"
    t4 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5());
        sensalg = ForwardSensitivity(autojacvec = false), $tols...)
    @info "  Running AD-Jv seeding"
    t5 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5());
        sensalg = ForwardSensitivity(autojacvec = true), $tols...)
    @info "  Running numerical differentiation"
    t6 = @belapsed numerical_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); $tols...)
    print('\n')
    [t1, t2, t3, t4, t5, t6]
end
```

```
Error: UndefVarError: `auto_sen` not defined
```



```julia
forward_bruss = let
    @info "Running the Brusselator model:"
    n = 5
    # Run low tolerance to test correctness
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
    sol1 = @time numerical_sen(
        bfun, b_u0, (0.0, 10.0), b_p, Rodas5(), abstol = 1e-5, reltol = 1e-7);
    sol2 = @time auto_sen(
        bfun, b_u0, (0.0, 10.0), b_p, Rodas5(), abstol = 1e-5, reltol = 1e-7);
    @test sol1 ≈ sol2 atol=1e-2
    sol3 = @time diffeq_sen(bfun, b_u0, (0.0, 10.0), b_p, Rodas5(autodiff = false),
        abstol = 1e-5, reltol = 1e-7);
    @test sol1 ≈ hcat(sol3...) atol=1e-3
    sol4 = @time diffeq_sen(
        ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac = brusselator_jac), b_u0,
        (0.0, 10.0), b_p, Rodas5(autodiff = false), abstol = 1e-5, reltol = 1e-7,
        sensalg = ForwardSensitivity(autodiff = false, autojacvec = false));
    @test sol1 ≈ hcat(sol4...) atol=1e-2
    sol5 = @time solve(brusselator_comp, Rodas5(autodiff = false), abstol = 1e-5, reltol = 1e-7);
    @test sol1 ≈ reshape(sol5[end][(2n * n + 1):end], 2n*n, 4n*n) atol=1e-3

    # High tolerance to benchmark
    @info "  Running compile-time CSA"
    t1 = @belapsed solve($brusselator_comp, $(Rodas5(autodiff = false)); $tols...);
    @info "  Running DSA"
    t2 = @belapsed auto_sen($bfun, $b_u0, $((0.0, 10.0)), $b_p, $(Rodas5()); $tols...);
    @info "  Running CSA user-Jacobian"
    t3 = @belapsed diffeq_sen(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac = brusselator_jac)),
        $b_u0, $((0.0, 10.0)), $b_p, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autodiff = false, autojacvec = false), $tols...);
    @info "  Running AD-Jacobian"
    t4 = @belapsed diffeq_sen(
        $bfun, $b_u0, $((0.0, 10.0)), $b_p, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autojacvec = false), $tols...);
    @info "  Running AD-Jv seeding"
    t5 = @belapsed diffeq_sen(
        $bfun, $b_u0, $((0.0, 10.0)), $b_p, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autojacvec = true), $tols...);
    @info "  Running numerical differentiation"
    t6 = @belapsed numerical_sen($bfun, $b_u0, $((0.0, 10.0)), $b_p, $(Rodas5()); $tols...);
    print('\n')
    [t1, t2, t3, t4, t5, t6]
end
```

```
Error: UndefVarError: `numerical_sen` not defined
```



```julia
forward_pollution = let
    @info "Running the pollution model:"
    pcomp, pu0, pp, pcompu0 = make_pollution()
    ptspan = (0.0, 60.0)
    @info "  Running compile-time CSA"
    t1 = 0#@belapsed solve($(ODEProblem(pcomp, pcompu0, ptspan, pp)), $(Rodas5(autodiff=false)),);
    @info "  Running DSA"
    t2 = @belapsed auto_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        $pu0, $ptspan, $pp, $(Rodas5()); $tols...);
    @info "  Running CSA user-Jacobian"
    t3 = @belapsed diffeq_sen(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac = pollution.jac)),
        $pu0, $ptspan, $pp, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autodiff = false, autojacvec = false), $tols...);
    @info "  Running AD-Jacobian"
    t4 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        $pu0, $ptspan, $pp, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autojacvec = false), $tols...);
    @info "  Running AD-Jv seeding"
    t5 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        $pu0, $ptspan, $pp, $(Rodas5(autodiff = false));
        sensalg = ForwardSensitivity(autojacvec = true), $tols...);
    @info "  Running numerical differentiation"
    t6 = @belapsed numerical_sen(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        $pu0, $ptspan, $pp, $(Rodas5()); $tols...);
    print('\n')
    [t1, t2, t3, t4, t5, t6]
end
```

```
Error: UndefVarError: `auto_sen` not defined
```



```julia
forward_pkpd = let
    @info "Running the PKPD model:"
    #sol1 = solve(pkpdcompprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240,)[end][6:end]
    sol2 = vec(auto_sen(pkpdprob, Tsit5(), abstol = 1e-5, reltol = 1e-7,
        callback = pkpdcb, tstops = 0:24:240))
    sol3 = vec(hcat(diffeq_sen(pkpdprob, Tsit5(), abstol = 1e-5, reltol = 1e-7,
        callback = pkpdcb, tstops = 0:24:240)...))
    #@test sol1 ≈ sol2 atol=1e-3
    @test sol2 ≈ sol3 atol=1e-3
    @info "  Running compile-time CSA"
    #t1 = @belapsed solve($pkpdcompprob, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240,);
    @info "  Running DSA"
    t2 = @belapsed auto_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5());
        callback = $pkpdcb, tstops = 0:24:240, $tols...);
    @info "  Running CSA user-Jacobian"
    t3 = @belapsed diffeq_sen(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac = pkpdf.jac)),
        $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()); callback = $pkpdcb, tstops = 0:24:240,
        sensalg = ForwardSensitivity(autodiff = false, autojacvec = false), $tols...);
    @info "  Running AD-Jacobian"
    t4 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp,
        $(Tsit5()); callback = $pkpdcb, tstops = 0:24:240,
        sensalg = ForwardSensitivity(autojacvec = false), $tols...);
    @info "  Running AD-Jv seeding"
    t5 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp,
        $(Tsit5()); callback = $pkpdcb, tstops = 0:24:240,
        sensalg = ForwardSensitivity(autojacvec = true), $tols...);
    @info "  Running numerical differentiation"
    t6 = @belapsed numerical_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5());
        callback = $pkpdcb, tstops = 0:24:240, $tols...);
    print('\n')
    [0, t2, t3, t4, t5, t6]
end
```

```
Error: UndefVarError: `auto_sen` not defined
```



```julia
forward_methods = ["Compile-time CSA", "DSA", "CSA user-Jacobian",
    "AD-Jacobian", "AD-Jv seeding", "Numerical Differentiation"]
forward_timings = DataFrame(
    methods = forward_methods, LV = forward_lv, Bruss = forward_bruss,
    Pollution = forward_pollution, PKPD = forward_pkpd)
display(forward_timings)
```

```
Error: UndefVarError: `forward_lv` not defined
```





## Run Adjoint Benchmarks

Adjoint requires a slightly different setup even with forward mode ADs since it requires
a loss function choice. For that we simply take the L2 norm of the solution.

```julia
adjoint_lv = let
    @info "Running the Lotka-Volerra model:"
    lvu0 = [1.0, 1.0];
    lvtspan = (0.0, 10.0);
    lvp = [1.5, 1.0, 3.0];
    lvt = 0:0.5:10
    @time lsol1 = auto_sen_l2(
        lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg = (ForwardDiff.gradient), tols...);
    @time lsol2 = auto_sen_l2(
        lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg = (ReverseDiff.gradient), tols...);
    @time lsol3 = map(ADJOINT_METHODS) do alg
        f = SciMLSensitivity.alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
        diffeq_sen_l2(f, lvu0, lvtspan, lvp, lvt, (Tsit5()); sensalg = alg, tols...)
    end
    @time lsol4 = numerical_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, Tsit5(); tols...);
    @test maximum(abs, lsol1 .- lsol2)/maximum(abs, lsol1) < 0.2
    @test all(i -> maximum(abs, lsol1 .- lsol3[i]')/maximum(abs, lsol1) < 0.2, eachindex(ADJOINT_METHODS))
    @test maximum(abs, lsol1 .- lsol4)/maximum(abs, lsol1) < 0.2
    t1 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5());
        diffalg = $(ForwardDiff.gradient), $tols...);
    t2 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5());
        diffalg = $(ReverseDiff.gradient), $tols...);
    t3 = map(ADJOINT_METHODS) do alg
        f = SciMLSensitivity.alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
        @belapsed diffeq_sen_l2(
            $f, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); sensalg = $alg, $tols...);
    end
    t4 = @belapsed numerical_sen_l2(
        $lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); $tols...);
    [t1; t2; t3; t4]
end
```

```
Error: UndefVarError: `auto_sen_l2` not defined
```



```julia
adjoint_bruss = let
    @info "Running the Brusselator model:"
    bt = 0:0.1:10
    tspan = (0.0, 10.0)
    n = 5
    bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
    @time bsol1 = auto_sen_l2(
        bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg = (ForwardDiff.gradient), tols...);
    #@time bsol2 = auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
    #@test maximum(abs, bsol1 .- bsol2)/maximum(abs,  bsol1) < 1e-2

    @time bsol3 = map(ADJOINT_METHODS) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac = brusselator_jac)
        solver = Rodas5(autodiff = false)
        diffeq_sen_l2(
            f, b_u0, tspan, b_p, bt, solver, reltol = 1e-7; sensalg = alg, tols...)
    end
    @time bsol4 = numerical_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); tols...);
    # NOTE: backsolve gives unstable results!!!
    @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) < 4e-2,
        eachindex(ADJOINT_METHODS)[1:(2end ÷ 3)])
    @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) >= 4e-2,
        eachindex(ADJOINT_METHODS)[(2end ÷ 3 + 1):end])
    @test maximum(abs, bsol1 .- bsol4)/maximum(abs, bsol1) < 2e-2
    t1 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5());
        diffalg = $(ForwardDiff.gradient), $tols...);
    #t2 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
    t2 = NaN
    t3 = map(ADJOINT_METHODS[1:(2end ÷ 3)]) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? bfun :
            ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac = brusselator_jac)
        solver = Rodas5(autodiff = false)
        @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg = alg, tols...);
    end
    t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
    t4 = @belapsed numerical_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()); $tols...);
    [t1; t2; t3; t4]
end
```

```
Error: UndefVarError: `auto_sen_l2` not defined
```



```julia
adjoint_pollution = let
    @info "Running the Pollution model:"
    pcomp, pu0, pp, pcompu0 = make_pollution();
    ptspan = (0.0, 60.0)
    pts = 0:0.5:60
    @time psol1 = auto_sen_l2(
        (ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), pu0, ptspan, pp,
        pts, (Rodas5(autodiff = false)); diffalg = (ForwardDiff.gradient), tols...);
    #@time psol2 = auto_sen_l2((ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
    #@test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
    @time psol3 = map(ADJOINT_METHODS) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? pollution.f :
            ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac = pollution.jac)
        solver = Rodas5(autodiff = false)
        diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg = alg, tols...);
    end
    @time psol4 = numerical_sen_l2(
        (ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        pu0, ptspan, pp, pts, (Rodas5(autodiff = false)); tols...);
    # NOTE: backsolve gives unstable results!!!
    @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) < 1e-2,
        eachindex(ADJOINT_METHODS)[1:(2end ÷ 3)])
    @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) >= 1e-2,
        eachindex(ADJOINT_METHODS)[(2end ÷ 3 + 1):end])
    @test maximum(abs, psol1 .- psol4)/maximum(abs, psol1) < 1e-2
    t1 = @belapsed auto_sen_l2(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp,
        $pts, $(Rodas5(autodiff = false)); diffalg = $(ForwardDiff.gradient), $tols...);
    #t2 = @belapsed auto_sen_l2($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
    t2 = NaN
    t3 = map(ADJOINT_METHODS[1:(2end ÷ 3)]) do alg
        @info "Running $alg"
        f = SciMLSensitivity.alg_autodiff(alg) ? pollution.f :
            ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac = pollution.jac)
        solver = Rodas5(autodiff = false)
        @elapsed diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg = alg, tols...);
    end
    t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
    t4 = @belapsed numerical_sen_l2(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)),
        $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff = false)); $tols...);
    [t1; t2; t3; t4]
end
```

```
Error: UndefVarError: `auto_sen_l2` not defined
```



```julia
adjoint_pkpd = let
    @info "Running the PKPD model:"
    pts = 0:0.5:50
    # need to use lower tolerances to avoid running into the complex domain because of exponentiation
    pkpdsol1 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts,
        (Tsit5()); callback = pkpdcb, tstops = 0:24:240,
        diffalg = (ForwardDiff.gradient), tols...);
    pkpdsol2 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts,
        (Tsit5()); callback = pkpdcb, tstops = 0:24:240,
        diffalg = (ReverseDiff.gradient), tols...);
    pkpdsol3 = @time map(ADJOINT_METHODS[1:(2end ÷ 3)]) do alg
        f = SciMLSensitivity.alg_autodiff(alg) ? pkpdf.f :
            ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac = pkpdf.jac)
        diffeq_sen_l2(f, pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); sensalg = alg,
            callback = pkpdcb, tstops = 0:24:240, tols...);
    end
    pkpdsol4 = @time numerical_sen_l2(
        (ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f)),
        pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5());
        callback = pkpdcb, tstops = 0:24:240, tols...);
    @test maximum(abs, pkpdsol1 .- pkpdsol2)/maximum(abs, pkpdsol1) < 0.2
    @test all(i->maximum(abs, pkpdsol1 .- pkpdsol3[i]')/maximum(abs, pkpdsol1) < 0.2,
        eachindex(ADJOINT_METHODS)[1:(2end ÷ 3)])
    @test maximum(abs, pkpdsol1 .- pkpdsol4)/maximum(abs, pkpdsol1) < 0.2
    t1 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts,
        $(Tsit5()); callback = pkpdcb, tstops = 0:24:240,
        diffalg = $(ForwardDiff.gradient), $tols...);
    t2 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts,
        $(Tsit5()); callback = pkpdcb, tstops = 0:24:240,
        diffalg = $(ReverseDiff.gradient), $tols...);
    t3 = map(ADJOINT_METHODS[1:(2end ÷ 3)]) do alg
        f = SciMLSensitivity.alg_autodiff(alg) ? pkpdf.f :
            ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac = pkpdf.jac)
        @belapsed diffeq_sen_l2(
            $f, $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops = 0:24:240,
            callback = pkpdcb, sensalg = $alg, tols...);
    end
    t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
    t4 = @belapsed numerical_sen_l2(
        $(ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f)), $pkpdu0,
        $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops = 0:24:240,
        callback = $pkpdcb, $tols...);
    [t1; t2; t3; t4]
end
```

```
Error: UndefVarError: `auto_sen_l2` not defined
```



```julia
adjoint_methods = ["ForwardDiff", "ReverseDiff",
    "InterpolatingAdjoint User Jac", "InterpolatingAdjoint AD Jac", "InterpolatingAdjoint v'J",
    "QuadratureAdjoint User Jac", "QuadratureAdjoint AD Jac", "QuadratureAdjoint v'J",
    "BacksolveAdjoint User Jac", "BacksolveAdjoint AD Jac", "BacksolveAdjoint v'J",
    "Numerical Differentiation"]
adjoint_timings = DataFrame(
    methods = adjoint_methods, LV = adjoint_lv, Bruss = adjoint_bruss,
    Pollution = adjoint_pollution, PKPD = adjoint_pkpd)
Markdown.parse(PrettyTables.pretty_table(
    String, adjoint_timings; backend = Val(:markdown), header = names(adjoint_timings)))
```

```
Error: UndefVarError: `adjoint_lv` not defined
```





## Appendix


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AutomaticDifferentiation","SimpleODEAD.jmd")
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
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Project.toml`
  [6e4b80f9] BenchmarkTools v1.6.3
  [a93c6f00] DataFrames v1.8.1
  [1313f7d8] DataFramesMeta v0.15.6
  [a0c0ee7d] DifferentiationInterface v0.7.16
  [a82114a7] DifferentiationInterfaceTest v0.11.0
  [7da242da] Enzyme v0.13.129
  [6a86dc24] FiniteDiff v2.29.0
  [f6369f11] ForwardDiff v1.3.2
  [da2b9cff] Mooncake v0.5.6
  [1dea7af3] OrdinaryDiffEq v6.108.0
  [65888b18] ParameterizedFunctions v5.22.0
  [91a5bcdd] Plots v1.41.6
⌅ [08abe8d2] PrettyTables v2.4.0
  [37e2e3b7] ReverseDiff v1.16.2
  [31c91b34] SciMLBenchmarks v0.1.3
  [1ed8b502] SciMLSensitivity v7.96.0
  [90137ffa] StaticArrays v1.9.16
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
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/AutomaticDifferentiation/Manifest.toml`
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
  [70df07ce] BracketingNonlinearSolve v1.8.0
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
  [2b5f629d] DiffEqBase v6.205.1
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
  [ccbc3e58] JumpProcesses v9.22.0
  [63c18a36] KernelAbstractions v0.9.40
  [ba0b0d4f] Krylov v0.10.5
  [929cbde3] LLVM v9.4.6
  [b964fa9f] LaTeXStrings v1.4.0
  [23fbe1c1] Latexify v0.16.10
  [10f19ff3] LayoutPointers v0.1.17
  [87fe0de2] LineSearch v0.1.6
  [d3d80556] LineSearches v7.6.0
⌃ [7ed4a6bd] LinearSolve v3.58.0
  [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
⌃ [961ee093] ModelingToolkit v11.10.0
⌃ [7771a370] ModelingToolkitBase v1.13.1
  [6bb917b9] ModelingToolkitTearing v1.3.1
  [da2b9cff] Mooncake v0.5.6
  [2e0e35c7] Moshi v0.3.7
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.13
  [ffc61752] Mustache v1.0.21
  [d8a4904e] MutableArithmetics v1.6.7
  [d41bc354] NLSolversBase v8.0.0
  [872c559c] NNlib v0.9.33
  [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.15.0
  [be0214bd] NonlinearSolveBase v2.12.0
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
  [6ad6398a] OrdinaryDiffEqBDF v1.16.0
  [bbf590c4] OrdinaryDiffEqCore v3.5.2
  [50262376] OrdinaryDiffEqDefault v1.12.0
  [4302a76b] OrdinaryDiffEqDifferentiation v2.0.0
  [9286f039] OrdinaryDiffEqExplicitRK v1.9.0
  [e0540318] OrdinaryDiffEqExponentialRK v1.13.0
  [becaefa8] OrdinaryDiffEqExtrapolation v1.15.0
  [5960d6e9] OrdinaryDiffEqFIRK v1.22.0
  [101fe9f7] OrdinaryDiffEqFeagin v1.8.0
  [d3585ca7] OrdinaryDiffEqFunctionMap v1.9.0
  [d28bc4f8] OrdinaryDiffEqHighOrderRK v1.9.0
  [9f002381] OrdinaryDiffEqIMEXMultistep v1.12.0
  [521117fe] OrdinaryDiffEqLinear v1.10.0
  [1344f307] OrdinaryDiffEqLowOrderRK v1.10.0
  [b0944070] OrdinaryDiffEqLowStorageRK v1.12.0
  [127b3ac7] OrdinaryDiffEqNonlinearSolve v1.20.0
  [c9986a66] OrdinaryDiffEqNordsieck v1.9.0
  [5dd0a6cf] OrdinaryDiffEqPDIRK v1.11.0
  [5b33eab2] OrdinaryDiffEqPRK v1.8.0
  [04162be5] OrdinaryDiffEqQPRK v1.8.0
  [af6ede74] OrdinaryDiffEqRKN v1.9.0
  [43230ef6] OrdinaryDiffEqRosenbrock v1.23.0
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
  [43287f4e] PtrArrays v1.3.0
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
  [0bca4576] SciMLBase v2.139.0
  [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.12
⌃ [a6db7da4] SciMLLogging v1.9.0
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
  [727e6d20] SimpleNonlinearSolve v2.10.0
  [699a6c99] SimpleTraits v0.9.5
  [a2af1166] SortingAlgorithms v1.2.2
  [dc90abb0] SparseInverseSubset v0.1.2
  [0a514795] SparseMatrixColorings v0.4.23
  [276daf66] SpecialFunctions v2.7.1
  [860ef19b] StableRNGs v1.0.4
  [64909d44] StateSelection v1.3.0
  [aedffcd0] Static v1.3.1
  [0d7ed370] StaticArrayInterface v1.9.0
  [90137ffa] StaticArrays v1.9.16
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
  [d1185830] SymbolicUtils v4.18.2
  [0c5d862f] Symbolics v7.15.1
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
  [629bc702] Qt6Declarative_jll v6.8.2+1
  [ce943373] Qt6ShaderTools_jll v6.8.2+1
  [e99dba38] Qt6Wayland_jll v6.8.2+2
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

