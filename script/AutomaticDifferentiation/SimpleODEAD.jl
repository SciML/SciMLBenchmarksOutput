
using ParameterizedFunctions, OrdinaryDiffEq, LinearAlgebra, StaticArrays
using SciMLSensitivity, ForwardDiff, FiniteDiff, ReverseDiff, BenchmarkTools, Test
using DataFrames, PrettyTables, Markdown
tols = (abstol=1e-5, reltol=1e-7)


function lvdf(du, u, p, t)
    a,b,c = p
    x, y = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    nothing
end

function lvcom_df(du, u, p, t)
    a,b,c = p
    x, y, s1, s2, s3, s4, s5, s6 = u
    du[1] = a*x - b*x*y
    du[2] = -c*y + x*y
    #####################
    #     [a-by -bx]
    # J = [        ]
    #     [y    x-c]
    #####################
    J  = @SMatrix [a-b*y -b*x
                   y    x-c]
    JS = J*@SMatrix[s1 s3 s5
                    s2 s4 s6]
    G  = @SMatrix [x -x*y 0
                   0  0  -y]
    du[3:end] .= vec(JS+G)
    nothing
end

lvdf_with_jacobian = ODEFunction{true, SciMLBase.FullSpecialize}(lvdf, jac=(J,u,p,t)->begin
                                   a,b,c = p
                                   x, y = u
                                   J[1] = a-b*y
                                   J[2] = y
                                   J[3] = -b*x
                                   J[4] = x-c
                                   nothing
                               end)

u0 = [1.,1.]; tspan = (0., 10.); p = [1.5,1.0,3.0]; lvcom_u0 = [u0...;zeros(6)]
lvprob = ODEProblem{true, SciMLBase.FullSpecialize}(lvcom_df, lvcom_u0, tspan, p)


pkpdf = @ode_def begin
  dEv      = -Ka1*Ev
  dCent    =  Ka1*Ev - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp) - Q2*(Cent/Vc)  + Q2*(Periph2/Vp2)
  dPeriph  =  Q*(Cent/Vc)  - Q*(Periph/Vp)
  dPeriph2 =  Q2*(Cent/Vc)  - Q2*(Periph2/Vp2)
  dResp   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
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

pkpdu0 = [100, eps(), eps(), eps(), 5.] # exact zero in the initial condition triggers NaN in Jacobian
#pkpdu0 = ones(5)
pkpdcondition = function (u,t,integrator)
  t in 0:24:240
end
pkpdaffect! = function (integrator)
  integrator.u[1] += 100
end
pkpdcb = DiscreteCallback(pkpdcondition, pkpdaffect!, save_positions=(false, true))
pkpdtspan = (0.,240.)
pkpdprob = ODEProblem{true, SciMLBase.FullSpecialize}(pkpdf.f, pkpdu0, pkpdtspan, pkpdp)

pkpdfcomp = let pkpdf=pkpdf, J=zeros(5,5), JP=zeros(5,14), tmpdu=zeros(5,14)
  function (du, u, p, t)
    pkpdf.f(@view(du[:, 1]), u, p, t)
    pkpdf.jac(J, u, p, t)
    pkpdf.paramjac(JP, u, p, t)
    mul!(tmpdu, J, @view(u[:, 2:end]))
    du[:, 2:end] .= tmpdu .+ JP
    nothing
  end
end
pkpdcompprob = ODEProblem{true, SciMLBase.FullSpecialize}(pkpdfcomp, hcat(pkpdprob.u0,zeros(5,14)), pkpdprob.tspan, pkpdprob.p)


pollution = @ode_def begin
  dy1  = -k1 *y1-k10*y11*y1-k14*y1*y6-k23*y1*y4-k24*y19*y1+
        k2 *y2*y4+k3 *y5*y2+k9 *y11*y2+k11*y13+k12*y10*y2+k22*y19+k25*y20
  dy2  = -k2 *y2*y4-k3 *y5*y2-k9 *y11*y2-k12*y10*y2+k1 *y1+k21*y19
  dy3  = -k15*y3+k1 *y1+k17*y4+k19*y16+k22*y19
  dy4  = -k2 *y2*y4-k16*y4-k17*y4-k23*y1*y4+k15*y3
  dy5  = -k3 *y5*y2+k4 *y7+k4 *y7+k6 *y7*y6+k7 *y9+k13*y14+k20*y17*y6
  dy6  = -k6 *y7*y6-k8 *y9*y6-k14*y1*y6-k20*y17*y6+k3 *y5*y2+k18*y16+k18*y16
  dy7  = -k4 *y7-k5 *y7-k6 *y7*y6+k13*y14
  dy8  = k4 *y7+k5 *y7+k6 *y7*y6+k7 *y9
  dy9  = -k7 *y9-k8 *y9*y6
  dy10 = -k12*y10*y2+k7 *y9+k9 *y11*y2
  dy11 = -k9 *y11*y2-k10*y11*y1+k8 *y9*y6+k11*y13
  dy12 = k9 *y11*y2
  dy13 = -k11*y13+k10*y11*y1
  dy14 = -k13*y14+k12*y10*y2
  dy15 = k14*y1*y6
  dy16 = -k18*y16-k19*y16+k16*y4
  dy17 = -k20*y17*y6
  dy18 = k20*y17*y6
  dy19 = -k21*y19-k22*y19-k24*y19*y1+k23*y1*y4+k25*y20
  dy20 = -k25*y20+k24*y19*y1
end k1  k2  k3  k4  k5  k6  k7  k8  k9  k10  k11  k12  k13  k14  k15  k16  k17  k18  k19  k20  k21  k22  k23  k24  k25

function make_pollution()
  comp = let pollution = pollution, J = zeros(20, 20), JP = zeros(20, 25), tmpdu = zeros(20,25), tmpu = zeros(20,25)
    function comp(du, u, p, t)
      tmpu  .= @view(u[:, 2:26])
      pollution(@view(du[:, 1]), u, p, t)
      pollution.jac(J,u,p,t)
      pollution.paramjac(JP,u,p,t)
      mul!(tmpdu, J, tmpu)
      du[:,2:26] .= tmpdu .+ JP
      nothing
    end
  end

  u0 = zeros(20)
  p = [.35e0, .266e2, .123e5, .86e-3, .82e-3, .15e5, .13e-3, .24e5, .165e5, .9e4, .22e-1, .12e5, .188e1, .163e5, .48e7, .35e-3, .175e-1, .1e9, .444e12, .124e4, .21e1, .578e1, .474e-1, .178e4, .312e1]
  u0[2]  = 0.2
  u0[4]  = 0.04
  u0[7]  = 0.1
  u0[8]  = 0.3
  u0[9]  = 0.01
  u0[17] = 0.007
  compu0 = zeros(20, 26)
  compu0[1:20] .= u0
  comp, u0, p, compu0
end


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
    brusselator_2d_loop, u0, p, brusselator_jac, ODEProblem{true, SciMLBase.FullSpecialize}(brusselator_comp, copy([u0;zeros((N^2*2)*(N^2*4))]), (0.,10.), p)
end


function diffeq_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
  diffeq_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end
function auto_sen(prob::DiffEqBase.DEProblem, args...; kwargs...)
  auto_sen(prob.f, prob.u0, prob.tspan, prob.p, args...; kwargs...)
end

function diffeq_sen(f, u0, tspan, p, alg=Tsit5(); sensalg=ForwardSensitivity(), kwargs...)
  prob = ODEForwardSensitivityProblem(f,u0,tspan,p,sensalg)
  sol = solve(prob,alg; save_everystep=false, kwargs...)
  extract_local_sensitivities(sol, length(sol))[2]
end

function auto_sen(f, u0, tspan, p, alg=Tsit5(); kwargs...)
  test_f(p) = begin
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,eltype(p).(u0),tspan,p)
    solve(prob,alg; save_everystep=false, kwargs...)[end]
  end
  ForwardDiff.jacobian(test_f, p)
end

function numerical_sen(f,u0, tspan, p, alg=Tsit5(); kwargs...)
  test_f(out,p) = begin
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,eltype(p).(u0),tspan,p)
    copyto!(out, solve(prob,alg; kwargs...)[end])
  end
  J = Matrix{Float64}(undef,length(u0),length(p))
  FiniteDiff.finite_difference_jacobian!(J, test_f, p, FiniteDiff.JacobianCache(p,Array{Float64}(undef,length(u0))))
  return J
end

function diffeq_sen_l2(df, u0, tspan, p, t, alg=Tsit5();
                       abstol=1e-5, reltol=1e-7,
                       sensalg=InterpolatingAdjoint(), kwargs...)
  prob = ODEProblem(df,u0,tspan,p)
  sol = solve(prob, alg, sensealg=DiffEqBase.SensitivityADPassThrough(), abstol=abstol, reltol=reltol; kwargs...)
  dg(out,u,p,t,i) = (out.=u.-1.0)
  adjoint_sensitivities(sol,alg;t,abstol=abstol,dgdu_discrete = dg,
                        reltol=reltol,sensealg=sensalg)[2]
end

function auto_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); diffalg=ReverseDiff.gradient, kwargs...)
  test_f(p) = begin
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f,eltype(p).(u0),tspan,p)
    sol = solve(prob,alg; sensealg=DiffEqBase.SensitivityADPassThrough(), kwargs...)(t)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  diffalg(test_f, p)
end

function numerical_sen_l2(f, u0, tspan, p, t, alg=Tsit5(); kwargs...)
  test_f(p) = begin
    prob = ODEProblem(f,eltype(p).(u0),tspan,p)
    sol = solve(prob,alg; kwargs...)(t)
    sum(sol.u) do x
      sum(z->(1-z)^2/2, x)
    end
  end
  FiniteDiff.finite_difference_gradient(test_f, p, Val{:central})
end


_adjoint_methods = ntuple(3) do ii
  Alg = (InterpolatingAdjoint, QuadratureAdjoint, BacksolveAdjoint)[ii]
  (
    user = Alg(autodiff=false,autojacvec=false), # user Jacobian
    adjc = Alg(autodiff=true,autojacvec=false), # AD Jacobian
    advj = Alg(autodiff=true,autojacvec=EnzymeVJP()), # AD vJ
  )
end |> NamedTuple{(:interp, :quad, :backsol)}
@isdefined(ADJOINT_METHODS) || (const ADJOINT_METHODS = mapreduce(collect, vcat, _adjoint_methods))


forward_lv = let
  @info "Running the Lotka-Volterra model:"
  @info "  Running compile-time CSA"
  t1 = @belapsed solve($lvprob, $(Tsit5()); $tols...)
  @info "  Running DSA"
  t2 = @belapsed auto_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); $tols...)
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($lvdf_with_jacobian, $u0, $tspan, $p, $(Tsit5()); sensalg=ForwardSensitivity(autodiff=false, autojacvec=false), $tols...)
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); sensalg=ForwardSensitivity(autojacvec=false), $tols...)
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); sensalg=ForwardSensitivity(autojacvec=true), $tols...)
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($lvdf, $u0, $tspan, $p, $(Tsit5()); $tols...)
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end


forward_bruss = let
  @info "Running the Brusselator model:"
  n = 5
  # Run low tolerance to test correctness
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  sol1 = @time numerical_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  sol2 = @time auto_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(), abstol=1e-5,reltol=1e-7);
  @test sol1 ≈ sol2 atol=1e-2
  sol3 = @time diffeq_sen(bfun, b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7);
  @test sol1 ≈ hcat(sol3...) atol=1e-3
  sol4 = @time diffeq_sen(ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac=brusselator_jac), b_u0, (0.,10.), b_p, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7, sensalg=ForwardSensitivity(autodiff=false, autojacvec=false));
  @test sol1 ≈ hcat(sol4...) atol=1e-2
  sol5 = @time solve(brusselator_comp, Rodas5(autodiff=false), abstol=1e-5,reltol=1e-7,);
  @test sol1 ≈ reshape(sol5[end][2n*n+1:end], 2n*n, 4n*n) atol=1e-3

  # High tolerance to benchmark
  @info "  Running compile-time CSA"
  t1 = @belapsed solve($brusselator_comp, $(Rodas5(autodiff=false)); $tols...);
  @info "  Running DSA"
  t2 = @belapsed auto_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5()); $tols...);
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac=brusselator_jac)), $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autodiff=false, autojacvec=false), $tols...);
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autojacvec=false), $tols...);
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autojacvec=true), $tols...);
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($bfun, $b_u0, $((0.,10.)), $b_p, $(Rodas5()); $tols...);
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end


forward_pollution = let
  @info "Running the pollution model:"
  pcomp, pu0, pp, pcompu0 = make_pollution()
  ptspan = (0.,60.)
  @info "  Running compile-time CSA"
  t1 = 0#@belapsed solve($(ODEProblem(pcomp, pcompu0, ptspan, pp)), $(Rodas5(autodiff=false)),);
  @info "  Running DSA"
  t2 = @belapsed auto_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()); $tols...);
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac=pollution.jac)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autodiff=false, autojacvec=false), $tols...);
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autojacvec=false), $tols...);
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5(autodiff=false)); sensalg=ForwardSensitivity(autojacvec=true), $tols...);
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $(Rodas5()); $tols...);
  print('\n')
  [t1, t2, t3, t4, t5, t6]
end


forward_pkpd = let
  @info "Running the PKPD model:"
  #sol1 = solve(pkpdcompprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240,)[end][6:end]
  sol2 = vec(auto_sen(pkpdprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240))
  sol3 = vec(hcat(diffeq_sen(pkpdprob, Tsit5(),abstol=1e-5,reltol=1e-7,callback=pkpdcb,tstops=0:24:240)...))
  #@test sol1 ≈ sol2 atol=1e-3
  @test sol2 ≈ sol3 atol=1e-3
  @info "  Running compile-time CSA"
  #t1 = @belapsed solve($pkpdcompprob, $(Tsit5()),callback=$pkpdcb,tstops=0:24:240,);
  @info "  Running DSA"
  t2 = @belapsed auto_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()); callback=$pkpdcb,tstops=0:24:240, $tols...);
  @info "  Running CSA user-Jacobian"
  t3 = @belapsed diffeq_sen($(ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac=pkpdf.jac)), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5());callback=$pkpdcb,tstops=0:24:240, sensalg=ForwardSensitivity(autodiff=false, autojacvec=false), $tols...);
  @info "  Running AD-Jacobian"
  t4 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()); callback=$pkpdcb,tstops=0:24:240,
                    sensalg=ForwardSensitivity(autojacvec=false), $tols...);
  @info "  Running AD-Jv seeding"
  t5 = @belapsed diffeq_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5());callback=$pkpdcb,tstops=0:24:240,
                         sensalg=ForwardSensitivity(autojacvec=true), $tols...);
  @info "  Running numerical differentiation"
  t6 = @belapsed numerical_sen($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $(Tsit5()); callback=$pkpdcb,tstops=0:24:240, $tols...);
  print('\n')
  [0, t2, t3, t4, t5, t6]
end


forward_methods = ["Compile-time CSA", "DSA", "CSA user-Jacobian", "AD-Jacobian", "AD-Jv seeding", "Numerical Differentiation"]
forward_timings = DataFrame(methods=forward_methods, LV=forward_lv, Bruss=forward_bruss, Pollution=forward_pollution, PKPD=forward_pkpd)
display(forward_timings)


adjoint_lv = let
  @info "Running the Lotka-Volerra model:"
  lvu0 = [1.,1.]; lvtspan = (0.0, 10.0); lvp = [1.5,1.0,3.0];
  lvt = 0:0.5:10
  @time lsol1 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ForwardDiff.gradient), tols...);
  @time lsol2 = auto_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, (Tsit5()); diffalg=(ReverseDiff.gradient), tols...);
  @time lsol3 = map(ADJOINT_METHODS) do alg
    f = SciMLSensitivity.alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
    diffeq_sen_l2(f, lvu0, lvtspan, lvp, lvt, (Tsit5()); sensalg=alg, tols...)
  end
  @time lsol4 = numerical_sen_l2(lvdf, lvu0, lvtspan, lvp, lvt, Tsit5(); tols...);
  @test maximum(abs, lsol1 .- lsol2)/maximum(abs,  lsol1) < 0.2
  @test all(i -> maximum(abs, lsol1 .- lsol3[i]')/maximum(abs, lsol1) < 0.2, eachindex(ADJOINT_METHODS))
  @test maximum(abs, lsol1 .- lsol4)/maximum(abs, lsol1) < 0.2
  t1 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(ADJOINT_METHODS) do alg
    f = SciMLSensitivity.alg_autodiff(alg) ? lvdf : lvdf_with_jacobian
    @belapsed diffeq_sen_l2($f, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); sensalg=$alg, $tols...);
  end
  t4 = @belapsed numerical_sen_l2($lvdf, $lvu0, $lvtspan, $lvp, $lvt, $(Tsit5()); $tols...);
  [t1; t2; t3; t4]
end


adjoint_bruss = let
  @info "Running the Brusselator model:"
  bt = 0:0.1:10
  tspan = (0.0, 10.0)
  n = 5
  bfun, b_u0, b_p, brusselator_jac, brusselator_comp = makebrusselator(n)
  @time bsol1 = auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); diffalg=(ForwardDiff.gradient), tols...);
  #@time bsol2 = auto_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
  #@test maximum(abs, bsol1 .- bsol2)/maximum(abs,  bsol1) < 1e-2

  @time bsol3 = map(ADJOINT_METHODS) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? bfun : ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver, reltol=1e-7; sensalg=alg, tols...)
  end
  @time bsol4 = numerical_sen_l2(bfun, b_u0, tspan, b_p, bt, (Rodas5()); tols...);
  # NOTE: backsolve gives unstable results!!!
  @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) < 4e-2, eachindex(ADJOINT_METHODS)[1:2end÷3])
  @test all(i->maximum(abs, bsol1 .- bsol3[i]')/maximum(abs, bsol1) >= 4e-2, eachindex(ADJOINT_METHODS)[2end÷3+1:end])
  @test maximum(abs, bsol1 .- bsol4)/maximum(abs, bsol1) < 2e-2
  t1 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()); diffalg=$(ForwardDiff.gradient), $tols...);
  #t2 = @belapsed auto_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
  t2 = NaN
  t3 = map(ADJOINT_METHODS[1:2end÷3]) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? bfun : ODEFunction{true, SciMLBase.FullSpecialize}(bfun, jac=brusselator_jac)
    solver = Rodas5(autodiff=false)
    @elapsed diffeq_sen_l2(f, b_u0, tspan, b_p, bt, solver; sensalg=alg, tols...);
  end
  t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
  t4 = @belapsed numerical_sen_l2($bfun, $b_u0, $tspan, $b_p, $bt, $(Rodas5()); $tols...);
  [t1; t2; t3; t4]
end


adjoint_pollution = let
  @info "Running the Pollution model:"
  pcomp, pu0, pp, pcompu0 = make_pollution();
  ptspan = (0.0, 60.0)
  pts = 0:0.5:60
  @time psol1 = auto_sen_l2((ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); diffalg=(ForwardDiff.gradient), tols...);
  #@time psol2 = auto_sen_l2((ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); diffalg=(ReverseDiff.gradient), tols...);
  #@test maximum(abs, psol1 .- psol2)/maximum(abs,  psol1) < 1e-2
  @time psol3 = map(ADJOINT_METHODS) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? pollution.f : ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac=pollution.jac)
    solver = Rodas5(autodiff=false)
    diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg=alg, tols...);
  end
  @time psol4 = numerical_sen_l2((ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), pu0, ptspan, pp, pts, (Rodas5(autodiff=false)); tols...);
  # NOTE: backsolve gives unstable results!!!
  @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) < 1e-2, eachindex(ADJOINT_METHODS)[1:2end÷3])
  @test all(i->maximum(abs, psol1 .- psol3[i]')/maximum(abs, psol1) >= 1e-2, eachindex(ADJOINT_METHODS)[2end÷3+1:end])
  @test maximum(abs, psol1 .- psol4)/maximum(abs, psol1) < 1e-2
  t1 = @belapsed auto_sen_l2($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); diffalg=$(ForwardDiff.gradient), $tols...);
  #t2 = @belapsed auto_sen_l2($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); diffalg=$(ReverseDiff.gradient), $tols...);
  t2 = NaN
  t3 = map(ADJOINT_METHODS[1:2end÷3]) do alg
    @info "Runing $alg"
    f = SciMLSensitivity.alg_autodiff(alg) ? pollution.f : ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f, jac=pollution.jac)
    solver = Rodas5(autodiff=false)
    @elapsed diffeq_sen_l2(f, pu0, ptspan, pp, pts, solver; sensalg=alg, tols...);
  end
  t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
  t4 = @belapsed numerical_sen_l2($(ODEFunction{true, SciMLBase.FullSpecialize}(pollution.f)), $pu0, $ptspan, $pp, $pts, $(Rodas5(autodiff=false)); $tols...);
  [t1; t2; t3; t4]
end


adjoint_pkpd = let
  @info "Running the PKPD model:"
  pts = 0:0.5:50
  # need to use lower tolerances to avoid running into the complex domain because of exponentiation
  pkpdsol1 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ForwardDiff.gradient), tols...);
  pkpdsol2 = @time auto_sen_l2((pkpdf.f), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=(ReverseDiff.gradient), tols...);
  pkpdsol3 = @time map(ADJOINT_METHODS[1:2end÷3]) do alg
    f = SciMLSensitivity.alg_autodiff(alg) ? pkpdf.f : ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac=pkpdf.jac)
    diffeq_sen_l2(f, pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5()); sensalg=alg,
                                  callback=pkpdcb, tstops=0:24:240, tols...);
  end
  pkpdsol4 = @time numerical_sen_l2((ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f)), pkpdu0, pkpdtspan, pkpdp, pts, (Tsit5());
                                     callback=pkpdcb, tstops=0:24:240, tols...);
  @test maximum(abs, pkpdsol1 .- pkpdsol2)/maximum(abs,  pkpdsol1) < 0.2
  @test all(i->maximum(abs, pkpdsol1 .- pkpdsol3[i]')/maximum(abs,  pkpdsol1) < 0.2, eachindex(ADJOINT_METHODS)[1:2end÷3])
  @test maximum(abs, pkpdsol1 .- pkpdsol4)/maximum(abs,  pkpdsol1) < 0.2
  t1 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ForwardDiff.gradient), $tols...);
  t2 = @belapsed auto_sen_l2($(pkpdf.f), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); callback=pkpdcb, tstops=0:24:240,
                                diffalg=$(ReverseDiff.gradient), $tols...);
  t3 = map(ADJOINT_METHODS[1:2end÷3]) do alg
    f = SciMLSensitivity.alg_autodiff(alg) ? pkpdf.f : ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f, jac=pkpdf.jac)
    @belapsed diffeq_sen_l2($f, $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops=0:24:240,
                                  callback=pkpdcb, sensalg=$alg, tols...);
  end
  t3 = [t3; fill(NaN, length(ADJOINT_METHODS)÷3)]
  t4 = @belapsed numerical_sen_l2($(ODEFunction{true, SciMLBase.FullSpecialize}(pkpdf.f)), $pkpdu0, $pkpdtspan, $pkpdp, $pts, $(Tsit5()); tstops=0:24:240,
                                     callback=$pkpdcb, $tols...);
  [t1; t2; t3; t4]
end


adjoint_methods = ["ForwardDiff", "ReverseDiff",
                   "InterpolatingAdjoint User Jac", "InterpolatingAdjoint AD Jac", "InterpolatingAdjoint v'J",
                   "QuadratureAdjoint User Jac", "QuadratureAdjoint AD Jac", "QuadratureAdjoint v'J",
                   "BacksolveAdjoint User Jac", "BacksolveAdjoint AD Jac", "BacksolveAdjoint v'J",
                   "Numerical Differentiation"]
adjoint_timings = DataFrame(methods=adjoint_methods, LV=adjoint_lv, Bruss=adjoint_bruss, Pollution=adjoint_pollution, PKPD=adjoint_pkpd)
Markdown.parse(PrettyTables.pretty_table(String, adjoint_timings; backend=Val(:markdown), header=names(adjoint_timings)))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

