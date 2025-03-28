
using OrdinaryDiffEq, DiffEqDevTools, Sundials, ParameterizedFunctions, Plots,
      ODEInterfaceDiffEq, LSODA, SparseArrays, LinearSolve,
      LinearAlgebra, IncompleteLU, AlgebraicMultigrid, Symbolics, ModelingToolkit,
      RecursiveFactorization
gr()

const N = 8

xyd_brusselator = range(0,stop=1,length=N)
brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a
function brusselator_2d_loop(du, u, p, t)
  A, B, alpha, dx = p
  alpha = alpha/dx^2
  @inbounds for I in CartesianIndices((N, N))
    i, j = Tuple(I)
    x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
    ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)
    du[i,j,1] = alpha*(u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4u[i,j,1]) +
                B + u[i,j,1]^2*u[i,j,2] - (A + 1)*u[i,j,1] + brusselator_f(x, y, t)
    du[i,j,2] = alpha*(u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4u[i,j,2]) +
                A*u[i,j,1] - u[i,j,1]^2*u[i,j,2]
    end
end
p = (3.4, 1., 10., step(xyd_brusselator))

input = rand(N,N,2)
output = similar(input)
sparsity_pattern = Symbolics.jacobian_sparsity(brusselator_2d_loop,output,input,p,0.0)
jac_sparsity = Float64.(sparse(sparsity_pattern))
f = ODEFunction{true, SciMLBase.FullSpecialize}(brusselator_2d_loop;jac_prototype=jac_sparsity)
function init_brusselator_2d(xyd)
  N = length(xyd)
  u = zeros(N, N, 2)
  for I in CartesianIndices((N, N))
    x = xyd[I[1]]
    y = xyd[I[2]]
    u[I,1] = 22*(y*(1-y))^(3/2)
    u[I,2] = 27*(x*(1-x))^(3/2)
  end
  u
end
u0 = init_brusselator_2d(xyd_brusselator)
prob = ODEProblem(f,u0,(0.,11.5),p);


prob_mtk = ODEProblem(complete(modelingtoolkitize(prob)),[],(0.0,11.5),jac=true,sparse=true);


using MethodOfLines, DomainSets
@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(u) = Dxx(u) + Dyy(u)

brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

α = 10.

u0_mol(x,y,t) = 22(y*(1-y))^(3/2)
v0_mol(x,y,t) = 27(x*(1-x))^(3/2)

eq = [Dt(u(x,y,t)) ~ 1. + v(x,y,t)*u(x,y,t)^2 - 4.4*u(x,y,t) + α*∇²(u(x,y,t)) + brusselator_f(x, y, t),
      Dt(v(x,y,t)) ~ 3.4*u(x,y,t) - v(x,y,t)*u(x,y,t)^2 + α*∇²(v(x,y,t))]

domains = [x ∈ Interval(x_min, x_max),
          y ∈ Interval(y_min, y_max),
          t ∈ Interval(t_min, t_max)]

bcs = [u(x,y,0) ~ u0_mol(x,y,0),
      u(0,y,t) ~ u(1,y,t),
      u(x,0,t) ~ u(x,1,t),

      v(x,y,0) ~ v0_mol(x,y,0),
      v(0,y,t) ~ v(1,y,t),
      v(x,0,t) ~ v(x,1,t)]

@named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[u(x,y,t),v(x,y,t)])

# Method of lines discretization

dx = 1/N
dy = 1/N

order = 2

discretization = MOLFiniteDifference([x=>dx, y=>dy], t; approx_order = order, jac = true, sparse = true, wrap = Val(false))

# Convert the PDE system into an ODE problem
prob_mol = discretize(pdesys,discretization)


using Base.Experimental: Const, @aliasscope
macro vp(expr)
  nodes = (Symbol("llvm.loop.vectorize.predicate.enable"), 1)
  if expr.head != :for
    error("Syntax error: loopinfo needs a for loop")
  end
  push!(expr.args[2].args, Expr(:loopinfo, nodes))
  return esc(expr)
end

struct Brusselator2DLoop <: Function
  N::Int
  s::Float64
end
function (b::Brusselator2DLoop)(du, unc, p, t)
  N = b.N
  s = b.s
  A, B, alpha, dx = p
  alpha = alpha/abs2(dx)
  u = Base.Experimental.Const(unc)
  Base.Experimental.@aliasscope begin
  @inbounds @fastmath begin
    b = ((abs2(-0.3) + abs2(-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
    du1 = alpha*(u[N,1,1] + u[2,1,1] + u[1,2,1] + u[1,N,1] - 4u[1,1,1]) +
      B + abs2(u[1,1,1])*u[1,1,2] - (A + 1)*u[1,1,1] + b
    du2 = alpha*(u[N,1,2] + u[2,1,2] + u[1,2,2] + u[1,N,2] - 4u[1,1,2]) +
      A*u[1,1,1] - abs2(u[1,1,1])*u[1,1,2]
    du[1,1,1] = du1
    du[1,1,2] = du2
    @vp for i = 2:N-1
      x = (i-1)*s
      ip1 = i+1
      im1 = i-1
      b = ((abs2(x-0.3) + abs2(-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
      du1 = alpha*(u[im1,1,1] + u[ip1,1,1] + u[i,2,1] + u[i,N,1] - 4u[i,1,1]) +
        B + abs2(u[i,1,1])*u[i,1,2] - (A + 1)*u[i,1,1] + b
      du2 = alpha*(u[im1,1,2] + u[ip1,1,2] + u[i,2,2] + u[i,N,2] - 4u[i,1,2]) +
        A*u[i,1,1] - abs2(u[i,1,1])*u[i,1,2]
      du[i,1,1] = du1
      du[i,1,2] = du2
    end
    b = ((abs2(0.7) + abs2(-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
    du1 = alpha*(u[N-1,1,1] + u[1,1,1] + u[N,2,1] + u[N,N,1] - 4u[N,1,1]) +
      B + abs2(u[N,1,1])*u[N,1,2] - (A + 1)*u[N,1,1] + b
    du2 = alpha*(u[N-1,1,2] + u[1,1,2] + u[N,2,2] + u[N,N,2] - 4u[N,1,2]) +
      A*u[N,1,1] - abs2(u[N,1,1])*u[N,1,2]
    du[N,1,1] = du1
    du[N,1,2] = du2
    for j = 2:N-1
      y = (j-1)*s
      jp1 = j+1
      jm1 = j-1
      b0 = ((abs2(-0.3) + abs2(y-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
      du[1,j,1] = alpha*(u[N,j,1] + u[2,j,1] + u[1,jp1,1] + u[1,jm1,1] - 4u[1,j,1]) +
        B + abs2(u[1,j,1])*u[1,j,2] - (A + 1)*u[1,j,1] + b0
      du[1,j,2] = alpha*(u[N,j,2] + u[2,j,2] + u[1,jp1,2] + u[1,jm1,2] - 4u[1,j,2]) +
        A*u[1,j,1] - abs2(u[1,j,1])*u[1,j,2]
      @vp for i = 2:N-1
        x = (i-1)*s
        b = ((abs2(x-0.3) + abs2(y-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
        du1 = alpha*(u[i-1,j,1] + u[i+1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4u[i,j,1]) +
          B + abs2(u[i,j,1])*u[i,j,2] - (A + 1)*u[i,j,1] + b
        du2 = alpha*(u[i-1,j,2] + u[i+1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4u[i,j,2]) +
          A*u[i,j,1] - abs2(u[i,j,1])*u[i,j,2]
        du[i,j,1] = du1
        du[i,j,2] = du2
      end
      bN = ((abs2(0.7) + abs2(y-0.6)) <= abs2(0.1)) * (t >= 1.1) * 5.0
      du[N,j,1] = alpha*(u[N-1,j,1] + u[1,j,1] + u[N,jp1,1] + u[N,jm1,1] - 4u[N,j,1]) +
        B + abs2(u[N,j,1])*u[N,j,2] - (A + 1)*u[N,j,1] + bN
      du[N,j,2] = alpha*(u[N-1,j,2] + u[1,j,2] + u[N,jp1,2] + u[N,jm1,2] - 4u[N,j,2]) +
        A*u[N,j,1] - abs2(u[N,j,1])*u[N,j,2]
    end
    b = ((abs2(-0.3) + abs2(0.4)) <= abs2(0.1)) * (t >= 1.1) * 5.0
    du1 = alpha*(u[N,N,1] + u[2,N,1] + u[1,1,1] + u[1,N-1,1] - 4u[1,N,1]) +
      B + abs2(u[1,N,1])*u[1,N,2] - (A + 1)*u[1,N,1] + b
    du2 = alpha*(u[N,N,2] + u[2,N,2] + u[1,1,2] + u[1,N-1,2] - 4u[1,N,2]) +
      A*u[1,N,1] - abs2(u[1,N,1])*u[1,N,2]
    du[1,N,1] = du1
    du[1,N,2] = du2
    @vp for i = 2:N-1
      x = (i-1)*s
      ip1 = i+1
      im1 = i-1
      b = ((abs2(x-0.3) + abs2(0.4)) <= abs2(0.1)) * (t >= 1.1) * 5.0
      du1 = alpha*(u[im1,N,1] + u[ip1,N,1] + u[i,1,1] + u[i,N-1,1] - 4u[i,N,1]) +
        B + abs2(u[i,N,1])*u[i,N,2] - (A + 1)*u[i,N,1] + b
      du2 = alpha*(u[im1,N,2] + u[ip1,N,2] + u[i,1,2] + u[i,N-1,2] - 4u[i,N,2]) +
        A*u[i,N,1] - abs2(u[i,N,1])*u[i,N,2]
      du[i,N,1] = du1
      du[i,N,2] = du2
    end
    b = ((abs2(0.7) + abs2(0.4)) <= abs2(0.1)) * (t >= 1.1) * 5.0
    du1 = alpha*(u[N-1,N,1] + u[1,N,1] + u[N,1,1] + u[N,N-1,1] - 4u[N,N,1]) +
      B + abs2(u[N,N,1])*u[N,N,2] - (A + 1)*u[N,N,1] + b
    du2 = alpha*(u[N-1,N,2] + u[1,N,2] + u[N,1,2] + u[N,N-1,2] - 4u[N,N,2]) +
      A*u[N,N,1] - abs2(u[N,N,1])*u[N,N,2]
    du[N,N,1] = du1
    du[N,N,2] = du2
  end
  end
end

function fast_bruss(N)
  xyd_brusselator = range(0,stop=1,length=N)
  brusselator_2d_loop = Brusselator2DLoop(N,Float64(step(xyd_brusselator)))
  p = (3.4, 1., 10., step(xyd_brusselator))

  input = rand(N,N,2)
  output = similar(input)
  sparsity_pattern = Symbolics.jacobian_sparsity(brusselator_2d_loop,output,input,p,0.0)
  jac_sparsity = Float64.(sparse(sparsity_pattern))
  f =  ODEFunction(brusselator_2d_loop;jac_prototype=jac_sparsity)
  u0 = zeros(N, N, 2)
  @inbounds for I in CartesianIndices((N, N))
    x = xyd_brusselator[I[1]]
    y = xyd_brusselator[I[2]]
    u0[I,1] = 22*(y*(1-y))^(3/2)
    u0[I,2] = 27*(x*(1-x))^(3/2)
  end
  return ODEProblem(f,u0,(0.,11.5),p)
end

fastprob = fast_bruss(N)


sol = solve(prob,CVODE_BDF(),abstol=1/10^14,reltol=1/10^14)
sol2 = solve(prob_mtk,CVODE_BDF(linear_solver = :KLU),abstol=1/10^14,reltol=1/10^14)
sol3 = solve(prob_mol,CVODE_BDF(linear_solver = :KLU),abstol=1/10^14,reltol=1/10^14,wrap=Val(false))


test_sol = [sol,sol2,sol,sol3]
probs = [prob,prob_mtk,fastprob,prob_mol];


plot(sol, idxs = 1)


plot(sol, idxs = 10)


function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
  if newW === nothing || newW
    Pl = ilu(convert(AbstractMatrix,W), τ = 50.0)
  else
    Pl = Plprev
  end
  Pl,nothing
end

function algebraicmultigrid(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
  if newW === nothing || newW
    Pl = aspreconditioner(ruge_stuben(convert(AbstractMatrix,W)))
  else
    Pl = Plprev
  end
  Pl,nothing
end


const jaccache = prob_mtk.f.jac(prob.u0,prob.p,0.0)
const W = I - 1.0*jaccache

prectmp = ilu(W, τ = 50.0)
const preccache = Ref(prectmp)

function psetupilu(p, t, u, du, jok, jcurPtr, gamma)
  if !jok
    prob_mtk.f.jac(jaccache,u,p,t)
    jcurPtr[] = true

    # W = I - gamma*J
    @. W = -gamma*jaccache
    idxs = diagind(W)
    @. @view(W[idxs]) = @view(W[idxs]) + 1

    # Build preconditioner on W
    preccache[] = ilu(W, τ = 5.0)
  end
end

function precilu(z,r,p,t,y,fy,gamma,delta,lr)
  ldiv!(z,preccache[],r)
end

prectmp2 = aspreconditioner(ruge_stuben(W, presmoother = AlgebraicMultigrid.Jacobi(rand(size(W,1))), postsmoother = AlgebraicMultigrid.Jacobi(rand(size(W,1)))))
const preccache2 = Ref(prectmp2)
function psetupamg(p, t, u, du, jok, jcurPtr, gamma)
  if !jok
    prob_mtk.f.jac(jaccache,u,p,t)
    jcurPtr[] = true

    # W = I - gamma*J
    @. W = -gamma*jaccache
    idxs = diagind(W)
    @. @view(W[idxs]) = @view(W[idxs]) + 1

    # Build preconditioner on W
    preccache2[] = aspreconditioner(ruge_stuben(W, presmoother = AlgebraicMultigrid.Jacobi(rand(size(W,1))), postsmoother = AlgebraicMultigrid.Jacobi(rand(size(W,1)))))
  end
end

function precamg(z,r,p,t,y,fy,gamma,delta,lr)
  ldiv!(z,preccache2[],r)
end


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
setups = [

          Dict(:alg => KenCarp47(linsolve=KLUFactorization())),
          Dict(:alg => KenCarp47(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg => KenCarp47(linsolve=KLUFactorization()), :prob_choice => 3),
          Dict(:alg => KenCarp47(linsolve=KLUFactorization()), :prob_choice => 4),
          Dict(:alg => KenCarp47(linsolve=KrylovJL_GMRES())),
          Dict(:alg => KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg => KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 3),
          Dict(:alg => KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 4),]
names = ["KenCarp47 KLU","KenCarp47 KLU MTK","KenCarp47 KLU FastBruss", "KenCarp47 KLU MOL",
         "KenCarp47 GMRES", "KenCarp47 GMRES MTK", "KenCarp47 GMRES FastBruss", "KenCarp47 GMRES MOL"];

wp = WorkPrecisionSet(probs,abstols,reltols,setups;names = names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10,wrap=Val(false))
plot(wp)


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
setups = [
    		  Dict(:alg=>CVODE_BDF(linear_solver = :KLU), :prob_choice => 2),
		      Dict(:alg=>CVODE_BDF(linear_solver = :GMRES)),
          Dict(:alg=>CVODE_BDF(linear_solver = :GMRES), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1)),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precamg,psetup=psetupamg,prec_side=1)),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precamg,psetup=psetupamg,prec_side=1), :prob_choice => 2),
          ]
names = ["CVODE MTK KLU","CVODE GMRES","CVODE MTK GMRES", "CVODE iLU GMRES", "CVODE AMG GMRES", "CVODE iLU MTK GMRES", "CVODE AMG MTK GMRES"];
wp = WorkPrecisionSet(probs,abstols,reltols,setups;names=names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


setups = [
          Dict(:alg=>KenCarp47(linsolve=KLUFactorization())),
          Dict(:alg=>KenCarp47(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=UMFPACKFactorization())),
          Dict(:alg=>KenCarp47(linsolve=UMFPACKFactorization()), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES())),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES(),precs=incompletelu,concrete_jac=true)),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES(),precs=incompletelu,concrete_jac=true), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES(),precs=algebraicmultigrid,concrete_jac=true)),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES(),precs=algebraicmultigrid,concrete_jac=true), :prob_choice => 2),

          ]
names = ["KenCarp47 KLU","KenCarp47 KLU MTK","KenCarp47 UMFPACK", "KenCarp47 UMFPACK MTK", "KenCarp47 GMRES",
        "KenCarp47 GMRES MTK", "KenCarp47 iLU GMRES", "KenCarp47 iLU GMRES MTK", "KenCarp47 AMG GMRES",
        "KenCarp47 AMG GMRES MTK"];
wp = WorkPrecisionSet(probs,abstols,reltols,setups;names = names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


setups = [
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>KenCarp47()),
    		  # Dict(:alg=>QNDF()), # bad
          Dict(:alg=>FBDF()),
          ]
wp = WorkPrecisionSet(probs,abstols,reltols,setups;
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


setups = [
          Dict(:alg=>KenCarp47(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>FBDF(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>FBDF(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver = :KLU), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1), :prob_choice => 2),
          ]
names = ["KenCarp47 KLU MTK", "KenCarp47 GMRES MTK",
         "FBDF KLU MTK", "FBDF GMRES MTK",
         "CVODE MTK KLU", "CVODE iLU MTK GMRES"
];
wp = WorkPrecisionSet(probs,abstols,reltols,setups;names = names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


abstols = 1.0 ./ 10.0 .^ (7:12)
reltols = 1.0 ./ 10.0 .^ (4:9)

setups = [
    		  Dict(:alg=>CVODE_BDF(linear_solver = :KLU), :prob_choice => 2),
		      Dict(:alg=>CVODE_BDF(linear_solver = :GMRES)),
          Dict(:alg=>CVODE_BDF(linear_solver = :GMRES), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1)),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precamg,psetup=psetupamg,prec_side=1)),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precamg,psetup=psetupamg,prec_side=1), :prob_choice => 2),
          ]
names = ["CVODE MTK KLU","CVODE GMRES","CVODE MTK GMRES", "CVODE iLU GMRES", "CVODE AMG GMRES", "CVODE iLU MTK GMRES", "CVODE AMG MTK GMRES"];
wp = WorkPrecisionSet(probs,abstols,reltols,setups;names = names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


setups = [
          Dict(:alg=>KenCarp47(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>KenCarp47(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>FBDF(linsolve=KLUFactorization()), :prob_choice => 2),
          Dict(:alg=>FBDF(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>Rodas5P(linsolve=KrylovJL_GMRES()), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver = :KLU), :prob_choice => 2),
          Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1), :prob_choice => 2),
          ]
names = ["KenCarp47 KLU MTK", "KenCarp47 GMRES MTK",
         "FBDF KLU MTK", "FBDF GMRES MTK",
         "Rodas5P GMRES MTK",
         "CVODE MTK KLU", "CVODE iLU MTK GMRES"
];
wp = WorkPrecisionSet(probs,abstols,reltols,setups;names = names,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

