
using DiffEqBase, OrdinaryDiffEq, Catalyst, ReactionNetworkImporters,
    Sundials, Plots, DiffEqDevTools, ODEInterface, ODEInterfaceDiffEq,
    LSODA, TimerOutputs, LinearAlgebra, ModelingToolkit, BenchmarkTools,
    LinearSolve

gr()
datadir  = joinpath(dirname(pathof(ReactionNetworkImporters)),"../data/bcr")
const to = TimerOutput()
tf       = 100000.0

# generate ModelingToolkit ODEs
@timeit to "Parse Network" prnbng = loadrxnetwork(BNGNetwork(), joinpath(datadir, "bcr.net"))
show(to)
rn    = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

@timeit to "Create ODESys" osys = convert(ODESystem, rn)
show(to)

tspan = (0.,tf)
@timeit to "ODEProb No Jac" oprob = ODEProblem(osys, Float64[], tspan, Float64[])
show(to)
oprob_sparse = ODEProblem(osys, Float64[], tspan, Float64[]; sparse=true);


@timeit to "ODEProb SparseJac" sparsejacprob = ODEProblem(osys, Float64[], tspan, Float64[], jac=true, sparse=true)
show(to)


@show numspecies(rn) # Number of ODEs
@show numreactions(rn) # Apprx. number of terms in the ODE
@show length(parameters(rn)); # Number of Parameters


u  = ModelingToolkit.varmap_to_vars(nothing, species(rn); defaults=ModelingToolkit.defaults(rn))
du = copy(u)
p  = ModelingToolkit.varmap_to_vars(nothing, parameters(rn); defaults=ModelingToolkit.defaults(rn))
@timeit to "ODE rhs Eval1" oprob.f(du,u,p,0.)
@timeit to "ODE rhs Eval2" oprob.f(du,u,p,0.)
sparsejacprob.f(du,u,p,0.)


@btime oprob.f($du,$u,$p,0.)


Js = similar(sparsejacprob.f.jac_prototype)
@timeit to "SparseJac Eval1" sparsejacprob.f.jac(Js,u,p,0.)
@timeit to "SparseJac Eval2" sparsejacprob.f.jac(Js,u,p,0.)
show(to)


sol = solve(oprob, CVODE_BDF(), saveat=tf/1000., reltol=1e-5, abstol=1e-5)
plot(sol; idxs=obs, legend=false, fmt=:png)


@time sol = solve(oprob, CVODE_BDF(), abstol=1/10^12, reltol=1/10^12)
test_sol  = TestSolution(sol);


default(legendfontsize=7,framestyle=:box,gridalpha=0.3,gridlinewidth=2.5)


function plot_settings(wp)
    times = vcat(map(wp -> wp.times, wp.wps)...)
    errors = vcat(map(wp -> wp.errors, wp.wps)...)
    xlimit = 10 .^ (floor(log10(minimum(errors))), ceil(log10(maximum(errors))))
    ylimit = 10 .^ (floor(log10(minimum(times))), ceil(log10(maximum(times))))
    return xlimit,ylimit
end


using IncompleteLU, LinearAlgebra

jaccache = sparsejacprob.f.jac(oprob.u0,oprob.p,0.0)
W = I - 1.0*jaccache
prectmp = ilu(W, τ = 50.0)
preccache = Ref(prectmp)

const τ1 = 1e2
function psetupilu(p, t, u, du, jok, jcurPtr, gamma)
    if !jok
        sparsejacprob.f.jac(jaccache,u,p,t)
        jcurPtr[] = true

        # W = I - gamma*J
        @. W = -gamma*jaccache
        idxs = diagind(W)
        @. @view(W[idxs]) = @view(W[idxs]) + 1

        # Build preconditioner on W
        preccache[] = ilu(W, τ = τ1)
    end
end
function precilu(z,r,p,t,y,fy,gamma,delta,lr)
    ldiv!(z,preccache[],r)
end

const τ2 = 1e2
function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
    if newW === nothing || newW
        Pl = ilu(convert(AbstractMatrix,W), τ = τ2)
    else
        Pl = Plprev
    end
    Pl,nothing
end;


abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (5:8);


solve(sparsejacprob,CVODE_BDF(linear_solver=:KLU), abstol=1e-8, reltol=1e-8);


setups = [
        Dict(:alg=>lsoda(), :prob_choice => 1),
        Dict(:alg=>CVODE_BDF(), :prob_choice => 1),
        Dict(:alg=>CVODE_BDF(linear_solver=:LapackDense), :prob_choice => 1),
        Dict(:alg=>CVODE_BDF(linear_solver=:GMRES), :prob_choice => 1),
        Dict(:alg=>CVODE_BDF(linear_solver=:GMRES,prec=precilu,psetup=psetupilu,prec_side=1), :prob_choice => 2),
        ];


wp = WorkPrecisionSet([oprob,oprob_sparse,sparsejacprob],abstols,reltols,setups;error_estimate=:l2,
                    saveat=tf/10000.,appxsol=[test_sol,test_sol,test_sol],maxiters=Int(1e6),numruns=1)

names = ["lsoda" "CVODE_BDF" "CVODE_BDF (LapackDense)" "CVODE_BDF (GMRES)" "CVODE_BDF (GMRES, iLU)" "CVODE_BDF (KLU, sparse jac)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
        Dict(:alg=>TRBDF2(autodiff=false)),
        Dict(:alg=>QNDF(autodiff=false)),
        Dict(:alg=>FBDF(autodiff=false)),
        Dict(:alg=>KenCarp4(autodiff=false))
        ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                    saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e6),numruns=1)

names = ["TRBDF2" "QNDF" "FBDF" "KenCarp4"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
        Dict(:alg=>TRBDF2(linsolve=KrylovJL_GMRES(),autodiff=false)),
        Dict(:alg=>QNDF(linsolve=KrylovJL_GMRES(),autodiff=false)),
        Dict(:alg=>FBDF(linsolve=KrylovJL_GMRES(),autodiff=false)),
        Dict(:alg=>KenCarp4(linsolve=KrylovJL_GMRES(),autodiff=false))
        ];


wp = WorkPrecisionSet(oprob,abstols,reltols,setups;error_estimate=:l2,
                    saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e6),numruns=1)

names = ["TRBDF2 (GMRES)" "QNDF (GMRES)" "FBDF (GMRES)" "KenCarp4 (GMRES)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
        Dict(:alg=>TRBDF2(linsolve=KrylovJL_GMRES(),autodiff=false,precs=incompletelu,concrete_jac=true)),
        Dict(:alg=>QNDF(linsolve=KrylovJL_GMRES(),autodiff=false,precs=incompletelu,concrete_jac=true)),
        Dict(:alg=>FBDF(linsolve=KrylovJL_GMRES(),autodiff=false,precs=incompletelu,concrete_jac=true)),
        Dict(:alg=>KenCarp4(linsolve=KrylovJL_GMRES(),autodiff=false,precs=incompletelu,concrete_jac=true))
        ];


wp = WorkPrecisionSet(oprob_sparse,abstols,reltols,setups;error_estimate=:l2,
                    saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e6),numruns=1)

names = ["TRBDF2 (GMRES, iLU)" "QNDF (GMRES, iLU)" "FBDF (GMRES, iLU)" "KenCarp4 (GMRES, iLU)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


setups = [
        Dict(:alg=>TRBDF2(linsolve=KLUFactorization(),autodiff=false)),
        Dict(:alg=>QNDF(linsolve=KLUFactorization(),autodiff=false)),
        Dict(:alg=>FBDF(linsolve=KLUFactorization(),autodiff=false)),
        Dict(:alg=>KenCarp4(linsolve=KLUFactorization(),autodiff=false))
        ];


wp = WorkPrecisionSet(sparsejacprob,abstols,reltols,setups;error_estimate=:l2,
                    saveat=tf/10000.,appxsol=test_sol,maxiters=Int(1e6),numruns=1)

names = ["TRBDF2 (KLU, sparse jac)" "QNDF (KLU, sparse jac)" "FBDF (KLU, sparse jac)" "KenCarp4 (KLU, sparse jac)"]
xlimit,ylimit = plot_settings(wp)
plot(wp;label=names,xlimit=xlimit,ylimit=ylimit)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

