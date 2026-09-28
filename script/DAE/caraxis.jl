
using OrdinaryDiffEq, DiffEqDevTools, Sundials, ModelingToolkit, ODEInterfaceDiffEq,
      Plots, DASSL, DASKR
using OrdinaryDiffEqBDF, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock
using LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D

# Constants
const M_ca    = 10.0
const eps_ca  = 1e-2
const L_ca    = 1.0
const L0_ca   = 0.5
const r_ca    = 0.1
const omega_ca = 10.0
const g_ca    = 1.0
const k_ca    = M_ca * eps_ca^2 / 2.0

# Shared initial conditions for all 10-variable formulations
u0_mm  = [0.0, 0.5, 1.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.0]


@variables xl(t) [guess = 0.0]   yl(t)    xr(t) [guess = 1.0]    yr(t)
@variables dxl(t) [guess = -0.5] dyl(t)   dxr(t) [guess = -0.5]  dyr(t)
@variables lam1(t) [guess = 0.0] lam2(t) [guess = 0.0]

yb_s = r_ca * sin(omega_ca * t)
xb_s = sqrt(L_ca^2 - yb_s^2)
Ll_s = sqrt(xl^2 + yl^2)
Lr_s = sqrt((xr - xb_s)^2 + (yr - yb_s)^2)

eqs = [
    D(xl)  ~ dxl,
    D(yl)  ~ dyl,
    D(xr)  ~ dxr,
    D(yr)  ~ dyr,
    k_ca * D(dxl) ~ (L0_ca - Ll_s)*xl/Ll_s + lam1*xb_s + 2.0*lam2*(xl - xr),
    k_ca * D(dyl) ~ (L0_ca - Ll_s)*yl/Ll_s + lam1*yb_s + 2.0*lam2*(yl - yr) - k_ca*g_ca,
    k_ca * D(dxr) ~ (L0_ca - Lr_s)*(xr - xb_s)/Lr_s    - 2.0*lam2*(xl - xr),
    k_ca * D(dyr) ~ (L0_ca - Lr_s)*(yr - yb_s)/Lr_s    - 2.0*lam2*(yl - yr) - k_ca*g_ca,
    0 ~ xb_s*xl + yb_s*yl,
    0 ~ (xl - xr)^2 + (yl - yr)^2 - L_ca^2,
]

@mtkbuild sys = ODESystem(eqs, t; maybe_zeros = [yb_s])
tspan = (0.0, 3.0)

mtkprob  = ODEProblem(sys, [yl => 0.5, yr => 0.5, dyl => 0.0, dyr => 0.0], tspan) # prob_choice = 1
println("MTK u0 all finite: ", all(isfinite, mtkprob.u0))
println("MTK RHS at t = 0 all finite: ", all(isfinite, mtkprob.f(mtkprob.u0, mtkprob.p, tspan[1])))


function caraxis_residual!(res, du, u, p, t)
    xl_,yl_,xr_,yr_     = u[1],u[2],u[3],u[4]
    dxl_,dyl_,dxr_,dyr_ = u[5],u[6],u[7],u[8]
    lam1_,lam2_          = u[9],u[10]
    yb_ = r_ca*sin(omega_ca*t);  xb_ = sqrt(L_ca^2 - yb_^2)
    Ll_ = sqrt(xl_^2 + yl_^2)
    Lr_ = sqrt((xr_-xb_)^2 + (yr_-yb_)^2)
    res[1] = du[1] - dxl_
    res[2] = du[2] - dyl_
    res[3] = du[3] - dxr_
    res[4] = du[4] - dyr_
    res[5] = k_ca*du[5] - ((L0_ca-Ll_)*xl_/Ll_ + lam1_*xb_ + 2.0*lam2_*(xl_-xr_))
    res[6] = k_ca*du[6] - ((L0_ca-Ll_)*yl_/Ll_ + lam1_*yb_ + 2.0*lam2_*(yl_-yr_) - k_ca*g_ca)
    res[7] = k_ca*du[7] - ((L0_ca-Lr_)*(xr_-xb_)/Lr_ - 2.0*lam2_*(xl_-xr_))
    res[8] = k_ca*du[8] - ((L0_ca-Lr_)*(yr_-yb_)/Lr_ - 2.0*lam2_*(yl_-yr_) - k_ca*g_ca)
    res[9]  = xb_*xl_ + yb_*yl_
    res[10] = (xl_-xr_)^2 + (yl_-yr_)^2 - L_ca^2
    nothing
end

du0_dae   = [-0.5, 0.0, -0.5, 0.0, 0.0, -g_ca, 0.0, -g_ca, 0.0, 0.0]
diff_vars = [true,true,true,true,true,true,true,true,false,false]
daeprob   = DAEProblem(caraxis_residual!, du0_dae, u0_mm, tspan;
                       differential_vars = diff_vars)              # prob_choice = 2


function caraxis_mm!(du, u, p, t)
    xl_,yl_,xr_,yr_     = u[1],u[2],u[3],u[4]
    dxl_,dyl_,dxr_,dyr_ = u[5],u[6],u[7],u[8]
    lam1_,lam2_          = u[9],u[10]
    yb_ = r_ca*sin(omega_ca*t);  xb_ = sqrt(L_ca^2 - yb_^2)
    Ll_ = sqrt(xl_^2 + yl_^2)
    Lr_ = sqrt((xr_-xb_)^2 + (yr_-yb_)^2)
    du[1]=dxl_; du[2]=dyl_; du[3]=dxr_; du[4]=dyr_
    du[5] = (L0_ca-Ll_)*xl_/Ll_ + lam1_*xb_ + 2.0*lam2_*(xl_-xr_)
    du[6] = (L0_ca-Ll_)*yl_/Ll_ + lam1_*yb_ + 2.0*lam2_*(yl_-yr_) - k_ca*g_ca
    du[7] = (L0_ca-Lr_)*(xr_-xb_)/Lr_         - 2.0*lam2_*(xl_-xr_)
    du[8] = (L0_ca-Lr_)*(yr_-yb_)/Lr_         - 2.0*lam2_*(yl_-yr_) - k_ca*g_ca
    du[9]  = xb_*xl_ + yb_*yl_
    du[10] = (xl_-xr_)^2 + (yl_-yr_)^2 - L_ca^2
    nothing
end

M_mat  = Matrix(Diagonal([1.0,1.0,1.0,1.0, k_ca,k_ca,k_ca,k_ca, 0.0,0.0]))
mmf    = ODEFunction(caraxis_mm!, mass_matrix=M_mat)
mmprob = ODEProblem(mmf, u0_mm, tspan)                            # prob_choice = 3


function caraxis_rescaled!(du, u, p, t)
    xl_,yl_,xr_,yr_     = u[1],u[2],u[3],u[4]
    dxl_,dyl_,dxr_,dyr_ = u[5],u[6],u[7],u[8]
    lam1_,lam2_          = u[9],u[10]
    yb_ = r_ca*sin(omega_ca*t);  xb_ = sqrt(L_ca^2 - yb_^2)
    Ll_ = sqrt(xl_^2 + yl_^2)
    Lr_ = sqrt((xr_-xb_)^2 + (yr_-yb_)^2)
    du[1]=dxl_; du[2]=dyl_; du[3]=dxr_; du[4]=dyr_
    du[5] = ((L0_ca-Ll_)*xl_/Ll_ + lam1_*xb_ + 2.0*lam2_*(xl_-xr_)) / k_ca
    du[6] = ((L0_ca-Ll_)*yl_/Ll_ + lam1_*yb_ + 2.0*lam2_*(yl_-yr_) - k_ca*g_ca) / k_ca
    du[7] = ((L0_ca-Lr_)*(xr_-xb_)/Lr_ - 2.0*lam2_*(xl_-xr_)) / k_ca
    du[8] = ((L0_ca-Lr_)*(yr_-yb_)/Lr_ - 2.0*lam2_*(yl_-yr_) - k_ca*g_ca) / k_ca
    du[9]  = xb_*xl_ + yb_*yl_
    du[10] = (xl_-xr_)^2 + (yl_-yr_)^2 - L_ca^2
    nothing
end

M_rsc    = Matrix(Diagonal([1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.0,0.0]))
f_rsc    = ODEFunction(caraxis_rescaled!, mass_matrix = M_rsc)
rscprob  = ODEProblem(f_rsc, u0_mm, tspan)                        # prob_choice = 4


const radau5_alg = radau5(DIMOFIND1VAR=4, DIMOFIND2VAR=4, DIMOFIND3VAR=2)

ref_sol = solve(rscprob, radau5_alg; abstol=1e-12, reltol=1e-12)
println("Reference retcode: ", ref_sol.retcode)
println("NaN in reference? ", any(isnan, ref_sol.u[end]))


mtk_ref = solve(mtkprob, Rodas5P(); abstol=1e-12, reltol=1e-12)
println("MTK reference retcode: ", mtk_ref.retcode)

testset_pos = [0.493455784275402809e-1, 0.496989460230171153,
               0.104174252488542151e1, 0.373911027265361256]
relerr(u) = maximum(abs.(u .- testset_pos) ./ abs.(testset_pos))
println("RADAU5 reference, max rel. error in positions at t = 3: ", relerr(ref_sol.u[end][1:4]))
println("MTK reference,    max rel. error in positions at t = 3: ",
        relerr(mtk_ref[[xl, yl, xr, yr]][end]))


probs = [mtkprob, daeprob, mmprob, rscprob]
refs  = [mtk_ref, ref_sol, ref_sol, ref_sol];


plot(ref_sol; idxs=[1,2,3,4],
     label=["xₗ" "yₗ" "xᵣ" "yᵣ"], title="Car Axis — positions",
     xlabel="t", ylabel="position", layout=(2,2), size=(900,600))


plot(ref_sol; idxs=[9,10],
     label=["λ₁" "λ₂"], title="Lagrange multipliers", xlabel="t")


abstols = 1.0 ./ 10.0 .^ (4:8)
reltols = 1.0 ./ 10.0 .^ (4:8)

setups = [Dict(:prob_choice => 4, :alg => radau5_alg),
          Dict(:prob_choice => 1, :alg => Rodas5P()),
          Dict(:prob_choice => 1, :alg => RadauIIA5()),
          Dict(:prob_choice => 1, :alg => FBDF()),
          Dict(:prob_choice => 1, :alg => QNDF())]
labels = ["RADAU5 (index-3 form)" "MTK + Rodas5P" "MTK + RadauIIA5" "MTK + FBDF" "MTK + QNDF"]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
    names = vec(labels), save_everystep = false, appxsol = refs, maxiters = Int(1e5), numruns = 3)
plot(wp; title = "Car Axis WPD — High Tolerances")


abstols = 1.0 ./ 10.0 .^ (7:11)
reltols = 1.0 ./ 10.0 .^ (7:11)

setups = [Dict(:prob_choice => 4, :alg => radau5_alg),
          Dict(:prob_choice => 1, :alg => Rodas5P()),
          Dict(:prob_choice => 1, :alg => RadauIIA5())]
labels = ["RADAU5 (index-3 form)" "MTK + Rodas5P" "MTK + RadauIIA5"]

wp = WorkPrecisionSet(probs, abstols, reltols, setups;
    names = vec(labels), save_everystep = false, appxsol = refs, maxiters = Int(1e5), numruns = 3)
plot(wp; title = "Car Axis WPD — Low Tolerances")


println("Standard Julia solvers on the raw mass-matrix form:")
for (name, alg) in [("Rodas4", Rodas4()), ("Rodas5P", Rodas5P()),
                     ("RadauIIA5", RadauIIA5()), ("FBDF", FBDF()), ("QNDF", QNDF()), ("NordsieckBDF", NordsieckBDF())]
    sol = solve(mmprob, alg; reltol=1e-5, abstol=1e-5, maxiters=Int(1e3))
    println("  ", rpad(name, 12), " → ", sol.retcode)
end


println("DAE solvers on the residual form:")
for (name, alg) in [("IDA", IDA()), ("DASSL", DASSL.dassl()), ("DASKR", DASKR.daskr())]
    try
        sol = solve(daeprob, alg; reltol=1e-5, abstol=1e-5, maxiters=Int(1e3))
        println("  ", rpad(name, 12), " → ", sol.retcode)
    catch e
        println("  ", rpad(name, 12), " → threw ", nameof(typeof(e)))
    end
end


g1_err = Float64[]
g2_err = Float64[]
for i in eachindex(ref_sol.t)
    u  = ref_sol.u[i]
    tc = ref_sol.t[i]
    xb = sqrt(L_ca^2 - (r_ca*sin(omega_ca*tc))^2)
    yb = r_ca*sin(omega_ca*tc)
    push!(g1_err, abs(xb*u[1] + yb*u[2]))
    push!(g2_err, abs((u[1]-u[3])^2 + (u[2]-u[4])^2 - L_ca^2))
end

g1_plot = max.(g1_err, eps())
g2_plot = max.(g2_err, eps())

plot(ref_sol.t, [g1_plot g2_plot]; yscale=:log10,
     label=["|g₁| orthogonality" "|g₂| rigid axis"],
     xlabel="t", ylabel="residual",
     title="Algebraic Constraint Satisfaction (RADAU5, rtol=1e-12)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

