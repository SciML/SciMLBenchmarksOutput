---
author: "dextorious, Chris Rackauckas"
title: "Filament Work-Precision Diagrams"
---


# Filament Benchmark

In this notebook we will benchmark a real-world biological model from a paper entitled [Magnetic dipole with a flexible tail as a self-propelling microdevice](https://doi.org/10.1103/PhysRevE.85.041502). This is a system of PDEs representing a Kirchhoff model of an elastic rod, where the equations of motion are given by the Rouse approximation with free boundary conditions.

## Model Implementation

First we will show the full model implementation. It is not necessary to understand the full model specification in order to understand the benchmark results, but it's all contained here for completeness. The model is highly optimized, with all internal vectors pre-cached, loops unrolled for efficiency (along with `@simd` annotations), a pre-defined Jacobian, matrix multiplications are all in-place, etc. Thus this model is a good stand-in for other optimized PDE solving cases.

The model is thus defined as follows:

```julia
using OrdinaryDiffEq, ODEInterfaceDiffEq, Sundials, DiffEqDevTools, LSODA
using LinearAlgebra
using Plots
gr()
```

```
Plots.GRBackend()
```



```julia
const T = Float64
abstract type AbstractFilamentCache end
abstract type AbstractMagneticForce end
abstract type AbstractInextensibilityCache end
abstract type AbstractSolver end
abstract type AbstractSolverCache end
```


```julia
struct FerromagneticContinuous <: AbstractMagneticForce
    ω :: T
    F :: Vector{T}
end

mutable struct FilamentCache{
        MagneticForce        <: AbstractMagneticForce,
        InextensibilityCache <: AbstractInextensibilityCache,
        SolverCache          <: AbstractSolverCache
            } <: AbstractFilamentCache
    N  :: Int
    μ  :: T
    Cm :: T
    x  :: SubArray{T,1,Vector{T},Tuple{StepRange{Int,Int}},true}
    y  :: SubArray{T,1,Vector{T},Tuple{StepRange{Int,Int}},true}
    z  :: SubArray{T,1,Vector{T},Tuple{StepRange{Int,Int}},true}
    A  :: Matrix{T}
    P  :: InextensibilityCache
    F  :: MagneticForce
    Sc :: SolverCache
end
```


```julia
struct NoHydroProjectionCache <: AbstractInextensibilityCache
    J         :: Matrix{T}
    P         :: Matrix{T}
    J_JT      :: Matrix{T}
    J_JT_LDLT :: LinearAlgebra.LDLt{T, SymTridiagonal{T}}
    P0        :: Matrix{T}

    NoHydroProjectionCache(N::Int) = new(
        zeros(N, 3*(N+1)),          # J
        zeros(3*(N+1), 3*(N+1)),    # P
        zeros(N,N),                 # J_JT
        LinearAlgebra.LDLt{T,SymTridiagonal{T}}(SymTridiagonal(zeros(N), zeros(N-1))),
        zeros(N, 3*(N+1))
    )
end
```


```julia
struct DiffEqSolverCache <: AbstractSolverCache
    S1 :: Vector{T}
    S2 :: Vector{T}

    DiffEqSolverCache(N::Integer) = new(zeros(T,3*(N+1)), zeros(T,3*(N+1)))
end
```


```julia
function FilamentCache(N=20; Cm=32, ω=200, Solver=SolverDiffEq)
    InextensibilityCache = NoHydroProjectionCache
    SolverCache = DiffEqSolverCache
    tmp = zeros(3*(N+1))
    FilamentCache{FerromagneticContinuous, InextensibilityCache, SolverCache}(
        N, N+1, Cm, view(tmp,1:3:3*(N+1)), view(tmp,2:3:3*(N+1)), view(tmp,3:3:3*(N+1)),
        zeros(3*(N+1), 3*(N+1)), # A
        InextensibilityCache(N), # P
        FerromagneticContinuous(ω, zeros(3*(N+1))),
        SolverCache(N)
    )
end
```

```
Main.var"##WeaveSandBox#312".FilamentCache
```



```julia
function stiffness_matrix!(f::AbstractFilamentCache)
    N, μ, A = f.N, f.μ, f.A
    @inbounds for j in axes(A, 2), i in axes(A, 1)
      A[i, j] = j == i ? 1 : 0
    end
    @inbounds for i in 1 : 3
        A[i,i] =    1
        A[i,3+i] = -2
        A[i,6+i] =  1

        A[3+i,i]   = -2
        A[3+i,3+i] =  5
        A[3+i,6+i] = -4
        A[3+i,9+i] =  1

        A[3*(N-1)+i,3*(N-3)+i] =  1
        A[3*(N-1)+i,3*(N-2)+i] = -4
        A[3*(N-1)+i,3*(N-1)+i] =  5
        A[3*(N-1)+i,3*N+i]     = -2

        A[3*N+i,3*(N-2)+i]     =  1
        A[3*N+i,3*(N-1)+i]     = -2
        A[3*N+i,3*N+i]         =  1

        for j in 2 : N-2
            A[3*j+i,3*j+i]     =  6
            A[3*j+i,3*(j-1)+i] = -4
            A[3*j+i,3*(j+1)+i] = -4
            A[3*j+i,3*(j-2)+i] =  1
            A[3*j+i,3*(j+2)+i] =  1
        end
    end
    rmul!(A, -μ^4)
    nothing
end
```

```
stiffness_matrix! (generic function with 1 method)
```



```julia
function update_separate_coordinates!(f::AbstractFilamentCache, r)
    N, x, y, z = f.N, f.x, f.y, f.z
    @inbounds for i in 1 : length(x)
        x[i] = r[3*i-2]
        y[i] = r[3*i-1]
        z[i] = r[3*i]
    end
    nothing
end

function update_united_coordinates!(f::AbstractFilamentCache, r)
    N, x, y, z = f.N, f.x, f.y, f.z
    @inbounds for i in 1 : length(x)
        r[3*i-2] = x[i]
        r[3*i-1] = y[i]
        r[3*i]   = z[i]
    end
    nothing
end

function update_united_coordinates(f::AbstractFilamentCache)
    r = zeros(T, 3*length(f.x))
    update_united_coordinates!(f, r)
    r
end
```

```
update_united_coordinates (generic function with 1 method)
```



```julia
function initialize!(initial_conf_type::Symbol, f::AbstractFilamentCache)
    N, x, y, z = f.N, f.x, f.y, f.z
    if initial_conf_type == :StraightX
        x .= range(0, stop=1, length=N+1)
        y .= 0
        z .= 0
    else
        error("Unknown initial configuration requested.")
    end
    update_united_coordinates(f)
end
```

```
initialize! (generic function with 1 method)
```



```julia
function magnetic_force!(::FerromagneticContinuous, f::AbstractFilamentCache, t)
    # TODO: generalize this for different magnetic fields as well
    N, μ, Cm, ω, F = f.N, f.μ, f.Cm, f.F.ω, f.F.F
    F[1]         = -μ * Cm * cos(ω*t)
    F[2]         = -μ * Cm * sin(ω*t)
    F[3*(N+1)-2] =  μ * Cm * cos(ω*t)
    F[3*(N+1)-1] =  μ * Cm * sin(ω*t)
    nothing
end
```

```
magnetic_force! (generic function with 1 method)
```



```julia
struct SolverDiffEq <: AbstractSolver end

function (f::FilamentCache)(dr, r, p, t)
    @views f.x, f.y, f.z = r[1:3:end], r[2:3:end], r[3:3:end]
    jacobian!(f)
    projection!(f)
    magnetic_force!(f.F, f, t)
    A, P, F, S1, S2 = f.A, f.P.P, f.F.F, f.Sc.S1, f.Sc.S2

    # implement dr = P * (A*r + F) in an optimized way to avoid temporaries
    mul!(S1, A, r)
    S1 .+= F
    mul!(S2, P, S1)
    copyto!(dr, S2)
    return dr
end
```


```julia
function jacobian!(f::FilamentCache)
    N, x, y, z, J = f.N, f.x, f.y, f.z, f.P.J
    @inbounds for i in 1 : N
        J[i, 3*i-2]     = -2 * (x[i+1]-x[i])
        J[i, 3*i-1]     = -2 * (y[i+1]-y[i])
        J[i, 3*i]       = -2 * (z[i+1]-z[i])
        J[i, 3*(i+1)-2] =  2 * (x[i+1]-x[i])
        J[i, 3*(i+1)-1] =  2 * (y[i+1]-y[i])
        J[i, 3*(i+1)]   =  2 * (z[i+1]-z[i])
    end
    nothing
end
```

```
jacobian! (generic function with 1 method)
```



```julia
function projection!(f::FilamentCache)
    # implement P[:] = I - J'/(J*J')*J in an optimized way to avoid temporaries
    J, P, J_JT, J_JT_LDLT, P0 = f.P.J, f.P.P, f.P.J_JT, f.P.J_JT_LDLT, f.P.P0
    mul!(J_JT, J, J')
    LDLt_inplace!(J_JT_LDLT, J_JT)
    ldiv!(P0, J_JT_LDLT, J)
    mul!(P, P0', J)
    subtract_from_identity!(P)
    nothing
end
```

```
projection! (generic function with 1 method)
```



```julia
function subtract_from_identity!(A)
    lmul!(-1, A)
    @inbounds for i in 1 : size(A,1)
        A[i,i] += 1
    end
    nothing
end
```

```
subtract_from_identity! (generic function with 1 method)
```



```julia
function LDLt_inplace!(L::LinearAlgebra.LDLt{T,SymTridiagonal{T}}, A::Matrix{T}) where {T<:Real}
    n = size(A,1)
    dv, ev = L.data.dv, L.data.ev
    @inbounds for (i,d) in enumerate(diagind(A))
        dv[i] = A[d]
    end
    @inbounds for (i,d) in enumerate(diagind(A,-1))
        ev[i] = A[d]
    end
    @inbounds @simd for i in 1 : n-1
        ev[i]   /= dv[i]
        dv[i+1] -= abs2(ev[i]) * dv[i]
    end
    L
end
```

```
LDLt_inplace! (generic function with 1 method)
```





# Investigating the model

Let's take a look at what results of the model look like:

```julia
function run(::SolverDiffEq; N=20, Cm=32, ω=200, time_end=1., solver=TRBDF2(autodiff=false), reltol=1e-6, abstol=1e-6)
    f = FilamentCache(N, Solver=SolverDiffEq, Cm=Cm, ω=ω)
    r0 = initialize!(:StraightX, f)
    stiffness_matrix!(f)
    prob = ODEProblem(ODEFunction(f, jac=(J, u, p, t)->(mul!(J, f.P.P, f.A); nothing)), r0, (0., time_end))
    sol = solve(prob, solver, dense=false, reltol=reltol, abstol=abstol)
end
```

```
run (generic function with 1 method)
```





This method runs the model with the `TRBDF2` method and the default parameters.

```julia
sol = run(SolverDiffEq())
plot(sol,vars = (0,25))
```

![](figures/Filament_17_1.png)



The model quickly falls into a highly oscillatory mode which then dominates throughout the rest of the solution.

# Work-Precision Diagrams

Now let's build the problem and solve it once at high accuracy to get a reference solution:

```julia
N=20
f = FilamentCache(N, Solver=SolverDiffEq)
r0 = initialize!(:StraightX, f)
stiffness_matrix!(f)
prob = ODEProblem(f, r0, (0., 0.01))

sol = solve(prob, Vern9(), reltol=1e-14, abstol=1e-14)
test_sol = TestSolution(sol);
```




## Omissions

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Rosenbrock23(autodiff=false)),
    Dict(:alg => Rodas4(autodiff=false)),
    Dict(:alg => radau()),
    Dict(:alg=>Exprb43(autodiff=false)),
    Dict(:alg=>Exprb32(autodiff=false)),
    Dict(:alg=>ImplicitEulerExtrapolation(autodiff=false)),
    Dict(:alg=>ImplicitDeuflhardExtrapolation(autodiff=false)),
    Dict(:alg=>ImplicitHairerWannerExtrapolation(autodiff=false)),
    ];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```



Rosenbrock23, Rodas4, Exprb32, Exprb43, extrapolation methods, and Rodas5 do
not perform well at all and are thus dropped from future tests. For reference,
they are in the 10^(2.5) range in for their most accurate run (with
ImplicitEulerExtrapolation takes over a day to run, and had to be prematurely
stopped), so about 500x slower than CVODE_BDF and
thus make the benchmarks take forever. It looks like `radau` fails on this
problem with high tolerance so its values should be ignored since it exits
early. It is thus removed from the next sections.

The EPIRK methods currently do not work on this problem

```julia
sol = solve(prob, EPIRK4s3B(autodiff=false), dt=2^-3)
```

```
Error: MethodError: Cannot `convert` an object of type 
  SubArray{ForwardDiff.Dual{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, F
loat64}, Float64, 1},1,Array{ForwardDiff.Dual{ForwardDiff.Tag{DiffEqBase.Or
dinaryDiffEqTag, Float64}, Float64, 1},1},Tuple{StepRange{Int64{},Int64{}}}
,true} to an object of type 
  SubArray{Float64,1,Array{Float64,1},Tuple{StepRange{Int64{},Int64{}}},tru
e}
Closest candidates are:
  convert(::Type{T}, !Matched::LinearAlgebra.Factorization) where T<:Abstra
ctArray at /cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.8/j
ulia-1.8-latest-linux-x86_64/share/julia/stdlib/v1.8/LinearAlgebra/src/fact
orization.jl:58
  convert(::Type{T}, !Matched::T) where T<:AbstractArray at abstractarray.j
l:16
  convert(::Type{T}, !Matched::T) where T at Base.jl:61
  ...
```





but would be called like:

```julia
abstols=1 ./10 .^(3:5)
reltols=1 ./10 .^(3:5)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => HochOst4(),:dts=>2.0.^(-3:-1:-5)),
    Dict(:alg => EPIRK4s3B(),:dts=>2.0.^(-3:-1:-5)),
    Dict(:alg => EXPRB53s3(),:dts=>2.0.^(-3:-1:-5)),
    ];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```



## High Tolerance (Low Accuracy)

### Endpoint Error

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff=false)),
    Dict(:alg => Trapezoid(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda()),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5())
    ];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_22_1.png)

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => ImplicitEuler(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => ABDF2(autodiff=false)),
    Dict(:alg => QNDF(autodiff=false)),
    Dict(:alg => RadauIIA5(autodiff=false)),
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_23_1.png)

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => CVODE_BDF(linear_solver=:GMRES)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false,linsolve=LinSolveGMRES())),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false,linsolve=LinSolveGMRES())),
];

names = [
    "CVODE-BDF",
    "CVODE-BDF (GMRES)",
    "TRBDF2",
    "TRBDF2 (GMRES)",
    "KenCarp4",
    "KenCarp4 (GMRES)",
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; names=names, appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

```
Error: UndefVarError: LinSolveGMRES not defined
```





### Timeseries Error

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Trapezoid(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => rodas()),
    Dict(:alg => lsoda()),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_25_1.png)



Timeseries errors seem to match final point errors very closely in this problem,
so these are turned off in future benchmarks.

(Confirmed in the other cases)

### Dense Error

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false, dense_errors = true, error_estimate=:L2)
plot(wp)
```

![](figures/Filament_26_1.png)



Dense errors seem to match timeseries errors very closely in this problem, so
these are turned off in future benchmarks.

(Confirmed in the other cases)

## Low Tolerance (High Accuracy)

```julia
abstols=1 ./10 .^(6:12)
reltols=1 ./10 .^(6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Vern7()),
    Dict(:alg => Vern9()),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => dop853()),
    Dict(:alg => ROCK4())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_27_1.png)

```julia
abstols=1 ./10 .^(6:12)
reltols=1 ./10 .^(6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno5(autodiff=false)),
    Dict(:alg => KenCarp5(autodiff=false)),
    Dict(:alg => lsoda()),
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                                    maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_28_1.png)



### Timeseries Error

```julia
abstols=1 ./10 .^(6:12)
reltols=1 ./10 .^(6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno5(autodiff=false)),
    Dict(:alg => KenCarp5(autodiff=false)),
    Dict(:alg => lsoda()),
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false, error_estimate = :l2)
plot(wp)
```



### Dense Error

```julia
abstols=1 ./10 .^(6:12)
reltols=1 ./10 .^(6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno5(autodiff=false)),
    Dict(:alg => KenCarp5(autodiff=false)),
    Dict(:alg => lsoda()),
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false, dense_errors=true, error_estimate = :L2)
plot(wp)
```



# No Jacobian Work-Precision Diagrams

In the previous cases the analytical Jacobian is given and is used by the solvers. Now we will solve the same problem without the analytical Jacobian.

Note that the pre-caching means that the model is not compatible with autodifferentiation by ForwardDiff. Thus all of the native Julia solvers are set to `autodiff=false` to use DiffEqDiffTools.jl's numerical differentiation backend. We'll only benchmark the methods that did well before.

```julia
N=20
f = FilamentCache(N, Solver=SolverDiffEq)
r0 = initialize!(:StraightX, f)
stiffness_matrix!(f)
prob = ODEProblem(ODEFunction(f, jac=nothing), r0, (0., 0.01))

sol = solve(prob, Vern9(), reltol=1e-14, abstol=1e-14)
test_sol = TestSolution(sol.t, sol.u);
```




## High Tolerance (Low Accuracy)

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff=false)),
    Dict(:alg => Trapezoid(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda())
    ];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_32_1.png)

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff=false)),
    Dict(:alg => Trapezoid(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda()),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5())
    ];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_33_1.png)

```julia
abstols=1 ./10 .^(3:8)
reltols=1 ./10 .^(3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => CVODE_BDF(linear_solver=:GMRES)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false,linsolve=LinSolveGMRES())),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false,linsolve=LinSolveGMRES())),
];

names = [
    "CVODE-BDF",
    "CVODE-BDF (GMRES)",
    "TRBDF2",
    "TRBDF2 (GMRES)",
    "KenCarp4",
    "KenCarp4 (GMRES)",
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; names=names, appxsol=test_sol,
                      maxiters=Int(1e6), verbose = false)
plot(wp)
```

```
Error: UndefVarError: LinSolveGMRES not defined
```





## Low Tolerance (High Accuracy)

```julia
abstols=1 ./10 .^(6:12)
reltols=1 ./10 .^(6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff=false)),
    Dict(:alg => TRBDF2(autodiff=false)),
    Dict(:alg => Kvaerno3(autodiff=false)),
    Dict(:alg => KenCarp3(autodiff=false)),
    Dict(:alg => Kvaerno4(autodiff=false)),
    Dict(:alg => KenCarp4(autodiff=false)),
    Dict(:alg => Kvaerno5(autodiff=false)),
    Dict(:alg => KenCarp5(autodiff=false)),
    Dict(:alg => lsoda()),
];
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol=test_sol,
                                    maxiters=Int(1e6), verbose = false)
plot(wp)
```

![](figures/Filament_35_1.png)



## Conclusion

Sundials' `CVODE_BDF` does the best in this test. When the Jacobian is given, the ESDIRK methods `TRBDF2` and `KenCarp3` are able to do almost as well as it until `<1e-6` error is needed. When Jacobians are not given, Sundials is the fastest without competition.


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/MOLPDE","Filament.jmd")
```

Computer Information:

```
Julia Version 1.8.3
Commit 0434deb161e (2022-11-14 20:14 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-13.0.1 (ORCJIT, znver2)
  Threads: 128 on 128 virtual cores
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
  LD_LIBRARY_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7661e5a9aa217ce3c468389d834a4fb43b0911e8/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/2e8fae88dcadc37883e31246fe7397f4f1039f88/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/d00220164876dea2cb19993200662745eed5e2db/lib:/cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.8/julia-1.8-latest-linux-x86_64/bin/../lib/julia:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/3e19866657986912870f596aecfee137473965a9/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/dc526f26fb179a3f68eb13fcbe5d2d2a5aa7eeac/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/5a508a62784097dab7c7ae5805f2c89d2cc97397/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/303fe895f57042ea41055187ec4af6322989b5cc/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f3337de0321b3370b90643d18bf63bd4ee79c991/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/ddfc455343aff48d27c1b39d7fcb07e0d9242b50/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/54c97eb1b0a6f74bac96297a815ddec2204a7db7/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/2b77b304b0975d15bd5aeb4d1d5097ac6256ea3c/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/fac7e6d8fc4c5775bf5118ab494120d2a0db4d64/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/387d89822da323c098aba6f8ab316874d4e90f2e/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/16154f990153825ec24b52aac11165df2084b9dc/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/92111ef825c608ea220f8e679dd8d908d7ac5b83/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f3ec73d7bf2f4419ba0943e94f7738cf56050797/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4487a7356408c3a92924e56f9d3891724855282c/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/51c48c945ae76d6c0102649044d9976d93b78125/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/5ba11d7fb2ceb4ca812844eb4af886a212b47f65/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/431a0e3706ffe717ab5d35c47bc38626c6169504/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/fc6071b99b67da0ae4e49ebab70c369ce9a76c9e/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/527e66fb9b12dfd1f58157fe0b3fd52b84062432/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/921a059ebce52878d7a7944c9c345327958d1f5b/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/8b0284dc2781b9481ff92e281f1db532d8421040/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/d11639e2a53726f2593e25ba98ed7b416f62bbc5/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/62c010876222f83fe8878bf2af0e362083d20ee3/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/ee20a84d0166c074dfa736b642902dd87b4da48d/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/459252c01ffcd08700841efdd4b6d3edfe5916e7/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/cc415631aeb190b075329ce756f690a90e1f873b/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/e19f3bb2eef5fb956b672235ea5323b5be9a0626/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/0631e2a6a31b5692eec7a575836451b16b734ec0/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/89ed5dda220da4354ada1970107e13679914bbbc/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/595f9476b128877ab5bf73883ff6c8dc8dacfe66/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/587de110e5f58fd435dc35b294df31bb7a75f692/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/fc239b3ff5739aeab252bd154fa4dd045fefe629/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/25fa81dbac6496585a91dbdc258273d39442466f/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/eff86eedadb59cff1a61399e3242b3f529ca6f59/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/b409c0eafb4254a980f9e730f6fbe56867890f6a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/a84cc58d5161b950f268bb562e105bbbf4d6004a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f0d193662fead3500b523f94b4f1878daab59a93/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/05616da88f6b36c7c94164d4070776aef18ce46b/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/2df316da869cd97f7d70029428ee1e2e521407cd/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7190f0cb0832b80761cc6d513dd9b935f3e26358/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4daa3879a820580557ef34945e2ae243dfcbba11/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/5aa80c7b8e919cbfee41019069d9b25269befe10/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/8793267ae1f4b96f626caa27147aa0218389c30d/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/694cae97bb3cbf8f1f73f2ecabd891602ccf1751/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/d22cde7583df1d5f71160a8e4676955a66a91f33/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/b610fc4e040c9a46c250ea4792cc64098003578a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/cacd8c147f866d6672e1aca9bb01fb919a81e96a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/b7dc5dce963737414a564aca8d4b82ee388f4fa1/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/0d364e900393f710a03a5bafe2852d76e4d2c2cd/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/79cc5446ced978de84b6e673e01da0ebfdd6e4a5/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/1a2adcee7d99fea18ead33c350332626b262e29a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/84f18b0f422f5d6a023fe871b59a9fc536d04f5c/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/3c7eef5f322b19cd4b5db6b21f8cafda87b8b26c/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4443e44120d70a97f8094a67268a886256077e69/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/988066cc974e30c89774b0c471f47201975a4423/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/26099abd23e11fc440259049997307380be745c8/lib:/cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.8/julia-1.8-latest-linux-x86_64/bin/../lib/julia:/cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.8/julia-1.8-latest-linux-x86_64/bin/../lib:

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/MOLPDE/Project.toml`
⌃ [28f2ccd6] ApproxFun v0.13.10
⌃ [f3b72e0c] DiffEqDevTools v2.32.0
⌃ [5b8099bc] DomainSets v0.5.13
⌃ [7f56f5a3] LSODA v0.7.0
⌃ [94925ecb] MethodOfLines v0.5.0
⌃ [961ee093] ModelingToolkit v8.27.0
  [09606e27] ODEInterfaceDiffEq v3.11.0
⌃ [1dea7af3] OrdinaryDiffEq v6.28.0
⌃ [91a5bcdd] Plots v1.35.3
  [31c91b34] SciMLBenchmarks v0.1.1
⌃ [c3572dad] Sundials v4.10.1
  [37e2e46d] LinearAlgebra
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/MOLPDE/Manifest.toml`
⌃ [c3fe647b] AbstractAlgebra v0.27.5
  [621f4979] AbstractFFTs v1.2.1
⌃ [1520ce14] AbstractTrees v0.4.2
  [79e6a3ab] Adapt v3.4.0
⌃ [28f2ccd6] ApproxFun v0.13.10
⌃ [fbd15aa5] ApproxFunBase v0.7.21
⌃ [59844689] ApproxFunFourier v0.3.2
⌃ [b70543e2] ApproxFunOrthogonalPolynomials v0.5.10
⌃ [f8fcb915] ApproxFunSingularities v0.3.1
  [dce04be8] ArgCheck v2.3.0
  [ec485272] ArnoldiMethod v0.2.0
⌃ [4fba245c] ArrayInterface v6.0.23
⌃ [30b0a656] ArrayInterfaceCore v0.1.20
  [6ba088a2] ArrayInterfaceGPUArrays v0.2.2
⌃ [015c0d05] ArrayInterfaceOffsetArrays v0.1.6
⌃ [b0d46f97] ArrayInterfaceStaticArrays v0.1.4
⌃ [dd5226c6] ArrayInterfaceStaticArraysCore v0.1.0
⌃ [4c555306] ArrayLayouts v0.8.11
  [15f4f7f2] AutoHashEquals v0.2.0
  [13072b0f] AxisAlgorithms v1.0.1
⌃ [aae01518] BandedMatrices v0.17.7
  [198e06fe] BangBang v0.3.37
  [9718e550] Baselet v0.1.1
  [e2ed5e7c] Bijections v0.1.4
  [9e28174c] BinDeps v1.0.2
⌃ [d1d4a3ce] BitFlags v0.1.5
⌃ [62783981] BitTwiddlingConvenienceFunctions v0.1.4
⌃ [8e7c35d0] BlockArrays v0.16.20
  [ffab5731] BlockBandedMatrices v0.11.9
  [fa961155] CEnum v0.4.2
⌃ [2a0fbf3d] CPUSummary v0.1.27
  [00ebfdb7] CSTParser v3.3.6
  [49dc2e85] Calculus v0.5.1
  [d360d2e6] ChainRulesCore v1.15.6
  [9e997f8a] ChangesOfVariables v0.1.4
⌃ [fb6a15b2] CloseOpenIntervals v0.1.10
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.19.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.9.9
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
⌃ [a80b9123] CommonMark v0.8.6
⌃ [38540f10] CommonSolve v0.2.1
  [bbf7d656] CommonSubexpressions v0.3.0
⌅ [34da2185] Compat v3.46.0
⌃ [b152e2b5] CompositeTypes v0.1.2
  [a33af91c] CompositionsBase v0.1.1
  [8f4d0f93] Conda v1.7.0
  [187b0558] ConstructionBase v1.4.1
  [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [717857b8] DSP v0.7.7
⌃ [9a962f9c] DataAPI v1.12.0
  [864edb3b] DataStructures v0.18.13
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [b429d917] DensityInterface v0.4.0
⌃ [2b5f629d] DiffEqBase v6.105.1
⌃ [459566f4] DiffEqCallbacks v2.24.1
⌃ [f3b72e0c] DiffEqDevTools v2.32.0
⌃ [77a26b50] DiffEqNoiseProcess v5.13.0
  [163ba53b] DiffResults v1.1.0
⌃ [b552c78f] DiffRules v1.11.1
  [b4f34e82] Distances v0.10.7
⌃ [31c24e10] Distributions v0.25.75
⌅ [ffbed154] DocStringExtensions v0.8.6
⌃ [5b8099bc] DomainSets v0.5.13
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.4.5
⌃ [d4d017d3] ExponentialUtilities v1.19.0
  [e2ba6199] ExprTools v0.1.8
  [c87230d0] FFMPEG v0.4.1
  [7a1cc6ca] FFTW v1.5.0
⌃ [7034ab61] FastBroadcast v0.2.1
  [9aa1b823] FastClosures v0.3.2
⌅ [442a2c76] FastGaussQuadrature v0.4.9
  [29a986be] FastLapackInterface v1.2.7
⌃ [057dd010] FastTransforms v0.14.5
⌃ [1a297f60] FillArrays v0.13.4
⌃ [6a86dc24] FiniteDiff v2.15.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
⌃ [f6369f11] ForwardDiff v0.10.32
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.1
  [46192b85] GPUArraysCore v0.1.2
⌅ [28b8d3ca] GR v0.69.4
  [a8297547] GenericFFT v0.1.1
  [c145ed77] GenericSchur v0.5.3
  [d7ba0133] Git v1.2.1
  [86223c79] Graphs v1.7.4
  [42e2da0e] Grisu v1.0.2
⌃ [0b43b601] Groebner v0.2.10
  [d5909c97] GroupsCore v0.4.0
⌃ [cd3eb016] HTTP v1.4.0
⌅ [eafb193a] Highlights v0.4.5
⌃ [3e5b6fbb] HostCPUFeatures v0.1.8
  [34004b35] HypergeometricFunctions v0.3.11
  [7073ff75] IJulia v1.23.3
  [615f187c] IfElse v0.1.1
⌃ [4858937d] InfiniteArrays v0.12.6
⌃ [cde9dba0] InfiniteLinearAlgebra v0.6.10
⌃ [e1ba4f0e] Infinities v0.1.5
  [d25df0c9] Inflate v0.1.3
  [83e8ac13] IniFile v0.5.1
  [22cec73e] InitialValues v0.3.1
  [18e54dd8] IntegerMathUtils v0.1.0
  [a98d9a8b] Interpolations v0.14.6
  [8197267c] IntervalSets v0.7.3
  [3587e190] InverseFunctions v0.1.8
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [42fd0dbc] IterativeSolvers v0.9.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.5
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.3
⌃ [98e50ef6] JuliaFormatter v1.0.11
⌃ [ccbc3e58] JumpProcesses v9.2.0
⌅ [ef3ab10e] KLU v0.3.0
⌅ [ba0b0d4f] Krylov v0.8.4
⌅ [0b1a1467] KrylovKit v0.5.4
⌃ [7f56f5a3] LSODA v0.7.0
  [b964fa9f] LaTeXStrings v1.3.0
⌃ [2ee39098] LabelledArrays v1.12.0
  [984bce1d] LambertW v0.4.5
  [23fbe1c1] Latexify v0.15.17
⌃ [10f19ff3] LayoutPointers v0.1.10
⌃ [5078a376] LazyArrays v0.22.11
⌃ [d7e5e226] LazyBandedMatrices v0.8.1
  [d3d80556] LineSearches v7.2.0
⌃ [7ed4a6bd] LinearSolve v1.26.1
⌃ [2ab3a3ac] LogExpFunctions v0.3.18
⌅ [e6f89c97] LoggingExtras v0.4.9
⌃ [bdcacae8] LoopVectorization v0.12.133
  [898213cb] LowRankApprox v0.5.2
  [1914dd2f] MacroTools v0.5.10
  [d125e4d3] ManualMemory v0.1.8
⌃ [a3b82374] MatrixFactorizations v0.9.2
⌃ [739be429] MbedTLS v1.1.6
⌃ [442fdcdd] Measures v0.3.1
  [e9d8d322] Metatheory v1.3.5
⌃ [94925ecb] MethodOfLines v0.5.0
  [128add7d] MicroCollections v0.1.3
  [e1d29d7a] Missings v1.0.2
⌃ [961ee093] ModelingToolkit v8.27.0
⌃ [46d2c3a1] MuladdMacro v0.2.2
  [102ac46a] MultivariatePolynomials v0.4.6
  [ffc61752] Mustache v1.0.14
⌃ [d8a4904e] MutableArithmetics v1.0.5
⌃ [d41bc354] NLSolversBase v7.8.2
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.1
⌅ [8913a72c] NonlinearSolve v0.3.22
  [4d1e1d77] Nullables v1.0.0
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.11.0
⌃ [6fe1bfb0] OffsetArrays v1.12.7
⌃ [4d8831e6] OpenSSL v1.2.1
⌃ [429524aa] Optim v1.7.3
  [bac558e1] OrderedCollections v1.4.1
⌃ [1dea7af3] OrdinaryDiffEq v6.28.0
  [90014a1f] PDMats v0.11.16
  [d96e819e] Parameters v0.12.3
⌃ [69de0a69] Parsers v2.4.1
  [b98c9c47] Pipe v1.3.0
⌃ [ccf2f8ad] PlotThemes v3.0.0
  [995b91a9] PlotUtils v1.3.1
⌃ [91a5bcdd] Plots v1.35.3
⌃ [e409e4f3] PoissonRandom v0.4.1
⌃ [f517fe37] Polyester v0.6.16
⌃ [1d0040c9] PolyesterWeave v0.1.10
  [f27b6e38] Polynomials v3.2.0
  [85a6dd25] PositiveFactorizations v0.2.4
  [d236fae5] PreallocationTools v0.4.4
  [21216c6a] Preferences v1.3.0
  [27ebfcd6] Primes v0.5.3
⌃ [1fd47b50] QuadGK v2.5.0
  [74087812] Random123 v1.6.0
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.5.3
  [c84ed2f1] Ratios v0.4.3
⌃ [3cdcf5f2] RecipesBase v1.3.0
⌃ [01d81517] RecipesPipeline v0.6.7
⌃ [731186ca] RecursiveArrayTools v2.32.0
  [f2c3362d] RecursiveFactorization v0.2.12
  [189a3867] Reexport v1.2.2
  [42d2dcc6] Referenceables v0.1.2
  [05181044] RelocatableFolders v1.0.0
  [ae029012] Requires v1.3.0
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.7.0
⌃ [47965b36] RootedTrees v2.15.0
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.3
  [3cdde19b] SIMDDualNumbers v0.1.1
  [94e857df] SIMDTypes v0.1.0
⌃ [476501e8] SLEEFPirates v0.6.36
  [1bc83da4] SafeTestsets v0.0.1
⌃ [0bca4576] SciMLBase v1.59.5
  [31c91b34] SciMLBenchmarks v0.1.1
  [6c6a2e73] Scratch v1.1.1
  [f8ebbe35] SemiseparableMatrices v0.3.4
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [699a6c99] SimpleTraits v0.9.4
  [47aef6b3] SimpleWeightedGraphs v1.2.1
  [66db9d55] SnoopPrecompile v1.0.1
  [b85f4697] SoftGlobalScope v1.1.0
⌃ [a2af1166] SortingAlgorithms v1.0.1
⌃ [47a9eef4] SparseDiffTools v1.27.0
  [276daf66] SpecialFunctions v2.1.7
  [171d559e] SplittablesBase v0.1.15
  [860ef19b] StableRNGs v1.0.0
⌅ [aedffcd0] Static v0.7.7
⌃ [90137ffa] StaticArrays v1.5.9
  [1e83bf80] StaticArraysCore v1.4.0
  [82ae8749] StatsAPI v1.5.0
  [2913bbd2] StatsBase v0.33.21
  [4c63d2b9] StatsFuns v1.0.1
⌅ [7792a7ef] StrideArraysCore v0.3.15
  [69024149] StringEncodings v0.3.5
⌃ [c3572dad] Sundials v4.10.1
  [d1185830] SymbolicUtils v0.19.11
⌃ [0c5d862f] Symbolics v4.11.1
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.9.0
  [62fd8b95] TensorCore v0.1.1
⌅ [8ea1fca8] TermInterface v0.2.3
  [8290d209] ThreadingUtilities v0.5.0
  [ac1d9e8a] ThreadsX v0.1.11
⌃ [a759f4b9] TimerOutputs v0.5.21
  [c751599d] ToeplitzMatrices v0.7.1
  [0796e94c] Tokenize v0.5.24
  [3bb67fe8] TranscodingStreams v0.9.9
  [28d57a85] Transducers v0.4.74
  [a2a6695c] TreeViews v0.3.0
⌃ [d5829a12] TriangularSolve v0.1.14
  [410a4b4d] Tricks v0.1.6
  [30578b45] URIParser v0.4.1
⌃ [5c2747f8] URIs v1.4.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌃ [1986cc42] Unitful v1.12.0
  [41fe7b60] Unzip v0.2.0
⌃ [3d5dd08c] VectorizationBase v0.21.51
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
⌃ [44d3d7a6] Weave v0.10.9
  [efce3f68] WoodburyMatrices v0.5.5
⌃ [ddb6d928] YAML v0.4.7
⌃ [c2297ded] ZMQ v1.2.1
  [700de1a5] ZygoteRules v0.2.2
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2e619515] Expat_jll v2.4.8+0
  [b22a6f82] FFMPEG_jll v4.4.2+2
  [f5851436] FFTW_jll v3.3.10+0
⌃ [34b6f7d7] FastTransforms_jll v0.6.0+0
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.10.4+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.8+0
⌅ [d2c73de3] GR_jll v0.69.1+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.34.1+0
  [7746bdde] Glib_jll v2.74.0+1
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.1.2+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
⌃ [1d63c593] LLVMOpenMP_jll v14.0.4+0
  [aae0fff6] LSODA_jll v0.1.1+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+1
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.4.0+0
  [38a345b3] Libuuid_jll v2.36.0+0
⌃ [856f044c] MKL_jll v2022.1.0+0
  [c771fb93] ODEInterface_jll v0.0.1+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌃ [458c3c95] OpenSSL_jll v1.1.17+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.40.1+0
⌃ [ea2cea3b] Qt5Base_jll v5.15.3+1
  [f50d1b31] Rmath_jll v0.3.0+0
  [fb77eaff] Sundials_jll v5.2.1+0
  [a2964d1f] Wayland_jll v1.19.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.9.14+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3
  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] Xorg_xtrans_jll v1.4.0+3
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [3161d3a3] Zstd_jll v1.5.2+0
⌅ [214eeab7] fzf_jll v0.29.0+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
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
  [44cfe95a] Pkg v1.8.0
  [de0858da] Printf
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.0
  [a4e569a6] Tar v1.10.1
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v0.5.2+0
  [781609d7] GMP_jll v6.2.1+2
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [3a97d323] MPFR_jll v4.1.1+1
  [c8ffd9c3] MbedTLS_jll v2.28.0+0
  [14a3606d] MozillaCACerts_jll v2022.2.1
  [4536629a] OpenBLAS_jll v0.3.20+0
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.40.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+0
  [83775a58] Zlib_jll v1.2.12+3
  [8e850b90] libblastrampoline_jll v5.1.1+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

