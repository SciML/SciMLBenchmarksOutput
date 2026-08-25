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
using OrdinaryDiffEq, ODEInterfaceDiffEq, Sundials, DiffEqDevTools, LSODA, LinearSolve
using OrdinaryDiffEqBDF, OrdinaryDiffEqExponentialRK, OrdinaryDiffEqExtrapolation,
    OrdinaryDiffEqFIRK, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqRosenbrock,
    OrdinaryDiffEqSDIRK, OrdinaryDiffEqStabilizedRK
using SciMLLogging, ADTypes
using LinearAlgebra, RecursiveFactorization
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
    ω::T
    F::Vector{T}
end

mutable struct FilamentCache{
    MagneticForce <: AbstractMagneticForce,
    InextensibilityCache <: AbstractInextensibilityCache,
    SolverCache <: AbstractSolverCache
} <: AbstractFilamentCache
    N::Int
    μ::T
    Cm::T
    x::SubArray{T, 1, Vector{T}, Tuple{StepRange{Int, Int}}, true}
    y::SubArray{T, 1, Vector{T}, Tuple{StepRange{Int, Int}}, true}
    z::SubArray{T, 1, Vector{T}, Tuple{StepRange{Int, Int}}, true}
    A::Matrix{T}
    P::InextensibilityCache
    F::MagneticForce
    Sc::SolverCache
end
```


```julia
struct NoHydroProjectionCache <: AbstractInextensibilityCache
    J::Matrix{T}
    P::Matrix{T}
    J_JT::Matrix{T}
    J_JT_LDLT::LinearAlgebra.LDLt{T, SymTridiagonal{T}}
    P0::Matrix{T}

    function NoHydroProjectionCache(N::Int)
        new(
            zeros(N, 3*(N+1)),          # J
            zeros(3*(N+1), 3*(N+1)),    # P
            zeros(N, N),                 # J_JT
            LinearAlgebra.LDLt{T, SymTridiagonal{T}}(SymTridiagonal(zeros(N), zeros(N-1))),
            zeros(N, 3*(N+1))
        )
    end
end
```


```julia
struct DiffEqSolverCache <: AbstractSolverCache
    S1::Vector{T}
    S2::Vector{T}

    DiffEqSolverCache(N::Integer) = new(zeros(T, 3*(N+1)), zeros(T, 3*(N+1)))
end
```


```julia
function FilamentCache(N = 20; Cm = 32, ω = 200, Solver = SolverDiffEq)
    InextensibilityCache = NoHydroProjectionCache
    SolverCache = DiffEqSolverCache
    tmp = zeros(3*(N+1))
    FilamentCache{FerromagneticContinuous, InextensibilityCache, SolverCache}(
        N, N+1, Cm, view(tmp, 1:3:(3 * (N + 1))),
        view(tmp, 2:3:(3 * (N + 1))), view(tmp, 3:3:(3 * (N + 1))),
        zeros(3*(N+1), 3*(N+1)), # A
        InextensibilityCache(N), # P
        FerromagneticContinuous(ω, zeros(3*(N+1))),
        SolverCache(N)
    )
end
```

```
Main.var"##WeaveSandBox#232".FilamentCache
```



```julia
function stiffness_matrix!(f::AbstractFilamentCache)
    N, μ, A = f.N, f.μ, f.A
    @inbounds for j in axes(A, 2), i in axes(A, 1)

        A[i, j] = j == i ? 1 : 0
    end
    @inbounds for i in 1:3
        A[i, i] = 1
        A[i, 3 + i] = -2
        A[i, 6 + i] = 1

        A[3 + i, i] = -2
        A[3 + i, 3 + i] = 5
        A[3 + i, 6 + i] = -4
        A[3 + i, 9 + i] = 1

        A[3 * (N - 1) + i, 3 * (N - 3) + i] = 1
        A[3 * (N - 1) + i, 3 * (N - 2) + i] = -4
        A[3 * (N - 1) + i, 3 * (N - 1) + i] = 5
        A[3 * (N - 1) + i, 3 * N + i] = -2

        A[3 * N + i, 3 * (N - 2) + i] = 1
        A[3 * N + i, 3 * (N - 1) + i] = -2
        A[3 * N + i, 3 * N + i] = 1

        for j in 2:(N - 2)
            A[3 * j + i, 3 * j + i] = 6
            A[3 * j + i, 3 * (j - 1) + i] = -4
            A[3 * j + i, 3 * (j + 1) + i] = -4
            A[3 * j + i, 3 * (j - 2) + i] = 1
            A[3 * j + i, 3 * (j + 2) + i] = 1
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
    @inbounds for i in 1:length(x)
        x[i] = r[3 * i - 2]
        y[i] = r[3 * i - 1]
        z[i] = r[3 * i]
    end
    nothing
end

function update_united_coordinates!(f::AbstractFilamentCache, r)
    N, x, y, z = f.N, f.x, f.y, f.z
    @inbounds for i in 1:length(x)
        r[3 * i - 2] = x[i]
        r[3 * i - 1] = y[i]
        r[3 * i] = z[i]
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
        x .= range(0, stop = 1, length = N+1)
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
    F[1] = -μ * Cm * cos(ω*t)
    F[2] = -μ * Cm * sin(ω*t)
    F[3 * (N + 1) - 2] = μ * Cm * cos(ω*t)
    F[3 * (N + 1) - 1] = μ * Cm * sin(ω*t)
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
    @inbounds for i in 1:N
        J[i, 3 * i - 2] = -2 * (x[i + 1]-x[i])
        J[i, 3 * i - 1] = -2 * (y[i + 1]-y[i])
        J[i, 3 * i] = -2 * (z[i + 1]-z[i])
        J[i, 3 * (i + 1) - 2] = 2 * (x[i + 1]-x[i])
        J[i, 3 * (i + 1) - 1] = 2 * (y[i + 1]-y[i])
        J[i, 3 * (i + 1)] = 2 * (z[i + 1]-z[i])
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
    @inbounds for i in 1:size(A, 1)
        A[i, i] += 1
    end
    nothing
end
```

```
subtract_from_identity! (generic function with 1 method)
```



```julia
function LDLt_inplace!(L::LinearAlgebra.LDLt{T, SymTridiagonal{T}}, A::Matrix{T}) where {T <:
                                                                                         Real}
    n = size(A, 1)
    dv, ev = L.data.dv, L.data.ev
    @inbounds for (i, d) in enumerate(diagind(A))
        dv[i] = A[d]
    end
    @inbounds for (i, d) in enumerate(diagind(A, -1))
        ev[i] = A[d]
    end
    @inbounds @simd for i in 1:(n - 1)
        ev[i] /= dv[i]
        dv[i + 1] -= abs2(ev[i]) * dv[i]
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
function run(::SolverDiffEq; N = 20, Cm = 32, ω = 200, time_end = 1.0,
        solver = TRBDF2(autodiff = AutoFiniteDiff()), reltol = 1e-6, abstol = 1e-6)
    f = FilamentCache(N, Solver = SolverDiffEq, Cm = Cm, ω = ω)
    r0 = initialize!(:StraightX, f)
    stiffness_matrix!(f)
    prob = ODEProblem(ODEFunction(f, jac = (J, u, p, t)->(mul!(J, f.P.P, f.A); nothing)), r0, (
        0.0, time_end))
    sol = solve(prob, solver, dense = false, reltol = reltol, abstol = abstol)
end
```

```
run (generic function with 1 method)
```





This method runs the model with the `TRBDF2` method and the default parameters.

```julia
sol = run(SolverDiffEq())
plot(sol, vars = (0, 25))
```

![](figures/Filament_17_1.png)



The model quickly falls into a highly oscillatory mode which then dominates throughout the rest of the solution.

# Work-Precision Diagrams

Now let's build the problem and solve it once at high accuracy to get a reference solution:

```julia
N=20
f = FilamentCache(N, Solver = SolverDiffEq)
r0 = initialize!(:StraightX, f)
stiffness_matrix!(f)
prob = ODEProblem(f, r0, (0.0, 0.01))

sol = solve(prob, Vern9(), reltol = 1e-14, abstol = 1e-14)
test_sol = TestSolution(sol);
```




## Omissions

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Rosenbrock23(autodiff = AutoFiniteDiff())),
    Dict(:alg => Rodas4(autodiff = AutoFiniteDiff())),
    Dict(:alg => radau()),
    Dict(:alg=>Exprb43(autodiff = AutoFiniteDiff())),
    Dict(:alg=>Exprb32(autodiff = AutoFiniteDiff())),
    Dict(:alg=>ImplicitEulerExtrapolation(autodiff = AutoFiniteDiff())),
    Dict(:alg=>ImplicitDeuflhardExtrapolation(autodiff = AutoFiniteDiff())),
    Dict(:alg=>ImplicitHairerWannerExtrapolation(autodiff = AutoFiniteDiff()))
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
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
try
    sol = solve(prob, EPIRK4s3B(autodiff = AutoFiniteDiff()), dt = 2^-3)
catch e
    println("EPIRK4s3B failed: $e")
end
```

```
retcode: Success
Interpolation: 3rd order Hermite
t: 2-element Vector{Float64}:
 0.0
 0.01
u: 2-element Vector{Vector{Float64}}:
 [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.15  …  0.0, 0.9, 0.0, 0.0
, 0.95, 0.0, 0.0, 1.0, 0.0, 0.0]
 [0.002095721720837016, 1.094023712191579, 0.0, 0.05181137843134699, 0.9324
864214423351, 0.0, 0.10127054158062286, 0.7787771092588137, 0.0, 0.15070550
878833638  …  0.0, 0.9012705640131832, -0.7787771092593901, 0.0, 0.95181139
45448426, -0.9324864214428036, 0.0, 1.0020957319453148, -1.0940237121919623
, 0.0]
```





but would be called like:

```julia
abstols=1 ./ 10 .^ (3:5)
reltols=1 ./ 10 .^ (3:5)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => HochOst4(), :dts=>2.0 .^ (-3:-1:-5)),
    Dict(:alg => EPIRK4s3B(), :dts=>2.0 .^ (-3:-1:-5)),
    Dict(:alg => EXPRB53s3(), :dts=>2.0 .^ (-3:-1:-5))
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```



## High Tolerance (Low Accuracy)

### Endpoint Error

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff = AutoFiniteDiff())),
    Dict(:alg => Trapezoid(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda()),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5()),
    Dict(:alg => RKC()),
    Dict(:alg => TSRKC3())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_22_1.png)

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => ImplicitEuler(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => ABDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => QNDF(autodiff = AutoFiniteDiff())),
    Dict(:alg => NordsieckBDF(autodiff = AutoFiniteDiff())),
    Dict(:alg => RadauIIA5(autodiff = AutoFiniteDiff()))
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_23_1.png)

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => CVODE_BDF(linear_solver = :GMRES)),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES()))
];

names = [
    "CVODE-BDF",
    "CVODE-BDF (GMRES)",
    "TRBDF2",
    "TRBDF2 (GMRES)",
    "KenCarp4",
    "KenCarp4 (GMRES)"
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; names = names, appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_24_1.png)



### Timeseries Error

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Trapezoid(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => rodas()),
    Dict(:alg => lsoda()),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5()),
    Dict(:alg => RKC()),
    Dict(:alg => TSRKC3())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_25_1.png)



Timeseries errors seem to match final point errors very closely in this problem,
so these are turned off in future benchmarks.

(Confirmed in the other cases)

### Dense Error

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5()),
    Dict(:alg => RKC()),
    Dict(:alg => TSRKC3())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None(), dense_errors = true, error_estimate = :L2)
plot(wp)
```

![](figures/Filament_26_1.png)



Dense errors seem to match timeseries errors very closely in this problem, so
these are turned off in future benchmarks.

(Confirmed in the other cases)

## Low Tolerance (High Accuracy)

```julia
abstols=1 ./ 10 .^ (6:12)
reltols=1 ./ 10 .^ (6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => Vern7()),
    Dict(:alg => Vern9()),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => dop853()),
    Dict(:alg => ROCK4())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_27_1.png)

```julia
abstols=1 ./ 10 .^ (6:12)
reltols=1 ./ 10 .^ (6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno5(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp5(autodiff = AutoFiniteDiff())),
    Dict(:alg => lsoda())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_28_1.png)



### Timeseries Error

```julia
abstols=1 ./ 10 .^ (6:12)
reltols=1 ./ 10 .^ (6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno5(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp5(autodiff = AutoFiniteDiff())),
    Dict(:alg => lsoda())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None(), error_estimate = :l2)
plot(wp)
```



### Dense Error

```julia
abstols=1 ./ 10 .^ (6:12)
reltols=1 ./ 10 .^ (6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno5(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp5(autodiff = AutoFiniteDiff())),
    Dict(:alg => lsoda())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None(), dense_errors = true, error_estimate = :L2)
plot(wp)
```



# No Jacobian Work-Precision Diagrams

In the previous cases the analytical Jacobian is given and is used by the solvers. Now we will solve the same problem without the analytical Jacobian.

Note that the pre-caching means that the model is not compatible with autodifferentiation by ForwardDiff. Thus all of the native Julia solvers are set to `autodiff=AutoFiniteDiff()` to use DiffEqDiffTools.jl's numerical differentiation backend. We'll only benchmark the methods that did well before.

```julia
N=20
f = FilamentCache(N, Solver = SolverDiffEq)
r0 = initialize!(:StraightX, f)
stiffness_matrix!(f)
prob = ODEProblem(ODEFunction(f, jac = nothing), r0, (0.0, 0.01))

sol = solve(prob, Vern9(), reltol = 1e-14, abstol = 1e-14)
test_sol = TestSolution(sol.t, sol.u);
```




## High Tolerance (Low Accuracy)

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff = AutoFiniteDiff())),
    Dict(:alg => Trapezoid(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_32_1.png)

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => BS3()),
    Dict(:alg => Tsit5()),
    Dict(:alg => ImplicitEuler(autodiff = AutoFiniteDiff())),
    Dict(:alg => Trapezoid(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => rodas()),
    Dict(:alg => dop853()),
    Dict(:alg => lsoda()),
    Dict(:alg => ROCK2()),
    Dict(:alg => ROCK4()),
    Dict(:alg => ESERK5()),
    Dict(:alg => RKC()),
    Dict(:alg => TSRKC3())
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_33_1.png)

```julia
abstols=1 ./ 10 .^ (3:8)
reltols=1 ./ 10 .^ (3:8)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => CVODE_BDF(linear_solver = :GMRES)),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES()))
];

names = [
    "CVODE-BDF",
    "CVODE-BDF (GMRES)",
    "TRBDF2",
    "TRBDF2 (GMRES)",
    "KenCarp4",
    "KenCarp4 (GMRES)"
];

wp = WorkPrecisionSet(prob, abstols, reltols, setups; names = names, appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
plot(wp)
```

![](figures/Filament_34_1.png)



## Low Tolerance (High Accuracy)

```julia
abstols=1 ./ 10 .^ (6:12)
reltols=1 ./ 10 .^ (6:12)
setups = [
    Dict(:alg => CVODE_BDF()),
    Dict(:alg => radau()),
    Dict(:alg => RadauIIA5(autodiff = AutoFiniteDiff())),
    Dict(:alg => TRBDF2(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno3(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp3(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno4(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp4(autodiff = AutoFiniteDiff())),
    Dict(:alg => Kvaerno5(autodiff = AutoFiniteDiff())),
    Dict(:alg => KenCarp5(autodiff = AutoFiniteDiff())),
    Dict(:alg => lsoda())
];
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
    maxiters = Int(1e6), verbose = SciMLLogging.None())
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
SciMLBenchmarks.weave_file("benchmarks/ComplicatedPDE","Filament.jmd")
```

Computer Information:

```
Julia Version 1.11.9
Commit 53a02c0720c (2026-02-06 00:27 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)
Threads: 128 default, 0 interactive, 64 GC (on 128 virtual cores)
Environment:
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/ComplicatedPDE/Project.toml`
  [47edcb42] ADTypes v1.23.0
⌃ [f3b72e0c] DiffEqDevTools v3.2.0
  [e4b2fa32] GaussianRandomFields v2.2.7
  [7073ff75] IJulia v1.34.4
  [7f56f5a3] LSODA v1.1.0
⌅ [7ed4a6bd] LinearSolve v3.87.0
⌃ [961ee093] ModelingToolkit v11.38.2
⌃ [8913a72c] NonlinearSolve v4.21.0
⌅ [09606e27] ODEInterfaceDiffEq v4.1.0
⌃ [1dea7af3] OrdinaryDiffEq v7.2.1
⌃ [6ad6398a] OrdinaryDiffEqBDF v2.4.2
  [e0540318] OrdinaryDiffEqExponentialRK v2.2.1
⌃ [becaefa8] OrdinaryDiffEqExtrapolation v2.3.1
⌃ [5960d6e9] OrdinaryDiffEqFIRK v2.4.1
  [1344f307] OrdinaryDiffEqLowOrderRK v2.2.3
⌃ [43230ef6] OrdinaryDiffEqRosenbrock v2.4.2
⌃ [2d112036] OrdinaryDiffEqSDIRK v2.8.1
  [358294b1] OrdinaryDiffEqStabilizedRK v2.5.1
  [91a5bcdd] Plots v1.41.6
  [f2c3362d] RecursiveFactorization v0.2.30
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [a6db7da4] SciMLLogging v2.0.4
  [860ef19b] StableRNGs v1.0.4
⌃ [c3572dad] Sundials v6.4.2
  [0c5d862f] Symbolics v7.36.0
  [2f01184e] SparseArrays v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/ComplicatedPDE/Manifest.toml`
  [47edcb42] ADTypes v1.23.0
  [14f7f29c] AMD v0.5.3
  [621f4979] AbstractFFTs v1.5.0
  [6e696c72] AbstractPlutoDingetjes v1.4.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [66dad0bd] AliasTables v1.1.3
  [ec485272] ArnoldiMethod v0.4.0
  [7d9fca2a] Arpack v0.5.4
⌃ [4fba245c] ArrayInterface v7.28.1
  [4c555306] ArrayLayouts v1.12.2
  [aae01518] BandedMatrices v1.11.0
  [e2ed5e7c] Bijections v0.2.2
  [b2a6c25c] BinaryHeaps v1.0.4
  [caf10ac8] BipartiteGraphs v0.1.11
  [d1d4a3ce] BitFlags v0.1.10
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
  [8e7c35d0] BlockArrays v1.10.0
  [70df07ce] BracketingNonlinearSolve v1.12.5
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.7
  [fb6a15b2] CloseOpenIntervals v0.1.13
⌃ [944b1d66] CodecZlib v0.7.8
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌅ [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.13
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.1.2
  [34da2185] Compat v4.18.1
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.7
  [f0e56b4a] ConcurrentUtilities v2.6.0
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [9a962f9c] DataAPI v1.16.0
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v7.13.1
  [459566f4] DiffEqCallbacks v4.19.2
⌃ [f3b72e0c] DiffEqDevTools v3.2.0
⌃ [77a26b50] DiffEqNoiseProcess v5.34.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
⌃ [a0c0ee7d] DifferentiationInterface v0.7.20
  [31c24e10] Distributions v0.25.130
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.8.1
  [7c1d4256] DynamicPolynomials v0.6.6
  [4e289a0a] EnumX v1.0.7
  [f151be2c] EnzymeCore v0.8.21
  [460bff9d] ExceptionUnwrapping v0.1.11
⌃ [d4d017d3] ExponentialUtilities v1.31.0
  [e2ba6199] ExprTools v0.1.11
  [55351af7] ExproniconLite v0.10.14
  [c87230d0] FFMPEG v0.4.5
  [7a1cc6ca] FFTW v1.10.0
  [7034ab61] FastBroadcast v1.3.6
  [9aa1b823] FastClosures v0.3.2
  [442a2c76] FastGaussQuadrature v1.3.0
  [a4df4552] FastPower v1.4.1
  [1a297f60] FillArrays v1.17.0
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.4.5
  [a85aefff] FunctionMaps v0.1.2
  [069b7b12] FunctionWrappers v1.1.3
⌃ [77dc65aa] FunctionWrappersWrappers v1.12.1
  [46192b85] GPUArraysCore v0.2.0
  [28b8d3ca] GR v0.73.26
  [a0844989] Gamma v1.2.0
  [e4b2fa32] GaussianRandomFields v2.2.7
  [c145ed77] GenericSchur v0.5.6
  [d7ba0133] Git v1.5.0
  [86223c79] Graphs v1.14.0
  [42e2da0e] Grisu v1.0.2
⌅ [cd3eb016] HTTP v1.11.0
⌅ [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.18
  [34004b35] HypergeometricFunctions v0.3.30
  [7073ff75] IJulia v1.34.4
  [615f187c] IfElse v0.1.1
⌃ [3263718b] ImplicitDiscreteSolve v2.1.4
  [d25df0c9] Inflate v0.1.5
  [18e54dd8] IntegerMathUtils v0.1.4
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
⌃ [ccbc3e58] JumpProcesses v9.29.2
  [ba0b0d4f] Krylov v0.10.9
  [7f56f5a3] LSODA v1.1.0
⌃ [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.11
  [10f19ff3] LayoutPointers v0.1.17
⌃ [87fe0de2] LineSearch v0.1.14
⌅ [7ed4a6bd] LinearSolve v3.87.0
  [2ab3a3ac] LogExpFunctions v1.0.1
  [e6f89c97] LoggingExtras v1.2.0
  [bdcacae8] LoopVectorization v0.12.174
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.7
  [739be429] MbedTLS v1.1.10
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
⌃ [961ee093] ModelingToolkit v11.38.2
⌃ [7771a370] ModelingToolkitBase v1.62.0
  [6bb917b9] ModelingToolkitTearing v1.20.5
  [2e0e35c7] Moshi v0.3.12
  [46d2c3a1] MuladdMacro v0.2.7
  [102ac46a] MultivariatePolynomials v0.5.19
  [ffc61752] Mustache v1.0.21
  [d8a4904e] MutableArithmetics v1.8.0
  [77ba4419] NaNMath v1.1.4
⌃ [8913a72c] NonlinearSolve v4.21.0
⌅ [be0214bd] NonlinearSolveBase v2.33.0
⌃ [5959db7a] NonlinearSolveFirstOrder v2.2.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.13.3
⌃ [26075421] NonlinearSolveSpectralMethods v1.7.4
  [54ca160b] ODEInterface v0.5.2
⌅ [09606e27] ODEInterfaceDiffEq v4.1.0
  [6fe1bfb0] OffsetArrays v1.17.0
  [4d8831e6] OpenSSL v1.6.1
⌅ [bac558e1] OrderedCollections v1.8.2
⌃ [1dea7af3] OrdinaryDiffEq v7.2.1
⌃ [6ad6398a] OrdinaryDiffEqBDF v2.4.2
⌃ [bbf590c4] OrdinaryDiffEqCore v4.14.1
⌃ [50262376] OrdinaryDiffEqDefault v2.3.0
⌃ [4302a76b] OrdinaryDiffEqDifferentiation v3.4.1
  [e0540318] OrdinaryDiffEqExponentialRK v2.2.1
⌃ [becaefa8] OrdinaryDiffEqExtrapolation v2.3.1
⌃ [5960d6e9] OrdinaryDiffEqFIRK v2.4.1
  [1344f307] OrdinaryDiffEqLowOrderRK v2.2.3
⌅ [127b3ac7] OrdinaryDiffEqNonlinearSolve v2.0.2
⌃ [43230ef6] OrdinaryDiffEqRosenbrock v2.4.2
  [b4bd8bb3] OrdinaryDiffEqRosenbrockTableaus v2.4.1
⌃ [2d112036] OrdinaryDiffEqSDIRK v2.8.1
  [358294b1] OrdinaryDiffEqStabilizedRK v2.5.1
⌃ [b1df2697] OrdinaryDiffEqTsit5 v2.1.2
⌃ [79d7bb75] OrdinaryDiffEqVerner v2.2.2
  [90014a1f] PDMats v0.11.41
⌅ [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.7
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.6
  [e409e4f3] PoissonRandom v0.4.13
  [f517fe37] Polyester v0.7.19
  [1d0040c9] PolyesterWeave v0.2.2
  [d236fae5] PreallocationTools v1.5.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [27ebfcd6] Primes v0.5.7
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.4.1
  [1fd47b50] QuadGK v2.11.3
  [988b38a3] ReadOnlyArrays v0.2.0
  [795d4caa] ReadOnlyDicts v1.0.1
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌃ [731186ca] RecursiveArrayTools v4.4.0
  [f2c3362d] RecursiveFactorization v0.2.30
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.3.0
  [9fe22ead] RespecializeParams v1.2.0
  [79098fc4] Rmath v0.9.0
⌃ [47965b36] RootedTrees v2.25.4
  [f2b01f46] Roots v3.0.6
  [7e49a35a] RuntimeGeneratedFunctions v0.5.24
⌃ [9dfe8606] SCCNonlinearSolve v1.13.3
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.46
⌅ [0bca4576] SciMLBase v3.39.1
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.17
  [a6db7da4] SciMLLogging v2.0.4
⌃ [c0aeaf25] SciMLOperators v1.26.1
  [431bcebd] SciMLPublic v1.2.4
  [53ae85a6] SciMLStructures v1.10.4
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.13.1
  [699a6c99] SimpleTraits v0.9.6
  [a2af1166] SortingAlgorithms v1.2.3
⌃ [bd59d7e1] SparseBandedMatrices v1.3.4
  [a57abbd0] SparseColumnPivotedQR v2.1.6
  [0a514795] SparseMatrixColorings v0.4.27
⌃ [276daf66] SpecialFunctions v2.8.3
  [860ef19b] StableRNGs v1.0.4
  [0c0c59c1] StarAlgebras v0.3.0
  [64909d44] StateSelection v1.11.0
  [aedffcd0] Static v1.4.6
  [0d7ed370] StaticArrayInterface v1.10.0
⌃ [90137ffa] StaticArrays v1.9.18
  [1e83bf80] StaticArraysCore v1.4.4
  [10745b16] Statistics v1.11.1
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.12
  [4c63d2b9] StatsFuns v2.2.1
  [7792a7ef] StrideArraysCore v0.5.9
  [69024149] StringEncodings v0.3.7
  [09ab397b] StructArrays v0.7.3
⌃ [c3572dad] Sundials v6.4.2
  [2efcf032] SymbolicIndexingInterface v0.3.54
  [19f23fe9] SymbolicLimits v1.1.5
  [d1185830] SymbolicUtils v4.45.0
  [0c5d862f] Symbolics v7.36.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.13.0
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [8290d209] ThreadingUtilities v0.5.6
⌅ [a759f4b9] TimerOutputs v0.5.29
  [3bb67fe8] TranscodingStreams v0.11.3
  [d5829a12] TriangularSolve v0.2.6
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [5c2747f8] URIs v1.6.3
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.74
  [33b4df10] VectorizedRNG v0.2.26
  [81def892] VersionParsing v1.3.0
  [d30d5f5c] WeakCacheSets v0.1.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
⌅ [68821587] Arpack_jll v3.5.2+0
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.8.2+0
⌅ [b22a6f82] FFMPEG_jll v8.1.2+0
  [f5851436] FFTW_jll v3.3.12+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.4.1+1
  [d2c73de3] GR_jll v0.73.26+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.55.0+0
  [7746bdde] Glib_jll v2.88.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
⌅ [2e76f6c2] HarfBuzz_jll v8.5.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.1.0+0
  [1d63c593] LLVMOpenMP_jll v22.1.7+0
  [aae0fff6] LSODA_jll v0.1.2+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [856f044c] MKL_jll v2025.2.0+0
  [c771fb93] ODEInterface_jll v0.0.2+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [656ef2d0] OpenBLAS32_jll v0.3.34+0
⌃ [9bd350c2] OpenSSH_jll v10.4.1+0
  [458c3c95] OpenSSL_jll v3.5.7+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.58.0+0
  [30392449] Pixman_jll v0.46.4+0
  [c0090381] Qt6Base_jll v6.10.2+2
  [629bc702] Qt6Declarative_jll v6.10.2+2
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
  [f50d1b31] Rmath_jll v0.5.2+0
  [ca45d3f4] SuiteSparse32_jll v7.12.1+0
  [fb77eaff] Sundials_jll v7.5.0+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
  [ffd25f8a] XZ_jll v5.8.3+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.13+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.8+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.2+0
  [a51aa0fd] Xorg_libXi_jll v1.8.4+0
  [d1454406] Xorg_libXinerama_jll v1.1.7+0
  [ec84b674] Xorg_libXrandr_jll v1.5.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [a65dc6b1] Xorg_libpciaccess_jll v0.19.0+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.2.0+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.6+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.47.0+2
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
⌅ [214eeab7] fzf_jll v0.61.1+0
  [a4ae2306] libaom_jll v3.14.1+0
  [0ac62f75] libass_jll v0.17.4+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [8e53e030] libdrm_jll v2.4.134+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.58+0
  [a9144af2] libsodium_jll v1.0.21+0
  [9a156e7d] libva_jll v2.23.0+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.3.0+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [37e2e46d] LinearAlgebra v1.11.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.11.0
  [de0858da] Printf v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.11.0
  [f489334b] StyledStrings v1.11.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.6.0+0
  [e37daf67] LibGit2_jll v1.7.2+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.6+0
  [14a3606d] MozillaCACerts_jll v2023.12.12
  [4536629a] OpenBLAS_jll v0.3.27+1
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.7.0+0
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

