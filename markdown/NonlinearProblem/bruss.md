---
author: "Avik Pal"
title: "Ill-Conditioned Nonlinear System Work-Precision Diagrams"
---


# Setup

Fetch required packages

```julia
using NonlinearSolve, NLsolve, MINPACK, SparseDiffTools, Sundials
using LinearAlgebra, SparseArrays, DiffEqDevTools, CairoMakie, Symbolics
using BenchmarkTools
RUS = RadiusUpdateSchemes;
```




Define the Brussletor problem.

```julia
brusselator_f(x, y) = (((x - 3 // 10) ^ 2 + (y - 6 // 10) ^ 2) ≤ 0.01) * 5

limit(a, N) = ifelse(a == N + 1, 1, ifelse(a == 0, N, a))

function init_brusselator_2d(xyd, N)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end

function generate_brusselator_problem(N::Int; sparsity = nothing, kwargs...)
    xyd_brusselator = range(0; stop = 1, length = N)

    function brusselator_2d_loop(du, u, p)
        A, B, α, δx = p
        α = α / δx ^ 2
        @inbounds for I in CartesianIndices((N, N))
            i, j = Tuple(I)
            x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
            ip1, im1 = limit(i + 1, N), limit(i - 1, N)
            jp1, jm1 = limit(j + 1, N), limit(j - 1, N)

            du[i, j, 1] = α * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                               4u[i, j, 1]) +
                          B + u[i, j, 1] ^ 2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
                          brusselator_f(x, y)

            du[i, j, 2] = α * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                               4u[i, j, 2]) +
                            A * u[i, j, 1] - u[i, j, 1] ^ 2 * u[i, j, 2]
        end
        return nothing
    end

    p = (3.4, 1.0, 10.0, step(xyd_brusselator))

    u0 = init_brusselator_2d(xyd_brusselator, N)

    nlfunc = NonlinearFunction(brusselator_2d_loop; sparsity)
    return NonlinearProblem(nlfunc, u0, p; kwargs...)
end
```

```
generate_brusselator_problem (generic function with 1 method)
```





# Scaling with Problem Size

First, let us experiment the scaling of each algorithm with the problem size.

```julia
prob_dense = generate_brusselator_problem(4)
prob_approx_sparse = generate_brusselator_problem(4;
    sparsity = ApproximateJacobianSparsity())
prob_exact_sparse = generate_brusselator_problem(4;
    sparsity = SymbolicsSparsityDetection())

@btime solve(prob_dense, NewtonRaphson())
@btime solve(prob_approx_sparse, NewtonRaphson())
@btime solve(prob_exact_sparse, NewtonRaphson())
```

```
57.839 μs (192 allocations: 38.44 KiB)
  400.617 μs (1374 allocations: 389.06 KiB)
  5.126 ms (27108 allocations: 1.99 MiB)
retcode: Success
u: 4×4×2 Array{Float64, 3}:
[:, :, 1] =
 1.30971  1.31119  1.31442  1.31119
 1.31119  1.31442  1.32754  1.31442
 1.30971  1.31119  1.31442  1.31119
 1.30882  1.30971  1.31119  1.30971

[:, :, 2] =
 2.59051  2.59046  2.59039  2.59046
 2.59046  2.59039  2.59025  2.59039
 2.59051  2.59046  2.59039  2.59046
 2.59054  2.59051  2.59046  2.59051
```


