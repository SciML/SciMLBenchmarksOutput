---
author: "Guillaume Dalle"
title: "Sparse AD benchmarks"
---
```julia
using ADTypes
using LinearAlgebra, SparseArrays
using BenchmarkTools, DataFrames
import DifferentiationInterface as DI
import SparseDiffTools as SDT
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using Symbolics: SymbolicsSparsityDetector
using Test
```




## Definitions

```julia
const N = 32
const xyd_brusselator = range(0; stop=1, length=N)
const p = (3.4, 1.0, 10.0, step(xyd_brusselator))

brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) =
    if a == N + 1
        1
    elseif a == 0
        N
    else
        a
    end;
```


```julia
function brusselator_2d_loop(du, u, p, t)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N),
        limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        du[i, j, 1] =
            alpha *
            (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] - 4u[i, j, 1]) +
            B +
            u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y, t)
        du[i, j, 2] =
            alpha *
            (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] - 4u[i, j, 2]) +
            A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end;
```


```julia
function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end;
```


```julia
x0 = init_brusselator_2d(xyd_brusselator);
y0 = similar(x0);

f!(y, x) = brusselator_2d_loop(y, x, p, 0.0);
```




## Sparsity detection

```julia
S1 = ADTypes.jacobian_sparsity(f!, y0, x0, TracerSparsityDetector())
S2 = ADTypes.jacobian_sparsity(f!, y0, x0, SymbolicsSparsityDetector())
@test S1 == S2
```

```
Test Passed
```



```julia
td1 = @belapsed ADTypes.jacobian_sparsity($f!, $y0, $x0, TracerSparsityDetector())
println("Sparsity detection with SparseConnectivityTracer: $td1 s")
```

```
Sparsity detection with SparseConnectivityTracer: 0.002196472 s
```



```julia
td2 = @belapsed ADTypes.jacobian_sparsity($f!, $y0, $x0, SymbolicsSparsityDetector())
println("Sparsity detection with Symbolics: $td2 s")
```

```
Sparsity detection with Symbolics: 0.234422523 s
```



```julia
println("Speedup from new sparsity detection method (>1 is better): $(td2 / td1)")
```

```
Speedup from new sparsity detection method (>1 is better): 106.726843319650
78
```





## Coloring

```julia
S = S1
c1 = ADTypes.column_coloring(S, GreedyColoringAlgorithm())
c2 = SDT.matrix_colors(S)
@test c1 == c2
```

```
Test Passed
```



```julia
tc1 = @belapsed ADTypes.column_coloring($S, GreedyColoringAlgorithm())
println("Coloring with SparseMatrixColorings: $tc1 s")
```

```
Coloring with SparseMatrixColorings: 0.000192108 s
```



```julia
tc2 = @belapsed SDT.matrix_colors($S)
println("Coloring with SDT: $tc2 s")
```

```
Coloring with SDT: 0.004081618 s
```



```julia
println("Speedup from new coloring method (>1 is better): $(tc2 / tc1)")
```

```
Speedup from new coloring method (>1 is better): 21.24647594061674
```





## Compressed differentiation

```julia
backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
);
```


```julia
extras = DI.prepare_jacobian(f!, similar(y0), backend, x0);
J1 = DI.jacobian!(f!, similar(y0), similar(S, eltype(x0)), backend, x0, extras)

cache = SDT.sparse_jacobian_cache(
    backend, SDT.JacPrototypeSparsityDetection(; jac_prototype=S), f!, similar(y0), x0
);
J2 = SDT.sparse_jacobian!(similar(S, eltype(x0)), backend, cache, f!, similar(y0), x0)

@test J1 == J2
```

```
Test Passed
```



```julia
tj1 = @belapsed DI.jacobian!($f!, _y, _J, $backend, $x0, _extras) evals = 1 samples = 100 setup = (
    _y = similar(y0);
    _J = similar(S, eltype(x0));
    _extras = DI.prepare_jacobian(f!, similar(y0), backend, x0)
)
println("Jacobian with DifferentiationInterface: $tj1 s")
```

```
Jacobian with DifferentiationInterface: 0.000214628 s
```



```julia
tj2 = @belapsed SDT.sparse_jacobian!(_J, $backend, _cache, $f!, _y, x0) evals = 1 samples = 100 setup = (
    _y = similar(y0);
    _J = similar(S, eltype(x0));
    _cache = SDT.sparse_jacobian_cache(
        backend, SDT.JacPrototypeSparsityDetection(; jac_prototype=S), f!, similar(y0), x0
    )
)
println("Jacobian with SparseDiffTools: $tj2 s")
```

```
Jacobian with SparseDiffTools: 0.000146579 s
```



```julia
println("Speedup from new differentiation method (>1 is better): $(tj2 / tj1)")
```

```
Speedup from new differentiation method (>1 is better): 0.6829444434090614
```


