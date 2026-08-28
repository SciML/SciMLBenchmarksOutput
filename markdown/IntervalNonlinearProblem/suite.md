---
author: "Fabian Gittins"
title: "Interval root-finding test suite"
---


In this benchmark, we will examine how the interval root-finding algorithms
provided in `BracketingNonlinearSolve.jl` and `SimpleNonlinearSolve.jl` fare against one another for a selection of
challenging test functions from the literature.

## `Roots.jl` baseline

To give us sensible measure to compare with, we will use the `Roots.jl` package
as a baseline,

```julia
using BenchmarkTools
using Roots
```




and search for the roots of the function

```julia
f(u, p) = u * sin(u) - p;
```




To get a good idea of the performance of the algorithms, we will use a large
number of random `p` values and determine the roots with all of them.
Specifically, we will draw `N = 100_000` random values (which we seed for
reproducibility),

```julia
using Random

Random.seed!(42)

const N = 100_000
ps = 1.5 .* rand(N)

function g!(out, ps, uspan)
    for i in 1:N
        out[i] = find_zero(f, uspan, ps[i])
    end
    return out
end;
```




Now, we can run the benchmark for `Roots.jl`:

```julia
out = zeros(N)
uspan = (0.0, 2.0)

@btime g!(out, ps, uspan);
```

```
344.476 ms (0 allocations: 0 bytes)
```





However, speed is not the only thing we care about. We also want the algorithms
to be accurate. We will use the mean of the absolute errors to measure the
accuracy,

```julia
println("Mean absolute error: $(mean(abs.(f.(out, ps))))")
```

```
Mean absolute error: 3.850522738114634e-17
```





For simplicity, we will assume the default tolerances of the methods, while
noting that these can be set.

## `BracketingNonlinearSolve.jl` algorithms

With the preliminaries out of the way, let's see how the `BracketingNonlinearSolve.jl`
solvers perform! We define a (non-allocating) function to benchmark,

```julia
using BracketingNonlinearSolve
using BracketingNonlinearSolve: Bisection # Roots also exports Bisection leading to a name conflict

function h!(out, ps, uspan, alg)
    for i in 1:N
        prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(f), uspan, ps[i])
        sol = solve(prob, alg)
        out[i] = sol.u
    end
    return out
end;
```




and loop through the methods,

```julia
for alg in (
        Alefeld, Bisection, Brent, Falsi,
        ITP, Muller, Ridder, ModAB,
    )
    println("Benchmark of $alg:")
    @btime h!($out, $ps, $uspan, $(alg()))
    println("Mean absolute error: $(mean(abs.(f.(out, ps))))\n")
end
```

```
Benchmark of BracketingNonlinearSolve.Alefeld:
  178.612 ms (0 allocations: 0 bytes)
Mean absolute error: 3.918693955483679e-17

Benchmark of BracketingNonlinearSolve.Bisection:
  128.644 ms (0 allocations: 0 bytes)
Mean absolute error: 1.3280442898291502e-13

Benchmark of BracketingNonlinearSolve.Brent:
  45.943 ms (0 allocations: 0 bytes)
Mean absolute error: 2.1665706599117464e-14

Benchmark of BracketingNonlinearSolve.Falsi:
  130.088 ms (0 allocations: 0 bytes)
Mean absolute error: 2.2704001668921334e-12

Benchmark of BracketingNonlinearSolve.ITP:
  46.231 ms (0 allocations: 0 bytes)
Mean absolute error: 1.6533434955381716e-16

Benchmark of BracketingNonlinearSolve.Muller:
  27.485 ms (0 allocations: 0 bytes)
Mean absolute error: 2.0138978308733098e-14

Benchmark of BracketingNonlinearSolve.Ridder:
  38.948 ms (0 allocations: 0 bytes)
Mean absolute error: 8.267108511975618e-15

Benchmark of BracketingNonlinearSolve.ModAB:
  28.375 ms (0 allocations: 0 bytes)
Mean absolute error: 4.442012919601487e-17
```





Although each method finds the roots with different accuracies, we can see that
all the `NonlinearSolve.jl` algorithms are performant and non-allocating.

## A different function

At this point, we will consider a separate function to solve. We will now
search for the root of

```julia
g(u) = exp(u) - 1.0e-15;
```




The root of this particular function is analytic and given by
`u = - 15 * log(10)`. Due to the nature of the function, it can be difficult to
numerically resolve the root.

Since we do not adjust the value of `p` here, we will just solve this same
function `N` times. As before, we start with `Roots.jl`,

```julia
function i!(out, uspan)
    for i in 1:N
        out[i] = find_zero(g, uspan)
    end
    return out
end

uspan = (-100.0, 0.0)

@btime i!(out, uspan)
println("Mean absolute error: $(mean(abs.(g.(out))))")
```

```
346.155 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30
```





So, how do the `BracketingNonlinearSolve.jl` methods fare?

```julia
g(u, p) = g(u)

function j!(out, uspan, alg)
    N = length(out)
    for i in 1:N
        prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(g), uspan)
        sol = solve(prob, alg)
        out[i] = sol.u
    end
    return out
end

for alg in (
        Alefeld, Bisection, Brent, Falsi,
        ITP, Muller, Ridder, ModAB,
    )
    println("Benchmark of $alg:")
    @btime j!($out, $uspan, $(alg()))
    println("Mean absolute error: $(mean(abs.(g.(out))))\n")
end
```

```
Benchmark of BracketingNonlinearSolve.Alefeld:
  337.928 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Bisection:
  71.732 ms (0 allocations: 0 bytes)
Mean absolute error: 3.4512664603419266e-29

Benchmark of BracketingNonlinearSolve.Brent:
  72.052 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Falsi:
  77.864 ms (0 allocations: 0 bytes)
Mean absolute error: 3.4512664603419266e-29

Benchmark of BracketingNonlinearSolve.ITP:
  131.603 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Muller:
  7.286 ms (0 allocations: 0 bytes)
Mean absolute error: 9.999998071250149e-16

Benchmark of BracketingNonlinearSolve.Ridder:
  88.675 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.ModAB:
  39.109 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30
```





Again, we see that the `BracketingNonlinearSolve.jl` root-finding algorithms are fast.
However, it is notable that some are able to resolve the root more accurately
than others. This is entirely to be expected as some of the algorithms, like
`Bisection`, bracket the root and thus will reliably converge to high accuracy.
Others, like `Muller`, are not bracketing methods, but can be extremely fast.

## Extended Test Suite with Challenging Functions

Now we'll test the algorithms on a comprehensive suite of challenging test functions
commonly used in the interval rootfinding literature. These functions exhibit various
difficulties such as multiple roots, nearly flat regions, discontinuities, and
extreme sensitivity.

```julia
using Statistics

# Define challenging test functions
test_functions = [
    # Function 1: Polynomial with multiple roots
    (
        name = "Wilkinson-like polynomial",
        f = (u, p) -> (u - 1) * (u - 2) * (u - 3) * (u - 4) * (u - 5) - p,
        interval = (0.5, 5.5),
        p = 0.05,
    ),

    # Function 2: Trigonometric with multiple roots
    (
        name = "sin(x) - 0.5x",
        f = (u, p) -> sin(u) - 0.5 * u - p,
        interval = (-10.0, 10.0),
        p = 0.3,
    ),

    # Function 3: Exponential function (sensitive near zero)
    (
        name = "exp(x) - 1 - x - x²/2",
        f = (u, p) -> exp(u) - 1 - u - u^2 / 2 - p,
        interval = (-2.0, 2.0),
        p = 0.005,
    ),

    # Function 4: Rational function with pole
    (
        name = "1/(x-0.5) - 2",
        f = (u, p) -> 1 / (u - 0.5) - 2 - p,
        interval = (0.6, 2.0),
        p = 0.05,
    ),

    # Function 5: Logarithmic function
    (
        name = "log(x) - x + 2",
        f = (u, p) -> log(u) - u + 2 - p,
        interval = (0.1, 3.0),
        p = 0.05,
    ),

    # Function 6: High oscillation function
    (
        name = "sin(20x) + sin(x) + x",
        f = (u, p) -> sin(20 * u) + sin(u) + u - p,
        interval = (-5.0, 5.0),
        p = 2.0,
    ),

    # Function 7: Function with very flat region
    (
        name = "x³ - 2x² + x",
        f = (u, p) -> u^3 - 2 * u^2 + u - p,
        interval = (-1.0, 2.0),
        p = 0.025,
    ),

    # Function 8: Bessel-like function
    (
        name = "x·sin(1/x) - 0.1",
        f = (u, p) -> u * sin(1 / u) - 0.1 - p,
        interval = (0.01, 1.0),
        p = 0.01,
    ),
]

# Add SimpleNonlinearSolve algorithms
using SimpleNonlinearSolve

# Combined algorithm list from both packages
all_algorithms = [
    (name = "Alefeld (BNS)", alg = () -> Alefeld(), package = "BracketingNonlinearSolve"),
    (name = "Bisection (BNS)", alg = () -> Bisection(), package = "BracketingNonlinearSolve"),
    (name = "Brent (BNS)", alg = () -> Brent(), package = "BracketingNonlinearSolve"),
    (name = "Falsi (BNS)", alg = () -> Falsi(), package = "BracketingNonlinearSolve"),
    (name = "ITP (BNS)", alg = () -> ITP(), package = "BracketingNonlinearSolve"),
    (name = "Ridder (BNS)", alg = () -> Ridder(), package = "BracketingNonlinearSolve"),
    (name = "ModAB (BNS)", alg = () -> ModAB(), package = "BracketingNonlinearSolve"),
    (
        name = "Bisection (SNS)", alg = () -> SimpleNonlinearSolve.Bisection(),
        package = "SimpleNonlinearSolve",
    ),
    (
        name = "Brent (SNS)", alg = () -> SimpleNonlinearSolve.Brent(),
        package = "SimpleNonlinearSolve",
    ),
    (
        name = "Falsi (SNS)", alg = () -> SimpleNonlinearSolve.Falsi(),
        package = "SimpleNonlinearSolve",
    ),
    (
        name = "Ridders (SNS)", alg = () -> SimpleNonlinearSolve.Ridder(),
        package = "SimpleNonlinearSolve",
    ),
]

# Benchmark function for testing all algorithms on a given function
function benchmark_function(test_func, N_samples = 10000)
    println("\\n=== Testing: $(test_func.name) ===")
    println("Interval: $(test_func.interval)")
    println("Parameter: $(test_func.p)")

    results = []

    # Test Roots.jl baseline
    try
        # Cache the function for Roots.jl
        roots_func = u -> test_func.f(u, test_func.p)

        # Warmup run to exclude compilation time
        find_zero(roots_func, test_func.interval)

        # Actual timing
        time_roots = @elapsed begin
            for i in 1:N_samples
                root = find_zero(roots_func, test_func.interval)
            end
        end

        # Calculate error using one solve
        final_root = find_zero(roots_func, test_func.interval)
        error_roots = abs(test_func.f(final_root, test_func.p))

        println("Roots.jl: $(round(time_roots * 1000, digits = 2)) ms, Error: $(round(error_roots, sigdigits = 3))")
        push!(
            results, (
                name = "Roots.jl", time = time_roots, error = error_roots, success = true,
            )
        )
    catch e
        println("Roots.jl: FAILED - $e")
        push!(results, (name = "Roots.jl", time = Inf, error = Inf, success = false))
    end

    # Test all algorithms
    for alg_info in all_algorithms
        try
            # Warmup run to exclude compilation time
            prob_warmup = IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(test_func.f),
                test_func.interval, test_func.p
            )
            solve(prob_warmup, alg_info.alg())

            # Actual timing
            time_taken = @elapsed begin
                for i in 1:N_samples
                    prob = IntervalNonlinearProblem{false}(
                        IntervalNonlinearFunction{false}(test_func.f),
                        test_func.interval, test_func.p
                    )
                    sol = solve(prob, alg_info.alg())
                end
            end

            # Calculate error using one solve
            prob_final = IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(test_func.f),
                test_func.interval, test_func.p
            )
            sol_final = solve(prob_final, alg_info.alg())
            error_val = abs(test_func.f(sol_final.u, test_func.p))

            println("$(alg_info.name): $(round(time_taken * 1000, digits = 2)) ms, Error: $(round(error_val, sigdigits = 3))")
            push!(
                results, (
                    name = alg_info.name, time = time_taken, error = error_val, success = true,
                )
            )
        catch e
            println("$(alg_info.name): FAILED - $e")
            push!(results, (name = alg_info.name, time = Inf, error = Inf, success = false))
        end
    end

    return results
end

# Run benchmarks on all test functions
all_results = []
for test_func in test_functions
    results = benchmark_function(test_func, 10000)  # Increased N since we're using fixed parameters
    push!(all_results, (func_name = test_func.name, results = results))
end
```

```
\n=== Testing: Wilkinson-like polynomial ===
Interval: (0.5, 5.5)
Parameter: 0.05
Roots.jl: 20.92 ms, Error: 1.87e-15
Alefeld (BNS): 12.49 ms, Error: 7.79e-15
Bisection (BNS): 5.55 ms, Error: 4.53e-12
Brent (BNS): 5.57 ms, Error: 5.55e-16
Falsi (BNS): 51.41 ms, Error: 2.79e-13
ITP (BNS): 5.93 ms, Error: 1.22e-15
Ridder (BNS): 6.4 ms, Error: 6.59e-16
ModAB (BNS): 4.6 ms, Error: 5.55e-16
Bisection (SNS): 5.38 ms, Error: 4.53e-12
Brent (SNS): 5.47 ms, Error: 5.55e-16
Falsi (SNS): 51.28 ms, Error: 2.79e-13
Ridders (SNS): 6.28 ms, Error: 6.59e-16
\n=== Testing: sin(x) - 0.5x ===
Interval: (-10.0, 10.0)
Parameter: 0.3
Roots.jl: 32.57 ms, Error: 5.55e-17
Alefeld (BNS): 28.34 ms, Error: 5.55e-17
Bisection (BNS): 10.05 ms, Error: 1.09e-13
Brent (BNS): 11.44 ms, Error: 1.86e-13
Falsi (BNS): 20.12 ms, Error: 9.63e-14
ITP (BNS): 8.3 ms, Error: 5.55e-17
Ridder (BNS): 7.67 ms, Error: 9.44e-16
ModAB (BNS): 5.75 ms, Error: 3.89e-16
Bisection (SNS): 10.15 ms, Error: 1.09e-13
Brent (SNS): 11.57 ms, Error: 1.86e-13
Falsi (SNS): 20.04 ms, Error: 9.63e-14
Ridders (SNS): 7.54 ms, Error: 9.44e-16
\n=== Testing: exp(x) - 1 - x - x²/2 ===
Interval: (-2.0, 2.0)
Parameter: 0.005
Roots.jl: 36.63 ms, Error: 5.12e-17
Alefeld (BNS): 60.84 ms, Error: 1.13e-17
Bisection (BNS): 10.51 ms, Error: 9.59e-15
Brent (BNS): 12.71 ms, Error: 7.64e-15
Falsi (BNS): 708.84 ms, Error: 2.98e-13
ITP (BNS): 8.87 ms, Error: 3.21e-17
Ridder (BNS): 8.56 ms, Error: 6.68e-17
ModAB (BNS): 6.6 ms, Error: 6.51e-17
Bisection (SNS): 9.65 ms, Error: 9.59e-15
Brent (SNS): 12.17 ms, Error: 7.64e-15
Falsi (SNS): 690.36 ms, Error: 2.98e-13
Ridders (SNS): 8.57 ms, Error: 6.68e-17
\n=== Testing: 1/(x-0.5) - 2 ===
Interval: (0.6, 2.0)
Parameter: 0.05
Roots.jl: 20.51 ms, Error: 2.64e-16
Alefeld (BNS): 7.94 ms, Error: 2.64e-16
Bisection (BNS): 4.69 ms, Error: 2.4e-13
Brent (BNS): 5.15 ms, Error: 2.64e-16
Falsi (BNS): 104.78 ms, Error: 4.65e-13
ITP (BNS): 6.75 ms, Error: 2.64e-16
Ridder (BNS): 4.38 ms, Error: 2.64e-16
ModAB (BNS): 3.01 ms, Error: 6.25e-16
Bisection (SNS): 4.15 ms, Error: 2.4e-13
Brent (SNS): 4.84 ms, Error: 2.64e-16
Falsi (SNS): 104.61 ms, Error: 4.65e-13
Ridders (SNS): 4.12 ms, Error: 2.64e-16
\n=== Testing: log(x) - x + 2 ===
Interval: (0.1, 3.0)
Parameter: 0.05
Roots.jl: 33.25 ms, Error: 4.16e-17
Alefeld (BNS): 35.91 ms, Error: 4.16e-17
Bisection (BNS): 10.68 ms, Error: 6.88e-13
Brent (BNS): 9.27 ms, Error: 4.16e-17
Falsi (BNS): 32.32 ms, Error: 1.01e-12
ITP (BNS): 9.15 ms, Error: 4.16e-17
Ridder (BNS): 8.81 ms, Error: 2.7e-14
ModAB (BNS): 6.49 ms, Error: 4.16e-17
Bisection (SNS): 10.93 ms, Error: 6.88e-13
Brent (SNS): 9.61 ms, Error: 4.16e-17
Falsi (SNS): 32.84 ms, Error: 1.01e-12
Ridders (SNS): 8.65 ms, Error: 2.7e-14
\n=== Testing: sin(20x) + sin(x) + x ===
Interval: (-5.0, 5.0)
Parameter: 2.0
Roots.jl: 42.62 ms, Error: 4.44e-16
Alefeld (BNS): 101.05 ms, Error: 8.88e-16
Bisection (BNS): 16.74 ms, Error: 1.73e-12
Brent (BNS): 12.64 ms, Error: 4.44e-16
Falsi (BNS): 24.89 ms, Error: 2.43e-13
ITP (BNS): 10.25 ms, Error: 2.66e-15
Ridder (BNS): 19.15 ms, Error: 3.11e-15
ModAB (BNS): 8.39 ms, Error: 0.0
Bisection (SNS): 16.93 ms, Error: 1.73e-12
Brent (SNS): 12.54 ms, Error: 4.44e-16
Falsi (SNS): 24.88 ms, Error: 2.43e-13
Ridders (SNS): 18.92 ms, Error: 3.11e-15
\n=== Testing: x³ - 2x² + x ===
Interval: (-1.0, 2.0)
Parameter: 0.025
Roots.jl: 23.26 ms, Error: 0.0
Alefeld (BNS): 9.72 ms, Error: 9.02e-17
Bisection (BNS): 4.1 ms, Error: 1.88e-14
Brent (BNS): 4.67 ms, Error: 9.02e-17
Falsi (BNS): 118.6 ms, Error: 2.98e-13
ITP (BNS): 5.74 ms, Error: 3.47e-18
Ridder (BNS): 4.29 ms, Error: 0.0
ModAB (BNS): 3.72 ms, Error: 0.0
Bisection (SNS): 4.53 ms, Error: 1.88e-14
Brent (SNS): 5.12 ms, Error: 9.02e-17
Falsi (SNS): 117.93 ms, Error: 2.98e-13
Ridders (SNS): 4.72 ms, Error: 0.0
\n=== Testing: x·sin(1/x) - 0.1 ===
Interval: (0.01, 1.0)
Parameter: 0.01
Roots.jl: 34.22 ms, Error: 8.67e-18
Alefeld (BNS): FAILED - DomainError(Inf, "sin(x) is only defined for finite
 x.")
Bisection (BNS): 10.54 ms, Error: 3.57e-13
Brent (BNS): 16.55 ms, Error: 5.34e-13
Falsi (BNS): 18.81 ms, Error: 5.3e-13
ITP (BNS): 8.03 ms, Error: 8.67e-18
Ridder (BNS): 7.12 ms, Error: 2.86e-16
ModAB (BNS): 7.19 ms, Error: 6.42e-17
Bisection (SNS): 10.96 ms, Error: 3.57e-13
Brent (SNS): 16.48 ms, Error: 5.34e-13
Falsi (SNS): 18.56 ms, Error: 5.3e-13
Ridders (SNS): 6.79 ms, Error: 2.86e-16
```





## Performance Summary

Let's create a summary table of the results:

```julia
using Printf

function print_summary_table(all_results)
    println("\\n" * "="^80)
    println("COMPREHENSIVE BENCHMARK SUMMARY")
    println("="^80)

    # Get all algorithm names
    alg_names = unique(
        [
            r.name for func_results in all_results
                for r in func_results.results
        ]
    )

    # Print header
    @printf "%-25s" "Function"
    for alg in alg_names
        @printf "%-15s" alg[1:min(14, length(alg))]
    end
    println()
    println("-"^(25 + 15 * length(alg_names)))

    # Print results for each function
    for func_result in all_results
        @printf "%-25s" func_result.func_name[1:min(24, length(func_result.func_name))]

        for alg in alg_names
            # Find result for this algorithm
            alg_result = findfirst(r -> r.name == alg, func_result.results)
            if alg_result !== nothing
                result = func_result.results[alg_result]
                if result.success && result.time < 1.0  # Reasonable time limit
                    @printf "%-15s" "$(round(result.time * 1000, digits = 1))ms"
                else
                    @printf "%-15s" "FAIL"
                end
            else
                @printf "%-15s" "N/A"
            end
        end
        println()
    end

    println("\\n" * "="^80)
    println("Notes:")
    println("- Times shown in milliseconds for 10000 function evaluations")
    println("- BNS = BracketingNonlinearSolve.jl, SNS = SimpleNonlinearSolve.jl")
    println("- FAIL indicates algorithm failed or took excessive time")
    println("- Compilation time excluded via warmup runs")
    return println("="^80)
end

print_summary_table(all_results)
```

```
\n=========================================================================
=======
COMPREHENSIVE BENCHMARK SUMMARY
===========================================================================
=====
Function                 Roots.jl       Alefeld (BNS)  Bisection (BNS Brent
 (BNS)    Falsi (BNS)    ITP (BNS)      Ridder (BNS)   ModAB (BNS)    Bisec
tion (SNS Brent (SNS)    Falsi (SNS)    Ridders (SNS)
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-------------------------------------------------------
Wilkinson-like polynomia 20.9ms         12.5ms         5.6ms          5.6ms
          51.4ms         5.9ms          6.4ms          4.6ms          5.4ms
          5.5ms          51.3ms         6.3ms
sin(x) - 0.5x            32.6ms         28.3ms         10.1ms         11.4m
s         20.1ms         8.3ms          7.7ms          5.8ms          10.1m
s         11.6ms         20.0ms         7.5ms
exp(x) - 1 - x - x²/     36.6ms         60.8ms         10.5ms         12.7m
s         708.8ms        8.9ms          8.6ms          6.6ms          9.6ms
          12.2ms         690.4ms        8.6ms
1/(x-0.5) - 2            20.5ms         7.9ms          4.7ms          5.2ms
          104.8ms        6.7ms          4.4ms          3.0ms          4.2ms
          4.8ms          104.6ms        4.1ms
log(x) - x + 2           33.3ms         35.9ms         10.7ms         9.3ms
          32.3ms         9.1ms          8.8ms          6.5ms          10.9m
s         9.6ms          32.8ms         8.6ms
sin(20x) + sin(x) + x    42.6ms         101.0ms        16.7ms         12.6m
s         24.9ms         10.3ms         19.2ms         8.4ms          16.9m
s         12.5ms         24.9ms         18.9ms
x³ - 2x² +               23.3ms         9.7ms          4.1ms          4.7ms
          118.6ms        5.7ms          4.3ms          3.7ms          4.5ms
          5.1ms          117.9ms        4.7ms
x·sin(1/x) - 0.          34.2ms         FAIL           10.5ms         16.5m
s         18.8ms         8.0ms          7.1ms          7.2ms          11.0m
s         16.5ms         18.6ms         6.8ms
\n=========================================================================
=======
Notes:
- Times shown in milliseconds for 10000 function evaluations
- BNS = BracketingNonlinearSolve.jl, SNS = SimpleNonlinearSolve.jl
- FAIL indicates algorithm failed or took excessive time
- Compilation time excluded via warmup runs
===========================================================================
=====
```





## Accuracy Analysis

Now let's examine the accuracy of each method:

```julia
function print_accuracy_table(all_results)
    println("\\n" * "="^80)
    println("ACCURACY ANALYSIS (Absolute Error)")
    println("="^80)

    alg_names = unique(
        [
            r.name for func_results in all_results
                for r in func_results.results
        ]
    )

    # Print header
    @printf "%-25s" "Function"
    for alg in alg_names
        @printf "%-15s" alg[1:min(14, length(alg))]
    end
    println()
    println("-"^(25 + 15 * length(alg_names)))

    # Print results for each function
    for func_result in all_results
        @printf "%-25s" func_result.func_name[1:min(24, length(func_result.func_name))]

        for alg in alg_names
            alg_result = findfirst(r -> r.name == alg, func_result.results)
            if alg_result !== nothing
                result = func_result.results[alg_result]
                if result.success && result.error < 1.0e10
                    @printf "%-15s" "$(round(result.error, sigdigits = 2))"
                else
                    @printf "%-15s" "FAIL"
                end
            else
                @printf "%-15s" "N/A"
            end
        end
        println()
    end

    return println("="^80)
end

print_accuracy_table(all_results)
```

```
\n=========================================================================
=======
ACCURACY ANALYSIS (Absolute Error)
===========================================================================
=====
Function                 Roots.jl       Alefeld (BNS)  Bisection (BNS Brent
 (BNS)    Falsi (BNS)    ITP (BNS)      Ridder (BNS)   ModAB (BNS)    Bisec
tion (SNS Brent (SNS)    Falsi (SNS)    Ridders (SNS)
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-------------------------------------------------------
Wilkinson-like polynomia 1.9e-15        7.8e-15        4.5e-12        5.6e-
16        2.8e-13        1.2e-15        6.6e-16        5.6e-16        4.5e-
12        5.6e-16        2.8e-13        6.6e-16
sin(x) - 0.5x            5.6e-17        5.6e-17        1.1e-13        1.9e-
13        9.6e-14        5.6e-17        9.4e-16        3.9e-16        1.1e-
13        1.9e-13        9.6e-14        9.4e-16
exp(x) - 1 - x - x²/     5.1e-17        1.1e-17        9.6e-15        7.6e-
15        3.0e-13        3.2e-17        6.7e-17        6.5e-17        9.6e-
15        7.6e-15        3.0e-13        6.7e-17
1/(x-0.5) - 2            2.6e-16        2.6e-16        2.4e-13        2.6e-
16        4.6e-13        2.6e-16        2.6e-16        6.2e-16        2.4e-
13        2.6e-16        4.6e-13        2.6e-16
log(x) - x + 2           4.2e-17        4.2e-17        6.9e-13        4.2e-
17        1.0e-12        4.2e-17        2.7e-14        4.2e-17        6.9e-
13        4.2e-17        1.0e-12        2.7e-14
sin(20x) + sin(x) + x    4.4e-16        8.9e-16        1.7e-12        4.4e-
16        2.4e-13        2.7e-15        3.1e-15        0.0            1.7e-
12        4.4e-16        2.4e-13        3.1e-15
x³ - 2x² +               0.0            9.0e-17        1.9e-14        9.0e-
17        3.0e-13        3.5e-18        0.0            0.0            1.9e-
14        9.0e-17        3.0e-13        0.0
x·sin(1/x) - 0.          8.7e-18        FAIL           3.6e-13        5.3e-
13        5.3e-13        8.7e-18        2.9e-16        6.4e-17        3.6e-
13        5.3e-13        5.3e-13        2.9e-16
===========================================================================
=====
```





## Algorithm Rankings

Finally, let's rank the algorithms by overall performance:

```julia
function rank_algorithms(all_results)
    println("\\n" * "="^60)
    println("ALGORITHM RANKINGS")
    println("="^60)

    # Calculate scores for each algorithm
    alg_scores = Dict()

    for func_result in all_results
        for result in func_result.results
            if !haskey(alg_scores, result.name)
                alg_scores[result.name] = Dict(:time_score => 0.0, :accuracy_score => 0.0, :success_count => 0)
            end

            if result.success
                alg_scores[result.name][:success_count] += 1
                # Lower time is better (inverse score)
                alg_scores[result.name][:time_score] += result.time < 1.0 ?
                    1.0 / result.time : 0.0
                # Lower error is better (inverse score)
                alg_scores[result.name][:accuracy_score] += result.error < 1.0e10 ?
                    1.0 / (result.error + 1.0e-15) :
                    0.0
            end
        end
    end

    # Normalize and combine scores
    total_functions = length(all_results)
    algorithm_rankings = []

    for (alg, scores) in alg_scores
        success_rate = scores[:success_count] / total_functions
        avg_speed_score = scores[:time_score] / total_functions
        avg_accuracy_score = scores[:accuracy_score] / total_functions

        # Combined score (weighted: 40% success rate, 30% speed, 30% accuracy)
        combined_score = 0.4 * success_rate + 0.3 * (avg_speed_score / 1000) +
            0.3 * (avg_accuracy_score / 1.0e12)

        push!(
            algorithm_rankings,
            (
                name = alg,
                success_rate = success_rate,
                speed_score = avg_speed_score,
                accuracy_score = avg_accuracy_score,
                combined_score = combined_score,
            )
        )
    end

    # Sort by combined score
    sort!(algorithm_rankings, by = x -> x.combined_score, rev = true)

    println("Rank | Algorithm          | Success Rate | Combined Score")
    println("-"^60)
    for (i, alg) in enumerate(algorithm_rankings)
        @printf "%-4d | %-18s | %-11.1f%% | %-12.3f\\n" i alg.name[1:min(18, length(alg.name))] (alg.success_rate * 100) alg.combined_score
    end

    println("="^60)
    return println("Note: Combined score weights success rate (40%), speed (30%), and accuracy (30%)")
end

rank_algorithms(all_results)
```

```
\n============================================================
ALGORITHM RANKINGS
============================================================
Rank | Algorithm          | Success Rate | Combined Score
------------------------------------------------------------
1    | ModAB (BNS)        | 100.0      % | 256.111     \n2    | Roots.jl
        | 100.0      % | 250.985     \n3    | ITP (BNS)          | 100.0
   % | 239.641     \n4    | Alefeld (BNS)      | 87.5       % | 197.174
 \n5    | Ridders (SNS)      | 100.0      % | 184.285     \n6    | Ridder (
BNS)       | 100.0      % | 184.285     \n7    | Brent (BNS)        | 100.0
      % | 155.207     \n8    | Brent (SNS)        | 100.0      % | 155.207
    \n9    | Bisection (SNS)    | 100.0      % | 6.556       \n10   | Bisec
tion (BNS)    | 100.0      % | 6.556       \n11   | Falsi (SNS)        | 10
0.0      % | 1.520       \n12   | Falsi (BNS)        | 100.0      % | 1.520
       \n============================================================
Note: Combined score weights success rate (40%), speed (30%), and accuracy
(30%)
```





## Conclusion

This extended benchmark suite demonstrates the performance and accuracy characteristics of interval rootfinding algorithms across a diverse set of challenging test functions. The test functions include:

 1. **Polynomial functions** with multiple roots
 2. **Trigonometric functions** with oscillatory behavior
 3. **Exponential functions** with high sensitivity
 4. **Rational functions** with singularities
 5. **Logarithmic functions** with domain restrictions
 6. **Highly oscillatory functions** testing robustness
 7. **Functions with flat regions** challenging convergence
 8. **Bessel-like functions** with complex behavior

The benchmark compares algorithms from both `BracketingNonlinearSolve.jl` and `SimpleNonlinearSolve.jl`, providing insights into:

  - **Robustness**: Which algorithms handle challenging functions
  - **Speed**: Computational efficiency across different problem types
  - **Accuracy**: Precision of the found roots
  - **Reliability**: Success rates across diverse test cases

This comprehensive evaluation helps users choose the most appropriate interval rootfinding algorithm for their specific applications.


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/IntervalNonlinearProblem","suite.jmd")
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
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/IntervalNonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.8.0
  [70df07ce] BracketingNonlinearSolve v1.12.6
  [f2b01f46] Roots v3.0.7
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [10745b16] Statistics v1.11.1
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
```

And the full manifest:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/IntervalNonlinearProblem/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [4fba245c] ArrayInterface v7.30.0
  [6e4b80f9] BenchmarkTools v1.8.0
  [70df07ce] BracketingNonlinearSolve v1.12.6
  [38540f10] CommonSolve v0.2.14
  [bbf7d656] CommonSubexpressions v0.3.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.8
  [8f4d0f93] Conda v1.10.3
  [187b0558] ConstructionBase v1.6.0
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [e2d170a0] DataValueInterfaces v1.0.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [ffbed154] DocStringExtensions v0.9.5
  [4e289a0a] EnumX v1.0.7
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [9aa1b823] FastClosures v0.3.2
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
  [f6369f11] ForwardDiff v1.4.5
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [46192b85] GPUArraysCore v0.2.0
  [d7ba0133] Git v1.5.0
⌅ [eafb193a] Highlights v0.5.3
  [7073ff75] IJulia v1.34.4
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [b964fa9f] LaTeXStrings v1.4.1
  [87fe0de2] LineSearch v0.1.16
  [2ab3a3ac] LogExpFunctions v1.0.1
  [e6f89c97] LoggingExtras v1.2.0
  [1914dd2f] MacroTools v0.5.16
  [bb5d69b7] MaybeInplace v0.1.8
  [ffc61752] Mustache v1.0.21
  [77ba4419] NaNMath v1.1.4
⌃ [be0214bd] NonlinearSolveBase v2.48.0
  [bac558e1] OrderedCollections v2.0.1
⌅ [69de0a69] Parsers v2.8.7
⌃ [d236fae5] PreallocationTools v1.7.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [08abe8d2] PrettyTables v3.4.8
  [3cdcf5f2] RecipesBase v1.3.4
  [731186ca] RecursiveArrayTools v4.5.1
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [9fe22ead] RespecializeParams v1.3.0
  [f2b01f46] Roots v3.0.7
  [7e49a35a] RuntimeGeneratedFunctions v0.5.25
⌃ [0bca4576] SciMLBase v3.49.2
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.18
  [a6db7da4] SciMLLogging v2.1.0
  [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
  [53ae85a6] SciMLStructures v1.10.5
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [276daf66] SpecialFunctions v2.9.0
  [1e83bf80] StaticArraysCore v1.4.4
  [10745b16] Statistics v1.11.1
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.5.0
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [a759f4b9] TimerOutputs v1.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [2e619515] Expat_jll v2.8.3+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.55.0+0
  [94ce4f54] Libiconv_jll v1.18.0+0
  [9bd350c2] OpenSSH_jll v10.5.1+0
  [458c3c95] OpenSSL_jll v3.5.8+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [a9144af2] libsodium_jll v1.0.21+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
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
  [9abbd945] Profile v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [f489334b] StyledStrings v1.11.0
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
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
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

