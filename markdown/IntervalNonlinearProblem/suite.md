---
author: "Fabian Gittins"
title: "Interval root-finding test suite"
---


In this benchmark, we will examine how the interval root-finding algorithms
provided in `NonlinearSolve.jl` and `SimpleNonlinearSolve.jl` fare against one another for a selection of
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
    out
end;
```




Now, we can run the benchmark for `Roots.jl`:

```julia
out = zeros(N)
uspan = (0.0, 2.0)

@btime g!(out, ps, uspan);
```

```
207.522 ms (0 allocations: 0 bytes)
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

## `NonlinearSolve.jl` algorithms

With the preliminaries out of the way, let's see how the `NonlinearSolve.jl`
solvers perform! We define a (non-allocating) function to benchmark,

```julia
using NonlinearSolve

function h!(out, ps, uspan, alg)
    for i in 1:N
        prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(f), uspan, ps[i])
        sol = solve(prob, alg)
        out[i] = sol.u
    end
    out
end;
```




and loop through the methods,

```julia
for alg in (Alefeld, NonlinearSolve.Bisection, Brent, Falsi,
            ITP, Muller, Ridder)
    println("Benchmark of $alg:")
    @btime h!($out, $ps, $uspan, $(alg()))
    println("Mean absolute error: $(mean(abs.(f.(out, ps))))\n")
end
```

```
Benchmark of BracketingNonlinearSolve.Alefeld:
  134.717 ms (0 allocations: 0 bytes)
Mean absolute error: 3.918693955483679e-17

Benchmark of BracketingNonlinearSolve.Bisection:
  96.996 ms (0 allocations: 0 bytes)
Mean absolute error: 1.3284481736999422e-13

Benchmark of BracketingNonlinearSolve.Brent:
  38.212 ms (0 allocations: 0 bytes)
Mean absolute error: 1.283331239427249e-13

Benchmark of BracketingNonlinearSolve.Falsi:
  92.344 ms (0 allocations: 0 bytes)
Mean absolute error: 2.2712276205532786e-12

Benchmark of BracketingNonlinearSolve.ITP:
  33.821 ms (0 allocations: 0 bytes)
Mean absolute error: 1.6533434955381716e-16

Benchmark of BracketingNonlinearSolve.Muller:
  20.445 ms (0 allocations: 0 bytes)
Mean absolute error: 2.0138978308733098e-14

Benchmark of BracketingNonlinearSolve.Ridder:
  39.593 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1236399759901633e-13
```





Although each method finds the roots with different accuracies, we can see that
all the `NonlinearSolve.jl` algorithms are performant and non-allocating.

## A different function

At this point, we will consider a separate function to solve. We will now
search for the root of

```julia
g(u) = exp(u) - 1e-15;
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
    out
end

uspan = (-100.0, 0.0)

@btime i!(out, uspan)
println("Mean absolute error: $(mean(abs.(g.(out))))")
```

```
178.252 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30
```





So, how do the `NonlinearSolve.jl` methods fare?

```julia
g(u, p) = g(u)

function j!(out, uspan, alg)
    N = length(out)
    for i in 1:N
        prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(g), uspan)
        sol = solve(prob, alg)
        out[i] = sol.u
    end
    out
end

for alg in (Alefeld, NonlinearSolve.Bisection, Brent, Falsi,
            ITP, Muller, Ridder)
    println("Benchmark of $alg:")
    @btime j!($out, $uspan, $(alg()))
    println("Mean absolute error: $(mean(abs.(g.(out))))\n")
end
```

```
Benchmark of BracketingNonlinearSolve.Alefeld:
  239.951 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Bisection:
  52.790 ms (0 allocations: 0 bytes)
Mean absolute error: 3.4512664603419266e-29

Benchmark of BracketingNonlinearSolve.Brent:
  51.722 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Falsi:
  42.983 ms (0 allocations: 0 bytes)
Mean absolute error: 0.9999999999998213

Benchmark of BracketingNonlinearSolve.ITP:
  109.903 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30

Benchmark of BracketingNonlinearSolve.Muller:
  4.933 ms (0 allocations: 0 bytes)
Mean absolute error: 9.999998071250149e-16

Benchmark of BracketingNonlinearSolve.Ridder:
  62.955 ms (0 allocations: 0 bytes)
Mean absolute error: 1.1832913578315177e-30
```





Again, we see that the `NonlinearSolve.jl` root-finding algorithms are fast.
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
    (name = "Wilkinson-like polynomial", 
     f = (u, p) -> (u - 1) * (u - 2) * (u - 3) * (u - 4) * (u - 5) - p,
     interval = (0.5, 5.5),
     p = 0.05),
     
    # Function 2: Trigonometric with multiple roots
    (name = "sin(x) - 0.5x",
     f = (u, p) -> sin(u) - 0.5*u - p,
     interval = (-10.0, 10.0), 
     p = 0.3),
     
    # Function 3: Exponential function (sensitive near zero)
    (name = "exp(x) - 1 - x - x²/2",
     f = (u, p) -> exp(u) - 1 - u - u^2/2 - p,
     interval = (-2.0, 2.0),
     p = 0.005),
     
    # Function 4: Rational function with pole
    (name = "1/(x-0.5) - 2",
     f = (u, p) -> 1/(u - 0.5) - 2 - p,
     interval = (0.6, 2.0),
     p = 0.05),
     
    # Function 5: Logarithmic function
    (name = "log(x) - x + 2",
     f = (u, p) -> log(u) - u + 2 - p,
     interval = (0.1, 3.0),
     p = 0.05),
     
    # Function 6: High oscillation function
    (name = "sin(20x) + 0.1x",
     f = (u, p) -> sin(20*u) + 0.1*u - p,
     interval = (-5.0, 5.0),
     p = 0.1),
     
    # Function 7: Function with very flat region
    (name = "x³ - 2x² + x",
     f = (u, p) -> u^3 - 2*u^2 + u - p,
     interval = (-1.0, 2.0),
     p = 0.025),
     
    # Function 8: Bessel-like function
    (name = "x·sin(1/x) - 0.1",
     f = (u, p) -> u * sin(1/u) - 0.1 - p,
     interval = (0.01, 1.0),
     p = 0.01),
]

# Add SimpleNonlinearSolve algorithms  
using SimpleNonlinearSolve

# Combined algorithm list from both packages
all_algorithms = [
    (name = "Alefeld (BNS)", alg = () -> Alefeld(), package = "BracketingNonlinearSolve"),
    (name = "Bisection (BNS)", alg = () -> NonlinearSolve.Bisection(), package = "BracketingNonlinearSolve"), 
    (name = "Brent (BNS)", alg = () -> Brent(), package = "BracketingNonlinearSolve"),
    (name = "Falsi (BNS)", alg = () -> Falsi(), package = "BracketingNonlinearSolve"),
    (name = "ITP (BNS)", alg = () -> ITP(), package = "BracketingNonlinearSolve"),
    (name = "Ridder (BNS)", alg = () -> Ridder(), package = "BracketingNonlinearSolve"),
    (name = "Bisection (SNS)", alg = () -> SimpleNonlinearSolve.Bisection(), package = "SimpleNonlinearSolve"),
    (name = "Brent (SNS)", alg = () -> SimpleNonlinearSolve.Brent(), package = "SimpleNonlinearSolve"),
    (name = "Falsi (SNS)", alg = () -> SimpleNonlinearSolve.Falsi(), package = "SimpleNonlinearSolve"),
    (name = "Ridders (SNS)", alg = () -> SimpleNonlinearSolve.Ridders(), package = "SimpleNonlinearSolve")
]

# Benchmark function for testing all algorithms on a given function
function benchmark_function(test_func, N_samples=10000)
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
        
        println("Roots.jl: $(round(time_roots*1000, digits=2)) ms, Error: $(round(error_roots, sigdigits=3))")
        push!(results, (name="Roots.jl", time=time_roots, error=error_roots, success=true))
    catch e
        println("Roots.jl: FAILED - $e")
        push!(results, (name="Roots.jl", time=Inf, error=Inf, success=false))
    end
    
    # Test all algorithms
    for alg_info in all_algorithms
        try
            # Warmup run to exclude compilation time
            prob_warmup = IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(test_func.f), 
                test_func.interval, test_func.p)
            solve(prob_warmup, alg_info.alg())
            
            # Actual timing
            time_taken = @elapsed begin
                for i in 1:N_samples
                    prob = IntervalNonlinearProblem{false}(
                        IntervalNonlinearFunction{false}(test_func.f), 
                        test_func.interval, test_func.p)
                    sol = solve(prob, alg_info.alg())
                end
            end
            
            # Calculate error using one solve
            prob_final = IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(test_func.f), 
                test_func.interval, test_func.p)
            sol_final = solve(prob_final, alg_info.alg())
            error_val = abs(test_func.f(sol_final.u, test_func.p))
            
            println("$(alg_info.name): $(round(time_taken*1000, digits=2)) ms, Error: $(round(error_val, sigdigits=3))")
            push!(results, (name=alg_info.name, time=time_taken, error=error_val, success=true))
        catch e
            println("$(alg_info.name): FAILED - $e")
            push!(results, (name=alg_info.name, time=Inf, error=Inf, success=false))
        end
    end
    
    return results
end

# Run benchmarks on all test functions
all_results = []
for test_func in test_functions
    results = benchmark_function(test_func, 10000)  # Increased N since we're using fixed parameters
    push!(all_results, (func_name=test_func.name, results=results))
end
```

```
\n=== Testing: Wilkinson-like polynomial ===
Interval: (0.5, 5.5)
Parameter: 0.05
Roots.jl: 12.54 ms, Error: 1.87e-15
Alefeld (BNS): 8.68 ms, Error: 7.79e-15
Bisection (BNS): 3.05 ms, Error: 4.53e-12
Brent (BNS): 3.27 ms, Error: 5.55e-16
Falsi (BNS): 45.87 ms, Error: 2.79e-13
ITP (BNS): 3.57 ms, Error: 1.22e-15
Ridder (BNS): 3.71 ms, Error: 6.59e-16
Bisection (SNS): 3.14 ms, Error: 4.53e-12
Brent (SNS): 3.11 ms, Error: 5.55e-16
Falsi (SNS): 37.08 ms, Error: 2.79e-13
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: sin(x) - 0.5x ===
Interval: (-10.0, 10.0)
Parameter: 0.3
Roots.jl: 17.75 ms, Error: 5.55e-17
Alefeld (BNS): 22.0 ms, Error: 5.55e-17
Bisection (BNS): 6.83 ms, Error: 1.09e-13
Brent (BNS): 7.66 ms, Error: 1.86e-13
Falsi (BNS): 17.01 ms, Error: 9.63e-14
ITP (BNS): 5.15 ms, Error: 5.55e-17
Ridder (BNS): 4.99 ms, Error: 9.44e-16
Bisection (SNS): 6.94 ms, Error: 1.09e-13
Brent (SNS): 7.65 ms, Error: 1.86e-13
Falsi (SNS): 13.32 ms, Error: 9.63e-14
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: exp(x) - 1 - x - x²/2 ===
Interval: (-2.0, 2.0)
Parameter: 0.005
Roots.jl: 18.88 ms, Error: 5.12e-17
Alefeld (BNS): 39.8 ms, Error: 1.13e-17
Bisection (BNS): 6.48 ms, Error: 9.59e-15
Brent (BNS): 8.49 ms, Error: 7.64e-15
Falsi (BNS): 555.11 ms, Error: 2.98e-13
ITP (BNS): 6.13 ms, Error: 3.21e-17
Ridder (BNS): 5.66 ms, Error: 6.68e-17
Bisection (SNS): 6.76 ms, Error: 9.59e-15
Brent (SNS): 8.73 ms, Error: 7.64e-15
Falsi (SNS): 555.03 ms, Error: 2.98e-13
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: 1/(x-0.5) - 2 ===
Interval: (0.6, 2.0)
Parameter: 0.05
Roots.jl: 12.85 ms, Error: 2.64e-16
Alefeld (BNS): 7.85 ms, Error: 2.64e-16
Bisection (BNS): 3.35 ms, Error: 2.4e-13
Brent (BNS): 3.8 ms, Error: 2.64e-16
Falsi (BNS): 77.53 ms, Error: 4.65e-13
ITP (BNS): 5.61 ms, Error: 2.64e-16
Ridder (BNS): 2.92 ms, Error: 2.64e-16
Bisection (SNS): 2.89 ms, Error: 2.4e-13
Brent (SNS): 3.38 ms, Error: 2.64e-16
Falsi (SNS): 77.22 ms, Error: 4.65e-13
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: log(x) - x + 2 ===
Interval: (0.1, 3.0)
Parameter: 0.05
Roots.jl: 17.7 ms, Error: 4.16e-17
Alefeld (BNS): 27.5 ms, Error: 4.16e-17
Bisection (BNS): 7.54 ms, Error: 6.88e-13
Brent (BNS): 6.47 ms, Error: 4.16e-17
Falsi (BNS): 23.77 ms, Error: 1.01e-12
ITP (BNS): 6.58 ms, Error: 4.16e-17
Ridder (BNS): 5.98 ms, Error: 2.7e-14
Bisection (SNS): 7.79 ms, Error: 6.88e-13
Brent (SNS): 6.73 ms, Error: 4.16e-17
Falsi (SNS): 23.99 ms, Error: 1.01e-12
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: sin(20x) + 0.1x ===
Interval: (-5.0, 5.0)
Parameter: 0.1
Roots.jl: FAILED - ArgumentError("The interval [a,b] is not a bracketing in
terval.\nYou need f(a) and f(b) to have different signs (f(a) * f(b) < 0).\
nConsider a different bracket or try fzero(f, c) with an initial guess c.\n
\n")
Alefeld (BNS): FAILED - UndefVarError(:ā)
Bisection (BNS): 345.68 ms, Error: 0.0936
Brent (BNS): 363.86 ms, Error: 0.0936
Falsi (BNS): 343.79 ms, Error: 0.0936
ITP (BNS): 343.63 ms, Error: 0.0936
Ridder (BNS): 358.86 ms, Error: 0.0936
Bisection (SNS): 355.33 ms, Error: 0.0936
Brent (SNS): 345.05 ms, Error: 0.0936
Falsi (SNS): 369.99 ms, Error: 0.0936
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: x³ - 2x² + x ===
Interval: (-1.0, 2.0)
Parameter: 0.025
Roots.jl: 13.1 ms, Error: 0.0
Alefeld (BNS): 7.31 ms, Error: 9.02e-17
Bisection (BNS): 2.77 ms, Error: 1.88e-14
Brent (BNS): 3.21 ms, Error: 9.02e-17
Falsi (BNS): 87.68 ms, Error: 2.98e-13
ITP (BNS): 4.04 ms, Error: 3.47e-18
Ridder (BNS): 2.86 ms, Error: 1.63e-16
Bisection (SNS): 2.82 ms, Error: 1.88e-14
Brent (SNS): 3.39 ms, Error: 9.02e-17
Falsi (SNS): 87.46 ms, Error: 2.98e-13
Ridders (SNS): FAILED - UndefVarError(:Ridders)
\n=== Testing: x·sin(1/x) - 0.1 ===
Interval: (0.01, 1.0)
Parameter: 0.01
Roots.jl: 18.9 ms, Error: 8.67e-18
Alefeld (BNS): FAILED - DomainError(Inf, "sin(x) is only defined for finite
 x.")
Bisection (BNS): 8.28 ms, Error: 3.57e-13
Brent (BNS): 11.88 ms, Error: 5.34e-13
Falsi (BNS): 13.03 ms, Error: 5.3e-13
ITP (BNS): 5.18 ms, Error: 8.67e-18
Ridder (BNS): 4.76 ms, Error: 2.86e-16
Bisection (SNS): 8.51 ms, Error: 3.57e-13
Brent (SNS): 11.83 ms, Error: 5.34e-13
Falsi (SNS): 13.08 ms, Error: 5.3e-13
Ridders (SNS): FAILED - UndefVarError(:Ridders)
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
    alg_names = unique([r.name for func_results in all_results for r in func_results.results])
    
    # Print header
    @printf "%-25s" "Function"
    for alg in alg_names
        @printf "%-15s" alg[1:min(14, length(alg))]
    end
    println()
    println("-"^(25 + 15*length(alg_names)))
    
    # Print results for each function
    for func_result in all_results
        @printf "%-25s" func_result.func_name[1:min(24, length(func_result.func_name))]
        
        for alg in alg_names
            # Find result for this algorithm
            alg_result = findfirst(r -> r.name == alg, func_result.results)
            if alg_result !== nothing
                result = func_result.results[alg_result]
                if result.success && result.time < 1.0  # Reasonable time limit
                    @printf "%-15s" "$(round(result.time*1000, digits=1))ms"
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
    println("="^80)
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
 (BNS)    Falsi (BNS)    ITP (BNS)      Ridder (BNS)   Bisection (SNS Brent
 (SNS)    Falsi (SNS)    Ridders (SNS)  
---------------------------------------------------------------------------
---------------------------------------------------------------------------
----------------------------------------
Wilkinson-like polynomia 12.5ms         8.7ms          3.1ms          3.3ms
          45.9ms         3.6ms          3.7ms          3.1ms          3.1ms
          37.1ms         FAIL           
sin(x) - 0.5x            17.8ms         22.0ms         6.8ms          7.7ms
          17.0ms         5.2ms          5.0ms          6.9ms          7.6ms
          13.3ms         FAIL           
exp(x) - 1 - x - x²/     18.9ms         39.8ms         6.5ms          8.5ms
          555.1ms        6.1ms          5.7ms          6.8ms          8.7ms
          555.0ms        FAIL           
1/(x-0.5) - 2            12.9ms         7.8ms          3.3ms          3.8ms
          77.5ms         5.6ms          2.9ms          2.9ms          3.4ms
          77.2ms         FAIL           
log(x) - x + 2           17.7ms         27.5ms         7.5ms          6.5ms
          23.8ms         6.6ms          6.0ms          7.8ms          6.7ms
          24.0ms         FAIL           
sin(20x) + 0.1x          FAIL           FAIL           345.7ms        363.9
ms        343.8ms        343.6ms        358.9ms        355.3ms        345.1
ms        370.0ms        FAIL           
x³ - 2x² +               13.1ms         7.3ms          2.8ms          3.2ms
          87.7ms         4.0ms          2.9ms          2.8ms          3.4ms
          87.5ms         FAIL           
x·sin(1/x) - 0.          18.9ms         FAIL           8.3ms          11.9m
s         13.0ms         5.2ms          4.8ms          8.5ms          11.8m
s         13.1ms         FAIL           
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
    
    alg_names = unique([r.name for func_results in all_results for r in func_results.results])
    
    # Print header
    @printf "%-25s" "Function"
    for alg in alg_names
        @printf "%-15s" alg[1:min(14, length(alg))]
    end
    println()
    println("-"^(25 + 15*length(alg_names)))
    
    # Print results for each function
    for func_result in all_results
        @printf "%-25s" func_result.func_name[1:min(24, length(func_result.func_name))]
        
        for alg in alg_names
            alg_result = findfirst(r -> r.name == alg, func_result.results)
            if alg_result !== nothing
                result = func_result.results[alg_result]
                if result.success && result.error < 1e10
                    @printf "%-15s" "$(round(result.error, sigdigits=2))"
                else
                    @printf "%-15s" "FAIL"
                end
            else
                @printf "%-15s" "N/A"
            end
        end
        println()
    end
    
    println("="^80)
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
 (BNS)    Falsi (BNS)    ITP (BNS)      Ridder (BNS)   Bisection (SNS Brent
 (SNS)    Falsi (SNS)    Ridders (SNS)  
---------------------------------------------------------------------------
---------------------------------------------------------------------------
----------------------------------------
Wilkinson-like polynomia 1.9e-15        7.8e-15        4.5e-12        5.6e-
16        2.8e-13        1.2e-15        6.6e-16        4.5e-12        5.6e-
16        2.8e-13        FAIL           
sin(x) - 0.5x            5.6e-17        5.6e-17        1.1e-13        1.9e-
13        9.6e-14        5.6e-17        9.4e-16        1.1e-13        1.9e-
13        9.6e-14        FAIL           
exp(x) - 1 - x - x²/     5.1e-17        1.1e-17        9.6e-15        7.6e-
15        3.0e-13        3.2e-17        6.7e-17        9.6e-15        7.6e-
15        3.0e-13        FAIL           
1/(x-0.5) - 2            2.6e-16        2.6e-16        2.4e-13        2.6e-
16        4.6e-13        2.6e-16        2.6e-16        2.4e-13        2.6e-
16        4.6e-13        FAIL           
log(x) - x + 2           4.2e-17        4.2e-17        6.9e-13        4.2e-
17        1.0e-12        4.2e-17        2.7e-14        6.9e-13        4.2e-
17        1.0e-12        FAIL           
sin(20x) + 0.1x          FAIL           FAIL           0.094          0.094
          0.094          0.094          0.094          0.094          0.094
          0.094          FAIL           
x³ - 2x² +               0.0            9.0e-17        1.9e-14        9.0e-
17        3.0e-13        3.5e-18        1.6e-16        1.9e-14        9.0e-
17        3.0e-13        FAIL           
x·sin(1/x) - 0.          8.7e-18        FAIL           3.6e-13        5.3e-
13        5.3e-13        8.7e-18        2.9e-16        3.6e-13        5.3e-
13        5.3e-13        FAIL           
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
                alg_scores[result.name][:time_score] += result.time < 1.0 ? 1.0 / result.time : 0.0
                # Lower error is better (inverse score) 
                alg_scores[result.name][:accuracy_score] += result.error < 1e10 ? 1.0 / (result.error + 1e-15) : 0.0
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
        combined_score = 0.4 * success_rate + 0.3 * (avg_speed_score / 1000) + 0.3 * (avg_accuracy_score / 1e12)
        
        push!(algorithm_rankings, (
            name = alg,
            success_rate = success_rate,
            speed_score = avg_speed_score,
            accuracy_score = avg_accuracy_score,
            combined_score = combined_score
        ))
    end
    
    # Sort by combined score
    sort!(algorithm_rankings, by = x -> x.combined_score, rev = true)
    
    println("Rank | Algorithm          | Success Rate | Combined Score")
    println("-"^60)
    for (i, alg) in enumerate(algorithm_rankings)
        @printf "%-4d | %-18s | %-11.1f%% | %-12.3f\\n" i alg.name[1:min(18, length(alg.name))] (alg.success_rate*100) alg.combined_score
    end
    
    println("="^60)
    println("Note: Combined score weights success rate (40%), speed (30%), and accuracy (30%)")
end

rank_algorithms(all_results)
```

```
\n============================================================
ALGORITHM RANKINGS
============================================================
Rank | Algorithm          | Success Rate | Combined Score
------------------------------------------------------------
1    | ITP (BNS)          | 100.0      % | 229.421     \n2    | Roots.jl   
        | 87.5       % | 224.973     \n3    | Alefeld (BNS)      | 75.0    
   % | 177.267     \n4    | Ridder (BNS)       | 100.0      % | 169.921    
 \n5    | Brent (SNS)        | 100.0      % | 129.253     \n6    | Brent (B
NS)        | 100.0      % | 129.253     \n7    | Bisection (SNS)    | 100.0
      % | 6.552       \n8    | Bisection (BNS)    | 100.0      % | 6.551   
    \n9    | Falsi (SNS)        | 100.0      % | 1.368       \n10   | Falsi
 (BNS)        | 100.0      % | 1.367       \n11   | Ridders (SNS)      | 0.
0        % | 0.000       \n================================================
============
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
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/IntervalNonlinearProblem/Project.toml`
  [6e4b80f9] BenchmarkTools v1.6.0
⌃ [8913a72c] NonlinearSolve v4.8.0
⌃ [f2b01f46] Roots v2.2.7
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
⌃ [727e6d20] SimpleNonlinearSolve v2.3.0
  [de0858da] Printf
  [9a3f8284] Random
  [10745b16] Statistics v1.10.0
Info Packages marked with ⌃ have new versions available and may be upgradable.
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/IntervalNonlinearProblem/Manifest.toml`
⌃ [47edcb42] ADTypes v1.14.0
  [7d9f7c33] Accessors v0.1.42
  [79e6a3ab] Adapt v4.3.0
⌃ [4fba245c] ArrayInterface v7.18.0
  [4c555306] ArrayLayouts v1.11.1
  [6e4b80f9] BenchmarkTools v1.6.0
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [70df07ce] BracketingNonlinearSolve v1.2.0
  [2a0fbf3d] CPUSummary v0.2.6
  [336ed68f] CSV v0.10.15
⌃ [d360d2e6] ChainRulesCore v1.25.1
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [944b1d66] CodecZlib v0.7.8
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
⌃ [34da2185] Compat v4.16.0
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [8f4d0f93] Conda v1.10.2
⌃ [187b0558] ConstructionBase v1.5.8
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.7.0
  [864edb3b] DataStructures v0.18.22
  [e2d170a0] DataValueInterfaces v1.0.0
⌃ [2b5f629d] DiffEqBase v6.170.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌅ [a0c0ee7d] DifferentiationInterface v0.6.52
⌃ [ffbed154] DocStringExtensions v0.9.4
  [4e289a0a] EnumX v1.0.5
⌃ [f151be2c] EnzymeCore v0.8.8
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
⌃ [a4df4552] FastPower v1.1.2
  [48062228] FilePathsBase v0.9.24
  [1a297f60] FillArrays v1.13.0
  [6a86dc24] FiniteDiff v2.27.0
  [f6369f11] ForwardDiff v1.0.1
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.2.0
⌃ [d7ba0133] Git v1.3.1
  [eafb193a] Highlights v0.5.3
⌃ [7073ff75] IJulia v1.27.0
  [615f187c] IfElse v0.1.1
⌃ [842dd82b] InlineStrings v1.4.3
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.4
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.7.0
  [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
  [ba0b0d4f] Krylov v0.10.1
  [b964fa9f] LaTeXStrings v1.4.0
  [10f19ff3] LayoutPointers v0.1.17
  [5078a376] LazyArrays v2.6.1
⌃ [9c8b4983] LightXML v0.9.1
  [87fe0de2] LineSearch v0.1.4
⌃ [7ed4a6bd] LinearSolve v3.9.0
  [2ab3a3ac] LogExpFunctions v0.3.29
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [e1d29d7a] Missings v1.2.0
⌃ [2e0e35c7] Moshi v0.3.5
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [ffc61752] Mustache v1.0.20
  [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.8.0
⌃ [be0214bd] NonlinearSolveBase v1.6.0
⌃ [5959db7a] NonlinearSolveFirstOrder v1.4.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.3.0
  [26075421] NonlinearSolveSpectralMethods v1.2.0
  [0f4fe800] OMJulia v0.3.2
⌃ [bac558e1] OrderedCollections v1.8.0
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.3
⌃ [f517fe37] Polyester v0.7.16
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [08abe8d2] PrettyTables v2.4.0
  [3cdcf5f2] RecipesBase v1.3.4
⌃ [731186ca] RecursiveArrayTools v3.33.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [f2b01f46] Roots v2.2.7
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.14
  [94e857df] SIMDTypes v0.1.0
⌃ [0bca4576] SciMLBase v2.86.2
  [31c91b34] SciMLBenchmarks v0.1.3 `../..`
⌃ [19f34311] SciMLJacobianOperators v0.1.3
⌅ [c0aeaf25] SciMLOperators v0.3.13
  [53ae85a6] SciMLStructures v1.7.0
⌃ [6c6a2e73] Scratch v1.2.1
  [91c51154] SentinelArrays v1.4.8
  [efcf1570] Setfield v1.1.2
⌃ [727e6d20] SimpleNonlinearSolve v2.3.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.1
⌃ [0a514795] SparseMatrixColorings v0.4.19
  [276daf66] SpecialFunctions v2.5.1
  [aedffcd0] Static v1.2.0
  [0d7ed370] StaticArrayInterface v1.8.0
  [1e83bf80] StaticArraysCore v1.4.3
  [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.4.1
⌃ [2efcf032] SymbolicIndexingInterface v0.3.40
  [3783bdb8] TableTraits v1.0.1
⌃ [bd369af6] Tables v1.12.0
⌃ [8290d209] ThreadingUtilities v0.5.3
⌃ [a759f4b9] TimerOutputs v0.5.28
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
  [3a884ed6] UnPack v1.0.2
  [81def892] VersionParsing v1.3.0
  [ea10d353] WeakRefStrings v1.4.2
  [44d3d7a6] Weave v0.10.12
  [76eceee3] WorkerUtilities v1.6.1
⌃ [ddb6d928] YAML v0.4.13
⌃ [c2297ded] ZMQ v1.4.0
  [2e619515] Expat_jll v2.6.5+0
⌃ [f8c6e375] Git_jll v2.49.0+0
  [1d5cc7b8] IntelOpenMP_jll v2025.0.4+0
  [94ce4f54] Libiconv_jll v1.18.0+0
  [856f044c] MKL_jll v2025.0.1+1
⌃ [458c3c95] OpenSSL_jll v3.5.0+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
⌅ [02c8fc9c] XML2_jll v2.13.6+1
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [a9144af2] libsodium_jll v1.0.21+0
  [1317d2d5] oneTBB_jll v2022.0.0+0
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
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.1+4
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

