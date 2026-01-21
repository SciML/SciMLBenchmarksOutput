---
author: "Alonso M. Cisneros"
title: "CUTEst Quadratic Programming with Linear Constraints Benchmarks"
---


# Introduction

CUTEst, the Constraind and Unconstrained Testing Environment is, as the name suggests is a
collection of around 1500 problems for general nonlinear optimization used to test
optimization routines. The wrapper
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) provides convenient access
to the problem collection, which we can leverage to test the optimizers made available by
Optimization.jl.
```julia
using Optimization
using OptimizationNLPModels
using CUTEst
using OptimizationOptimJL
using OptimizationOptimisers
using Ipopt
using OptimizationMOI
using OptimizationMOI: MOI as MOI
using DataFrames
using Plots
using StatsPlots
using StatsBase: countmap

optimizers = [
    ("Ipopt", MOI.OptimizerWithAttributes(Ipopt.Optimizer,
        "max_iter" => 5000,
        "tol" => 1e-6,
        "print_level" => 5)),
]

function get_stats(sol, optimizer_name)
    # Robustly get solve_time, even if stats or time is missing
    solve_time = try
        hasfield(typeof(sol), :stats) && hasfield(typeof(sol.stats), :time) ? getfield(sol.stats, :time) : NaN
    catch
        NaN
    end
    return (length(sol.u), solve_time, optimizer_name, Symbol(sol.retcode))
end

function run_benchmarks(problems, optimizers; chunk_size=1)
    problem = String[]
    n_vars = Int64[]
    secs = Float64[]
    solver = String[]
    retcode = Symbol[]
    optz = length(optimizers)
    n = length(problems)
    @info "Processing $(n) problems with $(optz) optimizers in chunks of $(chunk_size)"
    broadcast(c -> sizehint!(c, optz * n), [problem, n_vars, secs, solver, retcode])
    for chunk_start in 1:chunk_size:n
        chunk_end = min(chunk_start + chunk_size - 1, n)
        chunk_problems = problems[chunk_start:chunk_end]
        @info "Processing chunk $(div(chunk_start-1, chunk_size)+1)/$(div(n-1, chunk_size)+1): problems $(chunk_start)-$(chunk_end)"
        for (idx, prob_name) in enumerate(chunk_problems)
            current_problem = chunk_start + idx - 1
            @info "Problem $(current_problem)/$(n): $(prob_name)"
            nlp_prob = nothing
            try
                nlp_prob = CUTEstModel(prob_name)
                if nlp_prob.meta.nvar > 10000
                    @info "  Skipping $(prob_name) (too large: $(nlp_prob.meta.nvar) variables)"
                    finalize(nlp_prob)
                    continue
                end
                prob = OptimizationNLPModels.OptimizationProblem(nlp_prob, Optimization.AutoFiniteDiff())
                for (optimizer_name, optimizer) in optimizers
                    try
                        sol = solve(prob, optimizer; maxiters = 1000, maxtime = 30.0)
                        @info "✓ Solved $(prob_name) with $(optimizer_name) - Status: $(sol.retcode)"
                        vars, time, alg, code = get_stats(sol, optimizer_name)
                        push!(problem, prob_name)
                        push!(n_vars, vars)
                        push!(secs, time)
                        push!(solver, alg)
                        push!(retcode, code)
                    catch e
                        push!(problem, prob_name)
                        push!(n_vars, nlp_prob !== nothing ? nlp_prob.meta.nvar : -1)
                        push!(secs, NaN)
                        push!(solver, optimizer_name)
                        push!(retcode, :FAILED)
                    end
                end
            catch e
                for (optimizer_name, optimizer) in optimizers
                    push!(problem, prob_name)
                    push!(n_vars, -1)
                    push!(secs, NaN)
                    push!(solver, optimizer_name)
                    push!(retcode, :LOAD_FAILED)
                end
            finally
                if nlp_prob !== nothing
                    try
                        finalize(nlp_prob)
                    catch e
                    end
                end
            end
        end
        GC.gc()
        @info "Completed chunk, memory usage cleaned up"
    end
    return DataFrame(problem = problem, n_vars = n_vars, secs = secs, solver = solver, retcode = retcode)
end
```

```
run_benchmarks (generic function with 1 method)
```




```

# Benchmarks

We will be testing the [Ipopt]() and the [LBFGS]() optimizers on these classes of
problems.


# Quadratic programs with linear constraints

Lastly, we examine the problems with a quadratic objective function and only linear
constraints. There are 252 such problems in the suite.

```julia

# Select a moderate subset of quadratic problems for a realistic mix of successes and failures
quad_problems = CUTEst.select_sif_problems(objtype="quadratic", contype="linear")
@info "Testing $(length(quad_problems)) quadratic problems with linear constraints"
quad_problems = quad_problems[1:min(30, length(quad_problems))]
 # Skip HIER13, BLOWEYA, LUKVLE8, PATTERNNE, READING2, NINENEW, READING6, DITTERT, CVXQP2, and MSS1 if present
quad_problems = filter(p -> !(lowercase(p) in ["hier13", "bloweya", "s268", "stcqp1", "cvxqp3", "avgasb", "lukvle8", "sosqp2", "patternne", "reading2", "ninenew", "reading6", "dittert", "liswet9", "cleuven4", "cvxqp2", "mss1", "mpc2", "cmpc10", "cmpc3"]), quad_problems)
@info "Testing $(length(quad_problems)) quadratic problems with linear constraints (subset)"



# Harmonized analysis block with robust error handling and chunked processing
function run_quadratic_benchmarks(problems, optimizers; chunk_size=3)
    problem = String[]
    n_vars = Int64[]
    secs = Float64[]
    solver = String[]
    retcode = Symbol[]
    optz = length(optimizers)
    n = length(problems)
    @info "Processing $(n) quadratic problems with $(optz) optimizers in chunks of $(chunk_size)"
    broadcast(c -> sizehint!(c, optz * n), [problem, n_vars, secs, solver, retcode])
    for chunk_start in 1:chunk_size:n
        chunk_end = min(chunk_start + chunk_size - 1, n)
        chunk_problems = problems[chunk_start:chunk_end]
        @info "Processing chunk $(div(chunk_start-1, chunk_size)+1)/$(div(n-1, chunk_size)+1): problems $(chunk_start)-$(chunk_end)"
        for (idx, prob_name) in enumerate(chunk_problems)
            current_problem = chunk_start + idx - 1
            @info "Problem $(current_problem)/$(n): $(prob_name)"
            nlp_prob = nothing
            try
                nlp_prob = CUTEstModel(prob_name)
                if nlp_prob.meta.nvar > 10000
                    @info "  Skipping $(prob_name) (too large: $(nlp_prob.meta.nvar) variables)"
                    finalize(nlp_prob)
                    continue
                end
                prob = OptimizationNLPModels.OptimizationProblem(nlp_prob, Optimization.AutoFiniteDiff())
                for (optimizer_name, optimizer) in optimizers
                    try
                        sol = solve(prob, optimizer; maxiters = 1000, maxtime = 30.0)
                        @info "✓ Solved $(prob_name) with $(optimizer_name) - Status: $(sol.retcode)"
                        vars, time, alg, code = get_stats(sol, optimizer_name)
                        push!(problem, prob_name)
                        push!(n_vars, vars)
                        push!(secs, time)
                        push!(solver, alg)
                        push!(retcode, code)
                    catch e
                        push!(problem, prob_name)
                        push!(n_vars, nlp_prob !== nothing ? nlp_prob.meta.nvar : -1)
                        push!(secs, NaN)
                        push!(solver, optimizer_name)
                        push!(retcode, :FAILED)
                        println("ERROR: ", e)
                        println("Stacktrace:")
                        for (i, frame) in enumerate(stacktrace(e))
                            println("  ", i, ": ", frame)
                        end
                    end
                end
            catch e
                for (optimizer_name, optimizer) in optimizers
                    push!(problem, prob_name)
                    push!(n_vars, -1)
                    push!(secs, NaN)
                    push!(solver, optimizer_name)
                    push!(retcode, :LOAD_FAILED)
                end
                println("LOAD ERROR: ", e)
                println("Stacktrace:")
                for (i, frame) in enumerate(stacktrace(e))
                    println("  ", i, ": ", frame)
                end
            finally
                if nlp_prob !== nothing
                    try
                        finalize(nlp_prob)
                    catch e
                    end
                end
            end
        end
        GC.gc()
        @info "Completed chunk, memory usage cleaned up"
    end
    return DataFrame(problem = problem, n_vars = n_vars, secs = secs, solver = solver, retcode = retcode)
end

quad_results = run_quadratic_benchmarks(quad_problems, optimizers; chunk_size=3)

# Calculate and display success rates for quadratic problems
successful_codes = [:Success, :MaxIters, :MaxTime, :FirstOrderOptimal]
successful_results = filter(row -> row.retcode in successful_codes, quad_results)
total_attempts = nrow(quad_results)
successful_attempts = nrow(successful_results)
success_rate = total_attempts > 0 ? round(successful_attempts / total_attempts * 100, digits=1) : 0

@info "QUADRATIC PROBLEMS SUCCESS RATE: $(success_rate)% ($(successful_attempts)/$(total_attempts))"

println("Full results table for quadratic problems:")
display(quad_results)

total_attempts = nrow(quad_results)
successful_codes = [:Success, :MaxIters, :MaxTime, :FirstOrderOptimal]
successful_results = filter(row -> row.retcode in successful_codes, quad_results)
successful_attempts = nrow(successful_results)
success_rate = total_attempts > 0 ? round(successful_attempts / total_attempts * 100, digits=1) : 0

println("SUCCESS RATE ANALYSIS (Quadratic Problems):")
println("Total attempts: ", total_attempts)
println("Successful attempts: ", successful_attempts)
println("Success rate: ", success_rate, "%")
println("Return code distribution:")
if total_attempts > 0
    for (code, count) in sort(collect(pairs(countmap(quad_results.retcode))), by=x->x[2], rev=true)
        println("  ", code, ": ", count, " occurrences")
    end
else
    println("  No results to analyze")
end

if nrow(quad_results) > 0
    @df quad_results scatter(:n_vars, :secs,
        group = :solver,
        xlabel = "n. variables",
        ylabel = "secs.",
        title = "Time to solution by optimizer and number of vars",
    )
    println("Plotted quadratic problem results.")
else
    println("No quadratic problem results to plot. DataFrame is empty.")
    println("Attempted problems:")
    println(quad_problems)
end
```

```
***************************************************************************
***
This program contains Ipopt, a library for large-scale nonlinear optimizati
on.
 Ipopt is released as open source code under the Eclipse Public License (EP
L).
         For more information visit https://github.com/coin-or/Ipopt
***************************************************************************
***

This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:     2070
Number of nonzeros in Lagrangian Hessian.............:    26565

Total number of variables............................:      230
                     variables with only lower bounds:      215
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        9
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        9

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  6.9998860e-04 2.14e+03 2.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  1.6683174e-03 2.13e+03 9.90e+03  -1.0 2.63e+00   6.0 6.28e-03 3.76e-0
3h  1
   2  1.6998513e-03 2.13e+03 2.67e+04  -1.0 2.67e+00   8.2 1.00e-02 4.29e-0
5h  1
   3r 1.6998513e-03 2.13e+03 9.99e+02   2.0 0.00e+00   8.7 0.00e+00 4.57e-0
7R  2
   4r 7.1438721e-03 2.10e+03 9.98e+02   2.0 2.48e+04    -  4.37e-03 5.44e-0
5f  1
   5r 3.6811146e-02 2.10e+03 9.95e+02   2.0 5.77e+01   2.0 1.05e-03 4.11e-0
3f  1
   6r 3.6893306e-02 2.10e+03 1.11e+06   2.0 1.23e-01   6.9 9.91e-01 2.38e-0
1f  3
   7r 3.6880816e-02 2.10e+03 4.49e+04   2.0 2.79e-05   9.2 9.94e-01 1.00e+0
0f  1
   8r 3.6880329e-02 2.10e+03 7.37e+03   1.3 1.87e-06   9.6 1.00e+00 1.00e+0
0f  1
   9r 3.6878928e-02 2.10e+03 7.40e+03   1.3 5.53e-06   9.1 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10r 3.6875137e-02 2.10e+03 7.56e+03   1.3 1.74e-05   8.6 1.00e+00 1.00e+0
0f  1
  11r 3.6873139e-02 2.10e+03 7.44e+03   1.3 6.33e-06   9.1 1.00e+00 1.00e+0
0f  1
  12r 3.6868051e-02 2.10e+03 6.95e+03   1.3 1.76e-05   8.6 1.00e+00 1.00e+0
0f  1
  13r 3.6866379e-02 2.10e+03 6.88e+03   1.3 6.64e-06   9.0 1.00e+00 1.00e+0
0f  1
  14r 3.6855158e-02 2.10e+03 7.82e+03   1.3 2.06e-05   8.5 1.00e+00 1.00e+0
0f  1
  15r 3.6852651e-02 2.10e+03 7.09e+03   1.3 7.59e-06   9.0 1.00e+00 1.00e+0
0f  1
  16r 3.6841171e-02 2.10e+03 7.51e+03   1.3 2.36e-05   8.5 1.00e+00 1.00e+0
0f  1
  17r 3.6838632e-02 2.10e+03 6.65e+03   1.3 7.95e-06   8.9 1.00e+00 1.00e+0
0f  1
  18r 3.6824962e-02 2.10e+03 8.81e+03   1.3 2.28e-05   8.4 1.00e+00 1.00e+0
0f  1
  19r 3.6821939e-02 2.10e+03 6.58e+03   1.3 8.82e-06   8.9 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20r 3.6820848e-02 2.10e+03 6.49e+03   1.3 3.27e-06   9.3 1.00e+00 1.00e+0
0f  1
  21r 3.6817952e-02 2.10e+03 6.21e+03   1.3 9.57e-06   8.8 1.00e+00 1.00e+0
0f  1
  22r 3.6816829e-02 2.10e+03 6.36e+03   1.3 3.66e-06   9.2 1.00e+00 1.00e+0
0f  1
  23r 3.6812495e-02 2.10e+03 6.51e+03   1.3 1.10e-05   8.8 1.00e+00 1.00e+0
0f  1
  24r 3.6778906e-02 2.10e+03 8.83e+03   1.3 4.47e-05   8.3 1.00e+00 1.00e+0
0f  1
  25r 3.6772810e-02 2.10e+03 6.62e+03   1.3 1.22e-05   8.7 1.00e+00 1.00e+0
0f  1
  26r 3.6771355e-02 2.10e+03 5.89e+03   1.3 4.14e-06   9.1 1.00e+00 1.00e+0
0f  1
  27r 3.6764939e-02 2.10e+03 6.18e+03   1.3 1.32e-05   8.7 1.00e+00 1.00e+0
0f  1
  28r 3.6763312e-02 2.10e+03 5.63e+03   1.3 4.64e-06   9.1 1.00e+00 1.00e+0
0f  1
  29r 3.6758784e-02 2.10e+03 5.42e+03   1.3 1.35e-05   8.6 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  30r 3.6756824e-02 2.10e+03 5.74e+03   1.3 5.17e-06   9.0 1.00e+00 1.00e+0
0f  1
  31r 3.6752687e-02 2.10e+03 5.50e+03   1.3 1.56e-05   8.6 1.00e+00 1.00e+0
0f  1

Number of Iterations....: 31

                                   (scaled)                 (unscaled)
Objective...............:   3.6752687389453066e-02    3.6752687389453066e-0
2
Dual infeasibility......:   5.4998814584083348e+03    5.4998814584083348e+0
3
Constraint violation....:   1.0358170696823382e+02    2.0985653831764175e+0
3
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   2.1000219021273175e+01    2.1000219021273175e+0
1
Overall NLP error.......:   2.0491729862306448e+02    5.4998814584083348e+0
3


Number of objective function evaluations             = 39
Number of objective gradient evaluations             = 5
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 39
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 33
Number of Lagrangian Hessian evaluations             = 31
Total seconds in IPOPT                               = 30.272

EXIT: Maximum wallclock time exceeded.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:     3456
Number of nonzeros in inequality constraint Jacobian.:   131328
Number of nonzeros in Lagrangian Hessian.............:    73920

Total number of variables............................:      384
                     variables with only lower bounds:      228
                variables with lower and upper bounds:      156
                     variables with only upper bounds:        0
Total number of equality constraints.................:        9
Total number of inequality constraints...............:      342
        inequality constraints with only lower bounds:      249
   inequality constraints with lower and upper bounds:       89
        inequality constraints with only upper bounds:        4

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.2095848e+01 2.66e+03 2.73e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  1.2112651e+01 2.66e+03 3.81e+04  -1.0 8.32e+02   6.0 2.41e-05 4.58e-0
5h  1
   2r 1.2112651e+01 2.66e+03 9.99e+02   3.4 0.00e+00  10.0 0.00e+00 2.30e-0
7R  2
   3r 1.3517874e+01 2.66e+03 9.99e+02   3.4 1.45e+05    -  1.35e-05 1.70e-0
5f  1
   4r 1.3516741e+01 2.66e+03 1.76e+05   1.3 1.09e-01   6.0 9.82e-01 6.02e-0
2f  2
   5r 1.3516877e+01 2.66e+03 9.75e+03   1.3 4.65e-04   7.3 1.00e+00 1.00e+0
0f  1

Number of Iterations....: 5

                                   (scaled)                 (unscaled)
Objective...............:   1.3516877060346227e+01    1.3516877060346227e+0
1
Dual infeasibility......:   9.7543137332571878e+03    9.7543137332571878e+0
3
Constraint violation....:   2.6558523048824864e+03    2.6558523048824864e+0
3
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   4.0344181589675742e+01    4.0344181589675742e+0
1
Overall NLP error.......:   2.6558523048824864e+03    9.7543137332571878e+0
3


Number of objective function evaluations             = 11
Number of objective gradient evaluations             = 4
Number of equality constraint evaluations            = 11
Number of inequality constraint evaluations          = 11
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 7
Number of Lagrangian Hessian evaluations             = 5
Total seconds in IPOPT                               = 30.186

EXIT: Maximum wallclock time exceeded.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:       96
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:     4656

Total number of variables............................:       96
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       96
                     variables with only upper bounds:        0
Total number of equality constraints.................:        1
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  4.1972504e-01 4.00e-02 4.77e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  9.6266670e-02 1.79e-15 3.49e-02  -1.0 1.46e-02    -  1.00e+00 1.00e+0
0h  1
   2  8.5167968e-02 3.19e-16 7.55e-01  -2.5 2.36e-03    -  9.21e-01 1.00e+0
0f  1
   3  5.7281281e-02 3.64e-17 7.13e-03  -2.5 1.20e-02    -  1.00e+00 1.00e+0
0f  1
   4  4.1007753e-02 4.48e-16 1.17e-01  -3.8 6.45e-03    -  7.94e-01 1.00e+0
0f  1
   5  3.5972456e-02 1.18e-16 6.24e-02  -3.8 4.94e-03    -  7.55e-01 1.00e+0
0f  1
   6  3.4734557e-02 3.47e-18 4.76e-03  -3.8 2.51e-03    -  1.00e+00 1.00e+0
0f  1
   7  3.4469655e-02 4.68e-17 1.56e-03  -3.8 5.21e-04    -  1.00e+00 1.00e+0
0f  1
   8  3.4463910e-02 2.98e-16 1.46e-04  -3.8 8.22e-05    -  1.00e+00 1.00e+0
0f  1
   9  3.3925583e-02 2.93e-16 8.18e-03  -5.7 1.51e-03    -  9.04e-01 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  3.3789299e-02 6.56e-16 2.32e-03  -5.7 6.83e-04    -  1.00e+00 1.00e+0
0f  1
  11  3.3744681e-02 7.63e-16 3.66e-03  -5.7 5.50e-04    -  1.00e+00 1.00e+0
0f  1
  12  3.3741502e-02 1.06e-16 4.00e-04  -5.7 1.80e-04    -  1.00e+00 1.00e+0
0f  1
  13  3.3741499e-02 1.39e-17 5.53e-05  -5.7 2.18e-05    -  1.00e+00 1.00e+0
0f  1
  14  3.3741494e-02 2.62e-16 5.66e-06  -5.7 1.83e-06    -  1.00e+00 1.00e+0
0h  1
  15  3.3734554e-02 6.77e-17 4.54e-04  -7.0 1.32e-04    -  1.00e+00 1.00e+0
0f  1
  16  3.3734043e-02 1.42e-16 7.76e-05  -7.0 5.13e-05    -  1.00e+00 1.00e+0
0f  1
  17  3.3734036e-02 3.89e-16 2.41e-05  -7.0 8.69e-06    -  1.00e+00 1.00e+0
0h  1
  18  3.3734037e-02 2.50e-16 3.17e-06  -7.0 1.85e-06    -  1.00e+00 1.00e+0
0h  1
  19  3.3734037e-02 3.05e-16 7.25e-07  -7.0 2.89e-07    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 19

                                   (scaled)                 (unscaled)
Objective...............:   3.3734036836557693e-02    3.3734036836557693e-0
2
Dual infeasibility......:   7.2498226292322491e-07    7.2498226292322491e-0
7
Constraint violation....:   3.0531133177191805e-16    3.0531133177191805e-1
6
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   9.0909090909090915e-08    9.0909090909090915e-0
8
Overall NLP error.......:   7.2498226292322491e-07    7.2498226292322491e-0
7


Number of objective function evaluations             = 20
Number of objective gradient evaluations             = 20
Number of equality constraint evaluations            = 20
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 20
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 19
Total seconds in IPOPT                               = 17.471

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:        3
Number of nonzeros in Lagrangian Hessian.............:        6

Total number of variables............................:        3
                     variables with only lower bounds:        3
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        1
        inequality constraints with only lower bounds:        1
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  2.2500000e+00 0.00e+00 2.71e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  7.9307609e-01 0.00e+00 2.71e+00  -1.0 1.12e+00    -  3.25e-01 8.25e-0
1f  1
   2  6.6886239e-01 0.00e+00 4.69e+00  -1.0 4.68e-02   2.0 1.00e+00 1.00e+0
0f  1
   3  4.1836368e-01 0.00e+00 8.14e-01  -1.0 2.04e-01    -  9.21e-01 1.00e+0
0f  1
   4  3.7672986e-01 0.00e+00 9.02e-01  -1.7 3.10e-02   1.5 1.00e+00 1.00e+0
0f  1
   5  2.1085739e-01 0.00e+00 3.76e-01  -1.7 2.17e-01    -  6.54e-01 1.00e+0
0f  1
   6  1.6849109e-01 0.00e+00 2.85e-01  -1.7 3.97e-01    -  1.00e+00 3.89e-0
1f  2
   7  1.6172448e-01 0.00e+00 2.16e-01  -1.7 2.07e-02   1.0 1.00e+00 1.00e+0
0f  1
   8  1.3674336e-01 0.00e+00 2.42e-01  -1.7 1.74e+00    -  4.05e-01 7.67e-0
2f  2
   9  1.2903102e-01 0.00e+00 4.50e-02  -1.7 6.69e-02    -  1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.2316475e-01 0.00e+00 9.18e-02  -2.5 2.48e-02   0.6 1.00e+00 1.00e+0
0f  1
  11  1.2059806e-01 0.00e+00 1.12e-01  -2.5 1.14e-02   1.0 1.00e+00 1.00e+0
0f  1
  12  1.1639040e-01 0.00e+00 6.53e-02  -2.5 1.07e-01    -  1.00e+00 1.99e-0
1f  2
  13  1.1442786e-01 0.00e+00 3.24e-02  -2.5 1.33e-02    -  1.00e+00 1.00e+0
0f  1
  14  1.1397550e-01 0.00e+00 6.35e-03  -2.5 1.93e-03   0.5 1.00e+00 1.00e+0
0f  1
  15  1.1119836e-01 0.00e+00 6.62e-03  -3.8 1.24e-02    -  1.00e+00 1.00e+0
0f  1
  16  1.1126059e-01 0.00e+00 2.32e-04  -3.8 2.80e-04    -  1.00e+00 1.00e+0
0f  1
  17  1.1111270e-01 0.00e+00 1.43e-03  -5.7 6.66e-04    -  1.00e+00 1.00e+0
0f  1
  18  1.1111295e-01 0.00e+00 1.68e-04  -5.7 1.53e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.1111295e-01 0.00e+00 2.54e-05  -5.7 2.25e-05    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.1111295e-01 0.00e+00 1.31e-05  -5.7 2.27e-06    -  1.00e+00 1.00e+0
0h  1
  21  1.1111120e-01 0.00e+00 1.92e-05  -7.0 1.60e-05   0.0 1.00e+00 1.00e+0
0f  1
  22  1.1111120e-01 0.00e+00 7.86e-06  -7.0 6.21e-06    -  1.00e+00 1.00e+0
0h  1
  23  1.1111120e-01 0.00e+00 2.17e-06  -7.0 1.03e-06   0.5 1.00e+00 1.00e+0
0h  1
  24  1.1111120e-01 0.00e+00 1.38e-06  -7.0 2.28e-06    -  1.00e+00 5.00e-0
1h  2
  25  1.1111120e-01 0.00e+00 6.94e-07  -7.0 6.70e-08   0.9 1.00e+00 1.00e+0
0h  1

Number of Iterations....: 25

                                   (scaled)                 (unscaled)
Objective...............:   1.1111119979804229e-01    1.1111119979804229e-0
1
Dual infeasibility......:   6.9435271937006593e-07    6.9435271937006593e-0
7
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+0
0
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   9.0909090909092715e-08    9.0909090909092715e-0
8
Overall NLP error.......:   6.9435271937006593e-07    6.9435271937006593e-0
7


Number of objective function evaluations             = 33
Number of objective gradient evaluations             = 26
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 33
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 26
Number of Lagrangian Hessian evaluations             = 25
Total seconds in IPOPT                               = 0.010

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:   183750
Number of nonzeros in Lagrangian Hessian.............:    15400

Total number of variables............................:      175
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:     1050
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:     1050

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  0.0000000e+00 4.98e-02 9.92e-01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  9.1177485e-02 4.91e-02 1.40e+02  -1.0 5.61e-02   4.0 1.00e+00 1.77e-0
1h  1
   2  9.1714461e-02 4.90e-02 5.26e+03  -1.0 4.03e-02   4.4 1.00e+00 3.52e-0
3h  1
   3  1.0187941e-01 4.66e-02 3.52e+04  -1.0 5.71e-02   4.9 1.00e+00 4.92e-0
2h  1

Number of Iterations....: 3

                                   (scaled)                 (unscaled)
Objective...............:   1.0187940613798399e-01    1.0187940613798399e-0
1
Dual infeasibility......:   3.5188718182804369e+04    3.5188718182804369e+0
4
Constraint violation....:   4.6630028933483177e-02    4.6630028933483177e-0
2
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   2.3426454706504036e+00    2.3426454706504036e+0
0
Overall NLP error.......:   3.5188718182804369e+04    3.5188718182804369e+0
4


Number of objective function evaluations             = 4
Number of objective gradient evaluations             = 4
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 4
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 4
Number of Lagrangian Hessian evaluations             = 3
Total seconds in IPOPT                               = 42.578

EXIT: Maximum wallclock time exceeded.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:     3456
Number of nonzeros in inequality constraint Jacobian.:   131328
Number of nonzeros in Lagrangian Hessian.............:    73920

Total number of variables............................:      384
                     variables with only lower bounds:      228
                variables with lower and upper bounds:      156
                     variables with only upper bounds:        0
Total number of equality constraints.................:        9
Total number of inequality constraints...............:      342
        inequality constraints with only lower bounds:      249
   inequality constraints with lower and upper bounds:       89
        inequality constraints with only upper bounds:        4

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.2057448e+01 2.66e+03 2.73e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  1.2070111e+01 2.66e+03 3.82e+04  -1.0 8.33e+02   6.0 2.41e-05 4.59e-0
5h  1
   2r 1.2070111e+01 2.66e+03 9.99e+02   3.4 0.00e+00  10.9 0.00e+00 2.29e-0
7R  2
   3r 1.3176285e+01 2.66e+03 9.99e+02   3.4 1.45e+05    -  1.34e-05 1.69e-0
5f  1
   4r 1.3175686e+01 2.66e+03 1.76e+05   1.3 1.09e-01   6.0 9.82e-01 6.04e-0
2f  2
   5r 1.3175805e+01 2.66e+03 1.00e+04   1.3 4.73e-04   7.3 1.00e+00 1.00e+0
0f  1

Number of Iterations....: 5

                                   (scaled)                 (unscaled)
Objective...............:   1.3175805238032627e+01    1.3175805238032627e+0
1
Dual infeasibility......:   1.0038478290146477e+04    1.0038478290146477e+0
4
Constraint violation....:   2.6558526154466922e+03    2.6558526154466922e+0
3
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   3.9844453499617295e+01    3.9844453499617295e+0
1
Overall NLP error.......:   2.6558526154466922e+03    1.0038478290146477e+0
4


Number of objective function evaluations             = 9
Number of objective gradient evaluations             = 4
Number of equality constraint evaluations            = 9
Number of inequality constraint evaluations          = 9
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 7
Number of Lagrangian Hessian evaluations             = 5
Total seconds in IPOPT                               = 30.253

EXIT: Maximum wallclock time exceeded.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:    91512
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:   808356

Total number of variables............................:     1271
                     variables with only lower bounds:        0
                variables with lower and upper bounds:     1271
                     variables with only upper bounds:        0
Total number of equality constraints.................:       72
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.8553735e+00 1.01e+01 2.12e-01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  1.8742171e+00 7.55e+00 1.13e+01  -1.0 3.73e-01   2.0 9.90e-01 2.52e-0
1h  1

Number of Iterations....: 1

                                   (scaled)                 (unscaled)
Objective...............:   1.8742171418847360e+00    1.8742171418847360e+0
0
Dual infeasibility......:   1.1340415586752636e+01    1.1340415586752636e+0
1
Constraint violation....:   7.5501556656700490e+00    7.5501556656700490e+0
0
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   2.0604460819772835e+03    2.0604460819772835e+0
3
Overall NLP error.......:   2.0604460819772835e+03    2.0604460819772835e+0
3


Number of objective function evaluations             = 2
Number of objective gradient evaluations             = 2
Number of equality constraint evaluations            = 2
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 2
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 1
Total seconds in IPOPT                               = 68.725

EXIT: Maximum wallclock time exceeded.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:        8
Number of nonzeros in inequality constraint Jacobian.:     2216
Number of nonzeros in Lagrangian Hessian.............:       36

Total number of variables............................:        8
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        8
                     variables with only upper bounds:        0
Total number of equality constraints.................:        1
Total number of inequality constraints...............:      277
        inequality constraints with only lower bounds:      277
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  6.0067449e+01 9.20e-01 1.00e+02  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  7.3421159e+01 9.02e-01 9.88e+01  -1.0 1.01e+02    -  1.51e-02 1.94e-0
2f  1
   2  7.3706124e+01 9.02e-01 3.00e+02  -1.0 1.28e+02   2.0 1.59e-02 9.00e-0
5h  1
   3  8.4670525e+01 8.99e-01 8.78e+02  -1.0 1.18e+02   1.5 3.42e-02 3.44e-0
3h  1
   4  8.4868270e+01 8.99e-01 5.27e+04  -1.0 1.25e+02   2.9 2.34e-02 5.00e-0
5h  1
   5  9.3349073e+01 8.97e-01 8.04e+05  -1.0 1.39e+02   3.3 4.90e-02 1.73e-0
3h  1
   6  9.5449773e+01 8.97e-01 2.76e+06  -1.0 1.21e+02   3.7 4.76e-02 6.47e-0
4h  1
   7  9.8210283e+01 8.96e-01 1.46e+07  -1.0 1.10e+02   4.1 7.51e-02 9.81e-0
4h  1
   8  1.5192091e+02 8.80e-01 4.02e+07  -1.0 9.75e+01   4.6 5.76e-02 1.76e-0
2h  1
   9  1.5788682e+02 8.78e-01 1.76e+09  -1.0 9.45e+01   5.9 1.09e-01 2.26e-0
3h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  2.2520920e+02 8.66e-01 2.42e+09  -1.0 1.60e+02   6.3 2.65e-02 1.40e-0
2h  1
  11  2.3290562e+02 8.63e-01 8.68e+10  -1.0 8.66e+01   7.6 1.15e-01 2.78e-0
3h  1
  12  2.3533260e+02 8.63e-01 9.85e+10  -1.0 1.84e+02   8.1 2.41e-02 3.80e-0
4h  1
  13  2.6938433e+02 8.53e-01 7.63e+10  -1.0 8.63e+01   8.5 3.05e-04 1.16e-0
2h  1
  14  2.7198683e+02 8.53e-01 2.27e+13  -1.0 8.71e+01   9.8 2.22e-01 6.61e-0
4h  1
  15  5.5045656e+02 8.10e-01 1.89e+13  -1.0 9.51e+01   9.3 1.38e-03 5.01e-0
2h  1
  16  5.5679191e+02 8.09e-01 1.67e+14  -1.0 8.56e+01  10.7 2.05e-01 1.19e-0
3h  1
  17  4.5543200e+02 8.09e-01 1.46e+14  -1.0 3.95e+08    -  3.12e-13 3.38e-0
9f  1
  18  6.3381611e+02 7.82e-01 1.87e+15  -1.0 8.51e+01  11.1 9.35e-01 3.32e-0
2h  1
  19  6.5300468e+02 7.81e-01 2.44e+15  -1.0 2.40e+02  12.4 9.42e-03 1.91e-0
3h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  6.5893001e+02 7.80e-01 1.49e+17  -1.0 8.04e+01  13.8 1.89e-01 1.19e-0
3h  1
  21  6.5900191e+02 7.80e-01 1.21e+18  -1.0 8.09e+01  14.2 5.20e-01 1.21e-0
5h  1
  22  1.4383138e+04 9.43e-17 4.26e+17  -1.0 7.80e+01  13.7 1.08e-07 1.00e+0
0h  1
  23  1.4383138e+04 9.41e-17 1.21e+16  -1.0 5.24e-14  15.9 9.90e-01 1.00e+0
0   0
  24  1.4383138e+04 1.90e-16 1.21e+14  -1.0 3.77e-14  15.5 9.90e-01 1.00e+0
0   0
  25  1.4383138e+04 3.88e-17 1.21e+12  -1.0 1.10e-13  15.0 9.90e-01 1.00e+0
0f  1
  26  1.4383138e+04 2.39e-17 1.21e+10  -1.0 1.23e-11  14.5 9.90e-01 1.00e+0
0f  1
  27  1.4383138e+04 1.92e-16 1.03e+08  -1.0 1.23e-09  14.0 9.91e-01 1.00e+0
0f  1
  28  1.4383138e+04 8.73e-17 3.61e+06  -1.0 1.01e-07  13.6 1.00e+00 1.00e+0
0f  1
  29  1.4383138e+04 6.13e-17 1.90e+05  -1.0 1.59e-08  13.1 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  30  1.4383138e+04 1.62e-17 4.59e+05  -1.0 1.16e-07  12.6 1.00e+00 1.00e+0
0f  1
  31  1.4383138e+04 1.44e-16 1.32e+05  -1.0 9.96e-08  12.1 1.00e+00 1.00e+0
0f  1
  32  1.4383138e+04 9.50e-17 1.21e+05  -1.0 2.74e-07  11.6 1.00e+00 1.00e+0
0f  1
  33  1.4383138e+04 1.51e-17 5.35e+04  -1.0 3.64e-07  11.2 1.00e+00 1.00e+0
0f  1
  34  1.4383137e+04 3.53e-17 3.70e+04  -1.0 7.57e-07  10.7 1.00e+00 1.00e+0
0f  1
  35  1.4383137e+04 1.18e-16 1.91e+04  -1.0 1.17e-06  10.2 1.00e+00 1.00e+0
0f  1
  36  1.4383136e+04 1.08e-16 1.19e+04  -1.0 2.19e-06   9.7 1.00e+00 1.00e+0
0f  1
  37  1.4383135e+04 7.05e-17 6.56e+03  -1.0 3.62e-06   9.3 1.00e+00 1.00e+0
0f  1
  38  1.4383133e+04 7.43e-17 3.91e+03  -1.0 6.48e-06   8.8 1.00e+00 1.00e+0
0f  1
  39  1.4383129e+04 9.97e-17 2.21e+03  -1.0 1.10e-05   8.3 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  40  1.4383122e+04 2.21e-16 1.30e+03  -1.0 1.93e-05   7.8 1.00e+00 1.00e+0
0f  1
  41  1.4383110e+04 1.03e-16 7.44e+02  -1.0 3.32e-05   7.3 1.00e+00 1.00e+0
0f  1
  42  1.4383089e+04 4.05e-17 4.33e+02  -1.0 5.81e-05   6.9 1.00e+00 1.00e+0
0f  1
  43  1.4383051e+04 1.13e-17 2.50e+02  -1.0 1.01e-04   6.4 1.00e+00 1.00e+0
0f  1
  44  1.4382981e+04 2.76e-17 1.46e+02  -1.0 1.76e-04   5.9 1.00e+00 1.00e+0
0f  1
  45  1.4382849e+04 9.10e-17 8.54e+01  -1.0 3.09e-04   5.4 1.00e+00 1.00e+0
0f  1
  46  1.4382584e+04 1.15e-16 5.06e+01  -1.0 5.50e-04   5.0 1.00e+00 1.00e+0
0f  1
  47  1.4382020e+04 7.59e-17 3.05e+01  -1.0 9.93e-04   4.5 1.00e+00 1.00e+0
0f  1
  48  1.4380721e+04 1.08e-16 1.92e+01  -1.0 1.87e-03   4.0 1.00e+00 1.00e+0
0f  1
  49  1.4377497e+04 4.48e-17 1.44e+01  -1.0 4.22e-03   3.5 1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  50  1.4368912e+04 6.72e-18 1.17e+01  -1.0 1.03e-02   3.1 1.00e+00 1.00e+0
0f  1
  51  1.4344651e+04 1.10e-17 1.03e+01  -1.0 2.72e-02   2.6 1.00e+00 1.00e+0
0f  1
  52  1.4273388e+04 6.08e-17 9.62e+00  -1.0 7.62e-02   2.1 1.00e+00 1.00e+0
0f  1
  53  1.4064935e+04 1.88e-17 1.71e+01  -1.0 2.20e-01   1.6 8.65e-01 1.00e+0
0f  1
  54  1.3458814e+04 9.63e-17 4.30e+01  -1.0 6.55e-01   1.1 1.00e+00 1.00e+0
0f  1
  55  1.1896493e+04 1.89e-17 1.27e+02  -1.0 1.83e+00   0.7 4.50e-01 1.00e+0
0f  1
  56  8.7653654e+03 2.03e-16 2.74e+02  -1.0 3.96e+00   0.2 7.40e-01 1.00e+0
0f  1
  57  7.6313046e+03 9.39e-17 1.40e+02  -1.0 1.64e+00   0.6 1.00e+00 1.00e+0
0f  1
  58  7.0602727e+03 2.27e-17 2.27e+02  -1.0 3.26e+00   0.1 1.00e+00 2.70e-0
1f  1
  59  4.9745073e+03 3.04e-18 1.66e+02  -1.0 7.08e+00  -0.3 3.11e-01 5.57e-0
1f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  60  4.7045249e+03 5.94e-17 4.47e+02  -1.0 1.07e+01  -0.8 1.00e+00 5.92e-0
2f  1
  61  2.4792342e+03 1.62e-16 2.21e+02  -1.0 7.16e+00  -0.4 1.00e+00 1.00e+0
0f  1
  62  1.3530002e+03 4.68e-17 6.98e+02  -1.0 1.62e+01  -0.9 4.62e-02 1.00e+0
0f  1
  63  8.5029316e+02 3.47e-18 1.07e+02  -1.0 4.65e+00  -0.4 1.00e+00 1.00e+0
0f  1
  64  7.1699805e+02 9.02e-17 3.31e+01  -1.0 1.38e+00  -0.0 4.84e-01 1.00e+0
0f  1
  65  6.7656107e+02 2.78e-17 3.60e+01  -1.0 2.25e+00  -0.5 7.25e-01 2.07e-0
1f  1
  66  5.0995086e+02 1.25e-16 6.57e+01  -1.0 7.53e+00    -  3.88e-01 4.53e-0
1f  1
  67  4.5284647e+02 1.11e-16 6.18e+01  -1.0 5.67e+00  -1.0 4.18e-01 2.65e-0
1f  1
  68  4.4401718e+02 0.00e+00 5.31e+00  -1.0 1.58e+00    -  1.00e+00 1.00e+0
0f  1
  69  4.3118372e+02 2.22e-16 6.59e-01  -1.0 4.37e-01    -  1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  70  4.2832398e+02 2.78e-17 3.93e-01  -1.7 1.23e-01    -  1.00e+00 1.00e+0
0f  1
  71  4.2816868e+02 8.33e-17 9.22e-02  -1.7 2.98e-02    -  1.00e+00 1.00e+0
0f  1
  72  4.2738920e+02 0.00e+00 8.02e-02  -2.5 3.84e-02    -  1.00e+00 1.00e+0
0f  1
  73  4.2736683e+02 5.55e-17 1.23e-02  -2.5 5.48e-03    -  1.00e+00 1.00e+0
0f  1
  74  4.2724048e+02 1.39e-16 6.07e-03  -3.8 7.08e-03    -  1.00e+00 1.00e+0
0f  1
  75  4.2723969e+02 5.55e-17 1.30e-03  -3.8 6.44e-04    -  1.00e+00 1.00e+0
0f  1
  76  4.2723263e+02 0.00e+00 7.22e-04  -5.7 4.16e-04    -  1.00e+00 1.00e+0
0f  1
  77  4.2723263e+02 1.94e-16 2.22e-05  -5.7 2.84e-05    -  1.00e+00 1.00e+0
0h  1
  78  4.2723263e+02 1.11e-16 5.24e-06  -5.7 1.22e-06    -  1.00e+00 1.00e+0
0h  1
  79  4.2723254e+02 1.11e-16 4.20e-06  -7.0 4.87e-06    -  1.00e+00 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  80  4.2723254e+02 8.33e-17 4.20e-07  -7.0 2.44e-07    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 80

                                   (scaled)                 (unscaled)
Objective...............:   2.6954156297485696e+01    4.2723254442370506e+0
2
Dual infeasibility......:   4.2045888000871561e-07    6.6644162462032336e-0
6
Constraint violation....:   8.3266726846886741e-17    8.3266726846886741e-1
7
Variable bound violation:   8.9075433662594991e-09    8.9075433662594991e-0
9
Complementarity.........:   9.0909090913713597e-08    1.4409400091641804e-0
6
Overall NLP error.......:   4.2045888000871561e-07    6.6644162462032336e-0
6


Number of objective function evaluations             = 81
Number of objective gradient evaluations             = 81
Number of equality constraint evaluations            = 81
Number of inequality constraint evaluations          = 81
Number of equality constraint Jacobian evaluations   = 81
Number of inequality constraint Jacobian evaluations = 81
Number of Lagrangian Hessian evaluations             = 80
Total seconds in IPOPT                               = 0.436

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:       50
Number of nonzeros in inequality constraint Jacobian.:     1200
Number of nonzeros in Lagrangian Hessian.............:       55

Total number of variables............................:       10
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       10
                     variables with only upper bounds:        0
Total number of equality constraints.................:        5
Total number of inequality constraints...............:      120
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:      120
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.2595275e+01 9.70e-01 7.66e-02  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  7.4290282e+00 0.00e+00 5.78e+01  -1.0 3.65e-01    -  3.61e-02 1.00e+0
0f  1
   2  7.3851998e+00 0.00e+00 5.80e+00  -1.0 1.32e-02   2.0 1.00e+00 1.00e+0
0f  1
   3  6.9491636e+00 0.00e+00 6.68e+00  -1.0 1.57e-01    -  7.88e-01 1.00e+0
0f  1
   4  6.1022782e+00 0.00e+00 1.13e+01  -1.0 4.18e-01    -  1.00e+00 1.00e+0
0f  1
   5  6.0717411e+00 0.00e+00 5.80e-01  -1.0 1.09e-02   1.5 1.00e+00 1.00e+0
0f  1
   6  4.8382266e+00 0.00e+00 7.73e+00  -1.7 6.69e-01    -  4.49e-01 1.00e+0
0f  1
   7  4.8222602e+00 0.00e+00 5.53e-01  -1.7 7.36e-03   1.0 1.00e+00 1.00e+0
0f  1
   8  4.7626985e+00 0.00e+00 1.01e-01  -1.7 2.65e-02   0.6 1.00e+00 1.00e+0
0f  1
   9  4.5136222e+00 0.00e+00 3.12e+00  -2.5 1.59e-01   0.1 7.91e-01 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  4.4716551e+00 0.00e+00 3.01e-01  -2.5 1.83e-02   0.5 1.00e+00 1.00e+0
0f  1
  11  4.2875154e+00 0.00e+00 1.06e+00  -2.5 1.06e-01   0.0 1.00e+00 1.00e+0
0f  1
  12  4.2373270e+00 0.00e+00 1.70e-01  -2.5 1.73e-02   0.5 1.00e+00 1.00e+0
0f  1
  13  4.0979731e+00 0.00e+00 6.42e-01  -2.5 5.95e-02  -0.0 1.00e+00 1.00e+0
0f  1
  14  2.8104315e+00 0.00e+00 1.48e+01  -2.5 3.02e+00  -0.5 1.91e-01 4.41e-0
1f  1
  15  2.6662950e+00 0.00e+00 1.17e+00  -2.5 1.41e-01  -0.1 1.00e+00 1.00e+0
0f  1
  16  1.8590501e+00 0.00e+00 4.21e+00  -2.5 6.21e-01  -0.5 1.00e+00 1.00e+0
0f  1
  17  1.7500074e+00 0.00e+00 3.27e-01  -2.5 6.62e-02  -0.1 1.00e+00 1.00e+0
0f  1
  18  1.3926176e+00 0.00e+00 4.25e-01  -2.5 1.99e-01  -0.6 1.00e+00 1.00e+0
0f  1
  19  9.4820859e-01 0.00e+00 4.11e-01  -2.5 4.04e-01  -1.1 7.94e-01 1.00e+0
0f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  5.3435528e-01 0.00e+00 4.98e-01  -2.5 2.38e+00    -  6.35e-01 1.93e-0
1f  1
  21  2.4569212e-01 0.00e+00 2.28e-01  -2.5 6.13e-01    -  1.00e+00 5.35e-0
1f  1
  22  3.5117687e-01 0.00e+00 2.39e-01  -2.5 1.59e-01    -  5.48e-01 1.00e+0
0f  1
  23  3.4202125e-01 0.00e+00 1.32e-02  -2.5 6.70e-02    -  1.00e+00 1.00e+0
0f  1
  24  6.4606872e-02 0.00e+00 1.50e-01  -3.8 1.15e-01    -  9.92e-01 9.85e-0
1f  1
  25  1.4902796e-02 0.00e+00 2.75e-03  -3.8 2.88e-02    -  1.00e+00 1.00e+0
0f  1
  26  1.9499720e-02 0.00e+00 6.33e-05  -3.8 3.97e-03    -  1.00e+00 1.00e+0
0f  1
  27  3.0836430e-04 0.00e+00 6.39e-04  -5.7 6.26e-03    -  1.00e+00 1.00e+0
0f  1
  28  2.3981178e-04 0.00e+00 1.82e-08  -5.7 2.78e-05    -  1.00e+00 1.00e+0
0f  1
  29  1.1789754e-05 0.00e+00 7.84e-08  -7.0 6.86e-05    -  1.00e+00 1.00e+0
0f  1

Number of Iterations....: 29

                                   (scaled)                 (unscaled)
Objective...............:   1.1789754348677797e-05    1.1789754348677797e-0
5
Dual infeasibility......:   7.8413121968802421e-08    7.8413121968802421e-0
8
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+0
0
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   1.2319818184323468e-07    1.2319818184323468e-0
7
Overall NLP error.......:   1.2319818184323468e-07    1.2319818184323468e-0
7


Number of objective function evaluations             = 30
Number of objective gradient evaluations             = 30
Number of equality constraint evaluations            = 30
Number of inequality constraint evaluations          = 30
Number of equality constraint Jacobian evaluations   = 30
Number of inequality constraint Jacobian evaluations = 30
Number of Lagrangian Hessian evaluations             = 29
Total seconds in IPOPT                               = 0.094

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:   340560
Number of nonzeros in Lagrangian Hessian.............:    64980

Total number of variables............................:      360
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      300
                     variables with only upper bounds:       59
Total number of equality constraints.................:        0
Total number of inequality constraints...............:      946
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      946

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.8536373e+00 7.80e+01 1.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  1.9171542e+00 7.80e+01 9.90e+03  -1.0 2.06e+01   6.0 5.18e-04 4.82e-0
4h  1

Number of Iterations....: 1

                                   (scaled)                 (unscaled)
Objective...............:   1.9171542410561992e+00    1.9171542410561992e+0
0
Dual infeasibility......:   9.8991286008272818e+03    9.8991286008272818e+0
3
Constraint violation....:   7.7995725448164009e+01    7.7995725448164009e+0
1
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+0
0
Complementarity.........:   1.0000000000000025e+09    1.0000000000000025e+0
9
Overall NLP error.......:   1.0000000000000025e+09    1.0000000000000025e+0
9


Number of objective function evaluations             = 2
Number of objective gradient evaluations             = 2
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 2
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 2
Number of Lagrangian Hessian evaluations             = 1
Total seconds in IPOPT                               = 64.172

EXIT: Maximum wallclock time exceeded.
Full results table for quadratic problems:
10×5 DataFrame
 Row │ problem   n_vars  secs       solver  retcode
     │ String    Int64   Float64    String  Symbol
─────┼──────────────────────────────────────────────
   1 │ PRIMALC1     230  30.9028    Ipopt   MaxTime
   2 │ QPCBOEI1     384  30.2168    Ipopt   MaxTime
   3 │ DUAL2         96  17.4721    Ipopt   Success
   4 │ HS35           3   0.01092   Ipopt   Success
   5 │ GMNCASE2     175  42.8585    Ipopt   MaxTime
   6 │ QPNBOEI1     384  30.289     Ipopt   MaxTime
   7 │ TABLE8      1271  68.7647    Ipopt   MaxTime
   8 │ DUALC5         8   0.438994  Ipopt   Success
   9 │ DEGENQP       10   0.096483  Ipopt   Success
  10 │ LEUVEN7      360  64.2364    Ipopt   MaxTime
SUCCESS RATE ANALYSIS (Quadratic Problems):
Total attempts: 10
Successful attempts: 10
Success rate: 100.0%
Return code distribution:
  MaxTime: 6 occurrences
  Success: 4 occurrences
Plotted quadratic problem results.
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/OptimizationCUTEst","CUTEst_quadratic.jmd")
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
Threads: 128 default, 0 interactive, 64 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953:

```

Package Information:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/OptimizationCUTEst/Project.toml`
⌃ [1b53aba6] CUTEst v1.3.2
⌃ [a93c6f00] DataFrames v1.7.0
⌃ [b6b21f68] Ipopt v1.10.6
⌃ [b8f27783] MathOptInterface v1.42.1
⌃ [a4795742] NLPModels v0.21.5
⌅ [7f7a1694] Optimization v4.5.0
⌃ [fd9f6733] OptimizationMOI v0.5.5
⌅ [064b21be] OptimizationNLPModels v0.0.2
⌃ [36348300] OptimizationOptimJL v0.4.3
⌃ [42dfb2eb] OptimizationOptimisers v0.3.8
⌃ [91a5bcdd] Plots v1.40.17
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [2913bbd2] StatsBase v0.34.6
⌃ [f3b207a7] StatsPlots v0.15.7
  [de0858da] Printf
  [10745b16] Statistics v1.10.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl/benchmarks/OptimizationCUTEst/Manifest.toml`
⌃ [47edcb42] ADTypes v1.16.0
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
⌃ [7d9f7c33] Accessors v0.1.42
⌃ [79e6a3ab] Adapt v4.3.0
  [66dad0bd] AliasTables v1.1.3
  [ec485272] ArnoldiMethod v0.4.0
  [7d9fca2a] Arpack v0.5.4
⌃ [4fba245c] ArrayInterface v7.19.0
⌃ [4c555306] ArrayLayouts v1.11.2
  [13072b0f] AxisAlgorithms v1.1.0
⌃ [6e4b80f9] BenchmarkTools v1.6.0
  [e2ed5e7c] Bijections v0.2.2
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [8e7c35d0] BlockArrays v1.7.0
⌃ [70df07ce] BracketingNonlinearSolve v1.3.0
⌃ [2a0fbf3d] CPUSummary v0.2.6
⌃ [1b53aba6] CUTEst v1.3.2
⌃ [d360d2e6] ChainRulesCore v1.25.2
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [aaaa29a8] Clustering v0.15.8
  [523fee87] CodecBzip2 v0.8.5
  [944b1d66] CodecZlib v0.7.8
⌃ [35d6a980] ColorSchemes v3.30.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
⌃ [861a8166] Combinatorics v1.0.3
⌅ [a80b9123] CommonMark v0.9.1
⌃ [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
⌃ [34da2185] Compat v4.18.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.5.0
⌃ [8f4d0f93] Conda v1.10.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.16.0
⌃ [a93c6f00] DataFrames v1.7.0
⌅ [864edb3b] DataStructures v0.18.22
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v6.181.0
⌃ [459566f4] DiffEqCallbacks v4.8.0
⌃ [77a26b50] DiffEqNoiseProcess v5.24.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌃ [a0c0ee7d] DifferentiationInterface v0.7.4
⌃ [8d63f2c5] DispatchDoctor v0.4.26
  [b4f34e82] Distances v0.10.12
⌃ [31c24e10] Distributions v0.25.120
  [ffbed154] DocStringExtensions v0.9.5
  [5b8099bc] DomainSets v0.7.16
⌃ [7c1d4256] DynamicPolynomials v0.6.2
⌃ [06fc5a27] DynamicQuantities v1.8.0
  [4e289a0a] EnumX v1.0.5
⌃ [f151be2c] EnzymeCore v0.8.12
  [460bff9d] ExceptionUnwrapping v0.1.11
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
⌃ [c87230d0] FFMPEG v0.4.4
⌃ [7a1cc6ca] FFTW v1.9.0
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
⌃ [a4df4552] FastPower v1.1.3
⌃ [1a297f60] FillArrays v1.13.0
⌃ [64ca27bc] FindFirstFunctions v1.4.1
⌃ [6a86dc24] FiniteDiff v2.27.0
  [53c48c17] FixedPointNumbers v0.8.5
  [1fa38f19] Format v1.3.7
⌃ [f6369f11] ForwardDiff v1.0.1
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.5.2
  [46192b85] GPUArraysCore v0.2.0
⌃ [28b8d3ca] GR v0.73.17
⌃ [d7ba0133] Git v1.4.0
⌃ [c27321d9] Glob v1.3.1
⌃ [86223c79] Graphs v1.13.0
  [42e2da0e] Grisu v1.0.2
⌃ [cd3eb016] HTTP v1.10.17
  [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.28
⌃ [7073ff75] IJulia v1.29.2
  [615f187c] IfElse v0.1.1
⌅ [3263718b] ImplicitDiscreteSolve v0.1.3
  [d25df0c9] Inflate v0.1.5
⌃ [842dd82b] InlineStrings v1.4.4
  [18e54dd8] IntegerMathUtils v0.1.3
⌅ [a98d9a8b] Interpolations v0.15.1
⌃ [8197267c] IntervalSets v0.7.11
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
⌃ [b6b21f68] Ipopt v1.10.6
⌃ [92d709cd] IrrationalConstants v0.2.4
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.7.1
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [ae98c720] Jieko v0.2.1
⌃ [98e50ef6] JuliaFormatter v2.1.6
⌅ [70703baa] JuliaSyntax v0.4.10
⌃ [ccbc3e58] JumpProcesses v9.16.1
⌃ [5ab0869b] KernelDensity v0.6.10
⌃ [ba0b0d4f] Krylov v0.10.1
  [5be7bae1] LBFGSB v0.4.1
  [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.8
  [10f19ff3] LayoutPointers v0.1.17
⌃ [5078a376] LazyArrays v2.6.2
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.1
⌃ [87fe0de2] LineSearch v0.1.4
⌃ [d3d80556] LineSearches v7.4.0
⌃ [5c8ed15e] LinearOperators v2.10.0
⌃ [7ed4a6bd] LinearSolve v3.25.0
  [2ab3a3ac] LogExpFunctions v0.3.29
⌃ [e6f89c97] LoggingExtras v1.1.0
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
⌃ [b8f27783] MathOptInterface v1.42.1
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
⌃ [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.2.0
⌅ [961ee093] ModelingToolkit v10.14.0
  [2e0e35c7] Moshi v0.3.7
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [102ac46a] MultivariatePolynomials v0.5.9
  [6f286f6a] MultivariateStats v0.10.3
  [ffc61752] Mustache v1.0.21
⌃ [d8a4904e] MutableArithmetics v1.6.4
⌃ [a4795742] NLPModels v0.21.5
⌅ [d41bc354] NLSolversBase v7.10.0
  [77ba4419] NaNMath v1.1.3
⌃ [b8a86587] NearestNeighbors v0.4.22
⌃ [8913a72c] NonlinearSolve v4.10.0
⌅ [be0214bd] NonlinearSolveBase v1.13.0
⌃ [5959db7a] NonlinearSolveFirstOrder v1.6.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.7.0
⌃ [26075421] NonlinearSolveSpectralMethods v1.2.0
  [510215fc] Observables v0.5.5
  [6fe1bfb0] OffsetArrays v1.17.0
⌃ [4d8831e6] OpenSSL v1.5.0
⌅ [429524aa] Optim v1.13.2
⌃ [3bd65402] Optimisers v0.4.6
⌅ [7f7a1694] Optimization v4.5.0
⌅ [bca83a33] OptimizationBase v2.10.0
⌃ [fd9f6733] OptimizationMOI v0.5.5
⌅ [064b21be] OptimizationNLPModels v0.0.2
⌃ [36348300] OptimizationOptimJL v0.4.3
⌃ [42dfb2eb] OptimizationOptimisers v0.3.8
  [bac558e1] OrderedCollections v1.8.1
⌅ [bbf590c4] OrdinaryDiffEqCore v1.26.2
⌃ [90014a1f] PDMats v0.11.35
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.3
  [ccf2f8ad] PlotThemes v3.3.0
⌃ [995b91a9] PlotUtils v1.4.3
⌃ [91a5bcdd] Plots v1.40.17
⌃ [e409e4f3] PoissonRandom v0.4.6
  [f517fe37] Polyester v0.7.18
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
⌅ [aea7be01] PrecompileTools v1.2.1
⌃ [21216c6a] Preferences v1.4.3
⌅ [08abe8d2] PrettyTables v2.4.0
  [27ebfcd6] Primes v0.5.7
⌃ [33c8b6b6] ProgressLogging v0.1.5
⌃ [92933f4c] ProgressMeter v1.10.4
  [43287f4e] PtrArrays v1.3.0
  [1fd47b50] QuadGK v2.11.2
  [be4d8f0f] Quadmath v0.5.13
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌃ [731186ca] RecursiveArrayTools v3.36.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
⌃ [ae5879a3] ResettableStacks v1.1.1
⌅ [79098fc4] Rmath v0.8.0
⌃ [7e49a35a] RuntimeGeneratedFunctions v0.5.15
⌃ [9dfe8606] SCCNonlinearSolve v1.4.0
  [94e857df] SIMDTypes v0.1.0
⌅ [0bca4576] SciMLBase v2.107.0
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [19f34311] SciMLJacobianOperators v0.1.8
⌃ [c0aeaf25] SciMLOperators v1.4.0
⌃ [431bcebd] SciMLPublic v1.0.0
⌃ [53ae85a6] SciMLStructures v1.7.0
  [6c6a2e73] Scratch v1.3.0
⌃ [91c51154] SentinelArrays v1.4.8
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.7.0
⌃ [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.2.2
⌃ [9f842d2f] SparseConnectivityTracer v1.0.0
⌃ [0a514795] SparseMatrixColorings v0.4.21
⌃ [276daf66] SpecialFunctions v2.5.1
⌃ [860ef19b] StableRNGs v1.0.3
⌃ [aedffcd0] Static v1.2.0
  [0d7ed370] StaticArrayInterface v1.8.0
⌃ [90137ffa] StaticArrays v1.9.14
⌃ [1e83bf80] StaticArraysCore v1.4.3
⌃ [82ae8749] StatsAPI v1.7.1
⌃ [2913bbd2] StatsBase v0.34.6
⌃ [4c63d2b9] StatsFuns v1.5.0
⌃ [f3b207a7] StatsPlots v0.15.7
⌃ [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
⌃ [892a3eda] StringManipulation v0.4.1
  [856f2bd8] StructTypes v1.11.0
⌃ [2efcf032] SymbolicIndexingInterface v0.3.42
⌅ [19f23fe9] SymbolicLimits v0.2.2
⌅ [d1185830] SymbolicUtils v3.29.0
⌅ [0c5d862f] Symbolics v6.48.1
  [ab02a1b2] TableOperations v1.2.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.1
  [ed4db957] TaskLocalValues v0.1.3
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [5d786b92] TerminalLoggers v0.1.7
  [1c621080] TestItems v1.0.0
  [8290d209] ThreadingUtilities v0.5.5
  [a759f4b9] TimerOutputs v0.5.29
  [3bb67fe8] TranscodingStreams v0.11.3
⌃ [410a4b4d] Tricks v0.1.11
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.6.1
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌅ [1986cc42] Unitful v1.24.0
  [45397f5d] UnitfulLatexify v1.7.0
  [a7c27f48] Unityper v0.1.6
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [897b6980] WeakValueDicts v0.1.0
  [44d3d7a6] Weave v0.10.12
  [cc8bc4a8] Widgets v0.6.7
⌃ [efce3f68] WoodburyMatrices v1.0.0
⌃ [ddb6d928] YAML v0.4.14
⌃ [c2297ded] ZMQ v1.4.1
  [ae81ac8f] ASL_jll v0.1.3+0
⌅ [68821587] Arpack_jll v3.5.1+1
  [6e34b625] Bzip2_jll v1.0.9+0
⌃ [bb5f6f25] CUTEst_jll v2.5.3+0
  [83423d85] Cairo_jll v1.18.5+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.6.5+0
⌅ [b22a6f82] FFMPEG_jll v7.1.1+0
  [f5851436] FFTW_jll v3.3.11+0
⌃ [a3f928ae] Fontconfig_jll v2.16.0+0
  [d7e528f0] FreeType2_jll v2.13.4+0
  [559328eb] FriBidi_jll v1.0.17+0
⌃ [0656b61e] GLFW_jll v3.4.0+2
⌅ [d2c73de3] GR_jll v0.73.17+0
  [b0724c58] GettextRuntime_jll v0.22.4+0
⌃ [f8c6e375] Git_jll v2.50.1+0
⌃ [7746bdde] Glib_jll v2.84.3+0
  [3b182d85] Graphite2_jll v1.3.15+0
  [2e76f6c2] HarfBuzz_jll v8.5.1+0
⌃ [e33a78d0] Hwloc_jll v2.12.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
⌅ [9cc047cb] Ipopt_jll v300.1400.1701+0
⌃ [aacddb02] JpegTurbo_jll v3.1.1+0
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.0.1+0
  [1d63c593] LLVMOpenMP_jll v18.1.8+0
  [dd4b983a] LZO_jll v2.10.3+0
  [81d17ec3] L_BFGS_B_jll v3.0.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
⌃ [4b2f31a3] Libmount_jll v2.41.0+0
⌃ [89763e89] Libtiff_jll v4.7.1+0
⌃ [38a345b3] Libuuid_jll v2.41.0+0
  [d00139f3] METIS_jll v5.1.3+0
  [856f044c] MKL_jll v2025.2.0+0
⌅ [d7ed1dd3] MUMPS_seq_jll v500.800.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
⌅ [656ef2d0] OpenBLAS32_jll v0.3.24+0
⌃ [9bd350c2] OpenSSH_jll v10.0.1+0
⌅ [458c3c95] OpenSSL_jll v3.5.1+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
⌃ [91d4177d] Opus_jll v1.5.2+0
⌃ [36c8627f] Pango_jll v1.56.3+0
⌅ [30392449] Pixman_jll v0.44.2+0
⌃ [c0090381] Qt6Base_jll v6.8.2+1
  [629bc702] Qt6Declarative_jll v6.8.2+1
  [ce943373] Qt6ShaderTools_jll v6.8.2+1
⌃ [e99dba38] Qt6Wayland_jll v6.8.2+1
  [f50d1b31] Rmath_jll v0.5.1+0
⌅ [54dcf436] SIFDecode_jll v2.6.3+0
⌅ [319450e9] SPRAL_jll v2025.5.20+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
⌃ [ffd25f8a] XZ_jll v5.8.1+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.12+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.7+0
⌃ [d091e8ba] Xorg_libXfixes_jll v6.0.1+0
  [a51aa0fd] Xorg_libXi_jll v1.8.3+0
  [d1454406] Xorg_libXinerama_jll v1.1.6+0
  [ec84b674] Xorg_libXrandr_jll v1.5.5+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.3+0
⌃ [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.5+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.44.0+0
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
  [214eeab7] fzf_jll v0.61.1+0
⌃ [a4ae2306] libaom_jll v3.12.1+0
  [0ac62f75] libass_jll v0.17.4+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
⌃ [b53b4c65] libpng_jll v1.6.50+0
  [a9144af2] libsodium_jll v1.0.21+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
⌃ [1317d2d5] oneTBB_jll v2022.0.0+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
⌃ [d8fb68d0] xkbcommon_jll v1.9.2+0
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
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

