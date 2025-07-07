
using BenchmarkTools
using Roots


f(u, p) = u * sin(u) - p;


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


out = zeros(N)
uspan = (0.0, 2.0)

@btime g!(out, ps, uspan);


println("Mean absolute error: $(mean(abs.(f.(out, ps))))")


using NonlinearSolve

function h!(out, ps, uspan, alg)
    for i in 1:N
        prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(f), uspan, ps[i])
        sol = solve(prob, alg)
        out[i] = sol.u
    end
    out
end;


for alg in (Alefeld, NonlinearSolve.Bisection, Brent, Falsi,
            ITP, Muller, Ridder)
    println("Benchmark of $alg:")
    @btime h!($out, $ps, $uspan, $(alg()))
    println("Mean absolute error: $(mean(abs.(f.(out, ps))))\n")
end


g(u) = exp(u) - 1e-15;


function i!(out, uspan)
    for i in 1:N
        out[i] = find_zero(g, uspan)
    end
    out
end

uspan = (-100.0, 0.0)

@btime i!(out, uspan)
println("Mean absolute error: $(mean(abs.(g.(out))))")


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


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])

