
using BenchmarkTools, Random, VectorizationBase, Statistics
using LinearAlgebra, SparseArrays, LinearSolve, Sparspak
import Pardiso
using Plots
using MatrixDepot

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

# Why do I need to set this ?
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

algs = [
    UMFPACKFactorization(),
    KLUFactorization(),
    MKLPardisoFactorize(),
    SparspakFactorization(),
]
algnames = ["UMFPACK", "KLU", "Pardiso", "Sparspak"]
algnames_transpose = reshape(algnames, 1, length(algnames))

cols = [:red, :blue, :green, :magenta, :turqoise] # one color per alg

matrices = ["HB/1138_bus", "HB/494_bus", "HB/662_bus", "HB/685_bus", "HB/bcsstk01", "HB/bcsstk02", "HB/bcsstk03", "HB/bcsstk04", 
            "HB/bcsstk05", "HB/bcsstk06", "HB/bcsstk07", "HB/bcsstk08", "HB/bcsstk09", "HB/bcsstk10", "HB/bcsstk11", "HB/bcsstk12",
            "HB/bcsstk13", "HB/bcsstk14", "HB/bcsstk15", "HB/bcsstk16"]

times = fill(NaN, length(matrices), length(algs))
percentage_sparsity = fill(NaN, length(matrices))
matrix_size = fill(NaN, length(matrices))


for z in 1:length(matrices)
    try
        rng = MersenneTwister(123)
        A = mdopen(matrices[z]).A
        A = convert(SparseMatrixCSC, A)
        n = size(A, 1)
        matrix_size[z] = n
        percentage_sparsity[z] = length(nonzeros(A)) / n^2
        @info "$n × $n"
        b = rand(rng, n)
        u0 = rand(rng, n)

        for j in 1:length(algs)
            bt = @belapsed solve(prob, $(algs[j])).u setup=(prob = LinearProblem(copy($A),
                copy($b);
                u0 = copy($u0),
                alias_A = true,
                alias_b = true))
            times[z,j] = bt
        end
        p = bar(algnames, times[z, :];
            ylabel = "Time/s",
            yscale = :log10,
            title = "Time on $(matrices[z])",
            legend = :outertopright)
        display(p)
    catch e
        println("$(matrices[z]) failed to factorize.")
        println(e)
    end
end


meantimes = vec(mean(times, dims=1))
p = bar(algnames, meantimes;
    ylabel = "Time/s",
    yscale = :log10,
    title = "Mean factorization time",
    legend = :outertopright)


p = scatter(percentage_sparsity, times;
    ylabel = "Time/s",
    yscale = :log10,
    xlabel = "Percentage Sparsity",
    xscale = :log10,
    label = algnames_transpose,
    title = "Factorization Time vs Percentage Sparsity",
    legend = :outertopright)


p = scatter(matrix_size, times;
    ylabel = "Time/s",
    yscale = :log10,
    xlabel = "Matrix Size",
    xscale = :log10,
    label = algnames_transpose,
    title = "Factorization Time vs Matrix Size",
    legend = :outertopright)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

