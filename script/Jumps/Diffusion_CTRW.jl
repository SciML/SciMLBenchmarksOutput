
using Catalyst, JumpProcesses, JumpProblemLibrary, Plots, Statistics, DataFrames


N = 256
h = 1 / N
u0 = 10 * ones(Int64, N)
tf = .01
methods = (Direct(), FRM(), SortingDirect(), NRM(), DirectCR(), RSSA(), RSSACR(), Coevolve(), RDirect())
shortlabels = [string(leg)[15:end-2] for leg in methods]
jprob = JumpProblemLibrary.prob_jump_diffnetwork
rn = jprob.network(N)
prob = DiscreteProblem(rn, u0, (0.0, tf), [1 / (h*h)])
ploth = plot(reuse=false)
for (i,method) in enumerate(methods)
    println("Benchmarking method: ", method)
    jump_prob = JumpProblem(rn, prob, method, save_positions=(false, false))
    sol = solve(jump_prob, SSAStepper(); saveat=tf/1000.)
    plot!(ploth, sol.t, sol[Int(N//2),:], label=shortlabels[i])
end
plot!(ploth, title="Population at middle lattice site", xlabel="time")


function run_benchmark!(t, jump_prob, stepper)
    sol = solve(jump_prob, stepper)
    @inbounds for i in 1:length(t)
        t[i] = @elapsed (sol = solve(jump_prob, stepper))
    end
end


nsims = 50
benchmarks = Vector{Vector{Float64}}()
for method in methods
    jump_prob = JumpProblem(rn, prob, method, save_positions=(false, false))
    stepper = SSAStepper()
    t = Vector{Float64}(undef, nsims)
    run_benchmark!(t, jump_prob, stepper)
    push!(benchmarks, t)
end


medtimes = Vector{Float64}(undef,length(methods))
stdtimes = Vector{Float64}(undef,length(methods))
avgtimes = Vector{Float64}(undef,length(methods))
for i in 1:length(methods)
    medtimes[i] = median(benchmarks[i])
    avgtimes[i] = mean(benchmarks[i])
    stdtimes[i] = std(benchmarks[i])
end

df = DataFrame(names=shortlabels, medtimes=medtimes, relmedtimes=(medtimes/medtimes[1]),
               avgtimes=avgtimes, std=stdtimes, cv=stdtimes./avgtimes)


sa = [string(round(mt,digits=4),"s") for mt in df.medtimes]
bar(df.names, df.relmedtimes, legend=:false)
scatter!(df.names, .05 .+ df.relmedtimes, markeralpha=0, series_annotations=sa)
ylabel!("median relative to Direct")
title!("256 Site 1D Diffusion CTRW")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

