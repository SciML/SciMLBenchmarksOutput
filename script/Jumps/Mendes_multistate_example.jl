
using Catalyst, JumpProcesses, JumpProblemLibrary, Plots, Statistics
fmt = :png


jprob = JumpProblemLibrary.prob_jump_multistate
rn  = jprob.network
reactions(rn)


methods = (Direct(), FRM(), SortingDirect(), NRM(), DirectCR(), RSSA(), RSSACR(), Coevolve(), RDirect())
shortlabels = [string(leg)[15:end-2] for leg in methods]
tf      = 10.0 * jprob.tstop
prob    = DiscreteProblem(rn, jprob.u0, (0.0, tf), jprob.rates)
varlegs = ["A_P" "A_bound_P" "A_unbound_P" "RLA_P"]
@variables t S7(t) S8(t) S9(t)
varsyms = [
    [S7,S8,S9],
    [S9],
    [S7,S8],
    [S7]
]
varidxs = []
for vars in varsyms
    push!(varidxs, [findfirst(isequal(sym),unknowns(rn)) for sym in vars])
end


p = []
for (i,method) in enumerate(methods)
    jump_prob = JumpProblem(rn, prob, method, save_positions=(false, false))
    sol = solve(jump_prob, SSAStepper(), saveat=tf/1000.)
    solv = zeros(1001, 4)
    for (i,varidx) in enumerate(varidxs)
        solv[:,i] = sum(sol[varidx,:], dims=1)
    end
    if i < length(methods)
        push!(p, plot(sol.t, solv, title=shortlabels[i], legend=false, format=fmt))
    else
        push!(p, plot(sol.t, solv, title=shortlabels[i], legend=false, format=fmt))
    end
end
push!(p, plot((1:4)', framestyle = :none, legend=:inside, labels=varlegs))
plot(p..., layout=(6,2), format=fmt)


function run_benchmark!(t, jump_prob, stepper)
    sol = solve(jump_prob, stepper)
    @inbounds for i in 1:length(t)
        t[i] = @elapsed (sol = solve(jump_prob, stepper))
    end
end


nsims = 100
benchmarks = Vector{Vector{Float64}}()
for method in methods
    jump_prob = JumpProblem(rn, prob, method, save_positions=(false, false))
    stepper = SSAStepper()
    time = Vector{Float64}(undef, nsims)
    run_benchmark!(time, jump_prob, stepper)
    push!(benchmarks, time)
end


medtimes = Vector{Float64}(undef,length(methods))
stdtimes = Vector{Float64}(undef,length(methods))
avgtimes = Vector{Float64}(undef,length(methods))
for i in 1:length(methods)
    medtimes[i] = median(benchmarks[i])
    avgtimes[i] = mean(benchmarks[i])
    stdtimes[i] = std(benchmarks[i])
end
using DataFrames

df = DataFrame(names=shortlabels, medtimes=medtimes, relmedtimes=(medtimes/medtimes[1]),
               avgtimes=avgtimes, std=stdtimes, cv=stdtimes./avgtimes)

sa = [text(string(round(mt,digits=3),"s"),:center,12) for mt in df.medtimes]
bar(df.names,df.relmedtimes,legend=:false, fmt=fmt)
scatter!(df.names, .05 .+ df.relmedtimes, markeralpha=0, series_annotations=sa, fmt=fmt)
ylabel!("median relative to Direct")
title!("Multistate Model")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

