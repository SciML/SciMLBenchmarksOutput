
using InteractiveUtils
versioninfo()


Threads.nthreads()


using Plots
plot(rand(20,20))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

