
using InteractiveUtils
versioninfo()


Threads.nthreads()


using Plots
plot(rand(10,10))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

