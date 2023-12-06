
using Roots, BenchmarkTools

const N = 100_000;
levels = 1.5 .* rand(N);
out = zeros(N);
myfun(x, lv) = x * sin(x) - lv
function froots(out, levels, u0)
    for i in 1:N
        out[i] = find_zero(x->myfun(x, levels[i]), u0)
    end
end

@btime froots(out, levels, (0, 2))


using NonlinearSolve, BenchmarkTools

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]), ITP()).u
    end
end

function f2(out, levels, u0)
    for i in 1:N
        out[i] = solve(IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]), NonlinearSolve.Bisection()).u
    end
end

function f3(out, levels, u0)
    for i in 1:N
        out[i] = solve(NonlinearProblem{false}(NonlinearFunction{false}(myfun),
                u0, levels[i]), SimpleNewtonRaphson()).u
    end
end

@btime f(out, levels, (0.0, 2.0))
@btime f2(out, levels, (0.0, 2.0))
@btime f3(out, levels, 1.0)


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

