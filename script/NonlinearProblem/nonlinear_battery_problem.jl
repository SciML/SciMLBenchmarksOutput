
using NonlinearSolve, MINPACK, NLsolve, LinearSolve, StaticArrays, Sundials
using BenchmarkTools, LinearAlgebra, DiffEqDevTools, CairoMakie
RUS = RadiusUpdateSchemes;


solvers_all = [
    (; name = "Newton Raphson (No Line Search)",                    solver = Dict(:alg => NewtonRaphson()),                                        color = :salmon1,         markershape = :star4),
    (; name = "Newton Raphson (Hager & Zhang Line Search)",         solver = Dict(:alg => NewtonRaphson(linesearch = HagerZhang())),               color = :tomato1,         markershape = :pentagon),
    (; name = "Newton Raphson (More & Thuente Line Search)",        solver = Dict(:alg => NewtonRaphson(linesearch = MoreThuente())),              color = :red3,            markershape = :star6),
    (; name = "Newton Raphson (BackTracking Line Search)",          solver = Dict(:alg => NewtonRaphson(linesearch = BackTracking())),             color = :firebrick,       markershape = :heptagon),
    (; name = "Newton Krylov with GMRES",                           solver = Dict(:alg => NewtonRaphson(; linsolve = KrylovJL_GMRES())),           color = :darkslategray1,  markershape = :utriangle),
    (; name = "Newton Trust Region",                                solver = Dict(:alg => TrustRegion()),                                          color = :deepskyblue1,    markershape = :rect),
    (; name = "Newton Trust Region (NLsolve Radius Update)",        solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.NLsolve)),        color = :cadetblue,       markershape = :diamond),
    (; name = "Newton Trust Region (Nocedal Wright Radius Update)", solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.NocedalWright)),  color = :lightslateblue,  markershape = :hexagon),
    (; name = "Newton Trust Region (Hei Radius Update)",            solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.Hei)),            color = :royalblue2,      markershape = :star1),
    (; name = "Newton Trust Region (Yuan Radius Update)",           solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.Yuan)),           color = :blue1,           markershape = :octagon),
    (; name = "Newton Trust Region (Bastin Radius Update)",         solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.Bastin)),         color = :navy,            markershape = :circle),
    (; name = "Newton Trust Region (Fan Radius Update)",            solver = Dict(:alg => TrustRegion(radius_update_scheme = RUS.Fan)),            color = :blue3,           markershape = :rtriangle),
    (; name = "Levenberg-Marquardt (α_geodesic=0.75)",              solver = Dict(:alg => LevenbergMarquardt()),                                   color = :fuchsia,         markershape = :ltriangle),
    (; name = "Levenberg-Marquardt (α_geodesic, with Cholesky)",    solver = Dict(:alg => LevenbergMarquardt(linsolve = CholeskyFactorization())), color = :orchid4,         markershape = :+),
    (; name = "Modified Powell (MINPACK)",                          solver = Dict(:alg => CMINPACK(method = :hybr)),                               color = :lightgoldenrod2, markershape = :x),
    (; name = "Levenberg-Marquardt (MINPACK)",                      solver = Dict(:alg => CMINPACK(method = :lm)),                                 color = :gold1,           markershape = :dtriangle),
    (; name = "Newton Raphson (NLsolve.jl)",                        solver = Dict(:alg => NLsolveJL(method = :newton)),                            color = :olivedrab1,      markershape = :rtriangle),
    (; name = "Newton Trust Region (NLsolve.jl)",                   solver = Dict(:alg => NLsolveJL()),                                            color = :darkorange,      markershape = :circle),
    (; name = "KINSOL (Sundials)",                                  solver = Dict(:alg => KINSOL()),                                               color = :darkorchid1,     markershape = :star5),
    (; name = "Broyden (No Line Search)",                           solver = Dict(:alg => Broyden()),                                              color = :purple4,         markershape = :star8),
    (; name = "Broyden (True Jacobian + No Line Search)",           solver = Dict(:alg => Broyden(; init_jacobian = Val(:true_jacobian))),         color = :purple2,         markershape = :star2),
];



function f!(out, u, p = nothing)
    out[1] = -u[33]
    out[2] = -u[32]
    out[3] = -u[31]
    out[4] = 1.9876764062810574e10(u[1] + u[4]) - 1.9876764062810574e10u[23]
    out[5] = -u[2] + (-1.5546404484393263e-11exp(-19.460872248113507(-0.4 - u[10] + u[26])))
    out[6] = -1.9876764062810574e10u[14] + 1.9876764062810574e10(u[3] + u[6])
    out[7] = -1.9876764062810574e10u[4] +
             114676.11822324689(-exp(-19.460872248113507(-0.6608489145760508 + u[25])) +
                                exp(19.460872248113507(-0.6608489145760508 + u[25])))
    out[8] = -1.9876764062810574e10u[12] + 1.9876764062810574e10(u[2] + u[5])
    out[9] = -1.9876764062810574e10u[6] +
             114676.1182232469(-exp(-19.460872248113507(-0.6608489145760508 - u[11] +
                                                        u[27])) +
                               exp(19.460872248113507(-0.6608489145760508 - u[11] + u[27])))
    out[10] = -4.284490145672665e10u[19] + 4.284490145672665e10(u[31] + u[7])
    out[11] = -4.284490145672665e10u[21] + 4.284490145672665e10(u[32] + u[8])
    out[12] = -4.284490145672665e10u[22] + 4.284490145672665e10(u[33] + u[9])
    out[13] = 0.025692579121085843(7.680104664733624e7(u[10] - u[11]) +
                                   7.680104664733624e7u[10]) - 1.793773306620288e9u[12]
    out[14] = -u[3] +
              (-1.5546404484393263e-11exp(-19.460872248113507(-0.4 - u[11] + u[27])))
    out[15] = -1.9876764062810574e10u[5] +
              114676.1182232469(-exp(-19.460872248113507(-0.6608489145760506 - u[10] +
                                                         u[26])) +
                                exp(19.460872248113507(-0.6608489145760506 - u[10] +
                                                       u[26])))
    out[16] = 0.025692579121085843(7.680104664733624e7(-u[10] + u[11]) +
                                   1.4529008434739566e8(u[11] - u[13])) -
              1.793773306620288e9u[14]
    out[17] = -1.793773306620288e9u[14] - 1.4404300298567445e9(-u[26] + u[27])
    out[18] = 0.025692579121085843(5.1142109690283257e8(-u[11] + u[13]) +
                                   4.7254130462088e9(u[13] - u[15]))
    out[19] = 0.025692579121085843(4.7254130462088e9(-u[13] + u[15]) +
                                   4.7254130462088e9(u[15] - u[16]))
    out[20] = 0.025692579121085843(4.7254130462088e9(-u[15] + u[16]) +
                                   2.3040372207628164e8(u[16] - u[17]))
    out[21] = 0.025692579121085843(7.200116314883803e7(-u[16] + u[17]) +
                                   3.6900178974461965e7(u[17] - u[18])) -
              2.193876971198113e9u[19]
    out[22] = -4.284490145672665e10u[7] +
              147554.10828979727(-exp(-19.460872248113507(-3.3618450059739535 - u[17] +
                                                          u[28])) +
                                 exp(19.460872248113507(-3.3618450059739535 - u[17] +
                                                        u[28])))
    out[23] = 0.025692579121085843(3.6900178974461965e7(-u[17] + u[18]) +
                                   3.6900178974461965e7(u[18] - u[20])) -
              2.193876971198113e9u[21]
    out[24] = -4.284490145672665e10u[8] +
              147554.10828979727(-exp(-19.460872248113507(-3.3618450059739535 - u[18] +
                                                          u[29])) +
                                 exp(19.460872248113507(-3.3618450059739535 - u[18] +
                                                        u[29])))
    out[25] = 948060.7678835923(-u[18] + u[20]) - 2.193876971198113e9u[22]
    out[26] = -u[1] + (-1.5546404484393263e-11exp(-19.460872248113507(-0.4 + u[25])))
    out[27] = -2.193876971198113e9u[22] +
              0.025692579121085843(-37499.99999999999u[24] -
                                   8.296874999999998e10(-u[29] + u[30]))
    out[28] = -1.793773306620288e9u[23] +
              0.025692579121085843(34090.90909090909u[24] -
                                   5.6064049586776855e10(u[25] - u[26]))
    out[29] = -1.793773306620288e9u[12] +
              0.025692579121085843(-5.6064049586776855e10(-u[25] + u[26]) -
                                   5.6064049586776855e10(u[26] - u[27]))
    out[30] = -2.193876971198113e9u[19] - 2.1316811739525905e9(u[28] - u[29])
    out[31] = -2.193876971198113e9u[21] +
              0.025692579121085843(-8.296874999999998e10(-u[28] + u[29]) -
                                   8.296874999999998e10(u[29] - u[30]))
    out[32] = -4.284490145672665e10u[9] +
              147554.10828979727(-exp(-19.460872248113507(-3.3618450059739535 - u[20] +
                                                          u[30])) +
                                 exp(19.460872248113507(-3.3618450059739535 - u[20] +
                                                        u[30])))
    out[33] = 292.3000724036127 + u[24]
    nothing
end

n = 1
x_sol = [
    -3.889310081682032e-13,
    -5.690845522092043e-13,
    -1.4900105367898274e-12,
    -2.1680981422696e-5,
    -3.284624075480569e-5,
    -8.820027287447222e-5,
    9.53999632159426e-5,
    2.1138249693289567e-5,
    1.1829446876191545e-5,
    0.019709320908045884,
    0.06927785744111935,
    -3.2846241323890243e-5,
    0.13786323434647954,
    -8.820027436448276e-5,
    0.14528607936456214,
    0.15270892438264475,
    0.3049460860584471,
    0.3812355737657502,
    9.53999632159426e-5,
    0.40860971681949443,
    2.1138249693289567e-5,
    1.1829446876191545e-5,
    -2.1680981811627007e-5,
    -292.3000724036127,
    0.5895178515117894,
    0.5896685912243755,
    0.5897784273806014,
    3.837532182598256,
    3.8376303660343676,
    3.837750304468262,
    0.0,
    0.0,
    0.0,
]
x_start = zeros(length(x_sol))
x_start[25:27] .= 0.6608489145760508
x_start[28:30] .= 3.3618450059739433

dict = Dict("n" => n, "start" => x_start, "sol" => x_sol, "title" => "Battery Problem")

testcase = (; prob = NonlinearProblem(f!, dict["start"]), true_sol = dict["sol"])


abstols = 1.0 ./ 10.0 .^ (3:0.5:6)
reltols = 1.0 ./ 10.0 .^ (3:0.5:6);



function check_solver(prob, solver)
    try
        sol = solve(prob.prob, solver.solver[:alg]; abstol = 1e-5, reltol = 1e-5,
            maxiters = 10000)
        err = norm(sol.resid)
        if !SciMLBase.successful_retcode(sol.retcode)
            Base.printstyled("[Warn] Solver $(solver.name) returned retcode $(sol.retcode) with an residual norm = $(norm(sol.resid)).\n";
                color = :red)
            return false
        elseif err > 1e3
            Base.printstyled("[Warn] Solver $(solver.name) had a very large residual (norm = $(norm(sol.resid))).\n";
                color = :red)
            return false
        elseif isinf(err) || isnan(err)
            Base.printstyled("[Warn] Solver $(solver.name) had a residual of $(err).\n";
                color = :red)
            return false
        end
        Base.printstyled("[Info] Solver $(solver.name) successfully solved the problem (norm = $(norm(sol.resid))).\n";
            color = :green)
    catch e
        Base.printstyled("[Warn] Solver $(solver.name) threw an error: $e.\n"; color = :red)
        return false
    end
    return true
end

function generate_wpset(prob, solvers)
    # Finds the solvers that can solve the problem
    successful_solvers = filter(solver -> check_solver(prob, solver), solvers)

    return WorkPrecisionSet(prob.prob, abstols, reltols,
        getfield.(successful_solvers, :solver);
        names = getfield.(successful_solvers, :name), numruns = 50, error_estimate = :l2,
        maxiters = 10000), successful_solvers
end


wp_set, successful_solvers = generate_wpset(testcase, solvers_all)


fig = with_theme(theme_latexfonts(); fontsize = 32) do 
    fig = Figure(; size = (1300, 1000))
    ax = Axis(fig[1, 1], ylabel = L"Time ($s$)", xlabel = L"Error: $f(x^\ast)$ $L_2$-norm",
        xscale = log10, yscale = log10, title = "Battery Problem")

    ls, scs = [], []
    
    for (wp, solver) in zip(wp_set.wps, successful_solvers)
        (; name, times, errors) = wp
        errors = [err.l2 for err in errors]
        l = lines!(ax, errors, times; label = name, linewidth = 3, color = solver.color)
        sc = scatter!(ax, errors, times; label = name, markersize = 16,
            marker = solver.markershape, color = solver.color, strokewidth = 2)
        push!(ls, l)
        push!(scs, sc)
    end

    Legend(fig[2, 1], [[l, sc] for (l, sc) in zip(ls, scs)],
        [solver.name for solver in successful_solvers], position = :ct, color = :white,
        framevisible=false, label = "Solvers", orientation = :horizontal,
        tellwidth = false, tellheight = true, nbanks = 5, labelsize = 20)

    fig
end

fig


save("battery_problem_work_precision.svg", fig)


solver_successes = [(solver in successful_solvers) ? "O" : "X" for solver in solvers_all];


using PrettyTables
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, reshape(solver_successes, 1, :); backend = Val(:html),
    header = getfield.(solvers_all, :name), alignment=:c)
println(io, "```")
Docs.Text(String(take!(io)))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

