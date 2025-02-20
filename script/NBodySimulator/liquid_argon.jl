
using ProgressLogging
using NBodySimulator, OrdinaryDiffEq, StaticArrays
using Plots, DataFrames, StatsPlots

function setup(t)
    T = 120.0 # K
    kb = 1.38e-23 # J/K
    ϵ = T * kb # J
    σ = 3.4e-10 # m
    ρ = 1374 # kg/m^3
    m = 39.95 * 1.6747 * 1e-27 # kg
    N = 350
    L = (m*N/ρ)^(1/3)
    R = 3.5σ
    v_dev = sqrt(kb * T / m) # m/s

    _L = L / σ
    _σ = 1.0
    _ϵ = 1.0
    _m = 1.0
    _v = v_dev / sqrt(ϵ / m)
    _R = R / σ

    bodies = generate_bodies_in_cell_nodes(N, _m, _v, _L)
    lj_parameters = LennardJonesParameters(_ϵ, _σ, _R)
    pbc = CubicPeriodicBoundaryConditions(_L)
    lj_system = PotentialNBodySystem(bodies, Dict(:lennard_jones => lj_parameters));
    simulation = NBodySimulation(lj_system, (0.0, t), pbc, _ϵ/T)

    return simulation
end


function benchmark(energyerr, rts, bytes, allocs, nt, nf, t, configs)
    simulation = setup(t)
    prob = SecondOrderODEProblem(simulation)
    for config in configs
        alg = config.alg
        sol, rt, b, gc, memalloc = @timed solve(prob, alg();
            save_everystep=false, progress=true, progress_name="$alg", config...)
        result = NBodySimulator.SimulationResult(sol, simulation)
        ΔE = total_energy(result, t) - total_energy(result, 0)
        energyerr[alg] = ΔE
        rts[alg] = rt
        bytes[alg] = b
        allocs[alg] = memalloc
        nt[alg] = sol.destats.naccept
        nf[alg] = sol.destats.nf + sol.destats.nf2
    end
end

function run_benchmark!(results, t, integrators, tol...; c=ones(length(integrators)))
    @progress "Benchmark at t=$t" for τ in zip(tol...)
        runtime = Dict()
        ΔE = Dict()
        nt = Dict()
        nf = Dict()
        b = Dict()
        allocs = Dict()
        cfg = config(integrators, c, τ...)

        GC.gc()
        benchmark(ΔE, runtime, b, allocs, nt, nf, t, cfg)
        get_tol(idx) = haskey(cfg[idx], :dt) ? cfg[idx].dt : (cfg[idx].abstol, cfg[idx].rtol)

        for (idx,i) in enumerate(integrators)
            push!(results, [string(i), runtime[i], get_tol(idx)..., abs(ΔE[i]), nt[i], nf[i], c[idx]])
        end
    end
    return results
end


symplectic_integrators = [
    VelocityVerlet,
    VerletLeapfrog,
    PseudoVerletLeapfrog,
    McAte2,
    CalvoSanz4,
    McAte5,
    Yoshida6,
    KahanLi8,
    SofSpa10
];


config(integrators, c, τ) = [ (alg=a, dt=τ*cₐ) for (a,cₐ) in zip(integrators, c)]

t = 35.0
τs = 1e-3

# warmup
c_symplectic = ones(length(symplectic_integrators))
benchmark(Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), 10.,
    config(symplectic_integrators, c_symplectic, τs))

# results = DataFrame(:integrator=>String[], :runtime=>Float64[], :τ=>Float64[],
#     :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
# run_benchmark!(results, t, symplectic_integrators, τs)

# c_symplectic .= results[!, :runtime] ./ results[!, :timesteps]
# c_Verlet = c_symplectic[1]
# c_symplectic /= c_Verlet

c_symplectic = [
    1.00,   # VelocityVerlet
    1.05,   # VerletLeapfrog
    0.98,   # PseudoVerletLeapfrog
    1.02,   # McAte2
    2.38,   # CalvoSanz4
    2.92,   # McAte5
    3.74,   # Yoshida6
    8.44,   # KahanLi8
    15.76   # SofSpa10
]


t = 40.0
τs = 10 .^range(-4, -3, length=10)

results = DataFrame(:integrator=>String[], :runtime=>Float64[], :τ=>Float64[],
    :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
run_benchmark!(results, t, symplectic_integrators, τs, c=c_symplectic)


@df results plot(:EnergyError, :runtime, group=:integrator,
    xscale=:log10, yscale=:log10, xlabel="Energy error", ylabel="Runtime (s)")


@df results plot(:timesteps, :runtime, group=:integrator,
    xscale=:log10, yscale=:log10, xlabel="Number of timesteps", ylabel="Runtime (s)")


function benchmark(energyerr, rts, ts, t, configs)
    simulation = setup(t)
    prob = SecondOrderODEProblem(simulation)
    for config in configs
        alg = config.alg
        sol, rt = @timed solve(prob, alg(); progress=true, progress_name="$alg", config...)
        result = NBodySimulator.SimulationResult(sol, simulation)
        ΔE(t) = total_energy(result, t) - total_energy(result, 0)
        energyerr[alg] = [ΔE(t) for t in sol.t[2:10^2:end]]
        rts[alg] = rt
        ts[alg] = sol.t[2:10^2:end]
    end
end

ΔE = Dict()
rt = Dict()
ts = Dict()
configs = config(symplectic_integrators, c_symplectic, 2.3e-4)
benchmark(ΔE, rt, ts, 40., configs)

plt = plot(xlabel="Rescaled Time", ylabel="Energy error", legend=:bottomleft);
for c in configs
    plot!(plt, ts[c.alg], abs.(ΔE[c.alg]), label="$(c.alg), $(rt[c.alg])s")
end
plt


adaptive_integrators=[
    # Non-stiff ODE methods
    Tsit5,
    Vern7,
    Vern9,
    # DPRKN
    DPRKN6,
    DPRKN8,
    DPRKN12,
];


config(integrators, c, at, rt) = [ (alg=a, abstol=at*2^cₐ, rtol=rt*2^cₐ) for (a,cₐ) in zip(integrators, c)]

t = 35.0
ats = 10 .^range(-14, -4, length=10)
rts = 10 .^range(-14, -4, length=10)

# warmup
c_adaptive = ones(length(adaptive_integrators))
benchmark(Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), 10.,
    config(adaptive_integrators, 1, ats[1], rts[1]))

# results = DataFrame(:integrator=>String[], :runtime=>Float64[], :abstol=>Float64[],
#    :reltol=>Float64[], :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
# run_benchmark!(results, t, adaptive_integrators, ats[1], rts[1])

# c_adaptive .= results[!, :runtime] ./ results[!, :timesteps]
# c_adaptive /= c_Verlet

c_adaptive = [
    3.55,   # Tsit5,
    7.84,   # Vern7,
    11.38,  # Vern9
    3.56,   # DPRKN6,
    5.10,   # DPRKN8,
    8.85    # DPRKN12,
]


t = 40.0

results = DataFrame(:integrator=>String[], :runtime=>Float64[], :abstol=>Float64[],
    :reltol=>Float64[], :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
run_benchmark!(results, t, adaptive_integrators, ats, rts, c=c_adaptive)


@df results plot(:EnergyError, :runtime, group=:integrator,
    xscale=:log10, yscale=:log10, xlabel="Energy error", ylabel="Runtime (s)")


@df results plot(:EnergyError, :f_evals, group=:integrator,
    xscale=:log10, yscale=:log10, xlabel="Energy error", ylabel="Number of f evals")


t = 40.0

symplectic_integrators = [
    VelocityVerlet,
    VerletLeapfrog,
    PseudoVerletLeapfrog,
    McAte2,
    CalvoSanz4
]

c_symplectic = [
    1.00,   # VelocityVerlet
    1.05,   # VerletLeapfrog
    0.98,   # PseudoVerletLeapfrog
    1.02,   # McAte2
    2.38,   # CalvoSanz4
]

results1 = DataFrame(:integrator=>String[], :runtime=>Float64[], :τ=>Float64[],
    :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
run_benchmark!(results1, t, symplectic_integrators, τs, c=c_symplectic)

adaptive_integrators=[
    DPRKN6,
    DPRKN8,
    DPRKN12,
]

c_adaptive = [
    3.56,   # DPRKN6,
    5.10,   # DPRKN8,
    8.85    # DPRKN12,
]

results2 = DataFrame(:integrator=>String[], :runtime=>Float64[], :abstol=>Float64[],
    :reltol=>Float64[], :EnergyError=>Float64[], :timesteps=>Int[], :f_evals=>Int[], :cost=>Float64[]);
run_benchmark!(results2, t, adaptive_integrators, ats, rts, c=c_adaptive)

append!(results1, results2, cols=:union)
results1


@df results1 plot(:EnergyError, :runtime, group=:integrator,
    xscale=:log10, yscale=:log10, xlabel="Energy error", ylabel="Runtime (s)")


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

