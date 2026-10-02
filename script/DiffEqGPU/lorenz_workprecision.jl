
# GRADSOLVE reads its UTF-8 CUDA templates with the locale encoding; CI runners default to ASCII,
# and Python fixes its encoding when PythonCall starts it.
ENV["LC_ALL"] = "C.UTF-8"
using CUDA, DiffEqGPU, OrdinaryDiffEqVerner, StaticArrays, SciMLBase, CondaPkg
using PythonCall, Plots, Printf, LinearAlgebra, MuladdMacro

@assert CUDA.functional() "This benchmark requires a functional CUDA GPU"
CUDA.versioninfo()
println("Julia ", VERSION, "; DiffEqGPU ", pkgversion(DiffEqGPU))
const KERNEL = EnsembleGPUKernel(CUDA.CUDABackend(), 0.0)
const METHODS = ("GPUTsit5 (PI)", "GPUTsit5 (I)", "GPUTsit5 (PI, vectorized_asolve)", "GRADSOLVE", "Diffrax")

ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV["GRADSOLVE_CUDA_CACHE"] = joinpath(@__DIR__, ".gradsolve_cuda")
ENV["PATH"] = joinpath(CondaPkg.envdir(), "bin") * ":" * ENV["PATH"]
sys = pyimport("sys")
sys.path.insert(0, @__DIR__)
gu = pyimport("gradsolve_utils")
println("JAX devices: ", pyimport("jax").devices())
metadata = pyimport("importlib.metadata")
for package in ("gradsolve", "jax", "jaxlib", "diffrax")
    println(package, " ", metadata.version(package))
end
gr()


@muladd function lorenz_wp(u, p, t)
    T = eltype(u)
    return SVector(
        T(10) * (u[2] - u[1]),
        p[1] * u[1] - u[2] - u[1] * u[3],
        u[1] * u[2] - T(8 / 3) * u[3]
    )
end

# Concatenating SVectors with hcat builds an SMatrix, which is catastrophic to compile at ensemble sizes.
endpoint_matrix(u::AbstractVector{<:SVector{3}}) = [u[i][j] for i in eachindex(u), j in 1:3]

function julia_runner(rhos::Vector{T}, method, tol) where {T}
    prob = ODEProblem{false}(lorenz_wp, SVector{3, T}(1, 0, 0), (zero(T), one(T)), SVector(rhos[1]))
    ens = EnsembleProblem(
        prob; safetycopy = false,
        prob_func = (prob, ctx) -> remake(prob; p = SVector(rhos[ctx.sim_id]))
    )
    alg = method == "GPUTsit5 (PI)" ? GPUTsit5() : GPUTsit5IController()
    return function ()
        sol = CUDA.@sync solve(
            ens, alg, KERNEL; trajectories = length(rhos),
            adaptive = true, dt = T(0.01), reltol = T(tol), abstol = T(tol / 1000),
            save_everystep = false, save_start = false, dense = false
        )
        states = endpoint_matrix([s.u[end] for s in sol.u])
        @assert size(states) == (length(rhos), 3)
        @assert all(isfinite, states)
        return states
    end
end

function measure_julia(run)
    run()
    return minimum([@elapsed(run()) for _ in 1:5])
end

function reference(rhos)
    return map(rhos) do rho
        prob = ODEProblem{false}(lorenz_wp, SVector(1.0, 0.0, 0.0), (0.0, 1.0), SVector(Float64(rho)))
        sol = solve(prob, Vern9(); abstol = 1.0e-14, reltol = 1.0e-14, save_everystep = false)
        @assert SciMLBase.successful_retcode(sol)
        sol.u[end]
    end
end

function julia_kernel_runner(rhos::Vector{T}, method, tol) where {T}
    prob = ODEProblem{false}(lorenz_wp, SVector{3, T}(1, 0, 0), (zero(T), one(T)), SVector(rhos[1]))
    probs = CuArray([DiffEqGPU.make_prob_compatible(remake(prob; p = SVector(rho))) for rho in rhos])
    alg = method == "GPUTsit5 (PI)" ? GPUTsit5() : GPUTsit5IController()
    call() = CUDA.@sync DiffEqGPU.vectorized_asolve(
        probs, prob, alg; dt = T(0.01), reltol = T(tol), abstol = T(tol / 1000),
        save_everystep = false
    )
    function host()
        _, us = call()
        states = endpoint_matrix(Array(us)[end, :])
        @assert size(states) == (length(rhos), 3)
        @assert eltype(states) === T
        @assert all(isfinite, states)
        return states
    end
    return call, host
end

# Host-to-host arm through the low-level API: problem construction, upload,
# solve, download, and result materialization are all inside the timed closure.
function julia_lowlevel_runner(rhos::Vector{T}, tol) where {T}
    prob = ODEProblem{false}(lorenz_wp, SVector{3, T}(1, 0, 0), (zero(T), one(T)), SVector(rhos[1]))
    return function ()
        probs = CuArray([DiffEqGPU.make_prob_compatible(remake(prob; p = SVector(rho))) for rho in rhos])
        _, us = CUDA.@sync DiffEqGPU.vectorized_asolve(
            probs, prob, GPUTsit5(); dt = T(0.01), reltol = T(tol), abstol = T(tol / 1000),
            save_everystep = false
        )
        states = endpoint_matrix(Array(us)[end, :])
        @assert size(states) == (length(rhos), 3)
        @assert eltype(states) === T
        @assert all(isfinite, states)
        return states
    end
end

function measurement(rhos::Vector{T}, method, tol, indices, refs; kernel = false) where {T}
    @info "measurement" T method tol n = length(rhos) kernel
    if method == "GPUTsit5 (PI, vectorized_asolve)"
        @assert !kernel "the low-level host-to-host arm is not a kernel-only method"
        run = julia_lowlevel_runner(rhos, tol)
        seconds = measure_julia(run)
        states = run()
    elseif method in ("GPUTsit5 (PI)", "GPUTsit5 (I)")
        call, run = kernel ? julia_kernel_runner(rhos, method, tol) :
            (r -> (r, r))(julia_runner(rhos, method, tol))
        seconds = measure_julia(call)
        states = run()
    else
        precision = lowercase(string(T))
        call, run = kernel ? gu.prepare_kernel(pylist(rhos), precision, tol) :
            (r -> (r, r))(gu.prepare(pylist(rhos), precision, tol, method))
        seconds = pyconvert(Float64, gu.measure(call))
        states = pyconvert(Matrix{T}, run())
    end
    error = maximum(enumerate(indices)) do (j, i)
        norm(Float64.(states[i, :]) - refs[j]) / max(norm(refs[j]), 1)
    end
    @assert isfinite(error)
    return (; method, precision = string(T), n = length(rhos), tol, seconds, error, kernel)
end


wp = NamedTuple[]
for T in (Float32, Float64)
    rhos = collect(range(zero(T), T(21); length = 8192))
    indices = unique(round.(Int, range(1, length(rhos); length = 257)))
    refs = reference(rhos[indices])
    for (rho, ref) in zip(rhos[indices], refs)
        prob = ODEProblem{false}(lorenz_wp, SVector(1.0, 0.0, 0.0), (0.0, 1.0), SVector(Float64(rho)))
        tighter = solve(prob, Vern9(); abstol = 5.0e-15, reltol = 5.0e-15, save_everystep = false)
        @assert norm(tighter.u[end] - ref) / max(norm(ref), 1) < 1.0e-11
    end
    tolerances = T === Float32 ? 10.0 .^ (-3:-1:-7) : 10.0 .^ (-3:-1:-10)
    for method in METHODS, tol in tolerances
        row = measurement(rhos, method, tol, indices, refs)
        push!(wp, row)
        @printf("%s %s rtol=%.1e error=%.3e time=%.3f ms\n", row.precision, method, tol, row.error, 1000row.seconds)
    end
end


panels = map(("Float32", "Float64")) do precision
    p = plot(;
        xscale = :log10, yscale = :log10, xlabel = "achieved endpoint error",
        ylabel = "host-to-host time (ms)", title = precision, legend = :topleft
    )
    for method in METHODS
        rows = filter(r -> r.method == method && r.precision == precision, wp)
        plot!(
            p, getproperty.(rows, :error), 1000 .* getproperty.(rows, :seconds);
            label = method, marker = :circle
        )
    end
    p
end
plot(panels...; layout = (1, 2), size = (1100, 450))


kernel_n = 2^20
kernel_methods = ("GPUTsit5 (PI)", "GPUTsit5 (I)", "GRADSOLVE")
kernel_wp = NamedTuple[]
for T in (Float32, Float64)
    rhos = collect(range(zero(T), T(21); length = kernel_n))
    indices = unique(round.(Int, range(1, kernel_n; length = 257)))
    refs = reference(rhos[indices])
    tolerances = T === Float32 ? 10.0 .^ (-3:-1:-7) : 10.0 .^ (-4:-1:-9)
    for method in kernel_methods, tol in tolerances
        row = measurement(rhos, method, tol, indices, refs; kernel = true)
        push!(kernel_wp, row)
        @printf("%s %s rtol=%.1e error=%.3e kernel time=%.3f ms\n", row.precision, method, tol, row.error, 1000row.seconds)
    end
end

for T in ("Float32", "Float64")
    at_paper = filter(r -> r.precision == T && r.tol == 1.0e-6, kernel_wp)
    pi_row, gs = (only(filter(r -> r.method == m, at_paper)) for m in ("GPUTsit5 (PI)", "GRADSOLVE"))
    @printf("%s at reltol=1e-6: GRADSOLVE/GPUTsit5 (PI) time ratio %.2f, error ratio %.2f\n", T, gs.seconds / pi_row.seconds, gs.error / pi_row.error)
end


panels = map(("Float32", "Float64")) do precision
    p = plot(;
        xscale = :log10, yscale = :log10, xlabel = "achieved endpoint error",
        ylabel = "kernel time (ms)", title = "$precision, N = 2^20", legend = :topleft
    )
    for method in kernel_methods
        rows = filter(r -> r.method == method && r.precision == precision, kernel_wp)
        plot!(
            p, getproperty.(rows, :error), 1000 .* getproperty.(rows, :seconds);
            label = method, marker = :circle
        )
    end
    p
end
plot(panels...; layout = (1, 2), size = (1100, 450))


scaling = NamedTuple[]
for T in (Float32, Float64)
    ceiling = T === Float32 ? 1.0e-4 : 1.0e-7
    for n in 2 .^ (11:2:21)
        rhos = collect(range(zero(T), T(21); length = n))
        indices = unique(round.(Int, range(1, n; length = 257)))
        refs = reference(rhos[indices])
        for method in METHODS
            eligible = filter(r -> r.method == method && r.precision == string(T) && r.error <= ceiling, wp)
            @assert !isempty(eligible) "No tolerance meets the requested accuracy for $method, $T"
            seed = eligible[argmin(getproperty.(eligible, :seconds))].tol
            candidates = [measurement(rhos, method, seed * factor, indices, refs) for factor in (0.1, 1.0, 10.0)]
            valid = filter(r -> r.error <= ceiling, candidates)
            @assert !isempty(valid) "Accuracy target missed at N=$n for $method, $T"
            best = valid[argmin(getproperty.(valid, :seconds))]
            push!(scaling, best)
            @printf("%s %s N=%d rtol=%.1e error=%.3e time=%.3f ms\n", best.precision, method, n, best.tol, best.error, 1000best.seconds)
        end
    end
end


panels = map(("Float32", "Float64")) do precision
    p = plot(;
        xscale = :log10, yscale = :log10, xlabel = "trajectories",
        ylabel = "host-to-host time (ms)", title = precision, legend = :topleft
    )
    for method in METHODS
        rows = filter(r -> r.method == method && r.precision == precision, scaling)
        plot!(
            p, getproperty.(rows, :n), 1000 .* getproperty.(rows, :seconds);
            label = method, marker = :circle
        )
    end
    p
end
plot(panels...; layout = (1, 2), size = (1100, 450))


let T = Float64, n = 2^21, tol = 1.0e-7
    rhos = collect(range(zero(T), T(21); length = n))
    prob = ODEProblem{false}(lorenz_wp, SVector{3, T}(1, 0, 0), (zero(T), one(T)), SVector(rhos[1]))
    construct() = [DiffEqGPU.make_prob_compatible(remake(prob; p = SVector(rho))) for rho in rhos]
    cpu_probs = construct()
    t_construct = minimum(@elapsed(construct()) for _ in 1:5)
    probs = CuArray(cpu_probs)
    t_upload = minimum(@elapsed(CUDA.@sync(CuArray(cpu_probs))) for _ in 1:5)
    asolve() = DiffEqGPU.vectorized_asolve(
        probs, prob, GPUTsit5(); dt = T(0.01), reltol = T(tol), abstol = T(tol / 1000),
        save_everystep = false
    )
    _, us = CUDA.@sync asolve()
    t_solve = minimum(@elapsed(CUDA.@sync asolve()) for _ in 1:5)
    download() = endpoint_matrix(Array(us)[end, :])
    download()
    t_download = minimum(@elapsed(download()) for _ in 1:5)
    t_highlevel = measure_julia(julia_runner(rhos, "GPUTsit5 (PI)", tol))
    @printf("N = 2^21, Float64, reltol = 1e-7 stage times (min of 5):\n")
    @printf("  CPU problem construction      %.3f ms\n", 1000t_construct)
    @printf("  CuArray upload                %.3f ms\n", 1000t_upload)
    @printf("  vectorized_asolve (synced)    %.3f ms\n", 1000t_solve)
    @printf("  endpoint download + matrix    %.3f ms\n", 1000t_download)
    @printf("  high-level solve total        %.3f ms\n", 1000t_highlevel)
end

