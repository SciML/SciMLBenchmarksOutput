---
author: "Guilherme Zagatti"
title: "Multivariate Hawkes Model"
---
```julia
using JumpProcesses, Graphs, Statistics, BenchmarkTools, Plots
using OrdinaryDiffEq: Tsit5
fmt = :png
width_px, height_px = default(:size);
```




# Model and example solutions

Let a graph with ``V`` nodes, then the multivariate Hawkes process is characterized by ``V`` point processes such that the conditional intensity rate of node ``i`` connected to a set of nodes ``E_i`` in the graph is given by:
```math
  \lambda_i^\ast (t) = \lambda + \sum_{j \in E_i} \sum_{t_{n_j} < t} \alpha \exp \left[-\beta (t - t_{n_j}) \right]
```
This process is known as self-exciting, because the occurrence of an event ``j`` at ``t_{n_j}`` will increase the conditional intensity of all the processes connected to it by ``\alpha``. The excited intensity then decreases at a rate proportional to ``\beta``.

The conditional intensity of this process has a recursive formulation which can significantly speed the simulation. The recursive formulation for the univariate case is derived in Laub et al. [2]. We derive the compound case here. Let ``t_{N_i} = \max \{ t_{n_j} < t \mid j \in E_i \}`` and
```math
\begin{split}
  \phi_i^\ast (t)
    &= \sum_{j \in E_i} \sum_{t_{n_j} < t} \alpha \exp \left[-\beta (t - t_{N_i} + t_{N_i} - t_{n_j}) \right] \\
    &= \exp \left[ -\beta (t - t_{N_i}) \right] \sum_{j \in E_i} \sum_{t_{n_j} \leq t_{N_i}} \alpha \exp \left[-\beta (t_{N_i} - t_{n_j}) \right] \\
    &= \exp \left[ -\beta (t - t_{N_i}) \right] \left( \alpha + \phi^\ast (t_{N_i}) \right)
\end{split}
```
Then the conditional intensity can be re-written in terms of ``\phi_i^\ast (t_{N_i})``
```math
  \lambda_i^\ast (t) = \lambda + \phi_i^\ast (t) = \lambda + \exp \left[ -\beta (t - t_{N_i}) \right] \left( \alpha + \phi_i^\ast (t_{N_i}) \right)
```

In Julia, we define a factory for the conditional intensity ``\lambda_i`` which returns the brute-force or recursive versions of the intensity given node ``i`` and network ``g``.

```julia
function hawkes_rate(i::Int, g; use_recursion = false)

    @inline @inbounds function rate_recursion(u, p, t)
        λ, α, β, h, urate, ϕ = p
        urate[i] = λ + exp(-β*(t - h[i]))*ϕ[i]
        return urate[i]
    end

    @inline @inbounds function rate_brute(u, p, t)
        λ, α, β, h, urate = p
        x = zero(typeof(t))
        for j in g[i]
            for _t in reverse(h[j])
                ϕij = α * exp(-β * (t - _t))
                if ϕij ≈ 0
                    break
                end
                x += ϕij
            end
        end
        urate[i] = λ + x
        return urate[i]
    end

    if use_recursion
        return rate_recursion
    else
        return rate_brute
    end

end
```

```
hawkes_rate (generic function with 1 method)
```





Given the rate factory, we can create a jump factory which will create all the jumps in our model.

```julia
function hawkes_jump(i::Int, g; use_recursion = false)
    rate = hawkes_rate(i, g; use_recursion)
    urate = rate
    @inbounds rateinterval(u, p, t) = p[5][i] == p[1] ? typemax(t) : 2 / p[5][i]
    @inbounds lrate(u, p, t) = p[1]
    @inbounds function affect_recursion!(integrator)
        λ, α, β, h, _, ϕ  = integrator.p
        for j in g[i]
          ϕ[j] *= exp(-β*(integrator.t - h[j]))
          ϕ[j] += α
          h[j] = integrator.t
        end
        integrator.u[i] += 1
    end
    @inbounds function affect_brute!(integrator)
        push!(integrator.p[4][i], integrator.t)
        integrator.u[i] += 1
    end
    return VariableRateJump(
        rate,
        use_recursion ? affect_recursion! : affect_brute!;
        lrate,
        urate,
        rateinterval,
    )
end

function hawkes_jump(u, g; use_recursion = false)
    return [hawkes_jump(i, g; use_recursion) for i = 1:length(u)]
end
```

```
hawkes_jump (generic function with 2 methods)
```





We can then create a factory for Multivariate Hawkes `JumpProblem`s. We can define two types of `JumpProblem`s depending on the aggregator. The `Direct()` aggregator expects an `ODEProblem` since it cannot handle the `SSAStepper` with `VariableRateJump`s.


```julia
function f!(du, u, p, t)
    du .= 0
    nothing
end

function hawkes_problem(
    p,
    agg;
    u = [0.0],
    tspan = (0.0, 50.0),
    save_positions = (false, true),
    g = [[1]],
    use_recursion = false,
)
    oprob = ODEProblem(f!, u, tspan, p)
    jumps = hawkes_jump(u, g; use_recursion)
    jprob = JumpProblem(oprob, agg, jumps...; save_positions = save_positions)
    return jprob
end
```

```
hawkes_problem (generic function with 1 method)
```





The `Coevolve()` aggregator knows how to handle the `SSAStepper`, so it accepts a `DiscreteProblem`.

```julia
function hawkes_problem(
    p,
    agg::Coevolve;
    u = [0.0],
    tspan = (0.0, 50.0),
    save_positions = (false, true),
    g = [[1]],
    use_recursion = false,
)
    dprob = DiscreteProblem(u, tspan, p)
    jumps = hawkes_jump(u, g; use_recursion)
    jprob =
        JumpProblem(dprob, agg, jumps...; dep_graph = g, save_positions = save_positions)
    return jprob
end
```

```
hawkes_problem (generic function with 2 methods)
```





Lets solve the problems defined so far. We sample a random graph sampled from the Erdős-Rényi model. This model assumes that the probability of an edge between two nodes is independent of other edges, which we fix at ``0.2``. For illustration purposes, we fix ``V = 10``.

```julia
V = 10
G = erdos_renyi(V, 0.2, seed = 9103)
g = [neighbors(G, i) for i = 1:nv(G)]
```

```
10-element Vector{Vector{Int64}}:
 [4, 7]
 [8, 9]
 [4, 5]
 [1, 3]
 [3]
 []
 [1, 8, 9]
 [2, 7]
 [2, 7, 10]
 [9]
```





We fix the Hawkes parameters at ``\lambda = 0.5 , \alpha = 0.1 , \beta = 2.0`` which ensures the process does not explode.

```julia
tspan = (0.0, 50.0)
u = [0.0 for i = 1:nv(G)]
p = (0.5, 0.1, 2.0)
```

```
(0.5, 0.1, 2.0)
```





Now, we instantiate the problems, find their solutions and plot the results.


```julia
algorithms = Tuple{Any, Any, Bool, String}[
  (Direct(), Tsit5(), false, "Direct (brute-force)"),
  (Coevolve(), SSAStepper(), false, "Coevolve (brute-force)"),
  (Direct(), Tsit5(), true, "Direct (recursive)"),
  (Coevolve(), SSAStepper(), true, "Coevolve (recursive)"),
]

let fig = []
  for (i, (algo, stepper, use_recursion, label)) in enumerate(algorithms)
    @info label
    if use_recursion
        h = zeros(eltype(tspan), nv(G))
        urate = zeros(eltype(tspan), nv(G))
        ϕ = zeros(eltype(tspan), nv(G))
        _p = (p[1], p[2], p[3], h, ϕ, urate)
    else
        h = [eltype(tspan)[] for _ = 1:nv(G)]
        urate = zeros(eltype(tspan), nv(G))
        _p = (p[1], p[2], p[3], h, urate)
    end
    jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
    sol = solve(jump_prob, stepper)
    push!(fig, plot(sol.t, sol[1:V, :]', title=label, legend=false, format=fmt))
  end
  fig = plot(fig..., layout=(2,2), format=fmt, size=(width_px, 2*height_px/2))
end
```

![](figures/MultivariateHawkes_8_1.png)



## Alternative libraries

We benchmark `JumpProcesses.jl` against `PiecewiseDeterministicMarkovProcesses.jl` and Python `Tick` library.

In order to compare with the `PiecewiseDeterministicMarkovProcesses.jl`, we need to reformulate our jump problem as a Piecewise Deterministic Markov Process (PDMP). In this setting, we have two options.

The simple version only requires the conditional intensity. Like above, we define a brute-force and recursive approach. Following the library's specification we define the following functions.

```julia
function hawkes_rate_simple_recursion(rate, xc, xd, p, t, issum::Bool)
  λ, _, β, h, ϕ, g = p
  for i in 1:length(g)
    rate[i] = λ + exp(-β * (t - h[i])) * ϕ[i]
  end
  if issum
    return sum(rate)
  end
  return 0.0
end

function hawkes_rate_simple_brute(rate, xc, xd, p, t, issum::Bool)
  λ, α, β, h, g = p
  for i in 1:length(g)
    x = zero(typeof(t))
    for j in g[i]
        for _t in reverse(h[j])
            ϕij = α * exp(-β * (t - _t))
            if ϕij ≈ 0
                break
            end
            x += ϕij
        end
    end
    rate[i] = λ + x
  end
  if issum
    return sum(rate)
  end
  return 0.0
end

function hawkes_affect_simple_recursion!(xc, xd, p, t, i::Int64)
  _, α, β, h, ϕ, g = p
  for j in g[i]
      ϕ[j] *= exp(-β * (t - h[j]))
      ϕ[j] += α
      h[j] = t
  end
end

function hawkes_affect_simple_brute!(xc, xd, p, t, i::Int64)
  push!(p[4][i], t)
end
```

```
hawkes_affect_simple_brute! (generic function with 1 method)
```





Since this is a library for PDMP, we also need to define the ODE problem. In the simple version, we simply set it to zero.

```julia
function hawkes_drate_simple(dxc, xc, xd, p, t)
    dxc .= 0
end
```

```
hawkes_drate_simple (generic function with 1 method)
```





Next, we create a factory for the Multivariate Hawkes `PDMPCHVSimple` problem.

```julia
import LinearAlgebra: I
using PiecewiseDeterministicMarkovProcesses
const PDMP = PiecewiseDeterministicMarkovProcesses

struct PDMPCHVSimple end

function hawkes_problem(p,
                        agg::PDMPCHVSimple;
                        u = [0.0],
                        tspan = (0.0, 50.0),
                        save_positions = (false, true),
                        g = [[1]],
                        use_recursion = true)
    xd0 = Array{Int}(u)
    xc0 = copy(u)
    nu = one(eltype(xd0)) * I(length(xd0))
    if use_recursion
      jprob = PDMPProblem(hawkes_drate_simple, hawkes_rate_simple_recursion,
          hawkes_affect_simple_recursion!, nu, xc0, xd0, p, tspan)
    else
      jprob = PDMPProblem(hawkes_drate_simple, hawkes_rate_simple_brute,
          hawkes_affect_simple_brute!, nu, xc0, xd0, p, tspan)
    end
    return jprob
end

push!(algorithms, (PDMPCHVSimple(), CHV(Tsit5()), false, "PDMPCHVSimple (brute-force)"));
push!(algorithms, (PDMPCHVSimple(), CHV(Tsit5()), true, "PDMPCHVSimple (recursive)"));
```




The full version requires that we describe how the conditional intensity changes with time which we derive below:
```math
\begin{split}
  \frac{d \lambda_i^\ast (t)}{d t}
    &= -\beta \sum_{j \in E_i} \sum_{t_{n_j} < t} \alpha \exp \left[-\beta (t - t_{n_j}) \right] \\
    &= -\beta \left( \lambda_i^\ast (t) - \lambda \right)
\end{split}
```

```julia
function hawkes_drate_full(dxc, xc, xd, p, t)
    λ, α, β, _, _, g = p
    for i = 1:length(g)
        dxc[i] = -β * (xc[i] - λ)
    end
end
```

```
hawkes_drate_full (generic function with 1 method)
```





Next, we need to define the intensity rate and the jumps according to library's specification.

```julia
function hawkes_rate_full(rate, xc, xd, p, t, issum::Bool)
    λ, α, β, _, _, g = p
    if issum
        return sum(@view(xc[1:length(g)]))
    end
    rate[1:length(g)] .= @view xc[1:length(g)]
    return 0.0
end

function hawkes_affect_full!(xc, xd, p, t, i::Int64)
    λ, α, β, _, _, g = p
    for j in g[i]
        xc[i] += α
    end
end
```

```
hawkes_affect_full! (generic function with 1 method)
```





Finally, we create a factory for the Multivariate Hawkes `PDMPCHVFull` problem.

```julia
struct PDMPCHVFull end

function hawkes_problem(
    p,
    agg::PDMPCHVFull;
    u = [0.0],
    tspan = (0.0, 50.0),
    save_positions = (false, true),
    g = [[1]],
    use_recursion = true,
)
    xd0 = Array{Int}(u)
    xc0 = [p[1] for i = 1:length(u)]
    nu = one(eltype(xd0)) * I(length(xd0))
    jprob = PDMPProblem(hawkes_drate_full, hawkes_rate_full, hawkes_affect_full!, nu, xc0, xd0, p, tspan)
    return jprob
end

push!(algorithms, (PDMPCHVFull(), CHV(Tsit5()), true, "PDMPCHVFull"));
```




The Python `Tick` library can be accessed with the `PyCall.jl`. We install the required Python dependencies with `Conda.jl` and define a factory for the Multivariate Hawkes `PyTick` problem.

```julia
const BENCHMARK_PYTHON::Bool = tryparse(Bool, get(ENV, "SCIMLBENCHMARK_PYTHON", "true"))
const REBUILD_PYCALL::Bool = tryparse(Bool, get(ENV, "SCIMLBENCHMARK_REBUILD_PYCALL", "true"))

struct PyTick end

if BENCHMARK_PYTHON
  if REBUILD_PYCALL
    using Pkg, Conda

    # PyCall only works with Conda.ROOTENV
    # tick requires python=3.8
    Conda.add("python=3.8", Conda.ROOTENV)
    Conda.add("numpy", Conda.ROOTENV)
    Conda.pip_interop(true, Conda.ROOTENV)
    Conda.pip("install", "tick", Conda.ROOTENV)

    # rebuild PyCall to ensure it links to the python provided by Conda.jl
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
  end

  ENV["PYTHON"] = ""
  using PyCall
  @info "PyCall" PyCall.libpython PyCall.pyversion PyCall.conda

  function hawkes_problem(
      p,
      agg::PyTick;
      u = [0.0],
      tspan = (0.0, 50.0),
      save_positions = (false, true),
      g = [[1]],
      use_recursion = true,
  )
      λ, α, β = p
      SimuHawkesSumExpKernels = pyimport("tick.hawkes")[:SimuHawkesSumExpKernels]
      jprob = SimuHawkesSumExpKernels(
          baseline = fill(λ, length(u)),
          adjacency = [i in j ? α / β : 0.0 for j in g, i = 1:length(u), u = 1:1],
          decays = [β],
          end_time = tspan[2],
          verbose = false,
          force_simulation = true,
      )
      return jprob
  end

  push!(algorithms, (PyTick(), nothing, true, "PyTick"));
end
```

```
Channels:
 - conda-forge
Platform: linux-64
Collecting package metadata (repodata.json): ...working... done
Solving environment: ...working... failed
Error: failed process: Process(setenv(`/cache/julia-buildkite-plugin/depots
/5b300254-1738-4989-ae0a-f4d2d937f953/conda/3/x86_64/bin/conda install -q -
y python=3.8`,["DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/21581/bus", "B
UILDKITE_PULL_REQUEST_REPO=", "BUILDKITE_SOURCE=webhook", "BUILDKITE_PLUGIN
_COPPERMIND_INPUTS_0=benchmarks/Jumps", "BUILDKITE_GIT_CLONE_FLAGS=-v", "BU
ILDKITE_PLUGIN_CRYPTIC_BASE64_AGENT_PUBLIC_KEY_SECRET=LS0tLS1CRUdJTiBQVUJMS
UMgS0VZLS0tLS0KTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUF0
WHNKMzFGbTFKN29IYzlveGZaWQpKY3FxRk5yaXRMUUhxaDVJUHNGS3YySis1K1FVQkZNYURjMHI
3czZ3NDNSMDFobkVNT1lYNDAreUVDT3h5bHErClo3dHdxWlNxS2U1MThwc0RyeWRna0xJNzRnQU
VZWWNTZGdvTGt4YWpWNy9rb0hFTDgrczRKdFRVNUJ6d1RFdXAKTllTZGNQOFhQSmJLekY1RE5qd
WJmeFA5ZjdSN2x6SUx2NWl2Z2lxZTVtbUxGd1lwb0hTRVFVNXRlT09IQStLYwpjUDZ3K2d1Q0Vx
MUZFb0N2MDRyaTFXaWpVZXorMytEWVM4UCtROGRxMGJYUWZUS1Vyc0thMkdnLzVmZ0h5Z0R1CmR
HT2ZsdzUvVEljR3VVbGNsd1hZb2tTRkpSWUJFa2pUOXBCZ2JNNEcyL2tXNGFmZ3d4bHNuN3VsUW
5QNDZVLzEKZFFJREFRQUIKLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0tCg==", "BUILDKITE_ENV
_FILE=/tmp/job-env-01932d42-ae19-478f-af10-cfb95d0cee8c807275329", "BUILDKI
TE_BUILD_NUMBER=3000", "BUILDKITE_GIT_CLONE_MIRROR_FLAGS=-v", "BUILDKITE_AG
ENT_DEBUG=false", "BUILDKITE_AGENT_META_DATA_QUEUE=juliaecosystem", "BUILDK
ITE_PLUGINS_PATH=/etc/buildkite-agent/plugins", "BUILDKITE_COMMAND_EVAL=tru
e", "BUILDKITE_AGENT_META_DATA_SANDBOX_CAPABLE=true", "BUILDKITE_ORGANIZATI
ON_SLUG=julialang", "BUILDKITE_PLUGIN_COPPERMIND_INPUTS_1=src/**/*.jl", "BU
ILDKITE_PIPELINE_PROVIDER=github", "BUILDKITE_AGENT_EXPERIMENT=resolve-comm
it-after-checkout", "BUILDKITE_CONFIG_PATH=/etc/buildkite-agent/buildkite-a
gent.cfg", "BUILDKITE_PIPELINE_TEAMS=sciml-full-access", "BUILDKITE_AGENT_M
ETA_DATA_CRYPTIC_CAPABLE=true", "BUILDKITE_AGENT_ACCESS_TOKEN=bkaj_eyJhbGci
OiJIUzUxMiJ9.eyJzdWIiOiIwMTkzMmQ0Mi1hZTE5LTQ3OGYtYWYxMC1jZmI5NWQwY2VlOGMiLC
JhdWQiOiIwMTkzMmYyZC05NjhhLTRlMjYtODgyYS1hNjRiZjYwN2YyMzAiLCJpc3MiOiJidWlsZ
GtpdGUiLCJleHAiOjE3MzIyNzE3NDcsImlhdCI6MTczMTY2MzM0N30.slFiHXyNm5R8F-Eik5Rq
ZMlqAfHjIaI66jUxiko8ldu-vm-mz1RunvKFbnUPU4KN1As4jbLN-xCDKUlTokfqvQ", "BUILD
KITE_PLUGIN_CRYPTIC_BASE64_SIGNED_JOB_ID_SECRET=sMIUBr9e+mz0YMtLMQxF5nug7gh
Cj0G5iA+CrR6WGq1/Lu3az6MQZayKbS/4ROow2warRS1NICiy+TA4pX1ucmfopWSeSIOsg8GnEV
5qV3rUVJb9TlxqTaT3tY2929rpIG/vsDr/lBFBSlRk5JeONpkdi9CX5u37EkJ1TxJrUIg7nkdvY
vs3YBTw+11axXLPa5Iv3wOBskz0uVlpYwM0nPiZ3AYL3HaTUdvVYxgysRtAz1CgWoJYgW86X1aD
18/gP6SzYa5AGMJ2tuH8V895voztE4SttpFXdE+k2K2zZMllLRpvSazN5973wUYkYyQkGOK1lf1
Z3bg94r2fTYJU2w==", "BUILDKITE_PLUGIN_COPPERMIND_S3_PREFIX=s3://julialang-b
uildkite-artifacts/scimlbenchmarks", "BUILDKITE_BUILD_CREATOR_EMAIL=account
s@chrisrackauckas.com", "XKB_CONFIG_ROOT=/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f8b49c7c45b400e3f5c4002d1
9645d4b88712c0c/share/X11/xkb", "BUILDKITE_SSH_KEYSCAN=true", "BUILDKITE_PR
OJECT_SLUG=julialang/scimlbenchmarks-dot-jl", "BUILDKITE_INITIAL_JOB_ID=019
32d40-f44c-4e50-972b-d911ec64b64e", "BUILDKITE_BIN_PATH=/usr/bin", "PWD=/ca
che/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-jl", "GRDIR=/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/231c36cbc48a78caf7818ee7f4cd260eb3d642e3", "BUILDKITE_GIT_CHECKOUT_FLA
GS=-f", "BUILDKITE_COMPUTE_TYPE=self-hosted", "BUILDKITE_GIT_SUBMODULES=tru
e", "BUILDKITE_AGENT_META_DATA_OS=linux", "CI=true", "BUILDKITE_STEP_KEY=be
nchmark-benchmarks-Jumps", "BUILDKITE_STEP_IDENTIFIER=benchmark-benchmarks-
Jumps", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_0=markdown/**/figures/*.png", 
"CONDA_PREFIX=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-
f4d2d937f953/conda/3/x86_64", "BUILDKITE_PLUGIN_COPPERMIND_INPUTS_2=./*.tom
l", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_3=markdown/**/*.svg", "BUILDKITE_A
GENT_PID=2", "BUILDKITE_LAST_HOOK_EXIT_STATUS=0", "BUILDKITE_AGENT_META_DAT
A_EXCLUSIVE=true", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_6=script/**/*.jl", 
"BUILDKITE_BUILD_CREATOR=Christopher Rackauckas", "BUILDKITE_REBUILT_FROM_B
UILD_ID=", "OPENBLAS_DEFAULT_NUM_THREADS=1", "BUILDKITE_BRANCH=master", "JU
LIA_DEPOT_PATH=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a
-f4d2d937f953", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_5=pdf/**/*.pdf", "BUIL
DKITE_AGENT_DEBUG_HTTP=false", "SHELL=/shells/bash", "BUILDKITE=true", "BUI
LDKITE_PLUGIN_CRYPTIC_PRIVILEGED=true", "BUILDKITE_HOOKS_PATH=/hooks", "BUI
LDKITE_PIPELINE_DEFAULT_BRANCH=master", "BUILDKITE_PLUGIN_NAME=COPPERMIND",
 "BUILDKITE_REBUILT_FROM_BUILD_NUMBER=", "HOME=/root", "BUILDKITE_PLUGIN_CO
PPERMIND_OUTPUTS_4=notebook/**/*.ipynb", "BUILDKITE_S3_DEFAULT_REGION=us-ea
st-1", "BUILDKITE_TRIGGERED_FROM_BUILD_PIPELINE_SLUG=", "BUILDKITE_BUILD_CH
ECKOUT_PATH=/cache/build/exclusive-amdci1-0/julialang/scimlbenchmarks-dot-j
l", "BUILDKITE_SCRIPT_PATH=# Clear out these secrets as they're not needed 
during the actual build\nBUILDKITE_S3_ACCESS_KEY_ID=\"\" BUILDKITE_S3_SECRE
T_ACCESS_KEY=\"\" ./.buildkite/build_benchmark.sh \"benchmarks/Jumps\"\n", 
"INVOCATION_ID=43c7087b30204514a938e3477988f18f", "BUILDKITE_PLUGIN_COPPERM
IND_OUTPUTS_2=markdown/**/*.pdf", "LANG=en_US.UTF-8", "BUILDKITE_PIPELINE_N
AME=SciMLBenchmarks.jl", "SHLVL=3", "XDG_RUNTIME_DIR=/run/user/21581", "OLD
PWD=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f9
53", "FONTCONFIG_FILE=/cache/julia-buildkite-plugin/depots/5b300254-1738-49
89-ae0a-f4d2d937f953/artifacts/558980a93131f08be5335521b84e137ee3172296/etc
/fonts/fonts.conf", "BUILDKITE_PLUGIN_CONFIGURATION={\"inputs\":[\"benchmar
ks/Jumps\",\"src/**/*.jl\",\"./*.toml\"],\"outputs\":[\"markdown/**/figures
/*.png\",\"markdown/**/*.md\",\"markdown/**/*.pdf\",\"markdown/**/*.svg\",\
"notebook/**/*.ipynb\",\"pdf/**/*.pdf\",\"script/**/*.jl\"],\"s3_prefix\":\
"s3://julialang-buildkite-artifacts/scimlbenchmarks\"}", "BUILDKITE_BUILD_P
ATH=/cache/build", "BUILDKITE_BUILD_AUTHOR_EMAIL=accounts@chrisrackauckas.c
om", "BUILDKITE_TIMEOUT=10080", "BUILDKITE_GIT_MIRRORS_PATH=/cache/repos", 
"BUILDKITE_STRICT_SINGLE_HOOKS=false", "BUILDKITE_LABEL=:hammer: benchmarks
/Jumps", "FONTCONFIG_PATH=/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/558980a93131f08be5335521b84e137ee3172296
/etc/fonts", "JOURNAL_STREAM=8:666003079", "BUILDKITE_AGENT_META_DATA_SANDB
OX_JL=true", "BUILDKITE_BUILD_CREATOR_TEAMS=juliagpu-full-access:sciml-full
-access", "BUILDKITE_PROJECT_PROVIDER=github", "QT_ACCESSIBILITY=1", "BUILD
KITE_S3_SECRET_ACCESS_KEY=", "GKS_USE_CAIRO_PNG=true", "BUILDKITE_REPO_MIRR
OR=/cache/repos/https---github-com-SciML-SciMLBenchmarks-jl-git", "GIT_TERM
INAL_PROMPT=0", "BUILDKITE_JOB_ID=01932d42-ae19-478f-af10-cfb95d0cee8c", "S
YSTEMD_EXEC_PID=3972767", "BUILDKITE_REDACTED_VARS=*_PASSWORD,*_SECRET,*_TO
KEN,*_PRIVATE_KEY,*_ACCESS_KEY,*_SECRET_KEY,*_CONNECTION_STRING", "BUILDKIT
E_BUILD_AUTHOR=Christopher Rackauckas", "USER=sabae", "GKSwstype=100", "BUI
LDKITE_REPO=https://github.com/SciML/SciMLBenchmarks.jl.git", "BUILDKITE_GI
T_FETCH_FLAGS=-v --prune --tags", "BUILDKITE_LOCAL_HOOKS_ENABLED=true", "MA
NAGERPID=59175", "BUILDKITE_S3_ACCESS_KEY_ID=", "BUILDKITE_GIT_MIRRORS_LOCK
_TIMEOUT=300", "BUILDKITE_AGENT_ID=01932f2d-968a-4e26-882a-a64bf607f230", "
BUILDKITE_GIT_MIRRORS_SKIP_UPDATE=false", "BUILDKITE_ARTIFACT_PATHS=", "BUI
LDKITE_BUILD_URL=https://buildkite.com/julialang/scimlbenchmarks-dot-jl/bui
lds/3000", "BUILDKITE_MESSAGE=Merge pull request #1124 from SciML/compathel
per/new_version/2024-11-15-00-33-40-352-03235991109\n\nCompatHelper: add ne
w compat entry for ReactionNetworkImporters at version 0.15 for package Jum
ps, (keep existing compat)", "BUILDKITE_RETRY_COUNT=0", "LOGNAME=sabae", "J
ULIA_CPU_THREADS=128", "BUILDKITE_PLUGIN_JULIA_CACHE_DIR=/cache/julia-build
kite-plugin", "BUILDKITE_PLUGIN_COPPERMIND_INPUT_HASH=f89dc72e7bc182c8ceace
b3e7fa17be5469f1e2559ad6f519981eb9dbcd18392", "BUILDKITE_PLUGIN_CRYPTIC_BAS
E64_AGENT_PRIVATE_KEY_SECRET=LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSU
lFcFFJQkFBS0NBUUVBdFhzSjMxRm0xSjdvSGM5b3hmWllKY3FxRk5yaXRMUUhxaDVJUHNGS3YyS
is1K1FVCkJGTWFEYzByN3M2dzQzUjAxaG5FTU9ZWDQwK3lFQ094eWxxK1o3dHdxWlNxS2U1MThw
c0RyeWRna0xJNzRnQUUKWVljU2Rnb0xreGFqVjcva29IRUw4K3M0SnRUVTVCendURXVwTllTZGN
QOFhQSmJLekY1RE5qdWJmeFA5ZjdSNwpseklMdjVpdmdpcWU1bW1MRndZcG9IU0VRVTV0ZU9PSE
ErS2NjUDZ3K2d1Q0VxMUZFb0N2MDRyaTFXaWpVZXorCjMrRFlTOFArUThkcTBiWFFmVEtVcnNLY
TJHZy81ZmdIeWdEdWRHT2ZsdzUvVEljR3VVbGNsd1hZb2tTRkpSWUIKRWtqVDlwQmdiTTRHMi9r
VzRhZmd3eGxzbjd1bFFuUDQ2VS8xZFFJREFRQUJBb0lCQVFDTU5sVjRUbUlPWC8raQpHSDh3ZzV
XekRSTy9MU1greXlFbzFHQ282NW9lcDdDNDVNUjZXdUpFUzRKbjdSVkpoczVHSkg0cDhYdi9TYk
dmCk9wVEFiTCt6VVdSSUFPNC9tMWRSYTJhN1NzY1d4RDN6N0dOMkhtK3E5elBlSHAxd3pIZU5aZ
29BR0htM3RyUU0KMGpidUczN09OSG1YdGQ1MEYyVHo1TmcwN0hURkJwV3hMMjJwNm9aZzgyUEk0
OXIrdUpWWmZ5MU5HZVRnaFA4cgp2dVRVTVJIcldZa25YbUR1eDVSMHNIdDFoU2FvTXBFbSsrMWc
1V09rSzZDTGFJbEV0ZitWVVBvR0piYlNYRzNJCmo5N1h5a3NGUDhGZ24wMWx4ZktGV1p4MXlnTV
dsUm00SFNCTWVkc1FpWStqeG5Sd3BtRnh5L2pIOVhFTUppT0wKQSsvVFdCbUJBb0dCQU52cXROQ
jRuVS9zODIxTU9oalRPTmtFOGNJOENxV1BRRTZXcEFOYVY5UkVVcGVCZzhmMgpjTXg2VFMweE9E
U0JRUk9PRnNEUG9pc1E0VVRGQmFialBLTU41d2IzNFVKYmN4V0ZwcGlLUHJMa09Zdmtqb01VCkN
Sb1pKK05Lb253RWh5bWJ0TG0yMHhmUUZCamY1R1QvMHJZUWcxUkN1OVllSmE0Z3NWYWtSNGh4QW
9HQkFOTkIKMzhxenJhTTBDeHhrSnZkVmlTY3g2MFowN2g4OHFISy9SV2dVK3ZZaitwZy9ibXBwM
mdiUXZkYTJxaS84UEl2OApSb0JwcmY2Y285TmdSa2JmY05xZmFnM1Z5SDhBNW1iUE1nK0s4YmFu
KzlwU003WkREVW1sdU03R3ZRSW5OVnBCCnBJcE1uWEk5eDZSSFlpOFF2MHhXOXcyUmpmS09TbEl
YZFlITjZwOUZBb0dCQUp0NXdwMkVPRXR5VE9NZnVnOGsKL1pMSVlSY2VGYlRZb3ZFc3BRWE4wRH
c4bFZ1UmNCWmx6M2R3bTdGd2s3ampESndEbjJodklzcHBvNmxYMVZnWQpYUjAxemZocU5QSVI3e
m52QkVuaHF0UVViKzdNQmtqN1dEZ0FRdWY1TXdpVXR1NGVxOVdFUUpjY1A2a2FXTUZpCjc1aFI4
bGNXMnU5VTN2VE5Iak1QNzVheEFvR0JBSm5HdExsZlMwQ21USVF4SHZBaE1rSDJvMVZaSGxCY25
oMVEKdjV3QTBhRkVGVkNuczU4QVNEVjMwd2d0VlBxeTkvdkoraVBWU1ZNeUFFcUlKUC9IKytVWD
cySDh3UUk1ekh6Lwp5MmZtOHdYTGg1ZW5DSDllbFppTGFsZ1I4RmxWNHc4OUF5R3NuVnNnUDJlR
WtxTEI1UTRUcTZnVDBLakVETE51CjRobEhvOGFsQW9HQUhBVGltTGRkS0JFTkN2MXZyNnZ0d3JC
ZGRCbWlGSWFwaVcvMk5acWxCTFp1bEp6MEwzdCsKM3FvSUF0Uisxd2xpWkQwZGJnRGdVeVRMcnN
5Y1RDSkZIczNIZTFXb3NCSzcxTmlncFZhWEVzWnFpOHNENjlvUQo2QkFnaEdvbnNGbTEydzhhRG
NDdm92WUxLTlhVV1BFT1c0akdvd2k0Tmx4NGZidHlkYXpIUEdnPQotLS0tLUVORCBSU0EgUFJJV
kFURSBLRVktLS0tLQo=", "BUILDKITE_PIPELINE_SLUG=scimlbenchmarks-dot-jl", "BU
ILDKITE_SHELL=/bin/bash -e -c", "BUILDKITE_PLUGINS_ENABLED=true", "LANGUAGE
=en_US", "GKS_FONTPATH=/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/231c36cbc48a78caf7818ee7f4cd260eb3d642e3", 
"BUILDKITE_AGENT_NAME=exclusive-amdci1.0", "BUILDKITE_STEP_ID=01932d42-adc4
-4afa-81d5-d6faefeea6eb", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_1=markdown/*
*/*.md", "BUILDKITE_TAG=", "OPENBLAS_MAIN_FREE=1", "PATH=/cache/julia-build
kite-plugin/julia_installs/bin/linux/x64/1.10/julia-1.10-latest-linux-x86_6
4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/gam
es:/usr/local/games:/snap/bin:/usr/bin:/usr/local/bin:/usr/bin:/bin", "GKS_
ENCODING=utf8", "BUILDKITE_AGENT_META_DATA_NUM_CPUS=128", "BUILDKITE_TRIGGE
RED_FROM_BUILD_NUMBER=", "BUILDKITE_AGENT_META_DATA_CONFIG_GITSHA=2a413ab3"
, "BUILDKITE_COMMAND=# Clear out these secrets as they're not needed during
 the actual build\nBUILDKITE_S3_ACCESS_KEY_ID=\"\" BUILDKITE_S3_SECRET_ACCE
SS_KEY=\"\" ./.buildkite/build_benchmark.sh \"benchmarks/Jumps\"\n", "LIBDE
COR_PLUGIN_DIR=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a
-f4d2d937f953/artifacts/38e215c51e5c0f77bc7a8813ba4586632a8fc750/lib/libdec
or/plugins-1", "BUILDKITE_PLUGIN_VALIDATION=false", "BUILDKITE_AGENT_META_D
ATA_ARCH=x86_64", "CONDARC=/cache/julia-buildkite-plugin/depots/5b300254-17
38-4989-ae0a-f4d2d937f953/conda/3/x86_64/condarc-julia.yml", "FORCE_SANDBOX
_MODE=unprivileged", "BUILDKITE_TRIGGERED_FROM_BUILD_ID=", "TERM=xterm-256c
olor", "PYTHONIOENCODING=UTF-8", "BUILDKITE_PULL_REQUEST_BASE_BRANCH=", "BU
ILDKITE_PIPELINE_ID=5b300254-1738-4989-ae0a-f4d2d937f953", "_=/cache/julia-
buildkite-plugin/julia_installs/bin/linux/x64/1.10/julia-1.10-latest-linux-
x86_64/bin/julia", "BUILDKITE_BUILD_ID=01932d40-f42a-404a-bad8-804d71c5163e
", "BUILDKITE_AGENT_ENDPOINT=https://agent.buildkite.com/v3", "BUILDKITE_PL
UGINS=[{\"github.com/staticfloat/cryptic-buildkite-plugin#v2\":{\"files\":[
\".buildkite/secrets/token.toml\"],\"variables\":[\"BUILDKITE_S3_ACCESS_KEY
_ID=\\\"U2FsdGVkX1+x3xs1ZRRZRt3FmwFQmYYKnpV7o8xKkX5Ib6y0o5fv0+rskVAj+JKu\\\
"\",\"BUILDKITE_S3_SECRET_ACCESS_KEY=\\\"U2FsdGVkX1+LWh1yX7LsMBlecEJLc08eJr
gOhurhd47CY1/jS3wCGVCQmS1t6f2j70spBoFdfc9kn2naj8HH5A==\\\"\",\"BUILDKITE_S3
_DEFAULT_REGION=\\\"U2FsdGVkX18ccoE9FmtkwsCm1x0MLMBlN/FLcAyKkY4=\\\"\"]}},{
\"github.com/JuliaCI/julia-buildkite-plugin#v1\":{\"version\":\"1.10\"}},{\
"github.com/staticfloat/sandbox-buildkite-plugin\":{\"gid\":1000,\"uid\":10
00,\"rootfs_url\":\"https://github.com/thazhemadam/openmodelica-rootfs-imag
e/releases/download/v1.23.0/rootfs-openmodelica-v1.23.0.amd64.tar.gz\",\"wo
rkspaces\":[\"/cache/julia-buildkite-plugin:/cache/julia-buildkite-plugin\"
],\"rootfs_treehash\":\"82970243dc4f188e599a976abc20951f4aba2912\"}},{\"git
hub.com/staticfloat/coppermind-buildkite-plugin#v1\":{\"inputs\":[\"benchma
rks/Jumps\",\"src/**/*.jl\",\"./*.toml\"],\"outputs\":[\"markdown/**/figure
s/*.png\",\"markdown/**/*.md\",\"markdown/**/*.pdf\",\"markdown/**/*.svg\",
\"notebook/**/*.ipynb\",\"pdf/**/*.pdf\",\"script/**/*.jl\"],\"s3_prefix\":
\"s3://julialang-buildkite-artifacts/scimlbenchmarks\"}}]", "BUILDKITE_SOCK
ETS_PATH=/root/.buildkite-agent/sockets", "SANDBOX_PERSISTENCE_DIR=/cache/s
andbox_persistence", "BUILDKITE_GIT_CLEAN_FLAGS=-ffxdq", "BUILDKITE_COMMIT=
449e59114745bdf31beb22db3be60ada0e57e18c", "BUILDKITE_PULL_REQUEST=false", 
"BUILDKITE_ORGANIZATION_ID=d409823c-5fa7-41c8-9033-7269c5fde4f3", "GKS_QT=e
nv LD_LIBRARY_PATH=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-
ae0a-f4d2d937f953/artifacts/f839432e3d2904a5c847b217ef0c0f489377ecc5/lib:/c
ache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/art
ifacts/2def0eca464bd6d89ccac85338474402359d4930/lib:/cache/julia-buildkite-
plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/d00220164876de
a2cb19993200662745eed5e2db/lib:/cache/julia-buildkite-plugin/julia_installs
/bin/linux/x64/1.10/julia-1.10-latest-linux-x86_64/bin/../lib/julia:/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/cb697355f42d1d0c8f70b15c0c3dc28952f774b4/lib:/cache/julia-buildkite-plugi
n/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/dc526f26fb179a3f68e
b13fcbe5d2d2a5aa7eeac/lib:/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/c9fd7a94d3c09eac4f3ca94d21bf40ccf65eccf5
/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/artifacts/b757190a3c47fcb65f573f631fdd549b98fcf2e4/lib:/cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/b3ddd58
3e7aec92a77cf5961fad01fd7063c1d40/lib:/cache/julia-buildkite-plugin/depots/
5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7099954ffb0b6e1641832a06a08e
479498ce479f/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae
0a-f4d2d937f953/artifacts/0803f8d074309498cdf55effdb9c55bc3ef88dde/lib:/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/f92cfdafb94fa8c50330be3684c9aeb80bd14750/lib:/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/1308e48c3f4f2fd9
adaa56b9bd4a86a995d50abd/lib:/cache/julia-buildkite-plugin/depots/5b300254-
1738-4989-ae0a-f4d2d937f953/artifacts/558980a93131f08be5335521b84e137ee3172
296/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d9
37f953/artifacts/d75cfbd8954fdbc933ebead40a9c8b91513c023a/lib:/cache/julia-
buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/9cfb
24edca23321a2dcebb63b4e196181359ecd6/lib:/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/aae093c71ea1b1dc04c457afc
ae880d26c532115/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989
-ae0a-f4d2d937f953/artifacts/bd965e3c7f9460155f06361da380c63fa0351ef6/lib:/
cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/ar
tifacts/060cf7829c3363638c29228ea4ab0bd033d8eab0/lib:/cache/julia-buildkite
-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/77d0e7c90e6a2
fd6f2f8457bbb7b86ed86d140d9/lib:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/1e69ef9fbf05e2896d3cb70eac8080c4d1
0f8696/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/e200b9737b27598b95b404cbc34e74f95b2bf5d0/lib:/cache/jul
ia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/a
8e2d77aed043a035fd970326d8f070080efa8fa/lib:/cache/julia-buildkite-plugin/d
epots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/6f98018cad6a09e91f9065
8f188c6be47e48a0c7/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/d4f3ff9736df0dda120f8dc1d27174b0d5696fb1/li
b:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
/artifacts/2ab21f29b30c228bd0e5215585f822730cad5a72/lib:/cache/julia-buildk
ite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/62c0108762
22f83fe8878bf2af0e362083d20ee3/lib:/cache/julia-buildkite-plugin/depots/5b3
00254-1738-4989-ae0a-f4d2d937f953/artifacts/75b657b876788e58671ab6b88e49019
aa36b67cd/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-
f4d2d937f953/artifacts/bd1f25e7053ebc00ee7d82f3c5ec4cf1e9a51e17/lib:/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/cf5d5f8a6109be3f9c460a39768f57a3e53ff11d/lib:/cache/julia-buildkite-plugi
n/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/c8a20a2030f10b70947
d8d2a6bff7f8b5f343fe9/lib:/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/0631e2a6a31b5692eec7a575836451b16b734ec0
/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/artifacts/4abd0521d210cb9e48ea5e711873ba34dc05fc70/lib:/cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/1cf7375
e8ec1bbe1219934488737c12237ba2012/lib:/cache/julia-buildkite-plugin/depots/
5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/587de110e5f58fd435dc35b294df
31bb7a75f692/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae
0a-f4d2d937f953/artifacts/fc239b3ff5739aeab252bd154fa4dd045fefe629/lib:/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/c951fb23b5652def1dea483af7095a38f3b3cefd/lib:/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/85dab0a3f6b9cc2e
5d61165ee376bf46260812a4/lib:/cache/julia-buildkite-plugin/depots/5b300254-
1738-4989-ae0a-f4d2d937f953/artifacts/5b83972689fb7dea5e89326f1c0ba60d68e96
2fb/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d9
37f953/artifacts/9d7f5887309a96013a2c75f48b5e137e60ccae8f/lib:/cache/julia-
buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/74fd
55820a62aa47ebf4d942aa59096980a1851d/lib:/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/aa52699bd1491b7de9c72fc1e
ab21e2e4bd649e2/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989
-ae0a-f4d2d937f953/artifacts/951960aa7e4599865406f432151003dd82cde65d/lib:/
cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/ar
tifacts/37dda4e57d9de95c99d1f8c6b3d8f4eca88c39a2/lib:/cache/julia-buildkite
-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/136b88dafbb4b
6b7bfa6d4cff225f7d697015bd1/lib:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/38e215c51e5c0f77bc7a8813ba4586632a
8fc750/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/f0d193662fead3500b523f94b4f1878daab59a93/lib:/cache/jul
ia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/0
5616da88f6b36c7c94164d4070776aef18ce46b/lib:/cache/julia-buildkite-plugin/d
epots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/2df316da869cd97f7d7002
9428ee1e2e521407cd/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/7190f0cb0832b80761cc6d513dd9b935f3e26358/li
b:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
/artifacts/4daa3879a820580557ef34945e2ae243dfcbba11/lib:/cache/julia-buildk
ite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/3643539f49
1c217e13c1595daad81dd1426fba07/lib:/cache/julia-buildkite-plugin/depots/5b3
00254-1738-4989-ae0a-f4d2d937f953/artifacts/71f3593804fb3a115f7fcb71b20c4e1
f9b32290f/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-
f4d2d937f953/artifacts/fbef68f6f587b5d3709af5b95701b92e3b890e4b/lib:/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/0ef7836563e0bb993333653a46424119858e5f1d/lib:/cache/julia-buildkite-plugi
n/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4c45bf9c8292490acd9
463bbfbf168277d9720b6/lib:/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/2efdb7b239e9f244a3a933925294ea27cc6a61c2
/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/artifacts/872754c2f795d19a3e2e205b2bbaea659f28d11e/lib:/cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7da37be
2742b3d1cfe1c14bf5bbd85aed4887f46/lib:/cache/julia-buildkite-plugin/depots/
5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/c35cb3f6f3043a4e962fd56b61ad
91b4adb557f7/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae
0a-f4d2d937f953/artifacts/13befbe35cffe7da192c7001ece18b4be3aa3720/lib:/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/cacd8c147f866d6672e1aca9bb01fb919a81e96a/lib:/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/b7dc5dce96373741
4a564aca8d4b82ee388f4fa1/lib:/cache/julia-buildkite-plugin/depots/5b300254-
1738-4989-ae0a-f4d2d937f953/artifacts/0d364e900393f710a03a5bafe2852d76e4d2c
2cd/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d9
37f953/artifacts/1a2adcee7d99fea18ead33c350332626b262e29a/lib:/cache/julia-
buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/40ee
a58ff37ecc8fb6f21f41079a33b511b3ff92/lib:/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/79cc5446ced978de84b6e673e
01da0ebfdd6e4a5/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989
-ae0a-f4d2d937f953/artifacts/fce445d991cf502908d681348eec2174c5e31ba8/lib:/
cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/ar
tifacts/b0d2538004dda9eb6449b72b0b85703aeac30a66/lib:/cache/julia-buildkite
-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f03dd5ac03468
009d5a99bbfcdf336c2dc372de4/lib:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/eae77862e95d04dfafb9fbe9ae39e688b3
5d756a/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/231c36cbc48a78caf7818ee7f4cd260eb3d642e3/lib:/cache/jul
ia-buildkite-plugin/julia_installs/bin/linux/x64/1.10/julia-1.10-latest-lin
ux-x86_64/bin/../lib/julia:/cache/julia-buildkite-plugin/julia_installs/bin
/linux/x64/1.10/julia-1.10-latest-linux-x86_64/bin/../lib /cache/julia-buil
dkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/231c36cb
c48a78caf7818ee7f4cd260eb3d642e3/bin/gksqt"]), ProcessExited(1)) [1]
```





Now, we instantiate the problems, find their solutions and plot the results.

```julia
let fig = []
  for (i, (algo, stepper, use_recursion, label)) in enumerate(algorithms[5:end])
    @info label
    if algo isa PyTick
        _p = (p[1], p[2], p[3])
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        jump_prob.reset()
        jump_prob.simulate()
        t = tspan[1]:0.1:tspan[2]
        N = [[sum(jumps .< _t) for _t in t] for jumps in jump_prob.timestamps]
        push!(fig, plot(t, N, title=label, legend=false, format=fmt))
    elseif algo isa PDMPCHVSimple
        if use_recursion
          h = zeros(eltype(tspan), nv(G))
          ϕ = zeros(eltype(tspan), nv(G))
          _p = (p[1], p[2], p[3], h, ϕ, g)
        else
          h = [eltype(tspan)[] for _ in 1:nv(G)]
          _p = (p[1], p[2], p[3], h, g)
        end
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        sol = solve(jump_prob, stepper)
        push!(fig, plot(sol.time, sol.xd[1:V, :]', title=label, legend=false, format=fmt))
    elseif algo isa PDMPCHVFull
        _p = (p[1], p[2], p[3], nothing, nothing, g)
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        sol = solve(jump_prob, stepper)
        push!(fig, plot(sol.time, sol.xd[1:V, :]', title=label, legend=false, format=fmt))
    end
  end
  fig = plot(fig..., layout=(2,2), format=fmt, size=(width_px, 2*height_px/2))
end
```

![](figures/MultivariateHawkes_16_1.png)



# Correctness: QQ-Plots

We check that the algorithms produce correct simulation by inspecting their QQ-plots. Point process theory says that transforming the simulated points using the compensator should produce points whose inter-arrival duration is distributed according to the exponential distribution (see Section 7.4 [1]).

The compensator of any point process is the integral of the conditional intensity ``\Lambda_i^\ast(t) = \int_0^t \lambda_i^\ast(u) du``. The compensator for the Multivariate Hawkes process is defined below.
```math
    \Lambda_i^\ast(t) = \lambda t + \frac{\alpha}{\beta} \sum_{j \in E_i} \sum_{t_{n_j} < t} ( 1 - \exp \left[-\beta (t - t_{n_j}) \right])
```

```julia
function hawkes_Λ(i::Int, g, p)
    @inline @inbounds function Λ(t, h)
        λ, α, β = p
        x = λ * t
        for j in g[i]
            for _t in h[j]
                if _t >= t
                    break
                end
                x += (α / β) * (1 - exp(-β * (t - _t)))
            end
        end
        return x
    end
    return Λ
end

function hawkes_Λ(g, p)
    return [hawkes_Λ(i, g, p) for i = 1:length(g)]
end

Λ = hawkes_Λ(g, p)
```

```
10-element Vector{Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vec
tor{Int64}}, Tuple{Float64, Float64, Float64}}}:
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
 (::Main.var"##WeaveSandBox#225".var"#Λ#33"{Int64, Vector{Vector{Int64}}, T
uple{Float64, Float64, Float64}}) (generic function with 1 method)
```





We need a method for extracting the history from a simulation run. Below, we define such functions for each type of algorithm.

```julia
"""
Given an ODE solution `sol`, recover the timestamp in which events occurred. It
returns a vector with the history of each process in `sol`.

It assumes that `JumpProblem` was initialized with `save_positions` equal to
`(true, false)`, `(false, true)` or `(true, true)` such the system's state is
saved before and/or after the jump occurs; and, that `sol.u` is a
non-decreasing series that counts the total number of events observed as a
function of time.
"""
function histories(u, t)
    _u = permutedims(reduce(hcat, u))
    k = size(_u)[2]
    # computes a mask that show when total counts change
    mask = cat(fill(0.0, 1, k), _u[2:end, :] .- _u[1:end-1, :], dims = 1) .≈ 1
    h = Vector{typeof(t)}(undef, k)
    @inbounds for i = 1:k
        h[i] = t[mask[:, i]]
    end
    return h
end

function histories(sol::S) where {S<:ODESolution}
    # get u and permute the dimensions to get a matrix n x k with n obsevations and k processes.
    if sol.u[1] isa ExtendedJumpArray
        u = map((u) -> u.u, sol.u)
    else
        u = sol.u
    end
    return histories(u, sol.t)
end

function histories(sol::S) where {S<:PDMP.PDMPResult}
    return histories(sol.xd.u, sol.time)
end

function histories(sols)
    map(histories, sols)
end
```

```
histories (generic function with 4 methods)
```





We also need to compute the quantiles of the empirical distribution given a history of events `hs`, the compensator `Λ` and the target quantiles `quant`.

```julia
import Distributions: Exponential

"""
Computes the empirical and expected quantiles given a history of events `hs`,
the compensator `Λ` and the target quantiles `quant`.

The history `hs` is a vector with the history of each process. Alternatively,
the function also takes a vector of histories containing the histories from
multiple runs.

The compensator `Λ` can either be an homogeneous compensator function that
equally applies to all the processes in `hs`. Alternatively, it accepts a
vector of compensator that applies to each process.
"""
function qq(hs, Λ, quant = 0.01:0.01:0.99)
    _hs = apply_Λ(hs, Λ)
    T = typeof(hs[1][1][1])
    Δs = Vector{Vector{T}}(undef, length(hs[1]))
    for k = 1:length(Δs)
        _Δs = Vector{Vector{T}}(undef, length(hs))
        for i = 1:length(_Δs)
            _Δs[i] = _hs[i][k][2:end] .- _hs[i][k][1:end-1]
        end
        Δs[k] = reduce(vcat, _Δs)
    end
    empirical_quant = map((_Δs) -> quantile(_Δs, quant), Δs)
    expected_quant = quantile(Exponential(1.0), quant)
    return empirical_quant, expected_quant
end

"""
Compute the compensator `Λ` value for each timestamp recorded in history `hs`.

The history `hs` is a vector with the history of each process. Alternatively,
the function also takes a vector of histories containing the histories from
multiple runs.

The compensator `Λ` can either be an homogeneous compensator function that
equally applies to all the processes in `hs`. Alternatively, it accepts a
vector of compensator that applies to each process.
"""
function apply_Λ(hs::V, Λ) where {V<:Vector{<:Number}}
    _hs = similar(hs)
    @inbounds for n = 1:length(hs)
        _hs[n] = Λ(hs[n], hs)
    end
    return _hs
end

function apply_Λ(k::Int, hs::V, Λ::A) where {V<:Vector{<:Vector{<:Number}},A<:Array}
    @inbounds hsk = hs[k]
    @inbounds Λk = Λ[k]
    _hs = similar(hsk)
    @inbounds for n = 1:length(hsk)
        _hs[n] = Λk(hsk[n], hs)
    end
    return _hs
end

function apply_Λ(hs::V, Λ) where {V<:Vector{<:Vector{<:Number}}}
    _hs = similar(hs)
    @inbounds for k = 1:length(_hs)
        _hs[k] = apply_Λ(hs[k], Λ)
    end
    return _hs
end

function apply_Λ(hs::V, Λ::A) where {V<:Vector{<:Vector{<:Number}},A<:Array}
    _hs = similar(hs)
    @inbounds for k = 1:length(_hs)
        _hs[k] = apply_Λ(k, hs, Λ)
    end
    return _hs
end

function apply_Λ(hs::V, Λ) where {V<:Vector{<:Vector{<:Vector{<:Number}}}}
    return map((_hs) -> apply_Λ(_hs, Λ), hs)
end
```

```
apply_Λ (generic function with 5 methods)
```





We can construct QQ-plots with a Plot recipe as following.

```julia
@userplot QQPlot
@recipe function f(x::QQPlot)
    empirical_quant, expected_quant = x.args
    max_empirical_quant = maximum(maximum, empirical_quant)
    max_expected_quant = maximum(expected_quant)
    upperlim = ceil(maximum([max_empirical_quant, max_expected_quant]))
    @series begin
        seriestype := :line
        linecolor := :lightgray
        label --> ""
        (x) -> x
    end
    @series begin
        seriestype := :scatter
        aspect_ratio := :equal
        xlims := (0.0, upperlim)
        ylims := (0.0, upperlim)
        xaxis --> "Expected"
        yaxis --> "Empirical"
        markerstrokewidth --> 0
        markerstrokealpha --> 0
        markersize --> 1.5
        size --> (400, 500)
        label --> permutedims(["quantiles $i" for i = 1:length(empirical_quant)])
        expected_quant, empirical_quant
    end
end
```




Now, we simulate all of the algorithms we defined in the previous Section ``250`` times to produce their QQ-plots.

```julia
let fig = []
    for (i, (algo, stepper, use_recursion, label)) in enumerate(algorithms)
        @info label
        if algo isa PyTick
            _p = (p[1], p[2], p[3])
        elseif algo isa PDMPCHVSimple
            if use_recursion
                h = zeros(eltype(tspan), nv(G))
                ϕ = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, ϕ, g)
            else
                h = [eltype(tspan)[] for _ in 1:nv(G)]
                _p = (p[1], p[2], p[3], h, g)
            end
        elseif algo isa PDMPCHVFull
            _p = (p[1], p[2], p[3], nothing, nothing, g)
        else
            if use_recursion
                h = zeros(eltype(tspan), nv(G))
                ϕ = zeros(eltype(tspan), nv(G))
                urate = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate, ϕ)
            else
                h = [eltype(tspan)[] for _ = 1:nv(G)]
                urate = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate)
            end
        end
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        runs = Vector{Vector{Vector{Number}}}(undef, 250)
        for n = 1:length(runs)
            if algo isa PyTick
                jump_prob.reset()
                jump_prob.simulate()
                runs[n] = jump_prob.timestamps
            else
                if ~(algo isa PDMPCHVFull)
                    if use_recursion
                        h .= 0
                        ϕ .= 0
                    else
                        for _h in h empty!(_h) end
                    end
                    if ~(algo isa PDMPCHVSimple)
                        urate .= 0
                    end
                end
                runs[n] = histories(solve(jump_prob, stepper))
            end
        end
        qqs = qq(runs, Λ)
        push!(fig, qqplot(qqs..., legend = false, aspect_ratio = :equal, title=label, fmt=fmt))
    end
    fig = plot(fig..., layout = (4, 2), fmt=fmt, size=(width_px, 4*height_px/2))
end
```

![](figures/MultivariateHawkes_21_1.png)



# Benchmarking performance

In this Section we benchmark all the algorithms introduced in the first Section.

We generate networks in the range from ``1`` to ``95`` nodes and simulate the Multivariate Hawkes process ``25`` units of time.

 and simulate models in the range from ``1`` to ``95`` nodes for ``25`` units of time. We fix the Hawkes parameters at ``\lambda = 0.5 , \alpha = 0.1 , \beta = 5.0`` which ensures the process does not explode. We simulate ``50`` trajectories with a limit of ten seconds to complete execution for each configuration.

```julia
tspan = (0.0, 25.0)
p = (0.5, 0.1, 5.0)
Vs = append!([1], 5:5:95)
Gs = [erdos_renyi(V, 0.2, seed = 6221) for V in Vs]

bs = Vector{Vector{BenchmarkTools.Trial}}()

for (algo, stepper, use_recursion, label) in algorithms
    @info label
    global _stepper = stepper
    push!(bs, Vector{BenchmarkTools.Trial}())
    _bs = bs[end]
    for (i, G) in enumerate(Gs)
        local g = [neighbors(G, i) for i = 1:nv(G)]
        local u = [0.0 for i = 1:nv(G)]
        if algo isa PyTick
            _p = (p[1], p[2], p[3])
        elseif algo isa PDMPCHVSimple
            if use_recursion
              global h = zeros(eltype(tspan), nv(G))
              global ϕ = zeros(eltype(tspan), nv(G))
              _p = (p[1], p[2], p[3], h, ϕ, g)
            else
              global h = [eltype(tspan)[] for _ in 1:nv(G)]
              _p = (p[1], p[2], p[3], h, g)
            end
        elseif algo isa PDMPCHVFull
            _p = (p[1], p[2], p[3], nothing, nothing, g)
        else
            if use_recursion
                global h = zeros(eltype(tspan), nv(G))
                global urate = zeros(eltype(tspan), nv(G))
                global ϕ = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate, ϕ)
            else
                global h = [eltype(tspan)[] for _ = 1:nv(G)]
                global urate = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate)
            end
        end
        global jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        trial = try
            if algo isa PyTick
                @benchmark(
                    jump_prob.simulate(),
                    setup = (jump_prob.reset()),
                    samples = 50,
                    evals = 1,
                    seconds = 10,
                )
            else
                if algo isa PDMPCHVFull
                    @benchmark(
                        solve(jump_prob, _stepper),
                        setup = (),
                        samples = 50,
                        evals = 1,
                        seconds = 10,
                    )
                elseif algo isa PDMPCHVSimple
                    if use_recursion
                        @benchmark(solve(jump_prob, _stepper),
                                   setup=(h .= 0; ϕ .= 0),
                                   samples=50,
                                   evals=1,
                                   seconds=10,)
                    else
                        @benchmark(solve(jump_prob, _stepper),
                                   setup=([empty!(_h) for _h in h]),
                                   samples=50,
                                   evals=1,
                                   seconds=10,)
                    end
                else
                    if use_recursion
                        @benchmark(
                            solve(jump_prob, _stepper),
                            setup = (h .= 0; urate .= 0; ϕ .= 0),
                            samples = 50,
                            evals = 1,
                            seconds = 10,
                        )
                    else
                        @benchmark(
                            solve(jump_prob, _stepper),
                            setup = ([empty!(_h) for _h in h]; urate .= 0),
                            samples = 50,
                            evals = 1,
                            seconds = 10,
                        )
                    end
                end
            end
        catch e
            BenchmarkTools.Trial(
                BenchmarkTools.Parameters(samples = 50, evals = 1, seconds = 10),
            )
        end
        push!(_bs, trial)
        if (nv(G) == 1 || nv(G) % 10 == 0)
            median_time =
                length(trial) > 0 ? "$(BenchmarkTools.prettytime(median(trial.times)))" :
                "nan"
            println("algo=$(label), V = $(nv(G)), length = $(length(trial.times)), median time = $median_time")
        end
    end
end
```

```
algo=Direct (brute-force), V = 1, length = 50, median time = 92.709 μs
algo=Direct (brute-force), V = 10, length = 50, median time = 10.842 ms
algo=Direct (brute-force), V = 20, length = 1, median time = 99.580 ms
algo=Direct (brute-force), V = 30, length = 1, median time = 315.483 ms
algo=Direct (brute-force), V = 40, length = 1, median time = 1.382 s
algo=Direct (brute-force), V = 50, length = 1, median time = 2.923 s
algo=Direct (brute-force), V = 60, length = 1, median time = 5.795 s
algo=Direct (brute-force), V = 70, length = 1, median time = 10.525 s
algo=Direct (brute-force), V = 80, length = 1, median time = 14.468 s
algo=Direct (brute-force), V = 90, length = 1, median time = 22.642 s
algo=Coevolve (brute-force), V = 1, length = 50, median time = 4.600 μs
algo=Coevolve (brute-force), V = 10, length = 50, median time = 213.569 μs
algo=Coevolve (brute-force), V = 20, length = 50, median time = 1.373 ms
algo=Coevolve (brute-force), V = 30, length = 50, median time = 3.452 ms
algo=Coevolve (brute-force), V = 40, length = 50, median time = 8.359 ms
algo=Coevolve (brute-force), V = 50, length = 50, median time = 16.756 ms
algo=Coevolve (brute-force), V = 60, length = 50, median time = 29.289 ms
algo=Coevolve (brute-force), V = 70, length = 50, median time = 50.937 ms
algo=Coevolve (brute-force), V = 80, length = 50, median time = 75.329 ms
algo=Coevolve (brute-force), V = 90, length = 50, median time = 124.838 ms
algo=Direct (recursive), V = 1, length = 50, median time = 98.309 μs
algo=Direct (recursive), V = 10, length = 50, median time = 4.858 ms
algo=Direct (recursive), V = 20, length = 1, median time = 25.477 ms
algo=Direct (recursive), V = 30, length = 1, median time = 70.623 ms
algo=Direct (recursive), V = 40, length = 1, median time = 1.034 s
algo=Direct (recursive), V = 50, length = 1, median time = 1.861 s
algo=Direct (recursive), V = 60, length = 1, median time = 3.563 s
algo=Direct (recursive), V = 70, length = 1, median time = 5.990 s
algo=Direct (recursive), V = 80, length = 1, median time = 9.056 s
algo=Direct (recursive), V = 90, length = 1, median time = 13.668 s
algo=Coevolve (recursive), V = 1, length = 50, median time = 4.850 μs
algo=Coevolve (recursive), V = 10, length = 50, median time = 79.544 μs
algo=Coevolve (recursive), V = 20, length = 50, median time = 280.113 μs
algo=Coevolve (recursive), V = 30, length = 50, median time = 509.991 μs
algo=Coevolve (recursive), V = 40, length = 50, median time = 924.353 μs
algo=Coevolve (recursive), V = 50, length = 50, median time = 1.583 ms
algo=Coevolve (recursive), V = 60, length = 50, median time = 2.337 ms
algo=Coevolve (recursive), V = 70, length = 50, median time = 3.313 ms
algo=Coevolve (recursive), V = 80, length = 50, median time = 4.346 ms
algo=Coevolve (recursive), V = 90, length = 50, median time = 5.846 ms
algo=PDMPCHVSimple (brute-force), V = 1, length = 50, median time = 57.974 
μs
algo=PDMPCHVSimple (brute-force), V = 10, length = 50, median time = 4.894 
ms
algo=PDMPCHVSimple (brute-force), V = 20, length = 50, median time = 41.634
 ms
algo=PDMPCHVSimple (brute-force), V = 30, length = 50, median time = 108.78
7 ms
algo=PDMPCHVSimple (brute-force), V = 40, length = 35, median time = 278.56
4 ms
algo=PDMPCHVSimple (brute-force), V = 50, length = 16, median time = 599.25
8 ms
algo=PDMPCHVSimple (brute-force), V = 60, length = 9, median time = 1.066 s
algo=PDMPCHVSimple (brute-force), V = 70, length = 5, median time = 1.725 s
algo=PDMPCHVSimple (brute-force), V = 80, length = 3, median time = 2.911 s
algo=PDMPCHVSimple (brute-force), V = 90, length = 2, median time = 4.933 s
algo=PDMPCHVSimple (recursive), V = 1, length = 50, median time = 58.910 μs
algo=PDMPCHVSimple (recursive), V = 10, length = 50, median time = 331.473 
μs
algo=PDMPCHVSimple (recursive), V = 20, length = 50, median time = 800.120 
μs
algo=PDMPCHVSimple (recursive), V = 30, length = 50, median time = 1.562 ms
algo=PDMPCHVSimple (recursive), V = 40, length = 50, median time = 2.493 ms
algo=PDMPCHVSimple (recursive), V = 50, length = 50, median time = 3.640 ms
algo=PDMPCHVSimple (recursive), V = 60, length = 50, median time = 5.116 ms
algo=PDMPCHVSimple (recursive), V = 70, length = 50, median time = 7.088 ms
algo=PDMPCHVSimple (recursive), V = 80, length = 50, median time = 9.381 ms
algo=PDMPCHVSimple (recursive), V = 90, length = 50, median time = 12.194 m
s
algo=PDMPCHVFull, V = 1, length = 50, median time = 59.014 μs
algo=PDMPCHVFull, V = 10, length = 50, median time = 475.007 μs
algo=PDMPCHVFull, V = 20, length = 50, median time = 762.584 μs
algo=PDMPCHVFull, V = 30, length = 50, median time = 1.229 ms
algo=PDMPCHVFull, V = 40, length = 50, median time = 1.532 ms
algo=PDMPCHVFull, V = 50, length = 50, median time = 1.888 ms
algo=PDMPCHVFull, V = 60, length = 50, median time = 2.511 ms
algo=PDMPCHVFull, V = 70, length = 50, median time = 3.075 ms
algo=PDMPCHVFull, V = 80, length = 50, median time = 3.687 ms
algo=PDMPCHVFull, V = 90, length = 50, median time = 4.518 ms
```



```julia
let fig = plot(
        yscale = :log10,
        xlabel = "V",
        ylabel = "Time (ns)",
        legend_position = :outertopright,
)
    for (i, (algo, stepper, use_recursion, label)) in enumerate(algorithms)
        _bs, _Vs = [], []
        for (j, b) in enumerate(bs[i])
            if length(b) == 50
                push!(_bs, median(b.times))
                push!(_Vs, Vs[j])
            end
        end
        plot!(_Vs, _bs, label=label)
    end
    title!("Simulations, 50 samples: nodes × time")
end
```

![](figures/MultivariateHawkes_23_1.png)



# References

[1] D. J. Daley and D. Vere-Jones. An Introduction to the Theory of Point Processes: Volume I: Elementary Theory and Methods. Probability and Its Applications, An Introduction to the Theory of Point Processes. Springer-Verlag, 2 edition. doi:10.1007/b97277.

[2] Patrick J. Laub, Young Lee, and Thomas Taimre. The Elements of Hawkes Processes. Springer International Publishing. doi:10.1007/978-3-030-84639-8.
