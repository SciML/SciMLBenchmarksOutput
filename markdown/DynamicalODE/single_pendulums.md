---
author: "Gen Kuroki (黒木玄), Chris Rackauckas"
title: "Single Pedulum Comparison"
---


# Solving single pendulums by DifferentialEquations.jl

In this notebook, we shall solve the single pendulum equation:

$$\ddot q = -\sin q,$$

where $q$ means the angle.

Hamiltonian:

$$H(q,p) = \frac{1}{2}p^2 - \cos q + 1.$$

Canonical equation:

$$\dot q = p, \quad \dot p = - \sin q.$$

Initial condition:

$$q(0) = 0, \quad p(0) = 2k.$$

Exact solution:

$$q(t) = 2\arcsin(k\,\mathrm{sn}(t,k)).$$

Maximum of $q(t)$:

$$\sin(q_{\max}/2) = k, \quad q_{\max} = \max\{q(t)\}.$$

Define $y(t)$ by

$$y(t) = \sin(q(t)/2) = k\,\mathrm{sn}(t,k), \quad y_{\max} = k.$$

```julia
# Single pendulums shall be solved numerically.
#
using OrdinaryDiffEq, Elliptic, Printf, DiffEqPhysics, Statistics

sol2q(sol) = [sol.u[i][j] for i in 1:length(sol.u), j in 1:length(sol.u[1])÷2]
sol2p(sol) = [sol.u[i][j] for i in 1:length(sol.u), j in length(sol.u[1])÷2+1:length(sol.u[1])]
sol2tqp(sol) = (sol.t, sol2q(sol), sol2p(sol))

# The exact solutions of single pendulums can be expressed by the Jacobian elliptic functions.
#
sn(u, k) = Jacobi.sn(u, k^2) # the Jacobian sn function

# Use PyPlot.
#
using PyPlot

colorlist = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
cc(k) = colorlist[mod1(k, length(colorlist))]

# plot the sulution of a Hamiltonian problem
#
function plotsol(sol::ODESolution)
    local t, q, p
    t, q, p = sol2tqp(sol)
    local d = size(q)[2]
    for j in 1:d
        j_str = d > 1 ? "[$j]" : ""
        plot(t, q[:,j], color=cc(2j-1), label="q$(j_str)", lw=1)
        plot(t, p[:,j], color=cc(2j),   label="p$(j_str)", lw=1, ls="--")
    end
    grid(ls=":")
    xlabel("t")
    legend()
end

# plot the solution of a Hamiltonian problem on the 2D phase space
#
function plotsol2(sol::ODESolution)
    local t, q, p
    t, q, p = sol2tqp(sol)
    local d = size(q)[2]
    for j in 1:d
        j_str = d > 1 ? "[$j]" : ""
        plot(q[:,j], p[:,j], color=cc(j), label="(q$(j_str),p$(j_str))", lw=1)
    end
    grid(ls=":")
    xlabel("q")
    ylabel("p")
    legend()
end

# plot the energy of a Hamiltonian problem
#
function plotenergy(H, sol::ODESolution)
    local t, q, p
    t, q, p = sol2tqp(sol)
    local energy = [H(q[i,:], p[i,:], nothing) for i in 1:size(q)[1]]
    plot(t, energy, label="energy", color="red", lw=1)
    grid(ls=":")
    xlabel("t")
    legend()
    local stdenergy_str = @sprintf("%.3e", std(energy))
    title("                    std(energy) = $stdenergy_str", fontsize=10)
end

# plot the numerical and exact solutions of a single pendulum
#
# Warning: Assume q(0) = 0, p(0) = 2k.   (for the sake of laziness)
#
function plotcomparison(k, sol::ODESolution)
    local t, q, p
    t, q, p = sol2tqp(sol)
    local y = sin.(q/2)
    local y_exact = k*sn.(t, k) # the exact solution

    plot(t, y,       label="numerical", lw=1)
    plot(t, y_exact, label="exact",     lw=1, ls="--")
    grid(ls=":")
    xlabel("t")
    ylabel("y = sin(q(t)/2)")
    legend()
    local error_str = @sprintf("%.3e", maximum(abs.(y - y_exact)))
    title("maximum(abs(numerical - exact)) = $error_str", fontsize=10)
end

# plot solution and energy
#
function plotsolenergy(H, integrator, Δt, sol::ODESolution)
    local integrator_str = replace("$integrator", r"^[^.]*\." => "")

    figure(figsize=(10,8))

    subplot2grid((21,20), ( 1, 0), rowspan=10, colspan=10)
    plotsol(sol)

    subplot2grid((21,20), ( 1,10), rowspan=10, colspan=10)
    plotsol2(sol)

    subplot2grid((21,20), (11, 0), rowspan=10, colspan=10)
    plotenergy(H, sol)

    suptitle("=====    $integrator_str,   Δt = $Δt    =====")
end

# Solve a single pendulum
#
function singlependulum(k, integrator, Δt; t0 = 0.0, t1 = 100.0)
    local H(p,q,params) = p[1]^2/2 - cos(q[1]) + 1
    local q0 = [0.0]
    local p0 = [2k]
    local prob = HamiltonianProblem(H, p0, q0, (t0, t1))

    local integrator_str = replace("$integrator", r"^[^.]*\." => "")
    @printf("%-25s", "$integrator_str:")
    sol = solve(prob, integrator, dt=Δt)
    @time local sol = solve(prob, integrator, dt=Δt)

    sleep(0.1)
    figure(figsize=(10,8))

    subplot2grid((21,20), ( 1, 0), rowspan=10, colspan=10)
    plotsol(sol)

    subplot2grid((21,20), ( 1,10), rowspan=10, colspan=10)
    plotsol2(sol)

    subplot2grid((21,20), (11, 0), rowspan=10, colspan=10)
    plotenergy(H, sol)

    subplot2grid((21,20), (11,10), rowspan=10, colspan=10)
    plotcomparison(k, sol)

    suptitle("=====    $integrator_str,   Δt = $Δt    =====")
end
```

```
Collecting package metadata (current_repodata.json): ...working... done
Solving environment: ...working... unsuccessful initial attempt using froze
n solve. Retrying with flexible solve.
Solving environment: ...working... unsuccessful attempt using repodata from
 current_repodata.json, retrying with next repodata source.
Error: InitError: failed process: Process(setenv(`/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/conda/3/x86_64/bin/conda i
nstall -q -y matplotlib`,["BUILDKITE_UNBLOCKER_TEAMS=juliagpu-full-access:s
ciml-full-access", "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/21581/bus"
, "BUILDKITE_PULL_REQUEST_REPO=https://github.com/SciML/SciMLBenchmarks.jl.
git", "BUILDKITE_SOURCE=webhook", "BUILDKITE_PLUGIN_COPPERMIND_INPUTS_0=ben
chmarks/DynamicalODE", "BUILDKITE_GIT_CLONE_FLAGS=-v", "BUILDKITE_PLUGIN_CR
YPTIC_BASE64_AGENT_PUBLIC_KEY_SECRET=LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTU
lJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUF0WHNKMzFGbTFKN29IY
zlveGZaWQpKY3FxRk5yaXRMUUhxaDVJUHNGS3YySis1K1FVQkZNYURjMHI3czZ3NDNSMDFobkVN
T1lYNDAreUVDT3h5bHErClo3dHdxWlNxS2U1MThwc0RyeWRna0xJNzRnQUVZWWNTZGdvTGt4YWp
WNy9rb0hFTDgrczRKdFRVNUJ6d1RFdXAKTllTZGNQOFhQSmJLekY1RE5qdWJmeFA5ZjdSN2x6SU
x2NWl2Z2lxZTVtbUxGd1lwb0hTRVFVNXRlT09IQStLYwpjUDZ3K2d1Q0VxMUZFb0N2MDRyaTFXa
WpVZXorMytEWVM4UCtROGRxMGJYUWZUS1Vyc0thMkdnLzVmZ0h5Z0R1CmRHT2ZsdzUvVEljR3VV
bGNsd1hZb2tTRkpSWUJFa2pUOXBCZ2JNNEcyL2tXNGFmZ3d4bHNuN3VsUW5QNDZVLzEKZFFJREF
RQUIKLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0tCg==", "BUILDKITE_ENV_FILE=/tmp/job-en
v-0189b404-9636-43e2-b3c4-ee575c16da70895109322", "BUILDKITE_BUILD_NUMBER=1
346", "BUILDKITE_GIT_CLONE_MIRROR_FLAGS=-v", "BUILDKITE_AGENT_DEBUG=false",
 "BUILDKITE_AGENT_META_DATA_QUEUE=juliaecosystem", "BUILDKITE_PLUGINS_PATH=
/etc/buildkite-agent/plugins", "BUILDKITE_COMMAND_EVAL=true", "BUILDKITE_AG
ENT_META_DATA_SANDBOX_CAPABLE=true", "BUILDKITE_ORGANIZATION_SLUG=julialang
", "BUILDKITE_PLUGIN_COPPERMIND_INPUTS_1=src/**/*.jl", "BUILDKITE_PIPELINE_
PROVIDER=github", "BUILDKITE_AGENT_EXPERIMENT=resolve-commit-after-checkout
,git-mirrors,output-redactor,ansi-timestamps", "BUILDKITE_CONFIG_PATH=/etc/
buildkite-agent/buildkite-agent.cfg", "BUILDKITE_PIPELINE_TEAMS=sciml-full-
access", "BUILDKITE_AGENT_META_DATA_CRYPTIC_CAPABLE=true", "BUILDKITE_AGENT
_ACCESS_TOKEN=YNYqp562hZCyEGk8vYtTj2EmY2HDf9AAdT2tXMTWL5ME7Tm3A6", "BUILDKI
TE_PLUGIN_CRYPTIC_BASE64_SIGNED_JOB_ID_SECRET=UwoEnyHfgFzE2xaNszIGqa+0o0iQf
yMCcyr/sK4qjKVXWaQmxMNuistdqsztBmhEfL1nzBJhWDx5504xg7q0zRXwMxP6/h9mNJoxZQE/
fxoeoJ6dFxZSMSHCCe+1jhxygG3Nc2mfvoQVIwA1qdarLYdpegXGX7v/d9MdhQOfnSUxEWDxfbq
RYAbk6vEsfOzrlK8lZftzmTMz50xWRFxe5Z5xKX0WrqbZ6KC2IUYt8NXSUKKVBIfxWuy1AETMTU
BWMrqm3daNf1o86iPPDZvKZuHE2Y6y6equJX2mzJ4L6Sz8UHld+I8NqnIj0KxE+MmHSQh0altk7
idBZHBtbd2kZQ==", "BUILDKITE_PLUGIN_COPPERMIND_S3_PREFIX=s3://julialang-bui
ldkite-artifacts/scimlbenchmarks", "BUILDKITE_BUILD_CREATOR_EMAIL=accounts@
chrisrackauckas.com", "XKB_CONFIG_ROOT=/cache/julia-buildkite-plugin/depots
/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/f8b49c7c45b400e3f5c4002d196
45d4b88712c0c/share/X11/xkb", "BUILDKITE_SSH_KEYSCAN=true", "BUILDKITE_UNBL
OCKER_EMAIL=accounts@chrisrackauckas.com", "BUILDKITE_PROJECT_SLUG=julialan
g/scimlbenchmarks-dot-jl", "BUILDKITE_INITIAL_JOB_ID=0189b355-3846-4774-bed
b-5ab473578463", "BUILDKITE_BIN_PATH=/usr/bin", "PWD=/cache/build/exclusive
-amdci3-0/julialang/scimlbenchmarks-dot-jl", "GRDIR=/cache/julia-buildkite-
plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4c9fec19de6bf8
14495fc8b030c8914d924707bd", "BUILDKITE_GIT_SUBMODULES=true", "BUILDKITE_AG
ENT_META_DATA_OS=linux", "CI=true", "BUILDKITE_STEP_KEY=benchmark-benchmark
s-DynamicalODE", "BUILDKITE_STEP_IDENTIFIER=benchmark-benchmarks-DynamicalO
DE", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_0=markdown/**/figures/*.png", "CO
NDA_PREFIX=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/conda/3/x86_64", "BUILDKITE_PLUGIN_COPPERMIND_INPUTS_2=./*.toml",
 "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS_3=pdf/**/*.pdf", "BUILDKITE_AGENT_PID
=2", "BUILDKITE_LAST_HOOK_EXIT_STATUS=0", "BUILDKITE_AGENT_META_DATA_EXCLUS
IVE=true", "BUILDKITE_BUILD_CREATOR=Christopher Rackauckas", "BUILDKITE_UNB
LOCKER=Christopher Rackauckas", "BUILDKITE_REBUILT_FROM_BUILD_ID=", "OPENBL
AS_DEFAULT_NUM_THREADS=1", "BUILDKITE_BRANCH=dynamical", "JULIA_DEPOT_PATH=
/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953",
 "BUILDKITE_AGENT_DEBUG_HTTP=false", "SHELL=/shells/bash", "BUILDKITE=true"
, "BUILDKITE_PLUGIN_CRYPTIC_PRIVILEGED=true", "BUILDKITE_HOOKS_PATH=/hooks"
, "BUILDKITE_PIPELINE_DEFAULT_BRANCH=master", "BUILDKITE_PLUGIN_NAME=COPPER
MIND", "BUILDKITE_REBUILT_FROM_BUILD_NUMBER=", "HOME=/root", "BUILDKITE_PLU
GIN_COPPERMIND_OUTPUTS_4=script/**/*.jl", "BUILDKITE_S3_DEFAULT_REGION=us-e
ast-1", "BUILDKITE_TRIGGERED_FROM_BUILD_PIPELINE_SLUG=", "BUILDKITE_BUILD_C
HECKOUT_PATH=/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-
jl", "BUILDKITE_SCRIPT_PATH=# Instantiate, to install the overall project d
ependencies, and `build()` for conda\necho \"--- Instantiate\"\njulia --pro
ject=. -e 'using Pkg; Pkg.instantiate(); Pkg.build()'\n\nif [[ benchmarks/D
ynamicalODE == *BayesianInference* ]]\nthen\n  export CMDSTAN_HOME=\"\$(pwd
)/cmdstan-2.29.2/\"\n  pwd\n  curl -LO https://github.com/stan-dev/cmdstan/
releases/download/v2.29.2/cmdstan-2.29.2.tar.gz\n  tar -xzpf cmdstan-2.29.2
.tar.gz --no-same-owner\n  ls -lia .\n  ls -lia ./cmdstan-2.29.2\n  ls -lia
 ./cmdstan-2.29.2/make\n  touch ./cmdstan-2.29.2/make/local\n  echo \"STAN_
THREADS=true\" > ./cmdstan-2.29.2/make/local\n  echo \"g++ version\"\n  ech
o \$(g++ -v)\n  cd cmdstan-2.29.2\n  make build\n  cd -\nfi\n\n# Run benchm
ark\necho \"+++ Run benchmark for benchmarks/DynamicalODE\"\njulia --thread
s=auto --project=. benchmark.jl \"benchmarks/DynamicalODE\"\n", "INVOCATION
_ID=b6aa8a3b5a6844989ba340052c33382e", "BUILDKITE_PLUGIN_COPPERMIND_OUTPUTS
_2=notebook/**/*.ipynb", "LANG=en_US.UTF-8", "BUILDKITE_PIPELINE_NAME=SciML
Benchmarks.jl", "SHLVL=2", "XDG_RUNTIME_DIR=/run/user/21581", "OLDPWD=/cach
e/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953", "FON
TCONFIG_FILE=/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f
4d2d937f953/artifacts/387d89822da323c098aba6f8ab316874d4e90f2e/etc/fonts/fo
nts.conf", "BUILDKITE_PLUGIN_CONFIGURATION={\"inputs\":[\"benchmarks/Dynami
calODE\",\"src/**/*.jl\",\"./*.toml\"],\"outputs\":[\"markdown/**/figures/*
.png\",\"markdown/**/*.md\",\"notebook/**/*.ipynb\",\"pdf/**/*.pdf\",\"scri
pt/**/*.jl\"],\"s3_prefix\":\"s3://julialang-buildkite-artifacts/scimlbench
marks\"}", "BUILDKITE_BUILD_PATH=/cache/build", "BUILDKITE_BUILD_AUTHOR_EMA
IL=accounts@chrisrackauckas.com", "BUILDKITE_TIMEOUT=4000", "BUILDKITE_GIT_
MIRRORS_PATH=/cache/repos", "BUILDKITE_LABEL=:hammer: benchmarks/DynamicalO
DE", "FONTCONFIG_PATH=/cache/julia-buildkite-plugin/depots/5b300254-1738-49
89-ae0a-f4d2d937f953/artifacts/387d89822da323c098aba6f8ab316874d4e90f2e/etc
/fonts", "JOURNAL_STREAM=8:246315495", "BUILDKITE_AGENT_META_DATA_SANDBOX_J
L=true", "BUILDKITE_BUILD_CREATOR_TEAMS=juliagpu-full-access:sciml-full-acc
ess", "BUILDKITE_PROJECT_PROVIDER=github", "BUILDKITE_S3_SECRET_ACCESS_KEY=
a/TQQ97Ej4UlsOOmbepu0WD6HaewYkUmV9g7Y6kb", "GKS_USE_CAIRO_PNG=true", "BUILD
KITE_REPO_MIRROR=/cache/repos/https---github-com-SciML-SciMLBenchmarks-jl-g
it", "JULIA_IMAGE_THREADS=1", "GIT_TERMINAL_PROMPT=0", "BUILDKITE_JOB_ID=01
89b404-9636-43e2-b3c4-ee575c16da70", "SYSTEMD_EXEC_PID=1898632", "BUILDKITE
_REDACTED_VARS=*_PASSWORD,*_SECRET,*_TOKEN,*_ACCESS_KEY,*_SECRET_KEY", "BUI
LDKITE_BUILD_AUTHOR=Christopher Rackauckas", "USER=sabae", "GKSwstype=100",
 "BUILDKITE_REPO=https://github.com/SciML/SciMLBenchmarks.jl.git", "BUILDKI
TE_GIT_FETCH_FLAGS=-v --prune --tags", "BUILDKITE_LOCAL_HOOKS_ENABLED=true"
, "MANAGERPID=7863", "BUILDKITE_S3_ACCESS_KEY_ID=AKIA4WZGSTHCCMX6VMOX", "BU
ILDKITE_GIT_MIRRORS_LOCK_TIMEOUT=300", "BUILDKITE_AGENT_ID=0189b487-08dd-41
ba-9ae1-4a3b82b2ebdb", "BUILDKITE_GIT_MIRRORS_SKIP_UPDATE=false", "BUILDKIT
E_ARTIFACT_PATHS=", "BUILDKITE_BUILD_URL=https://buildkite.com/julialang/sc
imlbenchmarks-dot-jl/builds/1346", "BUILDKITE_MESSAGE=bump dynamical ode be
nchmarks", "BUILDKITE_RETRY_COUNT=0", "LOGNAME=sabae", "JULIA_CPU_THREADS=1
28", "BUILDKITE_PLUGIN_JULIA_CACHE_DIR=/cache/julia-buildkite-plugin", "BUI
LDKITE_PLUGIN_COPPERMIND_INPUT_HASH=3c6ff94a7c183013a483915718b9b7f3cb99abb
bf077bffd9c81407f1b35d25f", "BUILDKITE_PLUGIN_CRYPTIC_BASE64_AGENT_PRIVATE_
KEY_SECRET=LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcFFJQkFBS0NBUUVB
dFhzSjMxRm0xSjdvSGM5b3hmWllKY3FxRk5yaXRMUUhxaDVJUHNGS3YySis1K1FVCkJGTWFEYzB
yN3M2dzQzUjAxaG5FTU9ZWDQwK3lFQ094eWxxK1o3dHdxWlNxS2U1MThwc0RyeWRna0xJNzRnQU
UKWVljU2Rnb0xreGFqVjcva29IRUw4K3M0SnRUVTVCendURXVwTllTZGNQOFhQSmJLekY1RE5qd
WJmeFA5ZjdSNwpseklMdjVpdmdpcWU1bW1MRndZcG9IU0VRVTV0ZU9PSEErS2NjUDZ3K2d1Q0Vx
MUZFb0N2MDRyaTFXaWpVZXorCjMrRFlTOFArUThkcTBiWFFmVEtVcnNLYTJHZy81ZmdIeWdEdWR
HT2ZsdzUvVEljR3VVbGNsd1hZb2tTRkpSWUIKRWtqVDlwQmdiTTRHMi9rVzRhZmd3eGxzbjd1bF
FuUDQ2VS8xZFFJREFRQUJBb0lCQVFDTU5sVjRUbUlPWC8raQpHSDh3ZzVXekRSTy9MU1greXlFb
zFHQ282NW9lcDdDNDVNUjZXdUpFUzRKbjdSVkpoczVHSkg0cDhYdi9TYkdmCk9wVEFiTCt6VVdS
SUFPNC9tMWRSYTJhN1NzY1d4RDN6N0dOMkhtK3E5elBlSHAxd3pIZU5aZ29BR0htM3RyUU0KMGp
idUczN09OSG1YdGQ1MEYyVHo1TmcwN0hURkJwV3hMMjJwNm9aZzgyUEk0OXIrdUpWWmZ5MU5HZV
RnaFA4cgp2dVRVTVJIcldZa25YbUR1eDVSMHNIdDFoU2FvTXBFbSsrMWc1V09rSzZDTGFJbEV0Z
itWVVBvR0piYlNYRzNJCmo5N1h5a3NGUDhGZ24wMWx4ZktGV1p4MXlnTVdsUm00SFNCTWVkc1Fp
WStqeG5Sd3BtRnh5L2pIOVhFTUppT0wKQSsvVFdCbUJBb0dCQU52cXROQjRuVS9zODIxTU9oalR
PTmtFOGNJOENxV1BRRTZXcEFOYVY5UkVVcGVCZzhmMgpjTXg2VFMweE9EU0JRUk9PRnNEUG9pc1
E0VVRGQmFialBLTU41d2IzNFVKYmN4V0ZwcGlLUHJMa09Zdmtqb01VCkNSb1pKK05Lb253RWh5b
WJ0TG0yMHhmUUZCamY1R1QvMHJZUWcxUkN1OVllSmE0Z3NWYWtSNGh4QW9HQkFOTkIKMzhxenJh
TTBDeHhrSnZkVmlTY3g2MFowN2g4OHFISy9SV2dVK3ZZaitwZy9ibXBwMmdiUXZkYTJxaS84UEl
2OApSb0JwcmY2Y285TmdSa2JmY05xZmFnM1Z5SDhBNW1iUE1nK0s4YmFuKzlwU003WkREVW1sdU
03R3ZRSW5OVnBCCnBJcE1uWEk5eDZSSFlpOFF2MHhXOXcyUmpmS09TbElYZFlITjZwOUZBb0dCQ
Up0NXdwMkVPRXR5VE9NZnVnOGsKL1pMSVlSY2VGYlRZb3ZFc3BRWE4wRHc4bFZ1UmNCWmx6M2R3
bTdGd2s3ampESndEbjJodklzcHBvNmxYMVZnWQpYUjAxemZocU5QSVI3em52QkVuaHF0UVViKzd
NQmtqN1dEZ0FRdWY1TXdpVXR1NGVxOVdFUUpjY1A2a2FXTUZpCjc1aFI4bGNXMnU5VTN2VE5Iak
1QNzVheEFvR0JBSm5HdExsZlMwQ21USVF4SHZBaE1rSDJvMVZaSGxCY25oMVEKdjV3QTBhRkVGV
kNuczU4QVNEVjMwd2d0VlBxeTkvdkoraVBWU1ZNeUFFcUlKUC9IKytVWDcySDh3UUk1ekh6Lwp5
MmZtOHdYTGg1ZW5DSDllbFppTGFsZ1I4RmxWNHc4OUF5R3NuVnNnUDJlRWtxTEI1UTRUcTZnVDB
LakVETE51CjRobEhvOGFsQW9HQUhBVGltTGRkS0JFTkN2MXZyNnZ0d3JCZGRCbWlGSWFwaVcvMk
5acWxCTFp1bEp6MEwzdCsKM3FvSUF0Uisxd2xpWkQwZGJnRGdVeVRMcnN5Y1RDSkZIczNIZTFXb
3NCSzcxTmlncFZhWEVzWnFpOHNENjlvUQo2QkFnaEdvbnNGbTEydzhhRGNDdm92WUxLTlhVV1BF
T1c0akdvd2k0Tmx4NGZidHlkYXpIUEdnPQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo
=", "BUILDKITE_PIPELINE_SLUG=scimlbenchmarks-dot-jl", "BUILDKITE_SHELL=/bin
/bash -e -c", "BUILDKITE_UNBLOCKER_ID=e8e997cc-aa1b-4cc5-8244-2786f678c8d6"
, "BUILDKITE_PLUGINS_ENABLED=true", "LANGUAGE=en_US", "GKS_FONTPATH=/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/4c9fec19de6bf814495fc8b030c8914d924707bd", "BUILDKITE_AGENT_NAME=exclusiv
e-amdci3.0", "BUILDKITE_STEP_ID=0189b404-961a-4366-af90-161c1a6ab045", "BUI
LDKITE_PLUGIN_COPPERMIND_OUTPUTS_1=markdown/**/*.md", "BUILDKITE_TAG=", "OP
ENBLAS_MAIN_FREE=1", "PATH=/cache/julia-buildkite-plugin/julia_installs/bin
/linux/x64/1.9/julia-1.9-latest-linux-x86_64/bin:/usr/local/sbin:/usr/local
/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/u
sr/bin:/usr/local/bin:/usr/bin:/bin", "GKS_ENCODING=utf8", "BUILDKITE_AGENT
_META_DATA_NUM_CPUS=128", "BUILDKITE_TRIGGERED_FROM_BUILD_NUMBER=", "BUILDK
ITE_AGENT_META_DATA_CONFIG_GITSHA=51f06640", "BUILDKITE_COMMAND=# Instantia
te, to install the overall project dependencies, and `build()` for conda\ne
cho \"--- Instantiate\"\njulia --project=. -e 'using Pkg; Pkg.instantiate()
; Pkg.build()'\n\nif [[ benchmarks/DynamicalODE == *BayesianInference* ]]\n
then\n  export CMDSTAN_HOME=\"\$(pwd)/cmdstan-2.29.2/\"\n  pwd\n  curl -LO 
https://github.com/stan-dev/cmdstan/releases/download/v2.29.2/cmdstan-2.29.
2.tar.gz\n  tar -xzpf cmdstan-2.29.2.tar.gz --no-same-owner\n  ls -lia .\n 
 ls -lia ./cmdstan-2.29.2\n  ls -lia ./cmdstan-2.29.2/make\n  touch ./cmdst
an-2.29.2/make/local\n  echo \"STAN_THREADS=true\" > ./cmdstan-2.29.2/make/
local\n  echo \"g++ version\"\n  echo \$(g++ -v)\n  cd cmdstan-2.29.2\n  ma
ke build\n  cd -\nfi\n\n# Run benchmark\necho \"+++ Run benchmark for bench
marks/DynamicalODE\"\njulia --threads=auto --project=. benchmark.jl \"bench
marks/DynamicalODE\"\n", "BUILDKITE_PLUGIN_VALIDATION=false", "BUILDKITE_AG
ENT_META_DATA_ARCH=x86_64", "CONDARC=/cache/julia-buildkite-plugin/depots/5
b300254-1738-4989-ae0a-f4d2d937f953/conda/3/x86_64/condarc-julia.yml", "FOR
CE_SANDBOX_MODE=unprivileged", "BUILDKITE_TRIGGERED_FROM_BUILD_ID=", "TERM=
xterm-256color", "PYTHONIOENCODING=UTF-8", "BUILDKITE_PULL_REQUEST_BASE_BRA
NCH=master", "BUILDKITE_PIPELINE_ID=5b300254-1738-4989-ae0a-f4d2d937f953", 
"_=/cache/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.9/julia-1.9
-latest-linux-x86_64/bin/julia", "BUILDKITE_BUILD_ID=0189b355-3820-4fba-9ca
c-646cb831ed20", "BUILDKITE_AGENT_ENDPOINT=https://agent.buildkite.com/v3",
 "BUILDKITE_PLUGINS=[{\"github.com/staticfloat/cryptic-buildkite-plugin#v2\
":{\"variables\":[\"BUILDKITE_S3_ACCESS_KEY_ID=\\\"U2FsdGVkX19PKaFjxITJxFlY
66D/vA9T/y/MzZ2Fs+bpWmlhRF0g5DbemJbuKKw9\\\"\",\"BUILDKITE_S3_SECRET_ACCESS
_KEY=\\\"U2FsdGVkX19MkMLkCxkclrpxYMfzHvYlUJssaSbX/wkdNiL+H5/aHwiYiTsBXyXh2m
/1pcIUMHCR0nerHevB8g==\\\"\",\"BUILDKITE_S3_DEFAULT_REGION=\\\"U2FsdGVkX18c
coE9FmtkwsCm1x0MLMBlN/FLcAyKkY4=\\\"\"]}},{\"github.com/JuliaCI/julia-build
kite-plugin#v1\":{\"version\":1.9}},{\"github.com/staticfloat/sandbox-build
kite-plugin\":{\"rootfs_url\":\"https://github.com/ven-k/Placeholder/releas
es/download/v0.23.0/aws_uploader.x86_64.tar.gz\",\"workspaces\":[\"/cache/j
ulia-buildkite-plugin:/cache/julia-buildkite-plugin\"],\"rootfs_treehash\":
\"d46b35aa927024de8729d63fde18442a0a590e62\"}},{\"github.com/staticfloat/co
ppermind-buildkite-plugin#v1\":{\"inputs\":[\"benchmarks/DynamicalODE\",\"s
rc/**/*.jl\",\"./*.toml\"],\"outputs\":[\"markdown/**/figures/*.png\",\"mar
kdown/**/*.md\",\"notebook/**/*.ipynb\",\"pdf/**/*.pdf\",\"script/**/*.jl\"
],\"s3_prefix\":\"s3://julialang-buildkite-artifacts/scimlbenchmarks\"}}]",
 "SANDBOX_PERSISTENCE_DIR=/cache/sandbox_persistence", "BUILDKITE_GIT_CLEAN
_FLAGS=-ffxdq", "BUILDKITE_COMMIT=05f6e96a251342af162a7dc3d9210deff105c37e"
, "BUILDKITE_PULL_REQUEST=616", "GKS_QT=env LD_LIBRARY_PATH=/cache/julia-bu
ildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7661e5
a9aa217ce3c468389d834a4fb43b0911e8/lib:/cache/julia-buildkite-plugin/depots
/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/e63503984ff7722ba80209eddd5
621acca0d2d5e/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-a
e0a-f4d2d937f953/artifacts/d00220164876dea2cb19993200662745eed5e2db/lib:/ca
che/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.9/julia-1.9-lates
t-linux-x86_64/bin/../lib/julia:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/f792596249694cc12db3689d386d3a6c5d
24e794/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/dc526f26fb179a3f68eb13fcbe5d2d2a5aa7eeac/lib:/cache/jul
ia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/5
a508a62784097dab7c7ae5805f2c89d2cc97397/lib:/cache/julia-buildkite-plugin/d
epots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/1cfe0ebb804cb8b0d7d1e8
f98e5cda94b2b31b3d/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/909c4b91b22279c16eaf5a1de24767fd30e25f28/li
b:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
/artifacts/ddfc455343aff48d27c1b39d7fcb07e0d9242b50/lib:/cache/julia-buildk
ite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4fff8dd69d
1996234cc96473780198ebc7b3809c/lib:/cache/julia-buildkite-plugin/depots/5b3
00254-1738-4989-ae0a-f4d2d937f953/artifacts/2b77b304b0975d15bd5aeb4d1d5097a
c6256ea3c/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-
f4d2d937f953/artifacts/2bce74229f51de64c33433272240af5734619b33/lib:/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/387d89822da323c098aba6f8ab316874d4e90f2e/lib:/cache/julia-buildkite-plugi
n/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/16154f990153825ec24
b52aac11165df2084b9dc/lib:/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/92111ef825c608ea220f8e679dd8d908d7ac5b83
/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/artifacts/f3ec73d7bf2f4419ba0943e94f7738cf56050797/lib:/cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/bd965e3
c7f9460155f06361da380c63fa0351ef6/lib:/cache/julia-buildkite-plugin/depots/
5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/060cf7829c3363638c29228ea4ab
0bd033d8eab0/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae
0a-f4d2d937f953/artifacts/066b74f697b047bab2476f57bb0c7a29bead921c/lib:/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/1e69ef9fbf05e2896d3cb70eac8080c4d10f8696/lib:/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/fc6071b99b67da0a
e4e49ebab70c369ce9a76c9e/lib:/cache/julia-buildkite-plugin/depots/5b300254-
1738-4989-ae0a-f4d2d937f953/artifacts/527e66fb9b12dfd1f58157fe0b3fd52b84062
432/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d9
37f953/artifacts/921a059ebce52878d7a7944c9c345327958d1f5b/lib:/cache/julia-
buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/8b02
84dc2781b9481ff92e281f1db532d8421040/lib:/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/d11639e2a53726f2593e25ba9
8ed7b416f62bbc5/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989
-ae0a-f4d2d937f953/artifacts/62c010876222f83fe8878bf2af0e362083d20ee3/lib:/
cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/ar
tifacts/ee20a84d0166c074dfa736b642902dd87b4da48d/lib:/cache/julia-buildkite
-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/459252c01ffcd
08700841efdd4b6d3edfe5916e7/lib:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/cc415631aeb190b075329ce756f690a90e
1f873b/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/e19f3bb2eef5fb956b672235ea5323b5be9a0626/lib:/cache/jul
ia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/0
631e2a6a31b5692eec7a575836451b16b734ec0/lib:/cache/julia-buildkite-plugin/d
epots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/89ed5dda220da4354ada19
70107e13679914bbbc/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/595f9476b128877ab5bf73883ff6c8dc8dacfe66/li
b:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
/artifacts/587de110e5f58fd435dc35b294df31bb7a75f692/lib:/cache/julia-buildk
ite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/fc239b3ff5
739aeab252bd154fa4dd045fefe629/lib:/cache/julia-buildkite-plugin/depots/5b3
00254-1738-4989-ae0a-f4d2d937f953/artifacts/46d0dbb4a9ceba3eef765dd6b9b6674
ac2d84863/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-
f4d2d937f953/artifacts/eff86eedadb59cff1a61399e3242b3f529ca6f59/lib:/cache/
julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifact
s/b409c0eafb4254a980f9e730f6fbe56867890f6a/lib:/cache/julia-buildkite-plugi
n/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/37dda4e57d9de95c99d
1f8c6b3d8f4eca88c39a2/lib:/cache/julia-buildkite-plugin/depots/5b300254-173
8-4989-ae0a-f4d2d937f953/artifacts/f0d193662fead3500b523f94b4f1878daab59a93
/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f
953/artifacts/05616da88f6b36c7c94164d4070776aef18ce46b/lib:/cache/julia-bui
ldkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/2df316d
a869cd97f7d70029428ee1e2e521407cd/lib:/cache/julia-buildkite-plugin/depots/
5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/7190f0cb0832b80761cc6d513dd9
b935f3e26358/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae
0a-f4d2d937f953/artifacts/4daa3879a820580557ef34945e2ae243dfcbba11/lib:/cac
he/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artif
acts/5aa80c7b8e919cbfee41019069d9b25269befe10/lib:/cache/julia-buildkite-pl
ugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/22a0f38792d33b65
e35189101e60e1fab0a56030/lib:/cache/julia-buildkite-plugin/depots/5b300254-
1738-4989-ae0a-f4d2d937f953/artifacts/694cae97bb3cbf8f1f73f2ecabd891602ccf1
751/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d9
37f953/artifacts/214deacf44273474118c5fe83871fdfa8039b4ad/lib:/cache/julia-
buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/00e6
77e85b66b11c69044b970c5a7d6d8c1534e9/lib:/cache/julia-buildkite-plugin/depo
ts/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/e18a60e84fd8aefa967cb9f1d
11dc6cbd9cac88b/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989
-ae0a-f4d2d937f953/artifacts/cacd8c147f866d6672e1aca9bb01fb919a81e96a/lib:/
cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/ar
tifacts/b7dc5dce963737414a564aca8d4b82ee388f4fa1/lib:/cache/julia-buildkite
-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/0d364e900393f
710a03a5bafe2852d76e4d2c2cd/lib:/cache/julia-buildkite-plugin/depots/5b3002
54-1738-4989-ae0a-f4d2d937f953/artifacts/79cc5446ced978de84b6e673e01da0ebfd
d6e4a5/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d
2d937f953/artifacts/1a2adcee7d99fea18ead33c350332626b262e29a/lib:/cache/jul
ia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/9
d7f5887309a96013a2c75f48b5e137e60ccae8f/lib:/cache/julia-buildkite-plugin/d
epots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/74fd55820a62aa47ebf4d9
42aa59096980a1851d/lib:/cache/julia-buildkite-plugin/depots/5b300254-1738-4
989-ae0a-f4d2d937f953/artifacts/4443e44120d70a97f8094a67268a886256077e69/li
b:/cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
/artifacts/9a539db13549d6713ba69383580253e9fb34f487/lib:/cache/julia-buildk
ite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953/artifacts/4c9fec19de
6bf814495fc8b030c8914d924707bd/lib:/cache/julia-buildkite-plugin/julia_inst
alls/bin/linux/x64/1.9/julia-1.9-latest-linux-x86_64/bin/../lib/julia:/cach
e/julia-buildkite-plugin/julia_installs/bin/linux/x64/1.9/julia-1.9-latest-
linux-x86_64/bin/../lib /cache/julia-buildkite-plugin/depots/5b300254-1738-
4989-ae0a-f4d2d937f953/artifacts/4c9fec19de6bf814495fc8b030c8914d924707bd/b
in/gksqt"]), ProcessExited(1)) [1]

during initialization of module PyPlot
```





## Tests

```julia
# Single pendulum

k = rand()
integrator = VelocityVerlet()
Δt = 0.1
singlependulum(k, integrator, Δt, t0=-20.0, t1=20.0)
```

```
Error: UndefVarError: `singlependulum` not defined
```



```julia
# Two single pendulums

H(q,p,param) = sum(p.^2/2 .- cos.(q) .+ 1)
q0 = pi*rand(2)
p0 = zeros(2)
t0, t1 = -20.0, 20.0
prob = HamiltonianProblem(H, q0, p0, (t0, t1))

integrator = McAte4()
Δt = 0.1
sol = solve(prob, integrator, dt=Δt)
@time sol = solve(prob, integrator, dt=Δt)

sleep(0.1)
plotsolenergy(H, integrator, Δt, sol)
```

```
0.001152 seconds (13.70 k allocations: 1.295 MiB)
Error: UndefVarError: `plotsolenergy` not defined
```





## Comparison of symplectic Integrators

```julia
SymplecticIntegrators = [
    SymplecticEuler(),
    VelocityVerlet(),
    VerletLeapfrog(),
    PseudoVerletLeapfrog(),
    McAte2(),
    Ruth3(),
    McAte3(),
    CandyRoz4(),
    McAte4(),
    CalvoSanz4(),
    McAte42(),
    McAte5(),
    Yoshida6(),
    KahanLi6(),
    McAte8(),
    KahanLi8(),
    SofSpa10(),
]

k = 0.999
Δt = 0.1
for integrator in SymplecticIntegrators
    singlependulum(k, integrator, Δt)
end
```

```
Error: UndefVarError: `singlependulum` not defined
```



```julia
k = 0.999
Δt = 0.01
for integrator in SymplecticIntegrators[1:4]
    singlependulum(k, integrator, Δt)
end
```

```
Error: UndefVarError: `singlependulum` not defined
```



```julia
k = 0.999
Δt = 0.001
singlependulum(k, SymplecticEuler(), Δt)
```

```
Error: UndefVarError: `singlependulum` not defined
```



```julia
k = 0.999
Δt = 0.0001
singlependulum(k, SymplecticEuler(), Δt)
```

```
Error: UndefVarError: `singlependulum` not defined
```




## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/DynamicalODE","single_pendulums.jmd")
```

Computer Information:

```
Julia Version 1.9.2
Commit e4ee485e909 (2023-07-05 09:39 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver2)
  Threads: 128 on 128 virtual cores
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953
  JULIA_IMAGE_THREADS = 1

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/DynamicalODE/Project.toml`
  [459566f4] DiffEqCallbacks v2.27.0
  [055956cb] DiffEqPhysics v3.11.0
  [b305315f] Elliptic v1.0.1
  [1dea7af3] OrdinaryDiffEq v6.53.4
  [65888b18] ParameterizedFunctions v5.15.0
  [91a5bcdd] Plots v1.38.17
  [d330b81b] PyPlot v2.11.1
  [31c91b34] SciMLBenchmarks v0.1.3
  [90137ffa] StaticArrays v1.6.2
  [92b13dbe] TaylorIntegration v0.14.2
  [37e2e46d] LinearAlgebra
  [de0858da] Printf
  [10745b16] Statistics v1.9.0
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/DynamicalODE/Manifest.toml`
  [47edcb42] ADTypes v0.1.6
  [c3fe647b] AbstractAlgebra v0.31.0
  [1520ce14] AbstractTrees v0.4.4
  [79e6a3ab] Adapt v3.6.2
  [ec485272] ArnoldiMethod v0.2.0
  [4fba245c] ArrayInterface v7.4.11
  [30b0a656] ArrayInterfaceCore v0.1.29
  [6e4b80f9] BenchmarkTools v1.3.2
  [e2ed5e7c] Bijections v0.1.4
  [d1d4a3ce] BitFlags v0.1.7
  [62783981] BitTwiddlingConvenienceFunctions v0.1.5
  [2a0fbf3d] CPUSummary v0.2.3
  [00ebfdb7] CSTParser v3.3.6
  [49dc2e85] Calculus v0.5.1
  [d360d2e6] ChainRulesCore v1.16.0
  [fb6a15b2] CloseOpenIntervals v0.1.12
  [944b1d66] CodecZlib v0.7.2
  [35d6a980] ColorSchemes v3.22.0
  [3da002f7] ColorTypes v0.11.4
  [c3611d14] ColorVectorSpace v0.10.0
  [5ae59095] Colors v0.12.10
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.12
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v4.8.0
  [b152e2b5] CompositeTypes v0.1.3
  [f0e56b4a] ConcurrentUtilities v2.2.1
  [8f4d0f93] Conda v1.9.1
  [187b0558] ConstructionBase v1.5.3
  [d38c429a] Contour v0.6.2
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.15.0
  [864edb3b] DataStructures v0.18.14
  [e2d170a0] DataValueInterfaces v1.0.0
  [8bb1440f] DelimitedFiles v1.9.1
  [2b5f629d] DiffEqBase v6.127.0
  [459566f4] DiffEqCallbacks v2.27.0
  [055956cb] DiffEqPhysics v3.11.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
  [b4f34e82] Distances v0.10.9
  [31c24e10] Distributions v0.25.98
  [ffbed154] DocStringExtensions v0.9.3
  [5b8099bc] DomainSets v0.6.7
  [fa6b7ba4] DualNumbers v0.6.8
  [7c1d4256] DynamicPolynomials v0.5.2
  [b305315f] Elliptic v1.0.1
  [4e289a0a] EnumX v1.0.4
  [6912e4f1] Espresso v0.6.1
  [460bff9d] ExceptionUnwrapping v0.1.9
  [d4d017d3] ExponentialUtilities v1.24.0
  [e2ba6199] ExprTools v0.1.10
  [c87230d0] FFMPEG v0.4.1
  [7034ab61] FastBroadcast v0.2.6
  [9aa1b823] FastClosures v0.3.2
  [29a986be] FastLapackInterface v2.0.0
  [1a297f60] FillArrays v1.5.0
  [6a86dc24] FiniteDiff v2.21.1
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.35
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [46192b85] GPUArraysCore v0.1.5
  [28b8d3ca] GR v0.72.9
  [c145ed77] GenericSchur v0.5.3
  [d7ba0133] Git v1.3.0
  [c27321d9] Glob v1.3.1
  [86223c79] Graphs v1.8.0
  [42e2da0e] Grisu v1.0.2
  [0b43b601] Groebner v0.4.2
  [d5909c97] GroupsCore v0.4.0
  [cd3eb016] HTTP v1.9.14
  [eafb193a] Highlights v0.5.2
  [3e5b6fbb] HostCPUFeatures v0.1.15
  [34004b35] HypergeometricFunctions v0.3.23
  [7073ff75] IJulia v1.24.2
  [615f187c] IfElse v0.1.1
  [d25df0c9] Inflate v0.1.3
  [18e54dd8] IntegerMathUtils v0.1.2
  [8197267c] IntervalSets v0.7.7
  [92d709cd] IrrationalConstants v0.2.2
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.5
  [692b3bcd] JLLWrappers v1.4.1
  [682c06a0] JSON v0.21.4
  [98e50ef6] JuliaFormatter v1.0.34
  [ccbc3e58] JumpProcesses v9.7.2
  [ef3ab10e] KLU v0.4.0
  [ba0b0d4f] Krylov v0.9.2
  [b964fa9f] LaTeXStrings v1.3.0
  [2ee39098] LabelledArrays v1.14.0
  [984bce1d] LambertW v0.4.6
⌅ [23fbe1c1] Latexify v0.15.21
  [10f19ff3] LayoutPointers v0.1.14
  [50d2b5c4] Lazy v0.15.1
  [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [d3d80556] LineSearches v7.2.0
  [7ed4a6bd] LinearSolve v2.4.2
  [2ab3a3ac] LogExpFunctions v0.3.24
  [e6f89c97] LoggingExtras v1.0.0
  [bdcacae8] LoopVectorization v0.12.165
  [d8e11817] MLStyle v0.4.17
  [1914dd2f] MacroTools v0.5.10
  [d125e4d3] ManualMemory v0.1.8
  [739be429] MbedTLS v1.1.7
  [442fdcdd] Measures v0.3.2
  [e1d29d7a] Missings v1.1.0
  [961ee093] ModelingToolkit v8.64.0
  [46d2c3a1] MuladdMacro v0.2.4
  [102ac46a] MultivariatePolynomials v0.5.1
  [ffc61752] Mustache v1.0.17
  [d8a4904e] MutableArithmetics v1.3.0
  [d41bc354] NLSolversBase v7.8.3
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.0.2
  [8913a72c] NonlinearSolve v1.9.0
  [6fe1bfb0] OffsetArrays v1.12.10
  [4d8831e6] OpenSSL v1.4.1
  [bac558e1] OrderedCollections v1.6.2
  [1dea7af3] OrdinaryDiffEq v6.53.4
  [90014a1f] PDMats v0.11.17
  [65ce6f38] PackageExtensionCompat v1.0.0
  [65888b18] ParameterizedFunctions v5.15.0
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.7.2
  [b98c9c47] Pipe v1.3.0
  [32113eaa] PkgBenchmark v0.2.12
  [ccf2f8ad] PlotThemes v3.1.0
  [995b91a9] PlotUtils v1.3.5
  [91a5bcdd] Plots v1.38.17
  [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.5
  [1d0040c9] PolyesterWeave v0.2.1
  [d236fae5] PreallocationTools v0.4.12
  [aea7be01] PrecompileTools v1.1.2
  [21216c6a] Preferences v1.4.0
  [27ebfcd6] Primes v0.5.4
  [33c8b6b6] ProgressLogging v0.1.4
  [438e738f] PyCall v1.96.1
  [d330b81b] PyPlot v2.11.1
  [1fd47b50] QuadGK v2.8.2
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.5.3
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v2.38.7
  [f2c3362d] RecursiveFactorization v0.2.18
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.0
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.1
  [7e49a35a] RuntimeGeneratedFunctions v0.5.11
  [fdea26ae] SIMD v3.4.5
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.39
  [0bca4576] SciMLBase v1.94.0
  [31c91b34] SciMLBenchmarks v0.1.3
  [e9a6253c] SciMLNLSolve v0.1.8
  [c0aeaf25] SciMLOperators v0.3.6
  [6c6a2e73] Scratch v1.2.0
  [efcf1570] Setfield v1.1.1
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.1.0
  [727e6d20] SimpleNonlinearSolve v0.1.19
  [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [66db9d55] SnoopPrecompile v1.0.3
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.1.1
  [47a9eef4] SparseDiffTools v2.4.1
  [e56a9233] Sparspak v0.3.9
  [276daf66] SpecialFunctions v2.3.0
  [aedffcd0] Static v0.8.8
  [0d7ed370] StaticArrayInterface v1.4.0
  [90137ffa] StaticArrays v1.6.2
  [1e83bf80] StaticArraysCore v1.4.2
  [82ae8749] StatsAPI v1.6.0
  [2913bbd2] StatsBase v0.34.0
  [4c63d2b9] StatsFuns v1.3.0
  [7792a7ef] StrideArraysCore v0.4.17
  [69024149] StringEncodings v0.3.7
  [2efcf032] SymbolicIndexingInterface v0.2.2
  [d1185830] SymbolicUtils v1.2.0
  [0c5d862f] Symbolics v5.5.1
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.10.1
  [92b13dbe] TaylorIntegration v0.14.2
  [6aa5eb33] TaylorSeries v0.15.2
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
  [8290d209] ThreadingUtilities v0.5.2
  [a759f4b9] TimerOutputs v0.5.23
  [0796e94c] Tokenize v0.5.25
  [3bb67fe8] TranscodingStreams v0.9.13
  [a2a6695c] TreeViews v0.3.0
  [d5829a12] TriangularSolve v0.1.19
  [410a4b4d] Tricks v0.1.7
  [781d530d] TruncatedStacktraces v1.4.0
  [5c2747f8] URIs v1.4.2
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
  [1986cc42] Unitful v1.16.0
  [45397f5d] UnitfulLatexify v1.6.3
  [a7c27f48] Unityper v0.1.5
  [41fe7b60] Unzip v0.2.0
  [3d5dd08c] VectorizationBase v0.21.64
  [81def892] VersionParsing v1.3.0
  [19fa3120] VertexSafeGraphs v0.2.0
  [44d3d7a6] Weave v0.10.12
  [ddb6d928] YAML v0.4.9
  [c2297ded] ZMQ v1.2.2
  [700de1a5] ZygoteRules v0.2.3
  [6e34b625] Bzip2_jll v1.0.8+0
  [83423d85] Cairo_jll v1.16.1+1
  [2e619515] Expat_jll v2.5.0+0
⌃ [b22a6f82] FFMPEG_jll v4.4.2+2
  [a3f928ae] Fontconfig_jll v2.13.93+0
  [d7e528f0] FreeType2_jll v2.13.1+0
  [559328eb] FriBidi_jll v1.0.10+0
  [0656b61e] GLFW_jll v3.3.8+0
  [d2c73de3] GR_jll v0.72.9+0
  [78b55507] Gettext_jll v0.21.0+0
  [f8c6e375] Git_jll v2.36.1+2
  [7746bdde] Glib_jll v2.74.0+2
  [3b182d85] Graphite2_jll v1.3.14+0
  [2e76f6c2] HarfBuzz_jll v2.8.1+1
  [aacddb02] JpegTurbo_jll v2.1.91+0
  [c1c5ebd0] LAME_jll v3.100.1+0
  [88015f11] LERC_jll v3.0.0+1
  [1d63c593] LLVMOpenMP_jll v15.0.4+0
  [dd4b983a] LZO_jll v2.10.1+0
⌅ [e9f186c6] Libffi_jll v3.2.2+1
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.6.0+0
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+2
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.5.1+1
  [38a345b3] Libuuid_jll v2.36.0+0
  [e7412a2a] Ogg_jll v1.3.5+1
⌅ [458c3c95] OpenSSL_jll v1.1.21+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [91d4177d] Opus_jll v1.3.2+0
  [30392449] Pixman_jll v0.42.2+0
  [c0090381] Qt6Base_jll v6.4.2+3
  [f50d1b31] Rmath_jll v0.4.0+0
  [a2964d1f] Wayland_jll v1.21.0+0
  [2381bf8a] Wayland_protocols_jll v1.25.0+0
  [02c8fc9c] XML2_jll v2.10.3+0
  [aed1982a] XSLT_jll v1.1.34+0
  [ffd25f8a] XZ_jll v5.4.3+1
  [4f6342f7] Xorg_libX11_jll v1.8.6+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.11+0
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.4+0
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.1+0
  [c7cfdc94] Xorg_libxcb_jll v1.15.0+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.2+0
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.6+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.39.0+0
  [c5fb5394] Xorg_xtrans_jll v1.5.0+0
  [8f1865be] ZeroMQ_jll v4.3.4+0
  [3161d3a3] Zstd_jll v1.5.5+0
⌅ [214eeab7] fzf_jll v0.29.0+0
  [a4ae2306] libaom_jll v3.4.0+0
  [0ac62f75] libass_jll v0.15.1+0
  [f638f0a6] libfdk_aac_jll v2.0.2+0
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.7+1
  [1270edf5] x264_jll v2021.5.5+0
  [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.4.1+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [b27032c2] LibCURL v0.6.3
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.9.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics v1.9.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.0.2+0
  [deac9b47] LibCURL_jll v7.84.0+0
  [29816b5a] LibSSH2_jll v1.10.2+0
  [c8ffd9c3] MbedTLS_jll v2.28.2+0
  [14a3606d] MozillaCACerts_jll v2022.10.11
  [4536629a] OpenBLAS_jll v0.3.21+4
  [05823500] OpenLibm_jll v0.8.1+0
  [efcefdf7] PCRE2_jll v10.42.0+0
  [bea87d4a] SuiteSparse_jll v5.10.1+6
  [83775a58] Zlib_jll v1.2.13+0
  [8e850b90] libblastrampoline_jll v5.8.0+0
  [8e850ede] nghttp2_jll v1.48.0+0
  [3f19e933] p7zip_jll v17.4.0+0
Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

