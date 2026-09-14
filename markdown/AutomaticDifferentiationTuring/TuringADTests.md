---
author: "Penelope Yong, Chris Rackauckas"
title: "Turing.jl Model AD Backend Comparison"
---


This benchmark differentiates the log density of a large collection of
[Turing.jl](https://turinglang.org/) probabilistic models with every AD backend
that Turing supports, and reports both whether each backend gets the right
answer and how much it costs relative to a single evaluation of the log density.

The models and the measurement methodology are taken from
[TuringLang/ADTests](https://github.com/TuringLang/ADTests), whose live results
are published at <https://turinglang.org/ADTests/>. The models are deliberately
diverse rather than uniformly large: they cover base Julia control flow and
threading, the corners of the `@model` DSL, individual distributions (including
constrained and matrix-variate ones), the models from the DynamicPPL arXiv
paper, a slice of [PosteriorDB](https://github.com/stan-dev/posteriordb), and
models that call out to ODE/DDE solvers, Lux neural networks, and Gaussian
processes. Between them they exercise most of the Julia language features that
an AD backend has to cope with, which is what makes this a useful stress test of
the AD ecosystem rather than only of Turing.

## What is measured

For each (model, backend) pair we call `DynamicPPL.TestUtils.AD.run_ad`, which

1. builds the `LogDensityFunction` for the model with all variables linked to
   unconstrained space,
2. checks the gradient against a reference backend (`FiniteDifferences`, except
   on `dppl_hmm_semisup` where finite differences itself fails and `ForwardDiff`
   is used instead), and
3. benchmarks the primal evaluation and the gradient evaluation with
   [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl).

The headline number is the **relative gradient time**, `gradient time / primal
time`. That ratio is the quantity of interest for a sampler: it is how much more
expensive one leapfrog step of NUTS is than one likelihood evaluation. Reporting
a ratio rather than an absolute time also makes the numbers comparable across
models of very different sizes. Absolute gradient times are reported separately
below.

A cell that is not a number is a failure, and failures are as much the point of
this benchmark as the timings:

| Status  | Meaning                                                                 |
|:--------|:------------------------------------------------------------------------|
| `wrong` | The backend returned a gradient that disagrees with the reference        |
| `NaN`   | The backend returned a gradient containing `NaN`                         |
| `error` | The backend threw an exception                                           |
| `crash` | The backend took the whole Julia process down (usually a segfault)       |

Because of that last row, each model is benchmarked in a separate worker
process; if the worker dies, the results already collected are kept and the
worker is restarted with the backends that have not run yet.

```julia
using DataFrames, Markdown, Plots, PrettyTables, Printf, Statistics

include("turing_ad_models.jl")

const WORKER_SCRIPT = joinpath(@__DIR__, "turing_ad_worker.jl")

struct Measurement
    status::String
    relative::Float64
    gradient::Float64
    primal::Float64
end

Measurement(status) = Measurement(status, NaN, NaN, NaN)

function worker_command(model_name, backends)
    # The threaded models are written to be run with 4 threads; everything else
    # is single threaded so that the timings are not at the mercy of how many
    # cores the benchmark machine happens to have.
    nthreads = startswith(model_name, "threaded_") ? 4 : 1
    return `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$nthreads
            $WORKER_SCRIPT $model_name $backends`
end
```

```
worker_command (generic function with 1 method)
```





Driving the worker: read its records as they arrive, and restart it if it dies
partway through.

```julia
function benchmark_model(model_name)
    remaining = copy(BACKEND_NAMES)
    results = Dict{String, Measurement}()
    messages = Dict{String, String}()
    dimension = -1

    while !isempty(remaining)
        inflight = nothing
        try
            open(worker_command(model_name, remaining), "r") do io
                for line in eachline(io)
                    # Anything that is not a well-formed record is either the
                    # worker's own chatter or a line truncated by a crash.
                    fields = split(line, '\t')
                    if fields[1] == "META" && length(fields) == 3
                        dimension = parse(Int, fields[3])
                    elseif fields[1] == "BEGIN" && length(fields) == 2
                        inflight = fields[2]
                    elseif fields[1] == "MSG" && length(fields) == 3
                        messages[fields[2]] = fields[3]
                    elseif fields[1] == "RESULT" && length(fields) == 6
                        results[fields[2]] = Measurement(
                            fields[3], parse.(Float64, fields[4:6])...
                        )
                        inflight = nothing
                    end
                end
            end
        catch err
            err isa ProcessFailedException || rethrow()
        end

        if inflight !== nothing
            # Record the backend that killed the worker before restarting, so
            # that the restart does not walk straight back into the same crash.
            results[inflight] = Measurement("crash")
            messages[inflight] = "the worker process died while running this backend"
        end
        filter!(backend -> !haskey(results, backend), remaining)
        if inflight === nothing && !isempty(remaining)
            # The worker died without starting any of the backends it was given,
            # so it never got as far as building the model; restarting it would
            # only reproduce that.
            for backend in remaining
                results[backend] = Measurement("crash")
                messages[backend] = "the worker process died before building the model"
            end
            empty!(remaining)
        end
    end

    return (dimension = dimension, results = results, messages = messages)
end
```

```
benchmark_model (generic function with 1 method)
```





Now run every model. This is the expensive part: each model pays for a fresh
Julia process, and `FiniteDifferences` alone needs `4 * dimension + 1` log
density evaluations for every reference gradient.

```julia
runs = Dict{String, Any}()
for (i, model_name) in enumerate(MODEL_NAMES)
    @info "[$i/$(length(MODEL_NAMES))] benchmarking $model_name"
    # The workers write to the inherited file descriptor rather than through
    # Julia, so without this the progress log lags hours behind their output.
    flush(stderr)
    runs[model_name] = benchmark_model(model_name)
end
```


```julia
measurement(model_name, backend) = runs[model_name].results[backend]

df = DataFrame(
    [
        (
            category = CATEGORY_OF[model_name],
            model = model_name,
            dimension = runs[model_name].dimension,
            backend = backend,
            status = measurement(model_name, backend).status,
            relative = measurement(model_name, backend).relative,
            gradient = measurement(model_name, backend).gradient,
            primal = measurement(model_name, backend).primal,
        )
        for model_name in MODEL_NAMES, backend in BACKEND_NAMES
    ][:]
)
first(df, 10)
```

```
10×8 DataFrame
 Row │ category             model                          dimension  backe
nd  ⋯
     │ String               String                         Int64      Strin
g   ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │ Base Julia features  control_flow                           2  Finit
eDi ⋯
   2 │ Base Julia features  threaded_assume                       50  Finit
eDi
   3 │ Base Julia features  threaded_observe                       1  Finit
eDi
   4 │ Core Turing syntax   assume_submodel                        2  Finit
eDi
   5 │ Core Turing syntax   broadcast_macro                        2  Finit
eDi ⋯
   6 │ Core Turing syntax   dot_assume                             5  Finit
eDi
   7 │ Core Turing syntax   dot_observe                            1  Finit
eDi
   8 │ Core Turing syntax   dynamic_constraint                     2  Finit
eDi
   9 │ Core Turing syntax   multiple_constraints_same_var          4  Finit
eDi ⋯
  10 │ Core Turing syntax   observe_index                          1  Finit
eDi
                                                               5 columns om
itted
```





## Relative gradient time by model

Each cell is `gradient time / primal time`; lower is better. Non-numeric cells
are failures, as described above.

```julia
function cell(m)
    m.status == "ok" || return m.status
    return m.relative < 100 ? @sprintf("%.1f", m.relative) : @sprintf("%.0f", m.relative)
end

function category_table(category, model_names)
    table = DataFrame(
        "Model" => model_names,
        "Dim" => [runs[name].dimension for name in model_names],
    )
    for backend in BACKEND_NAMES
        table[!, backend] = [cell(measurement(name, backend)) for name in model_names]
    end
    return Markdown.parse(
        "### $category\n\n" * PrettyTables.pretty_table(
            String, table; backend = :markdown, column_labels = names(table)
        )
    )
end

for (category, model_names) in MODEL_CATEGORIES
    display(category_table(category, model_names))
end
```


### Base Julia features

|        **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ----------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|     control_flow |       2 |                   434 |             5.1 |             853 |                    78.3 |            11.4 |             376 |          26.3 |           5.1 |
|  threaded_assume |      50 |                   443 |             4.9 |           error |                   error |           error |           error |         error |         error |
| threaded_observe |       1 |                  12.3 |             1.0 |           error |                   error |           error |           crash |         error |         error |

### Core Turing syntax

|                     **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |       2 |                   686 |             9.1 |            1378 |                     123 |            17.6 |             612 |          42.6 |           7.9 |
|               broadcast_macro |       2 |                  92.8 |             1.8 |             357 |                    30.3 |             6.4 |            72.9 |           5.0 |           1.6 |
|                    dot_assume |       5 |                   166 |             1.9 |             201 |                    21.0 |             3.5 |            94.1 |           3.5 |           1.1 |
|                   dot_observe |       1 |                   141 |             4.1 |             883 |                    82.8 |            16.0 |             185 |          14.9 |           3.2 |
|            dynamic_constraint |       2 |                  77.9 |             1.7 |             207 |                    20.3 |             5.0 |            54.1 |           4.5 |           2.8 |
| multiple_constraints_same_var |       4 |                  58.8 |             1.1 |            38.3 |                     3.9 |             6.9 |            28.4 |           3.4 |           5.2 |
|                 observe_index |       1 |                   151 |             4.0 |             987 |                    83.4 |            16.0 |             189 |          15.5 |           3.2 |
|               observe_literal |       1 |                   421 |             9.6 |            1548 |                     146 |            10.7 |             511 |          48.8 |           8.8 |
|          observe_multivariate |       3 |                   102 |             1.4 |             175 |                    17.3 |             4.3 |            64.3 |           4.4 |          20.0 |
|              observe_submodel |       1 |                   411 |            10.3 |            1473 |                     160 |             9.4 |             507 |          47.4 |           8.5 |

### Distributions

|           **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |       1 |                  36.9 |             2.0 |            98.7 |                     8.2 |             4.0 |            32.9 |           4.1 |           2.6 |
|    assume_dirichlet |       1 |                  26.4 |             1.2 |            35.9 |                     5.1 |             6.8 |            22.5 |           7.6 |          11.1 |
|     assume_lkjcholu |      10 |                   151 |             2.0 |            88.9 |                    12.1 |             6.6 |            60.5 |           3.8 |          26.1 |
|     assume_mvnormal |       2 |                  41.3 |             0.9 |            24.1 |                     2.3 |            15.1 |            50.7 |         error |           5.1 |
|       assume_normal |       1 |                   470 |             7.8 |             869 |                    96.9 |             9.9 |             517 |          52.9 |           8.7 |
|      assume_wishart |       3 |                  48.0 |             1.0 |            63.9 |                     6.5 |            24.1 |            44.4 |          22.6 |          29.3 |
|   observe_bernoulli |       1 |                  30.4 |             1.7 |            88.3 |                     8.5 |             4.7 |            28.1 |           3.1 |           3.2 |
| observe_categorical |       1 |                  21.2 |             1.1 |            32.7 |                     5.7 |             9.6 |            14.1 |           2.1 |           9.2 |
|   observe_von_mises |       1 |                  27.2 |             NaN |             NaN |                     7.8 |             3.6 |            21.9 |           3.9 |           3.2 |

### DynamicPPL arXiv paper

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |       2 |                  21.3 |             2.8 |            1465 |                     239 |             1.3 |             2.9 |           1.6 |           3.0 |
|        dppl_hier_poisson |      13 |                   179 |             6.2 |             120 |                    12.9 |             7.7 |            51.4 |           6.0 |           2.0 |
|      dppl_high_dim_gauss |   10000 |                276855 |           22362 |             987 |                     187 |             1.9 |           63264 |        134193 |           7.1 |
|         dppl_hmm_semisup |     115 |                   NaN |            23.6 |            83.3 |                    14.9 |             6.8 |            5150 |          63.3 |          10.0 |
|                 dppl_lda |     535 |                  8662 |             206 |             229 |                    35.1 |            10.8 |            1747 |           214 |           3.0 |
| dppl_logistic_regression |     100 |                  1610 |            59.5 |            92.5 |                    13.4 |            20.6 |            1589 |          28.8 |          11.7 |
|         dppl_naive_bayes |     400 |                  5975 |             427 |             375 |                    50.9 |             5.8 |             889 |          1597 |           1.7 |
|      dppl_sto_volatility |     503 |                  6034 |             102 |             286 |                    29.4 |             2.9 |             764 |          80.8 |           3.3 |

### DynamicPPL demo models

|                                  **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |       2 |                  81.9 |             2.7 |             306 |                    23.9 |             5.5 |            63.2 |           4.6 |           1.6 |
|            demo_assume_dot_observe_literal |       2 |                  93.5 |             3.2 |             360 |                    29.6 |             6.8 |            80.5 |           5.5 |           1.7 |
|                  demo_assume_index_observe |       4 |                  99.5 |             2.1 |             165 |                    17.6 |             5.3 |            67.7 |           9.5 |          43.2 |
|    demo_assume_matrix_observe_matrix_index |       4 |                  78.4 |             1.4 |             106 |                    10.1 |             7.5 |            48.7 |          10.2 |          21.7 |
|           demo_assume_multivariate_observe |       4 |                  87.9 |             1.6 |             127 |                    13.0 |             6.0 |            45.4 |          13.5 |          26.9 |
|   demo_assume_multivariate_observe_literal |       4 |                  86.3 |             1.5 |             122 |                    13.0 |             5.8 |            43.0 |          13.3 |          26.6 |
|                demo_assume_observe_literal |       2 |                  91.1 |             1.9 |             360 |                    32.2 |             4.8 |            72.0 |           5.3 |           1.6 |
| demo_assume_submodel_observe_index_literal |       4 |                  89.2 |             1.6 |             162 |                    15.2 |             5.5 |            47.9 |           9.5 |          11.6 |
|                    demo_dot_assume_observe |       4 |                  84.4 |             1.7 |             133 |                    12.8 |             7.6 |            56.9 |          12.3 |          22.9 |
|              demo_dot_assume_observe_index |       4 |                  86.4 |             2.2 |             151 |                    14.3 |             5.8 |            60.1 |           8.7 |          13.4 |
|      demo_dot_assume_observe_index_literal |       4 |                  94.5 |             1.6 |             155 |                    15.5 |             5.3 |            50.0 |           9.0 |          10.0 |
|       demo_dot_assume_observe_matrix_index |       4 |                  79.2 |             1.5 |             121 |                    11.2 |             8.4 |            50.1 |          11.0 |          21.1 |
|           demo_dot_assume_observe_submodel |       4 |                  83.6 |             1.5 |             154 |                    12.1 |             7.3 |            56.9 |          11.7 |          22.1 |

### Effect of model size

| **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |      10 |                   229 |             3.3 |             188 |                    22.4 |             2.6 |             126 |           2.4 |           1.0 |
|      n050 |      50 |                   723 |            16.6 |             180 |                    23.4 |             1.4 |             128 |           7.2 |           0.6 |
|      n100 |     100 |                  1364 |            21.1 |             177 |                    22.0 |             1.2 |             157 |          14.3 |           0.6 |
|      n500 |     500 |                  5845 |            95.9 |             178 |                    22.4 |             1.4 |             505 |          84.0 |           0.9 |

### PosteriorDB

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |       4 |                  56.4 |             3.7 |             612 |                    70.1 |            10.6 |            14.8 |           3.2 |           2.6 |
|             pdb_earnings |       3 |                  34.2 |             3.0 |             715 |                     139 |            22.3 |            11.8 |           4.4 |           7.4 |
|        pdb_earnings_male |       4 |                  46.9 |             2.3 |             368 |                    69.8 |            16.4 |            10.8 |           3.1 |           4.1 |
|    pdb_eightsch_centered |      10 |                   241 |             3.2 |             296 |                    31.8 |             5.8 |             150 |           5.8 |           2.7 |
| pdb_eightsch_noncentered |      10 |                   246 |             4.6 |             326 |                    33.6 |             5.9 |             155 |           5.6 |           2.8 |
|              pdb_garch11 |       4 |                  50.9 |             2.1 |             378 |                    39.6 |             4.4 |             8.6 |           1.4 |           1.9 |
|                pdb_kidiq |       3 |                  37.2 |             2.6 |             297 |                    52.1 |            12.1 |            13.1 |           3.5 |           5.1 |
|                pdb_radon |      90 |                  1028 |            28.9 |             601 |                    53.0 |             8.7 |             223 |          32.0 |           1.8 |
|                 pdb_rats |      65 |                   874 |            22.4 |             598 |                    53.1 |             6.8 |             214 |          25.8 |           2.2 |
|                pdb_sblrc |       6 |                   108 |             6.4 |             276 |                    49.3 |             8.0 |            68.9 |           6.9 |           4.0 |
|                pdb_sblri |       6 |                   106 |             6.8 |             275 |                    49.2 |             8.1 |            73.5 |           7.0 |           3.9 |

### External libraries

|      **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |       7 |                  89.3 |             1.6 |           error |                   error |            11.0 |            46.0 |         error |          15.1 |
|    delaydiffeq |       5 |                  61.0 |             1.2 |             4.4 |                     1.3 |           error |           error |         wrong |          11.1 |
|         lux_nn |      20 |                   222 |             3.0 |            65.3 |                   wrong |            14.0 |            43.9 |         error |          16.8 |
| ordinarydiffeq |       5 |                  59.5 |             6.8 |            26.2 |                     5.3 |           error |           error |         wrong |           6.2 |




## Absolute gradient time by model

The same runs, reported as the wall time of a single gradient evaluation in
milliseconds. This is what matters when comparing models rather than backends.

```julia
function ms_cell(m)
    m.status == "ok" || return m.status
    return @sprintf("%.3g", 1000 * m.gradient)
end

# All backends time the same primal evaluation, so any spread across the row is
# measurement noise; take the fastest as the estimate.
function primal_time(model_name)
    times = [
        measurement(model_name, backend).primal
        for backend in BACKEND_NAMES if isfinite(measurement(model_name, backend).primal)
    ]
    return isempty(times) ? NaN : minimum(times)
end

function absolute_table(category, model_names)
    table = DataFrame(
        "Model" => model_names,
        "Primal (ms)" => [
            @sprintf("%.3g", 1000 * primal_time(name)) for name in model_names
        ],
    )
    for backend in BACKEND_NAMES
        table[!, backend] = [ms_cell(measurement(name, backend)) for name in model_names]
    end
    return Markdown.parse(
        "### $category\n\n" * PrettyTables.pretty_table(
            String, table; backend = :markdown, column_labels = names(table)
        )
    )
end

for (category, model_names) in MODEL_CATEGORIES
    display(absolute_table(category, model_names))
end
```


### Base Julia features

|        **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ----------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|     control_flow |        1.75e-05 |               0.00758 |        9.57e-05 |          0.0157 |                 0.00157 |        0.000226 |         0.00726 |      0.000487 |      9.05e-05 |
|  threaded_assume |          0.0675 |                  29.9 |           0.336 |           error |                   error |           error |           error |         error |         error |
| threaded_observe |          0.0725 |                 0.892 |          0.0766 |           error |                   error |           error |           crash |         error |         error |

### Core Turing syntax

|                     **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |        1.08e-05 |               0.00743 |        0.000102 |           0.016 |                 0.00153 |        0.000215 |         0.00736 |      0.000495 |      8.82e-05 |
|               broadcast_macro |        0.000108 |                0.0101 |          0.0002 |          0.0394 |                 0.00348 |        0.000727 |         0.00831 |      0.000571 |      0.000188 |
|                    dot_assume |        0.000166 |                0.0276 |        0.000322 |          0.0338 |                 0.00356 |        0.000607 |          0.0156 |       0.00058 |      0.000183 |
|                   dot_observe |        2.78e-05 |               0.00427 |        0.000126 |          0.0274 |                 0.00254 |        0.000477 |         0.00545 |      0.000457 |      8.86e-05 |
|            dynamic_constraint |        0.000149 |                0.0116 |        0.000259 |          0.0312 |                  0.0032 |        0.000802 |         0.00818 |       0.00069 |       0.00041 |
| multiple_constraints_same_var |         0.00134 |                0.0789 |         0.00154 |          0.0516 |                  0.0052 |         0.00928 |          0.0382 |       0.00464 |       0.00701 |
|                 observe_index |        2.73e-05 |               0.00413 |        0.000124 |          0.0277 |                 0.00254 |        0.000475 |         0.00554 |      0.000472 |      8.77e-05 |
|               observe_literal |        9.22e-06 |               0.00388 |         9.2e-05 |          0.0149 |                 0.00147 |        0.000109 |         0.00512 |       0.00045 |      8.12e-05 |
|          observe_multivariate |         0.00018 |                0.0184 |        0.000264 |          0.0324 |                 0.00318 |        0.000792 |          0.0118 |      0.000797 |       0.00363 |
|              observe_submodel |        9.22e-06 |               0.00379 |        9.48e-05 |          0.0148 |                  0.0016 |        9.84e-05 |         0.00509 |      0.000437 |      8.18e-05 |

### Distributions

|           **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |         0.00016 |               0.00623 |        0.000337 |          0.0161 |                 0.00141 |        0.000676 |         0.00556 |      0.000668 |      0.000424 |
|    assume_dirichlet |         0.00028 |               0.00761 |        0.000351 |          0.0101 |                 0.00144 |         0.00201 |         0.00642 |       0.00212 |       0.00314 |
|     assume_lkjcholu |         0.00102 |                 0.155 |         0.00209 |          0.0907 |                  0.0125 |         0.00683 |          0.0624 |       0.00399 |         0.027 |
|     assume_mvnormal |         0.00048 |                0.0207 |        0.000414 |          0.0118 |                 0.00109 |          0.0073 |          0.0249 |         error |       0.00245 |
|       assume_normal |        8.42e-06 |               0.00395 |        7.25e-05 |         0.00801 |                0.000959 |        9.69e-05 |         0.00491 |      0.000445 |      7.64e-05 |
|      assume_wishart |         0.00104 |                  0.05 |         0.00102 |          0.0666 |                 0.00669 |          0.0252 |          0.0467 |        0.0242 |        0.0315 |
|   observe_bernoulli |        0.000221 |                0.0068 |        0.000384 |          0.0197 |                 0.00195 |         0.00105 |         0.00621 |      0.000704 |      0.000711 |
| observe_categorical |        0.000539 |                0.0114 |        0.000621 |          0.0178 |                  0.0031 |         0.00522 |         0.00772 |       0.00117 |         0.005 |
|   observe_von_mises |        0.000308 |               0.00838 |             NaN |             NaN |                 0.00245 |         0.00111 |         0.00679 |       0.00122 |      0.000995 |

### DynamicPPL arXiv paper

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |          0.0076 |                 0.164 |          0.0218 |            11.1 |                    1.82 |          0.0189 |          0.0318 |        0.0244 |        0.0419 |
|        dppl_hier_poisson |         0.00193 |                 0.346 |           0.012 |           0.234 |                  0.0254 |          0.0152 |             0.1 |        0.0118 |       0.00384 |
|      dppl_high_dim_gauss |         0.00593 |              1.64e+03 |             134 |            5.93 |                    1.11 |          0.0121 |             381 |      1.06e+03 |        0.0522 |
|         dppl_hmm_semisup |            0.26 |                   NaN |            6.14 |            21.6 |                    3.88 |             1.8 |        1.35e+03 |          16.6 |          2.65 |
|                 dppl_lda |          0.0924 |                   804 |              19 |            21.2 |                    3.32 |            1.17 |             171 |          20.2 |         0.294 |
| dppl_logistic_regression |           0.303 |                   487 |            18.2 |            28.6 |                    4.83 |            9.33 |             500 |          9.25 |          5.15 |
|         dppl_naive_bayes |           0.547 |              3.27e+03 |             234 |             205 |                    27.8 |            3.18 |             488 |           877 |         0.936 |
|      dppl_sto_volatility |          0.0267 |                   161 |            2.83 |             7.9 |                   0.787 |          0.0778 |              21 |           2.2 |        0.0902 |

### DynamicPPL demo models

|                                  **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |        0.000131 |                0.0107 |        0.000356 |          0.0404 |                 0.00342 |        0.000739 |         0.00843 |      0.000622 |      0.000209 |
|            demo_assume_dot_observe_literal |        0.000107 |               0.00997 |        0.000349 |          0.0388 |                 0.00342 |         0.00074 |         0.00873 |      0.000596 |       0.00019 |
|                  demo_assume_index_observe |        0.000307 |                0.0306 |        0.000642 |          0.0506 |                 0.00559 |         0.00168 |          0.0211 |       0.00295 |        0.0134 |
|    demo_assume_matrix_observe_matrix_index |          0.0005 |                0.0399 |        0.000723 |          0.0539 |                 0.00511 |         0.00373 |          0.0244 |       0.00517 |        0.0109 |
|           demo_assume_multivariate_observe |        0.000379 |                0.0339 |        0.000599 |           0.048 |                 0.00504 |         0.00236 |          0.0176 |       0.00522 |        0.0107 |
|   demo_assume_multivariate_observe_literal |         0.00039 |                0.0338 |        0.000596 |          0.0476 |                 0.00508 |         0.00225 |          0.0168 |       0.00521 |        0.0104 |
|                demo_assume_observe_literal |        0.000109 |                  0.01 |        0.000211 |          0.0393 |                 0.00355 |        0.000526 |         0.00795 |      0.000606 |      0.000175 |
| demo_assume_submodel_observe_index_literal |        0.000357 |                0.0323 |        0.000595 |          0.0579 |                 0.00545 |         0.00207 |          0.0177 |       0.00342 |       0.00417 |
|                    demo_dot_assume_observe |        0.000409 |                0.0352 |        0.000684 |          0.0552 |                 0.00532 |         0.00321 |          0.0239 |       0.00519 |       0.00954 |
|              demo_dot_assume_observe_index |        0.000382 |                0.0338 |        0.000824 |          0.0576 |                 0.00554 |         0.00225 |          0.0231 |       0.00334 |       0.00524 |
|      demo_dot_assume_observe_index_literal |        0.000351 |                0.0335 |        0.000558 |          0.0546 |                  0.0056 |         0.00191 |          0.0177 |       0.00328 |        0.0035 |
|       demo_dot_assume_observe_matrix_index |        0.000486 |                 0.039 |        0.000718 |          0.0592 |                 0.00549 |         0.00413 |          0.0269 |       0.00541 |        0.0103 |
|           demo_dot_assume_observe_submodel |        0.000423 |                 0.036 |        0.000646 |          0.0653 |                 0.00527 |         0.00316 |          0.0246 |       0.00514 |       0.00961 |

### Effect of model size

| **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |        0.000298 |                0.0691 |        0.000973 |          0.0562 |                 0.00676 |        0.000787 |          0.0379 |      0.000739 |      0.000284 |
|      n050 |         0.00138 |                 0.999 |          0.0233 |           0.248 |                  0.0324 |         0.00192 |           0.178 |        0.0101 |      0.000919 |
|      n100 |         0.00278 |                  3.79 |          0.0596 |           0.495 |                  0.0617 |          0.0035 |            0.44 |        0.0408 |       0.00174 |
|      n500 |          0.0138 |                  83.2 |            1.32 |            2.53 |                   0.309 |          0.0202 |            7.12 |          1.22 |        0.0122 |

### PosteriorDB

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |         0.00172 |                0.0974 |         0.00647 |            1.05 |                   0.124 |          0.0187 |          0.0258 |        0.0057 |       0.00452 |
|             pdb_earnings |         0.00188 |                0.0766 |         0.00644 |            1.43 |                   0.267 |          0.0478 |          0.0296 |       0.00839 |        0.0139 |
|        pdb_earnings_male |         0.00386 |                 0.189 |          0.0099 |            1.42 |                    0.27 |          0.0711 |          0.0459 |        0.0133 |        0.0211 |
|    pdb_eightsch_centered |        0.000281 |                0.0681 |        0.000931 |          0.0846 |                 0.00925 |         0.00162 |          0.0425 |       0.00165 |      0.000773 |
| pdb_eightsch_noncentered |        0.000274 |                0.0674 |         0.00127 |          0.0902 |                 0.00927 |         0.00165 |          0.0428 |       0.00157 |       0.00078 |
|              pdb_garch11 |         0.00565 |                 0.288 |          0.0118 |            2.14 |                   0.225 |          0.0251 |           0.049 |       0.00788 |         0.011 |
|                pdb_kidiq |         0.00188 |                0.0756 |         0.00516 |            0.56 |                  0.0987 |          0.0232 |          0.0269 |        0.0065 |        0.0099 |
|                pdb_radon |          0.0151 |                  16.2 |           0.439 |            9.12 |                   0.849 |           0.138 |            3.37 |         0.486 |        0.0292 |
|                 pdb_rats |         0.00217 |                  1.89 |          0.0487 |             1.3 |                   0.115 |          0.0148 |           0.466 |        0.0572 |       0.00467 |
|                pdb_sblrc |        0.000568 |                0.0621 |         0.00375 |           0.157 |                  0.0287 |         0.00464 |          0.0391 |         0.004 |        0.0023 |
|                pdb_sblri |        0.000565 |                0.0611 |         0.00383 |           0.159 |                  0.0279 |          0.0047 |          0.0417 |       0.00406 |       0.00226 |

### External libraries

|      **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |         0.00291 |                  0.26 |         0.00466 |           error |                   error |          0.0321 |           0.137 |         error |        0.0453 |
|    delaydiffeq |           0.634 |                  38.7 |           0.742 |            2.83 |                     0.8 |           error |           error |         wrong |          7.38 |
|         lux_nn |          0.0542 |                  12.1 |           0.165 |            3.55 |                   wrong |           0.804 |            2.57 |         error |          1.03 |
| ordinarydiffeq |          0.0979 |                  5.83 |           0.672 |            2.58 |                   0.533 |           error |           error |         wrong |         0.615 |




## Summary

How often does each backend work at all, and how fast is it when it does?

```julia
statuses = ["ok", "wrong", "NaN", "error", "crash"]

summary = DataFrame(
    "Backend" => BACKEND_NAMES,
    [
        status => [count(
            name -> measurement(name, backend).status == status, MODEL_NAMES
        ) for backend in BACKEND_NAMES]
        for status in statuses
    ]...,
)
Markdown.parse(
    PrettyTables.pretty_table(
        String, summary; backend = :markdown, column_labels = names(summary)
    )
)
```


|         **Backend** | **ok** | **wrong** | **NaN** | **error** | **crash** |
| -------------------:| ------:| ---------:| -------:| ---------:| ---------:|
|   FiniteDifferences |     61 |         0 |       1 |         0 |         0 |
|         ForwardDiff |     61 |         0 |       1 |         0 |         0 |
|         ReverseDiff |     58 |         0 |       1 |         3 |         0 |
| ReverseDiffCompiled |     58 |         1 |       0 |         3 |         0 |
|         MooncakeRvs |     58 |         0 |       0 |         4 |         0 |
|         MooncakeFwd |     58 |         0 |       0 |         3 |         1 |
|           EnzymeFwd |     55 |         2 |       0 |         5 |         0 |
|           EnzymeRvs |     60 |         0 |       0 |         2 |         0 |




Aggregating the timings needs some care, because every backend fails on a
different subset of the models. The first three columns below summarise each
backend over the models *it* handles, so they are not directly comparable to one
another: a backend that only works on the easy models will look good. The last
two columns are the comparable ones — they compare each backend against
ForwardDiff over exactly the models that both of them get right.

```julia
geomean(xs) = isempty(xs) ? NaN : exp(mean(log, xs))
safe_median(xs) = isempty(xs) ? NaN : median(xs)
succeeded(backend) = [
    name for name in MODEL_NAMES if measurement(name, backend).status == "ok"
]
relative_times(backend) = [
    measurement(name, backend).relative for name in succeeded(backend)
]

function ratio_to_forwarddiff(backend)
    shared = intersect(succeeded(backend), succeeded("ForwardDiff"))
    ratios = [
        measurement(name, backend).relative / measurement(name, "ForwardDiff").relative
        for name in shared
    ]
    return length(shared), geomean(ratios)
end

comparison = DataFrame(
    "Backend" => BACKEND_NAMES,
    "Models OK" => [length(succeeded(b)) for b in BACKEND_NAMES],
    "Geometric mean" => [geomean(relative_times(b)) for b in BACKEND_NAMES],
    "Median" => [safe_median(relative_times(b)) for b in BACKEND_NAMES],
    "Shared with FD" => [first(ratio_to_forwarddiff(b)) for b in BACKEND_NAMES],
    "vs ForwardDiff" => [last(ratio_to_forwarddiff(b)) for b in BACKEND_NAMES],
)
Markdown.parse(
    PrettyTables.pretty_table(
        String, comparison;
        backend = :markdown, column_labels = names(comparison),
        formatters = [PrettyTables.fmt__printf("%.2f", [3, 4, 6])],
    )
)
```


|         **Backend** | **Models OK** | **Geometric mean** | **Median** | **Shared with FD** | **vs ForwardDiff** |
| -------------------:| -------------:| ------------------:| ----------:| ------------------:| ------------------:|
|   FiniteDifferences |            61 |             169.41 |      93.48 |                 60 |              36.95 |
|         ForwardDiff |            61 |               4.85 |       2.83 |                 61 |               1.00 |
|         ReverseDiff |            58 |             208.88 |     194.64 |                 58 |              41.11 |
| ReverseDiffCompiled |            58 |              24.13 |      22.44 |                 57 |               4.80 |
|         MooncakeRvs |            58 |               6.63 |       6.82 |                 57 |               1.32 |
|         MooncakeFwd |            58 |              96.10 |      63.75 |                 57 |              19.41 |
|           EnzymeFwd |            55 |              12.56 |       8.73 |                 54 |               2.37 |
|           EnzymeRvs |            60 |               5.22 |       5.06 |                 59 |               1.06 |




The scaling behaviour is the clearest signal in the whole benchmark: forward
mode costs a pass per parameter, so it climbs with the dimension of the model,
while reverse mode is bounded by a constant multiple of the primal.

```julia
scaling = plot(
    xscale = :log10, yscale = :log10,
    xlabel = "model dimension", ylabel = "gradient time / primal time",
    title = "Cost of a gradient against the number of parameters",
    legend = :outertopright, size = (900, 500),
)
for backend in BACKEND_NAMES
    names = succeeded(backend)
    isempty(names) && continue
    scatter!(
        scaling,
        [max(runs[name].dimension, 1) for name in names],
        [measurement(name, backend).relative for name in names];
        label = backend, markersize = 4, markerstrokewidth = 0,
    )
end
scaling
```

![](figures/TuringADTests_9_1.png)

```julia
relatives = [
    measurement(name, backend).relative
    for backend in BACKEND_NAMES, name in MODEL_NAMES
]
heatmap(
    log10.(relatives);
    xticks = (1:length(MODEL_NAMES), MODEL_NAMES), xrotation = 90,
    yticks = (1:length(BACKEND_NAMES), BACKEND_NAMES),
    colorbar_title = "log10(gradient time / primal time)",
    title = "Relative gradient time (blank = failure)",
    size = (1600, 620), bottom_margin = 60Plots.mm, left_margin = 12Plots.mm,
)
```

![](figures/TuringADTests_10_1.png)



## Failures

Every non-`ok` result, with the error it produced. Backends that a model's
maintainers already know to be unsupported show up here too, so this table is
long by design.

```julia
# `|` would be read as a column separator by the markdown table, and the longer
# Enzyme messages run to several hundred characters.
function readable(message)
    trimmed = length(message) > 120 ? first(message, 117) * "..." : message
    return replace(trimmed, '|' => '/')
end

failures = filter(:status => !=("ok"), df)[:, [:model, :backend, :status]]
failures.message = [
    readable(get(runs[row.model].messages, row.backend, "")) for row in eachrow(failures)
]
Markdown.parse(
    PrettyTables.pretty_table(
        String, failures; backend = :markdown, column_labels = names(failures)
    )
)
```


|         **model** |         **backend** | **status** |                                                                                                              **message** |
| -----------------:| -------------------:| ----------:| ------------------------------------------------------------------------------------------------------------------------:|
|  dppl_hmm_semisup |   FiniteDifferences |        NaN |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
| observe_von_mises |         ForwardDiff |        NaN |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
|   threaded_assume |         ReverseDiff |      error |                                                                                                      TaskFailedException |
|  threaded_observe |         ReverseDiff |      error |                                                                                                      TaskFailedException |
| observe_von_mises |         ReverseDiff |        NaN |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
|       abstractgps |         ReverseDiff |      error | MethodError: -(::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}, ::FillArrays.Zeros{... |
|   threaded_assume | ReverseDiffCompiled |      error |                                                                                                      TaskFailedException |
|  threaded_observe | ReverseDiffCompiled |      error |                                                                                                      TaskFailedException |
|       abstractgps | ReverseDiffCompiled |      error | MethodError: -(::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}, ::FillArrays.Zeros{... |
|            lux_nn | ReverseDiffCompiled |      wrong |                                           ADIncorrectException: The AD backend returned an incorrect value and gradient. |
|   threaded_assume |         MooncakeRvs |      error |                                                                   Mooncake failed to differentiate the following method: |
|  threaded_observe |         MooncakeRvs |      error | TypeError: in typeassert, expected Mooncake.CoDual{IdDict{Any, Any}, IdDict{Any, Any}}, got a value of type Mooncake.... |
|       delaydiffeq |         MooncakeRvs |      error |                                                 MethodError: no method matching +(::Vector{Float64}, ::Mooncake.NoRData) |
|    ordinarydiffeq |         MooncakeRvs |      error |                                                 MethodError: no method matching +(::Vector{Float64}, ::Mooncake.NoRData) |
|   threaded_assume |         MooncakeFwd |      error |                                                                                                      TaskFailedException |
|  threaded_observe |         MooncakeFwd |      crash |                                                                       the worker process died while running this backend |
|       delaydiffeq |         MooncakeFwd |      error | Mooncake.IntrinsicsWrappers.MissingIntrinsicWrapperException("Unable to translate the intrinsic Val{Core.Intrinsics.l... |
|    ordinarydiffeq |         MooncakeFwd |      error |                                                                          MethodError: Cannot `convert` an object of type |
|   threaded_assume |           EnzymeFwd |      error |                                                                                                      TaskFailedException |
|  threaded_observe |           EnzymeFwd |      error |                                                                         EnzymeRuntimeException: Enzyme execution failed. |
|   assume_mvnormal |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       abstractgps |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       delaydiffeq |           EnzymeFwd |      wrong |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
|            lux_nn |           EnzymeFwd |      error |                                                                         EnzymeRuntimeException: Enzyme execution failed. |
|    ordinarydiffeq |           EnzymeFwd |      wrong |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
|   threaded_assume |           EnzymeRvs |      error |                                                                                                 EnzymeNoDerivativeError: |
|  threaded_observe |           EnzymeRvs |      error |                                                                                                 EnzymeNoDerivativeError: |



## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:
```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AutomaticDifferentiationTuring","TuringADTests.jmd")
```

Computer Information:

```
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver2)
  GC: Built with stock GC
Threads: 128 default, 1 interactive, 128 GC (on 128 virtual cores)
Environment:
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Project.toml`
  [47edcb42] ADTypes v1.24.0
  [99985d1d] AbstractGPs v0.5.24
  [0ca39b1e] Chairmarks v1.3.1
  [a93c6f00] DataFrames v1.8.2
  [bcd4f6db] DelayDiffEq v6.4.0
  [8bb1440f] DelimitedFiles v1.9.1
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [31c24e10] Distributions v0.25.131
  [366bfd00] DynamicPPL v0.42.13
⌃ [7da242da] Enzyme v0.13.203
  [1a297f60] FillArrays v1.17.0
  [26cc04aa] FiniteDifferences v0.12.34
  [f6369f11] ForwardDiff v1.4.6
  [d9f16b24] Functors v0.5.3
  [6fdf6af0] LogDensityProblems v2.2.0
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [b2108857] Lux v1.31.4
⌃ [da2b9cff] Mooncake v0.5.56
  [1dea7af3] OrdinaryDiffEq v7.8.1
  [b1df2697] OrdinaryDiffEqTsit5 v2.1.4
  [91a5bcdd] Plots v1.41.7
  [1c4bc282] PosteriorDB v0.7.0
  [08abe8d2] PrettyTables v3.4.8
  [37e2e3b7] ReverseDiff v1.17.0
  [31c91b34] SciMLBenchmarks v0.2.1 [loaded: `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/src/SciMLBenchmarks.jl` (v0.2.1) expected `/home/crackauc/.julia/packages/SciMLBenchmarks/ceJyd/src/SciMLBenchmarks.jl` (v0.2.1)]
⌃ [1ed8b502] SciMLSensitivity v7.119.7
  [10745b16] Statistics v1.11.5
⌅ [4c63d2b9] StatsFuns v1.5.3
⌅ [fce5fe82] Turing v0.48.0
  [37e2e46d] LinearAlgebra v1.12.0
  [d6f4376e] Markdown v1.11.0
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
  [14f7f29c] AMD v0.5.4
  [621f4979] AbstractFFTs v1.5.0
  [99985d1d] AbstractGPs v0.5.24
  [80f14c24] AbstractMCMC v5.16.0
  [7a57a42e] AbstractPPL v0.15.5
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [0bf59076] AdvancedHMC v0.8.7
  [5b7e9947] AdvancedMH v0.8.10
  [b5ca4192] AdvancedVI v0.7.0
  [66dad0bd] AliasTables v1.1.3
  [dce04be8] ArgCheck v2.5.0
⌃ [4fba245c] ArrayInterface v7.30.1
  [a9b6321e] Atomix v1.2.1
  [ab4f0b2a] BFloat16s v0.6.1
  [198e06fe] BangBang v0.4.9
  [76274a88] Bijectors v0.16.3
  [b2a6c25c] BinaryHeaps v1.1.0
  [70df07ce] BracketingNonlinearSolve v1.12.7
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.7
  [082447d4] ChainRules v1.73.0
  [d360d2e6] ChainRulesCore v1.26.1
  [0ca39b1e] Chairmarks v1.3.1
  [9e997f8a] ChangesOfVariables v0.1.11
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
  [38540f10] CommonSolve v0.2.14
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.2.2
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.8
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.8.2
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [bcd4f6db] DelayDiffEq v6.4.0
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
  [2b5f629d] DiffEqBase v7.21.1
  [459566f4] DiffEqCallbacks v4.19.4
  [77a26b50] DiffEqNoiseProcess v5.36.3
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [0703355e] DimensionalData v0.30.2
  [8d63f2c5] DispatchDoctor v0.4.28
  [b4f34e82] Distances v0.10.12
  [31c24e10] Distributions v0.25.131
  [ffbed154] DocStringExtensions v0.9.5
  [366bfd00] DynamicPPL v0.42.13
  [cad2338a] EllipticalSliceSampling v2.0.0
  [4e289a0a] EnumX v1.0.7
⌃ [7da242da] Enzyme v0.13.203
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [21656369] ExpressionExplorer v1.1.5
  [411431e0] Extents v0.1.6
  [c87230d0] FFMPEG v0.4.5
  [7034ab61] FastBroadcast v1.4.0
  [9aa1b823] FastClosures v0.3.2
  [a4df4552] FastPower v1.5.0
  [1a297f60] FillArrays v1.17.0
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
  [26cc04aa] FiniteDifferences v0.12.34
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [4a37a8b9] FlexiChains v0.6.39
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.4.6
  [f62d2435] FunctionProperties v1.2.0
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [d9f16b24] Functors v0.5.3
  [46192b85] GPUArraysCore v0.2.0
⌅ [61eb1bfa] GPUCompiler v1.23.0
  [28b8d3ca] GR v0.73.27
⌃ [a0844989] Gamma v1.1.0
  [076d061b] HashArrayMappedTries v0.2.0
⌅ [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.30
  [7869d1d1] IRTools v0.4.20
  [615f187c] IfElse v0.1.1
  [22cec73e] InitialValues v0.3.1
⌅ [842dd82b] InlineStrings v1.4.6
  [85a1e053] Interfaces v0.3.2
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.6
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [63c18a36] KernelAbstractions v0.9.42
⌅ [ec8451be] KernelFunctions v0.10.67
  [ba0b0d4f] Krylov v0.10.10
  [2faa5264] LHLFactorization v2.2.2
  [929cbde3] LLVM v9.13.1
  [b964fa9f] LaTeXStrings v1.4.1
  [23fbe1c1] Latexify v0.16.12
  [1d6d02ad] LeftChildRightSiblingTrees v0.3.0
  [6f1fad26] Libtask v0.9.19
  [87fe0de2] LineSearch v0.1.18
  [d3d80556] LineSearches v7.8.1
⌃ [7ed4a6bd] LinearSolve v5.17.3
  [6fdf6af0] LogDensityProblems v2.2.0
  [996a588d] LogDensityProblemsAD v1.13.1
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.2.0
  [b2108857] Lux v1.31.4
  [bb33d45b] LuxCore v1.5.3
  [82251201] LuxLib v1.15.9
  [be115224] MCMCDiagnosticTools v0.3.19
  [7e8f7934] MLDataDevices v1.17.10
  [e80e1ace] MLJModelInterface v1.12.1
  [1914dd2f] MacroTools v0.5.16
  [dbb5928d] MappedArrays v0.4.3
  [bb5d69b7] MaybeInplace v0.1.8
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
⌃ [da2b9cff] Mooncake v0.5.56
  [46d2c3a1] MuladdMacro v0.2.7
  [ffc61752] Mustache v1.0.21
  [d41bc354] NLSolversBase v8.0.1
  [872c559c] NNlib v0.9.45
  [77ba4419] NaNMath v1.1.4
⌃ [8913a72c] NonlinearSolve v4.30.0
⌃ [be0214bd] NonlinearSolveBase v2.49.5
⌃ [5959db7a] NonlinearSolveFirstOrder v2.6.1
  [9a2c21bd] NonlinearSolveQuasiNewton v1.15.3
  [26075421] NonlinearSolveSpectralMethods v1.8.3
  [d8793406] ObjectFile v0.5.1
  [429524aa] Optim v2.3.1
  [3bd65402] Optimisers v0.4.9
  [7f7a1694] Optimization v5.9.1
  [bca83a33] OptimizationBase v5.6.1
  [36348300] OptimizationOptimJL v0.4.21
⌅ [bac558e1] OrderedCollections v1.8.2 [loaded: v2.0.1]
  [1dea7af3] OrdinaryDiffEq v7.8.1
⌃ [6ad6398a] OrdinaryDiffEqBDF v2.4.9
⌃ [bbf590c4] OrdinaryDiffEqCore v4.17.2
  [50262376] OrdinaryDiffEqDefault v2.6.2
⌃ [4302a76b] OrdinaryDiffEqDifferentiation v3.12.0
  [d3585ca7] OrdinaryDiffEqFunctionMap v2.3.0
  [127b3ac7] OrdinaryDiffEqNonlinearSolve v2.9.8
⌃ [43230ef6] OrdinaryDiffEqRosenbrock v2.7.3
  [b4bd8bb3] OrdinaryDiffEqRosenbrockTableaus v2.4.2
⌃ [2d112036] OrdinaryDiffEqSDIRK v2.9.4
  [b1df2697] OrdinaryDiffEqTsit5 v2.1.4
  [79d7bb75] OrdinaryDiffEqVerner v2.4.1
  [90014a1f] PDMats v0.11.41
⌅ [69de0a69] Parsers v2.8.8
  [569bd051] PartitionedDistributions v0.1.0
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.7
  [e409e4f3] PoissonRandom v0.4.13
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
  [1c4bc282] PosteriorDB v0.7.0
  [d236fae5] PreallocationTools v1.7.1
  [aea7be01] PrecompileTools v1.3.4
  [21216c6a] Preferences v1.6.0
  [08abe8d2] PrettyTables v3.4.8
  [33c8b6b6] ProgressLogging v0.1.6
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.5.0
  [1fd47b50] QuadGK v2.11.3
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [a3311ec8] ReactantCore v0.1.22
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [731186ca] RecursiveArrayTools v4.5.1
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.4.0
  [9fe22ead] RespecializeParams v1.3.0
  [37e2e3b7] ReverseDiff v1.17.0
  [708f8203] Richardson v1.4.3
  [79098fc4] Rmath v0.9.0
  [f2b01f46] Roots v3.0.8
  [7e49a35a] RuntimeGeneratedFunctions v0.5.26
  [0bca4576] SciMLBase v3.54.0
  [31c91b34] SciMLBenchmarks v0.2.1 [loaded: `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/src/SciMLBenchmarks.jl` (v0.2.1) expected `/home/crackauc/.julia/packages/SciMLBenchmarks/ceJyd/src/SciMLBenchmarks.jl` (v0.2.1)]
  [19f34311] SciMLJacobianOperators v0.1.19
  [a6db7da4] SciMLLogging v2.1.0
⌃ [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
⌃ [1ed8b502] SciMLSensitivity v7.119.7
  [53ae85a6] SciMLStructures v1.10.5
  [30f210dd] ScientificTypesBase v3.1.0
  [7e506255] ScopedValues v1.6.2
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.10
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.1.1
  [727e6d20] SimpleNonlinearSolve v2.14.5
  [a2af1166] SortingAlgorithms v1.2.3
  [a57abbd0] SparseColumnPivotedQR v2.1.8
  [9f842d2f] SparseConnectivityTracer v1.2.3
  [dc90abb0] SparseInverseSubset v0.1.3
  [0a514795] SparseMatrixColorings v0.4.28
  [276daf66] SpecialFunctions v2.9.0
  [860ef19b] StableRNGs v1.0.4
  [aedffcd0] Static v1.4.6
  [90137ffa] StaticArrays v1.9.20
  [1e83bf80] StaticArraysCore v1.4.4
  [64bff920] StatisticalTraits v3.5.0
  [10745b16] Statistics v1.11.5
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.13
⌅ [4c63d2b9] StatsFuns v1.5.3
  [69024149] StringEncodings v0.3.7
⌅ [892a3eda] StringManipulation v0.5.0
  [09ab397b] StructArrays v0.7.3
  [53d494c1] StructIO v0.3.1
  [856f2bd8] StructTypes v1.11.0
  [2efcf032] SymbolicIndexingInterface v0.3.55
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.14.0
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.8
  [a759f4b9] TimerOutputs v1.2.1
  [9f7883ad] Tracker v0.2.39
  [e689c965] Tracy v0.1.6
  [781d530d] TruncatedStacktraces v1.4.0
⌅ [fce5fe82] Turing v0.48.0
  [1cfade01] UnicodeFun v0.4.1
  [013be700] UnsafeAtomics v0.3.2
  [41fe7b60] Unzip v0.2.0
  [44d3d7a6] Weave v0.10.12
  [d49dbf32] WeightInitializers v1.3.4
  [ddb6d928] YAML v0.4.16
  [a5390f91] ZipFile v0.10.1
  [e88e6eb3] Zygote v0.7.13
  [700de1a5] ZygoteRules v0.2.8
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [7cc45869] Enzyme_jll v0.0.293+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.8.4+0
⌅ [b22a6f82] FFMPEG_jll v8.1.2+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.5.1+0
  [d2c73de3] GR_jll v0.73.27+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [7746bdde] Glib_jll v2.88.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
  [2e76f6c2] HarfBuzz_jll v100.14004.0+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.2.0+0
  [dad2f222] LLVMExtra_jll v0.0.47+0
  [1d63c593] LLVMOpenMP_jll v23.1.1+0
  [ad6e5548] LibTracyClient_jll v0.13.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [856f044c] MKL_jll v2025.2.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.58.2+0
  [30392449] Pixman_jll v0.46.4+0
  [c0090381] Qt6Base_jll v6.10.2+2
  [629bc702] Qt6Declarative_jll v6.10.2+2
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
  [f50d1b31] Rmath_jll v0.5.2+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
  [a2964d1f] Wayland_jll v1.24.0+0
  [ffd25f8a] XZ_jll v5.8.4+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.13+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.8+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.2+0
  [a51aa0fd] Xorg_libXi_jll v1.8.4+0
  [d1454406] Xorg_libXinerama_jll v1.1.7+0
  [ec84b674] Xorg_libXrandr_jll v1.5.6+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [a65dc6b1] Xorg_libpciaccess_jll v0.19.0+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.2.0+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.6+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
→ [33bec58e] Xorg_xkeyboard_config_jll v2.47.0+2
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
⌅ [214eeab7] fzf_jll v0.61.1+0
  [a4ae2306] libaom_jll v3.14.1+0
  [0ac62f75] libass_jll v0.17.5+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [8e53e030] libdrm_jll v2.4.134+0
  [2db6ffa8] libevdev_jll v1.13.4+0
  [f638f0a6] libfdk_aac_jll v2.0.4+0
  [36db933b] libinput_jll v1.28.1+0
  [b53b4c65] libpng_jll v1.6.58+0
  [9a156e7d] libva_jll v2.23.0+0
  [f27f6e37] libvorbis_jll v1.3.8+0
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.3.0+0
⌅ [1270edf5] x264_jll v10164.0.1+0
  [dfaa095f] x265_jll v4.1.0+0
  [d8fb68d0] xkbcommon_jll v1.13.0+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [8ba89e20] Distributed v1.11.0
  [f43a241f] Downloads v1.7.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [ac6e5ff7] JuliaSyntaxHighlighting v1.12.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [37e2e46d] LinearAlgebra v1.12.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.3.0
  [44cfe95a] Pkg v1.12.1
  [de0858da] Printf v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.12.0
  [f489334b] StyledStrings v1.11.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [e66e0078] CompilerSupportLibraries_jll v1.3.1+2
  [deac9b47] LibCURL_jll v8.15.0+0
  [e37daf67] LibGit2_jll v1.9.0+0
  [29816b5a] LibSSH2_jll v1.11.3+1
  [14a3606d] MozillaCACerts_jll v2025.11.4
  [4536629a] OpenBLAS_jll v0.3.29+0
  [05823500] OpenLibm_jll v0.8.7+0
  [458c3c95] OpenSSL_jll v3.5.6+0
  [efcefdf7] PCRE2_jll v10.44.0+1
  [bea87d4a] SuiteSparse_jll v7.8.3+2
  [83775a58] Zlib_jll v1.3.1+2
  [8e850b90] libblastrampoline_jll v5.15.0+0
  [8e850ede] nghttp2_jll v1.64.0+1
  [3f19e933] p7zip_jll v17.7.0+0
Info Packages marked with → are not downloaded, use `instantiate` to download
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

