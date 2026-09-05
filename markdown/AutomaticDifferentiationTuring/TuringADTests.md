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
|     control_flow |       2 |                   426 |             5.8 |             776 |                    81.5 |            10.9 |            47.5 |          22.8 |           3.8 |
|  threaded_assume |      50 |                   417 |             4.3 |           error |                   error |           error |             123 |          28.4 |           8.9 |
| threaded_observe |       1 |                  11.8 |             1.0 |           error |                   error |           error |             5.3 |           3.8 |           9.9 |

### Core Turing syntax

|                     **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |       2 |                   688 |             9.6 |            1335 |                     130 |            18.2 |            75.0 |          35.7 |           5.8 |
|               broadcast_macro |       2 |                  93.6 |             1.8 |             330 |                    29.9 |             6.3 |            12.8 |           4.9 |           1.5 |
|                    dot_assume |       5 |                   170 |             1.7 |             190 |                    21.2 |             3.1 |            17.7 |           3.4 |           1.0 |
|                   dot_observe |       1 |                   143 |             5.2 |             990 |                    91.8 |            15.6 |            19.6 |          15.6 |           2.3 |
|            dynamic_constraint |       2 |                  77.2 |             1.7 |             198 |                    20.7 |             5.6 |            11.5 |           4.1 |           2.5 |
| multiple_constraints_same_var |       4 |                  59.1 |             1.2 |            35.6 |                     3.8 |             6.5 |            16.9 |           3.5 |           5.5 |
|                 observe_index |       1 |                   158 |             5.1 |            1001 |                    81.6 |            15.4 |            20.1 |          16.0 |           2.5 |
|               observe_literal |       1 |                   537 |            12.6 |            1786 |                     147 |            12.6 |            43.0 |          52.9 |           7.9 |
|          observe_multivariate |       3 |                  98.5 |             1.5 |             166 |                    17.1 |             3.3 |            10.2 |           4.0 |          20.2 |
|              observe_submodel |       1 |                   524 |            12.4 |            1790 |                     148 |            12.6 |            42.8 |          52.0 |           8.9 |

### Distributions

|           **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |       1 |                  37.6 |             2.3 |            96.8 |                     9.2 |             4.2 |             5.9 |           3.8 |           2.5 |
|    assume_dirichlet |       1 |                  27.8 |             1.2 |            36.0 |                     5.0 |             7.7 |             5.1 |           7.6 |          10.7 |
|     assume_lkjcholu |      10 |                   153 |             2.0 |            87.3 |                    12.3 |             6.5 |            29.1 |           3.8 |          25.1 |
|     assume_mvnormal |       2 |                  44.4 |             0.9 |            24.8 |                     2.2 |            15.5 |            25.5 |         error |           5.1 |
|       assume_normal |       1 |                   567 |            10.6 |            1063 |                    98.8 |            12.9 |            45.1 |          59.5 |           8.0 |
|      assume_wishart |       3 |                  47.2 |             1.0 |            60.8 |                     6.5 |            26.1 |            34.7 |          26.5 |          18.1 |
|   observe_bernoulli |       1 |                  32.8 |             1.9 |            91.2 |                     9.6 |             5.1 |             5.4 |           3.1 |           3.4 |
| observe_categorical |       1 |                  21.2 |             1.2 |            30.4 |                     5.5 |             9.3 |             4.6 |           2.1 |           9.3 |
|   observe_von_mises |       1 |                  26.1 |             NaN |             NaN |                     7.2 |             3.8 |             5.5 |           3.8 |           3.1 |

### DynamicPPL arXiv paper

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |       2 |                  18.5 |             2.6 |            1364 |                     211 |             0.5 |             1.9 |           1.6 |           3.2 |
|        dppl_hier_poisson |      13 |                   179 |             6.8 |             111 |                    12.5 |             7.7 |            29.7 |           6.4 |           1.9 |
|      dppl_high_dim_gauss |   10000 |                263034 |           17251 |             783 |                     138 |             1.5 |           10639 |        105469 |           5.3 |
|         dppl_hmm_semisup |     115 |                   NaN |            21.9 |            77.0 |                    13.4 |             6.7 |            4532 |          62.0 |           9.6 |
|                 dppl_lda |     535 |                  8436 |             139 |             222 |                    35.3 |            11.4 |             739 |           210 |           3.0 |
| dppl_logistic_regression |     100 |                  1433 |            58.1 |            89.9 |                    15.5 |            11.6 |             138 |          29.4 |          11.3 |
|         dppl_naive_bayes |     400 |                  5607 |             322 |             339 |                    50.2 |             5.9 |             871 |          1015 |           1.8 |
|      dppl_sto_volatility |     503 |                  5856 |             104 |             246 |                    30.1 |             2.8 |             649 |          78.9 |           3.1 |

### DynamicPPL demo models

|                                  **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |       2 |                  85.7 |             2.9 |             283 |                    25.4 |             5.3 |            10.6 |           4.2 |           1.4 |
|            demo_assume_dot_observe_literal |       2 |                  96.7 |             3.4 |             337 |                    32.1 |             7.2 |            13.4 |           5.0 |           1.6 |
|                  demo_assume_index_observe |       4 |                  99.3 |             2.1 |             160 |                    18.6 |             4.4 |            14.2 |          10.0 |          39.4 |
|    demo_assume_matrix_observe_matrix_index |       4 |                  78.5 |             1.4 |            92.6 |                    10.1 |             6.8 |            16.9 |          11.8 |          35.7 |
|           demo_assume_multivariate_observe |       4 |                  88.6 |             1.6 |             121 |                    13.2 |             6.2 |            16.7 |          15.5 |          29.3 |
|   demo_assume_multivariate_observe_literal |       4 |                  90.6 |             1.7 |             118 |                    13.3 |             6.0 |            16.2 |          16.1 |          29.4 |
|                demo_assume_observe_literal |       2 |                  95.5 |             2.3 |             319 |                    33.3 |             5.2 |            12.6 |           4.9 |           1.5 |
| demo_assume_submodel_observe_index_literal |       4 |                  87.4 |             1.5 |             134 |                    14.5 |             4.7 |            13.6 |           8.8 |          11.6 |
|                    demo_dot_assume_observe |       4 |                  87.5 |             1.7 |             125 |                    13.8 |             7.4 |            17.4 |          13.0 |          24.4 |
|              demo_dot_assume_observe_index |       4 |                  88.7 |             2.0 |             127 |                    13.8 |             5.0 |            14.4 |           8.7 |          13.5 |
|      demo_dot_assume_observe_index_literal |       4 |                  92.2 |             1.6 |             136 |                    15.4 |             5.0 |            14.4 |           9.5 |          10.2 |
|       demo_dot_assume_observe_matrix_index |       4 |                  81.1 |             1.5 |             102 |                    10.9 |             8.3 |            18.3 |          12.1 |          26.6 |
|           demo_dot_assume_observe_submodel |       4 |                  83.3 |             1.5 |             114 |                    12.1 |             6.6 |            16.1 |          11.9 |          23.3 |

### Effect of model size

| **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |      10 |                   239 |             3.2 |             178 |                    22.6 |             2.4 |            23.1 |           2.3 |           0.8 |
|      n050 |      50 |                   737 |            16.7 |             164 |                    23.2 |             1.6 |            56.9 |           8.2 |           0.6 |
|      n100 |     100 |                  1342 |            27.1 |             162 |                    22.7 |             1.5 |            98.7 |          15.1 |           0.6 |
|      n500 |     500 |                  6546 |             105 |             173 |                    23.8 |             1.4 |             407 |          85.8 |           0.8 |

### PosteriorDB

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |       4 |                  55.0 |             4.0 |             571 |                    74.9 |            10.5 |             7.6 |           3.5 |           2.5 |
|             pdb_earnings |       3 |                  31.3 |             3.2 |             713 |                     126 |            21.4 |             5.1 |           9.2 |           5.7 |
|        pdb_earnings_male |       4 |                  42.2 |             2.1 |             357 |                    69.2 |            14.9 |             5.6 |           3.2 |           5.3 |
|    pdb_eightsch_centered |      10 |                   237 |             3.3 |             271 |                    31.3 |             5.2 |            36.0 |           5.5 |           2.6 |
| pdb_eightsch_noncentered |      10 |                   241 |             4.7 |             294 |                    31.9 |             5.6 |            37.3 |           5.4 |           2.6 |
|              pdb_garch11 |       4 |                  50.7 |             2.1 |             331 |                    39.2 |             4.5 |             7.1 |           1.4 |           1.9 |
|                pdb_kidiq |       3 |                  38.2 |             2.8 |             309 |                    60.2 |            12.3 |             6.3 |           3.5 |           6.2 |
|                pdb_radon |      90 |                  1068 |            27.0 |             433 |                    52.8 |             9.0 |             171 |          28.9 |           1.9 |
|                 pdb_rats |      65 |                   862 |            22.2 |             463 |                    51.6 |             6.7 |             139 |          25.8 |           2.1 |
|                pdb_sblrc |       6 |                   105 |             6.3 |             274 |                    48.0 |             5.8 |            21.3 |           7.0 |           4.0 |
|                pdb_sblri |       6 |                   106 |             6.5 |             272 |                    49.0 |             6.0 |            21.3 |           7.3 |           3.9 |

### External libraries

|      **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |       7 |                  87.5 |             1.6 |           error |                   error |            11.2 |            30.0 |         error |          14.9 |
|    delaydiffeq |       5 |                  60.9 |             1.1 |             4.9 |                     1.4 |           error |           error |         error |         error |
|         lux_nn |      20 |                   222 |             2.9 |            65.0 |                   wrong |            14.2 |            43.4 |         error |          15.0 |
| ordinarydiffeq |       5 |                  59.6 |             5.4 |            24.9 |                     5.3 |           error |           error |         wrong |          78.6 |




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
|     control_flow |        1.75e-05 |               0.00773 |        0.000105 |          0.0141 |                 0.00157 |         0.00022 |        0.000901 |      0.000416 |      6.68e-05 |
|  threaded_assume |          0.0804 |                  35.2 |           0.347 |           error |                   error |           error |            10.3 |          2.77 |          1.05 |
| threaded_observe |          0.0716 |                 0.843 |          0.0737 |           error |                   error |           error |           0.382 |         0.301 |         0.903 |

### Core Turing syntax

|                     **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |        1.04e-05 |               0.00719 |        0.000108 |           0.015 |                 0.00152 |         0.00022 |        0.000874 |      0.000402 |      6.28e-05 |
|               broadcast_macro |        0.000108 |                0.0101 |        0.000201 |          0.0362 |                  0.0033 |        0.000713 |          0.0014 |       0.00056 |      0.000164 |
|                    dot_assume |        0.000154 |                0.0279 |        0.000265 |          0.0314 |                 0.00351 |         0.00053 |         0.00275 |      0.000557 |      0.000168 |
|                   dot_observe |        2.62e-05 |               0.00403 |        0.000147 |          0.0262 |                 0.00254 |        0.000438 |        0.000546 |      0.000409 |      6.38e-05 |
|            dynamic_constraint |        0.000149 |                0.0116 |        0.000258 |            0.03 |                 0.00313 |        0.000845 |         0.00176 |       0.00062 |      0.000373 |
| multiple_constraints_same_var |         0.00134 |                0.0794 |         0.00158 |           0.048 |                 0.00512 |          0.0089 |          0.0234 |       0.00481 |       0.00781 |
|                 observe_index |        2.53e-05 |                 0.004 |         0.00013 |          0.0265 |                 0.00249 |        0.000435 |        0.000556 |      0.000418 |      6.43e-05 |
|               observe_literal |        7.23e-06 |               0.00391 |        9.14e-05 |          0.0146 |                 0.00139 |        0.000119 |        0.000393 |      0.000401 |      5.76e-05 |
|          observe_multivariate |        0.000179 |                0.0183 |        0.000277 |          0.0302 |                 0.00309 |         0.00059 |         0.00192 |      0.000731 |        0.0037 |
|              observe_submodel |        7.32e-06 |               0.00385 |        9.39e-05 |          0.0147 |                 0.00141 |        0.000131 |         0.00039 |      0.000395 |      6.48e-05 |

### Distributions

|           **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |        0.000152 |               0.00592 |        0.000356 |          0.0159 |                 0.00142 |         0.00066 |        0.000899 |      0.000584 |      0.000389 |
|    assume_dirichlet |        0.000275 |               0.00765 |        0.000342 |            0.01 |                 0.00141 |         0.00215 |         0.00143 |       0.00211 |       0.00295 |
|     assume_lkjcholu |         0.00103 |                 0.158 |         0.00204 |          0.0896 |                  0.0126 |         0.00672 |          0.0304 |       0.00404 |        0.0263 |
|     assume_mvnormal |        0.000474 |                 0.021 |        0.000409 |          0.0118 |                 0.00106 |         0.00736 |          0.0122 |         error |       0.00246 |
|       assume_normal |        6.65e-06 |               0.00379 |        7.41e-05 |         0.00802 |                0.000864 |        0.000111 |         0.00038 |      0.000415 |      5.31e-05 |
|      assume_wishart |         0.00106 |                0.0504 |         0.00107 |          0.0644 |                 0.00683 |          0.0278 |          0.0376 |        0.0285 |        0.0198 |
|   observe_bernoulli |        0.000201 |               0.00658 |        0.000388 |          0.0196 |                 0.00196 |         0.00103 |         0.00111 |      0.000652 |      0.000695 |
| observe_categorical |        0.000549 |                0.0117 |        0.000638 |          0.0168 |                 0.00307 |         0.00516 |          0.0026 |       0.00113 |       0.00527 |
|   observe_von_mises |        0.000303 |               0.00819 |             NaN |             NaN |                 0.00221 |         0.00118 |         0.00166 |       0.00117 |      0.000974 |

### DynamicPPL arXiv paper

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |         0.00849 |                  0.16 |          0.0223 |            11.6 |                    1.81 |         0.00638 |          0.0177 |        0.0205 |        0.0402 |
|        dppl_hier_poisson |         0.00196 |                 0.351 |          0.0133 |           0.219 |                  0.0246 |          0.0154 |          0.0587 |        0.0125 |        0.0039 |
|      dppl_high_dim_gauss |         0.00758 |              2.11e+03 |             140 |            6.15 |                    1.04 |          0.0213 |             156 |      1.09e+03 |          0.06 |
|         dppl_hmm_semisup |           0.264 |                   NaN |            5.78 |            20.3 |                    3.55 |            1.79 |        1.21e+03 |          16.9 |          2.61 |
|                 dppl_lda |          0.0918 |                   782 |            12.9 |              21 |                    3.24 |            1.15 |            76.6 |          20.6 |         0.296 |
| dppl_logistic_regression |           0.315 |                   621 |            18.5 |            39.8 |                    6.84 |            3.75 |            44.3 |          9.39 |          3.56 |
|         dppl_naive_bayes |           0.548 |              3.08e+03 |             177 |             186 |                    27.6 |            3.23 |             480 |           558 |         0.974 |
|      dppl_sto_volatility |          0.0273 |                   161 |            2.85 |            6.95 |                   0.858 |           0.079 |              18 |          2.22 |        0.0901 |

### DynamicPPL demo models

|                                  **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |        0.000128 |                0.0111 |        0.000383 |          0.0362 |                 0.00338 |        0.000707 |         0.00139 |      0.000552 |      0.000182 |
|            demo_assume_dot_observe_literal |        0.000104 |                  0.01 |        0.000355 |          0.0357 |                 0.00347 |        0.000771 |         0.00142 |      0.000535 |      0.000166 |
|                  demo_assume_index_observe |        0.000311 |                0.0309 |        0.000657 |            0.05 |                 0.00603 |          0.0014 |         0.00452 |       0.00315 |        0.0126 |
|    demo_assume_matrix_observe_matrix_index |        0.000495 |                0.0391 |        0.000692 |          0.0465 |                 0.00516 |         0.00345 |         0.00855 |       0.00595 |         0.018 |
|           demo_assume_multivariate_observe |        0.000372 |                0.0333 |        0.000603 |          0.0456 |                 0.00507 |         0.00234 |          0.0062 |       0.00582 |         0.011 |
|   demo_assume_multivariate_observe_literal |        0.000368 |                 0.034 |         0.00061 |          0.0443 |                 0.00494 |         0.00222 |         0.00616 |       0.00598 |        0.0109 |
|                demo_assume_observe_literal |        0.000106 |                0.0102 |        0.000246 |          0.0347 |                 0.00364 |         0.00057 |         0.00138 |      0.000533 |      0.000155 |
| demo_assume_submodel_observe_index_literal |        0.000391 |                0.0345 |        0.000592 |          0.0525 |                 0.00571 |         0.00194 |         0.00563 |       0.00345 |       0.00454 |
|                    demo_dot_assume_observe |        0.000395 |                0.0348 |        0.000671 |          0.0502 |                 0.00548 |           0.003 |         0.00711 |       0.00526 |       0.00995 |
|              demo_dot_assume_observe_index |        0.000386 |                0.0344 |        0.000784 |          0.0495 |                 0.00538 |         0.00198 |         0.00555 |       0.00344 |       0.00523 |
|      demo_dot_assume_observe_index_literal |        0.000349 |                0.0322 |        0.000558 |          0.0503 |                 0.00544 |         0.00177 |         0.00506 |       0.00336 |       0.00358 |
|       demo_dot_assume_observe_matrix_index |        0.000478 |                0.0392 |        0.000717 |          0.0495 |                 0.00534 |         0.00396 |         0.00898 |       0.00589 |        0.0134 |
|           demo_dot_assume_observe_submodel |        0.000441 |                0.0368 |        0.000678 |          0.0508 |                 0.00533 |         0.00294 |         0.00724 |       0.00533 |        0.0104 |

### Effect of model size

| **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |        0.000275 |                0.0707 |        0.000968 |          0.0526 |                  0.0067 |        0.000726 |         0.00636 |      0.000692 |      0.000238 |
|      n050 |         0.00124 |                  1.01 |          0.0231 |           0.226 |                  0.0321 |         0.00218 |          0.0707 |        0.0104 |      0.000872 |
|      n100 |         0.00243 |                  3.68 |          0.0659 |           0.445 |                  0.0624 |         0.00411 |           0.241 |        0.0427 |       0.00174 |
|      n500 |          0.0127 |                  83.3 |            1.52 |            2.22 |                   0.305 |          0.0188 |             5.7 |          1.25 |        0.0115 |

### PosteriorDB

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |         0.00175 |                0.0982 |         0.00711 |               1 |                   0.133 |          0.0188 |          0.0138 |       0.00615 |       0.00454 |
|             pdb_earnings |         0.00201 |                0.0747 |         0.00645 |            1.47 |                   0.254 |          0.0473 |          0.0144 |        0.0203 |        0.0143 |
|        pdb_earnings_male |         0.00381 |                 0.187 |         0.00931 |            1.49 |                    0.27 |          0.0708 |          0.0288 |        0.0127 |        0.0202 |
|    pdb_eightsch_centered |        0.000285 |                0.0678 |        0.000936 |          0.0778 |                 0.00907 |         0.00155 |          0.0104 |        0.0016 |      0.000736 |
| pdb_eightsch_noncentered |        0.000278 |                0.0674 |         0.00129 |           0.082 |                 0.00912 |         0.00158 |          0.0105 |       0.00151 |      0.000733 |
|              pdb_garch11 |         0.00568 |                 0.289 |           0.012 |            1.89 |                   0.223 |          0.0256 |          0.0407 |       0.00825 |         0.011 |
|                pdb_kidiq |         0.00162 |                0.0679 |         0.00489 |           0.573 |                  0.0978 |          0.0211 |           0.011 |       0.00676 |        0.0103 |
|                pdb_radon |          0.0152 |                  16.2 |           0.433 |            7.18 |                   0.874 |           0.138 |            2.89 |         0.487 |        0.0301 |
|                 pdb_rats |         0.00219 |                  1.89 |          0.0486 |            1.02 |                   0.113 |          0.0148 |           0.307 |        0.0575 |       0.00466 |
|                pdb_sblrc |        0.000575 |                0.0605 |          0.0037 |            0.16 |                  0.0276 |         0.00338 |          0.0124 |       0.00409 |       0.00237 |
|                pdb_sblri |        0.000573 |                0.0613 |          0.0038 |           0.157 |                  0.0281 |         0.00344 |          0.0122 |       0.00423 |       0.00225 |

### External libraries

|      **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |         0.00295 |                 0.258 |          0.0046 |           error |                   error |          0.0338 |          0.0927 |         error |        0.0467 |
|    delaydiffeq |           0.569 |                  35.6 |           0.639 |            2.78 |                    0.79 |           error |           error |         error |         error |
|         lux_nn |          0.0549 |                  12.2 |           0.161 |            3.59 |                   wrong |           0.817 |            2.57 |         error |          1.03 |
| ordinarydiffeq |          0.0979 |                  5.83 |           0.528 |             2.5 |                    0.53 |           error |           error |         wrong |          7.96 |




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
|         MooncakeFwd |     60 |         0 |       0 |         2 |         0 |
|           EnzymeFwd |     57 |         1 |       0 |         4 |         0 |
|           EnzymeRvs |     61 |         0 |       0 |         1 |         0 |




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
|   FiniteDifferences |            61 |             170.74 |      95.55 |                 60 |              36.71 |
|         ForwardDiff |            61 |               4.92 |       2.90 |                 61 |               1.00 |
|         ReverseDiff |            58 |             197.09 |     183.78 |                 58 |              38.14 |
| ReverseDiffCompiled |            58 |              24.22 |      22.96 |                 57 |               4.74 |
|         MooncakeRvs |            58 |               6.37 |       6.48 |                 57 |               1.24 |
|         MooncakeFwd |            60 |              29.25 |      18.96 |                 59 |               5.97 |
|           EnzymeFwd |            57 |              12.58 |       8.80 |                 56 |               2.39 |
|           EnzymeRvs |            61 |               5.29 |       5.27 |                 60 |               1.06 |




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
|  threaded_observe |         MooncakeRvs |      error |                                                                   Mooncake failed to differentiate the following method: |
|       delaydiffeq |         MooncakeRvs |      error |                                                 MethodError: no method matching +(::Vector{Float64}, ::Mooncake.NoRData) |
|    ordinarydiffeq |         MooncakeRvs |      error |                                                 MethodError: no method matching +(::Vector{Float64}, ::Mooncake.NoRData) |
|       delaydiffeq |         MooncakeFwd |      error | Mooncake.IntrinsicsWrappers.MissingIntrinsicWrapperException("Unable to translate the intrinsic Val{Core.Intrinsics.l... |
|    ordinarydiffeq |         MooncakeFwd |      error |                                                                          MethodError: Cannot `convert` an object of type |
|   assume_mvnormal |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       abstractgps |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       delaydiffeq |           EnzymeFwd |      error |                                                              EnzymeNoShadowError: Enzyme could not find shadow for value |
|            lux_nn |           EnzymeFwd |      error |                                                                         EnzymeRuntimeException: Enzyme execution failed. |
|    ordinarydiffeq |           EnzymeFwd |      wrong |                                                     ADIncorrectException: The AD backend returned an incorrect gradient. |
|       delaydiffeq |           EnzymeRvs |      error |                                                              EnzymeNoShadowError: Enzyme could not find shadow for value |



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
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Project.toml`
  [47edcb42] ADTypes v1.24.0
  [99985d1d] AbstractGPs v0.5.24
  [0ca39b1e] Chairmarks v1.3.1
  [a93c6f00] DataFrames v1.8.2
  [bcd4f6db] DelayDiffEq v6.3.0
  [8bb1440f] DelimitedFiles v1.9.1
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [31c24e10] Distributions v0.25.131
⌅ [366bfd00] DynamicPPL v0.41.8
⌃ [7da242da] Enzyme v0.13.199
  [1a297f60] FillArrays v1.17.0
  [26cc04aa] FiniteDifferences v0.12.34
  [f6369f11] ForwardDiff v1.4.5
  [d9f16b24] Functors v0.5.3
  [6fdf6af0] LogDensityProblems v2.2.0
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [b2108857] Lux v1.31.4
⌃ [da2b9cff] Mooncake v0.5.51
  [1dea7af3] OrdinaryDiffEq v7.8.1
  [b1df2697] OrdinaryDiffEqTsit5 v2.1.4
  [91a5bcdd] Plots v1.41.7
⌅ [1c4bc282] PosteriorDB v0.5.3
  [08abe8d2] PrettyTables v3.4.8
  [37e2e3b7] ReverseDiff v1.17.0
⌃ [31c91b34] SciMLBenchmarks v0.1.3 [loaded: v0.2.0]
  [1ed8b502] SciMLSensitivity v7.119.2
  [10745b16] Statistics v1.11.5
⌅ [4c63d2b9] StatsFuns v1.5.3
⌅ [fce5fe82] Turing v0.44.5
  [37e2e46d] LinearAlgebra v1.12.0
  [d6f4376e] Markdown v1.11.0
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `~/github-runners/amdci3-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Manifest.toml`
  [47edcb42] ADTypes v1.24.0
⌃ [14f7f29c] AMD v0.5.3
  [621f4979] AbstractFFTs v1.5.0
  [99985d1d] AbstractGPs v0.5.24
  [80f14c24] AbstractMCMC v5.16.0
⌅ [7a57a42e] AbstractPPL v0.14.2
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [0bf59076] AdvancedHMC v0.8.7
  [5b7e9947] AdvancedMH v0.8.10
⌅ [576499cb] AdvancedPS v0.7.2
⌅ [b5ca4192] AdvancedVI v0.6.2
  [66dad0bd] AliasTables v1.1.3
  [dce04be8] ArgCheck v2.5.0
  [4fba245c] ArrayInterface v7.30.1
  [a9b6321e] Atomix v1.1.3
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.8
  [ab4f0b2a] BFloat16s v0.6.1
  [198e06fe] BangBang v0.4.9
⌅ [76274a88] Bijectors v0.15.24
  [b2a6c25c] BinaryHeaps v1.1.0
  [70df07ce] BracketingNonlinearSolve v1.12.6
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
  [f70d9fcc] CommonWorldInvalidations v1.2.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.8
  [8f4d0f93] Conda v1.10.3
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.6.0
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.2.0
  [9a962f9c] DataAPI v1.16.0
  [a93c6f00] DataFrames v1.8.2
  [864edb3b] DataStructures v0.19.6
  [e2d170a0] DataValueInterfaces v1.0.0
  [bcd4f6db] DelayDiffEq v6.3.0
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
  [2b5f629d] DiffEqBase v7.20.0
  [459566f4] DiffEqCallbacks v4.19.3
  [77a26b50] DiffEqNoiseProcess v5.36.2
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.21
  [8d63f2c5] DispatchDoctor v0.4.28
  [b4f34e82] Distances v0.10.12
  [31c24e10] Distributions v0.25.131
  [ffbed154] DocStringExtensions v0.9.5
⌅ [366bfd00] DynamicPPL v0.41.8
  [cad2338a] EllipticalSliceSampling v2.0.0
  [4e289a0a] EnumX v1.0.7
⌃ [7da242da] Enzyme v0.13.199
  [f151be2c] EnzymeCore v0.8.21
  [e2ba6199] ExprTools v0.1.11
  [21656369] ExpressionExplorer v1.1.5
  [c87230d0] FFMPEG v0.4.5
  [b86e33f2] FFTA v0.3.1
  [7034ab61] FastBroadcast v1.4.0
  [9aa1b823] FastClosures v0.3.2
  [a4df4552] FastPower v1.5.0
  [1a297f60] FillArrays v1.17.0
  [64ca27bc] FindFirstFunctions v3.2.1
  [6a86dc24] FiniteDiff v2.33.0
  [26cc04aa] FiniteDifferences v0.12.34
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
  [f6369f11] ForwardDiff v1.4.5
  [f62d2435] FunctionProperties v1.2.0
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.13.0
  [d9f16b24] Functors v0.5.3
  [46192b85] GPUArraysCore v0.2.0
⌅ [61eb1bfa] GPUCompiler v1.23.0
  [28b8d3ca] GR v0.73.27
⌃ [a0844989] Gamma v1.1.0
  [d7ba0133] Git v1.5.0
  [076d061b] HashArrayMappedTries v0.2.0
⌅ [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.30
  [7073ff75] IJulia v1.34.4
  [7869d1d1] IRTools v0.4.20
  [615f187c] IfElse v0.1.1
  [22cec73e] InitialValues v0.3.1
⌅ [842dd82b] InlineStrings v1.4.5
  [18e54dd8] IntegerMathUtils v0.1.4
  [a98d9a8b] Interpolations v0.16.3
  [8197267c] IntervalSets v0.7.14
  [3587e190] InverseFunctions v0.1.17
  [41ab1584] InvertedIndices v1.3.1
  [92d709cd] IrrationalConstants v0.2.6
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
  [692b3bcd] JLLWrappers v1.8.0
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [63c18a36] KernelAbstractions v0.9.42
  [5ab0869b] KernelDensity v0.6.12
⌅ [ec8451be] KernelFunctions v0.10.67
  [ba0b0d4f] Krylov v0.10.9
  [2faa5264] LHLFactorization v2.2.2
  [929cbde3] LLVM v9.13.1
  [b964fa9f] LaTeXStrings v1.4.1
  [23fbe1c1] Latexify v0.16.12
  [1d6d02ad] LeftChildRightSiblingTrees v0.3.0
  [6f1fad26] Libtask v0.9.19
  [87fe0de2] LineSearch v0.1.16
  [d3d80556] LineSearches v7.8.1
  [7ed4a6bd] LinearSolve v5.15.1
  [6fdf6af0] LogDensityProblems v2.2.0
  [996a588d] LogDensityProblemsAD v1.13.1
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.2.0
  [b2108857] Lux v1.31.4
  [bb33d45b] LuxCore v1.5.3
  [82251201] LuxLib v1.15.9
  [c7f686f2] MCMCChains v7.7.0
  [be115224] MCMCDiagnosticTools v0.3.19
  [7e8f7934] MLDataDevices v1.17.10
  [e80e1ace] MLJModelInterface v1.12.1
  [1914dd2f] MacroTools v0.5.16
  [dbb5928d] MappedArrays v0.4.3
  [bb5d69b7] MaybeInplace v0.1.8
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
⌃ [da2b9cff] Mooncake v0.5.51
  [46d2c3a1] MuladdMacro v0.2.7
  [ffc61752] Mustache v1.0.21
  [d41bc354] NLSolversBase v8.0.1
  [872c559c] NNlib v0.9.45
  [77ba4419] NaNMath v1.1.4
  [c020b1a1] NaturalSort v1.0.0
  [8913a72c] NonlinearSolve v4.29.1
  [be0214bd] NonlinearSolveBase v2.49.2
  [5959db7a] NonlinearSolveFirstOrder v2.5.0
  [9a2c21bd] NonlinearSolveQuasiNewton v1.15.3
  [26075421] NonlinearSolveSpectralMethods v1.8.1
  [d8793406] ObjectFile v0.5.1
  [6fe1bfb0] OffsetArrays v1.17.0
  [429524aa] Optim v2.3.1
  [3bd65402] Optimisers v0.4.9
  [7f7a1694] Optimization v5.9.0
  [bca83a33] OptimizationBase v5.5.3
  [36348300] OptimizationOptimJL v0.4.20
⌅ [bac558e1] OrderedCollections v1.8.2 [loaded: v2.0.1]
  [1dea7af3] OrdinaryDiffEq v7.8.1
  [6ad6398a] OrdinaryDiffEqBDF v2.4.6
  [bbf590c4] OrdinaryDiffEqCore v4.16.0
  [50262376] OrdinaryDiffEqDefault v2.6.0
  [4302a76b] OrdinaryDiffEqDifferentiation v3.11.4
  [d3585ca7] OrdinaryDiffEqFunctionMap v2.3.0
  [127b3ac7] OrdinaryDiffEqNonlinearSolve v2.9.4
  [43230ef6] OrdinaryDiffEqRosenbrock v2.7.1
  [b4bd8bb3] OrdinaryDiffEqRosenbrockTableaus v2.4.2
  [2d112036] OrdinaryDiffEqSDIRK v2.9.2
  [b1df2697] OrdinaryDiffEqTsit5 v2.1.4
  [79d7bb75] OrdinaryDiffEqVerner v2.4.1
  [90014a1f] PDMats v0.11.41
⌅ [69de0a69] Parsers v2.8.7 [loaded: v2.8.8]
⌅ [569bd051] PartitionedDistributions v0.0.1
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.7
  [e409e4f3] PoissonRandom v0.4.13
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
⌅ [1c4bc282] PosteriorDB v0.5.3
  [d236fae5] PreallocationTools v1.7.1
  [aea7be01] PrecompileTools v1.3.4
  [21216c6a] Preferences v1.5.2
  [08abe8d2] PrettyTables v3.4.8
  [27ebfcd6] Primes v0.5.7
  [33c8b6b6] ProgressLogging v0.1.6
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.4.1
  [1fd47b50] QuadGK v2.11.3
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [a3311ec8] ReactantCore v0.1.21
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
  [7e49a35a] RuntimeGeneratedFunctions v0.5.25
  [26aad666] SSMProblems v0.6.1
  [0bca4576] SciMLBase v3.50.2
⌃ [31c91b34] SciMLBenchmarks v0.1.3 [loaded: v0.2.0]
  [19f34311] SciMLJacobianOperators v0.1.18
  [a6db7da4] SciMLLogging v2.1.0
  [c0aeaf25] SciMLOperators v1.30.0
  [431bcebd] SciMLPublic v1.3.0
  [1ed8b502] SciMLSensitivity v7.119.2
  [53ae85a6] SciMLStructures v1.10.5
  [30f210dd] ScientificTypesBase v3.1.0
  [7e506255] ScopedValues v1.6.2
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.10
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.1.1
  [727e6d20] SimpleNonlinearSolve v2.14.1
  [a2af1166] SortingAlgorithms v1.2.3
⌃ [a57abbd0] SparseColumnPivotedQR v2.1.7
  [9f842d2f] SparseConnectivityTracer v1.2.3
  [dc90abb0] SparseInverseSubset v0.1.3
⌃ [0a514795] SparseMatrixColorings v0.4.27
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
⌅ [fce5fe82] Turing v0.44.5
  [1cfade01] UnicodeFun v0.4.1
  [013be700] UnsafeAtomics v0.3.2
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [d49dbf32] WeightInitializers v1.3.4
  [efce3f68] WoodburyMatrices v1.1.0
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [a5390f91] ZipFile v0.10.1
  [e88e6eb3] Zygote v0.7.13
  [700de1a5] ZygoteRules v0.2.8
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
⌅ [7cc45869] Enzyme_jll v0.0.290+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.8.3+0
⌅ [b22a6f82] FFMPEG_jll v8.1.2+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.5.1+0
  [d2c73de3] GR_jll v0.73.27+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.55.0+0
  [7746bdde] Glib_jll v2.88.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
  [2e76f6c2] HarfBuzz_jll v100.14003.0+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+1
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.1.0+0
  [dad2f222] LLVMExtra_jll v0.0.47+0
  [1d63c593] LLVMOpenMP_jll v22.1.7+0
  [ad6e5548] LibTracyClient_jll v0.13.1+0
⌅ [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
  [4b2f31a3] Libmount_jll v2.42.0+0
  [89763e89] Libtiff_jll v4.7.3+0
  [38a345b3] Libuuid_jll v2.42.0+0
  [856f044c] MKL_jll v2025.2.0+0
  [e7412a2a] Ogg_jll v1.3.6+0
  [9bd350c2] OpenSSH_jll v10.5.1+0
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
  [ffd25f8a] XZ_jll v5.8.3+0
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
  [33bec58e] Xorg_xkeyboard_config_jll v2.47.0+2
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
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
  [a9144af2] libsodium_jll v1.0.21+0
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
  [1a1011a3] SharedArrays v1.11.0
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
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

