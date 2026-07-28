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
|     control_flow |       2 |                   486 |             5.6 |             585 |                    48.8 |            22.5 |            47.3 |          22.3 |           3.7 |
|  threaded_assume |      50 |                   427 |             4.2 |           error |                   error |           error |             227 |          30.3 |           9.7 |
| threaded_observe |       1 |                  11.0 |             1.0 |           error |                   error |           error |             6.9 |           3.7 |           9.8 |

### Core Turing syntax

|                     **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |       2 |                   238 |             3.0 |             358 |                    25.0 |             9.3 |            24.0 |          11.7 |           1.8 |
|               broadcast_macro |       2 |                   120 |             1.9 |             324 |                    22.3 |             7.9 |            13.3 |           5.3 |           1.6 |
|                    dot_assume |       5 |                   185 |             1.7 |             158 |                    11.9 |             5.7 |            23.0 |           3.5 |           1.0 |
|                   dot_observe |       1 |                   163 |             4.3 |            1310 |                    86.4 |            29.1 |            24.9 |          22.3 |           3.6 |
|            dynamic_constraint |       2 |                  88.0 |             1.6 |             173 |                    13.8 |             5.2 |            10.1 |           4.1 |           2.4 |
| multiple_constraints_same_var |       4 |                  60.8 |             1.2 |            29.8 |                     2.7 |             6.2 |            14.6 |           3.1 |           3.2 |
|                 observe_index |       1 |                   266 |             7.6 |            1377 |                    89.8 |            29.4 |            24.6 |          22.1 |           3.6 |
|               observe_literal |       1 |                   586 |            11.9 |            1663 |                     113 |            31.0 |            50.4 |          49.1 |           7.4 |
|          observe_multivariate |       3 |                   111 |             1.4 |             144 |                    10.2 |             6.5 |            14.0 |           4.0 |          13.7 |
|              observe_submodel |       1 |                   574 |            13.1 |            1693 |                     118 |            32.5 |            54.6 |          49.5 |           7.9 |

### Distributions

|           **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |       1 |                  44.6 |             2.2 |            91.8 |                     6.0 |             4.0 |             5.6 |           3.4 |           1.4 |
|    assume_dirichlet |       1 |                  29.1 |             1.2 |            30.2 |                     3.7 |             6.2 |             4.7 |           6.6 |          10.3 |
|     assume_lkjcholu |      10 |                   158 |             2.1 |            79.6 |                     8.6 |             6.3 |            24.9 |           3.8 |          27.5 |
|     assume_mvnormal |       2 |                  38.2 |             0.8 |            15.3 |                     1.0 |             9.9 |            18.2 |         error |           4.4 |
|       assume_normal |       1 |                   639 |            10.2 |            1009 |                    70.0 |            26.7 |            52.0 |          53.0 |           7.3 |
|      assume_wishart |       3 |                  45.1 |             0.7 |            37.9 |                     3.0 |            15.5 |            25.0 |          17.3 |          13.3 |
|   observe_bernoulli |       1 |                  33.4 |             1.9 |             113 |                     6.6 |             4.5 |             4.9 |           2.9 |           2.2 |
| observe_categorical |       1 |                  21.4 |             1.2 |            24.4 |                     3.8 |             7.5 |             4.1 |           1.9 |           7.6 |
|   observe_von_mises |       1 |                  31.5 |             NaN |             NaN |                     5.8 |             3.7 |             5.0 |           3.7 |           3.1 |

### DynamicPPL arXiv paper

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |       2 |                  25.5 |             3.3 |            1788 |                     224 |            31.8 |             5.1 |           2.1 |           6.0 |
|        dppl_hier_poisson |      13 |                   180 |             5.9 |            97.6 |                     7.7 |             6.9 |            29.0 |           5.9 |           1.8 |
|      dppl_high_dim_gauss |   10000 |                234725 |           39134 |             948 |                     130 |            35.0 |           58846 |         46651 |           5.4 |
|         dppl_hmm_semisup |     115 |                   NaN |            24.3 |            76.3 |                    11.6 |             6.1 |            4647 |          62.0 |           9.4 |
|                 dppl_lda |     535 |                  8251 |             133 |             215 |                    26.4 |            10.8 |            1617 |           423 |           3.4 |
| dppl_logistic_regression |     100 |                  1341 |            60.9 |             106 |                    16.3 |             8.7 |             223 |          26.1 |           9.7 |
|         dppl_naive_bayes |     400 |                  5573 |             332 |             314 |                    37.4 |             8.5 |             910 |           304 |           1.7 |
|      dppl_sto_volatility |     503 |                  5969 |             116 |             226 |                    20.9 |             4.8 |            1068 |          67.5 |           4.5 |

### DynamicPPL demo models

|                                  **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |       2 |                   112 |             3.4 |             295 |                    21.8 |             7.4 |            12.4 |           5.0 |           1.7 |
|            demo_assume_dot_observe_literal |       2 |                  90.6 |             2.6 |             206 |                    15.4 |             6.2 |            11.0 |           3.9 |           1.5 |
|                  demo_assume_index_observe |       4 |                  99.1 |             1.8 |             124 |                     9.2 |             5.9 |            15.0 |           7.5 |          21.1 |
|    demo_assume_matrix_observe_matrix_index |       4 |                  83.2 |             1.4 |            76.7 |                     6.4 |             7.0 |            16.3 |           9.6 |          18.9 |
|           demo_assume_multivariate_observe |       4 |                  92.2 |             1.5 |            94.6 |                     7.9 |             7.1 |            17.0 |          11.4 |          21.4 |
|   demo_assume_multivariate_observe_literal |       4 |                  88.9 |             1.4 |            86.1 |                     7.3 |             6.1 |            15.9 |          11.3 |          20.2 |
|                demo_assume_observe_literal |       2 |                   115 |             1.9 |             297 |                    22.2 |             6.3 |            12.3 |           5.0 |           1.5 |
| demo_assume_submodel_observe_index_literal |       4 |                  86.7 |             1.3 |             104 |                     8.5 |             6.2 |            14.8 |           7.1 |           9.0 |
|                    demo_dot_assume_observe |       4 |                  82.8 |             1.4 |            86.4 |                     6.9 |             6.7 |            16.4 |           9.7 |          17.3 |
|              demo_dot_assume_observe_index |       4 |                  84.9 |             1.8 |            97.9 |                     8.0 |             5.9 |            13.5 |           6.8 |          13.0 |
|      demo_dot_assume_observe_index_literal |       4 |                  90.8 |             1.4 |             110 |                     9.2 |             6.1 |            15.4 |           7.4 |           7.9 |
|       demo_dot_assume_observe_matrix_index |       4 |                  79.6 |             1.3 |            78.3 |                     6.1 |             7.4 |            15.5 |           9.2 |          16.2 |
|           demo_dot_assume_observe_submodel |       4 |                  85.8 |             1.3 |            88.3 |                     7.1 |             6.9 |            15.5 |           9.8 |          17.7 |

### Effect of model size

| **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |      10 |                   257 |             2.9 |             158 |                    12.5 |             5.4 |            39.9 |           2.8 |           0.8 |
|      n050 |      50 |                   738 |            17.3 |             143 |                    12.5 |             5.0 |             144 |           9.7 |           0.6 |
|      n100 |     100 |                  1349 |            20.4 |             148 |                    12.0 |             5.2 |             284 |          16.3 |           0.6 |
|      n500 |     500 |                  5714 |            85.4 |             142 |                    12.7 |             4.7 |            1367 |          76.6 |           0.7 |

### PosteriorDB

|                **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |       4 |                  63.6 |             4.9 |             828 |                    82.2 |            19.1 |            13.9 |           4.1 |           2.8 |
|             pdb_earnings |       3 |                  37.8 |             3.5 |             805 |                    90.4 |            30.0 |             8.5 |           3.6 |           6.6 |
|        pdb_earnings_male |       4 |                  48.7 |             2.8 |             414 |                    55.0 |            22.8 |             8.1 |           3.3 |           4.8 |
|    pdb_eightsch_centered |      10 |                   260 |             3.2 |             246 |                    20.1 |             6.9 |            38.6 |           6.0 |           2.5 |
| pdb_eightsch_noncentered |      10 |                   259 |             4.1 |             259 |                    20.2 |             6.8 |            39.4 |           5.7 |           2.5 |
|              pdb_garch11 |       4 |                  51.6 |             2.1 |             348 |                    29.8 |             4.7 |             6.8 |           1.4 |           1.8 |
|                pdb_kidiq |       3 |                  39.8 |             2.0 |             350 |                    47.0 |            17.5 |             8.6 |           4.1 |           4.7 |
|                pdb_radon |      90 |                  1075 |            31.7 |             437 |                    38.7 |             9.1 |             156 |          28.4 |           1.8 |
|                 pdb_rats |      65 |                   885 |            24.4 |             385 |                    33.5 |             8.5 |             124 |          24.7 |           1.9 |
|                pdb_sblrc |       6 |                   118 |             6.1 |             261 |                    36.6 |            12.2 |            27.1 |           7.8 |           3.3 |
|                pdb_sblri |       6 |                   110 |             5.7 |             252 |                    35.5 |            11.9 |            24.9 |           7.3 |           2.8 |

### External libraries

|      **Model** | **Dim** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| -------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |       7 |                  86.2 |             1.7 |           error |                   error |            10.5 |            30.2 |         error |          10.0 |
|    delaydiffeq |       5 |                  59.7 |             0.8 |           error |                   error |           error |           error |         error |         error |
|         lux_nn |      20 |                   222 |             2.8 |            62.3 |                   wrong |            17.3 |            40.1 |         error |          15.2 |
| ordinarydiffeq |       5 |                  58.0 |             4.4 |           error |                   error |             5.4 |           error |         error |          66.3 |




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
|     control_flow |        1.97e-05 |               0.00962 |        0.000112 |          0.0137 |                0.000965 |        0.000446 |        0.000935 |       0.00044 |      7.24e-05 |
|  threaded_assume |          0.0847 |                  36.4 |           0.359 |           error |                   error |           error |            19.4 |          2.75 |         0.927 |
| threaded_observe |          0.0714 |                 0.784 |          0.0743 |           error |                   error |           error |           0.523 |         0.304 |           0.9 |

### Core Turing syntax

|                     **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -----------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               assume_submodel |        3.82e-05 |               0.00985 |        0.000113 |          0.0142 |                0.000989 |        0.000357 |        0.000918 |      0.000446 |      6.81e-05 |
|               broadcast_macro |        0.000102 |                0.0122 |        0.000199 |          0.0333 |                 0.00226 |        0.000799 |         0.00136 |      0.000541 |      0.000167 |
|                    dot_assume |        0.000184 |                0.0346 |        0.000315 |          0.0293 |                 0.00218 |         0.00106 |         0.00427 |      0.000657 |      0.000191 |
|                   dot_observe |        1.83e-05 |               0.00482 |        0.000139 |           0.025 |                 0.00165 |        0.000539 |        0.000468 |      0.000409 |      6.63e-05 |
|            dynamic_constraint |        0.000157 |                 0.014 |        0.000256 |          0.0276 |                 0.00219 |         0.00081 |         0.00159 |      0.000651 |      0.000381 |
| multiple_constraints_same_var |         0.00142 |                0.0865 |         0.00176 |          0.0431 |                 0.00396 |         0.00914 |          0.0219 |       0.00456 |       0.00478 |
|                 observe_index |        1.82e-05 |               0.00494 |        0.000139 |          0.0256 |                 0.00163 |         0.00055 |        0.000459 |      0.000405 |      6.73e-05 |
|               observe_literal |        8.25e-06 |               0.00483 |        9.82e-05 |          0.0137 |                0.000949 |        0.000256 |        0.000417 |      0.000407 |      6.11e-05 |
|          observe_multivariate |        0.000196 |                0.0218 |        0.000287 |          0.0285 |                 0.00201 |         0.00128 |          0.0028 |      0.000786 |       0.00272 |
|              observe_submodel |        7.85e-06 |               0.00466 |        0.000103 |          0.0139 |                0.000948 |        0.000256 |        0.000428 |       0.00039 |      6.22e-05 |

### Distributions

|           **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| -------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|         assume_beta |        0.000163 |               0.00725 |        0.000357 |           0.015 |                0.000998 |        0.000659 |         0.00091 |      0.000553 |      0.000225 |
|    assume_dirichlet |        0.000306 |               0.00901 |        0.000386 |          0.0094 |                 0.00114 |         0.00192 |         0.00143 |       0.00202 |       0.00317 |
|     assume_lkjcholu |         0.00111 |                 0.176 |         0.00233 |          0.0888 |                 0.00951 |         0.00695 |          0.0277 |       0.00423 |        0.0307 |
|     assume_mvnormal |        0.000751 |                0.0291 |        0.000591 |          0.0117 |                0.000748 |         0.00764 |           0.014 |         error |        0.0033 |
|       assume_normal |        7.62e-06 |               0.00488 |        7.82e-05 |          0.0077 |                0.000534 |        0.000204 |        0.000396 |      0.000403 |      5.68e-05 |
|      assume_wishart |         0.00163 |                0.0741 |         0.00117 |          0.0623 |                 0.00508 |           0.026 |          0.0421 |        0.0296 |        0.0224 |
|   observe_bernoulli |        0.000208 |               0.00779 |          0.0004 |          0.0236 |                 0.00138 |        0.000954 |         0.00103 |      0.000615 |      0.000465 |
| observe_categorical |         0.00064 |                0.0138 |        0.000757 |          0.0157 |                 0.00248 |         0.00483 |         0.00267 |       0.00121 |       0.00495 |
|   observe_von_mises |        0.000304 |               0.00958 |             NaN |             NaN |                 0.00179 |         0.00115 |         0.00155 |       0.00114 |      0.000956 |

### DynamicPPL arXiv paper

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|       dppl_gauss_unknown |         0.00628 |                 0.166 |          0.0213 |            11.4 |                    1.44 |           0.205 |          0.0357 |        0.0134 |        0.0374 |
|        dppl_hier_poisson |         0.00215 |                 0.385 |          0.0129 |            0.21 |                  0.0167 |          0.0151 |          0.0632 |        0.0129 |       0.00393 |
|      dppl_high_dim_gauss |         0.00568 |              1.58e+03 |             222 |             6.1 |                   0.893 |           0.221 |             377 |           551 |        0.0483 |
|         dppl_hmm_semisup |           0.271 |                   NaN |            6.57 |            20.7 |                    3.18 |            1.67 |        1.27e+03 |            17 |          2.61 |
|                 dppl_lda |          0.0976 |                   814 |            12.9 |            21.3 |                     2.6 |            1.07 |             160 |          42.5 |         0.347 |
| dppl_logistic_regression |           0.328 |                   643 |            29.4 |            34.9 |                    5.37 |            2.92 |            75.7 |          12.7 |          3.19 |
|         dppl_naive_bayes |           0.587 |              3.27e+03 |             195 |             184 |                      22 |            5.03 |             535 |           179 |          1.01 |
|      dppl_sto_volatility |          0.0297 |                   178 |            3.45 |            6.77 |                   0.619 |           0.143 |            32.4 |          2.01 |         0.134 |

### DynamicPPL demo models

|                                  **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|                    demo_assume_dot_observe |         0.00011 |                0.0123 |        0.000378 |          0.0325 |                  0.0024 |        0.000809 |         0.00136 |      0.000554 |      0.000189 |
|            demo_assume_dot_observe_literal |        0.000154 |                0.0141 |        0.000397 |          0.0321 |                 0.00237 |        0.000957 |          0.0017 |      0.000605 |      0.000226 |
|                  demo_assume_index_observe |        0.000377 |                0.0376 |        0.000689 |          0.0475 |                  0.0036 |         0.00225 |         0.00571 |       0.00284 |       0.00808 |
|    demo_assume_matrix_observe_matrix_index |        0.000549 |                0.0457 |        0.000768 |          0.0424 |                 0.00354 |          0.0039 |         0.00913 |       0.00539 |        0.0105 |
|           demo_assume_multivariate_observe |        0.000448 |                0.0414 |        0.000672 |          0.0424 |                 0.00361 |         0.00322 |         0.00765 |       0.00512 |        0.0096 |
|   demo_assume_multivariate_observe_literal |        0.000485 |                0.0433 |        0.000712 |          0.0417 |                  0.0036 |         0.00301 |         0.00792 |       0.00556 |        0.0098 |
|                demo_assume_observe_literal |        0.000109 |                0.0126 |        0.000208 |          0.0324 |                 0.00243 |        0.000689 |         0.00134 |      0.000542 |      0.000166 |
| demo_assume_submodel_observe_index_literal |        0.000464 |                0.0405 |        0.000616 |           0.049 |                 0.00394 |         0.00291 |         0.00697 |       0.00342 |       0.00419 |
|                    demo_dot_assume_observe |         0.00054 |                0.0447 |        0.000735 |          0.0471 |                 0.00373 |         0.00367 |         0.00891 |       0.00527 |       0.00939 |
|              demo_dot_assume_observe_index |        0.000502 |                0.0428 |        0.000884 |          0.0494 |                 0.00404 |         0.00297 |         0.00686 |       0.00343 |        0.0066 |
|      demo_dot_assume_observe_index_literal |         0.00044 |                0.0402 |        0.000634 |          0.0488 |                 0.00405 |         0.00269 |         0.00681 |       0.00335 |        0.0035 |
|       demo_dot_assume_observe_matrix_index |        0.000601 |                0.0482 |        0.000804 |          0.0471 |                 0.00373 |         0.00447 |          0.0095 |       0.00557 |        0.0098 |
|           demo_dot_assume_observe_submodel |        0.000535 |                0.0465 |        0.000715 |          0.0481 |                 0.00378 |         0.00373 |         0.00849 |       0.00529 |       0.00979 |

### Effect of model size

| **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ---------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|      n010 |        0.000332 |                0.0852 |        0.000987 |          0.0527 |                 0.00414 |         0.00181 |          0.0135 |      0.000936 |       0.00028 |
|      n050 |         0.00156 |                  1.15 |          0.0272 |           0.226 |                  0.0196 |         0.00776 |           0.226 |        0.0153 |      0.000952 |
|      n100 |         0.00309 |                  4.17 |          0.0631 |           0.461 |                   0.038 |          0.0162 |            0.88 |        0.0509 |       0.00183 |
|      n500 |          0.0156 |                  95.1 |             1.4 |            2.22 |                   0.201 |          0.0734 |            21.6 |          1.19 |        0.0108 |

### PosteriorDB

|                **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| ------------------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|               pdb_arma11 |         0.00113 |                 0.072 |         0.00572 |           0.942 |                  0.0935 |          0.0227 |          0.0162 |       0.00471 |       0.00328 |
|             pdb_earnings |         0.00183 |                0.0772 |         0.00721 |            1.47 |                   0.193 |          0.0667 |          0.0173 |       0.00803 |        0.0132 |
|        pdb_earnings_male |          0.0036 |                 0.178 |          0.0104 |            1.49 |                   0.204 |          0.0893 |          0.0332 |        0.0136 |        0.0193 |
|    pdb_eightsch_centered |        0.000299 |                0.0795 |        0.000971 |          0.0739 |                   0.006 |         0.00208 |          0.0118 |       0.00182 |      0.000751 |
| pdb_eightsch_noncentered |        0.000302 |                0.0796 |         0.00124 |          0.0796 |                 0.00622 |          0.0021 |          0.0121 |       0.00173 |       0.00075 |
|              pdb_garch11 |         0.00538 |                 0.278 |          0.0114 |            1.89 |                   0.162 |          0.0255 |          0.0374 |       0.00758 |       0.00997 |
|                pdb_kidiq |         0.00161 |                0.0726 |         0.00334 |           0.575 |                  0.0758 |          0.0308 |          0.0153 |       0.00723 |       0.00773 |
|                pdb_radon |          0.0158 |                    17 |           0.525 |            6.97 |                   0.639 |           0.152 |            2.58 |         0.471 |        0.0298 |
|                 pdb_rats |         0.00245 |                  2.18 |          0.0599 |           0.944 |                  0.0823 |          0.0211 |           0.305 |        0.0608 |       0.00481 |
|                pdb_sblrc |        0.000588 |                0.0694 |         0.00381 |           0.156 |                  0.0219 |         0.00738 |          0.0165 |       0.00481 |       0.00202 |
|                pdb_sblri |        0.000619 |                0.0698 |         0.00362 |           0.156 |                   0.022 |         0.00757 |          0.0159 |       0.00485 |       0.00187 |

### External libraries

|      **Model** | **Primal (ms)** | **FiniteDifferences** | **ForwardDiff** | **ReverseDiff** | **ReverseDiffCompiled** | **MooncakeRvs** | **MooncakeFwd** | **EnzymeFwd** | **EnzymeRvs** |
| --------------:| ---------------:| ---------------------:| ---------------:| ---------------:| -----------------------:| ---------------:| ---------------:| -------------:| -------------:|
|    abstractgps |         0.00369 |                 0.318 |         0.00641 |           error |                   error |          0.0394 |           0.115 |         error |        0.0383 |
|    delaydiffeq |           0.778 |                  48.7 |           0.636 |           error |                   error |           error |           error |         error |         error |
|         lux_nn |          0.0565 |                  12.7 |           0.159 |            3.52 |                   wrong |            1.02 |            2.35 |         error |          1.03 |
| ordinarydiffeq |           0.115 |                  6.72 |           0.512 |           error |                   error |           0.638 |           error |         error |          7.91 |




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
|         ReverseDiff |     56 |         0 |       1 |         5 |         0 |
| ReverseDiffCompiled |     56 |         1 |       0 |         5 |         0 |
|         MooncakeRvs |     59 |         0 |       0 |         3 |         0 |
|         MooncakeFwd |     60 |         0 |       0 |         2 |         0 |
|           EnzymeFwd |     57 |         0 |       0 |         5 |         0 |
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
|   FiniteDifferences |            61 |             176.71 |     109.83 |                 60 |              39.67 |
|         ForwardDiff |            61 |               4.71 |       2.81 |                 61 |               1.00 |
|         ReverseDiff |            56 |             195.40 |     165.54 |                 56 |              38.29 |
| ReverseDiffCompiled |            56 |              17.40 |      14.61 |                 55 |               3.44 |
|         MooncakeRvs |            59 |               9.09 |       7.01 |                 58 |               1.85 |
|         MooncakeFwd |            60 |              34.45 |      20.63 |                 59 |               7.33 |
|           EnzymeFwd |            57 |              11.38 |       7.44 |                 56 |               2.25 |
|           EnzymeRvs |            61 |               4.69 |       4.48 |                 60 |               0.97 |




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
|       delaydiffeq |         ReverseDiff |      error |   MethodError: no method matching arraydist(::Distributions.Poisson{ReverseDiff.TrackedReal{Float64, Float64, Nothing}}) |
|    ordinarydiffeq |         ReverseDiff |      error |                                                                         DimensionMismatch: inconsistent array dimensions |
|   threaded_assume | ReverseDiffCompiled |      error |                                                                                                      TaskFailedException |
|  threaded_observe | ReverseDiffCompiled |      error |                                                                                                      TaskFailedException |
|       abstractgps | ReverseDiffCompiled |      error | MethodError: -(::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}, ::FillArrays.Zeros{... |
|       delaydiffeq | ReverseDiffCompiled |      error |   MethodError: no method matching arraydist(::Distributions.Poisson{ReverseDiff.TrackedReal{Float64, Float64, Nothing}}) |
|            lux_nn | ReverseDiffCompiled |      wrong |                                           ADIncorrectException: The AD backend returned an incorrect value and gradient. |
|    ordinarydiffeq | ReverseDiffCompiled |      error |                                                                         DimensionMismatch: inconsistent array dimensions |
|   threaded_assume |         MooncakeRvs |      error |              Differentiating through threading is not safe and is unsupported in reverse mode. Use forward mode instead. |
|  threaded_observe |         MooncakeRvs |      error |              Differentiating through threading is not safe and is unsupported in reverse mode. Use forward mode instead. |
|       delaydiffeq |         MooncakeRvs |      error | MethodError: no method matching increment!!(::Mooncake.NoRData, ::Mooncake.RData{@NamedTuple{u::Mooncake.NoRData, u_a... |
|       delaydiffeq |         MooncakeFwd |      error | MethodError: no method matching frule!!(::Mooncake.Dual{typeof(getindex), Mooncake.NoTangent}, ::Mooncake.Dual{SciMLB... |
|    ordinarydiffeq |         MooncakeFwd |      error |                                                                          MethodError: Cannot `convert` an object of type |
|   assume_mvnormal |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       abstractgps |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       delaydiffeq |           EnzymeFwd |      error |                                    IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis. |
|            lux_nn |           EnzymeFwd |      error |                                                                         EnzymeRuntimeException: Enzyme execution failed. |
|    ordinarydiffeq |           EnzymeFwd |      error |                                                                                  EnzymeNoDerivativeError: Current scope: |
|       delaydiffeq |           EnzymeRvs |      error | MethodError: no method matching EnzymeCore.MixedDuplicated(::SciMLBase.ODESolution{Float64, 2, Vector{Vector{Float64}... |



## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/AutomaticDifferentiationTuring","TuringADTests.jmd")
```

Computer Information:

```
Julia Version 1.11.9
Commit 53a02c0720c (2026-02-06 00:27 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)
Threads: 128 default, 0 interactive, 64 GC (on 128 virtual cores)
Environment:
  JULIA_NUM_THREADS = auto

```

Package Information:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Project.toml`
  [47edcb42] ADTypes v1.22.2
  [99985d1d] AbstractGPs v0.5.24
  [0ca39b1e] Chairmarks v1.3.1
  [a93c6f00] DataFrames v1.8.2
⌅ [bcd4f6db] DelayDiffEq v5.74.1
  [8bb1440f] DelimitedFiles v1.9.1
  [a0c0ee7d] DifferentiationInterface v0.7.20
⌃ [31c24e10] Distributions v0.25.129
⌅ [366bfd00] DynamicPPL v0.41.8
  [7da242da] Enzyme v0.13.195
  [1a297f60] FillArrays v1.17.0
  [26cc04aa] FiniteDifferences v0.12.34
⌃ [f6369f11] ForwardDiff v1.4.1
  [d9f16b24] Functors v0.5.2
  [6fdf6af0] LogDensityProblems v2.2.0
⌅ [2ab3a3ac] LogExpFunctions v0.3.29
  [b2108857] Lux v1.31.4
  [da2b9cff] Mooncake v0.5.40
⌃ [1dea7af3] OrdinaryDiffEq v6.111.0
⌅ [b1df2697] OrdinaryDiffEqTsit5 v1.12.0
  [91a5bcdd] Plots v1.41.6
⌅ [1c4bc282] PosteriorDB v0.5.3
  [08abe8d2] PrettyTables v3.4.2
  [37e2e3b7] ReverseDiff v1.17.0
⌃ [31c91b34] SciMLBenchmarks v0.1.3
⌃ [1ed8b502] SciMLSensitivity v7.106.0
  [10745b16] Statistics v1.11.1
⌅ [4c63d2b9] StatsFuns v1.5.2
⌅ [fce5fe82] Turing v0.44.5
  [37e2e46d] LinearAlgebra v1.11.0
  [d6f4376e] Markdown v1.11.0
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
```

And the full manifest:

```
Status `/julia/github-runners/amdci1-1/_work/SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/AutomaticDifferentiationTuring/Manifest.toml`
  [47edcb42] ADTypes v1.22.2
  [14f7f29c] AMD v0.5.3
  [621f4979] AbstractFFTs v1.5.0
  [99985d1d] AbstractGPs v0.5.24
  [80f14c24] AbstractMCMC v5.16.0
⌅ [7a57a42e] AbstractPPL v0.14.2
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.45
  [79e6a3ab] Adapt v4.7.0
  [0bf59076] AdvancedHMC v0.8.6
  [5b7e9947] AdvancedMH v0.8.10
⌅ [576499cb] AdvancedPS v0.7.2
⌅ [b5ca4192] AdvancedVI v0.6.2
  [66dad0bd] AliasTables v1.1.3
  [dce04be8] ArgCheck v2.5.0
  [4fba245c] ArrayInterface v7.28.1
  [a9b6321e] Atomix v1.1.3
  [13072b0f] AxisAlgorithms v1.1.0
  [39de3d68] AxisArrays v0.4.8
  [198e06fe] BangBang v0.4.9
⌅ [76274a88] Bijectors v0.15.24
  [d1d4a3ce] BitFlags v0.1.10
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [70df07ce] BracketingNonlinearSolve v1.12.1
  [fa961155] CEnum v0.5.0
  [2a0fbf3d] CPUSummary v0.2.7
  [082447d4] ChainRules v1.73.0
  [d360d2e6] ChainRulesCore v1.26.1
  [0ca39b1e] Chairmarks v1.3.1
  [9e997f8a] ChangesOfVariables v0.1.10
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [944b1d66] CodecZlib v0.7.8
  [35d6a980] ColorSchemes v3.31.0
  [3da002f7] ColorTypes v0.12.1
  [c3611d14] ColorVectorSpace v0.11.0
  [5ae59095] Colors v0.13.1
  [38540f10] CommonSolve v0.2.11
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.1.1
  [34da2185] Compat v4.18.1
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.6
  [f0e56b4a] ConcurrentUtilities v2.5.1
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
⌅ [bcd4f6db] DelayDiffEq v5.74.1
  [8bb1440f] DelimitedFiles v1.9.1
  [b429d917] DensityInterface v0.4.0
⌅ [2b5f629d] DiffEqBase v6.218.0
  [459566f4] DiffEqCallbacks v4.18.3
⌃ [77a26b50] DiffEqNoiseProcess v5.32.0
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.16.0
  [a0c0ee7d] DifferentiationInterface v0.7.20
  [8d63f2c5] DispatchDoctor v0.4.28
  [b4f34e82] Distances v0.10.12
⌃ [31c24e10] Distributions v0.25.129
  [ffbed154] DocStringExtensions v0.9.5
⌅ [366bfd00] DynamicPPL v0.41.8
  [cad2338a] EllipticalSliceSampling v2.0.0
  [4e289a0a] EnumX v1.0.7
  [7da242da] Enzyme v0.13.195
  [f151be2c] EnzymeCore v0.8.21
  [460bff9d] ExceptionUnwrapping v0.1.11
⌃ [d4d017d3] ExponentialUtilities v1.31.0
  [e2ba6199] ExprTools v0.1.11
  [21656369] ExpressionExplorer v1.1.4
  [55351af7] ExproniconLite v0.10.14
  [c87230d0] FFMPEG v0.4.5
  [b86e33f2] FFTA v0.3.1
  [7034ab61] FastBroadcast v1.3.4
  [9aa1b823] FastClosures v0.3.2
  [442a2c76] FastGaussQuadrature v1.3.0
  [a4df4552] FastPower v1.3.4
  [1a297f60] FillArrays v1.17.0
  [6a86dc24] FiniteDiff v2.32.0
  [26cc04aa] FiniteDifferences v0.12.34
⌅ [53c48c17] FixedPointNumbers v0.8.6
  [1fa38f19] Format v1.3.7
⌃ [f6369f11] ForwardDiff v1.4.1
⌅ [f62d2435] FunctionProperties v0.1.7
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v1.10.1
  [d9f16b24] Functors v0.5.2
  [46192b85] GPUArraysCore v0.2.0
⌅ [61eb1bfa] GPUCompiler v1.23.0
  [28b8d3ca] GR v0.73.26
  [a0844989] Gamma v1.1.0
  [c145ed77] GenericSchur v0.5.6
  [d7ba0133] Git v1.5.0
  [42e2da0e] Grisu v1.0.2
⌅ [cd3eb016] HTTP v1.11.0
  [076d061b] HashArrayMappedTries v0.2.0
⌅ [eafb193a] Highlights v0.5.3
  [34004b35] HypergeometricFunctions v0.3.29
  [7073ff75] IJulia v1.34.4
  [7869d1d1] IRTools v0.4.19
  [615f187c] IfElse v0.1.1
  [22cec73e] InitialValues v0.3.1
  [842dd82b] InlineStrings v1.4.5
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
  [ae98c720] Jieko v0.2.1
  [63c18a36] KernelAbstractions v0.9.42
  [5ab0869b] KernelDensity v0.6.12
⌅ [ec8451be] KernelFunctions v0.10.67
  [ba0b0d4f] Krylov v0.10.8
  [929cbde3] LLVM v9.11.0
  [b964fa9f] LaTeXStrings v1.4.0
  [23fbe1c1] Latexify v0.16.11
  [10f19ff3] LayoutPointers v0.1.17
⌅ [1d6d02ad] LeftChildRightSiblingTrees v0.2.1
  [6f1fad26] Libtask v0.9.18
  [87fe0de2] LineSearch v0.1.12
  [d3d80556] LineSearches v7.7.1
⌅ [7ed4a6bd] LinearSolve v3.87.0
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
  [d125e4d3] ManualMemory v0.1.8
  [dbb5928d] MappedArrays v0.4.3
⌃ [bb5d69b7] MaybeInplace v0.1.6
  [739be429] MbedTLS v1.1.10
  [442fdcdd] Measures v0.3.3
  [e1d29d7a] Missings v1.2.0
  [dbe65cb8] MistyClosures v2.1.0
  [da2b9cff] Mooncake v0.5.40
  [2e0e35c7] Moshi v0.3.12
  [46d2c3a1] MuladdMacro v0.2.6
  [ffc61752] Mustache v1.0.21
  [d41bc354] NLSolversBase v8.0.0
⌃ [872c559c] NNlib v0.9.38
  [77ba4419] NaNMath v1.1.4
  [c020b1a1] NaturalSort v1.0.0
⌃ [8913a72c] NonlinearSolve v4.19.1
⌅ [be0214bd] NonlinearSolveBase v2.30.3
⌃ [5959db7a] NonlinearSolveFirstOrder v2.1.1
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.13.1
⌃ [26075421] NonlinearSolveSpectralMethods v1.7.1
  [d8793406] ObjectFile v0.5.1
  [6fe1bfb0] OffsetArrays v1.17.0
  [4d8831e6] OpenSSL v1.6.1
  [429524aa] Optim v2.2.1
  [3bd65402] Optimisers v0.4.7
  [7f7a1694] Optimization v5.6.5
  [bca83a33] OptimizationBase v5.2.2
  [36348300] OptimizationOptimJL v0.4.16
⌅ [bac558e1] OrderedCollections v1.8.2
⌃ [1dea7af3] OrdinaryDiffEq v6.111.0
⌅ [89bda076] OrdinaryDiffEqAdamsBashforthMoulton v1.11.0
⌅ [6ad6398a] OrdinaryDiffEqBDF v1.26.0
⌅ [bbf590c4] OrdinaryDiffEqCore v3.33.1
⌅ [50262376] OrdinaryDiffEqDefault v1.14.0
⌅ [4302a76b] OrdinaryDiffEqDifferentiation v2.9.0
⌅ [9286f039] OrdinaryDiffEqExplicitRK v1.12.0
⌅ [e0540318] OrdinaryDiffEqExponentialRK v1.15.0
⌅ [becaefa8] OrdinaryDiffEqExtrapolation v1.18.0
⌅ [5960d6e9] OrdinaryDiffEqFIRK v1.26.0
⌅ [101fe9f7] OrdinaryDiffEqFeagin v1.10.0
⌅ [d3585ca7] OrdinaryDiffEqFunctionMap v1.11.0
⌅ [d28bc4f8] OrdinaryDiffEqHighOrderRK v1.12.0
⌅ [9f002381] OrdinaryDiffEqIMEXMultistep v1.14.0
⌅ [521117fe] OrdinaryDiffEqLinear v1.12.0
⌅ [1344f307] OrdinaryDiffEqLowOrderRK v1.13.0
⌅ [b0944070] OrdinaryDiffEqLowStorageRK v1.15.0
⌅ [127b3ac7] OrdinaryDiffEqNonlinearSolve v1.28.0
⌅ [c9986a66] OrdinaryDiffEqNordsieck v1.11.0
⌅ [5dd0a6cf] OrdinaryDiffEqPDIRK v1.14.0
⌅ [5b33eab2] OrdinaryDiffEqPRK v1.10.0
⌅ [04162be5] OrdinaryDiffEqQPRK v1.10.0
⌅ [af6ede74] OrdinaryDiffEqRKN v1.12.0
⌅ [43230ef6] OrdinaryDiffEqRosenbrock v1.31.1
⌅ [2d112036] OrdinaryDiffEqSDIRK v1.14.0
⌅ [669c94d9] OrdinaryDiffEqSSPRK v1.14.0
⌅ [e3e12d00] OrdinaryDiffEqStabilizedIRK v1.14.0
⌅ [358294b1] OrdinaryDiffEqStabilizedRK v1.11.1
⌅ [fa646aed] OrdinaryDiffEqSymplecticRK v1.13.0
⌅ [b1df2697] OrdinaryDiffEqTsit5 v1.12.0
⌅ [79d7bb75] OrdinaryDiffEqVerner v1.14.0
  [90014a1f] PDMats v0.11.40
  [69de0a69] Parsers v2.8.6
⌅ [569bd051] PartitionedDistributions v0.0.1
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.4
  [91a5bcdd] Plots v1.41.6
  [e409e4f3] PoissonRandom v0.4.12
  [f517fe37] Polyester v0.7.19
  [1d0040c9] PolyesterWeave v0.2.2
  [2dfb63ee] PooledArrays v1.4.3
  [85a6dd25] PositiveFactorizations v0.2.4
⌅ [1c4bc282] PosteriorDB v0.5.3
  [d236fae5] PreallocationTools v1.3.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.5.2
  [08abe8d2] PrettyTables v3.4.2
  [27ebfcd6] Primes v0.5.7
  [33c8b6b6] ProgressLogging v0.1.6
  [92933f4c] ProgressMeter v1.11.0
  [43287f4e] PtrArrays v1.4.0
  [0c0d3e7f] PureKLU v1.1.1
  [1fd47b50] QuadGK v2.11.3
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.5
  [a3311ec8] ReactantCore v0.1.21
  [c1ae055f] RealDot v0.1.0
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
⌅ [731186ca] RecursiveArrayTools v3.54.0
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.2.3
  [37e2e3b7] ReverseDiff v1.17.0
  [708f8203] Richardson v1.4.3
  [79098fc4] Rmath v0.9.0
  [f2b01f46] Roots v3.0.6
  [7e49a35a] RuntimeGeneratedFunctions v0.5.22
  [94e857df] SIMDTypes v0.1.0
  [26aad666] SSMProblems v0.6.1
⌅ [0bca4576] SciMLBase v2.155.1
⌃ [31c91b34] SciMLBenchmarks v0.1.3
  [19f34311] SciMLJacobianOperators v0.1.16
⌅ [a6db7da4] SciMLLogging v1.10.1
⌃ [c0aeaf25] SciMLOperators v1.24.4
  [431bcebd] SciMLPublic v1.2.3
⌃ [1ed8b502] SciMLSensitivity v7.106.0
  [53ae85a6] SciMLStructures v1.10.3
  [30f210dd] ScientificTypesBase v3.1.0
  [7e506255] ScopedValues v1.6.2
  [6c6a2e73] Scratch v1.3.0
  [91c51154] SentinelArrays v1.4.10
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.12.0
  [a2af1166] SortingAlgorithms v1.2.3
  [a57abbd0] SparseColumnPivotedQR v2.1.4
  [9f842d2f] SparseConnectivityTracer v1.2.2
  [dc90abb0] SparseInverseSubset v0.1.2
  [0a514795] SparseMatrixColorings v0.4.27
  [276daf66] SpecialFunctions v2.8.0
  [860ef19b] StableRNGs v1.0.4
  [aedffcd0] Static v1.4.4
  [0d7ed370] StaticArrayInterface v1.10.0
  [90137ffa] StaticArrays v1.9.18
  [1e83bf80] StaticArraysCore v1.4.4
  [64bff920] StatisticalTraits v3.5.0
  [10745b16] Statistics v1.11.1
  [82ae8749] StatsAPI v1.8.0
  [2913bbd2] StatsBase v0.34.12
⌅ [4c63d2b9] StatsFuns v1.5.2
  [7792a7ef] StrideArraysCore v0.5.9
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.4.6
  [09ab397b] StructArrays v0.7.3
  [53d494c1] StructIO v0.3.1
  [856f2bd8] StructTypes v1.11.0
  [2efcf032] SymbolicIndexingInterface v0.3.51
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.13.0
  [62fd8b95] TensorCore v0.1.1
  [5d786b92] TerminalLoggers v0.1.7
  [8290d209] ThreadingUtilities v0.5.6
⌅ [a759f4b9] TimerOutputs v0.5.29
  [9f7883ad] Tracker v0.2.38
  [e689c965] Tracy v0.1.6
  [3bb67fe8] TranscodingStreams v0.11.3
  [781d530d] TruncatedStacktraces v1.4.0
⌅ [fce5fe82] Turing v0.44.5
  [5c2747f8] URIs v1.6.1
  [1cfade01] UnicodeFun v0.4.1
  [013be700] UnsafeAtomics v0.3.1
  [41fe7b60] Unzip v0.2.0
  [81def892] VersionParsing v1.3.0
  [44d3d7a6] Weave v0.10.12
  [d49dbf32] WeightInitializers v1.3.4
  [efce3f68] WoodburyMatrices v1.1.0
  [ddb6d928] YAML v0.4.16
  [c2297ded] ZMQ v1.5.1
  [a5390f91] ZipFile v0.10.1
  [e88e6eb3] Zygote v0.7.12
  [700de1a5] ZygoteRules v0.2.7
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.7+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [7cc45869] Enzyme_jll v0.0.289+0
  [2702e6a9] EpollShim_jll v0.0.20230411+1
  [2e619515] Expat_jll v2.8.2+0
  [b22a6f82] FFMPEG_jll v8.1.2+0
  [a3f928ae] Fontconfig_jll v2.17.1+0
  [d7e528f0] FreeType2_jll v2.14.3+1
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.4.1+1
  [d2c73de3] GR_jll v0.73.26+0
⌅ [b0724c58] GettextRuntime_jll v0.22.4+0
  [61579ee1] Ghostscript_jll v9.55.1+0
  [020c3dae] Git_LFS_jll v3.7.1+0
  [f8c6e375] Git_jll v2.54.0+0
  [7746bdde] Glib_jll v2.86.3+0
  [3b182d85] Graphite2_jll v1.3.16+0
⌅ [2e76f6c2] HarfBuzz_jll v8.5.1+0
  [1d5cc7b8] IntelOpenMP_jll v2025.2.0+0
  [aacddb02] JpegTurbo_jll v3.2.0+0
  [c1c5ebd0] LAME_jll v3.100.3+0
  [88015f11] LERC_jll v4.1.0+0
  [dad2f222] LLVMExtra_jll v0.0.44+0
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
  [9bd350c2] OpenSSH_jll v10.4.1+0
  [458c3c95] OpenSSL_jll v3.5.7+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
  [91d4177d] Opus_jll v1.6.1+0
  [36c8627f] Pango_jll v1.57.1+0
  [30392449] Pixman_jll v0.46.4+0
  [c0090381] Qt6Base_jll v6.10.2+2
  [629bc702] Qt6Declarative_jll v6.10.2+2
  [ce943373] Qt6ShaderTools_jll v6.10.2+1
  [6de9746b] Qt6Svg_jll v6.10.2+0
  [e99dba38] Qt6Wayland_jll v6.10.2+1
  [f50d1b31] Rmath_jll v0.5.1+0
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
  [a51aa0fd] Xorg_libXi_jll v1.8.3+0
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
  [214eeab7] fzf_jll v0.61.1+0
  [a4ae2306] libaom_jll v3.13.3+0
  [0ac62f75] libass_jll v0.17.4+0
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
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [9fa8497b] Future v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [37e2e46d] LinearAlgebra v1.11.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.11.0
  [de0858da] Printf v1.11.0
  [3fa0cd96] REPL v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [1a1011a3] SharedArrays v1.11.0
  [6462fe0b] Sockets v1.11.0
  [2f01184e] SparseArrays v1.11.0
  [f489334b] StyledStrings v1.11.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [deac9b47] LibCURL_jll v8.6.0+0
  [e37daf67] LibGit2_jll v1.7.2+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.6+0
  [14a3606d] MozillaCACerts_jll v2023.12.12
  [4536629a] OpenBLAS_jll v0.3.27+1
  [05823500] OpenLibm_jll v0.8.5+0
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.7.0+0
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
```

