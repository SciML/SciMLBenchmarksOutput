---
author: "Mridul Jain, Sathvik Bhagavan, Chris Rackauckas"
title: "Tensor Product Function"
---


The tensor product function is defined as:

``f(x) =  \prod_{i=1}^{d} \cos(a\pi x_i)``

where:

- \(d\): Represents the dimensionality of the input vector \(x\).
- \(x_i\): Represents the \(i\)-th component of the input vector.
- \(a\): A constant parameter.

## Package Imports

```julia
using Surrogates
using XGBoost
using Plots
using Statistics
using PrettyTables
using BenchmarkTools

## Python SMT
using PythonCall
using CondaPkg
smt = pyimport("smt.surrogate_models")
np = pyimport("numpy")
```

```
Python: <module 'numpy' from '/home/crackauc/github-runners/amdci3-1/_work/
SciMLBenchmarks.jl/SciMLBenchmarks.jl/benchmarks/Surrogates/.CondaPkg/.pixi
/envs/default/lib/python3.12/site-packages/numpy/__init__.py'>
```





## Define the function

```julia
function tensor_product_function(x)
    a = 0.5
    return prod(cos.(a * π * x))
end
```

```
tensor_product_function (generic function with 1 method)
```





## Define parameters for training and test data

```julia
lb = -5.0 # Lower bound of sampling range
ub = 5.0  # Upper bound of sampling range
n_train = 100 # Number of training points
n_test = 150 # Number of testing points
```

```
150
```





## Sample training and test data points

```julia
x_train = sample(n_train, lb, ub, SobolSample())  # Sample training data points
y_train = tensor_product_function.(x_train)  # Calculate corresponding function values
x_test = sample(n_test, lb, ub, RandomSample())  # Sample larger test data set
y_test = tensor_product_function.(x_test)  # Calculate corresponding true function values
```

```
150-element Vector{Float64}:
  0.935887427145325
  0.9975900873907227
 -0.6458498405604215
  0.9988799120262536
  0.8735104495131871
  0.30816165968068104
 -0.9423712894821147
  0.9839522785702726
  0.2645183346283872
  0.9990054010563754
  ⋮
  0.9189086331663047
 -0.9002958533162877
 -0.5414301669379763
 -0.3582489052163669
  0.9322570785046133
 -0.017929317429171747
 -0.5608687856139128
  0.3056911219390553
  0.12790792540657953
```





## Plot training and test points

```julia
scatter(x_train, y_train, label="Training Points", xlabel="X-axis", ylabel="Y-axis", legend=:topright)
scatter!(x_test, y_test, label="Testing Points")
```

![](figures/tensor_product_5_1.png)


## Fit surrogate models

```julia
# SMT
radial_surrogate_smt = smt.RBF(d0=1.0, poly_degree=1.0, print_global=Py(false))
radial_surrogate_smt.set_training_values(np.array(x_train), np.array(y_train))
radial_surrogate_smt.train()

theta = 0.5 / max(1e-6 * abs(ub - lb), std(x_train))^2.0
kriging_surrogate_smt = smt.KRG(theta0=np.array([theta]), poly="quadratic", print_global=Py(false))
kriging_surrogate_smt.set_training_values(np.array(x_train), np.array(y_train))
kriging_surrogate_smt.train()

# Julia
xgboost_surrogate = XGBoostSurrogate(x_train, y_train, lb, ub, num_round = 10)
radial_surrogate = RadialBasis(x_train, y_train, lb, ub)
kriging_surrogate = Kriging(x_train, y_train, lb, ub)
loba_surrogate = LobachevskySurrogate(x_train, y_train, lb, ub, alpha = 2.0, n = 6)
```

```
(::Surrogates.LobachevskySurrogate{Vector{Float64}, Vector{Float64}, Float6
4, Int64, Float64, Float64, Vector{Float64}, Bool}) (generic function with 
2 methods)
```





## Predictions on training and test data

```julia
## Training data
radial_train_pred_smt = radial_surrogate_smt.predict_values(np.array(x_train))
radial_train_pred_smt = pyconvert(Matrix{Float64}, radial_train_pred_smt)[:, 1]
kriging_train_pred_smt = kriging_surrogate_smt.predict_values(np.array(x_train))
kriging_train_pred_smt = pyconvert(Matrix{Float64}, kriging_train_pred_smt)[:, 1]
xgboost_train_pred = xgboost_surrogate.(x_train)
radial_train_pred = radial_surrogate.(x_train)
kriging_train_pred = kriging_surrogate.(x_train)
loba_train_pred = loba_surrogate.(x_train)

## Test data
radial_test_pred_smt = radial_surrogate_smt.predict_values(np.array(x_test))
radial_test_pred_smt = pyconvert(Matrix{Float64}, radial_test_pred_smt)[:, 1]
kriging_test_pred_smt = kriging_surrogate_smt.predict_values(np.array(x_test))
kriging_test_pred_smt = pyconvert(Matrix{Float64}, kriging_test_pred_smt)[:, 1]
xgboost_test_pred = xgboost_surrogate.(x_test)
radial_test_pred = radial_surrogate.(x_test)
kriging_test_pred = kriging_surrogate.(x_test)
loba_test_pred = loba_surrogate.(x_test)
```

```
150-element Vector{Float64}:
  0.9358861674140941
  0.9975388748850074
 -0.6458414303147375
  0.9988688360960992
  0.8735132460667197
  0.30816320911072015
 -0.9423730631225773
  0.9839513342522227
  0.2644368989534278
  0.9990089094065537
  ⋮
  0.9189085653477829
 -0.9002783753025287
 -0.5414421139578379
 -0.3582736740847303
  0.9322701659182252
 -0.017911006375438545
 -0.5608742487977938
  0.3056741710372062
  0.12793344988085864
```





## Define the MSE function

```julia
function calculate_mse(predictions, true_values)
    return mean((predictions .- true_values).^2)  # Calculate mean of squared errors
end
```

```
calculate_mse (generic function with 1 method)
```





## Calculate MSE for the models

```julia
## Training MSE
mse_radial_train_smt = calculate_mse(radial_train_pred_smt, y_train)
mse_krig_train_smt = calculate_mse(kriging_train_pred_smt, y_train)
mse_xgb_train = calculate_mse(xgboost_train_pred, y_train)
mse_radial_train = calculate_mse(radial_train_pred, y_train)
mse_krig_train = calculate_mse(kriging_train_pred, y_train)
mse_loba_train = calculate_mse(loba_train_pred, y_train)

## Test MSE
mse_radial_test_smt = calculate_mse(radial_test_pred_smt, y_test)
mse_krig_test_smt = calculate_mse(kriging_test_pred_smt, y_test)
mse_xgb_test = calculate_mse(xgboost_test_pred, y_test)
mse_radial_test = calculate_mse(radial_test_pred, y_test)
mse_krig_test = calculate_mse(kriging_test_pred, y_test)
mse_loba_test = calculate_mse(loba_test_pred, y_test)
```

```
1.3998771549368414e-9
```





## Compare MSE

```julia
models = ["XGBoost", "Radial Basis", "Kriging", "Lobachevsky", "Radial Basis (SMT)", "Kriging (SMT)"]
train_mses = [mse_xgb_train, mse_radial_train, mse_krig_train, mse_loba_train, mse_radial_train_smt, mse_krig_train_smt]
test_mses = [mse_xgb_test, mse_radial_test, mse_krig_test, mse_loba_test, mse_radial_test_smt, mse_krig_test_smt]
mses = sort(collect(zip(test_mses, train_mses, models)))
pretty_table(hcat(getindex.(mses, 3), getindex.(mses, 2), getindex.(mses, 1)), column_labels=["Model", "Training MSE", "Test MSE"])
```

```
┌────────────────────┬──────────────┬─────────────┐
│              Model │ Training MSE │    Test MSE │
├────────────────────┼──────────────┼─────────────┤
│      Kriging (SMT) │  3.51665e-18 │ 3.64066e-18 │
│            Kriging │  3.50414e-14 │ 8.35139e-14 │
│ Radial Basis (SMT) │  2.34051e-14 │  1.2855e-13 │
│        Lobachevsky │   3.7375e-19 │  1.39988e-9 │
│       Radial Basis │  1.02176e-30 │  2.70087e-5 │
│            XGBoost │   0.00176894 │  0.00770549 │
└────────────────────┴──────────────┴─────────────┘
```





## Plot predictions

```julia
xs = -5:0.01:5

radial_pred_smt = radial_surrogate_smt.predict_values(np.array(xs))
radial_pred_smt = pyconvert(Matrix{Float64}, radial_pred_smt)[:, 1]
kriging_pred_smt = kriging_surrogate_smt.predict_values(np.array(xs))
kriging_pred_smt = pyconvert(Matrix{Float64}, kriging_pred_smt)[:, 1]

plot(xs, radial_pred_smt, label="Radial Basis (SMT)", legend=:top, color=:cyan)
plot!(xs, kriging_surrogate.(xs), label="Kriging (SMT)", legend=:top, color=:magenta)
plot!(xs, tensor_product_function.(xs), label="True function", legend=:top, color=:black)
plot!(xs, xgboost_surrogate.(xs), label="XGBoost", legend=:top, color=:green)
plot!(xs, radial_surrogate.(xs), label="Radial Basis", legend=:top, color=:red)
plot!(xs, kriging_surrogate.(xs), label="Kriging", legend=:top, color=:blue)
plot!(xs, loba_surrogate.(xs), label="Lobachevsky", legend=:top, color=:purple)
```

![](figures/tensor_product_11_1.png)



## Time evaluation

```julia
time_original = @belapsed tensor_product_function.(x_test)
time_radial_smt = @belapsed radial_surrogate_smt.predict_values(np.array(x_test))
time_krig_smt = @belapsed kriging_surrogate_smt.predict_values(np.array(x_test))
time_xgb = @belapsed xgboost_surrogate.(x_test)
time_radial = @belapsed radial_surrogate.(x_test)
time_krig = @belapsed kriging_surrogate.(x_test)
time_loba = @belapsed loba_surrogate.(x_test)
```

```
0.001080974
```





## Compare time performance

```julia
times = ["XGBoost" => time_xgb, "Radial Basis" => time_radial, "Kriging" => time_krig, "Lobachevsky" => time_loba, "Radial Basis (SMT)" => time_radial_smt, "Kriging (SMT)" => time_krig_smt, "Original Function" => time_original]
sorted_times = sort(times, by=x->x[2])
pretty_table(hcat(first.(sorted_times), last.(sorted_times)), column_labels=["Model", "Time(s)"])
```

```
┌────────────────────┬─────────────┐
│              Model │     Time(s) │
├────────────────────┼─────────────┤
│  Original Function │    1.517e-6 │
│       Radial Basis │   2.2869e-5 │
│ Radial Basis (SMT) │ 0.000410218 │
│            Kriging │ 0.000488377 │
│      Kriging (SMT) │ 0.000702746 │
│        Lobachevsky │  0.00108097 │
│            XGBoost │   0.0582976 │
└────────────────────┴─────────────┘
```


