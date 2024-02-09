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
using SurrogatesRandomForest
using Plots
using Statistics
using PrettyTables
using BenchmarkTools
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
  0.6423429368716482
 -0.7248728290811458
  0.782449624406479
  0.768980422203787
 -0.8680956922945071
  0.8135459525545304
  0.9028175030672306
  0.9469435427810834
 -0.10103665435398895
 -0.45897561664645603
  ⋮
  0.68561921982168
 -0.8385787222394305
  0.9832473659148661
  0.9907772321553996
  0.1366326552059298
 -0.4446422940052878
  0.058946729414397506
  0.9924243467700988
  0.09227883249780619
```





## Plot training and test points

```julia
scatter(x_train, y_train, label="Training Points", xlabel="X-axis", ylabel="Y-axis", legend=:topright)
scatter!(x_test, y_test, label="Testing Points")
```

![](figures/tensor_product_5_1.png)


## Fit surrogate models

```julia
randomforest_surrogate = RandomForestSurrogate(x_train, y_train, lb, ub, num_round = 10)
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
random_forest_train_pred = randomforest_surrogate.(x_train)
radial_train_pred = radial_surrogate.(x_train)
kriging_train_pred = kriging_surrogate.(x_train)
loba_train_pred = loba_surrogate.(x_train)

## Test data
random_forest_test_pred = randomforest_surrogate.(x_test)
radial_test_pred = radial_surrogate.(x_test)
kriging_test_pred = kriging_surrogate.(x_test)
loba_test_pred = loba_surrogate.(x_test)
```

```
150-element Vector{Float64}:
  0.6423439771186126
 -0.7248683835440488
  0.7824477925762459
  0.7689820594874365
 -0.8681000132553149
  0.8135517206821925
  0.9028151639095192
  0.946926956458695
 -0.1010349708655274
 -0.45897986158808735
  ⋮
  0.6856217452466795
 -0.8385799343033842
  0.983245745760918
  0.9907771841711847
  0.13653315618871242
 -0.44465609939785644
  0.058973681013981594
  0.9924057262133301
  0.09228677639294092
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
mse_rf_train = calculate_mse(random_forest_train_pred, y_train)
mse_radial_train = calculate_mse(radial_train_pred, y_train)
mse_krig_train = calculate_mse(kriging_train_pred, y_train)
mse_loba_train = calculate_mse(loba_train_pred, y_train)

## Test MSE
mse_rf_test = calculate_mse(random_forest_test_pred, y_test)
mse_radial_test = calculate_mse(radial_test_pred, y_test)
mse_krig_test = calculate_mse(kriging_test_pred, y_test)
mse_loba_test = calculate_mse(loba_test_pred, y_test)
```

```
2.566964355434295e-8
```





## Compare MSE

```julia
models = ["Random Forest", "Radial Basis", "Kriging", "Lobachevsky"]
train_mses = [mse_rf_train, mse_radial_train, mse_krig_train, mse_loba_train]
test_mses = [mse_rf_test, mse_radial_test, mse_krig_test, mse_loba_test]
mses = sort(collect(zip(test_mses, train_mses, models)))
pretty_table(hcat(getindex.(mses, 3), getindex.(mses, 2), getindex.(mses, 1)), header=["Model", "Training MSE", "Test MSE"])
```

```
┌───────────────┬──────────────┬────────────┐
│         Model │ Training MSE │   Test MSE │
├───────────────┼──────────────┼────────────┤
│   Lobachevsky │   3.7375e-19 │ 2.56696e-8 │
│       Kriging │   1.16824e-5 │  1.4371e-5 │
│  Radial Basis │  1.74579e-30 │  8.3866e-5 │
│ Random Forest │   0.00176894 │ 0.00739329 │
└───────────────┴──────────────┴────────────┘
```





## Plot predictions

```julia
xs = -5:0.01:5
plot(xs, tensor_product_function.(xs), label="True function", legend=:top, color=:black)
plot!(xs, randomforest_surrogate.(xs), label="Random Forest", legend=:top, color=:green)
plot!(xs, radial_surrogate.(xs), label="Radial Basis", legend=:top, color=:red)
plot!(xs, kriging_surrogate.(xs), label="Kriging", legend=:top, color=:blue)
plot!(xs, loba_surrogate.(xs), label="Lobachevsky", legend=:top, color=:purple)
```

![](figures/tensor_product_11_1.png)



## Time evaluation

```julia
time_original = @belapsed tensor_product_function.(x_test)
time_rf = @belapsed randomforest_surrogate.(x_test)
time_radial = @belapsed radial_surrogate.(x_test)
time_krig = @belapsed kriging_surrogate.(x_test)
time_loba = @belapsed loba_surrogate.(x_test)
```

```
0.001351646
```





## Compare time performance

```julia
times = ["Random Forest" => time_rf, "Radial Basis" => time_radial, "Kriging" => time_krig, "Lobachevsky" => time_loba, "Original Function" => time_original]
sorted_times = sort(times, by=x->x[2])
pretty_table(hcat(first.(sorted_times), last.(sorted_times)), header=["Model", "Time(s)"])
```

```
┌───────────────────┬─────────────┐
│             Model │     Time(s) │
├───────────────────┼─────────────┤
│ Original Function │    1.508e-6 │
│           Kriging │ 0.000338447 │
│       Lobachevsky │  0.00135165 │
│      Radial Basis │  0.00188795 │
│     Random Forest │  0.00409138 │
└───────────────────┴─────────────┘
```


