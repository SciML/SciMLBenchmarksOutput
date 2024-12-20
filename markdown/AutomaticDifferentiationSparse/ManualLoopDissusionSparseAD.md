---
author: "Yolhan Mannes"
title: "Diffusion operator loop sparse AD benchmarks"
---
```julia
using DifferentiationInterface
using DifferentiationInterfaceTest
using LinearAlgebra
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings
import Enzyme,ForwardDiff,Mooncake
```




## Backends tested

```julia
bcks = [
    AutoEnzyme(mode=Enzyme.Reverse),
    AutoEnzyme(mode=Enzyme.Forward),
    AutoMooncake(config=nothing),
    AutoForwardDiff(),
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm()
    ),
    AutoSparse(
        AutoEnzyme(mode=Enzyme.Forward);
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm()
    )
    ]
```

```
6-element Vector{ADTypes.AbstractADType}:
 ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, EnzymeCore.FF
IABI, false, false}())
 ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, f
alse, false}())
 ADTypes.AutoMooncake{Nothing}(nothing)
 ADTypes.AutoForwardDiff()
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=S
parseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Sparse
MatrixColorings.GreedyColoringAlgorithm{:direct, SparseMatrixColorings.Natu
ralOrder}(SparseMatrixColorings.NaturalOrder()))
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode
{false, EnzymeCore.FFIABI, false, false}()), sparsity_detector=SparseConnec
tivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMatrixColor
ings.GreedyColoringAlgorithm{:direct, SparseMatrixColorings.NaturalOrder}(S
parseMatrixColorings.NaturalOrder()))
```





## Diffusion operator simple loop

```julia
uin() = 0.0
uout() = 0.0
function Diffusion(u)
    du = zero(u)
    for i in eachindex(du,u)
        if i == 1
            ug = uin()
            ud = u[i+1]
        elseif i == length(u)
            ug = u[i-1]
            ud = uout()
        else
            ug = u[i-1]
            ud = u[i+1] 
        end
        du[i] = ug + ud -2*u[i]
    end
    return du
end;
```




## Manual jacobian
```julia
function DDiffusion(u)
    A = diagm(
        -1 => ones(length(u)-1),
        0=>-2 .*ones(length(u)),
        1 => ones(length(u)-1))
    return A
end;
```




## Define Scenarios

```julia
u = rand(1000)
scenarios = [ Scenario{:jacobian,:out}(Diffusion,u,res1=DDiffusion(u))];
```




## Run Benchmarks

```julia
df = benchmark_differentiation(bcks, scenarios)
```

```
Test Summary:                                                              
                                                                           
                                                                           
                                                                           
                                       | Pass  Total     Time
Testing benchmarks                                                         
                                                                           
                                                                           
                                                                           
                                       |   12     12  1m32.6s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, EnzymeCore.F
FIABI, false, false}())                                                    
                                                                           
                                                                           
                                       |    2      2    23.8s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, 
false, false}())                                                           
                                                                           
                                                                           
                                       |    2      2    22.7s
  ADTypes.AutoMooncake{Nothing}(nothing)                                   
                                                                           
                                                                           
                                                                           
                                       |    2      2    34.5s
  ADTypes.AutoForwardDiff()                                                
                                                                           
                                                                           
                                                                           
                                       |    2      2     4.1s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=
SparseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Spars
eMatrixColorings.GreedyColoringAlgorithm{:direct, SparseMatrixColorings.Nat
uralOrder}(SparseMatrixColorings.NaturalOrder()))                          
                                       |    2      2     4.1s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMod
e{false, EnzymeCore.FFIABI, false, false}()), sparsity_detector=SparseConne
ctivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMatrixColo
rings.GreedyColoringAlgorithm{:direct, SparseMatrixColorings.NaturalOrder}(
SparseMatrixColorings.NaturalOrder())) |    2      2     3.2s
12×12 DataFrame
 Row │ backend                            scenario                         
  o ⋯
     │ Abstract…                          Scenario…                        
  S ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │ AutoEnzyme(mode=ReverseMode{fals…  Scenario{:jacobian,:out} Diffusi…
  v ⋯
   2 │ AutoEnzyme(mode=ReverseMode{fals…  Scenario{:jacobian,:out} Diffusi…
  j
   3 │ AutoEnzyme(mode=ForwardMode{fals…  Scenario{:jacobian,:out} Diffusi…
  v
   4 │ AutoEnzyme(mode=ForwardMode{fals…  Scenario{:jacobian,:out} Diffusi…
  j
   5 │ AutoMooncake{Nothing}(nothing)     Scenario{:jacobian,:out} Diffusi…
  v ⋯
   6 │ AutoMooncake{Nothing}(nothing)     Scenario{:jacobian,:out} Diffusi…
  j
   7 │ AutoForwardDiff()                  Scenario{:jacobian,:out} Diffusi…
  v
   8 │ AutoForwardDiff()                  Scenario{:jacobian,:out} Diffusi…
  j
   9 │ AutoSparse(dense_ad=AutoForwardD…  Scenario{:jacobian,:out} Diffusi…
  v ⋯
  10 │ AutoSparse(dense_ad=AutoForwardD…  Scenario{:jacobian,:out} Diffusi…
  j
  11 │ AutoSparse(dense_ad=AutoEnzyme(m…  Scenario{:jacobian,:out} Diffusi…
  v
  12 │ AutoSparse(dense_ad=AutoEnzyme(m…  Scenario{:jacobian,:out} Diffusi…
  j
                                                              10 columns om
itted
```


