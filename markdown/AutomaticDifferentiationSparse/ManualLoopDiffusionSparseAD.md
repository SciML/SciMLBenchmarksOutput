---
author: "Yolhan Mannes"
title: "Diffusion operator loop sparse AD benchmarks"
---
```julia
using DifferentiationInterface
using DifferentiationInterfaceTest
using Chairmarks
using DataFrames
using LinearAlgebra
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings
import Chairmarks
import Enzyme, ForwardDiff, Mooncake
import Markdown, PrettyTables, Printf
```




## Backends tested

```julia
bcks = [
    AutoEnzyme(mode = Enzyme.Reverse),
    AutoEnzyme(mode = Enzyme.Forward),
    AutoMooncake(config = nothing),
    AutoForwardDiff(),
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    ),
    AutoSparse(
        AutoEnzyme(mode = Enzyme.Forward);
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
]
```

```
6-element Vector{ADTypes.AbstractADType}:
 ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, false, Enzyme
Core.FFIABI, false, false}())
 ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, f
alse, false, false}())
 ADTypes.AutoMooncake()
 ADTypes.AutoForwardDiff()
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=S
parseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Sparse
MatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColor
ings.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))
 ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode
{false, EnzymeCore.FFIABI, false, false, false}()), sparsity_detector=Spars
eConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMatr
ixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings
.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))
```





## Diffusion operator simple loop

```julia
uin() = 0.0
uout() = 0.0
function Diffusion(u)
    du = zero(u)
    for i in eachindex(du, u)
        if i == 1
            ug = uin()
            ud = u[i + 1]
        elseif i == length(u)
            ug = u[i - 1]
            ud = uout()
        else
            ug = u[i - 1]
            ud = u[i + 1]
        end
        du[i] = ug + ud - 2*u[i]
    end
    return du
end;
```




## Manual jacobian

```julia
function DDiffusion(u)
    A = diagm(
        -1 => ones(length(u)-1),
        0=>-2 .* ones(length(u)),
        1 => ones(length(u)-1))
    return A
end;
```




## Define Scenarios

```julia
u = rand(1000)
scenarios = [Scenario{:jacobian, :out}(Diffusion, u; res1 = DDiffusion(u))];
```




## Run Benchmarks

```julia
df = DataFrame(benchmark_differentiation(bcks, scenarios))
table = PrettyTables.pretty_table(
    String,
    df;
    backend = :markdown,
    column_labels = names(df),
    formatters = [PrettyTables.fmt__printf("%.1e")]
)

Markdown.parse(table)
```

```
Test Summary:                                                              
                                                                           
                                                                           
                                                                           
                                                                  | Pass  T
otal     Time
Testing benchmarks                                                         
                                                                           
                                                                           
                                                                           
                                                                  |   12   
  12  2m32.7s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ReverseMode{false, false, false, Enzym
eCore.FFIABI, false, false}())                                             
                                                                           
                                                                           
                                                                  |    2   
   2    46.5s
  ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMode{false, EnzymeCore.FFIABI, 
false, false, false}())                                                    
                                                                           
                                                                           
                                                                  |    2   
   2    50.8s
  ADTypes.AutoMooncake()                                                   
                                                                           
                                                                           
                                                                           
                                                                  |    2   
   2    42.6s
  ADTypes.AutoForwardDiff()                                                
                                                                           
                                                                           
                                                                           
                                                                  |    2   
   2     4.1s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoForwardDiff(), sparsity_detector=
SparseConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=Spars
eMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColo
rings.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false))      
                                                                  |    2   
   2     4.5s
  ADTypes.AutoSparse(dense_ad=ADTypes.AutoEnzyme(mode=EnzymeCore.ForwardMod
e{false, EnzymeCore.FFIABI, false, false, false}()), sparsity_detector=Spar
seConnectivityTracer.TracerSparsityDetector(), coloring_algorithm=SparseMat
rixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColoring
s.NaturalOrder}}((SparseMatrixColorings.NaturalOrder(),), false)) |    2   
   2     3.8s
```



|                                                                                                                                                                                                                                **backend** |                                                             **scenario** |       **operator** | **prepared** | **calls** | **samples** | **evals** | **time** | **allocs** | **bytes** | **gc_fraction** | **compile_fraction** |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| ------------------------------------------------------------------------:| ------------------:| ------------:| ---------:| -----------:| ---------:| --------:| ----------:| ---------:| ---------------:| --------------------:|
|                                                                                                                                                                  AutoEnzyme(mode=ReverseMode{false, false, false, FFIABI, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   6.4e+01 |     6.0e+00 |   1.0e+00 |  1.1e-01 |    6.6e+03 |   2.8e+08 |         3.1e-01 |              0.0e+00 |
|                                                                                                                                                                  AutoEnzyme(mode=ReverseMode{false, false, false, FFIABI, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   6.3e+01 |     7.0e+00 |   1.0e+00 |  8.9e-02 |    6.6e+03 |   2.8e+08 |         2.0e-01 |              0.0e+00 |
|                                                                                                                                                                         AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   6.3e+01 |     4.4e+01 |   1.0e+00 |  4.6e-03 |    5.3e+03 |   1.8e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                         AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   6.3e+01 |     6.3e+01 |   1.0e+00 |  4.6e-03 |    5.3e+03 |   1.8e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                             AutoMooncake() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   1.0e+00 |     1.4e+01 |   1.0e+00 |  5.2e-02 |    1.1e+04 |   3.3e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                             AutoMooncake() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   0.0e+00 |     1.3e+01 |   1.0e+00 |  5.2e-02 |    1.1e+04 |   3.3e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                          AutoForwardDiff() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   8.5e+01 |     9.2e+01 |   1.0e+00 |  3.0e-03 |    2.6e+02 |   1.7e+07 |         0.0e+00 |              0.0e+00 |
|                                                                                                                                                                                                                          AutoForwardDiff() | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   8.4e+01 |     1.1e+02 |   1.0e+00 |  3.0e-03 |    2.6e+02 |   1.7e+07 |         0.0e+00 |              0.0e+00 |
|                                                  AutoSparse(dense_ad=AutoForwardDiff(), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   2.0e+00 |     2.2e+04 |   1.0e+00 |  1.5e-05 |    1.5e+01 |   9.6e+04 |         0.0e+00 |              0.0e+00 |
|                                                  AutoSparse(dense_ad=AutoForwardDiff(), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   1.0e+00 |     2.5e+04 |   1.0e+00 |  1.4e-05 |    1.2e+01 |   8.8e+04 |         0.0e+00 |              0.0e+00 |
| AutoSparse(dense_ad=AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} | value_and_jacobian |      1.0e+00 |   2.0e+00 |     1.7e+04 |   1.0e+00 |  1.6e-05 |    2.3e+01 |   9.7e+04 |         0.0e+00 |              0.0e+00 |
| AutoSparse(dense_ad=AutoEnzyme(mode=ForwardMode{false, FFIABI, false, false, false}()), sparsity_detector=TracerSparsityDetector(), coloring_algorithm=GreedyColoringAlgorithm{:direct, 1, Tuple{NaturalOrder}}((NaturalOrder(),), false)) | Scenario{:jacobian,:out} Diffusion : Vector{Float64} -\> Vector{Float64} |           jacobian |      1.0e+00 |   1.0e+00 |     1.9e+04 |   1.0e+00 |  1.1e-05 |    2.0e+01 |   8.9e+04 |         0.0e+00 |              0.0e+00 |

