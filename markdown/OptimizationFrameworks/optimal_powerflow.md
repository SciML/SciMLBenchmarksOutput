---
author: "Chris Rackauckas"
title: "Optimal Powerflow Nonlinear Optimization Benchmark"
---


## Data Load and Setup Code

This is generic setup code usable for all solver setups. Basically removing some unnecessary untyped dictionaries
before getting to the benchmarks.

```julia
PRINT_LEVEL = 0

# This is a soft upper limit to the time of each optimization.
# If times go above this, they will halt early
MAX_CPU_TIME = 100.0

# Maximum number of variables in an optimization problem for the benchmark
# Anything with more variables is rejected and not run
# This is for testing. 100 is a good size for running a test of changes
# Should be set to typemax(Int) to run the whole benchmark
SIZE_LIMIT = 1000
```

```
1000
```



```julia
import PowerModels
import ConcreteStructs
using BenchmarkTools
using DataFrames

ConcreteStructs.@concrete struct DataRepresentation
    data
    ref
    var_lookup
    var_init
    var_lb
    var_ub
    ref_gen_idxs
    lookup_pg
    lookup_qg
    lookup_va
    lookup_vm
    lookup_lij
    lookup_p_lij
    lookup_q_lij
    cost_arrs
    f_bus
    t_bus
    ref_bus_idxs
    ref_buses_idxs
    ref_bus_gens
    ref_bus_arcs
    ref_branch_idxs
    ref_arcs_from
    ref_arcs_to
    p_idxmap
    q_idxmap
    bus_pd
    bus_qd
    bus_gs
    bus_bs
    br_g
    br_b
    br_tr
    br_ti
    br_ttm
    br_g_fr
    br_b_fr
    br_g_to
    br_b_to
end

function load_and_setup_data(file_name)
    data = PowerModels.parse_file(file_name)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    # Some data munging to type-stable forms

    var_lookup = Dict{String,Int}()

    var_init = Float64[]
    var_lb = Float64[]
    var_ub = Float64[]

    var_idx = 1
    for (i,bus) in ref[:bus]
        push!(var_init, 0.0) #va
        push!(var_lb, -Inf)
        push!(var_ub, Inf)
        var_lookup["va_$(i)"] = var_idx
        var_idx += 1

        push!(var_init, 1.0) #vm
        push!(var_lb, bus["vmin"])
        push!(var_ub, bus["vmax"])
        var_lookup["vm_$(i)"] = var_idx
        var_idx += 1
    end

    for (i,gen) in ref[:gen]
        push!(var_init, 0.0) #pg
        push!(var_lb, gen["pmin"])
        push!(var_ub, gen["pmax"])
        var_lookup["pg_$(i)"] = var_idx
        var_idx += 1

        push!(var_init, 0.0) #qg
        push!(var_lb, gen["qmin"])
        push!(var_ub, gen["qmax"])
        var_lookup["qg_$(i)"] = var_idx
        var_idx += 1
    end

    for (l,i,j) in ref[:arcs]
        branch = ref[:branch][l]

        push!(var_init, 0.0) #p
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["p_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1

        push!(var_init, 0.0) #q
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["q_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1
    end

    @assert var_idx == length(var_init)+1

    ref_gen_idxs = [i for i in keys(ref[:gen])]
    lookup_pg = Dict{Int,Int}()
    lookup_qg = Dict{Int,Int}()
    lookup_va = Dict{Int,Int}()
    lookup_vm = Dict{Int,Int}()
    lookup_lij = Tuple{Int,Int,Int}[]
    lookup_p_lij = Int[]
    lookup_q_lij = Int[]
    cost_arrs = Dict{Int,Vector{Float64}}()

    for (i,gen) in ref[:gen]
        lookup_pg[i] = var_lookup["pg_$(i)"]
        lookup_qg[i] = var_lookup["qg_$(i)"]
        cost_arrs[i] = gen["cost"]
    end

    for (i,bus) in ref[:bus]
        lookup_va[i] = var_lookup["va_$(i)"]
        lookup_vm[i] = var_lookup["vm_$(i)"]
    end

    for (l,i,j) in ref[:arcs]
        push!(lookup_lij, (l,i,j))
        push!(lookup_p_lij,var_lookup["p_$(l)_$(i)_$(j)"])
        push!(lookup_q_lij,var_lookup["q_$(l)_$(i)_$(j)"])
    end

    f_bus = Dict{Int,Int}()
    t_bus = Dict{Int,Int}()

    for (l,branch) in ref[:branch]
        f_bus[l] = branch["f_bus"]
        t_bus[l] = branch["t_bus"]
    end

    ref_bus_idxs = [i for i in keys(ref[:bus])]
    ref_buses_idxs = [i for i in keys(ref[:ref_buses])]
    ref_bus_gens = ref[:bus_gens]
    ref_bus_arcs = ref[:bus_arcs]
    ref_branch_idxs = [i for i in keys(ref[:branch])]
    ref_arcs_from = ref[:arcs_from]
    ref_arcs_to = ref[:arcs_to]

    p_idxmap = Dict(lookup_lij[i] => lookup_p_lij[i] for i in 1:length(lookup_lij))
    q_idxmap = Dict(lookup_lij[i] => lookup_q_lij[i] for i in 1:length(lookup_lij))

    bus_pd = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_qd = Dict(i => 0.0 for (i,bus) in ref[:bus])

    bus_gs = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_bs = Dict(i => 0.0 for (i,bus) in ref[:bus])

    for (i,bus) in ref[:bus]
        if length(ref[:bus_loads][i]) > 0
            bus_pd[i] = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])
            bus_qd[i] = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i])
        end

        if length(ref[:bus_shunts][i]) > 0
            bus_gs[i] = sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i])
            bus_bs[i] = sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i])
        end
    end


    br_g = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_tr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ti = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ttm = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_g_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_g_to = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_to = Dict(i => 0.0 for (i,branch) in ref[:branch])

    for (i,branch) in ref[:branch]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)

        br_g[i] = g
        br_b[i] = b

        br_tr[i] = tr
        br_ti[i] = ti
        br_ttm[i] = tr^2 + ti^2

        br_g_fr[i] = branch["g_fr"]
        br_b_fr[i] = branch["b_fr"]
        br_g_to[i] = branch["g_to"]
        br_b_to[i] = branch["b_to"]
    end

    DataRepresentation(
        data,
        ref,
        var_lookup,
        var_init,
        var_lb,
        var_ub,
        ref_gen_idxs,
        lookup_pg,
        lookup_qg,
        lookup_va,
        lookup_vm,
        lookup_lij,
        lookup_p_lij,
        lookup_q_lij,
        cost_arrs,
        f_bus,
        t_bus,
        ref_bus_idxs,
        ref_buses_idxs,
        ref_bus_gens,
        ref_bus_arcs,
        ref_branch_idxs,
        ref_arcs_from,
        ref_arcs_to,
        p_idxmap,
        q_idxmap,
        bus_pd,
        bus_qd,
        bus_gs,
        bus_bs,
        br_g,
        br_b,
        br_tr,
        br_ti,
        br_ttm,
        br_g_fr,
        br_b_fr,
        br_g_to,
        br_b_to)
end

file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
dataset = load_and_setup_data(file_name);
```




## Test Setup

Ensure that all objectives and constraints evaluate to the same value on a random vector on the same dataset

```julia
test_u0 = [0.6292298794022337, 0.30740951571225206, 0.0215258802699263, 0.38457509230779996, 0.9419186480931858, 0.34961116773074874, 0.875763562401991, 0.3203478635827923, 0.6354060958226175, 0.45537545721771266, 0.3120599359696674, 0.2421238802331842, 0.886455177641366, 0.49797378087768696, 0.652913329799645, 0.03590201299300255, 0.5618806749518928, 0.8142146688533769, 0.3973557130434364, 0.27827135011662674, 0.16456134856048643, 0.7465018431665373, 0.4898329811551083, 0.6966035226583556, 0.7419662648518377, 0.8505905798503723, 0.27102126066405097, 0.1988238097281576, 0.09684601934490256, 0.49238142828542797, 0.1366594202307445, 0.6337080281764231, 0.28814906958008235, 0.5404996094640431, 0.015153517398975858, 0.6338449294034381, 0.5165464961007717, 0.572879113636733, 0.9652420600585092, 0.26535868365228543, 0.865686920119479, 0.38426996353892773, 0.007412077949221274, 0.3889835001514599] 
test_obj = 7079.190664351089
test_cons = [0.0215258802699263, -1.0701734802505833, -5.108902216849063, -3.49724505910433, -2.617834191007569, 0.5457423426033834, -0.7150251969424766, -2.473175092089014, -2.071687022809815, -1.5522321037165985, -1.0107399030803794, 3.0047739260369246, 0.2849522377447594, -2.8227966798520674, 3.2236954017592256, 1.0793383525116511, -1.633412293595111, -3.1618224299953224, -0.7775962590542184, 1.7252573527333024, -4.23535583005632, -1.7030832394691608, 1.5810450617647889, -0.33289810365419437, 0.19476447251065077, 1.0688558672739048, 1.563372246165339, 9.915310272572729, 1.4932615291788414, 2.0016715378998793, -1.4038702698147258, -0.8834081057449231, 0.21730536348839036, -7.40879932706212, -1.6000837514115611, 0.8542376821320647, 0.06615508569119477, -0.6077039991323074, 0.6138802155526912, 0.0061762164203837955, -0.3065125522705683, 0.5843454392910835, 0.7251928172073308, 1.2740182727083802, 0.11298343104675009, 0.2518186223833513, 0.4202616621130535, 0.3751697141306502, 0.4019890236200105, 0.5950107614751935, 1.0021074654956683, 0.897077248544158, 0.15136310228960612]
```

```
53-element Vector{Float64}:
  0.0215258802699263
 -1.0701734802505833
 -5.108902216849063
 -3.49724505910433
 -2.617834191007569
  0.5457423426033834
 -0.7150251969424766
 -2.473175092089014
 -2.071687022809815
 -1.5522321037165985
  ⋮
  0.11298343104675009
  0.2518186223833513
  0.4202616621130535
  0.3751697141306502
  0.4019890236200105
  0.5950107614751935
  1.0021074654956683
  0.897077248544158
  0.15136310228960612
```





## Setup and Validations

Now is the setup code for each optimization framework, along with the validation runs on the test case. Any test which fails the validation
case, i.e. has `x_test_res[1] !≈ test_obj` or `x_test_res[2] !≈ test_cons` should be considered invalidated as this means that the model
in that modeling platform does not evaluate to give the same results

### Optimization.jl

Constraint optimization implementation reference: https://github.com/SciML/Optimization.jl/blob/master/lib/OptimizationMOI/test/runtests.jl
Other AD libraries can be considered: https://docs.sciml.ai/dev/modules/Optimization/API/optimization_function/

```julia
import Optimization
import OptimizationMOI
import ModelingToolkit
import Ipopt
import Enzyme
import ReverseDiff

function build_opf_optimization_prob(dataset; adchoice = Optimization.AutoEnzyme())
    (;data,
    ref,
    var_lookup,
    var_init,
    var_lb,
    var_ub,
    ref_gen_idxs,
    lookup_pg,
    lookup_qg,
    lookup_va,
    lookup_vm,
    lookup_lij,
    lookup_p_lij,
    lookup_q_lij,
    cost_arrs,
    f_bus,
    t_bus,
    ref_bus_idxs,
    ref_buses_idxs,
    ref_bus_gens,
    ref_bus_arcs,
    ref_branch_idxs,
    ref_arcs_from,
    ref_arcs_to,
    p_idxmap,
    q_idxmap,
    bus_pd,
    bus_qd,
    bus_gs,
    bus_bs,
    br_g,
    br_b,
    br_tr,
    br_ti,
    br_ttm,
    br_g_fr,
    br_b_fr,
    br_g_to,
    br_b_to) = dataset

    #total_callback_time = 0.0
    function opf_objective(x, param)
        #start = time()
        cost = 0.0
        for i in ref_gen_idxs
            pg = x[lookup_pg[i]]
            _cost_arr = cost_arrs[i]
            cost += _cost_arr[1]*pg^2 + _cost_arr[2]*pg + _cost_arr[3]
        end
        #total_callback_time += time() - start
        return cost
    end

    function opf_constraints(ret, x, param)
        offsetidx = 0

        # va_con
        for (reti,i) in enumerate(ref_buses_idxs)
            ret[reti + offsetidx] = x[lookup_va[i]]
        end

        offsetidx += length(ref_buses_idxs)
        
        #     @constraint(model,
        #         sum(p[a] for a in ref[:bus_arcs][i]) ==
        #         sum(pg[g] for g in ref_bus_gens[i]) -
        #         sum(load["pd"] for load in bus_loads) -
        #         sum(shunt["gs"] for shunt in bus_shunts)*x[lookup_vm[i]]^2
        #     )

        # power_balance_p_con
        for (reti,i) in enumerate(ref_bus_idxs)
            ret[reti + offsetidx] = sum(x[lookup_pg[j]] for j in ref_bus_gens[i]; init=0.0) -
            bus_pd[i] -
            bus_gs[i]*x[lookup_vm[i]]^2 -
            sum(x[p_idxmap[a]] for a in ref_bus_arcs[i])
        end

        offsetidx += length(ref_bus_idxs)

        #     @constraint(model,
        #         sum(q[a] for a in ref[:bus_arcs][i]) ==
        #         sum(x[lookup_qg[g]] for g in ref_bus_gens[i]) -
        #         sum(load["qd"] for load in bus_loads) +
        #         sum(shunt["bs"] for shunt in bus_shunts)*x[lookup_vm[i]]^2
        #     )
        # power_balance_q_con
        for (reti,i) in enumerate(ref_bus_idxs)
        ret[reti + offsetidx] = sum(x[lookup_qg[j]] for j in ref_bus_gens[i]; init=0.0) -
        bus_qd[i] +
        bus_bs[i]*x[lookup_vm[i]]^2 -
        sum(x[q_idxmap[a]] for a in ref_bus_arcs[i])
        end

        offsetidx += length(ref_bus_idxs)

        # @NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        # power_flow_p_from_con =
        for (reti,(l,i,j)) in enumerate(ref_arcs_from)
        ret[reti + offsetidx] = (br_g[l]+br_g_fr[l])/br_ttm[l]*x[lookup_vm[f_bus[l]]]^2 +
        (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[f_bus[l]]]*x[lookup_vm[t_bus[l]]]*cos(x[lookup_va[f_bus[l]]]-x[lookup_va[t_bus[l]]])) +
        (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[f_bus[l]]]*x[lookup_vm[t_bus[l]]]*sin(x[lookup_va[f_bus[l]]]-x[lookup_va[t_bus[l]]])) -
        x[p_idxmap[(l,i,j)]]
        end

        offsetidx += length(ref_arcs_from)

        # @NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        # power_flow_p_to_con
        for (reti,(l,i,j)) in enumerate(ref_arcs_to)
        ret[reti + offsetidx] = (br_g[l]+br_g_to[l])*x[lookup_vm[t_bus[l]]]^2 +
        (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[t_bus[l]]]*x[lookup_vm[f_bus[l]]]*cos(x[lookup_va[t_bus[l]]]-x[lookup_va[f_bus[l]]])) +
        (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[t_bus[l]]]*x[lookup_vm[f_bus[l]]]*sin(x[lookup_va[t_bus[l]]]-x[lookup_va[f_bus[l]]])) -
        x[p_idxmap[(l,i,j)]]
        end

        offsetidx += length(ref_arcs_to)

        # @NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        # power_flow_q_from_con
        for (reti,(l,i,j)) in enumerate(ref_arcs_from)
        ret[reti + offsetidx] = -(br_b[l]+br_b_fr[l])/br_ttm[l]*x[lookup_vm[f_bus[l]]]^2 -
        (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[f_bus[l]]]*x[lookup_vm[t_bus[l]]]*cos(x[lookup_va[f_bus[l]]]-x[lookup_va[t_bus[l]]])) +
        (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[f_bus[l]]]*x[lookup_vm[t_bus[l]]]*sin(x[lookup_va[f_bus[l]]]-x[lookup_va[t_bus[l]]])) -
        x[q_idxmap[(l,i,j)]]
        end

        offsetidx += length(ref_arcs_from)

        # @NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        # power_flow_q_to_con
        for (reti,(l,i,j)) in enumerate(ref_arcs_to)
        ret[reti + offsetidx] = -(br_b[l]+br_b_to[l])*x[lookup_vm[t_bus[l]]]^2 -
        (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[t_bus[l]]]*x[lookup_vm[f_bus[l]]]*cos(x[lookup_va[t_bus[l]]]-x[lookup_va[f_bus[l]]])) +
        (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(x[lookup_vm[t_bus[l]]]*x[lookup_vm[f_bus[l]]]*sin(x[lookup_va[t_bus[l]]]-x[lookup_va[f_bus[l]]])) -
        x[q_idxmap[(l,i,j)]]
        end

        offsetidx += length(ref_arcs_to)

        # @constraint(model, va_fr - va_to <= branch["angmax"])
        # @constraint(model, va_fr - va_to >= branch["angmin"])
        # power_flow_vad_con
        for (reti,(l,i,j)) in enumerate(ref_arcs_from)
        ret[reti + offsetidx] = x[lookup_va[f_bus[l]]] - x[lookup_va[t_bus[l]]]
        end

        offsetidx += length(ref_arcs_from)

        # @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        # power_flow_mva_from_con
        for (reti,(l,i,j)) in enumerate(ref_arcs_from)
        ret[reti + offsetidx] = x[p_idxmap[(l,i,j)]]^2 + x[q_idxmap[(l,i,j)]]^2
        end

        offsetidx += length(ref_arcs_from)

        # @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
        # power_flow_mva_to_con 
        for (reti,(l,i,j)) in enumerate(ref_arcs_to)
        ret[reti + offsetidx] = x[p_idxmap[(l,i,j)]]^2 + x[q_idxmap[(l,i,j)]]^2
        end

        offsetidx += length(ref_arcs_to)

        @assert offsetidx == length(ret)
        nothing
    end

    con_lbs = Float64[]
    con_ubs = Float64[]

    #@constraint(model, va[i] == 0)
    for (i,bus) in ref[:ref_buses]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_balance_p_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_balance_q_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_p_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_p_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_vad_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, branch["angmin"])
        push!(con_ubs, branch["angmax"])
    end

    #power_flow_mva_from_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    #power_flow_mva_to_con
    for (l,i,j) in ref[:arcs_to]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    model_variables = length(var_init)
    ret = Array{Float64}(undef, length(con_lbs))
    model_constraints = length(con_lbs)

    optf = Optimization.OptimizationFunction(opf_objective, adchoice; cons=opf_constraints)
    prob = Optimization.OptimizationProblem(optf, var_init; lb=var_lb, ub=var_ub, lcons=con_lbs, ucons=con_ubs)
end

function solve_opf_optimization(dataset; adchoice = Optimization.AutoSparseReverseDiff(true))
    model_build_time = @elapsed prob = build_opf_optimization_prob(dataset; adchoice)

    # Correctness tests
    ret = zeros(length(prob.lcons))
    prob.f.cons(ret, prob.u0, nothing)
    @allocated prob.f(prob.u0, nothing) == 0
    @allocated prob.f.cons(ret, prob.u0, nothing) == 0

    solve_time_with_compilation = @elapsed sol = Optimization.solve(prob, Ipopt.Optimizer(), print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME)
    cost = sol.minimum
    feasible = (sol.retcode == Optimization.SciMLBase.ReturnCode.Success)
    #println(sol.u) # solution vector

    solve_time_without_compilation = @elapsed sol = Optimization.solve(prob, Ipopt.Optimizer(), print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME)
    
    return (prob,sol),Dict(
        "case" => file_name,
        "variables" => length(prob.u0),
        "constraints" => length(prob.lcons),
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_optimization_prob(dataset, test_u0)
    prob = build_opf_optimization_prob(dataset)
    ret = zeros(length(prob.lcons))
    prob.f.cons(ret, test_u0, nothing)
    obj = prob.f(test_u0, nothing)
    obj, ret
end
```

```
test_optimization_prob (generic function with 1 method)
```



```julia
optimization_test_res = test_optimization_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, -1.0701734802505833, -5.1089022168
49063, -3.49724505910433, -2.617834191007569, 0.5457423426033834, -0.715025
1969424766, -2.473175092089014, -2.071687022809815, -1.5522321037165985  … 
 1.2740182727083802, 0.11298343104675009, 0.2518186223833513, 0.42026166211
30535, 0.3751697141306502, 0.4019890236200105, 0.5950107614751935, 1.002107
4654956683, 0.897077248544158, 0.15136310228960612])
```



```julia
@assert optimization_test_res[1] == test_obj
```


```julia
@assert optimization_test_res[2] == test_cons
```




## ModelingToolkit.jl

Showcases symbolic interface to Optimization.jl, through ModelingToolkit.jl. 
Equivalent to using `AutoModelingToolkit` as the AD backend in `OptimizationFunction`, the values for the objectives don't match exactly 
because of structural simplification.

```julia

import PowerModels
import Ipopt
using ModelingToolkit, Optimization, OptimizationMOI
import ModelingToolkit: ≲, unknowns

function build_opf_mtk_prob(dataset)
    (;data, ref) = dataset

    vars = Num[]
    lb = Float64[]
    ub = Float64[]

    ModelingToolkit.@variables va[1:maximum(keys(ref[:bus]))]
    for i in keys(ref[:bus])
        push!(lb, -Inf)
        push!(ub, Inf)
    end
    ModelingToolkit.@variables vm[1:maximum(keys(ref[:bus]))]
    for i in keys(ref[:bus])
        push!(lb, ref[:bus][i]["vmin"])
        push!(ub, ref[:bus][i]["vmax"])
    end
    vars = vcat(vars, [va[i] for i in keys(ref[:bus])], [vm[i] for i in keys(ref[:bus])])
    ModelingToolkit.@variables pg[1:maximum(keys(ref[:gen]))]
    for i in keys(ref[:gen])
        push!(lb, ref[:gen][i]["pmin"])
        push!(ub, ref[:gen][i]["pmax"])
    end
    ModelingToolkit.@variables qg[1:maximum(keys(ref[:gen]))]
    for i in keys(ref[:gen])
        push!(lb, ref[:gen][i]["qmin"])
        push!(ub, ref[:gen][i]["qmax"])
    end
    vars = vcat(vars, [pg[i] for i in keys(ref[:gen])], [qg[i] for i in keys(ref[:gen])])
    i_inds, j_inds, l_inds = maximum(first.(ref[:arcs])), maximum(getindex.(ref[:arcs], Ref(2))), maximum(last.(ref[:arcs]))
    ModelingToolkit.@variables p[1:i_inds, 1:j_inds, 1:l_inds]
    ModelingToolkit.@variables q[1:i_inds, 1:j_inds, 1:l_inds]

    for (l, i, j) in ref[:arcs]
        push!(vars, p[l, i, j])
        push!(lb, -ref[:branch][l]["rate_a"])
        push!(ub, ref[:branch][l]["rate_a"])
    end

    for (l, i, j) in ref[:arcs]
        push!(vars, q[l, i, j])
        push!(lb, -ref[:branch][l]["rate_a"])
        push!(ub, ref[:branch][l]["rate_a"])
    end

    loss = sum(gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for (i, gen) in ref[:gen])

    cons = Array{Union{ModelingToolkit.Equation,ModelingToolkit.Inequality}}([])
    for (i, bus) in ref[:ref_buses]
        push!(cons, va[i] ~ 0)
    end

    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        push!(cons,
            sum(p[a...] for a in ref[:bus_arcs][i]) ~
                (sum(pg[g] for g in ref[:bus_gens][i]; init = 0.0)) -
                (sum(load["pd"] for load in bus_loads; init = 0.0)) -
             sum(shunt["gs"] for shunt in bus_shunts; init = 0.0)*vm[i]^2
        )

        push!(cons,
            sum(q[a...] for a in ref[:bus_arcs][i]) ~
                (sum(qg[g] for g in ref[:bus_gens][i]; init = 0.0)) -
                (sum(load["qd"] for load in bus_loads; init = 0.0))
             + sum(shunt["bs"] for shunt in bus_shunts; init = 0.0)*vm[i]^2
        )
    end

    # Branch power flow physics and limit constraints
    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx...]
        q_fr = q[f_idx...]
        p_to = p[t_idx...]
        q_to = q[t_idx...]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]

        # From side of the branch flow
        push!(cons, p_fr ~ (g + g_fr) / ttm * vm_fr^2 + (-g * tr + b * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) + (-b * tr - g * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to)))
        push!(cons, q_fr ~ -(b + b_fr) / ttm * vm_fr^2 - (-b * tr - g * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) + (-g * tr + b * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to)))

        # To side of the branch flow
        push!(cons, p_to ~ (g + g_to) * vm_to^2 + (-g * tr - b * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) + (-b * tr + g * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr)))
        push!(cons, q_to ~ -(b + b_to) * vm_to^2 - (-b * tr + g * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) + (-g * tr - b * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr)))

        # Voltage angle difference limit
        push!(cons, va_fr - va_to ≲ branch["angmax"])
        push!(cons, branch["angmin"] ≲ va_fr - va_to)

        # Apparent power limit, from side and to side
        push!(cons, p_fr^2 + q_fr^2 ≲ branch["rate_a"]^2)
        push!(cons, p_to^2 + q_to^2 ≲ branch["rate_a"]^2)
    end

    optsys = ModelingToolkit.OptimizationSystem(loss, vars, [], constraints=cons, name=:rosetta)
    optsys = ModelingToolkit.structural_simplify(optsys)
    u0map = [Num(k) => 0.0 for k in collect(unknowns(optsys))]
    ks = collect(Num.(unknowns(optsys)))
    for key in keys(ref[:bus])
        ind = findfirst(x -> isequal(x, vm[key]), ks)
        if ind !== nothing
            u0map[ind] = vm[key] => 1.0
        end
    end

    inds = Int[]
    for k in collect(unknowns(optsys))
        push!(inds, findall(x -> isequal(x, k), vars)[1])
    end
    prob = Optimization.OptimizationProblem(optsys, Dict(u0map), lb = lb[inds], ub = ub[inds], grad=true, hess=true, cons_j=true, cons_h=true, cons_sparse=true, sparse=true)
end

function solve_opf_mtk(dataset)
    model_build_time = @elapsed prob = build_opf_mtk_prob(dataset)

    # @assert prob.f(prob.u0, nothing) == 0.0 #MTK with simplification doesn't evaluate the same
    ret = zeros(length(prob.lcons))
    prob.f.cons(ret, prob.u0, nothing, )
    @allocated prob.f(prob.u0, nothing) == 0
    @allocated prob.f.cons(ret, prob.u0, nothing) == 0


    solve_time_with_compilation = @elapsed sol = OptimizationMOI.solve(prob, Ipopt.Optimizer())
    solve_time_without_compilation = @elapsed sol = OptimizationMOI.solve(prob, Ipopt.Optimizer())

    cost = sol.minimum
    feasible = (sol.retcode == Optimization.SciMLBase.ReturnCode.Success)

   return (prob,sol),Dict(
        "case" => file_name,
        "variables" => length(prob.u0),
        "constraints" => length(prob.lcons),
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_mtk_prob(dataset, test_u0)
    prob = build_opf_mtk_prob(dataset)
    ret = zeros(length(prob.lcons))
    prob.f.cons(ret, test_u0, nothing)
    obj = prob.f(test_u0, nothing)
    obj, ret
end
```

```
test_mtk_prob (generic function with 1 method)
```



```julia
mtk_test_res = test_optimization_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, -1.0701734802505833, -5.1089022168
49063, -3.49724505910433, -2.617834191007569, 0.5457423426033834, -0.715025
1969424766, -2.473175092089014, -2.071687022809815, -1.5522321037165985  … 
 1.2740182727083802, 0.11298343104675009, 0.2518186223833513, 0.42026166211
30535, 0.3751697141306502, 0.4019890236200105, 0.5950107614751935, 1.002107
4654956683, 0.897077248544158, 0.15136310228960612])
```



```julia
@assert mtk_test_res[1] == test_obj
```


```julia
@assert mtk_test_res[2] == test_cons
```





## JuMP.jl

Implementation reference: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/ac-opf.jl
Only the built-in AD library is supported

```julia
import PowerModels
import Ipopt
import JuMP

function build_opf_jump_prob(dataset)
    (;data, ref) = dataset
    constraints = Any[]
    model = JuMP.Model(Ipopt.Optimizer)

    vars = [JuMP.@variable(model, va[i in keys(ref[:bus])]),
            JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0),
            JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"]),
            JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"]),
            JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"]),
            JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])]

    JuMP.@objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i,bus) in ref[:ref_buses]
        push!(constraints,JuMP.@constraint(model, va[i] == 0))
    end

    for (i,bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        push!(constraints,JuMP.@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        ))

        push!(constraints,JuMP.@constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        ))
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]

        # From side of the branch flow
        push!(constraints,JuMP.@NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) ))
        push!(constraints,JuMP.@NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) ))

        # To side of the branch flow
        push!(constraints,JuMP.@NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) ))
        push!(constraints,JuMP.@NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) ))

        # Voltage angle difference limit
        push!(constraints,JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"]))

        # Apparent power limit, from side and to side
        push!(constraints,JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2))
        push!(constraints,JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2))
    end

    model_variables = JuMP.num_variables(model)

    # for consistency with other solvers, skip the variable bounds in the constraint count
    non_nl_constraints = sum(JuMP.num_constraints(model, ft, st) for (ft, st) in JuMP.list_of_constraint_types(model) if ft != JuMP.VariableRef)
    model_constraints = JuMP.num_nonlinear_constraints(model) + non_nl_constraints

    model, vars, constraints
end

function solve_opf_jump(dataset)
    model_build_time = @elapsed model = build_opf_jump_prob(dataset)[1]
    JuMP.set_attribute(model, "max_cpu_time", MAX_CPU_TIME)
    JuMP.set_attribute(model, "print_level", PRINT_LEVEL)

    solve_time_with_compilation = @elapsed JuMP.optimize!(model)
    solve_time_without_compilation = @elapsed JuMP.optimize!(model)

    cost = JuMP.objective_value(model)
    feasible = (JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED)

    nlp_block = JuMP.MOI.get(model, JuMP.MOI.NLPBlock())
    total_callback_time =
        nlp_block.evaluator.eval_objective_timer +
        nlp_block.evaluator.eval_objective_gradient_timer +
        nlp_block.evaluator.eval_constraint_timer +
        nlp_block.evaluator.eval_constraint_jacobian_timer +
        nlp_block.evaluator.eval_hessian_lagrangian_timer
    model_variables = JuMP.num_variables(model)
    non_nl_constraints = sum(JuMP.num_constraints(model, ft, st) for (ft, st) in JuMP.list_of_constraint_types(model) if ft != JuMP.VariableRef)
    model_constraints = JuMP.num_nonlinear_constraints(model) + non_nl_constraints
    
    return model, Dict(
        "case" => file_name,
        "variables" => model_variables,
        "constraints" => model_constraints,
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_jump_prob(dataset, test_u0)
    model, vars, constraints  = build_opf_jump_prob(dataset)
    (;
    lookup_pg,
    lookup_qg,
    lookup_va,
    lookup_vm,
    lookup_lij,
    lookup_p_lij,
    lookup_q_lij) = dataset
    f = JuMP.objective_function(model)

    flatvars = reduce(vcat,[reduce(vcat,vars[i]) for i in 1:length(vars)])
    point = Dict()
    for v in flatvars
        varname, varint = split(JuMP.name(v), "[")
        idx = if varint[1] == '('
            varint = (parse(Int, varint[2]), parse(Int, varint[5]), parse(Int, varint[8]))
            if varname == "p"
                lookup_p_lij[findfirst(x->x==varint,lookup_lij)]
            elseif varname == "q"
                lookup_q_lij[findfirst(x->x==varint,lookup_lij)]
            else
                error("Invalid $varname, $varint")
            end
        else
            varint = parse(Int, varint[1:end-1])
            if varname == "va"
                lookup_va[varint]
            elseif varname == "pg"
                lookup_pg[varint]
            elseif varname == "qg"
                lookup_qg[varint]
            elseif varname == "vm"
                lookup_vm[varint]
            else
                error("Invalid $varname, $varint")
            end
        end
        point[v] = test_u0[idx]
    end
    obj = JuMP.value(x->point[x], f)

    # The JuMP assertion error is because JuMP and optimization.jl build different problems. JuMP builds f(x) == a and optimization.jl builds f(x) - a == 0
    # Workaround this for consistent evaluation
    # It's not a general purpose approach because only some of the Optimization.jl constraints are written as f(x) - a = 0 . 
    # Others are written as f(x) <= a, like the p_fr^2 + q_fr^2 <= branch["rate_a"]^2 constraints

    primal_value(set::JuMP.MOI.EqualTo) = JuMP.MOI.constant(set)
    primal_value(set) = 0.0
    function primal_value(f, constraint)
        object = JuMP.constraint_object(constraint)
        fx = JuMP.value(f, object.func)
        return fx - primal_value(object.set)
    end
    function primal_value(f, constraint::JuMP.NonlinearConstraintRef)
        return JuMP.value(f, constraint)
    end
    obj = JuMP.value(x->point[x], f)
    cons = [primal_value(x->point[x], c) for c in constraints]
    obj, cons
end
```

```
test_jump_prob (generic function with 1 method)
```



```julia
jump_test_res = test_jump_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, 1.0701734802505833, 0.715025196942
4766, 5.108902216849064, 2.473175092089014, 3.49724505910433, 2.07168702280
9815, 2.6178341910075695, 1.5522321037165985, -0.5457423426033834  …  0.006
1762164203837955, 0.2518186223833513, 0.897077248544158, 1.633412293595111,
 -1.4932615291788414, -1.5810450617647889, 1.6000837514115611, -0.306512552
2705683, 0.4202616621130535, 0.15136310228960612])
```



```julia
@assert jump_test_res[1] ≈ test_obj
```


```julia
@assert sort(abs.(jump_test_res[2])) ≈ sort(abs.(test_cons))
```




## NLPModels.jl

Implementation reference: https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/tutorial/
Other AD libraries can be considered: https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/

```julia
import ADNLPModels
import NLPModelsIpopt

function build_opf_nlpmodels_prob(dataset)
    (;data, ref) = dataset

    bus_pd = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_qd = Dict(i => 0.0 for (i,bus) in ref[:bus])

    bus_gs = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_bs = Dict(i => 0.0 for (i,bus) in ref[:bus])

    for (i,bus) in ref[:bus]
        if length(ref[:bus_loads][i]) > 0
            bus_pd[i] = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])
            bus_qd[i] = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i])
        end

        if length(ref[:bus_shunts][i]) > 0
            bus_gs[i] = sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i])
            bus_bs[i] = sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i])
        end
    end


    br_g = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_tr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ti = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ttm = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_g_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_g_to = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_to = Dict(i => 0.0 for (i,branch) in ref[:branch])

    for (i,branch) in ref[:branch]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)

        br_g[i] = g
        br_b[i] = b

        br_tr[i] = tr
        br_ti[i] = ti
        br_ttm[i] = tr^2 + ti^2

        br_g_fr[i] = branch["g_fr"]
        br_b_fr[i] = branch["b_fr"]
        br_g_to[i] = branch["g_to"]
        br_b_to[i] = branch["b_to"]
    end

    var_lookup = Dict{String,Int}()

    var_init = Float64[]
    var_lb = Float64[]
    var_ub = Float64[]

    var_idx = 1
    for (i,bus) in ref[:bus]
        push!(var_init, 0.0) #va
        push!(var_lb, -Inf)
        push!(var_ub, Inf)
        var_lookup["va_$(i)"] = var_idx
        var_idx += 1

        push!(var_init, 1.0) #vm
        push!(var_lb, bus["vmin"])
        push!(var_ub, bus["vmax"])
        var_lookup["vm_$(i)"] = var_idx
        var_idx += 1
    end

    for (i,gen) in ref[:gen]
        push!(var_init, 0.0) #pg
        push!(var_lb, gen["pmin"])
        push!(var_ub, gen["pmax"])
        var_lookup["pg_$(i)"] = var_idx
        var_idx += 1

        push!(var_init, 0.0) #qg
        push!(var_lb, gen["qmin"])
        push!(var_ub, gen["qmax"])
        var_lookup["qg_$(i)"] = var_idx
        var_idx += 1
    end

    for (l,i,j) in ref[:arcs]
        branch = ref[:branch][l]

        push!(var_init, 0.0) #p
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["p_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1

        push!(var_init, 0.0) #q
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["q_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1
    end

    @assert var_idx == length(var_init)+1
    #total_callback_time = 0.0
    function opf_objective(x)
        #start = time()
        cost = 0.0
        for (i,gen) in ref[:gen]
            pg = x[var_lookup["pg_$(i)"]]
            cost += gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
        end
        #total_callback_time += time() - start
        return cost
    end

    function opf_constraints!(cx, x)
        #start = time()

        va = Dict(i => x[var_lookup["va_$(i)"]] for (i,bus) in ref[:bus])
        vm = Dict(i => x[var_lookup["vm_$(i)"]] for (i,bus) in ref[:bus])

        pg = Dict(i => x[var_lookup["pg_$(i)"]] for (i,gen) in ref[:gen])
        qg = Dict(i => x[var_lookup["qg_$(i)"]] for (i,gen) in ref[:gen])

        p = Dict((l,i,j) => x[var_lookup["p_$(l)_$(i)_$(j)"]] for (l,i,j) in ref[:arcs])
        q = Dict((l,i,j) => x[var_lookup["q_$(l)_$(i)_$(j)"]] for (l,i,j) in ref[:arcs])

        vm_fr = Dict(l => vm[branch["f_bus"]] for (l,branch) in ref[:branch])
        vm_to = Dict(l => vm[branch["t_bus"]] for (l,branch) in ref[:branch])
        va_fr = Dict(l => va[branch["f_bus"]] for (l,branch) in ref[:branch])
        va_to = Dict(l => va[branch["t_bus"]] for (l,branch) in ref[:branch])


        # va_con = [va[i] for (i,bus) in ref[:ref_buses]]
        k = 0
        for (i,bus) in ref[:ref_buses]
            k += 1
            cx[k] = va[i]
        end

        #     @constraint(model,
        #         sum(p[a] for a in ref[:bus_arcs][i]) ==
        #         sum(pg[g] for g in ref[:bus_gens][i]) -
        #         sum(load["pd"] for load in bus_loads) -
        #         sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        #     )
        for (i, bus) in ref[:bus]
            k += 1
            cx[k] = sum(pg[j] for j in ref[:bus_gens][i]; init=0.0) - bus_pd[i] - bus_gs[i]*vm[i]^2 - sum(p[a] for a in ref[:bus_arcs][i])
        end

        #     @constraint(model,
        #         sum(q[a] for a in ref[:bus_arcs][i]) ==
        #         sum(qg[g] for g in ref[:bus_gens][i]) -
        #         sum(load["qd"] for load in bus_loads) +
        #         sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        #     )
        for (i, bus) in ref[:bus]
            k += 1
            cx[k] = sum(qg[j] for j in ref[:bus_gens][i]; init=0.0) - bus_qd[i] + bus_bs[i]*vm[i]^2 - sum(q[a] for a in ref[:bus_arcs][i])
        end


        # @NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        for (l,i,j) in ref[:arcs_from]
            k += 1
            cx[k] = (br_g[l]+br_g_fr[l])/br_ttm[l]*vm_fr[l]^2 +
            (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*cos(va_fr[l]-va_to[l])) +
            (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*sin(va_fr[l]-va_to[l])) -
            p[(l,i,j)]
        end

        # @NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        for (l,i,j) in ref[:arcs_to]
            k += 1
            cx[k] = (br_g[l]+br_g_to[l])*vm_to[l]^2 +
            (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*cos(va_to[l]-va_fr[l])) +
            (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*sin(va_to[l]-va_fr[l])) -
            p[(l,i,j)]
        end

        # @NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        for (l,i,j) in ref[:arcs_from]
            k += 1
            cx[k] = -(br_b[l]+br_b_fr[l])/br_ttm[l]*vm_fr[l]^2 -
            (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*cos(va_fr[l]-va_to[l])) +
            (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*sin(va_fr[l]-va_to[l])) -
            q[(l,i,j)]
        end

        # @NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        for (l,i,j) in ref[:arcs_to]
            k += 1
            cx[k] = -(br_b[l]+br_b_to[l])*vm_to[l]^2 -
            (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*cos(va_to[l]-va_fr[l])) +
            (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*sin(va_to[l]-va_fr[l])) -
            q[(l,i,j)]
        end

        # @constraint(model, va_fr - va_to <= branch["angmax"])
        # @constraint(model, va_fr - va_to >= branch["angmin"])
        for (l,i,j) in ref[:arcs_from]
            k += 1
            cx[k] = va_fr[l] - va_to[l]
        end

        # @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        for (l,i,j) in ref[:arcs_from]
            k += 1
            cx[k] = p[(l,i,j)]^2 + q[(l,i,j)]^2
        end

        # @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
        for (l,i,j) in ref[:arcs_to]
            k += 1
            cx[k] = p[(l,i,j)]^2 + q[(l,i,j)]^2
        end

        #total_callback_time += time() - start
        return cx
    end

    con_lbs = Float64[]
    con_ubs = Float64[]

    #@constraint(model, va[i] == 0)
    for (i,bus) in ref[:ref_buses]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_balance_p_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_balance_q_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_p_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_p_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_vad_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, branch["angmin"])
        push!(con_ubs, branch["angmax"])
    end

    #power_flow_mva_from_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    #power_flow_mva_to_con
    for (l,i,j) in ref[:arcs_to]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    model_variables = length(var_init)
    model_constraints = length(opf_constraints!(similar(con_lbs), var_init))
    #=
    backend = ADNLPModels.ADModelBackend(model_variables, opf_objective, model_constraints, opf_constraints!;
                   gradient_backend = ADNLPModels.ReverseDiffADGradient,
                   hprod_backend = ADNLPModels.SDTForwardDiffADHvprod,
                   jprod_backend = ADNLPModels.ForwardDiffADJprod, 
                   jtprod_backend = ADNLPModels.ReverseDiffADJtprod, 
                   jacobian_backend = ADNLPModels.ForwardDiffADJacobian, # SDTSparseADJacobian, 
                   hessian_backend = ADNLPModels.ForwardDiffADHessian, # SparseADJacobian, 
                   ghjvprod_backend = ADNLPModels.ForwardDiffADGHjvprod, 
                   hprod_residual_backend = ADNLPModels.ReverseDiffADHvprod, 
                   jprod_residual_backend = ADNLPModels.ForwardDiffADJprod, 
                   jtprod_residual_backend = ADNLPModels.ReverseDiffADJtprod, 
                   jacobian_residual_backend = ADNLPModels.ForwardDiffADHessian, # SparseADJacobian, 
                   hessian_residual_backend = ADNLPModels.ForwardDiffADHessian
                   )
    =#
    nlp = ADNLPModels.ADNLPModel!(opf_objective, var_init, var_lb, var_ub, opf_constraints!, con_lbs, con_ubs, backend = :optimized)
end

function solve_opf_nlpmodels(dataset)
    model_build_time = @elapsed nlp = build_opf_nlpmodels_prob(dataset)
    solve_time_with_compilation = @elapsed output = NLPModelsIpopt.ipopt(nlp, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME)
    solve_time_without_compilation = @elapsed output = NLPModelsIpopt.ipopt(nlp, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME)
    cost = output.objective
    feasible = (output.primal_feas <= 1e-6)  

    model_variables = nlp.meta.nvar
    model_constraints = nlp.meta.ncon

    return (nlp, output), Dict(
        "case" => file_name,
        "variables" => model_variables,
        "constraints" => model_constraints,
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_nlpmodels_prob(dataset, test_u0)
    nlp = build_opf_nlpmodels_prob(dataset)
    ret = zeros(nlp.meta.ncon)
    nlp.c!(ret, test_u0)
    obj = nlp.f(test_u0)
    obj, ret
end
```

```
test_nlpmodels_prob (generic function with 1 method)
```



```julia
nlpmodels_test_res = test_nlpmodels_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, -1.0701734802505833, -5.1089022168
49063, -3.49724505910433, -2.617834191007569, 0.5457423426033834, -0.715025
1969424766, -2.473175092089014, -2.071687022809815, -1.5522321037165985  … 
 1.2740182727083802, 0.11298343104675009, 0.2518186223833513, 0.42026166211
30535, 0.3751697141306502, 0.4019890236200105, 0.5950107614751935, 1.002107
4654956683, 0.897077248544158, 0.15136310228960612])
```



```julia
@assert nlpmodels_test_res[1] == test_obj
```


```julia
@assert nlpmodels_test_res[2] == test_cons
```




## Nonconvex

Implementation reference: https://julianonconvex.github.io/Nonconvex.jl/stable/problem/
Currently does not converge due to an upstream issue with the AD backend Zygote: https://github.com/JuliaNonconvex/Nonconvex.jl/issues/130

```julia
import Nonconvex
Nonconvex.@load Ipopt

function build_opf_nonconvex_prob(dataset)
    (;data, ref) = dataset
    time_model_start = time()

    model = Nonconvex.DictModel()

    bus_pd = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_qd = Dict(i => 0.0 for (i,bus) in ref[:bus])

    bus_gs = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_bs = Dict(i => 0.0 for (i,bus) in ref[:bus])

    for (i,bus) in ref[:bus]
        if length(ref[:bus_loads][i]) > 0
            bus_pd[i] = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])
            bus_qd[i] = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i])
        end

        if length(ref[:bus_shunts][i]) > 0
            bus_gs[i] = sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i])
            bus_bs[i] = sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i])
        end
    end


    br_g = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_tr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ti = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ttm = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_g_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_g_to = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_to = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_rate_a = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_angmin = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_angmax = Dict(i => 0.0 for (i,branch) in ref[:branch])

    for (i,branch) in ref[:branch]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)

        br_g[i] = g
        br_b[i] = b

        br_tr[i] = tr
        br_ti[i] = ti
        br_ttm[i] = tr^2 + ti^2

        br_g_fr[i] = branch["g_fr"]
        br_b_fr[i] = branch["b_fr"]
        br_g_to[i] = branch["g_to"]
        br_b_to[i] = branch["b_to"]

        br_rate_a[i] = branch["rate_a"]
        br_angmin[i] = branch["angmin"]
        br_angmax[i] = branch["angmax"]
    end


    for (i,bus) in ref[:bus]
        addvar!(model, "va_$(i)", -Inf, Inf, init=0.0) #va
        addvar!(model, "vm_$(i)", bus["vmin"], bus["vmax"], init=1.0) #vm
    end

    for (i,gen) in ref[:gen]
        addvar!(model, "pg_$(i)", gen["pmin"], gen["pmax"], init=0.0) #pg
        addvar!(model, "qg_$(i)", gen["qmin"], gen["qmax"], init=0.0) #qg
    end

    for (l,i,j) in ref[:arcs]
        branch = ref[:branch][l]
        addvar!(model, "p_$(l)_$(i)_$(j)", -branch["rate_a"], branch["rate_a"], init=0.0) #p
        addvar!(model, "q_$(l)_$(i)_$(j)", -branch["rate_a"], branch["rate_a"], init=0.0) #q
    end


    # JuMP.@objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))
    function opf_objective(x::OrderedDict)
        cost = 0.0
        for (i,gen) in ref[:gen]
            pg = x["pg_$(i)"]
            cost += gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
        end
        return cost
    end
    Nonconvex.set_objective!(model, opf_objective)

    # JuMP.@constraint(model, va[i] == 0)
    function const_ref_bus(x::OrderedDict, i)
        return x["va_$(i)"]
    end
    for (i,bus) in ref[:ref_buses]
        add_eq_constraint!(model, x -> const_ref_bus(x,i))
    end

    # @constraint(model,
    #     sum(p[a] for a in ref[:bus_arcs][i]) ==
    #     sum(pg[g] for g in ref[:bus_gens][i]) -
    #     sum(load["pd"] for load in bus_loads) -
    #     sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
    # )
    function const_power_balance_p(x::OrderedDict, b)
        balance = - bus_pd[b] - bus_gs[b]*x["vm_$(b)"]^2
        for (l,i,j) in ref[:bus_arcs][b]
            balance -= x["p_$(l)_$(i)_$(j)"]
        end
        for j in ref[:bus_gens][b]
            balance += x["pg_$(j)"]
        end
        return balance
    end

    # @constraint(model,
    #     sum(q[a] for a in ref[:bus_arcs][i]) ==
    #     sum(qg[g] for g in ref[:bus_gens][i]) -
    #     sum(load["qd"] for load in bus_loads) +
    #     sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
    # )
    function const_power_balance_q(x::OrderedDict, b)
        balance = - bus_qd[b] + bus_bs[b]*x["vm_$(b)"]^2
        for (l,i,j) in ref[:bus_arcs][b]
            balance -= x["q_$(l)_$(i)_$(j)"]
        end
        for j in ref[:bus_gens][b]
            balance += x["qg_$(j)"]
        end
        return balance
    end

    for (i,bus) in ref[:bus]
        add_eq_constraint!(model, x -> const_power_balance_p(x,i))
        add_eq_constraint!(model, x -> const_power_balance_q(x,i))
    end


    # @NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
    function const_flow_p_from(x::OrderedDict,l,i,j)
        return (br_g[l]+br_g_fr[l])/br_ttm[l]*x["vm_$(i)"]^2 +
        (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(x["vm_$(i)"]*x["vm_$(j)"]*cos(x["va_$(i)"]-x["va_$(j)"])) +
        (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(x["vm_$(i)"]*x["vm_$(j)"]*sin(x["va_$(i)"]-x["va_$(j)"])) -
        x["p_$(l)_$(i)_$(j)"]
    end
    # @NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
    function const_flow_q_from(x::OrderedDict,l,i,j)
        return -(br_b[l]+br_b_fr[l])/br_ttm[l]*x["vm_$(i)"]^2 -
       (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(x["vm_$(i)"]*x["vm_$(j)"]*cos(x["va_$(i)"]-x["va_$(j)"])) +
       (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(x["vm_$(i)"]*x["vm_$(j)"]*sin(x["va_$(i)"]-x["va_$(j)"])) -
       x["q_$(l)_$(i)_$(j)"]
    end

    # @NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
    function const_flow_p_to(x::OrderedDict,l,i,j)
        return (br_g[l]+br_g_to[l])*x["vm_$(j)"]^2 +
        (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(x["vm_$(j)"]*x["vm_$(i)"]*cos(x["va_$(j)"]-x["va_$(i)"])) +
        (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(x["vm_$(j)"]*x["vm_$(i)"]*sin(x["va_$(j)"]-x["va_$(i)"])) -
        x["p_$(l)_$(j)_$(i)"]
    end
    # @NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
    function const_flow_q_to(x::OrderedDict,l,i,j)
       return -(br_b[l]+br_b_to[l])*x["vm_$(j)"]^2 -
       (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(x["vm_$(j)"]*x["vm_$(i)"]*cos(x["va_$(j)"]-x["va_$(i)"])) +
       (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(x["vm_$(j)"]*x["vm_$(i)"]*sin(x["va_$(j)"]-x["va_$(i)"])) -
       x["q_$(l)_$(j)_$(i)"]
    end

    function const_thermal_limit(x::OrderedDict,l,i,j)
       return x["p_$(l)_$(i)_$(j)"]^2 + x["q_$(l)_$(i)_$(j)"]^2 - br_rate_a[l]^2
    end

    function const_voltage_angle_difference_lb(x::OrderedDict,l,i,j)
       return br_angmin[l] - x["va_$(i)"] + x["va_$(j)"]
    end

    function const_voltage_angle_difference_ub(x::OrderedDict,l,i,j)
       return x["va_$(i)"] - x["va_$(j)"] - br_angmax[l]
    end

    for (l,i,j) in ref[:arcs_from]
        add_eq_constraint!(model, x -> const_flow_p_from(x,l,i,j))
        add_eq_constraint!(model, x -> const_flow_q_from(x,l,i,j))

        add_eq_constraint!(model, x -> const_flow_p_to(x,l,i,j))
        add_eq_constraint!(model, x -> const_flow_q_to(x,l,i,j))

        add_ineq_constraint!(model, x -> const_thermal_limit(x,l,i,j))
        add_ineq_constraint!(model, x -> const_thermal_limit(x,l,j,i))

        add_ineq_constraint!(model, x -> const_voltage_angle_difference_lb(x,l,i,j))
        add_ineq_constraint!(model, x -> const_voltage_angle_difference_ub(x,l,i,j))
    end
    model
end

function solve_opf_nonconvex(dataset)
    model_build_time = @elapsed model = build_opf_nonconvex_prob(dataset)

    solve_time_with_compilation = @elapsed result = Nonconvex.optimize(
        model,
        IpoptAlg(),
        NonconvexCore.getinit(model);
        options = IpoptOptions(; first_order=false, symbolic=false, sparse=true, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME),
    )

    solve_time_without_compilation = @elapsed result = Nonconvex.optimize(
        model,
        IpoptAlg(),
        NonconvexCore.getinit(model);
        options = IpoptOptions(; first_order=false, symbolic=false, sparse=true, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME),
    )

    cost = result.minimum
    feasible = result.status == 0 # just guessing this is correct for Ipopt

    model_variables = Nonconvex.NonconvexCore.getnvars(model)
    model_constraints = Nonconvex.NonconvexCore.getnconstraints(model)

    return (model, result), Dict(
        "case" => file_name,
        "variables" => model_variables,
        "constraints" => model_constraints,
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_nonconvex_prob(dataset, test_u0)
    model = build_opf_nonconvex_prob(dataset)
        (;
    lookup_pg,
    lookup_qg,
    lookup_va,
    lookup_vm,
    lookup_lij,
    lookup_p_lij,
    lookup_q_lij) = dataset

    point = Dict()
    for v in keys(model.init)
        varsplit = split(v, "_")
        varname = varsplit[1]
        varint = parse.(Int, varsplit[2:end])

        idx = if varname == "p"
            lookup_p_lij[findfirst(x->x==Tuple(varint),lookup_lij)]
        elseif varname == "q"
            lookup_q_lij[findfirst(x->x==Tuple(varint),lookup_lij)]
        elseif varname == "va"
            lookup_va[varint[1]]
        elseif varname == "pg"
            lookup_pg[varint[1]]
        elseif varname == "qg"
            lookup_qg[varint[1]]
        elseif varname == "vm"
            lookup_vm[varint[1]]
        else
            error("Invalid $varname, $varint")
        end
        point[v] = test_u0[idx]
    end
    u0 = OrderedDict(keys(model.init) .=> getindex.((point,),keys(model.init)))
    obj = model.objective(u0)
    cons = vcat(model.eq_constraints(u0), model.ineq_constraints(u0))
    obj, cons
end
```

```
test_nonconvex_prob (generic function with 1 method)
```



```julia
nonconvex_test_res = test_nonconvex_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, -1.0701734802505833, -0.7150251969
424766, -5.108902216849064, -2.4731750920890136, -3.4972450591043294, -2.07
1687022809815, -2.617834191007569, -1.5522321037165985, 0.5457423426033834 
 …  -1.13747899115099, 0.09028143995439242, -17.895781377616647, -17.250522
75145584, -0.5297749920186825, -0.517422559177915, -15.579738337886946, -15
.848636897710394, -0.21708622332773042, -0.8301113278688671])
```



```julia
@assert nonconvex_test_res[1] ≈ test_obj
```


```julia
@assert sort(abs.(nonconvex_test_res[2])) ≈ sort(abs.(test_cons))
```

```
Error: DimensionMismatch: dimensions must match: a has dims (Base.OneTo(59)
,), b has dims (Base.OneTo(53),), mismatch at 1
```



```julia
println(sort(abs.(nonconvex_test_res[2])))
```

```
[0.0215258802699263, 0.08410522353400862, 0.09028143995439242, 0.1947644725
1065077, 0.21708622332773042, 0.21730536348839036, 0.2849522377447594, 0.33
063890653376593, 0.33289810365419437, 0.45744368990710405, 0.51742255917791
5, 0.5297749920186825, 0.5457423426033834, 0.5897538612894937, 0.7150251969
424766, 0.7775962590542184, 0.8301113278688671, 0.8834081057449231, 1.01073
99030803794, 1.0688558672739048, 1.0701734802505833, 1.0793383525116511, 1.
1313027747306061, 1.13747899115099, 1.3778364577303637, 1.4038702698147258,
 1.4932615291788414, 1.5522321037165985, 1.563372246165339, 1.5810450617647
889, 1.6000837514115611, 1.633412293595111, 1.7030832394691608, 1.725257352
7333024, 2.0016715378998793, 2.071687022809815, 2.4731750920890136, 2.61783
4191007569, 2.8227966798520674, 3.0047739260369246, 3.1618224299953224, 3.2
236954017592256, 3.4972450591043294, 4.23535583005632, 4.48598172729162, 5.
108902216849064, 5.164989238524806, 7.40879932706212, 9.915310272572729, 15
.579738337886946, 15.848636897710394, 17.145492534504328, 17.25052275145584
, 17.422407182792668, 17.563254560708913, 17.745610976379986, 17.7724302858
69348, 17.895781377616647, 18.034616568953247]
```



```julia
println(sort(abs.(test_cons)))
```

```
[0.0061762164203837955, 0.0215258802699263, 0.06615508569119477, 0.11298343
104675009, 0.15136310228960612, 0.19476447251065077, 0.21730536348839036, 0
.2518186223833513, 0.2849522377447594, 0.3065125522705683, 0.33289810365419
437, 0.3751697141306502, 0.4019890236200105, 0.4202616621130535, 0.54574234
26033834, 0.5843454392910835, 0.5950107614751935, 0.6077039991323074, 0.613
8802155526912, 0.7150251969424766, 0.7251928172073308, 0.7775962590542184, 
0.8542376821320647, 0.8834081057449231, 0.897077248544158, 1.00210746549566
83, 1.0107399030803794, 1.0688558672739048, 1.0701734802505833, 1.079338352
5116511, 1.2740182727083802, 1.4038702698147258, 1.4932615291788414, 1.5522
321037165985, 1.563372246165339, 1.5810450617647889, 1.6000837514115611, 1.
633412293595111, 1.7030832394691608, 1.7252573527333024, 2.0016715378998793
, 2.071687022809815, 2.473175092089014, 2.617834191007569, 2.82279667985206
74, 3.0047739260369246, 3.1618224299953224, 3.2236954017592256, 3.497245059
10433, 4.23535583005632, 5.108902216849063, 7.40879932706212, 9.91531027257
2729]
```





## Optim.jl

Implementation reference: https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/
Currently does not converge to a feasible point, root cause in unclear
`debug/optim-debug.jl` can be used to confirm it will converge if given a suitable starting point

```julia
import Optim

function build_opf_optim_prob(dataset)
    (;data, ref) = dataset

    bus_pd = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_qd = Dict(i => 0.0 for (i,bus) in ref[:bus])

    bus_gs = Dict(i => 0.0 for (i,bus) in ref[:bus])
    bus_bs = Dict(i => 0.0 for (i,bus) in ref[:bus])

    for (i,bus) in ref[:bus]
        if length(ref[:bus_loads][i]) > 0
            bus_pd[i] = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])
            bus_qd[i] = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i])
        end

        if length(ref[:bus_shunts][i]) > 0
            bus_gs[i] = sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i])
            bus_bs[i] = sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i])
        end
    end


    br_g = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_tr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ti = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_ttm = Dict(i => 0.0 for (i,branch) in ref[:branch])

    br_g_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_fr = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_g_to = Dict(i => 0.0 for (i,branch) in ref[:branch])
    br_b_to = Dict(i => 0.0 for (i,branch) in ref[:branch])

    for (i,branch) in ref[:branch]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)

        br_g[i] = g
        br_b[i] = b

        br_tr[i] = tr
        br_ti[i] = ti
        br_ttm[i] = tr^2 + ti^2

        br_g_fr[i] = branch["g_fr"]
        br_b_fr[i] = branch["b_fr"]
        br_g_to[i] = branch["g_to"]
        br_b_to[i] = branch["b_to"]
    end

    var_lookup = Dict{String,Int}()

    var_init = Float64[]
    var_lb = Float64[]
    var_ub = Float64[]

    var_idx = 1
    for (i,bus) in ref[:bus]
        push!(var_init, 0.0) #va
        push!(var_lb, -Inf)
        push!(var_ub, Inf)
        var_lookup["va_$(i)"] = var_idx
        var_idx += 1

        push!(var_init, 1.0) #vm
        push!(var_lb, bus["vmin"])
        push!(var_ub, bus["vmax"])
        var_lookup["vm_$(i)"] = var_idx
        var_idx += 1
    end

    for (i,gen) in ref[:gen]
        #push!(var_init, 0.0) #pg
        push!(var_init, (gen["pmax"]+gen["pmin"])/2) # non-standard start
        push!(var_lb, gen["pmin"])
        push!(var_ub, gen["pmax"])
        var_lookup["pg_$(i)"] = var_idx
        var_idx += 1

        #push!(var_init, 0.0) #qg
        push!(var_init, (gen["qmax"]+gen["qmin"])/2) # non-standard start
        push!(var_lb, gen["qmin"])
        push!(var_ub, gen["qmax"])
        var_lookup["qg_$(i)"] = var_idx
        var_idx += 1
    end

    for (l,i,j) in ref[:arcs]
        branch = ref[:branch][l]

        push!(var_init, 0.0) #p
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["p_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1

        push!(var_init, 0.0) #q
        push!(var_lb, -branch["rate_a"])
        push!(var_ub,  branch["rate_a"])
        var_lookup["q_$(l)_$(i)_$(j)"] = var_idx
        var_idx += 1
    end

    @assert var_idx == length(var_init)+1
    #total_callback_time = 0.0
    function opf_objective(x)
        #start = time()
        cost = 0.0
        for (i,gen) in ref[:gen]
            pg = x[var_lookup["pg_$(i)"]]
            cost += gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
        end
        #total_callback_time += time() - start
        return cost
    end

    function opf_constraints(c,x)
        #start = time()
        va = Dict(i => x[var_lookup["va_$(i)"]] for (i,bus) in ref[:bus])
        vm = Dict(i => x[var_lookup["vm_$(i)"]] for (i,bus) in ref[:bus])

        pg = Dict(i => x[var_lookup["pg_$(i)"]] for (i,gen) in ref[:gen])
        qg = Dict(i => x[var_lookup["qg_$(i)"]] for (i,gen) in ref[:gen])

        p = Dict((l,i,j) => x[var_lookup["p_$(l)_$(i)_$(j)"]] for (l,i,j) in ref[:arcs])
        q = Dict((l,i,j) => x[var_lookup["q_$(l)_$(i)_$(j)"]] for (l,i,j) in ref[:arcs])

        vm_fr = Dict(l => vm[branch["f_bus"]] for (l,branch) in ref[:branch])
        vm_to = Dict(l => vm[branch["t_bus"]] for (l,branch) in ref[:branch])
        va_fr = Dict(l => va[branch["f_bus"]] for (l,branch) in ref[:branch])
        va_to = Dict(l => va[branch["t_bus"]] for (l,branch) in ref[:branch])


        va_con = [va[i] for (i,bus) in ref[:ref_buses]]

        #     @constraint(model,
        #         sum(p[a] for a in ref[:bus_arcs][i]) ==
        #         sum(pg[g] for g in ref[:bus_gens][i]) -
        #         sum(load["pd"] for load in bus_loads) -
        #         sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        #     )
        power_balance_p_con = [
           sum(pg[j] for j in ref[:bus_gens][i]; init=0.0) -
           bus_pd[i] -
           bus_gs[i]*vm[i]^2 -
           sum(p[a] for a in ref[:bus_arcs][i])
           for (i,bus) in ref[:bus]
        ]

        #     @constraint(model,
        #         sum(q[a] for a in ref[:bus_arcs][i]) ==
        #         sum(qg[g] for g in ref[:bus_gens][i]) -
        #         sum(load["qd"] for load in bus_loads) +
        #         sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        #     )
        power_balance_q_con = [
           sum(qg[j] for j in ref[:bus_gens][i]; init=0.0) -
           bus_qd[i] +
           bus_bs[i]*vm[i]^2 -
           sum(q[a] for a in ref[:bus_arcs][i])
           for (i,bus) in ref[:bus]
        ]


        # @NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        power_flow_p_from_con = [
           (br_g[l]+br_g_fr[l])/br_ttm[l]*vm_fr[l]^2 +
           (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*cos(va_fr[l]-va_to[l])) +
           (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*sin(va_fr[l]-va_to[l])) -
           p[(l,i,j)]
           for (l,i,j) in ref[:arcs_from]
        ]

        # @NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        power_flow_p_to_con = [
           (br_g[l]+br_g_to[l])*vm_to[l]^2 +
           (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*cos(va_to[l]-va_fr[l])) +
           (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*sin(va_to[l]-va_fr[l])) -
           p[(l,i,j)]
           for (l,i,j) in ref[:arcs_to]
        ]

        # @NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        power_flow_q_from_con = [
           -(br_b[l]+br_b_fr[l])/br_ttm[l]*vm_fr[l]^2 -
           (-br_b[l]*br_tr[l]-br_g[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*cos(va_fr[l]-va_to[l])) +
           (-br_g[l]*br_tr[l]+br_b[l]*br_ti[l])/br_ttm[l]*(vm_fr[l]*vm_to[l]*sin(va_fr[l]-va_to[l])) -
           q[(l,i,j)]
           for (l,i,j) in ref[:arcs_from]
        ]

        # @NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        power_flow_q_to_con = [
           -(br_b[l]+br_b_to[l])*vm_to[l]^2 -
           (-br_b[l]*br_tr[l]+br_g[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*cos(va_to[l]-va_fr[l])) +
           (-br_g[l]*br_tr[l]-br_b[l]*br_ti[l])/br_ttm[l]*(vm_to[l]*vm_fr[l]*sin(va_to[l]-va_fr[l])) -
           q[(l,i,j)]
           for (l,i,j) in ref[:arcs_to]
        ]

        # @constraint(model, va_fr - va_to <= branch["angmax"])
        # @constraint(model, va_fr - va_to >= branch["angmin"])
        power_flow_vad_con = [
           va_fr[l] - va_to[l]
           for (l,i,j) in ref[:arcs_from]
        ]

        # @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        power_flow_mva_from_con = [
           p[(l,i,j)]^2 + q[(l,i,j)]^2
           for (l,i,j) in ref[:arcs_from]
        ]

        # @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
        power_flow_mva_to_con = [
           p[(l,i,j)]^2 + q[(l,i,j)]^2
           for (l,i,j) in ref[:arcs_to]
        ]

        c .= [
            va_con...,
            power_balance_p_con...,
            power_balance_q_con...,
            power_flow_p_from_con...,
            power_flow_p_to_con...,
            power_flow_q_from_con...,
            power_flow_q_to_con...,
            power_flow_vad_con...,
            power_flow_mva_from_con...,
            power_flow_mva_to_con...,
        ]
        #total_callback_time += time() - start
        return c
    end

    con_lbs = Float64[]
    con_ubs = Float64[]

    #@constraint(model, va[i] == 0)
    for (i,bus) in ref[:ref_buses]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end


    #power_balance_p_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
        #push!(con_lbs, -Inf)
        #push!(con_ubs, Inf)
    end

    #power_balance_q_con
    for (i,bus) in ref[:bus]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
        #push!(con_lbs, -Inf)
        #push!(con_ubs, Inf)
    end


    #power_flow_p_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_p_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_from_con
    for (l,i,j) in ref[:arcs_from]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_q_to_con
    for (l,i,j) in ref[:arcs_to]
        push!(con_lbs, 0.0)
        push!(con_ubs, 0.0)
    end

    #power_flow_vad_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, branch["angmin"])
        push!(con_ubs, branch["angmax"])
    end

    #power_flow_mva_from_con
    for (l,i,j) in ref[:arcs_from]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    #power_flow_mva_to_con
    for (l,i,j) in ref[:arcs_to]
        branch = ref[:branch][l]
        push!(con_lbs, -Inf)
        push!(con_ubs, branch["rate_a"]^2)
    end

    df = Optim.TwiceDifferentiable(opf_objective, var_init)
    dfc = Optim.TwiceDifferentiableConstraints(opf_constraints, var_lb, var_ub, con_lbs, con_ubs)
    df, dfc, var_init, con_lbs, con_ubs
end

function solve_opf_optim(dataset)
    model_build_time = @elapsed df, dfc, var_init, con_lbs, con_ubs = build_opf_optim_prob(dataset)

    options = Optim.Options(show_trace=PRINT_LEVEL != 0,time_limit=MAX_CPU_TIME)
    solve_time_with_compilation = @elapsed res = Optim.optimize(df, dfc, var_init, Optim.IPNewton(), options)
    solve_time_without_compilation = @elapsed res = Optim.optimize(df, dfc, var_init, Optim.IPNewton(), options)

    sol = res.minimizer
    cost = res.minimum

    # NOTE: confirmed these constraint violations can be eliminated
    # if a better starting point is used
    sol_eval = dfc.c!(zeros(dfc.bounds.nc), sol)
    vio_lb = [max(v,0) for v in (con_lbs .- sol_eval)]
    vio_ub = [max(v,0) for v in (sol_eval .- con_ubs)]
    const_vio = vio_lb .+ vio_ub
    constraint_tol = 1e-6
    feasible = (sum(const_vio) <= constraint_tol)

    return (res,), Dict(
        "case" => file_name,
        "variables" => length(var_init),
        "constraints" => dfc.bounds.nc,
        "feasible" => feasible,
        "cost" => cost,
        "time_build" => model_build_time,
        "time_solve" => solve_time_without_compilation,
        "time_solve_compilation" => solve_time_with_compilation,
    )
end

function test_optim_prob(dataset, test_u0)
    df, dfc, var_init, con_lbs, con_ubs = build_opf_optim_prob(dataset)
    obj = df.f(test_u0)
    cons = dfc.c!(zeros(dfc.bounds.nc), test_u0)
    obj, cons
end
```

```
test_optim_prob (generic function with 1 method)
```



```julia
optim_test_res = test_optim_prob(dataset, test_u0)
```

```
(7079.190664351089, [0.0215258802699263, -1.0701734802505833, -5.1089022168
49063, -3.49724505910433, -2.617834191007569, 0.5457423426033834, -0.715025
1969424766, -2.473175092089014, -2.071687022809815, -1.5522321037165985  … 
 1.2740182727083802, 0.11298343104675009, 0.2518186223833513, 0.42026166211
30535, 0.3751697141306502, 0.4019890236200105, 0.5950107614751935, 1.002107
4654956683, 0.897077248544158, 0.15136310228960612])
```



```julia
@assert optim_test_res[1] == test_obj
```


```julia
@assert optim_test_res[2] == test_cons
```




## CASADI

Implementation reference: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/ac-opf.jl

CASADI Segfaults so removed for now.

```
import PowerModels
import PythonCall
import CondaPkg
CondaPkg.add("casadi")

function solve_opf_casadi(dataset)
    (;data, ref) = dataset

    time_model_start = time()

    casadi = PythonCall.pyimport("casadi")

    x, x0, lbx, ubx, cons, lbg, ubg = [], [], [], [], [], [], []
    va, vm = Dict{Int,Any}(), Dict{Int,Any}()
    for (k, _) in ref[:bus]
        va[k] = casadi.SX.sym("va$k")
        push!(x, va[k])
        push!(x0, 0.0)
        push!(lbx, -casadi.inf)
        push!(ubx, casadi.inf)
        vm[k] = casadi.SX.sym("vm$k")
        push!(x, vm[k])
        push!(x0, 1.0)
        push!(lbx, ref[:bus][k]["vmin"])
        push!(ubx, ref[:bus][k]["vmax"])
    end
    pg, qg = Dict{Int,Any}(), Dict{Int,Any}()
    for (k, ) in ref[:gen]
        pg[k] = casadi.SX.sym("pg$k")
        push!(x, pg[k])
        push!(x0, 0.0)
        push!(lbx, ref[:gen][k]["pmin"])
        push!(ubx, ref[:gen][k]["pmax"])
        qg[k] = casadi.SX.sym("qg$k")
        push!(x, qg[k])
        push!(x0, 0.0)
        push!(lbx, ref[:gen][k]["qmin"])
        push!(ubx, ref[:gen][k]["qmax"])
    end
    p, q = Dict{NTuple{3,Int},Any}(), Dict{NTuple{3,Int},Any}()
    for k in ref[:arcs]
        a = ref[:branch][k[1]]["rate_a"]
        p[k] = casadi.SX.sym("p$k")
        push!(x, p[k])
        push!(x0, 0.0)
        push!(lbx, -a)
        push!(ubx, a)
        q[k] = casadi.SX.sym("q$k")
        push!(x, q[k])
        push!(x0, 0.0)
        push!(lbx, -a)
        push!(ubx, a)
    end
    f = sum(
        cost["cost"][1] * pg[k]^2 +
        cost["cost"][2] * pg[k] +
        cost["cost"][3] for (k, cost) in ref[:gen]
    )
    for (k, _) in ref[:ref_buses]
        push!(cons, va[k])
        push!(lbg, 0)
        push!(ubg, 0)
    end
    for (i, _) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        push!(
            cons,
            sum(p[k] for k in ref[:bus_arcs][i]) -
            sum(pg[g] for g in ref[:bus_gens][i]; init = 0) +
            sum(load["pd"] for load in bus_loads; init = 0) +
            sum(shunt["gs"] for shunt in bus_shunts; init = 0) * vm[i]^2
        )
        push!(lbg, 0)
        push!(ubg, 0)
        push!(
            cons,
            sum(q[k] for k in ref[:bus_arcs][i]) -
            sum(qg[g] for g in ref[:bus_gens][i]; init = 0) +
            sum(load["qd"] for load in bus_loads; init = 0) -
            sum(shunt["bs"] for shunt in bus_shunts; init = 0) * vm[i]^2
        )
        push!(lbg, 0)
        push!(ubg, 0)
    end
    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])
        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]
        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]
        push!(
            cons,
            (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*casadi.cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*casadi.sin(va_fr-va_to)) - p_fr
        )
        push!(
            cons,
            -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*casadi.cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*casadi.sin(va_fr-va_to)) - q_fr
        )
        push!(
            cons,
            (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*casadi.cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*casadi.sin(va_to-va_fr)) - p_to
        )
        push!(
            cons,
            -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*casadi.cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*casadi.sin(va_to-va_fr)) - q_to
        )
        for i in 1:4
            push!(lbg, 0)
            push!(ubg, 0)
        end
        push!(cons, va_fr - va_to)
        push!(lbg, branch["angmin"])
        push!(ubg, branch["angmax"])
        push!(cons, p_fr^2 + q_fr^2)
        push!(lbg, -casadi.inf)
        push!(ubg, branch["rate_a"]^2)
        push!(cons, p_to^2 + q_to^2)
        push!(lbg, -casadi.inf)
        push!(ubg, branch["rate_a"]^2)
    end

    nlp = Dict("x" => casadi.vcat(x), "f" => f, "g" => casadi.vcat(cons))
    options = PythonCall.pydict(Dict("error_on_fail" => true))
    model = casadi.nlpsol("model", "ipopt", PythonCall.pydict(nlp), options)

    model_variables = length(x)
    model_constraints = length(lbg)

    model_build_time = time() - time_model_start

    time_solve_start = time()
    solution = model(; lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg, x0 = x0)
    cost = PythonCall.pyconvert(Float64, (PythonCall.pyfloat(solution["f"])))
    feasible = true # error if not feasible

    solve_time = time() - time_solve_start
    total_time = time() - time_data_start

    println("")
    println("\033[1mSummary\033[0m")
    println("   case........: $(file_name)")
    println("   variables...: $(model_variables)")
    println("   constraints.: $(model_constraints)")
    println("   feasible....: $(feasible)")
    println("   cost........: $(round(Int, cost))")
    println("   total time..: $(total_time)")
    println("     data time.: $(data_load_time)")
    println("     build time: $(model_build_time)")
    println("     solve time: $(solve_time)")
    # println("      callbacks: $(total_callback_time)")
    println("")

    return Dict(
        "case" => file_name,
        "variables" => model_variables,
        "constraints" => model_constraints,
        "feasible" => feasible,
        "cost" => cost,
        "time_total" => total_time,
        "time_data" => data_load_time,
        "time_build" => model_build_time,
        "time_solve" => solve_time,
        #"time_callbacks" => TBD,
    )
end

solve_opf_casadi("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
```

## Test the Benchmarking

```julia
file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
dataset = load_and_setup_data(file_name);
```


```julia
model, res = solve_opf_optimization(dataset);
res
```

```
***************************************************************************
***
This program contains Ipopt, a library for large-scale nonlinear optimizati
on.
 Ipopt is released as open source code under the Eclipse Public License (EP
L).
         For more information visit https://github.com/coin-or/Ipopt
***************************************************************************
***

Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.000190159
  "time_solve_compilation" => 16.3793
  "time_solve"             => 1.54655
  "feasible"               => true
```



```julia
model, res = solve_opf_jump(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.00209108
  "time_solve_compilation" => 1.03216
  "time_solve"             => 0.0124845
  "feasible"               => true
```



```julia
model, res = solve_opf_nlpmodels(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.639783
  "time_solve_compilation" => 2.59046
  "time_solve"             => 0.0349775
  "feasible"               => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
Error: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDi
ff.Tag{NonconvexUtils.var"#101#108"{NonconvexUtils.var"#100#107"{NonconvexU
tils.var"#97#104"{NonconvexCore.VectorOfFunctions{Tuple{NonconvexCore.IneqC
onstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"
##WeaveSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_t
hermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentia
bleFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}
, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int
64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity
), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconve
xCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrappe
r{Main.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".v
ar"#const_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, 
DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String
, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}
, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{type
of(identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64
}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Fun
ctionWrapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSan
dBox#225".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}
, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedColle
ctions.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_
to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#u
nflatten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{Strin
g, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"
#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220
#262"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#2
54"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unf
latten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiab
leFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Dif
ferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColl
ections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCons
traint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##W
eaveSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_ther
mal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiable
Flatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, D
ifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}
, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), 
OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCo
re.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{M
ain.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"
#const_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Dif
ferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, F
loat64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, T
uple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(
identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, 
NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Functi
onWrapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBo
x#225".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, I
nt64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollecti
ons.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_
Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unfl
atten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, 
Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80
#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#26
2"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"
{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflat
ten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableF
latten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differ
entiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollect
ions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstra
int{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##Weav
eSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_thermal
_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFla
tten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Diff
erentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, T
uple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ord
eredCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.
IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main
.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"#co
nst_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differ
entiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Floa
t64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tupl
e{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(ide
ntity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Non
convexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionW
rapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBox#2
25".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, Int6
4, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections
.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tup
le#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatt
en_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Flo
at64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82
"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#262"{
Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"{Di
ct{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten
{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlat
ten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Different
iableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollection
s.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint
{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSa
ndBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_thermal_li
mit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatte
n.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differe
ntiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tupl
e{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ordere
dCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.Ine
qConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.va
r"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"#const
_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Different
iableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64
}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{I
nt64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identi
ty), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Noncon
vexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrap
per{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBox#225"
.var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, Int64, 
Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.Or
deredDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#
11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_
to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float6
4}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{N
onconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#262"{Mai
n.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"{Dict{
Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tu
ple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlatten
.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differentiab
leFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollections.O
rderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{No
nconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandB
ox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_thermal_limit
#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.U
nflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differenti
ableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{D
ifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCo
llections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCo
nstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"#
#WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"#const_th
ermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiab
leFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}},
 DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int6
4}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity)
, OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconvex
Core.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper
{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBox#225".va
r"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, Int64, Int
64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.Order
edDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"
{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_
Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float64}}
}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{Nonc
onvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#262"{Main.v
ar"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"{Dict{Int
64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple
{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlatten.va
r"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableF
latten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollections.Orde
redDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{Nonco
nvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#
225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_thermal_limit#25
2"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unfl
atten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiabl
eFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Diff
erentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColle
ctions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConst
raint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##We
aveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"#const_therm
al_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableF
latten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Di
fferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64},
 Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), O
rderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCor
e.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Ma
in.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBox#225".var"#
const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, Int64, Int64,
 Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedD
ict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tu
ple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dic
t#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float64}}}}}
}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{Nonconv
exCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#262"{Main.var"
##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"{Dict{Int64,
 Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{Or
deredCollections.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#
unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlat
ten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollections.Ordered
Dict{String, Float64}}}}}}, Float64}}}}}}, Float64}, Float64, 1})

Closest candidates are:
  (::Type{T})(::Real, !Matched::RoundingMode) where T<:AbstractFloat
   @ Base rounding.jl:207
  (::Type{T})(::T) where T<:Number
   @ Core boot.jl:792
  Float64(!Matched::IrrationalConstants.Logπ)
   @ IrrationalConstants /cache/julia-buildkite-plugin/depots/5b300254-1738
-4989-ae0a-f4d2d937f953/packages/IrrationalConstants/vp5v4/src/macro.jl:112
  ...
```



```julia
model, res = solve_opf_optim(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 77.9548
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.000471267
  "time_solve_compilation" => 22.3103
  "time_solve"             => 18.1145
  "feasible"               => false
```



```julia
file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m"
dataset = load_and_setup_data(file_name);
```


```julia
model, res = solve_opf_optimization(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 5812.64
  "variables"              => 24
  "constraints"            => 28
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 9.53e-5
  "time_solve_compilation" => 0.214868
  "time_solve"             => 0.207903
  "feasible"               => true
```



```julia
model, res = solve_opf_jump(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 5812.64
  "variables"              => 24
  "constraints"            => 28
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.00192784
  "time_solve_compilation" => 0.00913105
  "time_solve"             => 0.00833238
  "feasible"               => true
```



```julia
model, res = solve_opf_nlpmodels(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 5812.64
  "variables"              => 24
  "constraints"            => 28
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.0136894
  "time_solve_compilation" => 0.0193051
  "time_solve"             => 0.0184144
  "feasible"               => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
Error: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDi
ff.Tag{NonconvexUtils.var"#101#108"{NonconvexUtils.var"#100#107"{NonconvexU
tils.var"#97#104"{NonconvexCore.VectorOfFunctions{Tuple{NonconvexCore.IneqC
onstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"
##WeaveSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_t
hermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentia
bleFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}
, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int
64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity
), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconve
xCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrappe
r{Main.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".v
ar"#const_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, 
DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String
, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}
, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{type
of(identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64
}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Fun
ctionWrapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSan
dBox#225".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}
, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedColle
ctions.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_
to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#u
nflatten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{Strin
g, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"
#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220
#262"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#2
54"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unf
latten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiab
leFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Dif
ferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColl
ections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCons
traint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##W
eaveSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_ther
mal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiable
Flatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, D
ifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}
, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), 
OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCo
re.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{M
ain.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"
#const_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Dif
ferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, F
loat64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, T
uple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(
identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, 
NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Functi
onWrapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBo
x#225".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, I
nt64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollecti
ons.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_
Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unfl
atten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, 
Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80
#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#26
2"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"
{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflat
ten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableF
latten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differ
entiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollect
ions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstra
int{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##Weav
eSandBox#225".var"#217#259"{Main.var"##WeaveSandBox#225".var"#const_thermal
_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFla
tten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Diff
erentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, T
uple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ord
eredCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.
IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main
.var"##WeaveSandBox#225".var"#218#260"{Main.var"##WeaveSandBox#225".var"#co
nst_thermal_limit#252"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differ
entiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Floa
t64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tupl
e{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(ide
ntity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Non
convexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionW
rapper{Main.var"##WeaveSandBox#225".var"#219#261"{Main.var"##WeaveSandBox#2
25".var"#const_voltage_angle_difference_lb#253"{Dict{Int64, Float64}}, Int6
4, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections
.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tup
le#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatt
en_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Flo
at64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82
"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#220#262"{
Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#254"{Di
ct{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten
{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlat
ten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Different
iableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollection
s.OrderedDict{String, Float64}}}}}}, Float64}}}}}}, Float64}, Float64, 1})

Closest candidates are:
  (::Type{T})(::Real, !Matched::RoundingMode) where T<:AbstractFloat
   @ Base rounding.jl:207
  (::Type{T})(::T) where T<:Number
   @ Core boot.jl:792
  Float64(!Matched::IrrationalConstants.Logπ)
   @ IrrationalConstants /cache/julia-buildkite-plugin/depots/5b300254-1738
-4989-ae0a-f4d2d937f953/packages/IrrationalConstants/vp5v4/src/macro.jl:112
  ...
```



```julia
model, res = solve_opf_optim(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 6273.63
  "variables"              => 24
  "constraints"            => 28
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.0515386
  "time_solve_compilation" => 2.30253
  "time_solve"             => 2.2211
  "feasible"               => false
```



```julia
using DataFrames, PrettyTables

function multidata_multisolver_benchmark(dataset_files; sizelimit = SIZE_LIMIT)

    cases = String[]
    vars = Int[]
    cons = Int[]

    optimization_time = Float64[]
    mtk_time = Float64[]
    jump_time = Float64[]
    nlpmodels_time = Float64[]
    nonconvex_time = Float64[]
    optim_time = Float64[]

    optimization_time_modelbuild = Float64[]
    mtk_time_modelbuild = Float64[]
    jump_time_modelbuild = Float64[]
    nlpmodels_time_modelbuild = Float64[]
    nonconvex_time_modelbuild = Float64[]
    optim_time_modelbuild = Float64[]

    optimization_time_compilation = Float64[]
    mtk_time_compilation = Float64[]
    jump_time_compilation = Float64[]
    nlpmodels_time_compilation = Float64[]
    nonconvex_time_compilation = Float64[]
    optim_time_compilation = Float64[]

    optimization_cost = Float64[]
    mtk_cost = Float64[]
    jump_cost = Float64[]
    nlpmodels_cost = Float64[]
    nonconvex_cost = Float64[]
    optim_cost = Float64[]

    for file in dataset_files
        @show file
        dataset = load_and_setup_data(file)

        prob = build_opf_optimization_prob(dataset)
        @info "Number of Variables: $(length(prob.u0))"
        @info "Number of Constraints: $(length(prob.lcons))"

        if length(prob.u0) > sizelimit
            @info "Variable size over global limit. Skipping for now"
            continue
        end
        
        @info "Running Optimization.jl"
        model, res = solve_opf_optimization(dataset)
        push!(cases, split(file,"/")[end])
        push!(vars, res["variables"])
        push!(cons, res["constraints"])
        push!(optimization_time, res["time_solve"])
        push!(optimization_time_modelbuild, res["time_build"])
        push!(optimization_time_compilation, res["time_solve_compilation"])
        push!(optimization_cost, res["cost"])

        @info "Running ModelingToolkit.jl"
        model, res = solve_opf_mtk(dataset)
        push!(mtk_time, res["time_solve"])
        push!(mtk_time_modelbuild, res["time_build"])
        push!(mtk_time_compilation, res["time_solve_compilation"])
        push!(mtk_cost, res["cost"])


        @info "Running JuMP.jl"
        model, res = solve_opf_jump(dataset)
        push!(jump_time, res["time_solve"])
        push!(jump_time_modelbuild, res["time_build"])
        push!(jump_time_compilation, res["time_solve_compilation"])
        push!(jump_cost, res["cost"])

        @info "Running NLPModels.jl"
        model, res = solve_opf_nlpmodels(dataset)
        push!(nlpmodels_time, res["time_solve"])
        push!(nlpmodels_time_modelbuild, res["time_build"])
        push!(nlpmodels_time_compilation, res["time_solve_compilation"])
        push!(nlpmodels_cost, res["cost"])
        
        #=
        @info "Running Nonconvex.jl"
        model, res = solve_opf_nonconvex(dataset)
        push!(nonconvex_time, res["time_solve"])
        push!(nonconvex_time_modelbuild, res["time_build"])
        push!(nonconvex_time_compilation, res["time_solve_compilation"])
        push!(nonconvex_cost, res["cost"])
        =#
        
        if length(prob.u0) > 400
            @info "Running Optim.jl"
            model, res = solve_opf_optim(dataset)
            push!(optim_time, NaN)
            push!(optim_time_modelbuild, NaN)
            push!(optim_time_compilation, NaN)
            push!(optim_cost, NaN)
        else
            @info "Running Optim.jl"
            model, res = solve_opf_optim(dataset)
            push!(optim_time, res["time_solve"])
            push!(optim_time_modelbuild, res["time_build"])
            push!(optim_time_compilation, res["time_solve_compilation"])
            push!(optim_cost, res["cost"])
        end
    end
    DataFrame(:case => cases, :vars => vars, :cons => cons, 
              :optimization => optimization_time, :optimization_modelbuild => optimization_time_modelbuild, :optimization_wcompilation => optimization_time_compilation, :optimization_cost => optimization_cost,
              :mtk => mtk_time, :mtk_time_modelbuild => mtk_time_modelbuild, :mtk_time_wcompilation => mtk_time_compilation, :mtk_cost => mtk_cost,
              :jump => jump_time, :jump_modelbuild => jump_time_modelbuild, :jump_wcompilation => jump_time_compilation, :jump_cost => jump_cost, 
              :nlpmodels => nlpmodels_time, :nlpmodels_modelbuild => nlpmodels_time_modelbuild, :nlpmodels_wcompilation => nlpmodels_time_compilation,  :nlpmodels_cost => nlpmodels_cost, 
              #:nonconvex => nonconvex_time, :nonconvex_modelbuild => nonconvex_time_modelbuild, :nonconvex_wcompilation => nonconvex_time_compilation,  :nonconvex_cost => nonconvex_cost,
              :optim => optim_time, :optim_modelbuild => optim_time_modelbuild, :optim_wcompilation => optim_time_compilation,  :optim_cost => optim_cost)
end

test_datasets = [
    "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m",
    "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
    ]
```

```
2-element Vector{String}:
 "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m"
 "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
```



```julia
timing_data = multidata_multisolver_benchmark(test_datasets)
```

```
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       12
Number of nonzeros in inequality constraint Jacobian.:       28
Number of nonzeros in Lagrangian Hessian.............:       81

Total number of variables............................:        7
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        5
                     variables with only upper bounds:        0
Total number of equality constraints.................:        2
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.0415000e+03 1.00e-01 2.18e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8281767e+03 4.88e-02 1.73e+01  -1.0 1.72e-01   2.0 8.71e-01 1.00e+00f  1
   2  1.2456491e+03 4.02e-02 1.24e+01  -1.0 3.52e-01   1.5 7.50e-01 4.67e-01f  1
   3  1.1757068e+03 3.55e-02 1.09e+01  -1.0 4.02e-01    -  2.11e-01 1.25e-01f  1
   4  9.7867738e+02 3.20e-02 6.94e+00  -1.0 4.50e-01    -  1.34e-01 3.62e-01f  1
   5  7.8792959e+02 2.52e-02 8.87e+00  -1.0 3.55e-01    -  4.83e-01 9.79e-01f  1
   6  8.6287363e+02 2.15e-03 1.01e+00  -1.0 1.00e-01    -  9.91e-01 1.00e+00h  1
   7  7.9572299e+02 1.24e-03 6.38e-01  -1.0 9.73e-02    -  1.00e+00 1.00e+00f  1
   8  7.5317979e+02 7.04e-04 1.40e-01  -1.7 6.32e-02    -  1.00e+00 1.00e+00f  1
   9  7.4438683e+02 8.02e-05 3.37e-02  -2.5 1.26e-02    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  7.4408348e+02 3.66e-06 3.72e-03  -2.5 2.53e-03    -  1.00e+00 1.00e+00f  1
  11  7.4309700e+02 2.80e-06 2.64e-03  -3.8 2.09e-03    -  1.00e+00 1.00e+00f  1
  12  7.4305680e+02 7.30e-08 9.05e-05  -3.8 3.30e-04    -  1.00e+00 1.00e+00h  1
  13  7.4300130e+02 1.40e-08 1.42e-05  -5.7 1.46e-04    -  1.00e+00 1.00e+00h  1
  14  7.4300039e+02 9.07e-12 1.02e-08  -8.6 3.65e-06    -  1.00e+00 1.00e+00h  1
  15  7.4300039e+02 3.05e-16 1.07e-14  -9.0 2.88e-09    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0165744010648609e+01    7.4300038920867655e+02
Dual infeasibility......:   1.0658141036401503e-14    7.7898901742874248e-13
Constraint violation....:   3.0531133177191805e-16    3.0531133177191805e-16
Variable bound violation:   1.0864721122771925e-08    1.0864721122771925e-08
Complementarity.........:   9.0909205311425006e-10    6.6444300444990875e-08
Overall NLP error.......:   9.0909205311425006e-10    6.6444300444990875e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 1.992

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       12
Number of nonzeros in inequality constraint Jacobian.:       28
Number of nonzeros in Lagrangian Hessian.............:       81

Total number of variables............................:        7
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        5
                     variables with only upper bounds:        0
Total number of equality constraints.................:        2
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.0415000e+03 1.00e-01 2.18e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8281767e+03 4.88e-02 1.73e+01  -1.0 1.72e-01   2.0 8.71e-01 1.00e+00f  1
   2  1.2456491e+03 4.02e-02 1.24e+01  -1.0 3.52e-01   1.5 7.50e-01 4.67e-01f  1
   3  1.1757068e+03 3.55e-02 1.09e+01  -1.0 4.02e-01    -  2.11e-01 1.25e-01f  1
   4  9.7867738e+02 3.20e-02 6.94e+00  -1.0 4.50e-01    -  1.34e-01 3.62e-01f  1
   5  7.8792959e+02 2.52e-02 8.87e+00  -1.0 3.55e-01    -  4.83e-01 9.79e-01f  1
   6  8.6287363e+02 2.15e-03 1.01e+00  -1.0 1.00e-01    -  9.91e-01 1.00e+00h  1
   7  7.9572299e+02 1.24e-03 6.38e-01  -1.0 9.73e-02    -  1.00e+00 1.00e+00f  1
   8  7.5317979e+02 7.04e-04 1.40e-01  -1.7 6.32e-02    -  1.00e+00 1.00e+00f  1
   9  7.4438683e+02 8.02e-05 3.37e-02  -2.5 1.26e-02    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  7.4408348e+02 3.66e-06 3.72e-03  -2.5 2.53e-03    -  1.00e+00 1.00e+00f  1
  11  7.4309700e+02 2.80e-06 2.64e-03  -3.8 2.09e-03    -  1.00e+00 1.00e+00f  1
  12  7.4305680e+02 7.30e-08 9.05e-05  -3.8 3.30e-04    -  1.00e+00 1.00e+00h  1
  13  7.4300130e+02 1.40e-08 1.42e-05  -5.7 1.46e-04    -  1.00e+00 1.00e+00h  1
  14  7.4300039e+02 9.07e-12 1.02e-08  -8.6 3.65e-06    -  1.00e+00 1.00e+00h  1
  15  7.4300039e+02 3.05e-16 1.07e-14  -9.0 2.88e-09    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0165744010648609e+01    7.4300038920867655e+02
Dual infeasibility......:   1.0658141036401503e-14    7.7898901742874248e-13
Constraint violation....:   3.0531133177191805e-16    3.0531133177191805e-16
Variable bound violation:   1.0864721122771925e-08    1.0864721122771925e-08
Complementarity.........:   9.0909205311425006e-10    6.6444300444990875e-08
Overall NLP error.......:   9.0909205311425006e-10    6.6444300444990875e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.008

EXIT: Optimal Solution Found.
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       40
Number of nonzeros in inequality constraint Jacobian.:       60
Number of nonzeros in Lagrangian Hessian.............:      200

Total number of variables............................:       13
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        9
                     variables with only upper bounds:        0
Total number of equality constraints.................:        6
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.0500000e+03 3.99e+00 4.24e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4085733e+04 2.02e-03 9.56e+02  -1.0 1.40e+00    -  1.48e-01 1.00e+00H  1
   2  2.3966039e+04 1.35e-02 1.34e+01  -1.0 8.05e-01    -  9.51e-01 1.00e+00h  1
   3  2.3907973e+04 4.00e-04 1.38e+02  -1.0 4.93e-02   2.0 1.00e+00 1.00e+00h  1
   4  2.3296939e+04 9.47e-04 8.48e+01  -1.0 7.49e-01    -  1.00e+00 3.86e-01f  2
   5  2.2869373e+04 1.15e-03 4.61e+01  -1.0 2.47e-01    -  1.00e+00 1.00e+00h  1
   6  1.6680309e+04 1.15e-01 1.94e+01  -1.0 1.40e+01    -  5.68e-01 4.66e-01F  1
   7  1.8872763e+04 4.50e-02 5.76e+00  -1.0 5.62e+00    -  1.00e+00 1.00e+00f  1
   8  1.8270004e+04 2.90e-03 3.68e-01  -1.0 2.89e+00    -  1.00e+00 1.00e+00h  1
   9  1.6971509e+04 2.56e-03 1.50e+00  -1.7 2.42e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.6728621e+04 5.23e-03 2.68e-02  -1.7 2.74e+00    -  1.00e+00 1.00e+00h  1
  11  1.6756707e+04 3.39e-04 7.94e-03  -1.7 1.19e+00    -  1.00e+00 1.00e+00h  1
  12  1.6334001e+04 4.68e-03 4.47e-01  -3.8 5.15e+00    -  8.43e-01 8.83e-01h  1
  13  1.6273539e+04 4.05e-02 2.21e-01  -3.8 5.96e+00    -  8.57e-01 6.23e-01h  1
  14  1.6189011e+04 2.73e-02 3.55e-01  -3.8 3.46e+00    -  1.00e+00 1.00e+00h  1
  15  1.6238285e+04 5.42e-04 1.65e-03  -3.8 6.43e-01    -  1.00e+00 1.00e+00h  1
  16  1.6240300e+04 2.03e-05 2.15e-04  -3.8 1.07e-01    -  1.00e+00 1.00e+00h  1
  17  1.6236954e+04 2.07e-04 3.13e-03  -5.7 2.07e-01    -  9.75e-01 7.29e-01h  1
  18  1.6236733e+04 1.94e-05 1.70e-04  -5.7 3.64e-02    -  1.00e+00 9.90e-01h  1
  19  1.6236739e+04 7.49e-09 4.19e-08  -5.7 1.71e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.6236704e+04 1.03e-08 4.43e-06  -8.6 1.72e-03    -  1.00e+00 9.92e-01h  1
  21  1.6236704e+04 1.81e-13 1.93e-12  -8.6 8.11e-06    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3293548355426417e+00    1.6236704303715280e+04
Dual infeasibility......:   1.9305324434281426e-12    7.2402206849240285e-09
Constraint violation....:   1.8141044222375058e-13    1.8141044222375058e-13
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059961047863117e-09    9.3984252354722182e-06
Overall NLP error.......:   2.5059961047863117e-09    9.3984252354722182e-06


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 3.667

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       40
Number of nonzeros in inequality constraint Jacobian.:       60
Number of nonzeros in Lagrangian Hessian.............:      200

Total number of variables............................:       13
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        9
                     variables with only upper bounds:        0
Total number of equality constraints.................:        6
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.0500000e+03 3.99e+00 4.24e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4085733e+04 2.02e-03 9.56e+02  -1.0 1.40e+00    -  1.48e-01 1.00e+00H  1
   2  2.3966039e+04 1.35e-02 1.34e+01  -1.0 8.05e-01    -  9.51e-01 1.00e+00h  1
   3  2.3907973e+04 4.00e-04 1.38e+02  -1.0 4.93e-02   2.0 1.00e+00 1.00e+00h  1
   4  2.3296939e+04 9.47e-04 8.48e+01  -1.0 7.49e-01    -  1.00e+00 3.86e-01f  2
   5  2.2869373e+04 1.15e-03 4.61e+01  -1.0 2.47e-01    -  1.00e+00 1.00e+00h  1
   6  1.6680309e+04 1.15e-01 1.94e+01  -1.0 1.40e+01    -  5.68e-01 4.66e-01F  1
   7  1.8872763e+04 4.50e-02 5.76e+00  -1.0 5.62e+00    -  1.00e+00 1.00e+00f  1
   8  1.8270004e+04 2.90e-03 3.68e-01  -1.0 2.89e+00    -  1.00e+00 1.00e+00h  1
   9  1.6971509e+04 2.56e-03 1.50e+00  -1.7 2.42e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.6728621e+04 5.23e-03 2.68e-02  -1.7 2.74e+00    -  1.00e+00 1.00e+00h  1
  11  1.6756707e+04 3.39e-04 7.94e-03  -1.7 1.19e+00    -  1.00e+00 1.00e+00h  1
  12  1.6334001e+04 4.68e-03 4.47e-01  -3.8 5.15e+00    -  8.43e-01 8.83e-01h  1
  13  1.6273539e+04 4.05e-02 2.21e-01  -3.8 5.96e+00    -  8.57e-01 6.23e-01h  1
  14  1.6189011e+04 2.73e-02 3.55e-01  -3.8 3.46e+00    -  1.00e+00 1.00e+00h  1
  15  1.6238285e+04 5.42e-04 1.65e-03  -3.8 6.43e-01    -  1.00e+00 1.00e+00h  1
  16  1.6240300e+04 2.03e-05 2.15e-04  -3.8 1.07e-01    -  1.00e+00 1.00e+00h  1
  17  1.6236954e+04 2.07e-04 3.13e-03  -5.7 2.07e-01    -  9.75e-01 7.29e-01h  1
  18  1.6236733e+04 1.94e-05 1.70e-04  -5.7 3.64e-02    -  1.00e+00 9.90e-01h  1
  19  1.6236739e+04 7.49e-09 4.19e-08  -5.7 1.71e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.6236704e+04 1.03e-08 4.43e-06  -8.6 1.72e-03    -  1.00e+00 9.92e-01h  1
  21  1.6236704e+04 1.81e-13 1.93e-12  -8.6 8.11e-06    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3293548355426417e+00    1.6236704303715280e+04
Dual infeasibility......:   1.9305324434281426e-12    7.2402206849240285e-09
Constraint violation....:   1.8141044222375058e-13    1.8141044222375058e-13
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059961047863117e-09    9.3984252354722182e-06
Overall NLP error.......:   2.5059961047863117e-09    9.3984252354722182e-06


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.016

EXIT: Optimal Solution Found.
2×23 DataFrame
 Row │ case                    vars   cons   optimization  optimization_modelb ⋯
     │ String                  Int64  Int64  Float64       Float64             ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ pglib_opf_case3_lmbd.m     24     28      0.166827                5.169 ⋯
   2 │ pglib_opf_case5_pjm.m      44     53      0.876303                5.333
                                                              19 columns omitted
```



```julia
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, timing_data; backend = Val(:html))
# show(io, "text/html", pretty_table(timing_data; backend = Val(:html)))
println(io, "```")
Text(String(take!(io)))
```

```@raw html
<table>
  <thead>
    <tr class = "header">
      <th style = "text-align: right;">case</th>
      <th style = "text-align: right;">vars</th>
      <th style = "text-align: right;">cons</th>
      <th style = "text-align: right;">optimization</th>
      <th style = "text-align: right;">optimization_modelbuild</th>
      <th style = "text-align: right;">optimization_wcompilation</th>
      <th style = "text-align: right;">optimization_cost</th>
      <th style = "text-align: right;">mtk</th>
      <th style = "text-align: right;">mtk_time_modelbuild</th>
      <th style = "text-align: right;">mtk_time_wcompilation</th>
      <th style = "text-align: right;">mtk_cost</th>
      <th style = "text-align: right;">jump</th>
      <th style = "text-align: right;">jump_modelbuild</th>
      <th style = "text-align: right;">jump_wcompilation</th>
      <th style = "text-align: right;">jump_cost</th>
      <th style = "text-align: right;">nlpmodels</th>
      <th style = "text-align: right;">nlpmodels_modelbuild</th>
      <th style = "text-align: right;">nlpmodels_wcompilation</th>
      <th style = "text-align: right;">nlpmodels_cost</th>
      <th style = "text-align: right;">optim</th>
      <th style = "text-align: right;">optim_modelbuild</th>
      <th style = "text-align: right;">optim_wcompilation</th>
      <th style = "text-align: right;">optim_cost</th>
    </tr>
    <tr class = "subheader headerLastRow">
      <th style = "text-align: right;">String</th>
      <th style = "text-align: right;">Int64</th>
      <th style = "text-align: right;">Int64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style = "text-align: right;">pglib_opf_case3_lmbd.m</td>
      <td style = "text-align: right;">24</td>
      <td style = "text-align: right;">28</td>
      <td style = "text-align: right;">0.166827</td>
      <td style = "text-align: right;">5.1699e-5</td>
      <td style = "text-align: right;">0.227767</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0161855</td>
      <td style = "text-align: right;">21.0429</td>
      <td style = "text-align: right;">4.24507</td>
      <td style = "text-align: right;">743.0</td>
      <td style = "text-align: right;">0.00762638</td>
      <td style = "text-align: right;">0.00194973</td>
      <td style = "text-align: right;">0.00848715</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0178365</td>
      <td style = "text-align: right;">0.0134798</td>
      <td style = "text-align: right;">0.0190073</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">2.35037</td>
      <td style = "text-align: right;">0.000292768</td>
      <td style = "text-align: right;">2.40261</td>
      <td style = "text-align: right;">6273.63</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case5_pjm.m</td>
      <td style = "text-align: right;">44</td>
      <td style = "text-align: right;">53</td>
      <td style = "text-align: right;">0.876303</td>
      <td style = "text-align: right;">5.333e-5</td>
      <td style = "text-align: right;">0.888701</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0293974</td>
      <td style = "text-align: right;">1.22185</td>
      <td style = "text-align: right;">5.01985</td>
      <td style = "text-align: right;">16236.7</td>
      <td style = "text-align: right;">0.0123032</td>
      <td style = "text-align: right;">0.00314486</td>
      <td style = "text-align: right;">0.0131673</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0376521</td>
      <td style = "text-align: right;">0.0261532</td>
      <td style = "text-align: right;">0.0394642</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">17.3324</td>
      <td style = "text-align: right;">0.000407517</td>
      <td style = "text-align: right;">19.0955</td>
      <td style = "text-align: right;">77.9548</td>
    </tr>
  </tbody>
</table>
```





## Run the Full Benchmark

```julia
using LibGit2
tmpdir = Base.Filesystem.mktempdir()
LibGit2.clone("https://github.com/power-grid-lib/pglib-opf", tmpdir)
benchmarkfiles = readdir(tmpdir)
benchmarkfiles = benchmarkfiles[endswith(".m").(benchmarkfiles)]
benchmark_datasets = joinpath.((tmpdir,),benchmarkfiles)
```

```
66-element Vector{String}:
 "/tmp/jl_ttujSi/pglib_opf_case10000_goc.m"
 "/tmp/jl_ttujSi/pglib_opf_case10192_epigrids.m"
 "/tmp/jl_ttujSi/pglib_opf_case10480_goc.m"
 "/tmp/jl_ttujSi/pglib_opf_case118_ieee.m"
 "/tmp/jl_ttujSi/pglib_opf_case1354_pegase.m"
 "/tmp/jl_ttujSi/pglib_opf_case13659_pegase.m"
 "/tmp/jl_ttujSi/pglib_opf_case14_ieee.m"
 "/tmp/jl_ttujSi/pglib_opf_case162_ieee_dtc.m"
 "/tmp/jl_ttujSi/pglib_opf_case179_goc.m"
 "/tmp/jl_ttujSi/pglib_opf_case1803_snem.m"
 ⋮
 "/tmp/jl_ttujSi/pglib_opf_case6515_rte.m"
 "/tmp/jl_ttujSi/pglib_opf_case7336_epigrids.m"
 "/tmp/jl_ttujSi/pglib_opf_case73_ieee_rts.m"
 "/tmp/jl_ttujSi/pglib_opf_case78484_epigrids.m"
 "/tmp/jl_ttujSi/pglib_opf_case793_goc.m"
 "/tmp/jl_ttujSi/pglib_opf_case8387_pegase.m"
 "/tmp/jl_ttujSi/pglib_opf_case89_pegase.m"
 "/tmp/jl_ttujSi/pglib_opf_case9241_pegase.m"
 "/tmp/jl_ttujSi/pglib_opf_case9591_goc.m"
```



```julia
timing_data = multidata_multisolver_benchmark(benchmark_datasets)
```

```
file = "/tmp/jl_ttujSi/pglib_opf_case10000_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case10192_epigrids.m"
file = "/tmp/jl_ttujSi/pglib_opf_case10480_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case118_ieee.m"
file = "/tmp/jl_ttujSi/pglib_opf_case1354_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case13659_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case14_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      180
Number of nonzeros in inequality constraint Jacobian.:      232
Number of nonzeros in Lagrangian Hessian.............:      858

Total number of variables............................:       31
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       18
                     variables with only upper bounds:        0
Total number of equality constraints.................:       23
Total number of inequality constraints...............:       80
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       80

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  5.1286896e+02 4.78e-01 6.04e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1 -2.7915938e+03 1.28e-01 2.89e+01  -1.0 2.57e-01    -  5.76e-01 7.84e-01f  1
   2 -1.0848401e+04 1.11e+00 2.29e+01  -1.0 2.71e+00    -  9.96e-01 8.58e-01f  1
   3 -1.0721797e+04 6.55e-01 1.42e+01  -1.0 8.99e+00    -  2.11e-01 4.38e-01h  1
   4 -1.0907759e+04 2.92e-01 6.65e+00  -1.0 8.23e+00    -  6.15e-01 5.60e-01f  1
   5 -1.0340414e+04 2.45e-03 3.35e-01  -1.0 8.90e-01    -  1.00e+00 1.00e+00h  1
   6 -1.0744091e+04 1.76e-03 6.00e-01  -1.7 1.01e+00    -  1.00e+00 9.08e-01f  1
   7 -1.0710587e+04 2.91e-05 1.87e-02  -1.7 6.98e-02    -  1.00e+00 1.00e+00h  1
   8 -1.0777420e+04 3.08e-04 2.78e-02  -3.8 7.68e-02    -  8.32e-01 8.19e-01f  1
   9 -1.0804382e+04 2.42e-04 1.25e-02  -3.8 1.40e-01    -  8.72e-01 9.25e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10 -1.0806089e+04 9.05e-05 1.10e-03  -3.8 4.31e-02    -  1.00e+00 1.00e+00h  1
  11 -1.0806759e+04 9.66e-06 1.54e-03  -5.7 1.15e-02    -  9.95e-01 9.79e-01h  1
  12 -1.0806757e+04 1.90e-07 2.29e-06  -5.7 1.74e-03    -  1.00e+00 1.00e+00h  1
  13 -1.0806766e+04 6.75e-10 7.53e-09  -8.6 1.17e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 13

                                   (scaled)                 (unscaled)
Objective...............:  -1.5299690122878349e+01   -1.0806765799751756e+04
Dual infeasibility......:   7.5309998237571563e-09    5.3194378892429578e-06
Constraint violation....:   6.7505911971466048e-10    6.7505911971466048e-10
Variable bound violation:   3.2451651144072002e-08    3.2451651144072002e-08
Complementarity.........:   6.7185777630203695e-09    4.7455926106509745e-06
Overall NLP error.......:   7.5309998237571563e-09    5.3194378892429578e-06


Number of objective function evaluations             = 14
Number of objective gradient evaluations             = 14
Number of equality constraint evaluations            = 14
Number of inequality constraint evaluations          = 14
Number of equality constraint Jacobian evaluations   = 14
Number of inequality constraint Jacobian evaluations = 14
Number of Lagrangian Hessian evaluations             = 13
Total seconds in IPOPT                               = 19.444

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      180
Number of nonzeros in inequality constraint Jacobian.:      232
Number of nonzeros in Lagrangian Hessian.............:      858

Total number of variables............................:       31
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       18
                     variables with only upper bounds:        0
Total number of equality constraints.................:       23
Total number of inequality constraints...............:       80
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       80

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  5.1286896e+02 4.78e-01 6.04e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1 -2.7915938e+03 1.28e-01 2.89e+01  -1.0 2.57e-01    -  5.76e-01 7.84e-01f  1
   2 -1.0848401e+04 1.11e+00 2.29e+01  -1.0 2.71e+00    -  9.96e-01 8.58e-01f  1
   3 -1.0721797e+04 6.55e-01 1.42e+01  -1.0 8.99e+00    -  2.11e-01 4.38e-01h  1
   4 -1.0907759e+04 2.92e-01 6.65e+00  -1.0 8.23e+00    -  6.15e-01 5.60e-01f  1
   5 -1.0340414e+04 2.45e-03 3.35e-01  -1.0 8.90e-01    -  1.00e+00 1.00e+00h  1
   6 -1.0744091e+04 1.76e-03 6.00e-01  -1.7 1.01e+00    -  1.00e+00 9.08e-01f  1
   7 -1.0710587e+04 2.91e-05 1.87e-02  -1.7 6.98e-02    -  1.00e+00 1.00e+00h  1
   8 -1.0777420e+04 3.08e-04 2.78e-02  -3.8 7.68e-02    -  8.32e-01 8.19e-01f  1
   9 -1.0804382e+04 2.42e-04 1.25e-02  -3.8 1.40e-01    -  8.72e-01 9.25e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10 -1.0806089e+04 9.05e-05 1.10e-03  -3.8 4.31e-02    -  1.00e+00 1.00e+00h  1
  11 -1.0806759e+04 9.66e-06 1.54e-03  -5.7 1.15e-02    -  9.95e-01 9.79e-01h  1
  12 -1.0806757e+04 1.90e-07 2.29e-06  -5.7 1.74e-03    -  1.00e+00 1.00e+00h  1
  13 -1.0806766e+04 6.75e-10 7.53e-09  -8.6 1.17e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 13

                                   (scaled)                 (unscaled)
Objective...............:  -1.5299690122878349e+01   -1.0806765799751756e+04
Dual infeasibility......:   7.5309998237571563e-09    5.3194378892429578e-06
Constraint violation....:   6.7505911971466048e-10    6.7505911971466048e-10
Variable bound violation:   3.2451651144072002e-08    3.2451651144072002e-08
Complementarity.........:   6.7185777630203695e-09    4.7455926106509745e-06
Overall NLP error.......:   7.5309998237571563e-09    5.3194378892429578e-06


Number of objective function evaluations             = 14
Number of objective gradient evaluations             = 14
Number of equality constraint evaluations            = 14
Number of inequality constraint evaluations          = 14
Number of equality constraint Jacobian evaluations   = 14
Number of inequality constraint Jacobian evaluations = 14
Number of Lagrangian Hessian evaluations             = 13
Total seconds in IPOPT                               = 0.019

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case162_ieee_dtc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case179_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case1803_snem.m"
file = "/tmp/jl_ttujSi/pglib_opf_case1888_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case19402_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case1951_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case197_snem.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2000_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case200_activ.m"
file = "/tmp/jl_ttujSi/pglib_opf_case20758_epigrids.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2312_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2383wp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case240_pserc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case24464_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case24_ieee_rts.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      264
Number of nonzeros in inequality constraint Jacobian.:      446
Number of nonzeros in Lagrangian Hessian.............:     1758

Total number of variables............................:       83
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       60
                     variables with only upper bounds:        0
Total number of equality constraints.................:       33
Total number of inequality constraints...............:      152
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      152

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  8.1648801e+04 1.94e+00 1.08e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  9.7063010e+04 1.17e+00 7.78e+01  -1.0 3.49e-01    -  2.81e-01 4.01e-01f  1
   2  1.0787562e+05 1.66e-01 1.28e+02  -1.0 2.27e+00    -  6.45e-01 8.48e-01f  1
   3  9.1196524e+04 2.40e-02 2.76e+01  -1.0 2.39e+00    -  1.00e+00 8.72e-01f  1
   4  8.5295147e+04 2.80e-02 1.36e+01  -1.0 5.16e+00    -  1.00e+00 5.14e-01f  1
   5  7.8741472e+04 3.85e-02 2.04e+00  -1.0 2.47e+00    -  1.00e+00 1.00e+00f  1
   6  7.2334034e+04 2.14e-02 8.21e-02  -1.0 3.63e+00    -  1.00e+00 1.00e+00h  1
   7  6.3455042e+04 3.51e-02 9.18e-01  -1.7 5.88e+00    -  1.00e+00 1.00e+00h  1
   8  5.2492373e+04 7.52e-02 1.15e-01  -1.7 1.00e+01    -  1.00e+00 1.00e+00h  1
   9  5.4318878e+04 2.85e-03 1.63e-02  -1.7 1.68e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  4.7919475e+04 1.63e-01 1.23e-01  -2.5 6.91e+00    -  9.37e-01 8.96e-01h  1
  11  4.5592531e+04 3.37e-02 3.55e-02  -2.5 6.66e+00    -  1.00e+00 8.81e-01h  1
  12  4.5680009e+04 7.85e-04 9.35e-03  -2.5 3.23e-01    -  1.00e+00 1.00e+00h  1
  13  4.4314887e+04 2.65e-03 4.61e-02  -3.8 2.63e+00    -  8.20e-01 8.13e-01h  1
  14  4.3834426e+04 2.27e-03 2.64e-02  -3.8 9.57e-01    -  9.20e-01 1.00e+00h  1
  15  4.3816708e+04 1.14e-03 7.53e-03  -3.8 7.35e-01    -  1.00e+00 1.00e+00h  1
  16  4.3814995e+04 2.44e-05 3.50e-04  -3.8 9.80e-02    -  1.00e+00 1.00e+00h  1
  17  4.3703915e+04 8.21e-04 1.06e-02  -5.7 6.35e-01    -  8.21e-01 9.45e-01h  1
  18  4.3691595e+04 7.22e-03 4.13e-03  -5.7 5.79e-01    -  9.69e-01 9.27e-01h  1
  19  4.3690790e+04 6.71e-05 5.51e-04  -5.7 2.19e-01    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  4.3690872e+04 1.90e-06 7.51e-06  -5.7 3.67e-02    -  1.00e+00 1.00e+00h  1
  21  4.3690871e+04 2.45e-08 2.64e-07  -5.7 4.00e-03    -  1.00e+00 1.00e+00h  1
  22  4.3689290e+04 9.55e-06 2.06e-05  -8.6 5.73e-02    -  9.81e-01 9.90e-01h  1
  23  4.3689274e+04 1.58e-07 1.41e-06  -8.6 1.04e-02    -  1.00e+00 1.00e+00h  1
  24  4.3689274e+04 1.09e-09 1.18e-08  -8.6 8.45e-04    -  1.00e+00 1.00e+00h  1
  25  4.3689272e+04 1.79e-11 3.67e-11  -9.5 1.11e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 25

                                   (scaled)                 (unscaled)
Objective...............:   1.4606218940836955e+00    4.3689272073857268e+04
Dual infeasibility......:   3.6724401297760778e-11    1.0984789195248430e-06
Constraint violation....:   1.7925216866387927e-11    1.7925216866387927e-11
Variable bound violation:   2.9379617849656370e-08    2.9379617849656370e-08
Complementarity.........:   3.0539679157473679e-10    9.1348511011893753e-06
Overall NLP error.......:   3.0539679157473679e-10    9.1348511011893753e-06


Number of objective function evaluations             = 26
Number of objective gradient evaluations             = 26
Number of equality constraint evaluations            = 26
Number of inequality constraint evaluations          = 26
Number of equality constraint Jacobian evaluations   = 26
Number of inequality constraint Jacobian evaluations = 26
Number of Lagrangian Hessian evaluations             = 25
Total seconds in IPOPT                               = 117.897

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      264
Number of nonzeros in inequality constraint Jacobian.:      446
Number of nonzeros in Lagrangian Hessian.............:     1758

Total number of variables............................:       83
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       60
                     variables with only upper bounds:        0
Total number of equality constraints.................:       33
Total number of inequality constraints...............:      152
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      152

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  8.1648801e+04 1.94e+00 1.08e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  9.7063010e+04 1.17e+00 7.78e+01  -1.0 3.49e-01    -  2.81e-01 4.01e-01f  1
   2  1.0787562e+05 1.66e-01 1.28e+02  -1.0 2.27e+00    -  6.45e-01 8.48e-01f  1
   3  9.1196524e+04 2.40e-02 2.76e+01  -1.0 2.39e+00    -  1.00e+00 8.72e-01f  1
   4  8.5295147e+04 2.80e-02 1.36e+01  -1.0 5.16e+00    -  1.00e+00 5.14e-01f  1
   5  7.8741472e+04 3.85e-02 2.04e+00  -1.0 2.47e+00    -  1.00e+00 1.00e+00f  1
   6  7.2334034e+04 2.14e-02 8.21e-02  -1.0 3.63e+00    -  1.00e+00 1.00e+00h  1
   7  6.3455042e+04 3.51e-02 9.18e-01  -1.7 5.88e+00    -  1.00e+00 1.00e+00h  1
   8  5.2492373e+04 7.52e-02 1.15e-01  -1.7 1.00e+01    -  1.00e+00 1.00e+00h  1
   9  5.4318878e+04 2.85e-03 1.63e-02  -1.7 1.68e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  4.7919475e+04 1.63e-01 1.23e-01  -2.5 6.91e+00    -  9.37e-01 8.96e-01h  1
  11  4.5592531e+04 3.37e-02 3.55e-02  -2.5 6.66e+00    -  1.00e+00 8.81e-01h  1
  12  4.5680009e+04 7.85e-04 9.35e-03  -2.5 3.23e-01    -  1.00e+00 1.00e+00h  1
  13  4.4314887e+04 2.65e-03 4.61e-02  -3.8 2.63e+00    -  8.20e-01 8.13e-01h  1
  14  4.3834426e+04 2.27e-03 2.64e-02  -3.8 9.57e-01    -  9.20e-01 1.00e+00h  1
  15  4.3816708e+04 1.14e-03 7.53e-03  -3.8 7.35e-01    -  1.00e+00 1.00e+00h  1
  16  4.3814995e+04 2.44e-05 3.50e-04  -3.8 9.80e-02    -  1.00e+00 1.00e+00h  1
  17  4.3703915e+04 8.21e-04 1.06e-02  -5.7 6.35e-01    -  8.21e-01 9.45e-01h  1
  18  4.3691595e+04 7.22e-03 4.13e-03  -5.7 5.79e-01    -  9.69e-01 9.27e-01h  1
  19  4.3690790e+04 6.71e-05 5.51e-04  -5.7 2.19e-01    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  4.3690872e+04 1.90e-06 7.51e-06  -5.7 3.67e-02    -  1.00e+00 1.00e+00h  1
  21  4.3690871e+04 2.45e-08 2.64e-07  -5.7 4.00e-03    -  1.00e+00 1.00e+00h  1
  22  4.3689290e+04 9.55e-06 2.06e-05  -8.6 5.73e-02    -  9.81e-01 9.90e-01h  1
  23  4.3689274e+04 1.58e-07 1.41e-06  -8.6 1.04e-02    -  1.00e+00 1.00e+00h  1
  24  4.3689274e+04 1.09e-09 1.18e-08  -8.6 8.45e-04    -  1.00e+00 1.00e+00h  1
  25  4.3689272e+04 1.79e-11 3.67e-11  -9.5 1.11e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 25

                                   (scaled)                 (unscaled)
Objective...............:   1.4606218940836955e+00    4.3689272073857268e+04
Dual infeasibility......:   3.6724401297760778e-11    1.0984789195248430e-06
Constraint violation....:   1.7925216866387927e-11    1.7925216866387927e-11
Variable bound violation:   2.9379617849656370e-08    2.9379617849656370e-08
Complementarity.........:   3.0539679157473679e-10    9.1348511011893753e-06
Overall NLP error.......:   3.0539679157473679e-10    9.1348511011893753e-06


Number of objective function evaluations             = 26
Number of objective gradient evaluations             = 26
Number of equality constraint evaluations            = 26
Number of inequality constraint evaluations          = 26
Number of equality constraint Jacobian evaluations   = 26
Number of inequality constraint Jacobian evaluations = 26
Number of Lagrangian Hessian evaluations             = 25
Total seconds in IPOPT                               = 0.069

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case2736sp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2737sop_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2742_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2746wop_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2746wp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2848_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2853_sdet.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2868_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case2869_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case30000_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case300_ieee.m"
file = "/tmp/jl_ttujSi/pglib_opf_case3012wp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case3022_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case30_as.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      396
Number of nonzeros in inequality constraint Jacobian.:      481
Number of nonzeros in Lagrangian Hessian.............:     1893

Total number of variables............................:       62
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       33
                     variables with only upper bounds:        0
Total number of equality constraints.................:       51
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  7.6240779e+02 2.28e-01 1.49e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.9224406e+02 2.78e-02 1.43e+01  -1.0 1.90e-01    -  3.40e-01 8.76e-01H  1
   2  8.4855325e+02 5.29e-03 4.38e+00  -1.0 2.46e-01    -  1.00e+00 8.80e-01f  1
   3  8.2262940e+02 8.92e-03 8.52e-01  -1.0 4.92e-01    -  1.00e+00 1.00e+00f  1
   4  8.1121257e+02 4.74e-04 4.47e-01  -1.7 2.48e-01    -  1.00e+00 1.00e+00h  1
   5  8.0707413e+02 9.35e-05 1.22e-01  -1.7 8.85e-02    -  1.00e+00 1.00e+00h  1
   6  8.0482039e+02 1.26e-04 1.49e-01  -2.5 1.21e-01    -  9.74e-01 1.00e+00h  1
   7  8.0389326e+02 4.80e-05 1.55e-02  -2.5 5.59e-02    -  1.00e+00 1.00e+00h  1
   8  8.0343678e+02 5.44e-05 2.98e-02  -3.8 5.16e-02    -  9.34e-01 1.00e+00h  1
   9  8.0319342e+02 1.29e-04 3.24e-03  -3.8 1.73e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.0317199e+02 5.39e-06 9.13e-05  -3.8 3.72e-03    -  1.00e+00 1.00e+00h  1
  11  8.0311973e+02 1.88e-05 7.62e-04  -5.7 7.16e-03    -  9.12e-01 9.76e-01h  1
  12  8.0311674e+02 1.97e-06 6.03e-05  -5.7 4.38e-03    -  1.00e+00 1.00e+00h  1
  13  8.0311848e+02 8.07e-09 2.03e-07  -5.7 4.43e-04    -  1.00e+00 1.00e+00h  1
  14  8.0311794e+02 5.83e-09 2.37e-07  -8.6 2.26e-04    -  9.99e-01 9.99e-01h  1
  15  8.0311795e+02 4.56e-13 1.32e-11  -8.6 2.60e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   5.5711615424297420e+00    8.0311794653018751e+02
Dual infeasibility......:   1.3169909607313457e-11    1.8985252319933346e-09
Constraint violation....:   4.5563552930616424e-13    4.5563552930616424e-13
Variable bound violation:   4.5789416702746166e-09    4.5789416702746166e-09
Complementarity.........:   2.5067419061931308e-09    3.6136259859824079e-07
Overall NLP error.......:   2.5067419061931308e-09    3.6136259859824079e-07


Number of objective function evaluations             = 17
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 17
Number of inequality constraint evaluations          = 17
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 73.176

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      396
Number of nonzeros in inequality constraint Jacobian.:      481
Number of nonzeros in Lagrangian Hessian.............:     1893

Total number of variables............................:       62
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       33
                     variables with only upper bounds:        0
Total number of equality constraints.................:       51
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  7.6240779e+02 2.28e-01 1.49e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.9224406e+02 2.78e-02 1.43e+01  -1.0 1.90e-01    -  3.40e-01 8.76e-01H  1
   2  8.4855325e+02 5.29e-03 4.38e+00  -1.0 2.46e-01    -  1.00e+00 8.80e-01f  1
   3  8.2262940e+02 8.92e-03 8.52e-01  -1.0 4.92e-01    -  1.00e+00 1.00e+00f  1
   4  8.1121257e+02 4.74e-04 4.47e-01  -1.7 2.48e-01    -  1.00e+00 1.00e+00h  1
   5  8.0707413e+02 9.35e-05 1.22e-01  -1.7 8.85e-02    -  1.00e+00 1.00e+00h  1
   6  8.0482039e+02 1.26e-04 1.49e-01  -2.5 1.21e-01    -  9.74e-01 1.00e+00h  1
   7  8.0389326e+02 4.80e-05 1.55e-02  -2.5 5.59e-02    -  1.00e+00 1.00e+00h  1
   8  8.0343678e+02 5.44e-05 2.98e-02  -3.8 5.16e-02    -  9.34e-01 1.00e+00h  1
   9  8.0319342e+02 1.29e-04 3.24e-03  -3.8 1.73e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.0317199e+02 5.39e-06 9.13e-05  -3.8 3.72e-03    -  1.00e+00 1.00e+00h  1
  11  8.0311973e+02 1.88e-05 7.62e-04  -5.7 7.16e-03    -  9.12e-01 9.76e-01h  1
  12  8.0311674e+02 1.97e-06 6.03e-05  -5.7 4.38e-03    -  1.00e+00 1.00e+00h  1
  13  8.0311848e+02 8.07e-09 2.03e-07  -5.7 4.43e-04    -  1.00e+00 1.00e+00h  1
  14  8.0311794e+02 5.83e-09 2.37e-07  -8.6 2.26e-04    -  9.99e-01 9.99e-01h  1
  15  8.0311795e+02 4.56e-13 1.32e-11  -8.6 2.60e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   5.5711615424297420e+00    8.0311794653018751e+02
Dual infeasibility......:   1.3169909607313457e-11    1.8985252319933346e-09
Constraint violation....:   4.5563552930616424e-13    4.5563552930616424e-13
Variable bound violation:   4.5789416702746166e-09    4.5789416702746166e-09
Complementarity.........:   2.5067419061931308e-09    3.6136259859824079e-07
Overall NLP error.......:   2.5067419061931308e-09    3.6136259859824079e-07


Number of objective function evaluations             = 17
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 17
Number of inequality constraint evaluations          = 17
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.049

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case30_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      394
Number of nonzeros in inequality constraint Jacobian.:      480
Number of nonzeros in Lagrangian Hessian.............:     1843

Total number of variables............................:       60
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       31
                     variables with only upper bounds:        0
Total number of equality constraints.................:       51
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.1323549e+03 3.00e-01 1.21e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1 -4.3228819e+03 6.85e-02 3.39e+01  -1.0 1.48e-01    -  9.07e-01 7.63e-01f  1
   2 -1.7306335e+04 1.81e+00 1.56e+01  -1.0 2.55e+00    -  1.00e+00 7.01e-01f  1
   3 -1.4734828e+04 7.45e-01 8.73e+00  -1.0 9.53e-01    -  8.84e-01 5.59e-01h  1
   4 -1.2094588e+04 7.60e-03 3.28e+00  -1.0 2.30e-01    -  1.00e+00 1.00e+00h  1
   5 -1.1798662e+04 9.28e-05 1.82e-01  -1.0 3.92e-02    -  1.00e+00 1.00e+00h  1
   6 -1.2617514e+04 4.38e-03 6.67e-01  -2.5 2.48e-01    -  8.15e-01 6.88e-01f  1
   7 -1.2771000e+04 3.63e-03 2.46e+00  -2.5 2.06e-01    -  8.09e-01 9.78e-01h  1
   8 -1.2828222e+04 1.31e-03 5.86e-01  -2.5 1.96e-01    -  1.00e+00 1.00e+00h  1
   9 -1.2815297e+04 4.54e-05 1.31e-03  -2.5 3.70e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10 -1.2841350e+04 1.09e-04 4.14e-02  -3.8 4.60e-02    -  9.83e-01 1.00e+00h  1
  11 -1.2841024e+04 1.58e-05 4.18e-04  -3.8 1.52e-02    -  1.00e+00 1.00e+00h  1
  12 -1.2842371e+04 5.05e-06 5.10e-04  -5.7 8.21e-03    -  9.97e-01 1.00e+00h  1
  13 -1.2842374e+04 2.54e-07 2.60e-06  -5.7 2.18e-03    -  1.00e+00 1.00e+00h  1
  14 -1.2842390e+04 4.69e-09 5.58e-08  -8.6 2.65e-04    -  1.00e+00 1.00e+00h  1
  15 -1.2842390e+04 3.43e-13 2.67e-12  -8.6 2.55e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:  -9.7624838273460757e+00   -1.2842390078812239e+04
Dual infeasibility......:   2.6748029600747472e-12    3.5186601693536225e-09
Constraint violation....:   3.4283687000424834e-13    3.4283687000424834e-13
Variable bound violation:   8.1627367087833136e-09    8.1627367087833136e-09
Complementarity.........:   2.5097537167041501e-09    3.3015405507130615e-06
Overall NLP error.......:   2.5097537167041501e-09    3.3015405507130615e-06


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 78.603

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      394
Number of nonzeros in inequality constraint Jacobian.:      480
Number of nonzeros in Lagrangian Hessian.............:     1843

Total number of variables............................:       60
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       31
                     variables with only upper bounds:        0
Total number of equality constraints.................:       51
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.1323549e+03 3.00e-01 1.21e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1 -4.3228819e+03 6.85e-02 3.39e+01  -1.0 1.48e-01    -  9.07e-01 7.63e-01f  1
   2 -1.7306335e+04 1.81e+00 1.56e+01  -1.0 2.55e+00    -  1.00e+00 7.01e-01f  1
   3 -1.4734828e+04 7.45e-01 8.73e+00  -1.0 9.53e-01    -  8.84e-01 5.59e-01h  1
   4 -1.2094588e+04 7.60e-03 3.28e+00  -1.0 2.30e-01    -  1.00e+00 1.00e+00h  1
   5 -1.1798662e+04 9.28e-05 1.82e-01  -1.0 3.92e-02    -  1.00e+00 1.00e+00h  1
   6 -1.2617514e+04 4.38e-03 6.67e-01  -2.5 2.48e-01    -  8.15e-01 6.88e-01f  1
   7 -1.2771000e+04 3.63e-03 2.46e+00  -2.5 2.06e-01    -  8.09e-01 9.78e-01h  1
   8 -1.2828222e+04 1.31e-03 5.86e-01  -2.5 1.96e-01    -  1.00e+00 1.00e+00h  1
   9 -1.2815297e+04 4.54e-05 1.31e-03  -2.5 3.70e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10 -1.2841350e+04 1.09e-04 4.14e-02  -3.8 4.60e-02    -  9.83e-01 1.00e+00h  1
  11 -1.2841024e+04 1.58e-05 4.18e-04  -3.8 1.52e-02    -  1.00e+00 1.00e+00h  1
  12 -1.2842371e+04 5.05e-06 5.10e-04  -5.7 8.21e-03    -  9.97e-01 1.00e+00h  1
  13 -1.2842374e+04 2.54e-07 2.60e-06  -5.7 2.18e-03    -  1.00e+00 1.00e+00h  1
  14 -1.2842390e+04 4.69e-09 5.58e-08  -8.6 2.65e-04    -  1.00e+00 1.00e+00h  1
  15 -1.2842390e+04 3.43e-13 2.67e-12  -8.6 2.55e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:  -9.7624838273460757e+00   -1.2842390078812239e+04
Dual infeasibility......:   2.6748029600747472e-12    3.5186601693536225e-09
Constraint violation....:   3.4283687000424834e-13    3.4283687000424834e-13
Variable bound violation:   8.1627367087833136e-09    8.1627367087833136e-09
Complementarity.........:   2.5097537167041501e-09    3.3015405507130615e-06
Overall NLP error.......:   2.5097537167041501e-09    3.3015405507130615e-06


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.050

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case3120sp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case3375wp_k.m"
file = "/tmp/jl_ttujSi/pglib_opf_case3970_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case39_epri.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      486
Number of nonzeros in inequality constraint Jacobian.:      555
Number of nonzeros in Lagrangian Hessian.............:     2201

Total number of variables............................:       86
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       48
                     variables with only upper bounds:        0
Total number of equality constraints.................:       67
Total number of inequality constraints...............:      184
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      184

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2569221e+03 1.10e+01 4.31e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.3839990e+03 1.09e+01 1.49e+02  -1.0 3.74e+00   2.0 5.47e-03 1.03e-02H  1
   2  5.6510553e+03 1.07e+01 1.49e+02  -1.0 9.49e+00    -  7.02e-03 2.43e-02h  6
   3  1.2127496e+04 1.02e+01 1.52e+02  -1.0 2.62e+01    -  2.06e-02 4.28e-02h  5
   4  8.5568506e+04 5.41e+00 1.70e+03  -1.0 4.53e+01    -  2.27e-02 4.76e-01H  1
   5  1.0855289e+05 3.76e+00 1.39e+03  -1.0 2.24e+01    -  2.35e-01 3.08e-01H  1
   6  1.1833112e+05 2.98e+00 1.19e+03  -1.0 2.96e+01    -  3.90e-01 2.07e-01h  1
   7  1.5315294e+05 1.66e-02 1.50e+03  -1.0 3.71e+01    -  6.93e-01 1.00e+00H  1
   8  1.5180722e+05 9.11e-03 3.46e+01  -1.0 4.45e+00    -  7.96e-01 1.00e+00f  1
   9  1.4014722e+05 7.84e-02 9.05e+00  -1.0 3.38e+01    -  6.67e-01 1.00e+00F  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.3367300e+05 8.03e-02 4.94e+00  -1.0 3.76e+01    -  2.74e-01 5.68e-01F  1
  11  1.3316419e+05 7.96e-02 4.85e+00  -1.0 8.24e+02    -  3.11e-02 8.77e-03h  5
  12  1.3293814e+05 7.96e-02 4.83e+00  -1.0 4.54e+04    -  4.63e-04 6.66e-05f  6
  13  1.3274159e+05 7.96e-02 4.67e+00  -1.0 4.51e+03    -  1.24e-02 7.22e-04f  6
  14  1.3262305e+05 4.95e-02 2.82e+03  -1.0 2.43e+00   1.5 2.10e-01 3.74e-01h  1
  15  1.3267819e+05 4.14e-02 2.33e+03  -1.0 1.93e+00   1.0 1.00e+00 1.63e-01h  1
  16  1.3292293e+05 1.08e-02 4.57e+02  -1.0 1.83e+00   0.6 7.45e-01 7.40e-01h  1
  17  1.3305910e+05 2.82e-03 1.18e+02  -1.0 5.76e-01   0.1 6.41e-01 1.00e+00f  1
  18  1.3081535e+05 7.90e-03 5.09e+01  -1.0 9.08e+00    -  5.33e-01 1.00e+00f  1
  19  1.2429782e+05 1.76e+01 1.91e+01  -1.0 1.01e+02    -  4.41e-01 6.31e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.2587931e+05 5.68e-02 3.05e+00  -1.0 3.13e+01    -  7.57e-01 1.00e+00h  1
  21  1.2497495e+05 1.24e-01 6.27e-01  -1.0 1.72e+01    -  1.00e+00 1.00e+00h  1
  22  1.2536907e+05 1.86e-02 7.99e-02  -1.0 4.65e+00    -  1.00e+00 1.00e+00h  1
  23  1.2538778e+05 3.74e-04 4.57e-03  -1.0 1.06e+00    -  1.00e+00 1.00e+00h  1
  24  1.2096111e+05 1.03e-01 1.97e+00  -2.5 2.53e+01    -  7.85e-01 9.23e-01f  1
  25  1.1978632e+05 5.05e-02 9.29e-01  -2.5 2.30e+01    -  8.49e-01 6.17e-01h  1
  26  1.1949451e+05 2.03e-01 3.56e-01  -2.5 8.81e+00    -  9.28e-01 7.77e-01h  1
  27  1.1953594e+05 6.10e-04 3.33e-02  -2.5 3.33e+00    -  1.00e+00 1.00e+00h  1
  28  1.1954786e+05 5.26e-06 1.09e-04  -2.5 2.35e-01    -  1.00e+00 1.00e+00h  1
  29  1.1944245e+05 2.67e-04 4.76e-01  -3.8 2.50e+00    -  1.00e+00 7.75e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  1.1942406e+05 5.92e-05 3.37e-03  -3.8 9.67e-01    -  1.00e+00 1.00e+00h  1
  31  1.1942476e+05 2.99e-06 2.03e-04  -3.8 1.90e-01    -  1.00e+00 1.00e+00h  1
  32  1.1941948e+05 6.68e-06 2.58e-01  -5.7 3.47e-01    -  1.00e+00 8.73e-01h  1
  33  1.1941887e+05 3.37e-06 2.09e-04  -5.7 1.97e-01    -  1.00e+00 1.00e+00h  1
  34  1.1941888e+05 1.53e-08 3.68e-06  -5.7 1.38e-02    -  1.00e+00 1.00e+00h  1
  35  1.1941881e+05 4.14e-09 9.94e-04  -8.6 7.48e-03    -  9.81e-01 9.98e-01h  1
  36  1.1941881e+05 8.24e-13 1.61e-10  -8.6 8.30e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 36

                                   (scaled)                 (unscaled)
Objective...............:   5.2199088989616655e+01    1.1941880845126325e+05
Dual infeasibility......:   1.6093538172775878e-10    3.6818097586724373e-07
Constraint violation....:   4.9012305433847061e-13    8.2422957348171611e-13
Variable bound violation:   2.7796368318178111e-08    2.7796368318178111e-08
Complementarity.........:   2.5089191802217999e-09    5.7397963221581475e-06
Overall NLP error.......:   2.5089191802217999e-09    5.7397963221581475e-06


Number of objective function evaluations             = 75
Number of objective gradient evaluations             = 37
Number of equality constraint evaluations            = 75
Number of inequality constraint evaluations          = 75
Number of equality constraint Jacobian evaluations   = 37
Number of inequality constraint Jacobian evaluations = 37
Number of Lagrangian Hessian evaluations             = 36
Total seconds in IPOPT                               = 165.389

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      486
Number of nonzeros in inequality constraint Jacobian.:      555
Number of nonzeros in Lagrangian Hessian.............:     2201

Total number of variables............................:       86
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       48
                     variables with only upper bounds:        0
Total number of equality constraints.................:       67
Total number of inequality constraints...............:      184
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      184

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2569221e+03 1.10e+01 4.31e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.3839990e+03 1.09e+01 1.49e+02  -1.0 3.74e+00   2.0 5.47e-03 1.03e-02H  1
   2  5.6510553e+03 1.07e+01 1.49e+02  -1.0 9.49e+00    -  7.02e-03 2.43e-02h  6
   3  1.2127496e+04 1.02e+01 1.52e+02  -1.0 2.62e+01    -  2.06e-02 4.28e-02h  5
   4  8.5568506e+04 5.41e+00 1.70e+03  -1.0 4.53e+01    -  2.27e-02 4.76e-01H  1
   5  1.0855289e+05 3.76e+00 1.39e+03  -1.0 2.24e+01    -  2.35e-01 3.08e-01H  1
   6  1.1833112e+05 2.98e+00 1.19e+03  -1.0 2.96e+01    -  3.90e-01 2.07e-01h  1
   7  1.5315294e+05 1.66e-02 1.50e+03  -1.0 3.71e+01    -  6.93e-01 1.00e+00H  1
   8  1.5180722e+05 9.11e-03 3.46e+01  -1.0 4.45e+00    -  7.96e-01 1.00e+00f  1
   9  1.4014722e+05 7.84e-02 9.05e+00  -1.0 3.38e+01    -  6.67e-01 1.00e+00F  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.3367300e+05 8.03e-02 4.94e+00  -1.0 3.76e+01    -  2.74e-01 5.68e-01F  1
  11  1.3316419e+05 7.96e-02 4.85e+00  -1.0 8.24e+02    -  3.11e-02 8.77e-03h  5
  12  1.3293814e+05 7.96e-02 4.83e+00  -1.0 4.54e+04    -  4.63e-04 6.66e-05f  6
  13  1.3274159e+05 7.96e-02 4.67e+00  -1.0 4.51e+03    -  1.24e-02 7.22e-04f  6
  14  1.3262305e+05 4.95e-02 2.82e+03  -1.0 2.43e+00   1.5 2.10e-01 3.74e-01h  1
  15  1.3267819e+05 4.14e-02 2.33e+03  -1.0 1.93e+00   1.0 1.00e+00 1.63e-01h  1
  16  1.3292293e+05 1.08e-02 4.57e+02  -1.0 1.83e+00   0.6 7.45e-01 7.40e-01h  1
  17  1.3305910e+05 2.82e-03 1.18e+02  -1.0 5.76e-01   0.1 6.41e-01 1.00e+00f  1
  18  1.3081535e+05 7.90e-03 5.09e+01  -1.0 9.08e+00    -  5.33e-01 1.00e+00f  1
  19  1.2429782e+05 1.76e+01 1.91e+01  -1.0 1.01e+02    -  4.41e-01 6.31e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.2587931e+05 5.68e-02 3.05e+00  -1.0 3.13e+01    -  7.57e-01 1.00e+00h  1
  21  1.2497495e+05 1.24e-01 6.27e-01  -1.0 1.72e+01    -  1.00e+00 1.00e+00h  1
  22  1.2536907e+05 1.86e-02 7.99e-02  -1.0 4.65e+00    -  1.00e+00 1.00e+00h  1
  23  1.2538778e+05 3.74e-04 4.57e-03  -1.0 1.06e+00    -  1.00e+00 1.00e+00h  1
  24  1.2096111e+05 1.03e-01 1.97e+00  -2.5 2.53e+01    -  7.85e-01 9.23e-01f  1
  25  1.1978632e+05 5.05e-02 9.29e-01  -2.5 2.30e+01    -  8.49e-01 6.17e-01h  1
  26  1.1949451e+05 2.03e-01 3.56e-01  -2.5 8.81e+00    -  9.28e-01 7.77e-01h  1
  27  1.1953594e+05 6.10e-04 3.33e-02  -2.5 3.33e+00    -  1.00e+00 1.00e+00h  1
  28  1.1954786e+05 5.26e-06 1.09e-04  -2.5 2.35e-01    -  1.00e+00 1.00e+00h  1
  29  1.1944245e+05 2.67e-04 4.76e-01  -3.8 2.50e+00    -  1.00e+00 7.75e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  1.1942406e+05 5.92e-05 3.37e-03  -3.8 9.67e-01    -  1.00e+00 1.00e+00h  1
  31  1.1942476e+05 2.99e-06 2.03e-04  -3.8 1.90e-01    -  1.00e+00 1.00e+00h  1
  32  1.1941948e+05 6.68e-06 2.58e-01  -5.7 3.47e-01    -  1.00e+00 8.73e-01h  1
  33  1.1941887e+05 3.37e-06 2.09e-04  -5.7 1.97e-01    -  1.00e+00 1.00e+00h  1
  34  1.1941888e+05 1.53e-08 3.68e-06  -5.7 1.38e-02    -  1.00e+00 1.00e+00h  1
  35  1.1941881e+05 4.14e-09 9.94e-04  -8.6 7.48e-03    -  9.81e-01 9.98e-01h  1
  36  1.1941881e+05 8.24e-13 1.61e-10  -8.6 8.30e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 36

                                   (scaled)                 (unscaled)
Objective...............:   5.2199088989616655e+01    1.1941880845126325e+05
Dual infeasibility......:   1.6093538172775878e-10    3.6818097586724373e-07
Constraint violation....:   4.9012305433847061e-13    8.2422957348171611e-13
Variable bound violation:   2.7796368318178111e-08    2.7796368318178111e-08
Complementarity.........:   2.5089191802217999e-09    5.7397963221581475e-06
Overall NLP error.......:   2.5089191802217999e-09    5.7397963221581475e-06


Number of objective function evaluations             = 75
Number of objective gradient evaluations             = 37
Number of equality constraint evaluations            = 75
Number of inequality constraint evaluations          = 75
Number of equality constraint Jacobian evaluations   = 37
Number of inequality constraint Jacobian evaluations = 37
Number of Lagrangian Hessian evaluations             = 36
Total seconds in IPOPT                               = 0.118

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case3_lmbd.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       12
Number of nonzeros in inequality constraint Jacobian.:       28
Number of nonzeros in Lagrangian Hessian.............:       81

Total number of variables............................:        7
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        5
                     variables with only upper bounds:        0
Total number of equality constraints.................:        2
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.0415000e+03 1.00e-01 2.18e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8281767e+03 4.88e-02 1.73e+01  -1.0 1.72e-01   2.0 8.71e-01 1.00e+00f  1
   2  1.2456491e+03 4.02e-02 1.24e+01  -1.0 3.52e-01   1.5 7.50e-01 4.67e-01f  1
   3  1.1757068e+03 3.55e-02 1.09e+01  -1.0 4.02e-01    -  2.11e-01 1.25e-01f  1
   4  9.7867738e+02 3.20e-02 6.94e+00  -1.0 4.50e-01    -  1.34e-01 3.62e-01f  1
   5  7.8792959e+02 2.52e-02 8.87e+00  -1.0 3.55e-01    -  4.83e-01 9.79e-01f  1
   6  8.6287363e+02 2.15e-03 1.01e+00  -1.0 1.00e-01    -  9.91e-01 1.00e+00h  1
   7  7.9572299e+02 1.24e-03 6.38e-01  -1.0 9.73e-02    -  1.00e+00 1.00e+00f  1
   8  7.5317979e+02 7.04e-04 1.40e-01  -1.7 6.32e-02    -  1.00e+00 1.00e+00f  1
   9  7.4438683e+02 8.02e-05 3.37e-02  -2.5 1.26e-02    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  7.4408348e+02 3.66e-06 3.72e-03  -2.5 2.53e-03    -  1.00e+00 1.00e+00f  1
  11  7.4309700e+02 2.80e-06 2.64e-03  -3.8 2.09e-03    -  1.00e+00 1.00e+00f  1
  12  7.4305680e+02 7.30e-08 9.05e-05  -3.8 3.30e-04    -  1.00e+00 1.00e+00h  1
  13  7.4300130e+02 1.40e-08 1.42e-05  -5.7 1.46e-04    -  1.00e+00 1.00e+00h  1
  14  7.4300039e+02 9.07e-12 1.02e-08  -8.6 3.65e-06    -  1.00e+00 1.00e+00h  1
  15  7.4300039e+02 3.05e-16 1.07e-14  -9.0 2.88e-09    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0165744010648609e+01    7.4300038920867655e+02
Dual infeasibility......:   1.0658141036401503e-14    7.7898901742874248e-13
Constraint violation....:   3.0531133177191805e-16    3.0531133177191805e-16
Variable bound violation:   1.0864721122771925e-08    1.0864721122771925e-08
Complementarity.........:   9.0909205311425006e-10    6.6444300444990875e-08
Overall NLP error.......:   9.0909205311425006e-10    6.6444300444990875e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.010

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       12
Number of nonzeros in inequality constraint Jacobian.:       28
Number of nonzeros in Lagrangian Hessian.............:       81

Total number of variables............................:        7
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        5
                     variables with only upper bounds:        0
Total number of equality constraints.................:        2
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.0415000e+03 1.00e-01 2.18e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8281767e+03 4.88e-02 1.73e+01  -1.0 1.72e-01   2.0 8.71e-01 1.00e+00f  1
   2  1.2456491e+03 4.02e-02 1.24e+01  -1.0 3.52e-01   1.5 7.50e-01 4.67e-01f  1
   3  1.1757068e+03 3.55e-02 1.09e+01  -1.0 4.02e-01    -  2.11e-01 1.25e-01f  1
   4  9.7867738e+02 3.20e-02 6.94e+00  -1.0 4.50e-01    -  1.34e-01 3.62e-01f  1
   5  7.8792959e+02 2.52e-02 8.87e+00  -1.0 3.55e-01    -  4.83e-01 9.79e-01f  1
   6  8.6287363e+02 2.15e-03 1.01e+00  -1.0 1.00e-01    -  9.91e-01 1.00e+00h  1
   7  7.9572299e+02 1.24e-03 6.38e-01  -1.0 9.73e-02    -  1.00e+00 1.00e+00f  1
   8  7.5317979e+02 7.04e-04 1.40e-01  -1.7 6.32e-02    -  1.00e+00 1.00e+00f  1
   9  7.4438683e+02 8.02e-05 3.37e-02  -2.5 1.26e-02    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  7.4408348e+02 3.66e-06 3.72e-03  -2.5 2.53e-03    -  1.00e+00 1.00e+00f  1
  11  7.4309700e+02 2.80e-06 2.64e-03  -3.8 2.09e-03    -  1.00e+00 1.00e+00f  1
  12  7.4305680e+02 7.30e-08 9.05e-05  -3.8 3.30e-04    -  1.00e+00 1.00e+00h  1
  13  7.4300130e+02 1.40e-08 1.42e-05  -5.7 1.46e-04    -  1.00e+00 1.00e+00h  1
  14  7.4300039e+02 9.07e-12 1.02e-08  -8.6 3.65e-06    -  1.00e+00 1.00e+00h  1
  15  7.4300039e+02 3.05e-16 1.07e-14  -9.0 2.88e-09    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0165744010648609e+01    7.4300038920867655e+02
Dual infeasibility......:   1.0658141036401503e-14    7.7898901742874248e-13
Constraint violation....:   3.0531133177191805e-16    3.0531133177191805e-16
Variable bound violation:   1.0864721122771925e-08    1.0864721122771925e-08
Complementarity.........:   9.0909205311425006e-10    6.6444300444990875e-08
Overall NLP error.......:   9.0909205311425006e-10    6.6444300444990875e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.010

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case4020_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case4601_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case4619_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case4661_sdet.m"
file = "/tmp/jl_ttujSi/pglib_opf_case4837_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case4917_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case500_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case5658_epigrids.m"
file = "/tmp/jl_ttujSi/pglib_opf_case57_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      792
Number of nonzeros in inequality constraint Jacobian.:      941
Number of nonzeros in Lagrangian Hessian.............:     3629

Total number of variables............................:      119
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       63
                     variables with only upper bounds:        0
Total number of equality constraints.................:      108
Total number of inequality constraints...............:      320
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      320

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.5464745e+04 1.72e+00 2.39e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.3135319e+04 5.29e-01 1.39e+02  -1.0 5.32e+00    -  7.01e-02 6.96e-01H  1
   2  3.3266892e+04 3.22e-01 8.70e+01  -1.0 2.15e+00    -  6.75e-01 3.99e-01h  1
   3  2.9037565e+04 1.39e-01 3.65e+01  -1.0 2.61e+00    -  5.49e-01 6.05e-01f  1
   4  2.1855561e+04 6.81e-02 2.30e+01  -1.0 7.89e+00    -  9.53e-01 6.85e-01f  1
   5  1.4875145e+04 1.29e-02 1.13e+01  -1.0 1.92e+01    -  1.00e+00 8.63e-01F  1
   6  1.4572648e+04 2.56e-03 1.99e+00  -1.0 3.07e+00    -  1.00e+00 8.34e-01H  1
   7  1.4718740e+04 1.22e-03 6.52e-02  -1.0 9.96e-01    -  1.00e+00 1.00e+00f  1
   8  1.3867335e+04 6.17e-03 5.66e+00  -2.5 5.16e+00    -  8.37e-01 6.27e-01f  1
   9  1.3652951e+04 3.36e-03 2.70e-01  -2.5 2.16e+00    -  9.55e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.3615786e+04 1.89e-03 2.11e-02  -2.5 2.28e+00    -  1.00e+00 1.00e+00h  1
  11  1.3618724e+04 1.57e-05 1.63e-04  -2.5 1.71e-01    -  1.00e+00 1.00e+00h  1
  12  1.3590750e+04 2.34e-04 6.28e-02  -3.8 1.04e+00    -  9.02e-01 1.00e+00h  1
  13  1.3588612e+04 5.33e-05 2.77e-04  -3.8 3.73e-01    -  1.00e+00 1.00e+00h  1
  14  1.3588425e+04 1.85e-05 8.28e-05  -3.8 2.53e-01    -  1.00e+00 1.00e+00h  1
  15  1.3586478e+04 2.93e-05 3.47e-03  -5.7 2.32e-01    -  9.13e-01 1.00e+00h  1
  16  1.3586460e+04 1.33e-05 2.42e-05  -5.7 1.33e-01    -  1.00e+00 1.00e+00h  1
  17  1.3586454e+04 1.92e-06 3.44e-06  -5.7 3.40e-02    -  1.00e+00 1.00e+00h  1
  18  1.3586454e+04 1.36e-07 2.56e-07  -5.7 8.52e-03    -  1.00e+00 1.00e+00h  1
  19  1.3586431e+04 2.84e-07 4.13e-06  -8.6 1.52e-02    -  9.95e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.3586431e+04 7.58e-09 1.41e-08  -8.6 2.04e-03    -  1.00e+00 1.00e+00h  1
  21  1.3586431e+04 7.04e-12 1.32e-11  -8.6 6.11e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   7.5635014503680873e+00    1.3586430965807842e+04
Dual infeasibility......:   1.3214820263864168e-11    2.3737979614165535e-08
Constraint violation....:   7.0375788530085970e-12    7.0375788530085970e-12
Variable bound violation:   9.3718373062046112e-09    9.3718373062046112e-09
Complementarity.........:   2.5123411665866831e-09    4.5129560754861688e-06
Overall NLP error.......:   2.5123411665866831e-09    4.5129560754861688e-06


Number of objective function evaluations             = 25
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 25
Number of inequality constraint evaluations          = 25
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 285.789

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      792
Number of nonzeros in inequality constraint Jacobian.:      941
Number of nonzeros in Lagrangian Hessian.............:     3629

Total number of variables............................:      119
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       63
                     variables with only upper bounds:        0
Total number of equality constraints.................:      108
Total number of inequality constraints...............:      320
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      320

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.5464745e+04 1.72e+00 2.39e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.3135319e+04 5.29e-01 1.39e+02  -1.0 5.32e+00    -  7.01e-02 6.96e-01H  1
   2  3.3266892e+04 3.22e-01 8.70e+01  -1.0 2.15e+00    -  6.75e-01 3.99e-01h  1
   3  2.9037565e+04 1.39e-01 3.65e+01  -1.0 2.61e+00    -  5.49e-01 6.05e-01f  1
   4  2.1855561e+04 6.81e-02 2.30e+01  -1.0 7.89e+00    -  9.53e-01 6.85e-01f  1
   5  1.4875145e+04 1.29e-02 1.13e+01  -1.0 1.92e+01    -  1.00e+00 8.63e-01F  1
   6  1.4572648e+04 2.56e-03 1.99e+00  -1.0 3.07e+00    -  1.00e+00 8.34e-01H  1
   7  1.4718740e+04 1.22e-03 6.52e-02  -1.0 9.96e-01    -  1.00e+00 1.00e+00f  1
   8  1.3867335e+04 6.17e-03 5.66e+00  -2.5 5.16e+00    -  8.37e-01 6.27e-01f  1
   9  1.3652951e+04 3.36e-03 2.70e-01  -2.5 2.16e+00    -  9.55e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.3615786e+04 1.89e-03 2.11e-02  -2.5 2.28e+00    -  1.00e+00 1.00e+00h  1
  11  1.3618724e+04 1.57e-05 1.63e-04  -2.5 1.71e-01    -  1.00e+00 1.00e+00h  1
  12  1.3590750e+04 2.34e-04 6.28e-02  -3.8 1.04e+00    -  9.02e-01 1.00e+00h  1
  13  1.3588612e+04 5.33e-05 2.77e-04  -3.8 3.73e-01    -  1.00e+00 1.00e+00h  1
  14  1.3588425e+04 1.85e-05 8.28e-05  -3.8 2.53e-01    -  1.00e+00 1.00e+00h  1
  15  1.3586478e+04 2.93e-05 3.47e-03  -5.7 2.32e-01    -  9.13e-01 1.00e+00h  1
  16  1.3586460e+04 1.33e-05 2.42e-05  -5.7 1.33e-01    -  1.00e+00 1.00e+00h  1
  17  1.3586454e+04 1.92e-06 3.44e-06  -5.7 3.40e-02    -  1.00e+00 1.00e+00h  1
  18  1.3586454e+04 1.36e-07 2.56e-07  -5.7 8.52e-03    -  1.00e+00 1.00e+00h  1
  19  1.3586431e+04 2.84e-07 4.13e-06  -8.6 1.52e-02    -  9.95e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.3586431e+04 7.58e-09 1.41e-08  -8.6 2.04e-03    -  1.00e+00 1.00e+00h  1
  21  1.3586431e+04 7.04e-12 1.32e-11  -8.6 6.11e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   7.5635014503680873e+00    1.3586430965807842e+04
Dual infeasibility......:   1.3214820263864168e-11    2.3737979614165535e-08
Constraint violation....:   7.0375788530085970e-12    7.0375788530085970e-12
Variable bound violation:   9.3718373062046112e-09    9.3718373062046112e-09
Complementarity.........:   2.5123411665866831e-09    4.5129560754861688e-06
Overall NLP error.......:   2.5123411665866831e-09    4.5129560754861688e-06


Number of objective function evaluations             = 25
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 25
Number of inequality constraint evaluations          = 25
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.117

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case588_sdet.m"
file = "/tmp/jl_ttujSi/pglib_opf_case5_pjm.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       40
Number of nonzeros in inequality constraint Jacobian.:       60
Number of nonzeros in Lagrangian Hessian.............:      200

Total number of variables............................:       13
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        9
                     variables with only upper bounds:        0
Total number of equality constraints.................:        6
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.0500000e+03 3.99e+00 4.24e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4085733e+04 2.02e-03 9.56e+02  -1.0 1.40e+00    -  1.48e-01 1.00e+00H  1
   2  2.3966039e+04 1.35e-02 1.34e+01  -1.0 8.05e-01    -  9.51e-01 1.00e+00h  1
   3  2.3907973e+04 4.00e-04 1.38e+02  -1.0 4.93e-02   2.0 1.00e+00 1.00e+00h  1
   4  2.3296939e+04 9.47e-04 8.48e+01  -1.0 7.49e-01    -  1.00e+00 3.86e-01f  2
   5  2.2869373e+04 1.15e-03 4.61e+01  -1.0 2.47e-01    -  1.00e+00 1.00e+00h  1
   6  1.6680309e+04 1.15e-01 1.94e+01  -1.0 1.40e+01    -  5.68e-01 4.66e-01F  1
   7  1.8872763e+04 4.50e-02 5.76e+00  -1.0 5.62e+00    -  1.00e+00 1.00e+00f  1
   8  1.8270004e+04 2.90e-03 3.68e-01  -1.0 2.89e+00    -  1.00e+00 1.00e+00h  1
   9  1.6971509e+04 2.56e-03 1.50e+00  -1.7 2.42e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.6728621e+04 5.23e-03 2.68e-02  -1.7 2.74e+00    -  1.00e+00 1.00e+00h  1
  11  1.6756707e+04 3.39e-04 7.94e-03  -1.7 1.19e+00    -  1.00e+00 1.00e+00h  1
  12  1.6334001e+04 4.68e-03 4.47e-01  -3.8 5.15e+00    -  8.43e-01 8.83e-01h  1
  13  1.6273539e+04 4.05e-02 2.21e-01  -3.8 5.96e+00    -  8.57e-01 6.23e-01h  1
  14  1.6189011e+04 2.73e-02 3.55e-01  -3.8 3.46e+00    -  1.00e+00 1.00e+00h  1
  15  1.6238285e+04 5.42e-04 1.65e-03  -3.8 6.43e-01    -  1.00e+00 1.00e+00h  1
  16  1.6240300e+04 2.03e-05 2.15e-04  -3.8 1.07e-01    -  1.00e+00 1.00e+00h  1
  17  1.6236954e+04 2.07e-04 3.13e-03  -5.7 2.07e-01    -  9.75e-01 7.29e-01h  1
  18  1.6236733e+04 1.94e-05 1.70e-04  -5.7 3.64e-02    -  1.00e+00 9.90e-01h  1
  19  1.6236739e+04 7.49e-09 4.19e-08  -5.7 1.71e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.6236704e+04 1.03e-08 4.43e-06  -8.6 1.72e-03    -  1.00e+00 9.92e-01h  1
  21  1.6236704e+04 1.81e-13 1.93e-12  -8.6 8.11e-06    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3293548355426417e+00    1.6236704303715280e+04
Dual infeasibility......:   1.9305324434281426e-12    7.2402206849240285e-09
Constraint violation....:   1.8141044222375058e-13    1.8141044222375058e-13
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059961047863117e-09    9.3984252354722182e-06
Overall NLP error.......:   2.5059961047863117e-09    9.3984252354722182e-06


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.018

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       40
Number of nonzeros in inequality constraint Jacobian.:       60
Number of nonzeros in Lagrangian Hessian.............:      200

Total number of variables............................:       13
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        9
                     variables with only upper bounds:        0
Total number of equality constraints.................:        6
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.0500000e+03 3.99e+00 4.24e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4085733e+04 2.02e-03 9.56e+02  -1.0 1.40e+00    -  1.48e-01 1.00e+00H  1
   2  2.3966039e+04 1.35e-02 1.34e+01  -1.0 8.05e-01    -  9.51e-01 1.00e+00h  1
   3  2.3907973e+04 4.00e-04 1.38e+02  -1.0 4.93e-02   2.0 1.00e+00 1.00e+00h  1
   4  2.3296939e+04 9.47e-04 8.48e+01  -1.0 7.49e-01    -  1.00e+00 3.86e-01f  2
   5  2.2869373e+04 1.15e-03 4.61e+01  -1.0 2.47e-01    -  1.00e+00 1.00e+00h  1
   6  1.6680309e+04 1.15e-01 1.94e+01  -1.0 1.40e+01    -  5.68e-01 4.66e-01F  1
   7  1.8872763e+04 4.50e-02 5.76e+00  -1.0 5.62e+00    -  1.00e+00 1.00e+00f  1
   8  1.8270004e+04 2.90e-03 3.68e-01  -1.0 2.89e+00    -  1.00e+00 1.00e+00h  1
   9  1.6971509e+04 2.56e-03 1.50e+00  -1.7 2.42e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.6728621e+04 5.23e-03 2.68e-02  -1.7 2.74e+00    -  1.00e+00 1.00e+00h  1
  11  1.6756707e+04 3.39e-04 7.94e-03  -1.7 1.19e+00    -  1.00e+00 1.00e+00h  1
  12  1.6334001e+04 4.68e-03 4.47e-01  -3.8 5.15e+00    -  8.43e-01 8.83e-01h  1
  13  1.6273539e+04 4.05e-02 2.21e-01  -3.8 5.96e+00    -  8.57e-01 6.23e-01h  1
  14  1.6189011e+04 2.73e-02 3.55e-01  -3.8 3.46e+00    -  1.00e+00 1.00e+00h  1
  15  1.6238285e+04 5.42e-04 1.65e-03  -3.8 6.43e-01    -  1.00e+00 1.00e+00h  1
  16  1.6240300e+04 2.03e-05 2.15e-04  -3.8 1.07e-01    -  1.00e+00 1.00e+00h  1
  17  1.6236954e+04 2.07e-04 3.13e-03  -5.7 2.07e-01    -  9.75e-01 7.29e-01h  1
  18  1.6236733e+04 1.94e-05 1.70e-04  -5.7 3.64e-02    -  1.00e+00 9.90e-01h  1
  19  1.6236739e+04 7.49e-09 4.19e-08  -5.7 1.71e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.6236704e+04 1.03e-08 4.43e-06  -8.6 1.72e-03    -  1.00e+00 9.92e-01h  1
  21  1.6236704e+04 1.81e-13 1.93e-12  -8.6 8.11e-06    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3293548355426417e+00    1.6236704303715280e+04
Dual infeasibility......:   1.9305324434281426e-12    7.2402206849240285e-09
Constraint violation....:   1.8141044222375058e-13    1.8141044222375058e-13
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059961047863117e-09    9.3984252354722182e-06
Overall NLP error.......:   2.5059961047863117e-09    9.3984252354722182e-06


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.017

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case60_c.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      734
Number of nonzeros in inequality constraint Jacobian.:     1052
Number of nonzeros in Lagrangian Hessian.............:     3674

Total number of variables............................:      139
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       80
                     variables with only upper bounds:        0
Total number of equality constraints.................:       95
Total number of inequality constraints...............:      352
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      352

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.6499999e+03 1.26e+02 5.72e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.5159983e+03 1.22e+02 8.10e+00  -1.0 7.03e+00    -  1.15e-02 2.99e-02h  4
   2  1.2060701e+04 1.13e+02 3.43e+01  -1.0 6.55e+00    -  1.01e-02 5.12e-02h  3
   3  2.2974643e+04 8.31e+01 1.10e+02  -1.0 9.32e+00    -  1.41e-02 8.77e-02h  3
   4  3.0719479e+04 5.14e+01 1.44e+02  -1.0 1.65e+01    -  8.10e-02 6.76e-02h  3
   5  4.3224414e+04 1.40e+01 2.32e+02  -1.0 2.26e+01    -  2.57e-01 1.18e-01h  2
   6  6.2754285e+04 1.12e+01 4.28e+02  -1.0 3.44e+01    -  3.94e-01 2.10e-01H  1
   7  1.3360791e+05 2.79e-01 3.49e+03  -1.0 5.60e+01    -  4.15e-01 9.78e-01H  1
   8  1.3180044e+05 1.70e-02 2.83e+01  -1.0 1.41e+01    -  9.07e-01 1.00e+00f  1
   9  1.2392664e+05 4.66e+00 2.07e+01  -1.0 4.70e+00  -2.0 4.34e-01 9.33e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.1769473e+05 8.65e-01 1.46e+01  -1.0 1.15e+02    -  2.95e-01 2.08e-01f  1
  11  1.1377505e+05 3.87e-01 1.44e+01  -1.0 3.70e+03    -  1.58e-02 1.14e-02f  4
  12  1.1254021e+05 4.80e-02 6.69e+00  -1.0 4.65e+00  -2.5 5.23e-01 1.00e+00h  1
  13  1.0348211e+05 2.28e-01 6.68e+00  -1.0 1.36e+02    -  2.34e-01 2.90e-01f  1
  14  9.6774244e+04 1.57e-01 1.64e+01  -1.0 1.77e+02    -  1.00e+00 4.95e-01F  1
  15  8.7771468e+04 9.76e-02 3.04e+01  -1.0 2.12e+02    -  9.18e-01 1.00e+00F  1
  16  8.6197091e+04 2.64e-02 5.71e-01  -1.0 2.16e+02    -  1.00e+00 1.00e+00H  1
  17  7.9769835e+04 5.26e-02 1.13e+00  -1.7 1.13e+02    -  9.07e-01 8.64e-01f  1
  18  7.5840211e+04 8.49e-02 1.19e-01  -1.7 1.32e+02    -  1.00e+00 1.00e+00h  1
  19  7.6297427e+04 1.78e-03 2.00e-02  -1.7 4.01e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  7.4470398e+04 1.02e-01 1.22e-01  -2.5 8.91e+01    -  8.38e-01 6.55e-01h  1
  21  7.3576845e+04 2.18e-01 1.48e-01  -2.5 3.46e+01    -  6.25e-01 8.69e-01h  1
  22  7.3418908e+04 1.53e-02 7.03e-02  -2.5 9.87e+00    -  1.00e+00 1.00e+00h  1
  23  7.3453970e+04 6.39e-04 1.05e-02  -2.5 7.76e-01    -  1.00e+00 1.00e+00h  1
  24  7.3120526e+04 1.32e-02 2.71e-02  -3.8 1.60e+01    -  8.78e-01 6.51e-01h  1
  25  7.2999736e+04 5.76e-03 4.55e-02  -3.8 5.65e+00    -  1.00e+00 6.30e-01h  1
  26  7.2999838e+04 3.40e-03 9.62e-02  -3.8 3.51e+00    -  8.70e-01 1.00e+00h  1
  27  7.3008139e+04 2.63e-04 2.61e-03  -3.8 5.46e-01    -  1.00e+00 1.00e+00h  1
  28  7.3009356e+04 1.11e-05 6.95e-05  -3.8 5.59e-02    -  1.00e+00 1.00e+00h  1
  29  7.2993687e+04 2.34e-04 2.63e-03  -5.7 1.46e+00    -  7.55e-01 7.43e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  7.2991483e+04 2.97e-03 1.70e-02  -5.7 3.53e-01    -  9.42e-01 8.14e-01h  1
  31  7.2991196e+04 1.03e-05 8.87e-05  -5.7 8.60e-02    -  1.00e+00 1.00e+00h  1
  32  7.2991215e+04 4.83e-08 6.59e-07  -5.7 1.47e-02    -  1.00e+00 1.00e+00h  1
  33  7.2991018e+04 3.15e-07 6.60e-05  -8.6 1.37e-02    -  9.95e-01 9.81e-01h  1
  34  7.2991015e+04 2.39e-10 3.60e-09  -8.6 5.88e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 34

                                   (scaled)                 (unscaled)
Objective...............:   2.4327905327884167e+01    7.2991015085161009e+04
Dual infeasibility......:   3.6025277792860484e-09    1.0808664204278574e-05
Constraint violation....:   2.3900526002762490e-10    2.3900526002762490e-10
Variable bound violation:   5.6456421759776276e-08    5.6456421759776276e-08
Complementarity.........:   2.7594491399162557e-09    8.2791753372824940e-06
Overall NLP error.......:   3.6025277792860484e-09    1.0808664204278574e-05


Number of objective function evaluations             = 64
Number of objective gradient evaluations             = 35
Number of equality constraint evaluations            = 64
Number of inequality constraint evaluations          = 64
Number of equality constraint Jacobian evaluations   = 35
Number of inequality constraint Jacobian evaluations = 35
Number of Lagrangian Hessian evaluations             = 34
Total seconds in IPOPT                               = 204.594

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      734
Number of nonzeros in inequality constraint Jacobian.:     1052
Number of nonzeros in Lagrangian Hessian.............:     3674

Total number of variables............................:      139
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       80
                     variables with only upper bounds:        0
Total number of equality constraints.................:       95
Total number of inequality constraints...............:      352
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      352

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.6499999e+03 1.26e+02 5.72e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.5159983e+03 1.22e+02 8.10e+00  -1.0 7.03e+00    -  1.15e-02 2.99e-02h  4
   2  1.2060701e+04 1.13e+02 3.43e+01  -1.0 6.55e+00    -  1.01e-02 5.12e-02h  3
   3  2.2974643e+04 8.31e+01 1.10e+02  -1.0 9.32e+00    -  1.41e-02 8.77e-02h  3
   4  3.0719479e+04 5.14e+01 1.44e+02  -1.0 1.65e+01    -  8.10e-02 6.76e-02h  3
   5  4.3224414e+04 1.40e+01 2.32e+02  -1.0 2.26e+01    -  2.57e-01 1.18e-01h  2
   6  6.2754285e+04 1.12e+01 4.28e+02  -1.0 3.44e+01    -  3.94e-01 2.10e-01H  1
   7  1.3360791e+05 2.79e-01 3.49e+03  -1.0 5.60e+01    -  4.15e-01 9.78e-01H  1
   8  1.3180044e+05 1.70e-02 2.83e+01  -1.0 1.41e+01    -  9.07e-01 1.00e+00f  1
   9  1.2392664e+05 4.66e+00 2.07e+01  -1.0 4.70e+00  -2.0 4.34e-01 9.33e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.1769473e+05 8.65e-01 1.46e+01  -1.0 1.15e+02    -  2.95e-01 2.08e-01f  1
  11  1.1377505e+05 3.87e-01 1.44e+01  -1.0 3.70e+03    -  1.58e-02 1.14e-02f  4
  12  1.1254021e+05 4.80e-02 6.69e+00  -1.0 4.65e+00  -2.5 5.23e-01 1.00e+00h  1
  13  1.0348211e+05 2.28e-01 6.68e+00  -1.0 1.36e+02    -  2.34e-01 2.90e-01f  1
  14  9.6774244e+04 1.57e-01 1.64e+01  -1.0 1.77e+02    -  1.00e+00 4.95e-01F  1
  15  8.7771468e+04 9.76e-02 3.04e+01  -1.0 2.12e+02    -  9.18e-01 1.00e+00F  1
  16  8.6197091e+04 2.64e-02 5.71e-01  -1.0 2.16e+02    -  1.00e+00 1.00e+00H  1
  17  7.9769835e+04 5.26e-02 1.13e+00  -1.7 1.13e+02    -  9.07e-01 8.64e-01f  1
  18  7.5840211e+04 8.49e-02 1.19e-01  -1.7 1.32e+02    -  1.00e+00 1.00e+00h  1
  19  7.6297427e+04 1.78e-03 2.00e-02  -1.7 4.01e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  7.4470398e+04 1.02e-01 1.22e-01  -2.5 8.91e+01    -  8.38e-01 6.55e-01h  1
  21  7.3576845e+04 2.18e-01 1.48e-01  -2.5 3.46e+01    -  6.25e-01 8.69e-01h  1
  22  7.3418908e+04 1.53e-02 7.03e-02  -2.5 9.87e+00    -  1.00e+00 1.00e+00h  1
  23  7.3453970e+04 6.39e-04 1.05e-02  -2.5 7.76e-01    -  1.00e+00 1.00e+00h  1
  24  7.3120526e+04 1.32e-02 2.71e-02  -3.8 1.60e+01    -  8.78e-01 6.51e-01h  1
  25  7.2999736e+04 5.76e-03 4.55e-02  -3.8 5.65e+00    -  1.00e+00 6.30e-01h  1
  26  7.2999838e+04 3.40e-03 9.62e-02  -3.8 3.51e+00    -  8.70e-01 1.00e+00h  1
  27  7.3008139e+04 2.63e-04 2.61e-03  -3.8 5.46e-01    -  1.00e+00 1.00e+00h  1
  28  7.3009356e+04 1.11e-05 6.95e-05  -3.8 5.59e-02    -  1.00e+00 1.00e+00h  1
  29  7.2993687e+04 2.34e-04 2.63e-03  -5.7 1.46e+00    -  7.55e-01 7.43e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  7.2991483e+04 2.97e-03 1.70e-02  -5.7 3.53e-01    -  9.42e-01 8.14e-01h  1
  31  7.2991196e+04 1.03e-05 8.87e-05  -5.7 8.60e-02    -  1.00e+00 1.00e+00h  1
  32  7.2991215e+04 4.83e-08 6.59e-07  -5.7 1.47e-02    -  1.00e+00 1.00e+00h  1
  33  7.2991018e+04 3.15e-07 6.60e-05  -8.6 1.37e-02    -  9.95e-01 9.81e-01h  1
  34  7.2991015e+04 2.39e-10 3.60e-09  -8.6 5.88e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 34

                                   (scaled)                 (unscaled)
Objective...............:   2.4327905327884167e+01    7.2991015085161009e+04
Dual infeasibility......:   3.6025277792860484e-09    1.0808664204278574e-05
Constraint violation....:   2.3900526002762490e-10    2.3900526002762490e-10
Variable bound violation:   5.6456421759776276e-08    5.6456421759776276e-08
Complementarity.........:   2.7594491399162557e-09    8.2791753372824940e-06
Overall NLP error.......:   3.6025277792860484e-09    1.0808664204278574e-05


Number of objective function evaluations             = 64
Number of objective gradient evaluations             = 35
Number of equality constraint evaluations            = 64
Number of inequality constraint evaluations          = 64
Number of equality constraint Jacobian evaluations   = 35
Number of inequality constraint Jacobian evaluations = 35
Number of Lagrangian Hessian evaluations             = 34
Total seconds in IPOPT                               = 0.166

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case6468_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case6470_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case6495_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case6515_rte.m"
file = "/tmp/jl_ttujSi/pglib_opf_case7336_epigrids.m"
file = "/tmp/jl_ttujSi/pglib_opf_case73_ieee_rts.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      938
Number of nonzeros in inequality constraint Jacobian.:     1425
Number of nonzeros in Lagrangian Hessian.............:     5509

Total number of variables............................:      264
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      192
                     variables with only upper bounds:        0
Total number of equality constraints.................:      112
Total number of inequality constraints...............:      480
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      480

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.2825909e+05 2.32e+00 1.10e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.5458019e+05 1.40e+00 8.50e+01  -1.0 1.07e+00    -  1.13e-01 4.00e-01f  1
   2  4.0941357e+05 6.92e-01 8.32e+01  -1.0 2.63e+00    -  2.08e-01 5.07e-01f  1
   3  4.2237366e+05 3.33e-02 6.10e+01  -1.0 2.99e+00    -  5.44e-01 1.00e+00f  1
   4  3.0526786e+05 1.17e-01 1.16e+01  -1.0 1.73e+00    -  9.47e-01 1.00e+00f  1
   5  2.3439897e+05 9.97e-02 2.58e+00  -1.0 3.33e+00    -  1.00e+00 1.00e+00h  1
   6  2.2346477e+05 1.64e-02 3.84e-01  -1.0 4.21e+00    -  1.00e+00 1.00e+00h  1
   7  1.7075285e+05 8.75e-02 1.99e+00  -1.7 7.87e+00    -  1.00e+00 1.00e+00h  1
   8  1.2531323e+05 1.12e-01 2.35e-01  -1.7 1.07e+01    -  1.00e+00 1.00e+00h  1
   9  1.3301887e+05 7.24e-03 2.66e-02  -1.7 2.33e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.0586954e+05 3.34e-01 5.34e-01  -2.5 6.00e+00    -  9.24e-01 8.75e-01h  1
  11  9.7620400e+04 1.34e-01 6.15e-02  -2.5 3.48e+00    -  1.00e+00 9.86e-01h  1
  12  9.7894332e+04 6.56e-04 3.39e-02  -2.5 1.01e+00    -  1.00e+00 1.00e+00h  1
  13  9.7741158e+04 5.13e-05 2.20e-03  -2.5 3.61e-01    -  1.00e+00 1.00e+00h  1
  14  9.4312704e+04 7.18e-04 1.70e-01  -3.8 1.69e+00    -  7.83e-01 5.78e-01f  1
  15  9.2441407e+04 5.24e-03 1.43e-01  -3.8 1.16e+00    -  8.12e-01 6.85e-01h  1
  16  9.1580886e+04 7.55e-03 3.78e-02  -3.8 5.93e-01    -  8.71e-01 1.00e+00h  1
  17  9.1561535e+04 1.05e-04 7.04e-04  -3.8 1.99e-01    -  1.00e+00 1.00e+00h  1
  18  9.1560512e+04 8.57e-07 4.78e-06  -3.8 1.57e-02    -  1.00e+00 1.00e+00h  1
  19  9.1241932e+04 5.66e-04 1.61e-02  -5.7 3.99e-01    -  7.66e-01 8.50e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  9.1171606e+04 2.73e-04 9.24e-03  -5.7 1.38e-01    -  9.11e-01 9.99e-01h  1
  21  9.1167889e+04 7.81e-05 1.21e-04  -5.7 6.21e-02    -  1.00e+00 1.00e+00h  1
  22  9.1167702e+04 2.63e-06 2.21e-07  -5.7 1.53e-02    -  1.00e+00 1.00e+00h  1
  23  9.1167699e+04 1.96e-08 8.55e-09  -5.7 8.98e-04    -  1.00e+00 1.00e+00h  1
  24  9.1162690e+04 3.14e-06 6.25e-06  -8.6 1.26e-02    -  9.81e-01 9.79e-01h  1
  25  9.1162568e+04 3.17e-08 7.43e-09  -8.6 1.36e-03    -  1.00e+00 1.00e+00h  1
  26  9.1162568e+04 2.60e-11 1.15e-11  -8.6 3.13e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 26

                                   (scaled)                 (unscaled)
Objective...............:   2.9396539456257083e+00    9.1162567524511353e+04
Dual infeasibility......:   1.1493028750919621e-11    3.5641406401794992e-07
Constraint violation....:   2.6005864128819667e-11    2.6005864128819667e-11
Variable bound violation:   2.4902964135264938e-08    2.4902964135264938e-08
Complementarity.........:   2.5101035813794565e-09    7.7841641044696616e-05
Overall NLP error.......:   2.5101035813794565e-09    7.7841641044696616e-05


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 27
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 27
Number of inequality constraint Jacobian evaluations = 27
Number of Lagrangian Hessian evaluations             = 26
Total seconds in IPOPT                               = 1083.422

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      938
Number of nonzeros in inequality constraint Jacobian.:     1425
Number of nonzeros in Lagrangian Hessian.............:     5509

Total number of variables............................:      264
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      192
                     variables with only upper bounds:        0
Total number of equality constraints.................:      112
Total number of inequality constraints...............:      480
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      480

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.2825909e+05 2.32e+00 1.10e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.5458019e+05 1.40e+00 8.50e+01  -1.0 1.07e+00    -  1.13e-01 4.00e-01f  1
   2  4.0941357e+05 6.92e-01 8.32e+01  -1.0 2.63e+00    -  2.08e-01 5.07e-01f  1
   3  4.2237366e+05 3.33e-02 6.10e+01  -1.0 2.99e+00    -  5.44e-01 1.00e+00f  1
   4  3.0526786e+05 1.17e-01 1.16e+01  -1.0 1.73e+00    -  9.47e-01 1.00e+00f  1
   5  2.3439897e+05 9.97e-02 2.58e+00  -1.0 3.33e+00    -  1.00e+00 1.00e+00h  1
   6  2.2346477e+05 1.64e-02 3.84e-01  -1.0 4.21e+00    -  1.00e+00 1.00e+00h  1
   7  1.7075285e+05 8.75e-02 1.99e+00  -1.7 7.87e+00    -  1.00e+00 1.00e+00h  1
   8  1.2531323e+05 1.12e-01 2.35e-01  -1.7 1.07e+01    -  1.00e+00 1.00e+00h  1
   9  1.3301887e+05 7.24e-03 2.66e-02  -1.7 2.33e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.0586954e+05 3.34e-01 5.34e-01  -2.5 6.00e+00    -  9.24e-01 8.75e-01h  1
  11  9.7620400e+04 1.34e-01 6.15e-02  -2.5 3.48e+00    -  1.00e+00 9.86e-01h  1
  12  9.7894332e+04 6.56e-04 3.39e-02  -2.5 1.01e+00    -  1.00e+00 1.00e+00h  1
  13  9.7741158e+04 5.13e-05 2.20e-03  -2.5 3.61e-01    -  1.00e+00 1.00e+00h  1
  14  9.4312704e+04 7.18e-04 1.70e-01  -3.8 1.69e+00    -  7.83e-01 5.78e-01f  1
  15  9.2441407e+04 5.24e-03 1.43e-01  -3.8 1.16e+00    -  8.12e-01 6.85e-01h  1
  16  9.1580886e+04 7.55e-03 3.78e-02  -3.8 5.93e-01    -  8.71e-01 1.00e+00h  1
  17  9.1561535e+04 1.05e-04 7.04e-04  -3.8 1.99e-01    -  1.00e+00 1.00e+00h  1
  18  9.1560512e+04 8.57e-07 4.78e-06  -3.8 1.57e-02    -  1.00e+00 1.00e+00h  1
  19  9.1241932e+04 5.66e-04 1.61e-02  -5.7 3.99e-01    -  7.66e-01 8.50e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  9.1171606e+04 2.73e-04 9.24e-03  -5.7 1.38e-01    -  9.11e-01 9.99e-01h  1
  21  9.1167889e+04 7.81e-05 1.21e-04  -5.7 6.21e-02    -  1.00e+00 1.00e+00h  1
  22  9.1167702e+04 2.63e-06 2.21e-07  -5.7 1.53e-02    -  1.00e+00 1.00e+00h  1
  23  9.1167699e+04 1.96e-08 8.55e-09  -5.7 8.98e-04    -  1.00e+00 1.00e+00h  1
  24  9.1162690e+04 3.14e-06 6.25e-06  -8.6 1.26e-02    -  9.81e-01 9.79e-01h  1
  25  9.1162568e+04 3.17e-08 7.43e-09  -8.6 1.36e-03    -  1.00e+00 1.00e+00h  1
  26  9.1162568e+04 2.60e-11 1.15e-11  -8.6 3.13e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 26

                                   (scaled)                 (unscaled)
Objective...............:   2.9396539456257083e+00    9.1162567524511353e+04
Dual infeasibility......:   1.1493028750919621e-11    3.5641406401794992e-07
Constraint violation....:   2.6005864128819667e-11    2.6005864128819667e-11
Variable bound violation:   2.4902964135264938e-08    2.4902964135264938e-08
Complementarity.........:   2.5101035813794565e-09    7.7841641044696616e-05
Overall NLP error.......:   2.5101035813794565e-09    7.7841641044696616e-05


Number of objective function evaluations             = 27
Number of objective gradient evaluations             = 27
Number of equality constraint evaluations            = 27
Number of inequality constraint evaluations          = 27
Number of equality constraint Jacobian evaluations   = 27
Number of inequality constraint Jacobian evaluations = 27
Number of Lagrangian Hessian evaluations             = 26
Total seconds in IPOPT                               = 0.227

EXIT: Optimal Solution Found.
file = "/tmp/jl_ttujSi/pglib_opf_case78484_epigrids.m"
file = "/tmp/jl_ttujSi/pglib_opf_case793_goc.m"
file = "/tmp/jl_ttujSi/pglib_opf_case8387_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case89_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case9241_pegase.m"
file = "/tmp/jl_ttujSi/pglib_opf_case9591_goc.m"
10×23 DataFrame
 Row │ case                         vars   cons   optimization  optimization_m ⋯
     │ String                       Int64  Int64  Float64       Float64        ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ pglib_opf_case14_ieee.m        118    169      8.00617               0. ⋯
   2 │ pglib_opf_case24_ieee_rts.m    266    315     63.5218                0.
   3 │ pglib_opf_case30_as.m          236    348     38.2858                0.
   4 │ pglib_opf_case30_ieee.m        236    348     59.1589                0.
   5 │ pglib_opf_case39_epri.m        282    401    122.373                 0. ⋯
   6 │ pglib_opf_case3_lmbd.m          24     28      0.211679              5.
   7 │ pglib_opf_case57_ieee.m        448    675    133.628                 0.
   8 │ pglib_opf_case5_pjm.m           44     53      0.976089              6.
   9 │ pglib_opf_case60_c.m           518    737    130.589                 0. ⋯
  10 │ pglib_opf_case73_ieee_rts.m    824    987    430.525                 0.
                                                              19 columns omitted
```



```julia
io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, timing_data; backend = Val(:html))
# show(io, "text/html", pretty_table(timing_data; backend = Val(:html)))
println(io, "```")
Text(String(take!(io)))
```

```@raw html
<table>
  <thead>
    <tr class = "header">
      <th style = "text-align: right;">case</th>
      <th style = "text-align: right;">vars</th>
      <th style = "text-align: right;">cons</th>
      <th style = "text-align: right;">optimization</th>
      <th style = "text-align: right;">optimization_modelbuild</th>
      <th style = "text-align: right;">optimization_wcompilation</th>
      <th style = "text-align: right;">optimization_cost</th>
      <th style = "text-align: right;">mtk</th>
      <th style = "text-align: right;">mtk_time_modelbuild</th>
      <th style = "text-align: right;">mtk_time_wcompilation</th>
      <th style = "text-align: right;">mtk_cost</th>
      <th style = "text-align: right;">jump</th>
      <th style = "text-align: right;">jump_modelbuild</th>
      <th style = "text-align: right;">jump_wcompilation</th>
      <th style = "text-align: right;">jump_cost</th>
      <th style = "text-align: right;">nlpmodels</th>
      <th style = "text-align: right;">nlpmodels_modelbuild</th>
      <th style = "text-align: right;">nlpmodels_wcompilation</th>
      <th style = "text-align: right;">nlpmodels_cost</th>
      <th style = "text-align: right;">optim</th>
      <th style = "text-align: right;">optim_modelbuild</th>
      <th style = "text-align: right;">optim_wcompilation</th>
      <th style = "text-align: right;">optim_cost</th>
    </tr>
    <tr class = "subheader headerLastRow">
      <th style = "text-align: right;">String</th>
      <th style = "text-align: right;">Int64</th>
      <th style = "text-align: right;">Int64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
      <th style = "text-align: right;">Float64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style = "text-align: right;">pglib_opf_case14_ieee.m</td>
      <td style = "text-align: right;">118</td>
      <td style = "text-align: right;">169</td>
      <td style = "text-align: right;">8.00617</td>
      <td style = "text-align: right;">0.000100219</td>
      <td style = "text-align: right;">8.82699</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">0.0688544</td>
      <td style = "text-align: right;">4.30231</td>
      <td style = "text-align: right;">20.9276</td>
      <td style = "text-align: right;">-10806.8</td>
      <td style = "text-align: right;">0.0155253</td>
      <td style = "text-align: right;">0.249278</td>
      <td style = "text-align: right;">0.0494466</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">1.0141</td>
      <td style = "text-align: right;">0.132813</td>
      <td style = "text-align: right;">0.0789104</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">123.826</td>
      <td style = "text-align: right;">0.0655969</td>
      <td style = "text-align: right;">123.467</td>
      <td style = "text-align: right;">1658.7</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case24_ieee_rts.m</td>
      <td style = "text-align: right;">266</td>
      <td style = "text-align: right;">315</td>
      <td style = "text-align: right;">63.5218</td>
      <td style = "text-align: right;">0.000145249</td>
      <td style = "text-align: right;">65.8718</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">0.202841</td>
      <td style = "text-align: right;">9.18944</td>
      <td style = "text-align: right;">119.399</td>
      <td style = "text-align: right;">43689.3</td>
      <td style = "text-align: right;">0.0321452</td>
      <td style = "text-align: right;">0.0384232</td>
      <td style = "text-align: right;">0.0329211</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">0.207503</td>
      <td style = "text-align: right;">0.16624</td>
      <td style = "text-align: right;">0.569162</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">384.284</td>
      <td style = "text-align: right;">0.0524681</td>
      <td style = "text-align: right;">382.569</td>
      <td style = "text-align: right;">63741.2</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case30_as.m</td>
      <td style = "text-align: right;">236</td>
      <td style = "text-align: right;">348</td>
      <td style = "text-align: right;">38.2858</td>
      <td style = "text-align: right;">0.000179098</td>
      <td style = "text-align: right;">37.0622</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">0.191836</td>
      <td style = "text-align: right;">8.77981</td>
      <td style = "text-align: right;">74.8257</td>
      <td style = "text-align: right;">803.118</td>
      <td style = "text-align: right;">0.0213372</td>
      <td style = "text-align: right;">0.0098225</td>
      <td style = "text-align: right;">0.0226292</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">0.146451</td>
      <td style = "text-align: right;">0.154562</td>
      <td style = "text-align: right;">0.149677</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">299.368</td>
      <td style = "text-align: right;">0.0169356</td>
      <td style = "text-align: right;">297.231</td>
      <td style = "text-align: right;">772.093</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case30_ieee.m</td>
      <td style = "text-align: right;">236</td>
      <td style = "text-align: right;">348</td>
      <td style = "text-align: right;">59.1589</td>
      <td style = "text-align: right;">0.000199298</td>
      <td style = "text-align: right;">62.1541</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">0.200102</td>
      <td style = "text-align: right;">7.16145</td>
      <td style = "text-align: right;">80.1533</td>
      <td style = "text-align: right;">-12842.4</td>
      <td style = "text-align: right;">0.0306336</td>
      <td style = "text-align: right;">0.00951944</td>
      <td style = "text-align: right;">0.0317445</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">0.231549</td>
      <td style = "text-align: right;">0.17021</td>
      <td style = "text-align: right;">0.233984</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">299.621</td>
      <td style = "text-align: right;">0.442476</td>
      <td style = "text-align: right;">299.971</td>
      <td style = "text-align: right;">4244.05</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case39_epri.m</td>
      <td style = "text-align: right;">282</td>
      <td style = "text-align: right;">401</td>
      <td style = "text-align: right;">122.373</td>
      <td style = "text-align: right;">0.000187709</td>
      <td style = "text-align: right;">118.449</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">0.359624</td>
      <td style = "text-align: right;">9.33898</td>
      <td style = "text-align: right;">166.947</td>
      <td style = "text-align: right;">1.19419e5</td>
      <td style = "text-align: right;">0.0535391</td>
      <td style = "text-align: right;">0.00624061</td>
      <td style = "text-align: right;">0.0553409</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">0.310181</td>
      <td style = "text-align: right;">0.17202</td>
      <td style = "text-align: right;">0.346189</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">363.792</td>
      <td style = "text-align: right;">0.109105</td>
      <td style = "text-align: right;">361.489</td>
      <td style = "text-align: right;">78346.0</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case3_lmbd.m</td>
      <td style = "text-align: right;">24</td>
      <td style = "text-align: right;">28</td>
      <td style = "text-align: right;">0.211679</td>
      <td style = "text-align: right;">5.1839e-5</td>
      <td style = "text-align: right;">0.18097</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0167586</td>
      <td style = "text-align: right;">0.42155</td>
      <td style = "text-align: right;">0.0191153</td>
      <td style = "text-align: right;">743.0</td>
      <td style = "text-align: right;">0.008994</td>
      <td style = "text-align: right;">0.00405742</td>
      <td style = "text-align: right;">0.00959787</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0186793</td>
      <td style = "text-align: right;">0.0142239</td>
      <td style = "text-align: right;">0.0197595</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">2.3701</td>
      <td style = "text-align: right;">0.000461976</td>
      <td style = "text-align: right;">2.3681</td>
      <td style = "text-align: right;">6273.63</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case57_ieee.m</td>
      <td style = "text-align: right;">448</td>
      <td style = "text-align: right;">675</td>
      <td style = "text-align: right;">133.628</td>
      <td style = "text-align: right;">0.000276328</td>
      <td style = "text-align: right;">130.421</td>
      <td style = "text-align: right;">35192.0</td>
      <td style = "text-align: right;">0.491086</td>
      <td style = "text-align: right;">19.8321</td>
      <td style = "text-align: right;">287.565</td>
      <td style = "text-align: right;">13586.4</td>
      <td style = "text-align: right;">0.0460831</td>
      <td style = "text-align: right;">0.013891</td>
      <td style = "text-align: right;">0.0474168</td>
      <td style = "text-align: right;">37589.3</td>
      <td style = "text-align: right;">0.897384</td>
      <td style = "text-align: right;">0.34733</td>
      <td style = "text-align: right;">0.365197</td>
      <td style = "text-align: right;">37589.3</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case5_pjm.m</td>
      <td style = "text-align: right;">44</td>
      <td style = "text-align: right;">53</td>
      <td style = "text-align: right;">0.976089</td>
      <td style = "text-align: right;">6.0079e-5</td>
      <td style = "text-align: right;">0.847475</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0306352</td>
      <td style = "text-align: right;">0.762705</td>
      <td style = "text-align: right;">0.0323985</td>
      <td style = "text-align: right;">16236.7</td>
      <td style = "text-align: right;">0.0140361</td>
      <td style = "text-align: right;">0.00425123</td>
      <td style = "text-align: right;">0.0148818</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0394116</td>
      <td style = "text-align: right;">0.025832</td>
      <td style = "text-align: right;">0.039499</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">20.5075</td>
      <td style = "text-align: right;">0.000676926</td>
      <td style = "text-align: right;">20.1869</td>
      <td style = "text-align: right;">77.9548</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case60_c.m</td>
      <td style = "text-align: right;">518</td>
      <td style = "text-align: right;">737</td>
      <td style = "text-align: right;">130.589</td>
      <td style = "text-align: right;">0.000256118</td>
      <td style = "text-align: right;">131.297</td>
      <td style = "text-align: right;">18338.5</td>
      <td style = "text-align: right;">1.16596</td>
      <td style = "text-align: right;">20.3741</td>
      <td style = "text-align: right;">206.313</td>
      <td style = "text-align: right;">72991.0</td>
      <td style = "text-align: right;">0.0798728</td>
      <td style = "text-align: right;">0.0294596</td>
      <td style = "text-align: right;">0.0819908</td>
      <td style = "text-align: right;">92693.7</td>
      <td style = "text-align: right;">0.949834</td>
      <td style = "text-align: right;">0.336046</td>
      <td style = "text-align: right;">0.644875</td>
      <td style = "text-align: right;">92693.7</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case73_ieee_rts.m</td>
      <td style = "text-align: right;">824</td>
      <td style = "text-align: right;">987</td>
      <td style = "text-align: right;">430.525</td>
      <td style = "text-align: right;">0.000358457</td>
      <td style = "text-align: right;">416.533</td>
      <td style = "text-align: right;">1.5864e5</td>
      <td style = "text-align: right;">0.944983</td>
      <td style = "text-align: right;">526.644</td>
      <td style = "text-align: right;">1085.53</td>
      <td style = "text-align: right;">91162.6</td>
      <td style = "text-align: right;">0.100855</td>
      <td style = "text-align: right;">0.443444</td>
      <td style = "text-align: right;">0.103228</td>
      <td style = "text-align: right;">1.89764e5</td>
      <td style = "text-align: right;">1.11119</td>
      <td style = "text-align: right;">0.572093</td>
      <td style = "text-align: right;">0.908917</td>
      <td style = "text-align: right;">1.89764e5</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
      <td style = "text-align: right;">NaN</td>
    </tr>
  </tbody>
</table>
```





## Appendix

```
Error: ArgumentError: Package SciMLBenchmarks not found in current path, ma
ybe you meant `import/using ..SciMLBenchmarks`.
- Otherwise, run `import Pkg; Pkg.add("SciMLBenchmarks")` to install the Sc
iMLBenchmarks package.
```


