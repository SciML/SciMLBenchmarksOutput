---
author: "Chris Rackauckas"
title: "Optimal Powerflow Nonlinear Optimization Benchmark"
---


## Data Load and Setup Code

This is generic setup code usable for all solver setups. Basically removing some unnecessary untyped dictionaries
before getting to the benchmarks.

```julia
import PowerModels
import ConcreteStructs
using BenchmarkTools

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

```
[info | PowerModels]: extending matpower format with data: areas 1x3
[info | PowerModels]: removing 1 cost terms from generator 4: [4000.0, 0.0]
[info | PowerModels]: removing 1 cost terms from generator 1: [1400.0, 0.0]
[info | PowerModels]: removing 1 cost terms from generator 5: [1000.0, 0.0]
[info | PowerModels]: removing 1 cost terms from generator 2: [1500.0, 0.0]
[info | PowerModels]: removing 1 cost terms from generator 3: [3000.0, 0.0]
[info | PowerModels]: updated generator 4 cost function with order 2 to a f
unction of order 3: [0.0, 4000.0, 0.0]
[info | PowerModels]: updated generator 1 cost function with order 2 to a f
unction of order 3: [0.0, 1400.0, 0.0]
[info | PowerModels]: updated generator 5 cost function with order 2 to a f
unction of order 3: [0.0, 1000.0, 0.0]
[info | PowerModels]: updated generator 2 cost function with order 2 to a f
unction of order 3: [0.0, 1500.0, 0.0]
[info | PowerModels]: updated generator 3 cost function with order 2 to a f
unction of order 3: [0.0, 3000.0, 0.0]
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

function build_opf_optimization_prob(dataset; adchoice = Optimization.AutoForwardDiff())
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

    optprob = Optimization.OptimizationFunction(opf_objective, adchoice; cons=opf_constraints)
    prob = Optimization.OptimizationProblem(optprob, var_init; lb=var_lb, ub=var_ub, lcons=con_lbs, ucons=con_ubs)
end

function solve_opf_optimization(dataset; adchoice = Optimization.AutoForwardDiff())
    model_build_time = @elapsed prob = build_opf_optimization_prob(dataset; adchoice)

    # Correctness tests
    @assert prob.f(prob.u0, nothing) == 0.0
    ret = zeros(length(prob.lcons))
    prob.f.cons(ret, prob.u0, nothing)
    @allocated prob.f(prob.u0, nothing) == 0
    @allocated prob.f.cons(ret, prob.u0, nothing) == 0

    solve_time_with_compilation = @elapsed sol = Optimization.solve(prob, Ipopt.Optimizer())
    cost = sol.minimum
    feasible = (sol.retcode == Optimization.SciMLBase.ReturnCode.Success)
    #println(sol.u) # solution vector

    solve_time_without_compilation = @elapsed sol = Optimization.solve(prob, Ipopt.Optimizer())
    
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
    #JuMP.set_optimizer_attribute(model, "print_level", 0)

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
    cons = [JuMP.value(x->point[x], c) for c in constraints]
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
4766, 1.1089022168490632, 1.158475092089014, 0.4972450591043296, 1.08558702
28098154, -0.38216580899243047, 0.5661321037165985, -0.5457423426033834  … 
 0.0061762164203837955, 0.2518186223833513, 0.897077248544158, 1.6334122935
95111, -1.4932615291788414, -1.5810450617647889, 1.6000837514115611, -0.306
5125522705683, 0.4202616621130535, 0.15136310228960612])
```



```julia
@assert jump_test_res[1] ≈ test_obj
```


```julia
@assert sort(abs.(jump_test_res[2])) ≈ sort(abs.(test_cons))
```

```
Error: AssertionError: sort(abs.(jump_test_res[2])) ≈ sort(abs.(test_cons))
```



```julia
println(sort(abs.(jump_test_res[2])))
```

```
[0.0061762164203837955, 0.0215258802699263, 0.06615508569119477, 0.11298343
104675009, 0.15136310228960612, 0.19476447251065077, 0.21730536348839036, 0
.2518186223833513, 0.2849522377447594, 0.3065125522705683, 0.33289810365419
437, 0.3751697141306502, 0.38216580899243047, 0.4019890236200105, 0.4202616
621130535, 0.4972450591043296, 0.5457423426033834, 0.5661321037165985, 0.58
43454392910835, 0.5950107614751935, 0.6077039991323074, 0.6138802155526912,
 0.7150251969424766, 0.7251928172073308, 0.7775962590542184, 0.854237682132
0647, 0.8834081057449231, 0.897077248544158, 1.0021074654956683, 1.01073990
30803794, 1.0688558672739048, 1.0701734802505833, 1.0793383525116511, 1.085
5870228098154, 1.1089022168490632, 1.158475092089014, 1.2740182727083802, 1
.4038702698147258, 1.4932615291788414, 1.563372246165339, 1.581045061764788
9, 1.6000837514115611, 1.633412293595111, 1.7030832394691608, 1.72525735273
33024, 2.0016715378998793, 2.8227966798520674, 3.0047739260369246, 3.161822
4299953224, 3.2236954017592256, 4.23535583005632, 7.40879932706212, 9.91531
0272572729]
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

    nlp = ADNLPModels.ADNLPModel!(opf_objective, var_init, var_lb, var_ub, opf_constraints!, con_lbs, con_ubs, backend = :optimized)
end

function solve_opf_nlpmodels(dataset)
    model_build_time = @elapsed nlp = build_opf_nlpmodels_prob(dataset)
    solve_time = @elapsed output = NLPModelsIpopt.ipopt(nlp)
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
        "time_solve" => solve_time,
        #"time_callbacks" => TBD,
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

    solve_time = @elapsed result = Nonconvex.optimize(
        model,
        IpoptAlg(),
        NonconvexCore.getinit(model);
        options = IpoptOptions(; first_order=false, symbolic=true, sparse=true),
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
        "time_solve" => solve_time,
        #"time_callbacks" => TBD,
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

    options = Optim.Options(show_trace=true)
    solve_time = @elapsed res = Optim.optimize(df, dfc, var_init, Optim.IPNewton(), options)

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
        "time_solve" => solve_time,
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

## Start the Benchmarking

```julia
file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
dataset = load_and_setup_data(file_name);
```

```
Error: IOError: stream is closed or unusable
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

This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:     1540
Number of nonzeros in inequality constraint Jacobian.:      792
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.22e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2215509237561167e-12    8.8862036950244670e-1
1
Constraint violation....:   1.3516965324811281e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302145149e-09    1.0023630520858059e-0
7
Overall NLP error.......:   2.5059076302145149e-09    1.0023630520858059e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 18.447

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:     1540
Number of nonzeros in inequality constraint Jacobian.:      792
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.22e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2215509237561167e-12    8.8862036950244670e-1
1
Constraint violation....:   1.3516965324811281e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302145149e-09    1.0023630520858059e-0
7
Overall NLP error.......:   2.5059076302145149e-09    1.0023630520858059e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.569

EXIT: Optimal Solution Found.
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.000188659
  "time_solve_compilation" => 43.2252
  "time_solve"             => 1.57175
  "feasible"               => true
```



```julia
model, res = solve_opf_jump(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      264

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        6
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       18

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308578e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572868e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 8.44e-15 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.43e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.4300297359373651e-12    9.7201189437494604e-1
1
Constraint violation....:   1.3489209749195652e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302141840e-09    1.0023630520856735e-0
7
Overall NLP error.......:   2.5059076302141840e-09    1.0023630520856735e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.525

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      264

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        6
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       18

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308578e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572868e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 8.44e-15 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.43e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.4300297359373651e-12    9.7201189437494604e-1
1
Constraint violation....:   1.3489209749195652e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302141840e-09    1.0023630520856735e-0
7
Overall NLP error.......:   2.5059076302141840e-09    1.0023630520856735e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.017

EXIT: Optimal Solution Found.
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 2.86974
  "time_solve_compilation" => 1.08693
  "time_solve"             => 0.0178948
  "feasible"               => true
```



```julia
model, res = solve_opf_nlpmodels(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       36
Number of nonzeros in Lagrangian Hessian.............:       63

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.80e-14 2.23e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2251036374334697e-12    8.9004145497338787e-1
1
Constraint violation....:   1.3544720900426910e-14    1.7985612998927536e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302144711e-09    1.0023630520857884e-0
7
Overall NLP error.......:   2.5059076302144711e-09    1.0023630520857884e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 3.067

EXIT: Optimal Solution Found.
Dict{String, Any} with 7 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 1.34624
  "time_solve"  => 3.06853
  "feasible"    => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 4.29e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 5.00e-14 2.34e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486802e+02    1.7551890899394719e+0
4
Dual infeasibility......:   2.3395132956297353e-12    9.3580531825189412e-1
1
Constraint violation....:   3.2294167340296559e-14    4.9960036108132044e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076126557705e-09    1.0023630450623082e-0
7
Overall NLP error.......:   2.5059076126557705e-09    1.0023630450623082e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.369

EXIT: Optimal Solution Found.
Dict{String, Any} with 7 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 59
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 0.119268
  "time_solve"  => 110.061
  "feasible"    => true
```



```julia
model, res = solve_opf_optim(dataset);
res
```

```
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694388021194502e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694388025767452e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694388026336165e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694388026906734e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694388027476881e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694388028108887e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694388028711103e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694388029316265e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694388029940883e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694388030569883e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694388031169234e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694388031769338e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694388032372386e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694388032975021e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694388033592024e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694388034193601e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694388034786938e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694388035426672e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694388036072048e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694388036691512e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694388037305968e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694388037957633e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.69438803860489e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694388039233447e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694388039878235e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694388040501365e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694388041139532e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694388041756594e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694388042371175e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694388042983937e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.69438804319164e9
Dict{String, Any} with 7 entries:
  "cost"        => 77.9548
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 0.000552015
  "time_solve"  => 30.1024
  "feasible"    => false
```



```julia
file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m"
dataset = load_and_setup_data(file_name);
```

```
Error: IOError: stream is closed or unusable
```



```julia
model, res = solve_opf_optimization(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:     1540
Number of nonzeros in inequality constraint Jacobian.:      792
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.22e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2215509237561167e-12    8.8862036950244670e-1
1
Constraint violation....:   1.3516965324811281e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302145149e-09    1.0023630520858059e-0
7
Overall NLP error.......:   2.5059076302145149e-09    1.0023630520858059e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.644

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:     1540
Number of nonzeros in inequality constraint Jacobian.:      792
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.22e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2215509237561167e-12    8.8862036950244670e-1
1
Constraint violation....:   1.3516965324811281e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302145149e-09    1.0023630520858059e-0
7
Overall NLP error.......:   2.5059076302145149e-09    1.0023630520858059e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.614

EXIT: Optimal Solution Found.
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.000146608
  "time_solve_compilation" => 1.64835
  "time_solve"             => 1.61735
  "feasible"               => true
```



```julia
model, res = solve_opf_jump(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      264

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        6
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       18

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308578e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572868e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 8.44e-15 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.43e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.4300297359373651e-12    9.7201189437494604e-1
1
Constraint violation....:   1.3489209749195652e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302141840e-09    1.0023630520856735e-0
7
Overall NLP error.......:   2.5059076302141840e-09    1.0023630520856735e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.017

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      264

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        6
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       18

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308578e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572868e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 8.44e-15 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.82e-14 2.43e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.4300297359373651e-12    9.7201189437494604e-1
1
Constraint violation....:   1.3489209749195652e-14    1.8207657603852567e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302141840e-09    1.0023630520856735e-0
7
Overall NLP error.......:   2.5059076302141840e-09    1.0023630520856735e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.017

EXIT: Optimal Solution Found.
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.00170842
  "time_solve_compilation" => 0.0297621
  "time_solve"             => 0.0175443
  "feasible"               => true
```



```julia
model, res = solve_opf_nlpmodels(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       36
Number of nonzeros in Lagrangian Hessian.............:       63

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       18
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        6
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066346e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7182484e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6691211e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.95e-01 2.40e-0
2f  1
   4  6.5744238e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8265929e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8541540e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572759e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.94e-01 2.00e-0
1h  1
   8  1.7308372e+04 3.63e-02 1.47e+01  -1.0 2.41e+00    -  7.66e-01 9.58e-0
1h  1
   9  1.7572883e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590632e+04 1.69e-03 1.61e-01  -1.0 5.03e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558725e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.11e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.23e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.84e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551892e+04 1.75e-07 2.11e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.82e-11 3.10e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 1.59e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 1.80e-14 2.23e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486897e+02    1.7551890838594758e+0
4
Dual infeasibility......:   2.2251036374334697e-12    8.9004145497338787e-1
1
Constraint violation....:   1.3544720900426910e-14    1.7985612998927536e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302144711e-09    1.0023630520857884e-0
7
Overall NLP error.......:   2.5059076302144711e-09    1.0023630520857884e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.040

EXIT: Optimal Solution Found.
Dict{String, Any} with 7 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 0.026254
  "time_solve"  => 0.0416916
  "feasible"    => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
This is Ipopt version 3.14.4, running with linear solver MUMPS 5.4.1.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      990

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+0
0   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-0
1h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-0
2f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-0
2f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-0
1f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-0
1h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-0
1h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-0
1h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-0
1h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+0
0h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-0
1f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-0
1h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+0
0h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-0
1h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+0
0f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+0
0h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-0
1f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+0
0f  1
  19  1.7551891e+04 4.29e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 5.00e-14 2.34e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486802e+02    1.7551890899394719e+0
4
Dual infeasibility......:   2.3395132956297353e-12    9.3580531825189412e-1
1
Constraint violation....:   3.2294167340296559e-14    4.9960036108132044e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076126557705e-09    1.0023630450623082e-0
7
Overall NLP error.......:   2.5059076126557705e-09    1.0023630450623082e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.052

EXIT: Optimal Solution Found.
Dict{String, Any} with 7 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 59
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 0.109284
  "time_solve"  => 1.39628
  "feasible"    => true
```



```julia
model, res = solve_opf_optim(dataset);
res
```

```
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694388055711191e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694388056306736e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694388056884043e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694388057461603e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694388058037071e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.69438805863252e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694388059202701e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694388059784471e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.69438806036076e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694388060935435e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694388061528528e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694388062099048e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694388062673343e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694388063248376e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694388063841136e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694388064414083e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694388064987488e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694388065559513e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694388066148238e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694388066718591e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694388067289699e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694388067861047e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694388068451676e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694388069021978e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694388069603128e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694388070171725e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694388070760819e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694388071328246e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694388071896315e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694388072465164e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694388072658899e9
Dict{String, Any} with 7 entries:
  "cost"        => 77.9548
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_build"  => 0.0168107
  "time_solve"  => 18.0287
  "feasible"    => false
```



```julia
using DataFrames, PrettyTables

function multidata_multisolver_benchmark(dataset_files)

    cases = String[]
    vars = Int[]
    cons = Int[]
    optimization_time = Float64[]
    jump_time = Float64[]
    nlpmodels_time = Float64[]
    nonconvex_time = Float64[]
    optim_time = Float64[]

    optimization_cost = Float64[]
    jump_cost = Float64[]
    nlpmodels_cost = Float64[]
    nonconvex_cost = Float64[]
    optim_cost = Float64[]

    for file in dataset_files
        @show file
        dataset = load_and_setup_data(file)
        model, res = solve_opf_optimization(dataset)
        push!(cases, split(res["case"],"/")[end])
        push!(vars, res["variables"])
        push!(cons, res["constraints"])
        push!(optimization_time, res["time_solve"])
        push!(optimization_cost, res["cost"])

        model, res = solve_opf_jump(dataset)
        push!(jump_time, res["time_solve"])
        push!(jump_cost, res["cost"])

        model, res = solve_opf_nlpmodels(dataset)
        push!(nlpmodels_time, res["time_solve"])
        push!(nlpmodels_cost, res["cost"])
        
        model, res = solve_opf_nonconvex(dataset)
        push!(nonconvex_time, res["time_solve"])
        push!(nonconvex_cost, res["cost"])

        model, res = solve_opf_optim(dataset)
        push!(optim_time, res["time_solve"])
        push!(optim_cost, res["cost"])
    end
    DataFrame(:case => cases, :vars => vars, :cons => cons, 
              :optimization => optimization_time, :optimization_cost => optimization_cost,
              :jump => jump_time, :jump_cost => jump_cost, 
              :nlpmodels => nlpmodels_time, :nlpmodels_cost => nlpmodels_cost, 
              :nonconvex => nonconvex_time, :nonconvex_cost => nonconvex_cost,
              :optim => optim_time, :optim_cost => optim_cost)
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
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lm
bd.m"
Error: IOError: stream is closed or unusable
```



```julia
pretty_table(timing_data)
```

```
Error: UndefVarError: `timing_data` not defined
```



```julia
#file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case118_ieee.m"
#dataset = load_and_setup_data(file_name);
#build_and_solve_optimization(dataset)
```

