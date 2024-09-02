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

Ensure that all objectives and constraints evaluate to the same value on a feasible point in the same dataset

```julia
test_u0 = [0.062436387733897314, 1.0711076238965598, 0.0, 1.066509799068872, -0.023231313776594726, 1.0879315976617783, -0.033094993289919016, 1.0999999581285527, 0.07121718642320936, 1.094374845084077, 0.4228458440068076, -3.7102746662566277, 1.8046846767604458e-8, -0.44810504067067086, 8.80063717152151, -0.0, 0.8709675496332583, 3.6803022758556523, -0.0, 4.618897246588245, -1.1691336178031877, 1.3748418519024668, 0.9623014391707738, -1.3174990482204871, -2.3850868109149004, 0.1445158405684026, 2.813869610747349, 0.8151138880859179, 1.9869253829584679, 3.768252275480421, 3.9998421778156934, 0.03553108302190666, 1.177155791026922, -1.3025310027752557, -0.9598988325635542, 1.3193604239530325, 2.399997991458022, -0.003103523654225171, -2.7920689620650667, -0.6047898784636468, -1.9771521474512397, -3.7071711426024025, -3.9623014391707136, 0.3313990482205271]
test_obj = 16236.704322376236
test_cons = [0.0, 2.5424107263916085e-14, -1.0835776720341528e-13, -6.039613253960852e-14, 0.0, 0.0, 0.0, -1.7075230118734908e-13, -3.9968028886505635e-14, 1.532107773982716e-13, 0.0, 6.661338147750939e-16, -1.7763568394002505e-15, 0.0, 8.881784197001252e-16, 4.440892098500626e-16, 0.0, 4.440892098500626e-16, -1.7763568394002505e-15, -8.881784197001252e-16, -4.440892098500626e-16, 2.220446049250313e-16, 0.0, 1.7763568394002505e-15, 0.0, 6.8833827526759706e-15, -8.992806499463768e-15, 3.9968028886505635e-14, -7.105427357601002e-15, 0.0, 0.0, 7.327471962526033e-15, -7.105427357601002e-15, 2.842170943040401e-14, -7.105427357601002e-15, -0.033094993289919016, 0.00986367951332429, -0.062436387733897314, 0.07121718642320936, 0.008780798689312044, 0.09444850019980408, 3.2570635340201743, 2.661827801892032, 5.709523923775402, 8.58227283683798, 18.147597689108025, 15.999999905294098, 3.0822827695389314, 2.6621176970504, 5.759999990861611, 8.161419886019171, 17.65224849471505, 15.80965802401578]
```

```
53-element Vector{Float64}:
  0.0
  2.5424107263916085e-14
 -1.0835776720341528e-13
 -6.039613253960852e-14
  0.0
  0.0
  0.0
 -1.7075230118734908e-13
 -3.9968028886505635e-14
  1.532107773982716e-13
  ⋮
  8.58227283683798
 18.147597689108025
 15.999999905294098
  3.0822827695389314
  2.6621176970504
  5.759999990861611
  8.161419886019171
 17.65224849471505
 15.80965802401578
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
(16236.704322376236, [0.0, 2.5424107263916085e-14, -1.0835776720341528e-13,
 -6.039613253960852e-14, 0.0, 0.0, 0.0, -1.7075230118734908e-13, -3.9968028
886505635e-14, 1.532107773982716e-13  …  5.709523923775402, 8.5822728368379
8, 18.147597689108025, 15.999999905294098, 3.0822827695389314, 2.6621176970
504, 5.759999990861611, 8.161419886019171, 17.65224849471505, 15.8096580240
1578])
```



```julia
@assert optimization_test_res[1] == test_obj
```


```julia
@assert optimization_test_res[2] == test_cons
```




## ModelingToolkit.jl

Showcases symbolic interface to Optimization.jl, through ModelingToolkit.jl. The simplification process in ModelingToolkit.jl transforms
the system to solve for a smaller subset of variables. As a result, while the optimization problem being solved is equivalent constraint
function values don't match. The test for this system is thus modified. The `test_u0` vector will be appropriately subsetted to match the
reduced set of variables, and the resultant point is ensured to be feasible as per the modified constraints. The subsetted point
will be used to re-generate the full point using the observed equations in the reduced system, and this point will be validated to
match `test_u0`.

```julia

import PowerModels
import Ipopt
using ModelingToolkit, Optimization, OptimizationMOI
import ModelingToolkit: ≲, unknowns
import SymbolicIndexingInterface
using SymbolicIndexingInterface: variable_symbols, all_variable_symbols, getname

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
    optsys = ModelingToolkit.complete(optsys)
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

# Given a ModelingToolkit.jl variable, translate it to the corresponding
# name it has in the dataset
function mtk_sym_to_name(sym)
    sym = ModelingToolkit.unwrap(sym)
    return string(getname(sym)) * "_" * join(ModelingToolkit.arguments(sym)[2:end], "_")
end

function test_mtk_prob(dataset, test_u0)
    prob = build_opf_mtk_prob(dataset)

    syms_subset = variable_symbols(prob)
    syms_names = map(mtk_sym_to_name, syms_subset)
    subset_idxs = [dataset.var_lookup[name] for name in syms_names]
    sub_u0 = test_u0[subset_idxs]

    objective = prob.f(sub_u0, prob.p)

    cons_buffer = zeros(length(prob.lcons))
    prob.f.cons(cons_buffer, sub_u0)

    all_syms = all_variable_symbols(prob)
    all_syms_names = map(mtk_sym_to_name, all_syms)
    all_syms_idxs = [dataset.var_lookup[name] for name in all_syms_names]
    reordered_syms = similar(all_syms)
    for (sym, idx) in zip(all_syms, all_syms_idxs)
        reordered_syms[idx] = sym
    end
    reconstructed_u0 = SymbolicIndexingInterface.observed(prob, reordered_syms)(sub_u0, prob.p)
    return objective, reconstructed_u0, cons_buffer, prob.lcons, prob.ucons
end
```

```
test_mtk_prob (generic function with 1 method)
```



```julia
objective, reconstructed_u0, cons_vals, lcons, ucons = test_mtk_prob(dataset, test_u0)
```

```
(16236.704322376236, [0.062436387733897314, 1.0711076238965598, 0.0, 1.0665
09799068872, -0.023231313776594726, 1.0879315976617783, -0.0330949932899190
16, 1.0999999581285527, 0.07121718642320936, 1.094374845084077  …  -0.95989
88325635542, 1.3193604239530325, 2.399997991458022, -0.003103523654225171, 
-2.7920689620650667, -0.6047898784636468, -1.9771521474512397, -3.707171142
6024025, -3.9623014391707136, 0.3313990482205271], [0.0, -2.542410726391608
5e-14, 0.0, 1.0782979045774244e-13, 1.7091883464104285e-13, 6.0174087934683
48e-14, 3.9968028886505635e-14, 0.0, -1.5276668818842154e-13, 0.0  …  -2.31
08919702252706e-6, -0.49535150528494754, 0.0, 1.5543122344752192e-15, 4.440
892098500626e-16, -1.5543122344752192e-15, -0.42915027539849476, -0.6180472
757981029, -9.470590205053371e-8, -0.19034197598422065], [0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, -Inf, -
Inf, -Inf, -Inf], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
```



```julia
@assert isapprox(objective, test_obj)
```


```julia
# Some pairs of variables have interchangeable values.
# Either one could be specified in terms of the other, and the choice
# is arbitrary during simplification of the symbolic system.
swappable_pairs = [
    [dataset.var_lookup["pg_1"], dataset.var_lookup["pg_2"]],
    [dataset.var_lookup["qg_1"], dataset.var_lookup["qg_2"]],
]

# Indexes that must match
non_swappable_idxs = setdiff(eachindex(reconstructed_u0), reduce(vcat, swappable_pairs))
@assert isapprox(reconstructed_u0[non_swappable_idxs], test_u0[non_swappable_idxs])
for (i, j) in swappable_pairs
    @assert isapprox(reconstructed_u0[[i, j]], test_u0[[i, j]]) || isapprox(reconstructed_u0[[j, i]], test_u0[[i, j]])
end
```


```julia
# Test all constraint values
@assert all(isapprox.(lcons, cons_vals, atol=1e-12) .|| (lcons .<= cons_vals .<= ucons) .|| isapprox.(cons_vals, ucons, atol=1e-12))
```


```julia
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
(16236.704322376236, [0.0, -2.55351295663786e-14, 0.0, 1.0835776720341528e-
13, 1.7075230118734908e-13, 6.039613253960852e-14, 3.9968028886505635e-14, 
0.0, -1.5298873279334657e-13, 4.440892098500626e-16  …  0.00878079868931204
4, 18.147597689108025, 17.65224849471505, 0.0, 7.105427357601002e-15, 0.0, 
7.105427357601002e-15, 0.09444850019980408, 15.999999905294098, 15.80965802
401578])
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
(16236.704322376236, [0.0, 2.5424107263916085e-14, -1.0835776720341528e-13,
 -6.039613253960852e-14, 0.0, 0.0, 0.0, -1.7075230118734908e-13, -3.9968028
886505635e-14, 1.532107773982716e-13  …  5.709523923775402, 8.5822728368379
8, 18.147597689108025, 15.999999905294098, 3.0822827695389314, 2.6621176970
504, 5.759999990861611, 8.161419886019171, 17.65224849471505, 15.8096580240
1578])
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
(16236.704322376236, [0.0, 2.5424107263916085e-14, 0.0, -1.0782979045774244
e-13, -1.7091883464104285e-13, -5.995204332975845e-14, -3.9968028886505635e
-14, 2.220446049250313e-16, 1.5276668818842154e-13, 0.0  …  -0.594815962021
5082, -0.45238158917508947, -2.3108919720016274e-6, -0.49535150528494754, -
0.532379574287611, -0.5148179769089868, -9.470590178750626e-8, -0.190341975
98421976, -0.6180472757981029, -0.4291502753984947])
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
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313e-16, 2
.220446049250313e-16, 4.440892098500626e-16, 4.440892098500626e-16, 4.44089
2098500626e-16, 6.661338147750939e-16, 8.881784197001252e-16, 8.88178419700
1252e-16, 1.7763568394002505e-15, 1.7763568394002505e-15, 1.776356839400250
5e-15, 6.8833827526759706e-15, 7.105427357601002e-15, 7.105427357601002e-15
, 7.105427357601002e-15, 7.327471962526033e-15, 8.992806499463768e-15, 2.54
24107263916085e-14, 2.842170943040401e-14, 3.9968028886505635e-14, 3.996802
8886505635e-14, 5.995204332975845e-14, 1.0782979045774244e-13, 1.5276668818
842154e-13, 1.7091883464104285e-13, 9.138388712415235e-9, 9.470590178750626
e-8, 2.3108919720016274e-6, 0.05047607622459793, 0.19034197598421976, 0.429
1502753984947, 0.45238158917508947, 0.4611623878644015, 0.4905037823083798,
 0.49535150528494754, 0.5137350960849745, 0.5148179769089868, 0.53237957428
7611, 0.5334624551116232, 0.5566937688882179, 0.5860351633321961, 0.5948159
620215082, 0.6180472757981029, 9.565327163162017, 9.986180113980826, 14.890
536465979823, 15.065317230461066, 15.485482302949597, 15.485772198107965]
```



```julia
println(sort(abs.(test_cons)))
```

```
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313e-
16, 4.440892098500626e-16, 4.440892098500626e-16, 4.440892098500626e-16, 6.
661338147750939e-16, 8.881784197001252e-16, 8.881784197001252e-16, 1.776356
8394002505e-15, 1.7763568394002505e-15, 1.7763568394002505e-15, 6.883382752
6759706e-15, 7.105427357601002e-15, 7.105427357601002e-15, 7.10542735760100
2e-15, 7.327471962526033e-15, 8.992806499463768e-15, 2.5424107263916085e-14
, 2.842170943040401e-14, 3.9968028886505635e-14, 3.9968028886505635e-14, 6.
039613253960852e-14, 1.0835776720341528e-13, 1.532107773982716e-13, 1.70752
30118734908e-13, 0.008780798689312044, 0.00986367951332429, 0.0330949932899
19016, 0.062436387733897314, 0.07121718642320936, 0.09444850019980408, 2.66
1827801892032, 2.6621176970504, 3.0822827695389314, 3.2570635340201743, 5.7
09523923775402, 5.759999990861611, 8.161419886019171, 8.58227283683798, 15.
80965802401578, 15.999999905294098, 17.65224849471505, 18.147597689108025]
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
(16236.704322376236, [0.0, 2.5424107263916085e-14, -1.0835776720341528e-13,
 -6.039613253960852e-14, 0.0, 0.0, 0.0, -1.7075230118734908e-13, -3.9968028
886505635e-14, 1.532107773982716e-13  …  5.709523923775402, 8.5822728368379
8, 18.147597689108025, 15.999999905294098, 3.0822827695389314, 2.6621176970
504, 5.759999990861611, 8.161419886019171, 17.65224849471505, 15.8096580240
1578])
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
  "time_build"             => 0.000181158
  "time_solve_compilation" => 16.9196
  "time_solve"             => 0.43622
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
  "time_build"             => 0.00212567
  "time_solve_compilation" => 1.04332
  "time_solve"             => 0.0126151
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
  "time_build"             => 0.654309
  "time_solve_compilation" => 3.28699
  "time_solve"             => 0.0761495
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
##WeaveSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_t
hermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentia
bleFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}
, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int
64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity
), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconve
xCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrappe
r{Main.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".v
ar"#const_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, 
DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String
, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}
, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{type
of(identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64
}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Fun
ctionWrapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSan
dBox#225".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}
, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedColle
ctions.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_
to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#u
nflatten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{Strin
g, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"
#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224
#266"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#2
58"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unf
latten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiab
leFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Dif
ferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColl
ections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCons
traint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##W
eaveSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_ther
mal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiable
Flatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, D
ifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}
, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), 
OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCo
re.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{M
ain.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"
#const_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Dif
ferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, F
loat64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, T
uple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(
identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, 
NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Functi
onWrapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBo
x#225".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, I
nt64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollecti
ons.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_
Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unfl
atten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, 
Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80
#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#26
6"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"
{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflat
ten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableF
latten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differ
entiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollect
ions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstra
int{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##Weav
eSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_thermal
_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFla
tten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Diff
erentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, T
uple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ord
eredCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.
IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main
.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"#co
nst_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differ
entiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Floa
t64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tupl
e{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(ide
ntity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Non
convexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionW
rapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBox#2
25".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, Int6
4, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections
.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tup
le#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatt
en_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Flo
at64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82
"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#266"{
Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"{Di
ct{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten
{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlat
ten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Different
iableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollection
s.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint
{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSa
ndBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_thermal_li
mit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatte
n.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differe
ntiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tupl
e{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ordere
dCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.Ine
qConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.va
r"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"#const
_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Different
iableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64
}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{I
nt64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identi
ty), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Noncon
vexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrap
per{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBox#225"
.var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, Int64, 
Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.Or
deredDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#
11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_
to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float6
4}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{N
onconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#266"{Mai
n.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"{Dict{
Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tu
ple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlatten
.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differentiab
leFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollections.O
rderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{No
nconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandB
ox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_thermal_limit
#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.U
nflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differenti
ableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{D
ifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCo
llections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCo
nstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"#
#WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"#const_th
ermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiab
leFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}},
 DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int6
4}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity)
, OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconvex
Core.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper
{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBox#225".va
r"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, Int64, Int
64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.Order
edDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"
{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_
Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float64}}
}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{Nonc
onvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#266"{Main.v
ar"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"{Dict{Int
64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple
{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableFlatten.va
r"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableF
latten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollections.Orde
redDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{Nonco
nvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#
225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_thermal_limit#25
6"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unfl
atten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiabl
eFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Diff
erentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColle
ctions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConst
raint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##We
aveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"#const_therm
al_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableF
latten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Di
fferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64},
 Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), O
rderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCor
e.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Ma
in.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBox#225".var"#
const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, Int64, Int64,
 Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedD
ict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tu
ple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dic
t#16"{typeof(identity), OrderedCollections.OrderedDict{String, Float64}}}}}
}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{Nonconv
exCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#266"{Main.var"
##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"{Dict{Int64,
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
  Float64(!Matched::IrrationalConstants.Invπ)
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
  "time_build"             => 0.000504616
  "time_solve_compilation" => 22.0266
  "time_solve"             => 17.6571
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
  "time_build"             => 0.000116609
  "time_solve_compilation" => 0.132158
  "time_solve"             => 0.0819346
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
  "time_build"             => 0.00191687
  "time_solve_compilation" => 0.00932118
  "time_solve"             => 0.0085241
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
  "time_build"             => 0.0157032
  "time_solve_compilation" => 0.0191925
  "time_solve"             => 0.0182361
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
##WeaveSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_t
hermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentia
bleFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}
, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int
64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity
), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Nonconve
xCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrappe
r{Main.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".v
ar"#const_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, 
DifferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String
, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}
, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{type
of(identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64
}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Fun
ctionWrapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSan
dBox#225".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}
, Int64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedColle
ctions.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_
to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#u
nflatten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{Strin
g, Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"
#80#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224
#266"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#2
58"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unf
latten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Differentiab
leFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Dif
ferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedColl
ections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqCons
traint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##W
eaveSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_ther
mal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differentiable
Flatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, D
ifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}
, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), 
OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCo
re.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{M
ain.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"
#const_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Dif
ferentiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, F
loat64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, T
uple{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(
identity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, 
NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.Functi
onWrapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBo
x#225".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, I
nt64, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollecti
ons.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_
Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unfl
atten_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, 
Float64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80
#82"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#26
6"{Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"
{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFlatten.Unflat
ten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, DifferentiableF
latten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, Tuple{Differ
entiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), OrderedCollect
ions.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.IneqConstra
int{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main.var"##Weav
eSandBox#225".var"#221#263"{Main.var"##WeaveSandBox#225".var"#const_thermal
_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, DifferentiableFla
tten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Float64}}, Diff
erentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tuple{Int64}, T
uple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(identity), Ord
eredCollections.OrderedDict{String, Float64}}}}}}, Float64}, NonconvexCore.
IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionWrapper{Main
.var"##WeaveSandBox#225".var"#222#264"{Main.var"##WeaveSandBox#225".var"#co
nst_thermal_limit#256"{Dict{Int64, Float64}}, Int64, Int64, Int64}}, Differ
entiableFlatten.Unflatten{Tuple{OrderedCollections.OrderedDict{String, Floa
t64}}, DifferentiableFlatten.var"#unflatten_to_Tuple#11"{Tuple{Int64}, Tupl
e{Int64}, Tuple{DifferentiableFlatten.var"#unflatten_to_Dict#16"{typeof(ide
ntity), OrderedCollections.OrderedDict{String, Float64}}}}}}, Float64}, Non
convexCore.IneqConstraint{NonconvexCore.var"#80#82"{NonconvexCore.FunctionW
rapper{Main.var"##WeaveSandBox#225".var"#223#265"{Main.var"##WeaveSandBox#2
25".var"#const_voltage_angle_difference_lb#257"{Dict{Int64, Float64}}, Int6
4, Int64, Int64}}, DifferentiableFlatten.Unflatten{Tuple{OrderedCollections
.OrderedDict{String, Float64}}, DifferentiableFlatten.var"#unflatten_to_Tup
le#11"{Tuple{Int64}, Tuple{Int64}, Tuple{DifferentiableFlatten.var"#unflatt
en_to_Dict#16"{typeof(identity), OrderedCollections.OrderedDict{String, Flo
at64}}}}}}, Float64}, NonconvexCore.IneqConstraint{NonconvexCore.var"#80#82
"{NonconvexCore.FunctionWrapper{Main.var"##WeaveSandBox#225".var"#224#266"{
Main.var"##WeaveSandBox#225".var"#const_voltage_angle_difference_ub#258"{Di
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
  Float64(!Matched::IrrationalConstants.Invπ)
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
  "time_build"             => 0.054688
  "time_solve_compilation" => 2.21626
  "time_solve"             => 2.15352
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

Number of nonzeros in equality constraint Jacobian...:       78
Number of nonzeros in inequality constraint Jacobian.:       24
Number of nonzeros in Lagrangian Hessian.............:      122

Total number of variables............................:       23
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       20
                     variables with only upper bounds:        0
Total number of equality constraints.................:       19
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.3949934e+00 1.09e+00 1.67e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1048421e+03 4.62e-01 1.04e+02  -1.0 1.81e+00    -  5.77e-03 5.76e-01h  1
   2  4.4503068e+03 1.36e-01 3.09e+01  -1.0 8.29e-01    -  8.15e-01 7.05e-01h  1
   3  4.6140270e+03 1.18e-01 2.67e+01  -1.0 2.99e-01    -  4.13e-01 1.37e-01h  1
   4  4.9343698e+03 8.25e-02 3.85e+01  -1.0 4.36e-01    -  4.94e-01 2.98e-01h  1
   5  5.4019378e+03 3.46e-02 2.51e+01  -1.0 3.62e-01    -  9.90e-01 5.81e-01h  1
   6  5.4116196e+03 3.37e-02 1.26e+02  -1.0 1.45e-01    -  1.84e-01 2.70e-02h  1
   7  5.6094774e+03 1.60e-02 5.13e+01  -1.0 2.53e-01    -  2.76e-01 5.25e-01h  1
   8  5.7031738e+03 8.53e-03 8.21e+01  -1.0 1.43e-01    -  7.00e-01 4.67e-01h  1
   9  5.8145457e+03 6.10e-04 2.19e+00  -1.0 5.77e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.8146316e+03 2.55e-05 7.83e-03  -1.0 1.23e-02    -  1.00e+00 1.00e+00h  1
  11  5.8127612e+03 1.60e-05 1.36e-02  -2.5 8.27e-03    -  1.00e+00 1.00e+00f  1
  12  5.8126464e+03 2.60e-07 1.15e-04  -3.8 1.05e-03    -  1.00e+00 1.00e+00f  1
  13  5.8126430e+03 1.32e-10 8.31e-08  -5.7 2.50e-05    -  1.00e+00 1.00e+00h  1
  14  5.8126429e+03 7.55e-15 6.25e-12  -8.6 1.78e-07    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 14

                                   (scaled)                 (unscaled)
Objective...............:   1.1625285870072360e+03    5.8126429350361796e+03
Dual infeasibility......:   6.2541517268298295e-12    3.1270758634149147e-11
Constraint violation....:   7.5495165674510645e-15    7.5495165674510645e-15
Variable bound violation:   1.0911841874516881e-08    1.0911841874516881e-08
Complementarity.........:   2.5102170848799089e-09    1.2551085424399544e-08
Overall NLP error.......:   2.5102170848799089e-09    1.2551085424399544e-08


Number of objective function evaluations             = 15
Number of objective gradient evaluations             = 15
Number of equality constraint evaluations            = 15
Number of inequality constraint evaluations          = 15
Number of equality constraint Jacobian evaluations   = 15
Number of inequality constraint Jacobian evaluations = 15
Number of Lagrangian Hessian evaluations             = 14
Total seconds in IPOPT                               = 0.871

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       78
Number of nonzeros in inequality constraint Jacobian.:       24
Number of nonzeros in Lagrangian Hessian.............:      122

Total number of variables............................:       23
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       20
                     variables with only upper bounds:        0
Total number of equality constraints.................:       19
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.3949934e+00 1.09e+00 1.67e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1048421e+03 4.62e-01 1.04e+02  -1.0 1.81e+00    -  5.77e-03 5.76e-01h  1
   2  4.4503068e+03 1.36e-01 3.09e+01  -1.0 8.29e-01    -  8.15e-01 7.05e-01h  1
   3  4.6140270e+03 1.18e-01 2.67e+01  -1.0 2.99e-01    -  4.13e-01 1.37e-01h  1
   4  4.9343698e+03 8.25e-02 3.85e+01  -1.0 4.36e-01    -  4.94e-01 2.98e-01h  1
   5  5.4019378e+03 3.46e-02 2.51e+01  -1.0 3.62e-01    -  9.90e-01 5.81e-01h  1
   6  5.4116196e+03 3.37e-02 1.26e+02  -1.0 1.45e-01    -  1.84e-01 2.70e-02h  1
   7  5.6094774e+03 1.60e-02 5.13e+01  -1.0 2.53e-01    -  2.76e-01 5.25e-01h  1
   8  5.7031738e+03 8.53e-03 8.21e+01  -1.0 1.43e-01    -  7.00e-01 4.67e-01h  1
   9  5.8145457e+03 6.10e-04 2.19e+00  -1.0 5.77e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.8146316e+03 2.55e-05 7.83e-03  -1.0 1.23e-02    -  1.00e+00 1.00e+00h  1
  11  5.8127612e+03 1.60e-05 1.36e-02  -2.5 8.27e-03    -  1.00e+00 1.00e+00f  1
  12  5.8126464e+03 2.60e-07 1.15e-04  -3.8 1.05e-03    -  1.00e+00 1.00e+00f  1
  13  5.8126430e+03 1.32e-10 8.31e-08  -5.7 2.50e-05    -  1.00e+00 1.00e+00h  1
  14  5.8126429e+03 7.55e-15 6.25e-12  -8.6 1.78e-07    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 14

                                   (scaled)                 (unscaled)
Objective...............:   1.1625285870072360e+03    5.8126429350361796e+03
Dual infeasibility......:   6.2541517268298295e-12    3.1270758634149147e-11
Constraint violation....:   7.5495165674510645e-15    7.5495165674510645e-15
Variable bound violation:   1.0911841874516881e-08    1.0911841874516881e-08
Complementarity.........:   2.5102170848799089e-09    1.2551085424399544e-08
Overall NLP error.......:   2.5102170848799089e-09    1.2551085424399544e-08


Number of objective function evaluations             = 15
Number of objective gradient evaluations             = 15
Number of equality constraint evaluations            = 15
Number of inequality constraint evaluations          = 15
Number of equality constraint Jacobian evaluations   = 15
Number of inequality constraint Jacobian evaluations = 15
Number of Lagrangian Hessian evaluations             = 14
Total seconds in IPOPT                               = 0.009

EXIT: Optimal Solution Found.
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      240

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-01h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-02f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-02f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-01f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-01h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-01h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-01h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-01h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+00h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-01f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-01h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+00h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-01h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+00f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+00h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-01f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+00f  1
  19  1.7551891e+04 5.68e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+00f  1
  21  1.7551891e+04 5.68e-14 3.38e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486898e+02    1.7551890899394759e+04
Dual infeasibility......:   3.3822003142280486e-12    1.3528801256912194e-10
Constraint violation....:   3.6743585951626306e-14    5.6843418860808015e-14
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-08
Complementarity.........:   2.5059076126917168e-09    1.0023630450766867e-07
Overall NLP error.......:   2.5059076126917168e-09    1.0023630450766867e-07


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.374

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      240

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-01h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-02f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-02f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-01f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-01h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-01h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-01h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-01h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+00h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-01f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-01h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+00h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-01h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+00f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+00h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-01f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+00f  1
  19  1.7551891e+04 5.68e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+00f  1
  21  1.7551891e+04 5.68e-14 3.38e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486898e+02    1.7551890899394759e+04
Dual infeasibility......:   3.3822003142280486e-12    1.3528801256912194e-10
Constraint violation....:   3.6743585951626306e-14    5.6843418860808015e-14
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-08
Complementarity.........:   2.5059076126917168e-09    1.0023630450766867e-07
Overall NLP error.......:   2.5059076126917168e-09    1.0023630450766867e-07


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.016

EXIT: Optimal Solution Found.
2×23 DataFrame
 Row │ case                    vars   cons   optimization  optimization_modelb ⋯
     │ String                  Int64  Int64  Float64       Float64             ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ pglib_opf_case3_lmbd.m     24     28     0.0762723                4.999 ⋯
   2 │ pglib_opf_case5_pjm.m      44     53     0.421325                 4.757
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
      <td style = "text-align: right;">0.0762723</td>
      <td style = "text-align: right;">4.999e-5</td>
      <td style = "text-align: right;">0.1407</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0172325</td>
      <td style = "text-align: right;">1.31762</td>
      <td style = "text-align: right;">3.14972</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.00758625</td>
      <td style = "text-align: right;">0.00186944</td>
      <td style = "text-align: right;">0.00811047</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0171803</td>
      <td style = "text-align: right;">0.0148622</td>
      <td style = "text-align: right;">0.0183478</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">2.21682</td>
      <td style = "text-align: right;">0.000270278</td>
      <td style = "text-align: right;">2.25415</td>
      <td style = "text-align: right;">6273.63</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case5_pjm.m</td>
      <td style = "text-align: right;">44</td>
      <td style = "text-align: right;">53</td>
      <td style = "text-align: right;">0.421325</td>
      <td style = "text-align: right;">4.7579e-5</td>
      <td style = "text-align: right;">0.420732</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0322933</td>
      <td style = "text-align: right;">0.402154</td>
      <td style = "text-align: right;">2.74896</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0121216</td>
      <td style = "text-align: right;">0.00199916</td>
      <td style = "text-align: right;">0.0125735</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0345445</td>
      <td style = "text-align: right;">0.025572</td>
      <td style = "text-align: right;">0.0352744</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">17.8314</td>
      <td style = "text-align: right;">0.000374736</td>
      <td style = "text-align: right;">17.7618</td>
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
 "/tmp/jl_bYOIDa/pglib_opf_case10000_goc.m"
 "/tmp/jl_bYOIDa/pglib_opf_case10192_epigrids.m"
 "/tmp/jl_bYOIDa/pglib_opf_case10480_goc.m"
 "/tmp/jl_bYOIDa/pglib_opf_case118_ieee.m"
 "/tmp/jl_bYOIDa/pglib_opf_case1354_pegase.m"
 "/tmp/jl_bYOIDa/pglib_opf_case13659_pegase.m"
 "/tmp/jl_bYOIDa/pglib_opf_case14_ieee.m"
 "/tmp/jl_bYOIDa/pglib_opf_case162_ieee_dtc.m"
 "/tmp/jl_bYOIDa/pglib_opf_case179_goc.m"
 "/tmp/jl_bYOIDa/pglib_opf_case1803_snem.m"
 ⋮
 "/tmp/jl_bYOIDa/pglib_opf_case6515_rte.m"
 "/tmp/jl_bYOIDa/pglib_opf_case7336_epigrids.m"
 "/tmp/jl_bYOIDa/pglib_opf_case73_ieee_rts.m"
 "/tmp/jl_bYOIDa/pglib_opf_case78484_epigrids.m"
 "/tmp/jl_bYOIDa/pglib_opf_case793_goc.m"
 "/tmp/jl_bYOIDa/pglib_opf_case8387_pegase.m"
 "/tmp/jl_bYOIDa/pglib_opf_case89_pegase.m"
 "/tmp/jl_bYOIDa/pglib_opf_case9241_pegase.m"
 "/tmp/jl_bYOIDa/pglib_opf_case9591_goc.m"
```



```julia
timing_data = multidata_multisolver_benchmark(benchmark_datasets)
```

```
file = "/tmp/jl_bYOIDa/pglib_opf_case10000_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case10192_epigrids.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case10480_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case118_ieee.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case1354_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case13659_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case14_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      489
Number of nonzeros in inequality constraint Jacobian.:      160
Number of nonzeros in Lagrangian Hessian.............:      791

Total number of variables............................:      115
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      101
                     variables with only upper bounds:        0
Total number of equality constraints.................:      109
Total number of inequality constraints...............:       80
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       80

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.1649922e+01 9.42e-01 1.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4464214e+03 2.50e-01 1.33e+02  -1.0 1.81e+00    -  5.61e-03 7.34e-01H  1
   2  2.3967630e+03 4.55e-02 8.94e+01  -1.0 1.75e+00    -  1.74e-02 9.97e-01f  1
   3  2.1633361e+03 4.42e-02 8.75e+01  -1.0 2.61e+01    -  2.18e-01 1.95e-02f  1
   4  2.1823081e+03 8.47e-04 1.41e+01  -1.0 8.82e-01    -  7.57e-01 1.00e+00h  1
   5  2.1890842e+03 3.76e-03 7.25e-01  -1.0 6.67e-02    -  1.00e+00 1.00e+00f  1
   6  2.1849633e+03 4.30e-04 5.24e-01  -1.7 1.38e-02    -  9.64e-01 1.00e+00h  1
   7  2.1794160e+03 2.31e-03 2.33e-02  -1.7 4.13e-02    -  1.00e+00 1.00e+00f  1
   8  2.1786673e+03 6.12e-04 2.89e-01  -3.8 7.90e-03    -  8.46e-01 7.40e-01h  1
   9  2.1780966e+03 1.45e-04 3.05e-01  -3.8 4.14e-02    -  4.43e-01 9.69e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  2.1780936e+03 7.00e-06 6.66e-04  -3.8 1.35e-02    -  1.00e+00 1.00e+00h  1
  11  2.1780803e+03 1.65e-06 9.68e-04  -5.7 4.57e-03    -  9.56e-01 9.45e-01h  1
  12  2.1780805e+03 1.82e-07 1.04e-05  -5.7 1.67e-03    -  1.00e+00 1.00e+00h  1
  13  2.1780804e+03 2.19e-09 2.91e-06  -8.6 1.68e-04    -  1.00e+00 9.99e-01h  1
  14  2.1780804e+03 1.10e-09 1.44e-05  -8.6 2.01e-06    -  1.00e+00 5.00e-01f  2
  15  2.1780804e+03 6.66e-14 3.85e-12  -8.6 1.01e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   9.3602396804143993e+01    2.1780804108196480e+03
Dual infeasibility......:   3.8475889141409425e-12    8.9531447152069183e-11
Constraint violation....:   6.6613381477509392e-14    6.6613381477509392e-14
Variable bound violation:   1.0340993394919451e-08    1.0340993394919451e-08
Complementarity.........:   2.5059040485519189e-09    5.8311119222354586e-08
Overall NLP error.......:   2.5059040485519189e-09    5.8311119222354586e-08


Number of objective function evaluations             = 18
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 18
Number of inequality constraint evaluations          = 18
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 4.500

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      489
Number of nonzeros in inequality constraint Jacobian.:      160
Number of nonzeros in Lagrangian Hessian.............:      791

Total number of variables............................:      115
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      101
                     variables with only upper bounds:        0
Total number of equality constraints.................:      109
Total number of inequality constraints...............:       80
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       80

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.1649922e+01 9.42e-01 1.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4464214e+03 2.50e-01 1.33e+02  -1.0 1.81e+00    -  5.61e-03 7.34e-01H  1
   2  2.3967630e+03 4.55e-02 8.94e+01  -1.0 1.75e+00    -  1.74e-02 9.97e-01f  1
   3  2.1633361e+03 4.42e-02 8.75e+01  -1.0 2.61e+01    -  2.18e-01 1.95e-02f  1
   4  2.1823081e+03 8.47e-04 1.41e+01  -1.0 8.82e-01    -  7.57e-01 1.00e+00h  1
   5  2.1890842e+03 3.76e-03 7.25e-01  -1.0 6.67e-02    -  1.00e+00 1.00e+00f  1
   6  2.1849633e+03 4.30e-04 5.24e-01  -1.7 1.38e-02    -  9.64e-01 1.00e+00h  1
   7  2.1794160e+03 2.31e-03 2.33e-02  -1.7 4.13e-02    -  1.00e+00 1.00e+00f  1
   8  2.1786673e+03 6.12e-04 2.89e-01  -3.8 7.90e-03    -  8.46e-01 7.40e-01h  1
   9  2.1780966e+03 1.45e-04 3.05e-01  -3.8 4.14e-02    -  4.43e-01 9.69e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  2.1780936e+03 7.00e-06 6.66e-04  -3.8 1.35e-02    -  1.00e+00 1.00e+00h  1
  11  2.1780803e+03 1.65e-06 9.68e-04  -5.7 4.57e-03    -  9.56e-01 9.45e-01h  1
  12  2.1780805e+03 1.82e-07 1.04e-05  -5.7 1.67e-03    -  1.00e+00 1.00e+00h  1
  13  2.1780804e+03 2.19e-09 2.91e-06  -8.6 1.68e-04    -  1.00e+00 9.99e-01h  1
  14  2.1780804e+03 1.10e-09 1.44e-05  -8.6 2.01e-06    -  1.00e+00 5.00e-01f  2
  15  2.1780804e+03 6.66e-14 3.85e-12  -8.6 1.01e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   9.3602396804143993e+01    2.1780804108196480e+03
Dual infeasibility......:   3.8475889141409425e-12    8.9531447152069183e-11
Constraint violation....:   6.6613381477509392e-14    6.6613381477509392e-14
Variable bound violation:   1.0340993394919451e-08    1.0340993394919451e-08
Complementarity.........:   2.5059040485519189e-09    5.8311119222354586e-08
Overall NLP error.......:   2.5059040485519189e-09    5.8311119222354586e-08


Number of objective function evaluations             = 18
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 18
Number of inequality constraint evaluations          = 18
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.024

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case162_ieee_dtc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case179_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case1803_snem.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case1888_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case19402_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case1951_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case197_snem.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2000_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case200_activ.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case20758_epigrids.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2312_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2383wp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case240_pserc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case24464_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case24_ieee_rts.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      979
Number of nonzeros in inequality constraint Jacobian.:      304
Number of nonzeros in Lagrangian Hessian.............:     1543

Total number of variables............................:      265
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      241
                     variables with only upper bounds:        0
Total number of equality constraints.................:      201
Total number of inequality constraints...............:      152
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      152

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.0097983e+04 2.52e+00 4.56e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.6173148e+04 1.56e+00 6.27e+01  -1.0 2.38e+00    -  1.00e-02 3.79e-01h  1
   2  5.3163376e+04 1.35e+00 5.43e+01  -1.0 6.42e+00    -  1.34e-02 1.39e-01f  1
   3  5.2196944e+04 1.33e+00 5.02e+01  -1.0 8.86e+00    -  3.43e-01 1.34e-02f  1
   4  5.3182043e+04 1.06e+00 3.73e+01  -1.0 7.58e+00    -  7.33e-01 2.03e-01h  1
   5  5.5137462e+04 3.87e-01 1.31e+01  -1.0 9.48e+00    -  9.74e-01 6.34e-01H  1
   6  5.4942906e+04 3.18e-01 1.19e+01  -1.0 1.03e+01    -  4.67e-01 1.79e-01f  1
   7  5.5363333e+04 2.73e-01 1.12e+01  -1.0 7.18e+00    -  1.00e+00 1.42e-01h  1
   8  5.5935463e+04 2.33e-01 6.62e+00  -1.0 3.08e+00    -  2.59e-01 1.44e-01h  1
   9  6.2388698e+04 6.41e-02 5.12e+00  -1.0 1.40e+00    -  6.92e-01 8.82e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  6.3791569e+04 2.81e-03 8.67e-01  -1.0 1.80e+00    -  7.56e-01 1.00e+00h  1
  11  6.3501031e+04 8.45e-04 6.21e-01  -1.7 1.69e-01    -  8.51e-01 1.00e+00f  1
  12  6.3454928e+04 1.01e-03 1.25e-02  -1.7 1.02e-01    -  1.00e+00 1.00e+00h  1
  13  6.3378078e+04 8.53e-04 1.25e-01  -3.8 1.49e-01    -  8.03e-01 7.30e-01f  1
  14  6.3364846e+04 4.37e-04 2.13e+00  -3.8 2.00e-01    -  7.95e-01 5.13e-01h  1
  15  6.3352718e+04 1.22e-04 8.47e-02  -3.8 2.47e-01    -  1.00e+00 9.79e-01h  1
  16  6.3352941e+04 1.86e-06 2.89e-05  -3.8 9.06e-03    -  1.00e+00 1.00e+00h  1
  17  6.3352216e+04 3.36e-07 5.07e-04  -5.7 9.37e-03    -  9.90e-01 9.93e-01h  1
  18  6.3352210e+04 1.17e-09 2.51e-08  -5.7 1.81e-04    -  1.00e+00 1.00e+00h  1
  19  6.3352201e+04 5.65e-11 8.23e-10  -8.6 1.04e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 19

                                   (scaled)                 (unscaled)
Objective...............:   4.8732462373895481e+02    6.3352201086064124e+04
Dual infeasibility......:   8.2331781542267553e-10    1.0703131600494782e-07
Constraint violation....:   5.6523674629715970e-11    5.6523674629715970e-11
Variable bound violation:   3.9922449346363464e-08    3.9922449346363464e-08
Complementarity.........:   3.7502813809752752e-09    4.8753657952678579e-07
Overall NLP error.......:   3.7502813809752752e-09    4.8753657952678579e-07


Number of objective function evaluations             = 21
Number of objective gradient evaluations             = 20
Number of equality constraint evaluations            = 21
Number of inequality constraint evaluations          = 21
Number of equality constraint Jacobian evaluations   = 20
Number of inequality constraint Jacobian evaluations = 20
Number of Lagrangian Hessian evaluations             = 19
Total seconds in IPOPT                               = 11.486

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      979
Number of nonzeros in inequality constraint Jacobian.:      304
Number of nonzeros in Lagrangian Hessian.............:     1543

Total number of variables............................:      265
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      241
                     variables with only upper bounds:        0
Total number of equality constraints.................:      201
Total number of inequality constraints...............:      152
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      152

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.0097983e+04 2.52e+00 4.56e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.6173148e+04 1.56e+00 6.27e+01  -1.0 2.38e+00    -  1.00e-02 3.79e-01h  1
   2  5.3163376e+04 1.35e+00 5.43e+01  -1.0 6.42e+00    -  1.34e-02 1.39e-01f  1
   3  5.2196944e+04 1.33e+00 5.02e+01  -1.0 8.86e+00    -  3.43e-01 1.34e-02f  1
   4  5.3182043e+04 1.06e+00 3.73e+01  -1.0 7.58e+00    -  7.33e-01 2.03e-01h  1
   5  5.5137462e+04 3.87e-01 1.31e+01  -1.0 9.48e+00    -  9.74e-01 6.34e-01H  1
   6  5.4942906e+04 3.18e-01 1.19e+01  -1.0 1.03e+01    -  4.67e-01 1.79e-01f  1
   7  5.5363333e+04 2.73e-01 1.12e+01  -1.0 7.18e+00    -  1.00e+00 1.42e-01h  1
   8  5.5935463e+04 2.33e-01 6.62e+00  -1.0 3.08e+00    -  2.59e-01 1.44e-01h  1
   9  6.2388698e+04 6.41e-02 5.12e+00  -1.0 1.40e+00    -  6.92e-01 8.82e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  6.3791569e+04 2.81e-03 8.67e-01  -1.0 1.80e+00    -  7.56e-01 1.00e+00h  1
  11  6.3501031e+04 8.45e-04 6.21e-01  -1.7 1.69e-01    -  8.51e-01 1.00e+00f  1
  12  6.3454928e+04 1.01e-03 1.25e-02  -1.7 1.02e-01    -  1.00e+00 1.00e+00h  1
  13  6.3378078e+04 8.53e-04 1.25e-01  -3.8 1.49e-01    -  8.03e-01 7.30e-01f  1
  14  6.3364846e+04 4.37e-04 2.13e+00  -3.8 2.00e-01    -  7.95e-01 5.13e-01h  1
  15  6.3352718e+04 1.22e-04 8.47e-02  -3.8 2.47e-01    -  1.00e+00 9.79e-01h  1
  16  6.3352941e+04 1.86e-06 2.89e-05  -3.8 9.06e-03    -  1.00e+00 1.00e+00h  1
  17  6.3352216e+04 3.36e-07 5.07e-04  -5.7 9.37e-03    -  9.90e-01 9.93e-01h  1
  18  6.3352210e+04 1.17e-09 2.51e-08  -5.7 1.81e-04    -  1.00e+00 1.00e+00h  1
  19  6.3352201e+04 5.65e-11 8.23e-10  -8.6 1.04e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 19

                                   (scaled)                 (unscaled)
Objective...............:   4.8732462373895481e+02    6.3352201086064124e+04
Dual infeasibility......:   8.2331781542267553e-10    1.0703131600494782e-07
Constraint violation....:   5.6523674629715970e-11    5.6523674629715970e-11
Variable bound violation:   3.9922449346363464e-08    3.9922449346363464e-08
Complementarity.........:   3.7502813809752752e-09    4.8753657952678579e-07
Overall NLP error.......:   3.7502813809752752e-09    4.8753657952678579e-07


Number of objective function evaluations             = 21
Number of objective gradient evaluations             = 20
Number of equality constraint evaluations            = 21
Number of inequality constraint evaluations          = 21
Number of equality constraint Jacobian evaluations   = 20
Number of inequality constraint Jacobian evaluations = 20
Number of Lagrangian Hessian evaluations             = 19
Total seconds in IPOPT                               = 0.056

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case2736sp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2737sop_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2742_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2746wop_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2746wp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2848_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2853_sdet.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2868_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case2869_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case30000_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case300_ieee.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case3012wp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case3022_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case30_as.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      999
Number of nonzeros in inequality constraint Jacobian.:      328
Number of nonzeros in Lagrangian Hessian.............:     1634

Total number of variables............................:      236
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      206
                     variables with only upper bounds:        0
Total number of equality constraints.................:      225
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.9330612e+02 7.89e-01 6.98e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  7.3680988e+02 1.06e-01 4.80e+01  -1.0 7.66e-01    -  1.76e-02 8.65e-01h  1
   2  7.6515454e+02 5.53e-02 2.51e+01  -1.0 9.70e-01    -  7.06e-01 4.79e-01h  1
   3  8.0390995e+02 9.81e-03 3.83e+00  -1.0 6.95e-01    -  1.00e+00 1.00e+00h  1
   4  8.0419681e+02 2.85e-04 1.48e-01  -1.0 1.06e-01    -  1.00e+00 1.00e+00h  1
   5  8.0355508e+02 1.29e-04 2.65e-02  -1.7 7.56e-02    -  1.00e+00 1.00e+00f  1
   6  8.0322260e+02 1.45e-04 2.45e-02  -2.5 4.50e-02    -  9.81e-01 1.00e+00f  1
   7  8.0312215e+02 2.25e-04 7.52e-02  -3.8 4.76e-02    -  8.87e-01 9.26e-01h  1
   8  8.0312831e+02 3.92e-05 5.43e-04  -3.8 1.60e-02    -  1.00e+00 1.00e+00h  1
   9  8.0312738e+02 1.51e-06 2.40e-04  -5.7 4.36e-03    -  1.00e+00 9.98e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.0312733e+02 3.89e-09 4.27e-07  -5.7 3.29e-04    -  1.00e+00 1.00e+00f  1
  11  8.0312731e+02 7.14e-11 1.82e-09  -8.6 2.73e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 11

                                   (scaled)                 (unscaled)
Objective...............:   2.4711609550316729e+02    8.0312731038529364e+02
Dual infeasibility......:   1.8152057634779339e-09    5.8994187313032853e-09
Constraint violation....:   7.1401551338112768e-11    7.1401551338112768e-11
Variable bound violation:   1.0369035186030828e-08    1.0369035186030828e-08
Complementarity.........:   3.8105847373601750e-09    1.2384400396420568e-08
Overall NLP error.......:   3.8105847373601750e-09    1.2384400396420568e-08


Number of objective function evaluations             = 12
Number of objective gradient evaluations             = 12
Number of equality constraint evaluations            = 12
Number of inequality constraint evaluations          = 12
Number of equality constraint Jacobian evaluations   = 12
Number of inequality constraint Jacobian evaluations = 12
Number of Lagrangian Hessian evaluations             = 11
Total seconds in IPOPT                               = 11.947

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      999
Number of nonzeros in inequality constraint Jacobian.:      328
Number of nonzeros in Lagrangian Hessian.............:     1634

Total number of variables............................:      236
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      206
                     variables with only upper bounds:        0
Total number of equality constraints.................:      225
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.9330612e+02 7.89e-01 6.98e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  7.3680988e+02 1.06e-01 4.80e+01  -1.0 7.66e-01    -  1.76e-02 8.65e-01h  1
   2  7.6515454e+02 5.53e-02 2.51e+01  -1.0 9.70e-01    -  7.06e-01 4.79e-01h  1
   3  8.0390995e+02 9.81e-03 3.83e+00  -1.0 6.95e-01    -  1.00e+00 1.00e+00h  1
   4  8.0419681e+02 2.85e-04 1.48e-01  -1.0 1.06e-01    -  1.00e+00 1.00e+00h  1
   5  8.0355508e+02 1.29e-04 2.65e-02  -1.7 7.56e-02    -  1.00e+00 1.00e+00f  1
   6  8.0322260e+02 1.45e-04 2.45e-02  -2.5 4.50e-02    -  9.81e-01 1.00e+00f  1
   7  8.0312215e+02 2.25e-04 7.52e-02  -3.8 4.76e-02    -  8.87e-01 9.26e-01h  1
   8  8.0312831e+02 3.92e-05 5.43e-04  -3.8 1.60e-02    -  1.00e+00 1.00e+00h  1
   9  8.0312738e+02 1.51e-06 2.40e-04  -5.7 4.36e-03    -  1.00e+00 9.98e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.0312733e+02 3.89e-09 4.27e-07  -5.7 3.29e-04    -  1.00e+00 1.00e+00f  1
  11  8.0312731e+02 7.14e-11 1.82e-09  -8.6 2.73e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 11

                                   (scaled)                 (unscaled)
Objective...............:   2.4711609550316729e+02    8.0312731038529364e+02
Dual infeasibility......:   1.8152057634779339e-09    5.8994187313032853e-09
Constraint violation....:   7.1401551338112768e-11    7.1401551338112768e-11
Variable bound violation:   1.0369035186030828e-08    1.0369035186030828e-08
Complementarity.........:   3.8105847373601750e-09    1.2384400396420568e-08
Overall NLP error.......:   3.8105847373601750e-09    1.2384400396420568e-08


Number of objective function evaluations             = 12
Number of objective gradient evaluations             = 12
Number of equality constraint evaluations            = 12
Number of inequality constraint evaluations          = 12
Number of equality constraint Jacobian evaluations   = 12
Number of inequality constraint Jacobian evaluations = 12
Number of Lagrangian Hessian evaluations             = 11
Total seconds in IPOPT                               = 0.033

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case30_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      995
Number of nonzeros in inequality constraint Jacobian.:      328
Number of nonzeros in Lagrangian Hessian.............:     1628

Total number of variables............................:      232
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      202
                     variables with only upper bounds:        0
Total number of equality constraints.................:      225
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.6429132e+01 9.42e-01 1.81e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  7.3805974e+03 1.92e-01 1.37e+02  -1.0 1.73e+00    -  5.86e-03 7.96e-01H  1
   2  7.0362117e+03 7.70e-02 5.47e+01  -1.0 1.62e+00    -  2.07e-02 6.00e-01f  1
   3  7.0544217e+03 7.45e-02 5.34e+01  -1.0 7.99e-01    -  9.68e-01 3.19e-02h  1
   4  8.2288089e+03 5.15e-04 1.49e+00  -1.0 3.21e-01    -  1.00e+00 1.00e+00h  1
   5  8.2390074e+03 3.30e-05 1.34e-01  -1.0 2.16e-02    -  1.00e+00 1.00e+00h  1
   6  8.2178583e+03 7.40e-05 4.92e-01  -2.5 2.35e-02    -  8.48e-01 9.57e-01f  1
   7  8.2101534e+03 7.44e-05 6.45e+00  -2.5 4.92e-02    -  1.00e+00 7.43e-01f  1
   8  8.2093612e+03 5.58e-06 6.10e-02  -2.5 2.33e-02    -  1.00e+00 1.00e+00f  1
   9  8.2094471e+03 1.15e-07 8.09e-03  -2.5 1.99e-03    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.2085399e+03 5.53e-06 8.48e-04  -3.8 8.62e-03    -  1.00e+00 1.00e+00f  1
  11  8.2085221e+03 8.56e-06 1.69e+01  -5.7 1.49e-03    -  8.58e-01 4.60e-01h  1
  12  8.2085142e+03 1.44e-06 1.47e-02  -5.7 1.50e-03    -  1.00e+00 1.00e+00h  1
  13  8.2085158e+03 5.61e-08 7.72e-05  -5.7 4.49e-04    -  1.00e+00 1.00e+00h  1
  14  8.2085159e+03 2.07e-09 5.33e-06  -5.7 2.37e-04    -  1.00e+00 1.00e+00h  1
  15  8.2085154e+03 5.30e-09 2.50e-03  -8.6 8.83e-05    -  9.80e-01 1.00e+00h  1
  16  8.2085154e+03 4.62e-09 2.33e-03  -8.6 1.11e-05    -  1.00e+00 1.25e-01f  4
  17  8.2085154e+03 3.63e-12 1.37e-08  -8.6 9.90e-06    -  1.00e+00 1.00e+00h  1
  18  8.2085154e+03 1.42e-14 2.54e-12  -9.0 4.10e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 18

                                   (scaled)                 (unscaled)
Objective...............:   1.5730473122733963e+02    8.2085154403067681e+03
Dual infeasibility......:   2.5374700336157655e-12    1.3241090581152641e-10
Constraint violation....:   1.4210854715202004e-14    1.4210854715202004e-14
Variable bound violation:   1.0549795703695963e-08    1.0549795703695963e-08
Complementarity.........:   9.0917662089346910e-10    4.7442885362324712e-08
Overall NLP error.......:   9.0917662089346910e-10    4.7442885362324712e-08


Number of objective function evaluations             = 23
Number of objective gradient evaluations             = 19
Number of equality constraint evaluations            = 23
Number of inequality constraint evaluations          = 23
Number of equality constraint Jacobian evaluations   = 19
Number of inequality constraint Jacobian evaluations = 19
Number of Lagrangian Hessian evaluations             = 18
Total seconds in IPOPT                               = 11.697

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      995
Number of nonzeros in inequality constraint Jacobian.:      328
Number of nonzeros in Lagrangian Hessian.............:     1628

Total number of variables............................:      232
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      202
                     variables with only upper bounds:        0
Total number of equality constraints.................:      225
Total number of inequality constraints...............:      164
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      164

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.6429132e+01 9.42e-01 1.81e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  7.3805974e+03 1.92e-01 1.37e+02  -1.0 1.73e+00    -  5.86e-03 7.96e-01H  1
   2  7.0362117e+03 7.70e-02 5.47e+01  -1.0 1.62e+00    -  2.07e-02 6.00e-01f  1
   3  7.0544217e+03 7.45e-02 5.34e+01  -1.0 7.99e-01    -  9.68e-01 3.19e-02h  1
   4  8.2288089e+03 5.15e-04 1.49e+00  -1.0 3.21e-01    -  1.00e+00 1.00e+00h  1
   5  8.2390074e+03 3.30e-05 1.34e-01  -1.0 2.16e-02    -  1.00e+00 1.00e+00h  1
   6  8.2178583e+03 7.40e-05 4.92e-01  -2.5 2.35e-02    -  8.48e-01 9.57e-01f  1
   7  8.2101534e+03 7.44e-05 6.45e+00  -2.5 4.92e-02    -  1.00e+00 7.43e-01f  1
   8  8.2093612e+03 5.58e-06 6.10e-02  -2.5 2.33e-02    -  1.00e+00 1.00e+00f  1
   9  8.2094471e+03 1.15e-07 8.09e-03  -2.5 1.99e-03    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.2085399e+03 5.53e-06 8.48e-04  -3.8 8.62e-03    -  1.00e+00 1.00e+00f  1
  11  8.2085221e+03 8.56e-06 1.69e+01  -5.7 1.49e-03    -  8.58e-01 4.60e-01h  1
  12  8.2085142e+03 1.44e-06 1.47e-02  -5.7 1.50e-03    -  1.00e+00 1.00e+00h  1
  13  8.2085158e+03 5.61e-08 7.72e-05  -5.7 4.49e-04    -  1.00e+00 1.00e+00h  1
  14  8.2085159e+03 2.07e-09 5.33e-06  -5.7 2.37e-04    -  1.00e+00 1.00e+00h  1
  15  8.2085154e+03 5.30e-09 2.50e-03  -8.6 8.83e-05    -  9.80e-01 1.00e+00h  1
  16  8.2085154e+03 4.62e-09 2.33e-03  -8.6 1.11e-05    -  1.00e+00 1.25e-01f  4
  17  8.2085154e+03 3.63e-12 1.37e-08  -8.6 9.90e-06    -  1.00e+00 1.00e+00h  1
  18  8.2085154e+03 1.42e-14 2.54e-12  -9.0 4.10e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 18

                                   (scaled)                 (unscaled)
Objective...............:   1.5730473122733963e+02    8.2085154403067681e+03
Dual infeasibility......:   2.5374700336157655e-12    1.3241090581152641e-10
Constraint violation....:   1.4210854715202004e-14    1.4210854715202004e-14
Variable bound violation:   1.0549795703695963e-08    1.0549795703695963e-08
Complementarity.........:   9.0917662089346910e-10    4.7442885362324712e-08
Overall NLP error.......:   9.0917662089346910e-10    4.7442885362324712e-08


Number of objective function evaluations             = 23
Number of objective gradient evaluations             = 19
Number of equality constraint evaluations            = 23
Number of inequality constraint evaluations          = 23
Number of equality constraint Jacobian evaluations   = 19
Number of inequality constraint Jacobian evaluations = 19
Number of Lagrangian Hessian evaluations             = 18
Total seconds in IPOPT                               = 0.054

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case3120sp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case3375wp_k.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case3970_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case39_epri.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     1125
Number of nonzeros in inequality constraint Jacobian.:      368
Number of nonzeros in Lagrangian Hessian.............:     1832

Total number of variables............................:      282
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      243
                     variables with only upper bounds:        0
Total number of equality constraints.................:      263
Total number of inequality constraints...............:      184
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      184

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.3768629e+02 1.10e+01 1.25e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.1096016e+04 1.02e+01 6.03e+01  -1.0 6.92e+00    -  1.45e-03 7.43e-02h  2
   2  2.0586599e+04 9.35e+00 5.53e+01  -1.0 1.21e+01    -  5.88e-02 8.41e-02h  4
   3  2.8045737e+04 8.65e+00 5.11e+01  -1.0 3.53e+01    -  1.19e-01 7.49e-02h  4
   4  3.4162191e+04 8.06e+00 4.76e+01  -1.0 5.29e+01    -  2.03e-01 6.79e-02h  4
   5  4.4158466e+04 7.09e+00 4.18e+01  -1.0 6.46e+01    -  2.50e-01 1.21e-01h  3
   6  5.4914496e+04 6.03e+00 3.55e+01  -1.0 7.52e+01    -  2.65e-01 1.50e-01h  2
   7  6.6043801e+04 4.95e+00 2.90e+01  -1.0 7.38e+01    -  5.73e-01 1.79e-01h  1
   8  7.6876929e+04 4.05e+00 2.38e+01  -1.0 5.15e+01    -  5.38e-01 1.80e-01h  1
   9  9.6582595e+04 2.65e+00 1.55e+01  -1.0 5.05e+01    -  3.04e-01 3.46e-01H  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.1317794e+05 1.47e+00 8.54e+00  -1.0 6.29e+01    -  3.50e-01 4.46e-01H  1
  11  1.1373986e+05 1.43e+00 1.46e+01  -1.0 3.33e+01    -  8.90e-01 2.45e-02h  1
  12  1.2662465e+05 6.62e-01 9.96e+00  -1.0 2.83e+01    -  1.00e+00 5.38e-01h  1
  13  1.3081352e+05 4.07e-01 4.65e+00  -1.0 5.24e+01    -  8.69e-01 3.86e-01h  1
  14  1.3171708e+05 3.56e-01 9.76e+00  -1.0 1.01e+01    -  3.50e-01 1.25e-01h  1
  15  1.3630115e+05 1.14e-01 2.18e+00  -1.0 6.05e+00    -  1.00e+00 6.81e-01h  1
  16  1.3815859e+05 2.30e-02 7.46e-01  -1.0 3.59e+00    -  8.11e-01 8.46e-01h  1
  17  1.3844499e+05 4.46e-03 1.00e-01  -1.7 1.19e+00    -  9.76e-01 1.00e+00h  1
  18  1.3842260e+05 3.88e-03 1.28e+00  -2.5 5.82e-01    -  1.00e+00 6.00e-01h  1
  19  1.3841745e+05 2.48e-03 9.52e+00  -2.5 5.85e-01    -  6.99e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.3841745e+05 3.85e-05 7.32e-04  -2.5 5.45e-02    -  1.00e+00 1.00e+00h  1
  21  1.3841575e+05 4.57e-05 5.16e-02  -3.8 1.08e-01    -  9.11e-01 9.17e-01h  1
  22  1.3841566e+05 7.70e-06 2.95e-04  -3.8 1.89e-02    -  1.00e+00 1.00e+00h  1
  23  1.3841557e+05 3.47e-06 1.17e-01  -5.7 9.00e-03    -  9.92e-01 9.04e-01h  1
  24  1.3841556e+05 1.69e-07 2.44e-06  -5.7 1.56e-03    -  1.00e+00 1.00e+00h  1
  25  1.3841556e+05 1.86e-10 4.07e-06  -8.6 7.90e-05    -  1.00e+00 1.00e+00h  1
  26  1.3841556e+05 1.14e-13 1.36e-11  -8.6 5.30e-08    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 26

                                   (scaled)                 (unscaled)
Objective...............:   3.9723627718148396e+03    1.3841556265037853e+05
Dual infeasibility......:   1.3577390042013769e-11    4.7309930888572472e-10
Constraint violation....:   4.9308361011817179e-14    1.1368683772161603e-13
Variable bound violation:   1.0982981990537155e-07    1.0982981990537155e-07
Complementarity.........:   2.5059244715281850e-09    8.7318043595363258e-08
Overall NLP error.......:   2.5059244715281850e-09    8.7318043595363258e-08


Number of objective function evaluations             = 55
Number of objective gradient evaluations             = 27
Number of equality constraint evaluations            = 55
Number of inequality constraint evaluations          = 55
Number of equality constraint Jacobian evaluations   = 27
Number of inequality constraint Jacobian evaluations = 27
Number of Lagrangian Hessian evaluations             = 26
Total seconds in IPOPT                               = 14.592

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     1125
Number of nonzeros in inequality constraint Jacobian.:      368
Number of nonzeros in Lagrangian Hessian.............:     1832

Total number of variables............................:      282
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      243
                     variables with only upper bounds:        0
Total number of equality constraints.................:      263
Total number of inequality constraints...............:      184
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      184

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.3768629e+02 1.10e+01 1.25e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.1096016e+04 1.02e+01 6.03e+01  -1.0 6.92e+00    -  1.45e-03 7.43e-02h  2
   2  2.0586599e+04 9.35e+00 5.53e+01  -1.0 1.21e+01    -  5.88e-02 8.41e-02h  4
   3  2.8045737e+04 8.65e+00 5.11e+01  -1.0 3.53e+01    -  1.19e-01 7.49e-02h  4
   4  3.4162191e+04 8.06e+00 4.76e+01  -1.0 5.29e+01    -  2.03e-01 6.79e-02h  4
   5  4.4158466e+04 7.09e+00 4.18e+01  -1.0 6.46e+01    -  2.50e-01 1.21e-01h  3
   6  5.4914496e+04 6.03e+00 3.55e+01  -1.0 7.52e+01    -  2.65e-01 1.50e-01h  2
   7  6.6043801e+04 4.95e+00 2.90e+01  -1.0 7.38e+01    -  5.73e-01 1.79e-01h  1
   8  7.6876929e+04 4.05e+00 2.38e+01  -1.0 5.15e+01    -  5.38e-01 1.80e-01h  1
   9  9.6582595e+04 2.65e+00 1.55e+01  -1.0 5.05e+01    -  3.04e-01 3.46e-01H  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.1317794e+05 1.47e+00 8.54e+00  -1.0 6.29e+01    -  3.50e-01 4.46e-01H  1
  11  1.1373986e+05 1.43e+00 1.46e+01  -1.0 3.33e+01    -  8.90e-01 2.45e-02h  1
  12  1.2662465e+05 6.62e-01 9.96e+00  -1.0 2.83e+01    -  1.00e+00 5.38e-01h  1
  13  1.3081352e+05 4.07e-01 4.65e+00  -1.0 5.24e+01    -  8.69e-01 3.86e-01h  1
  14  1.3171708e+05 3.56e-01 9.76e+00  -1.0 1.01e+01    -  3.50e-01 1.25e-01h  1
  15  1.3630115e+05 1.14e-01 2.18e+00  -1.0 6.05e+00    -  1.00e+00 6.81e-01h  1
  16  1.3815859e+05 2.30e-02 7.46e-01  -1.0 3.59e+00    -  8.11e-01 8.46e-01h  1
  17  1.3844499e+05 4.46e-03 1.00e-01  -1.7 1.19e+00    -  9.76e-01 1.00e+00h  1
  18  1.3842260e+05 3.88e-03 1.28e+00  -2.5 5.82e-01    -  1.00e+00 6.00e-01h  1
  19  1.3841745e+05 2.48e-03 9.52e+00  -2.5 5.85e-01    -  6.99e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.3841745e+05 3.85e-05 7.32e-04  -2.5 5.45e-02    -  1.00e+00 1.00e+00h  1
  21  1.3841575e+05 4.57e-05 5.16e-02  -3.8 1.08e-01    -  9.11e-01 9.17e-01h  1
  22  1.3841566e+05 7.70e-06 2.95e-04  -3.8 1.89e-02    -  1.00e+00 1.00e+00h  1
  23  1.3841557e+05 3.47e-06 1.17e-01  -5.7 9.00e-03    -  9.92e-01 9.04e-01h  1
  24  1.3841556e+05 1.69e-07 2.44e-06  -5.7 1.56e-03    -  1.00e+00 1.00e+00h  1
  25  1.3841556e+05 1.86e-10 4.07e-06  -8.6 7.90e-05    -  1.00e+00 1.00e+00h  1
  26  1.3841556e+05 1.14e-13 1.36e-11  -8.6 5.30e-08    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 26

                                   (scaled)                 (unscaled)
Objective...............:   3.9723627718148396e+03    1.3841556265037853e+05
Dual infeasibility......:   1.3577390042013769e-11    4.7309930888572472e-10
Constraint violation....:   4.9308361011817179e-14    1.1368683772161603e-13
Variable bound violation:   1.0982981990537155e-07    1.0982981990537155e-07
Complementarity.........:   2.5059244715281850e-09    8.7318043595363258e-08
Overall NLP error.......:   2.5059244715281850e-09    8.7318043595363258e-08


Number of objective function evaluations             = 55
Number of objective gradient evaluations             = 27
Number of equality constraint evaluations            = 55
Number of inequality constraint evaluations          = 55
Number of equality constraint Jacobian evaluations   = 27
Number of inequality constraint Jacobian evaluations = 27
Number of Lagrangian Hessian evaluations             = 26
Total seconds in IPOPT                               = 0.088

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case3_lmbd.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       78
Number of nonzeros in inequality constraint Jacobian.:       24
Number of nonzeros in Lagrangian Hessian.............:      122

Total number of variables............................:       23
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       20
                     variables with only upper bounds:        0
Total number of equality constraints.................:       19
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.3949934e+00 1.09e+00 1.67e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1048421e+03 4.62e-01 1.04e+02  -1.0 1.81e+00    -  5.77e-03 5.76e-01h  1
   2  4.4503068e+03 1.36e-01 3.09e+01  -1.0 8.29e-01    -  8.15e-01 7.05e-01h  1
   3  4.6140270e+03 1.18e-01 2.67e+01  -1.0 2.99e-01    -  4.13e-01 1.37e-01h  1
   4  4.9343698e+03 8.25e-02 3.85e+01  -1.0 4.36e-01    -  4.94e-01 2.98e-01h  1
   5  5.4019378e+03 3.46e-02 2.51e+01  -1.0 3.62e-01    -  9.90e-01 5.81e-01h  1
   6  5.4116196e+03 3.37e-02 1.26e+02  -1.0 1.45e-01    -  1.84e-01 2.70e-02h  1
   7  5.6094774e+03 1.60e-02 5.13e+01  -1.0 2.53e-01    -  2.76e-01 5.25e-01h  1
   8  5.7031738e+03 8.53e-03 8.21e+01  -1.0 1.43e-01    -  7.00e-01 4.67e-01h  1
   9  5.8145457e+03 6.10e-04 2.19e+00  -1.0 5.77e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.8146316e+03 2.55e-05 7.83e-03  -1.0 1.23e-02    -  1.00e+00 1.00e+00h  1
  11  5.8127612e+03 1.60e-05 1.36e-02  -2.5 8.27e-03    -  1.00e+00 1.00e+00f  1
  12  5.8126464e+03 2.60e-07 1.15e-04  -3.8 1.05e-03    -  1.00e+00 1.00e+00f  1
  13  5.8126430e+03 1.32e-10 8.31e-08  -5.7 2.50e-05    -  1.00e+00 1.00e+00h  1
  14  5.8126429e+03 7.55e-15 6.25e-12  -8.6 1.78e-07    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 14

                                   (scaled)                 (unscaled)
Objective...............:   1.1625285870072360e+03    5.8126429350361796e+03
Dual infeasibility......:   6.2541517268298295e-12    3.1270758634149147e-11
Constraint violation....:   7.5495165674510645e-15    7.5495165674510645e-15
Variable bound violation:   1.0911841874516881e-08    1.0911841874516881e-08
Complementarity.........:   2.5102170848799089e-09    1.2551085424399544e-08
Overall NLP error.......:   2.5102170848799089e-09    1.2551085424399544e-08


Number of objective function evaluations             = 15
Number of objective gradient evaluations             = 15
Number of equality constraint evaluations            = 15
Number of inequality constraint evaluations          = 15
Number of equality constraint Jacobian evaluations   = 15
Number of inequality constraint Jacobian evaluations = 15
Number of Lagrangian Hessian evaluations             = 14
Total seconds in IPOPT                               = 0.010

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:       78
Number of nonzeros in inequality constraint Jacobian.:       24
Number of nonzeros in Lagrangian Hessian.............:      122

Total number of variables............................:       23
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       20
                     variables with only upper bounds:        0
Total number of equality constraints.................:       19
Total number of inequality constraints...............:       12
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       12

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.3949934e+00 1.09e+00 1.67e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1048421e+03 4.62e-01 1.04e+02  -1.0 1.81e+00    -  5.77e-03 5.76e-01h  1
   2  4.4503068e+03 1.36e-01 3.09e+01  -1.0 8.29e-01    -  8.15e-01 7.05e-01h  1
   3  4.6140270e+03 1.18e-01 2.67e+01  -1.0 2.99e-01    -  4.13e-01 1.37e-01h  1
   4  4.9343698e+03 8.25e-02 3.85e+01  -1.0 4.36e-01    -  4.94e-01 2.98e-01h  1
   5  5.4019378e+03 3.46e-02 2.51e+01  -1.0 3.62e-01    -  9.90e-01 5.81e-01h  1
   6  5.4116196e+03 3.37e-02 1.26e+02  -1.0 1.45e-01    -  1.84e-01 2.70e-02h  1
   7  5.6094774e+03 1.60e-02 5.13e+01  -1.0 2.53e-01    -  2.76e-01 5.25e-01h  1
   8  5.7031738e+03 8.53e-03 8.21e+01  -1.0 1.43e-01    -  7.00e-01 4.67e-01h  1
   9  5.8145457e+03 6.10e-04 2.19e+00  -1.0 5.77e-02    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.8146316e+03 2.55e-05 7.83e-03  -1.0 1.23e-02    -  1.00e+00 1.00e+00h  1
  11  5.8127612e+03 1.60e-05 1.36e-02  -2.5 8.27e-03    -  1.00e+00 1.00e+00f  1
  12  5.8126464e+03 2.60e-07 1.15e-04  -3.8 1.05e-03    -  1.00e+00 1.00e+00f  1
  13  5.8126430e+03 1.32e-10 8.31e-08  -5.7 2.50e-05    -  1.00e+00 1.00e+00h  1
  14  5.8126429e+03 7.55e-15 6.25e-12  -8.6 1.78e-07    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 14

                                   (scaled)                 (unscaled)
Objective...............:   1.1625285870072360e+03    5.8126429350361796e+03
Dual infeasibility......:   6.2541517268298295e-12    3.1270758634149147e-11
Constraint violation....:   7.5495165674510645e-15    7.5495165674510645e-15
Variable bound violation:   1.0911841874516881e-08    1.0911841874516881e-08
Complementarity.........:   2.5102170848799089e-09    1.2551085424399544e-08
Overall NLP error.......:   2.5102170848799089e-09    1.2551085424399544e-08


Number of objective function evaluations             = 15
Number of objective gradient evaluations             = 15
Number of equality constraint evaluations            = 15
Number of inequality constraint evaluations          = 15
Number of equality constraint Jacobian evaluations   = 15
Number of inequality constraint Jacobian evaluations = 15
Number of Lagrangian Hessian evaluations             = 14
Total seconds in IPOPT                               = 0.009

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case4020_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case4601_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case4619_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case4661_sdet.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case4837_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case4917_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case500_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case5658_epigrids.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case57_ieee.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     1935
Number of nonzeros in inequality constraint Jacobian.:      640
Number of nonzeros in Lagrangian Hessian.............:     3167

Total number of variables............................:      445
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      388
                     variables with only upper bounds:        0
Total number of equality constraints.................:      435
Total number of inequality constraints...............:      320
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      320

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0503586e+02 3.76e+00 1.91e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.0716549e+04 2.65e+00 1.19e+02  -1.0 3.70e+00    -  2.74e-03 2.95e-01h  1
   2  1.3984057e+04 2.15e+00 9.71e+01  -1.0 7.02e+00    -  5.91e-02 1.90e-01h  1
   3  1.4673527e+04 2.08e+00 7.81e+01  -1.0 3.90e+00    -  3.67e-01 2.99e-02h  1
   4  2.0340384e+04 1.57e+00 5.90e+01  -1.0 4.22e+00    -  7.79e-01 2.46e-01h  1
   5  3.5210269e+04 2.14e-01 4.79e+01  -1.0 4.88e+00    -  7.60e-01 8.64e-01h  1
   6  3.6360055e+04 1.01e-01 2.37e+01  -1.0 1.06e+01    -  7.33e-01 5.27e-01h  1
   7  3.7555239e+04 1.23e-02 1.55e+00  -1.0 6.75e+00    -  1.00e+00 1.00e+00h  1
   8  3.7614438e+04 3.64e-04 1.91e-02  -1.0 1.25e+00    -  1.00e+00 1.00e+00h  1
   9  3.7591308e+04 4.39e-04 3.47e+00  -2.5 1.26e+00    -  8.64e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  3.7590571e+04 8.45e-05 2.31e+00  -2.5 1.37e-01    -  9.45e-01 8.36e-01h  1
  11  3.7590149e+04 1.43e-05 1.36e-03  -2.5 2.81e-02    -  1.00e+00 1.00e+00f  1
  12  3.7589396e+04 2.20e-06 1.90e-02  -3.8 4.24e-02    -  9.89e-01 1.00e+00h  1
  13  3.7589383e+04 8.07e-09 2.05e-06  -3.8 5.69e-04    -  1.00e+00 1.00e+00h  1
  14  3.7589339e+04 7.59e-09 1.23e-06  -5.7 2.32e-03    -  1.00e+00 1.00e+00h  1
  15  3.7589338e+04 1.86e-12 2.72e-10  -8.6 2.92e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0107655336327775e+03    3.7589338204193162e+04
Dual infeasibility......:   2.7159293080629876e-10    1.0100263800303899e-08
Constraint violation....:   1.8649361568923162e-12    1.8649361568923162e-12
Variable bound violation:   2.4448082225347889e-08    2.4448082225347889e-08
Complementarity.........:   2.6288227592950274e-09    9.7763234390144846e-08
Overall NLP error.......:   2.6288227592950274e-09    9.7763234390144846e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 35.796

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     1935
Number of nonzeros in inequality constraint Jacobian.:      640
Number of nonzeros in Lagrangian Hessian.............:     3167

Total number of variables............................:      445
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      388
                     variables with only upper bounds:        0
Total number of equality constraints.................:      435
Total number of inequality constraints...............:      320
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      320

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0503586e+02 3.76e+00 1.91e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.0716549e+04 2.65e+00 1.19e+02  -1.0 3.70e+00    -  2.74e-03 2.95e-01h  1
   2  1.3984057e+04 2.15e+00 9.71e+01  -1.0 7.02e+00    -  5.91e-02 1.90e-01h  1
   3  1.4673527e+04 2.08e+00 7.81e+01  -1.0 3.90e+00    -  3.67e-01 2.99e-02h  1
   4  2.0340384e+04 1.57e+00 5.90e+01  -1.0 4.22e+00    -  7.79e-01 2.46e-01h  1
   5  3.5210269e+04 2.14e-01 4.79e+01  -1.0 4.88e+00    -  7.60e-01 8.64e-01h  1
   6  3.6360055e+04 1.01e-01 2.37e+01  -1.0 1.06e+01    -  7.33e-01 5.27e-01h  1
   7  3.7555239e+04 1.23e-02 1.55e+00  -1.0 6.75e+00    -  1.00e+00 1.00e+00h  1
   8  3.7614438e+04 3.64e-04 1.91e-02  -1.0 1.25e+00    -  1.00e+00 1.00e+00h  1
   9  3.7591308e+04 4.39e-04 3.47e+00  -2.5 1.26e+00    -  8.64e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  3.7590571e+04 8.45e-05 2.31e+00  -2.5 1.37e-01    -  9.45e-01 8.36e-01h  1
  11  3.7590149e+04 1.43e-05 1.36e-03  -2.5 2.81e-02    -  1.00e+00 1.00e+00f  1
  12  3.7589396e+04 2.20e-06 1.90e-02  -3.8 4.24e-02    -  9.89e-01 1.00e+00h  1
  13  3.7589383e+04 8.07e-09 2.05e-06  -3.8 5.69e-04    -  1.00e+00 1.00e+00h  1
  14  3.7589339e+04 7.59e-09 1.23e-06  -5.7 2.32e-03    -  1.00e+00 1.00e+00h  1
  15  3.7589338e+04 1.86e-12 2.72e-10  -8.6 2.92e-05    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 15

                                   (scaled)                 (unscaled)
Objective...............:   1.0107655336327775e+03    3.7589338204193162e+04
Dual infeasibility......:   2.7159293080629876e-10    1.0100263800303899e-08
Constraint violation....:   1.8649361568923162e-12    1.8649361568923162e-12
Variable bound violation:   2.4448082225347889e-08    2.4448082225347889e-08
Complementarity.........:   2.6288227592950274e-09    9.7763234390144846e-08
Overall NLP error.......:   2.6288227592950274e-09    9.7763234390144846e-08


Number of objective function evaluations             = 16
Number of objective gradient evaluations             = 16
Number of equality constraint evaluations            = 16
Number of inequality constraint evaluations          = 16
Number of equality constraint Jacobian evaluations   = 16
Number of inequality constraint Jacobian evaluations = 16
Number of Lagrangian Hessian evaluations             = 15
Total seconds in IPOPT                               = 0.083

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case588_sdet.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case5_pjm.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      240

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-01h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-02f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-02f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-01f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-01h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-01h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-01h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-01h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+00h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-01f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-01h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+00h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-01h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+00f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+00h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-01f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+00f  1
  19  1.7551891e+04 5.68e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+00f  1
  21  1.7551891e+04 5.68e-14 3.38e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486898e+02    1.7551890899394759e+04
Dual infeasibility......:   3.3822003142280486e-12    1.3528801256912194e-10
Constraint violation....:   3.6743585951626306e-14    5.6843418860808015e-14
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-08
Complementarity.........:   2.5059076126917168e-09    1.0023630450766867e-07
Overall NLP error.......:   2.5059076126917168e-09    1.0023630450766867e-07


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.019

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       48
Number of nonzeros in Lagrangian Hessian.............:      240

Total number of variables............................:       44
                     variables with only lower bounds:        0
                variables with lower and upper bounds:       39
                     variables with only upper bounds:        0
Total number of equality constraints.................:       35
Total number of inequality constraints...............:       24
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:       24

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.0059989e+02 3.99e+00 2.88e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.3066305e+03 2.47e+00 1.01e+02  -1.0 2.78e+00    -  4.11e-03 3.82e-01h  1
   2  6.7181372e+03 2.36e+00 9.62e+01  -1.0 1.60e+01    -  7.37e-02 4.44e-02f  1
   3  6.6689587e+03 2.30e+00 9.34e+01  -1.0 1.30e+01    -  4.94e-01 2.40e-02f  1
   4  6.5741805e+03 2.04e+00 8.25e+01  -1.0 1.29e+01    -  3.67e-01 1.12e-01f  2
   5  6.8264259e+03 1.80e+00 7.10e+01  -1.0 1.23e+01    -  8.72e-01 1.20e-01h  2
   6  8.8540136e+03 1.08e+00 4.20e+01  -1.0 9.14e+00    -  5.92e-01 4.00e-01h  1
   7  1.0572806e+04 8.62e-01 3.58e+01  -1.0 2.94e+00    -  4.93e-01 2.00e-01h  1
   8  1.7308577e+04 3.63e-02 1.46e+01  -1.0 2.41e+00    -  7.65e-01 9.58e-01h  1
   9  1.7572869e+04 1.33e-02 1.10e+00  -1.0 2.11e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7590631e+04 1.68e-03 1.61e-01  -1.0 5.04e-01    -  1.00e+00 1.00e+00h  1
  11  1.7558724e+04 5.24e-03 5.03e-01  -2.5 6.03e-01    -  8.35e-01 9.36e-01f  1
  12  1.7553111e+04 3.34e-03 4.12e+00  -2.5 2.84e-01    -  1.00e+00 8.20e-01h  1
  13  1.7552956e+04 3.24e-05 1.26e-02  -2.5 6.35e-02    -  1.00e+00 1.00e+00h  1
  14  1.7551990e+04 1.35e-05 1.09e+00  -3.8 2.53e-02    -  1.00e+00 9.25e-01h  1
  15  1.7551938e+04 4.46e-08 1.22e-02  -3.8 7.00e-03    -  1.00e+00 1.00e+00f  1
  16  1.7551940e+04 2.35e-10 2.06e-04  -3.8 3.83e-04    -  1.00e+00 1.00e+00h  1
  17  1.7551893e+04 1.75e-07 2.10e-01  -5.7 2.49e-03    -  1.00e+00 9.68e-01f  1
  18  1.7551891e+04 6.80e-11 3.09e-05  -5.7 2.38e-04    -  1.00e+00 1.00e+00f  1
  19  1.7551891e+04 5.68e-14 6.47e-10  -5.7 5.17e-07    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.7551891e+04 6.26e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+00f  1
  21  1.7551891e+04 5.68e-14 3.38e-12  -8.6 3.33e-08    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727248486898e+02    1.7551890899394759e+04
Dual infeasibility......:   3.3822003142280486e-12    1.3528801256912194e-10
Constraint violation....:   3.6743585951626306e-14    5.6843418860808015e-14
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-08
Complementarity.........:   2.5059076126917168e-09    1.0023630450766867e-07
Overall NLP error.......:   2.5059076126917168e-09    1.0023630450766867e-07


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.018

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case60_c.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     2170
Number of nonzeros in inequality constraint Jacobian.:      704
Number of nonzeros in Lagrangian Hessian.............:     3460

Total number of variables............................:      517
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      457
                     variables with only upper bounds:        0
Total number of equality constraints.................:      473
Total number of inequality constraints...............:      352
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      352

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.8499996e+03 2.00e+01 1.41e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.3245588e+04 1.86e+01 3.73e+01  -1.0 7.57e+00    -  2.13e-03 7.03e-02h  3
   2  1.7309771e+04 1.73e+01 3.47e+01  -1.0 1.53e+01    -  3.49e-02 6.78e-02h  1
   3  1.8342944e+04 1.71e+01 3.43e+01  -1.0 4.18e+01    -  1.55e-01 1.62e-02h  1
   4  2.4426939e+04 1.55e+01 3.37e+01  -1.0 4.49e+01    -  5.05e-01 8.84e-02h  1
   5  3.5654049e+04 1.29e+01 3.04e+01  -1.0 7.04e+01    -  2.66e-01 1.69e-01H  1
   6  4.5981864e+04 1.06e+01 3.76e+01  -1.0 1.07e+02    -  7.81e-01 1.83e-01h  1
   7  6.4990845e+04 6.28e+00 2.46e+01  -1.0 1.29e+02    -  7.62e-01 4.05e-01H  1
   8  7.2941782e+04 4.45e+00 1.98e+01  -1.0 1.46e+02    -  6.50e-01 2.92e-01h  1
   9  7.8657212e+04 3.11e+00 1.34e+01  -1.0 1.50e+02    -  2.32e-01 3.01e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.2721881e+04 2.11e+00 8.87e+00  -1.0 1.56e+02    -  2.37e-01 3.22e-01h  1
  11  8.6494258e+04 1.31e+00 8.94e+00  -1.0 1.47e+02    -  8.66e-01 3.79e-01h  1
  12  9.1978730e+04 1.81e-01 3.49e+00  -1.0 1.42e+02    -  6.80e-01 8.76e-01h  1
  13  9.2499709e+04 1.79e-01 6.09e+00  -1.0 1.18e+02    -  8.21e-01 6.38e-01h  1
  14  9.2809697e+04 7.56e-03 5.99e-01  -1.0 8.99e+01    -  1.00e+00 1.00e+00H  1
  15  9.2748777e+04 2.86e-03 2.52e+00  -1.7 2.65e+01    -  9.39e-01 7.19e-01h  1
  16  9.2718202e+04 4.67e-03 8.70e-02  -1.7 1.35e+01    -  1.00e+00 1.00e+00h  1
  17  9.2701386e+04 3.06e-03 1.06e+00  -2.5 1.16e+01    -  8.99e-01 7.12e-01h  1
  18  9.2698201e+04 1.55e-03 2.09e+00  -2.5 5.16e+00    -  1.00e+00 5.76e-01h  1
  19  9.2696531e+04 5.10e-04 3.57e-03  -2.5 2.53e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  9.2694394e+04 2.74e-04 1.07e+00  -3.8 2.72e+00    -  9.84e-01 7.42e-01h  1
  21  9.2693778e+04 1.15e-04 6.40e-04  -3.8 1.13e+00    -  1.00e+00 1.00e+00h  1
  22  9.2693813e+04 4.96e-06 2.97e-05  -3.8 9.20e-02    -  1.00e+00 1.00e+00h  1
  23  9.2693672e+04 7.42e-06 1.07e-02  -5.7 2.34e-01    -  9.98e-01 9.80e-01h  1
  24  9.2693671e+04 8.19e-07 9.01e-06  -5.7 1.39e-02    -  1.00e+00 1.00e+00h  1
  25  9.2693670e+04 1.48e-07 8.95e-04  -8.6 3.09e-03    -  1.00e+00 9.48e-01h  1
  26  9.2693670e+04 5.13e-09 5.45e-08  -8.6 3.72e-04    -  1.00e+00 1.00e+00f  1
  27  9.2693670e+04 6.00e-10 3.72e-11  -8.6 2.53e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 27

                                   (scaled)                 (unscaled)
Objective...............:   3.0897889867007520e+03    9.2693669601022557e+04
Dual infeasibility......:   3.7179050873715082e-11    1.1153715262114524e-09
Constraint violation....:   6.0020255432391423e-10    6.0020255432391423e-10
Variable bound violation:   7.6218864109023343e-08    7.6218864109023343e-08
Complementarity.........:   2.5423138023358464e-09    7.6269414070075392e-08
Overall NLP error.......:   2.5423138023358464e-09    7.6269414070075392e-08


Number of objective function evaluations             = 35
Number of objective gradient evaluations             = 28
Number of equality constraint evaluations            = 35
Number of inequality constraint evaluations          = 35
Number of equality constraint Jacobian evaluations   = 28
Number of inequality constraint Jacobian evaluations = 28
Number of Lagrangian Hessian evaluations             = 27
Total seconds in IPOPT                               = 39.421

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     2170
Number of nonzeros in inequality constraint Jacobian.:      704
Number of nonzeros in Lagrangian Hessian.............:     3460

Total number of variables............................:      517
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      457
                     variables with only upper bounds:        0
Total number of equality constraints.................:      473
Total number of inequality constraints...............:      352
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      352

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.8499996e+03 2.00e+01 1.41e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.3245588e+04 1.86e+01 3.73e+01  -1.0 7.57e+00    -  2.13e-03 7.03e-02h  3
   2  1.7309771e+04 1.73e+01 3.47e+01  -1.0 1.53e+01    -  3.49e-02 6.78e-02h  1
   3  1.8342944e+04 1.71e+01 3.43e+01  -1.0 4.18e+01    -  1.55e-01 1.62e-02h  1
   4  2.4426939e+04 1.55e+01 3.37e+01  -1.0 4.49e+01    -  5.05e-01 8.84e-02h  1
   5  3.5654049e+04 1.29e+01 3.04e+01  -1.0 7.04e+01    -  2.66e-01 1.69e-01H  1
   6  4.5981864e+04 1.06e+01 3.76e+01  -1.0 1.07e+02    -  7.81e-01 1.83e-01h  1
   7  6.4990845e+04 6.28e+00 2.46e+01  -1.0 1.29e+02    -  7.62e-01 4.05e-01H  1
   8  7.2941782e+04 4.45e+00 1.98e+01  -1.0 1.46e+02    -  6.50e-01 2.92e-01h  1
   9  7.8657212e+04 3.11e+00 1.34e+01  -1.0 1.50e+02    -  2.32e-01 3.01e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  8.2721881e+04 2.11e+00 8.87e+00  -1.0 1.56e+02    -  2.37e-01 3.22e-01h  1
  11  8.6494258e+04 1.31e+00 8.94e+00  -1.0 1.47e+02    -  8.66e-01 3.79e-01h  1
  12  9.1978730e+04 1.81e-01 3.49e+00  -1.0 1.42e+02    -  6.80e-01 8.76e-01h  1
  13  9.2499709e+04 1.79e-01 6.09e+00  -1.0 1.18e+02    -  8.21e-01 6.38e-01h  1
  14  9.2809697e+04 7.56e-03 5.99e-01  -1.0 8.99e+01    -  1.00e+00 1.00e+00H  1
  15  9.2748777e+04 2.86e-03 2.52e+00  -1.7 2.65e+01    -  9.39e-01 7.19e-01h  1
  16  9.2718202e+04 4.67e-03 8.70e-02  -1.7 1.35e+01    -  1.00e+00 1.00e+00h  1
  17  9.2701386e+04 3.06e-03 1.06e+00  -2.5 1.16e+01    -  8.99e-01 7.12e-01h  1
  18  9.2698201e+04 1.55e-03 2.09e+00  -2.5 5.16e+00    -  1.00e+00 5.76e-01h  1
  19  9.2696531e+04 5.10e-04 3.57e-03  -2.5 2.53e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  9.2694394e+04 2.74e-04 1.07e+00  -3.8 2.72e+00    -  9.84e-01 7.42e-01h  1
  21  9.2693778e+04 1.15e-04 6.40e-04  -3.8 1.13e+00    -  1.00e+00 1.00e+00h  1
  22  9.2693813e+04 4.96e-06 2.97e-05  -3.8 9.20e-02    -  1.00e+00 1.00e+00h  1
  23  9.2693672e+04 7.42e-06 1.07e-02  -5.7 2.34e-01    -  9.98e-01 9.80e-01h  1
  24  9.2693671e+04 8.19e-07 9.01e-06  -5.7 1.39e-02    -  1.00e+00 1.00e+00h  1
  25  9.2693670e+04 1.48e-07 8.95e-04  -8.6 3.09e-03    -  1.00e+00 9.48e-01h  1
  26  9.2693670e+04 5.13e-09 5.45e-08  -8.6 3.72e-04    -  1.00e+00 1.00e+00f  1
  27  9.2693670e+04 6.00e-10 3.72e-11  -8.6 2.53e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 27

                                   (scaled)                 (unscaled)
Objective...............:   3.0897889867007520e+03    9.2693669601022557e+04
Dual infeasibility......:   3.7179050873715082e-11    1.1153715262114524e-09
Constraint violation....:   6.0020255432391423e-10    6.0020255432391423e-10
Variable bound violation:   7.6218864109023343e-08    7.6218864109023343e-08
Complementarity.........:   2.5423138023358464e-09    7.6269414070075392e-08
Overall NLP error.......:   2.5423138023358464e-09    7.6269414070075392e-08


Number of objective function evaluations             = 35
Number of objective gradient evaluations             = 28
Number of equality constraint evaluations            = 35
Number of inequality constraint evaluations          = 35
Number of equality constraint Jacobian evaluations   = 28
Number of inequality constraint Jacobian evaluations = 28
Number of Lagrangian Hessian evaluations             = 27
Total seconds in IPOPT                               = 0.162

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case6468_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case6470_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case6495_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case6515_rte.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case7336_epigrids.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case73_ieee_rts.m"
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     3079
Number of nonzeros in inequality constraint Jacobian.:      960
Number of nonzeros in Lagrangian Hessian.............:     4867

Total number of variables............................:      821
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      748
                     variables with only upper bounds:        0
Total number of equality constraints.................:      627
Total number of inequality constraints...............:      480
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      480

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2029395e+05 2.52e+00 4.58e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.6840177e+05 1.57e+00 6.28e+01  -1.0 2.15e+00    -  9.96e-03 3.79e-01h  1
   2  1.5863973e+05 1.38e+00 5.55e+01  -1.0 5.37e+00    -  1.33e-02 1.21e-01f  1
   3  1.5548642e+05 1.36e+00 5.17e+01  -1.0 7.22e+00    -  3.52e-01 1.30e-02f  1
   4  1.5695662e+05 1.11e+00 4.06e+01  -1.0 6.22e+00    -  7.22e-01 1.83e-01h  1
   5  1.5887394e+05 6.95e-01 2.54e+01  -1.0 7.76e+00    -  3.89e-01 3.74e-01h  1
   6  1.5957285e+05 4.99e-01 2.10e+01  -1.0 9.38e+00    -  1.00e+00 2.82e-01h  1
   7  1.6281289e+05 3.69e-01 1.55e+01  -1.0 6.50e+00    -  3.44e-01 2.60e-01h  1
   8  1.6421420e+05 3.25e-01 1.22e+01  -1.0 5.49e+00    -  1.00e+00 1.19e-01h  1
   9  1.6632915e+05 2.66e-01 7.11e+00  -1.0 3.63e+00    -  5.01e-01 1.83e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7504296e+05 1.51e-01 1.06e+01  -1.0 2.44e+00    -  6.58e-02 4.32e-01h  1
  11  1.8861399e+05 6.24e-02 1.08e+01  -1.0 2.53e+00    -  2.74e-01 8.66e-01h  1
  12  1.9092060e+05 2.17e-02 3.46e+00  -1.0 1.11e+00    -  7.53e-01 1.00e+00h  1
  13  1.9108513e+05 3.84e-03 3.77e-01  -1.0 8.49e-01    -  1.00e+00 1.00e+00h  1
  14  1.9022312e+05 1.55e-03 4.99e-01  -1.7 2.53e-01    -  8.65e-01 1.00e+00f  1
  15  1.9008605e+05 1.06e-03 1.15e-02  -1.7 1.07e-01    -  1.00e+00 1.00e+00f  1
  16  1.8987249e+05 8.00e-04 3.18e-01  -3.8 1.55e-01    -  8.03e-01 6.68e-01f  1
  17  1.8982422e+05 5.39e-04 1.68e+00  -3.8 1.76e-01    -  8.21e-01 3.99e-01f  1
  18  1.8979502e+05 3.18e-04 2.71e+00  -3.8 2.56e-01    -  8.80e-01 4.71e-01h  1
  19  1.8976750e+05 9.31e-05 3.39e-01  -3.8 1.97e-01    -  1.00e+00 9.20e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.8976634e+05 3.16e-06 4.97e-05  -3.8 1.13e-02    -  1.00e+00 1.00e+00h  1
  21  1.8976428e+05 8.44e-07 4.71e-02  -5.7 1.03e-02    -  9.81e-01 9.21e-01h  1
  22  1.8976410e+05 4.22e-08 9.88e-07  -5.7 9.91e-04    -  1.00e+00 1.00e+00h  1
  23  1.8976408e+05 3.98e-10 4.49e-07  -8.6 1.35e-04    -  1.00e+00 1.00e+00h  1
  24  1.8976408e+05 2.84e-14 1.42e-12  -8.6 6.22e-05    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 24

                                   (scaled)                 (unscaled)
Objective...............:   1.4597236705182570e+03    1.8976407716737341e+05
Dual infeasibility......:   1.4208530936394057e-12    1.8471090217312274e-10
Constraint violation....:   2.8421709430404007e-14    2.8421709430404007e-14
Variable bound violation:   3.9923579997491743e-08    3.9923579997491743e-08
Complementarity.........:   2.5067203665655705e-09    3.2587364765352413e-07
Overall NLP error.......:   2.5067203665655705e-09    3.2587364765352413e-07


Number of objective function evaluations             = 25
Number of objective gradient evaluations             = 25
Number of equality constraint evaluations            = 25
Number of inequality constraint evaluations          = 25
Number of equality constraint Jacobian evaluations   = 25
Number of inequality constraint Jacobian evaluations = 25
Number of Lagrangian Hessian evaluations             = 24
Total seconds in IPOPT                               = 94.479

EXIT: Optimal Solution Found.
This is Ipopt version 3.14.14, running with linear solver MUMPS 5.6.2.

Number of nonzeros in equality constraint Jacobian...:     3079
Number of nonzeros in inequality constraint Jacobian.:      960
Number of nonzeros in Lagrangian Hessian.............:     4867

Total number of variables............................:      821
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      748
                     variables with only upper bounds:        0
Total number of equality constraints.................:      627
Total number of inequality constraints...............:      480
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:      480

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2029395e+05 2.52e+00 4.58e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.6840177e+05 1.57e+00 6.28e+01  -1.0 2.15e+00    -  9.96e-03 3.79e-01h  1
   2  1.5863973e+05 1.38e+00 5.55e+01  -1.0 5.37e+00    -  1.33e-02 1.21e-01f  1
   3  1.5548642e+05 1.36e+00 5.17e+01  -1.0 7.22e+00    -  3.52e-01 1.30e-02f  1
   4  1.5695662e+05 1.11e+00 4.06e+01  -1.0 6.22e+00    -  7.22e-01 1.83e-01h  1
   5  1.5887394e+05 6.95e-01 2.54e+01  -1.0 7.76e+00    -  3.89e-01 3.74e-01h  1
   6  1.5957285e+05 4.99e-01 2.10e+01  -1.0 9.38e+00    -  1.00e+00 2.82e-01h  1
   7  1.6281289e+05 3.69e-01 1.55e+01  -1.0 6.50e+00    -  3.44e-01 2.60e-01h  1
   8  1.6421420e+05 3.25e-01 1.22e+01  -1.0 5.49e+00    -  1.00e+00 1.19e-01h  1
   9  1.6632915e+05 2.66e-01 7.11e+00  -1.0 3.63e+00    -  5.01e-01 1.83e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.7504296e+05 1.51e-01 1.06e+01  -1.0 2.44e+00    -  6.58e-02 4.32e-01h  1
  11  1.8861399e+05 6.24e-02 1.08e+01  -1.0 2.53e+00    -  2.74e-01 8.66e-01h  1
  12  1.9092060e+05 2.17e-02 3.46e+00  -1.0 1.11e+00    -  7.53e-01 1.00e+00h  1
  13  1.9108513e+05 3.84e-03 3.77e-01  -1.0 8.49e-01    -  1.00e+00 1.00e+00h  1
  14  1.9022312e+05 1.55e-03 4.99e-01  -1.7 2.53e-01    -  8.65e-01 1.00e+00f  1
  15  1.9008605e+05 1.06e-03 1.15e-02  -1.7 1.07e-01    -  1.00e+00 1.00e+00f  1
  16  1.8987249e+05 8.00e-04 3.18e-01  -3.8 1.55e-01    -  8.03e-01 6.68e-01f  1
  17  1.8982422e+05 5.39e-04 1.68e+00  -3.8 1.76e-01    -  8.21e-01 3.99e-01f  1
  18  1.8979502e+05 3.18e-04 2.71e+00  -3.8 2.56e-01    -  8.80e-01 4.71e-01h  1
  19  1.8976750e+05 9.31e-05 3.39e-01  -3.8 1.97e-01    -  1.00e+00 9.20e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.8976634e+05 3.16e-06 4.97e-05  -3.8 1.13e-02    -  1.00e+00 1.00e+00h  1
  21  1.8976428e+05 8.44e-07 4.71e-02  -5.7 1.03e-02    -  9.81e-01 9.21e-01h  1
  22  1.8976410e+05 4.22e-08 9.88e-07  -5.7 9.91e-04    -  1.00e+00 1.00e+00h  1
  23  1.8976408e+05 3.98e-10 4.49e-07  -8.6 1.35e-04    -  1.00e+00 1.00e+00h  1
  24  1.8976408e+05 2.84e-14 1.42e-12  -8.6 6.22e-05    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 24

                                   (scaled)                 (unscaled)
Objective...............:   1.4597236705182570e+03    1.8976407716737341e+05
Dual infeasibility......:   1.4208530936394057e-12    1.8471090217312274e-10
Constraint violation....:   2.8421709430404007e-14    2.8421709430404007e-14
Variable bound violation:   3.9923579997491743e-08    3.9923579997491743e-08
Complementarity.........:   2.5067203665655705e-09    3.2587364765352413e-07
Overall NLP error.......:   2.5067203665655705e-09    3.2587364765352413e-07


Number of objective function evaluations             = 25
Number of objective gradient evaluations             = 25
Number of equality constraint evaluations            = 25
Number of inequality constraint evaluations          = 25
Number of equality constraint Jacobian evaluations   = 25
Number of inequality constraint Jacobian evaluations = 25
Number of Lagrangian Hessian evaluations             = 24
Total seconds in IPOPT                               = 0.233

EXIT: Optimal Solution Found.
file = "/tmp/jl_bYOIDa/pglib_opf_case78484_epigrids.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case793_goc.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case8387_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case89_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case9241_pegase.m"
file = "/tmp/jl_bYOIDa/pglib_opf_case9591_goc.m"
10×23 DataFrame
 Row │ case                         vars   cons   optimization  optimization_m ⋯
     │ String                       Int64  Int64  Float64       Float64        ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ pglib_opf_case14_ieee.m        118    169     5.85981                8. ⋯
   2 │ pglib_opf_case24_ieee_rts.m    266    315    50.2096                 0.
   3 │ pglib_opf_case30_as.m          236    348    28.972                  0.
   4 │ pglib_opf_case30_ieee.m        236    348    44.3778                 0.
   5 │ pglib_opf_case39_epri.m        282    401    97.4717                 0. ⋯
   6 │ pglib_opf_case3_lmbd.m          24     28     0.0796157              5.
   7 │ pglib_opf_case57_ieee.m        448    675   159.022                  0.
   8 │ pglib_opf_case5_pjm.m           44     53     0.469747               7.
   9 │ pglib_opf_case60_c.m           518    737   161.926                  0. ⋯
  10 │ pglib_opf_case73_ieee_rts.m    824    987   335.107                  0.
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
      <td style = "text-align: right;">5.85981</td>
      <td style = "text-align: right;">8.8839e-5</td>
      <td style = "text-align: right;">6.42808</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">0.0787533</td>
      <td style = "text-align: right;">1.62002</td>
      <td style = "text-align: right;">5.95628</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">0.0154334</td>
      <td style = "text-align: right;">0.255563</td>
      <td style = "text-align: right;">0.051691</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">0.0726395</td>
      <td style = "text-align: right;">0.144123</td>
      <td style = "text-align: right;">0.0777518</td>
      <td style = "text-align: right;">2178.08</td>
      <td style = "text-align: right;">127.466</td>
      <td style = "text-align: right;">0.0732341</td>
      <td style = "text-align: right;">127.1</td>
      <td style = "text-align: right;">1658.7</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case24_ieee_rts.m</td>
      <td style = "text-align: right;">266</td>
      <td style = "text-align: right;">315</td>
      <td style = "text-align: right;">50.2096</td>
      <td style = "text-align: right;">0.000156749</td>
      <td style = "text-align: right;">50.8349</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">0.205455</td>
      <td style = "text-align: right;">4.00808</td>
      <td style = "text-align: right;">13.0159</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">0.0309108</td>
      <td style = "text-align: right;">0.0362117</td>
      <td style = "text-align: right;">0.0321842</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">0.194248</td>
      <td style = "text-align: right;">0.173607</td>
      <td style = "text-align: right;">0.332749</td>
      <td style = "text-align: right;">63352.2</td>
      <td style = "text-align: right;">388.316</td>
      <td style = "text-align: right;">0.052828</td>
      <td style = "text-align: right;">381.243</td>
      <td style = "text-align: right;">63741.2</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case30_as.m</td>
      <td style = "text-align: right;">236</td>
      <td style = "text-align: right;">348</td>
      <td style = "text-align: right;">28.972</td>
      <td style = "text-align: right;">0.000187809</td>
      <td style = "text-align: right;">28.4406</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">0.18173</td>
      <td style = "text-align: right;">3.61604</td>
      <td style = "text-align: right;">13.4924</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">0.021553</td>
      <td style = "text-align: right;">0.00495203</td>
      <td style = "text-align: right;">0.0225807</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">0.140024</td>
      <td style = "text-align: right;">0.169739</td>
      <td style = "text-align: right;">0.142165</td>
      <td style = "text-align: right;">803.127</td>
      <td style = "text-align: right;">295.234</td>
      <td style = "text-align: right;">0.146261</td>
      <td style = "text-align: right;">296.133</td>
      <td style = "text-align: right;">772.093</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case30_ieee.m</td>
      <td style = "text-align: right;">236</td>
      <td style = "text-align: right;">348</td>
      <td style = "text-align: right;">44.3778</td>
      <td style = "text-align: right;">0.000175249</td>
      <td style = "text-align: right;">44.812</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">0.205372</td>
      <td style = "text-align: right;">3.37621</td>
      <td style = "text-align: right;">13.2592</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">0.0301136</td>
      <td style = "text-align: right;">0.00515074</td>
      <td style = "text-align: right;">0.181655</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">0.211708</td>
      <td style = "text-align: right;">0.161883</td>
      <td style = "text-align: right;">0.214189</td>
      <td style = "text-align: right;">8208.52</td>
      <td style = "text-align: right;">296.548</td>
      <td style = "text-align: right;">0.0456094</td>
      <td style = "text-align: right;">296.286</td>
      <td style = "text-align: right;">4244.05</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case39_epri.m</td>
      <td style = "text-align: right;">282</td>
      <td style = "text-align: right;">401</td>
      <td style = "text-align: right;">97.4717</td>
      <td style = "text-align: right;">0.000189588</td>
      <td style = "text-align: right;">95.4604</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">0.276064</td>
      <td style = "text-align: right;">4.34619</td>
      <td style = "text-align: right;">16.2012</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">0.0509085</td>
      <td style = "text-align: right;">0.00552338</td>
      <td style = "text-align: right;">0.0519264</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">0.323011</td>
      <td style = "text-align: right;">0.203323</td>
      <td style = "text-align: right;">0.324828</td>
      <td style = "text-align: right;">1.38416e5</td>
      <td style = "text-align: right;">376.71</td>
      <td style = "text-align: right;">0.335883</td>
      <td style = "text-align: right;">373.075</td>
      <td style = "text-align: right;">78346.0</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case3_lmbd.m</td>
      <td style = "text-align: right;">24</td>
      <td style = "text-align: right;">28</td>
      <td style = "text-align: right;">0.0796157</td>
      <td style = "text-align: right;">5.698e-5</td>
      <td style = "text-align: right;">0.105444</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0181882</td>
      <td style = "text-align: right;">0.182174</td>
      <td style = "text-align: right;">0.0198149</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.00913632</td>
      <td style = "text-align: right;">0.00461632</td>
      <td style = "text-align: right;">0.00979563</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">0.0189162</td>
      <td style = "text-align: right;">0.0158461</td>
      <td style = "text-align: right;">0.0197862</td>
      <td style = "text-align: right;">5812.64</td>
      <td style = "text-align: right;">2.35775</td>
      <td style = "text-align: right;">0.000425226</td>
      <td style = "text-align: right;">2.33142</td>
      <td style = "text-align: right;">6273.63</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case57_ieee.m</td>
      <td style = "text-align: right;">448</td>
      <td style = "text-align: right;">675</td>
      <td style = "text-align: right;">159.022</td>
      <td style = "text-align: right;">0.000268028</td>
      <td style = "text-align: right;">142.471</td>
      <td style = "text-align: right;">36353.6</td>
      <td style = "text-align: right;">0.470279</td>
      <td style = "text-align: right;">12.4449</td>
      <td style = "text-align: right;">37.591</td>
      <td style = "text-align: right;">37589.3</td>
      <td style = "text-align: right;">0.0452667</td>
      <td style = "text-align: right;">0.00759491</td>
      <td style = "text-align: right;">0.0456267</td>
      <td style = "text-align: right;">37589.3</td>
      <td style = "text-align: right;">0.283151</td>
      <td style = "text-align: right;">0.483329</td>
      <td style = "text-align: right;">0.287178</td>
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
      <td style = "text-align: right;">0.469747</td>
      <td style = "text-align: right;">7.1109e-5</td>
      <td style = "text-align: right;">0.380704</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0344086</td>
      <td style = "text-align: right;">0.352682</td>
      <td style = "text-align: right;">0.0360302</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0142296</td>
      <td style = "text-align: right;">0.00468598</td>
      <td style = "text-align: right;">0.0149376</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">0.0384412</td>
      <td style = "text-align: right;">0.0286415</td>
      <td style = "text-align: right;">0.039405</td>
      <td style = "text-align: right;">17551.9</td>
      <td style = "text-align: right;">19.142</td>
      <td style = "text-align: right;">0.000629054</td>
      <td style = "text-align: right;">19.1821</td>
      <td style = "text-align: right;">77.9548</td>
    </tr>
    <tr>
      <td style = "text-align: right;">pglib_opf_case60_c.m</td>
      <td style = "text-align: right;">518</td>
      <td style = "text-align: right;">737</td>
      <td style = "text-align: right;">161.926</td>
      <td style = "text-align: right;">0.000274777</td>
      <td style = "text-align: right;">143.511</td>
      <td style = "text-align: right;">35648.5</td>
      <td style = "text-align: right;">0.608479</td>
      <td style = "text-align: right;">14.4639</td>
      <td style = "text-align: right;">41.2895</td>
      <td style = "text-align: right;">92693.7</td>
      <td style = "text-align: right;">0.0799585</td>
      <td style = "text-align: right;">0.0301956</td>
      <td style = "text-align: right;">0.0815161</td>
      <td style = "text-align: right;">92693.7</td>
      <td style = "text-align: right;">0.746166</td>
      <td style = "text-align: right;">0.504203</td>
      <td style = "text-align: right;">0.632022</td>
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
      <td style = "text-align: right;">335.107</td>
      <td style = "text-align: right;">0.000331947</td>
      <td style = "text-align: right;">324.897</td>
      <td style = "text-align: right;">1.5864e5</td>
      <td style = "text-align: right;">1.14022</td>
      <td style = "text-align: right;">27.5426</td>
      <td style = "text-align: right;">96.8314</td>
      <td style = "text-align: right;">1.89764e5</td>
      <td style = "text-align: right;">0.322377</td>
      <td style = "text-align: right;">0.0122453</td>
      <td style = "text-align: right;">0.101324</td>
      <td style = "text-align: right;">1.89764e5</td>
      <td style = "text-align: right;">1.00862</td>
      <td style = "text-align: right;">0.622706</td>
      <td style = "text-align: right;">0.878662</td>
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


