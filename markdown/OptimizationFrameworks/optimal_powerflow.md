---
author: "Chris Rackauckas"
title: "Optimal Powerflow Nonlinear Optimization Benchmark"
---


## Data Load and Setup Code

This is generic setup code usable for all solver setups. Basically removing some unnecessary untyped dictionaries
before getting to the benchmarks.

```julia
PRINT_LEVEL = 0
MAX_CPU_TIME = 100.0
```

```
100.0
```



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

    optprob = Optimization.OptimizationFunction(opf_objective, adchoice; cons=opf_constraints)
    prob = Optimization.OptimizationProblem(optprob, var_init; lb=var_lb, ub=var_ub, lcons=con_lbs, ucons=con_ubs)
end

function solve_opf_optimization(dataset; adchoice = Optimization.AutoSparseReverseDiff())
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
        options = IpoptOptions(; first_order=false, symbolic=true, sparse=true, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME),
    )

    solve_time_without_compilation = @elapsed result = Nonconvex.optimize(
        model,
        IpoptAlg(),
        NonconvexCore.getinit(model);
        options = IpoptOptions(; first_order=false, symbolic=true, sparse=true, print_level = PRINT_LEVEL, max_cpu_time = MAX_CPU_TIME),
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

    options = Optim.Options(show_trace=PRINT_LEVEL == 0,time_limit=MAX_CPU_TIME)
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
  "time_build"             => 0.000175299
  "time_solve_compilation" => 44.5058
  "time_solve"             => 0.612124
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
  "time_build"             => 2.85569
  "time_solve_compilation" => 0.900935
  "time_solve"             => 0.0177942
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
  "time_build"             => 1.007
  "time_solve_compilation" => 3.32775
  "time_solve"             => 0.088694
  "feasible"               => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 17551.9
  "variables"              => 44
  "constraints"            => 59
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.0898298
  "time_solve_compilation" => 113.041
  "time_solve"             => 1.39275
  "feasible"               => true
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
 * time: 1.69455105617939e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694551060500263e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694551061137237e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694551061773226e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.69455106240885e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.69455106306876e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694551063697348e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694551064333302e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694551064967926e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694551065628767e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694551066262993e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694551066898138e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694551067533869e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694551068194976e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694551068827774e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.69455106946116e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.69455107009696e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694551070731559e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694551071388031e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694551072020374e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694551072653801e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694551073286308e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694551073942449e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694551074574335e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694551075206555e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.69455107584013e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694551076496175e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694551077126758e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694551077758963e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694551078390503e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694551078613725e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694551079801079e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694551080458835e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694551081087079e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694551081717841e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694551082349246e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694551083006335e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694551083630093e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694551084259918e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694551084890279e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694551085520947e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694551086175345e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694551086803598e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694551087433503e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694551088062175e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694551088718532e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694551089346206e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694551089975642e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694551090604297e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694551091258441e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.69455109188432e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694551092513707e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694551093142843e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694551093798461e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694551094425635e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694551095053963e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694551095683015e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694551096313665e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694551096972953e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694551097601749e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694551098230844e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694551098453568e9
Dict{String, Any} with 8 entries:
  "cost"                   => 77.9548
  "variables"              => 44
  "constraints"            => 53
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.000544005
  "time_solve_compilation" => 29.8954
  "time_solve"             => 19.8398
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
  "time_build"             => 7.4259e-5
  "time_solve_compilation" => 5.03163
  "time_solve"             => 0.183199
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
  "time_build"             => 0.00163046
  "time_solve_compilation" => 0.00933164
  "time_solve"             => 0.00869074
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
  "time_build"             => 0.0136088
  "time_solve_compilation" => 0.0192469
  "time_solve"             => 0.0184632
  "feasible"               => true
```



```julia
model, res = solve_opf_nonconvex(dataset);
res
```

```
Dict{String, Any} with 8 entries:
  "cost"                   => 5812.64
  "variables"              => 24
  "constraints"            => 31
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.0206367
  "time_solve_compilation" => 3.89369
  "time_solve"             => 0.664286
  "feasible"               => true
```



```julia
model, res = solve_opf_optim(dataset);
res
```

```
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.69455110882973e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.694551108951053e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694551109065561e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694551109181968e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694551109298323e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.694551109414922e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.69455110953175e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.694551109648883e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.694551109766021e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.694551109882925e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.69455110999827e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.69455111011335e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694551110265608e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694551110393801e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694551110523301e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.69455111065959e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.69455111082157e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551110881274e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.694551111100427e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.694551111214955e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694551111329739e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694551111468116e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694551111584666e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.694551111701311e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.694551111817781e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.69455111193424e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.694551112068921e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.694551112186396e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.694551112301817e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.694551112416502e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694551112548351e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694551112698232e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694551112826462e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.694551112962376e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551113125521e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551113183538e9
Dict{String, Any} with 8 entries:
  "cost"                   => 6273.63
  "variables"              => 24
  "constraints"            => 28
  "case"                   => "../../benchmarks/OptimizationFrameworks/opf_
data…
  "time_build"             => 0.0878533
  "time_solve_compilation" => 2.28765
  "time_solve"             => 2.30226
  "feasible"               => false
```



```julia
using DataFrames, PrettyTables

function multidata_multisolver_benchmark(dataset_files; sizelimit = 300)

    cases = String[]
    vars = Int[]
    cons = Int[]

    optimization_time = Float64[]
    jump_time = Float64[]
    nlpmodels_time = Float64[]
    nonconvex_time = Float64[]
    optim_time = Float64[]

    optimization_time_modelbuild = Float64[]
    jump_time_modelbuild = Float64[]
    nlpmodels_time_modelbuild = Float64[]
    nonconvex_time_modelbuild = Float64[]
    optim_time_modelbuild = Float64[]

    optimization_time_compilation = Float64[]
    jump_time_compilation = Float64[]
    nlpmodels_time_compilation = Float64[]
    nonconvex_time_compilation = Float64[]
    optim_time_compilation = Float64[]

    optimization_cost = Float64[]
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
            @info "Variable size over limit. Skipping for now"
            continue
        end
        
        model, res = solve_opf_optimization(dataset)
        push!(cases, split(res["case"],"/")[end])
        push!(vars, res["variables"])
        push!(cons, res["constraints"])
        push!(optimization_time, res["time_solve"])
        push!(optimization_time_modelbuild, res["time_build"])
        push!(optimization_time_compilation, res["time_solve_compilation"])
        push!(optimization_cost, res["cost"])

        model, res = solve_opf_jump(dataset)
        push!(jump_time, res["time_solve"])
        push!(jump_time_modelbuild, res["time_build"])
        push!(jump_time_compilation, res["time_solve_compilation"])
        push!(jump_cost, res["cost"])

        model, res = solve_opf_nlpmodels(dataset)
        push!(nlpmodels_time, res["time_solve"])
        push!(nlpmodels_time_modelbuild, res["time_build"])
        push!(nlpmodels_time_compilation, res["time_solve_compilation"])
        push!(nlpmodels_cost, res["cost"])
        
        model, res = solve_opf_nonconvex(dataset)
        push!(nonconvex_time, res["time_solve"])
        push!(nonconvex_time_modelbuild, res["time_build"])
        push!(nonconvex_time_compilation, res["time_solve_compilation"])
        push!(nonconvex_cost, res["cost"])

        model, res = solve_opf_optim(dataset)
        push!(optim_time, res["time_solve"])
        push!(optim_time_modelbuild, res["time_build"])
        push!(optim_time_compilation, res["time_solve_compilation"])
        push!(optim_cost, res["cost"])
    end
    DataFrame(:case => cases, :vars => vars, :cons => cons, 
              :optimization => optimization_time, :optimization_modelbuild => optimization_time_modelbuild, :optimization_wcompilation => optimization_time_compilation, :optimization_cost => optimization_cost,
              :jump => jump_time, :jump_modelbuild => jump_time_modelbuild, :jump_wcompilation => jump_time_compilation, :jump_cost => jump_cost, 
              :nlpmodels => nlpmodels_time, :nlpmodels_modelbuild => nlpmodels_time_modelbuild, :nlpmodels_wcompilation => nlpmodels_time_compilation,  :nlpmodels_cost => nlpmodels_cost, 
              :nonconvex => nonconvex_time, :nonconvex_modelbuild => nonconvex_time_modelbuild, :nonconvex_wcompilation => nonconvex_time_compilation,  :nonconvex_cost => nonconvex_cost,
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
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lm
bd.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.694551148351231e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.69455115139224e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694551151502516e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694551151613276e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694551151724945e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.694551151836112e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.694551151947435e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.694551152078582e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.694551152189648e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.694551152300874e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.694551152410036e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.694551152519362e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694551152643505e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694551152766471e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694551152910051e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.694551153038066e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551153171698e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551153248127e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.694551153455847e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.694551153564677e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694551153673889e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694551153784435e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694551153895541e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.69455115400615e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.694551154117468e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.69455115422876e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.694551154340345e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.69455115447098e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.694551154579854e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.694551154689039e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694551154813416e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694551154936064e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694551155059132e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.694551155207663e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551155341571e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694551155417846e9
file = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pj
m.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694551163494984e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694551164117355e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694551164721136e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694551165322887e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694551165925473e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694551166548367e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.69455116714527e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694551167748089e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694551168352224e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694551168976944e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694551169579818e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694551170184135e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694551170793644e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694551171437377e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694551172051462e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.6945511726762e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694551173285369e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694551173900114e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694551174530914e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694551175141008e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694551175746759e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694551176353178e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694551176979106e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694551177585546e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694551178191489e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694551178798126e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694551179424913e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.69455118003156e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694551180637023e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694551181242525e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694551181454325e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694551182591981e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.69455118321885e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694551183823369e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694551184428981e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694551185034025e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694551185659613e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694551186259009e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694551186864549e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694551187470416e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694551188074705e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694551188699568e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694551189304211e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694551189908663e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694551190513352e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.69455119113873e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694551191743327e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694551192347849e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694551192952458e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694551193578386e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694551194182342e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694551194792003e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694551195395853e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694551196020747e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694551196625046e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694551197229516e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694551197834555e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694551198440343e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694551199064287e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694551199668221e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694551200271316e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694551200482295e9
2×23 DataFrame
 Row │ case                    vars   cons   optimization  optimization_mod
elb ⋯
     │ String                  Int64  Int64  Float64       Float64         
    ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │ pglib_opf_case3_lmbd.m     24     28      0.165133                5.
203 ⋯
   2 │ pglib_opf_case3_lmbd.m     44     53      0.623138                4.
577
                                                              19 columns om
itted
```



```julia
pretty_table(timing_data)
```

```
┌────────────────────────┬───────┬───────┬──────────────┬──────────────────
───────┬───────────────────────────┬───────────────────┬────────────┬──────
───────────┬───────────────────┬───────────┬───────────┬───────────────────
───┬────────────────────────┬────────────────┬───────────┬─────────────────
─────┬────────────────────────┬────────────────┬─────────┬─────────────────
─┬────────────────────┬────────────┐
│                   case │  vars │  cons │ optimization │ optimization_mode
lbuild │ optimization_wcompilation │ optimization_cost │       jump │ jump_
modelbuild │ jump_wcompilation │ jump_cost │ nlpmodels │ nlpmodels_modelbui
ld │ nlpmodels_wcompilation │ nlpmodels_cost │ nonconvex │ nonconvex_modelb
uild │ nonconvex_wcompilation │ nonconvex_cost │   optim │ optim_modelbuild
 │ optim_wcompilation │ optim_cost │
│                 String │ Int64 │ Int64 │      Float64 │                 F
loat64 │                   Float64 │           Float64 │    Float64 │      
   Float64 │           Float64 │   Float64 │   Float64 │              Float
64 │                Float64 │        Float64 │   Float64 │              Flo
at64 │                Float64 │        Float64 │ Float64 │          Float64
 │            Float64 │    Float64 │
├────────────────────────┼───────┼───────┼──────────────┼──────────────────
───────┼───────────────────────────┼───────────────────┼────────────┼──────
───────────┼───────────────────┼───────────┼───────────┼───────────────────
───┼────────────────────────┼────────────────┼───────────┼─────────────────
─────┼────────────────────────┼────────────────┼─────────┼─────────────────
─┼────────────────────┼────────────┤
│ pglib_opf_case3_lmbd.m │    24 │    28 │     0.165133 │               5.2
039e-5 │                  0.404159 │           5812.64 │ 0.00866955 │      
0.00150822 │        0.00924134 │   5812.64 │ 0.0196715 │              4.779
01 │               0.485115 │        5812.64 │  0.676627 │              3.0
6456 │                2.75782 │        5812.64 │ 2.16971 │      0.000559726
 │            8.65092 │    6273.63 │
│ pglib_opf_case3_lmbd.m │    44 │    53 │     0.623138 │                4.
577e-5 │                  0.654535 │           17551.9 │  0.0174878 │      
0.00191648 │          0.018232 │   17551.9 │ 0.0391158 │            0.04408
27 │              0.0378932 │        17551.9 │   1.31771 │             0.10
8724 │                4.10194 │        17551.9 │  19.028 │      0.000387407
 │            19.0669 │    77.9548 │
└────────────────────────┴───────┴───────┴──────────────┴──────────────────
───────┴───────────────────────────┴───────────────────┴────────────┴──────
───────────┴───────────────────┴───────────┴───────────┴───────────────────
───┴────────────────────────┴────────────────┴───────────┴─────────────────
─────┴────────────────────────┴────────────────┴─────────┴─────────────────
─┴────────────────────┴────────────┘
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
 "/tmp/jl_uYfBkN/pglib_opf_case10000_goc.m"
 "/tmp/jl_uYfBkN/pglib_opf_case10192_epigrids.m"
 "/tmp/jl_uYfBkN/pglib_opf_case10480_goc.m"
 "/tmp/jl_uYfBkN/pglib_opf_case118_ieee.m"
 "/tmp/jl_uYfBkN/pglib_opf_case1354_pegase.m"
 "/tmp/jl_uYfBkN/pglib_opf_case13659_pegase.m"
 "/tmp/jl_uYfBkN/pglib_opf_case14_ieee.m"
 "/tmp/jl_uYfBkN/pglib_opf_case162_ieee_dtc.m"
 "/tmp/jl_uYfBkN/pglib_opf_case179_goc.m"
 "/tmp/jl_uYfBkN/pglib_opf_case1803_snem.m"
 ⋮
 "/tmp/jl_uYfBkN/pglib_opf_case6515_rte.m"
 "/tmp/jl_uYfBkN/pglib_opf_case7336_epigrids.m"
 "/tmp/jl_uYfBkN/pglib_opf_case73_ieee_rts.m"
 "/tmp/jl_uYfBkN/pglib_opf_case78484_epigrids.m"
 "/tmp/jl_uYfBkN/pglib_opf_case793_goc.m"
 "/tmp/jl_uYfBkN/pglib_opf_case8387_pegase.m"
 "/tmp/jl_uYfBkN/pglib_opf_case89_pegase.m"
 "/tmp/jl_uYfBkN/pglib_opf_case9241_pegase.m"
 "/tmp/jl_uYfBkN/pglib_opf_case9591_goc.m"
```



```julia
timing_data = multidata_multisolver_benchmark(benchmark_datasets)
```

```
file = "/tmp/jl_uYfBkN/pglib_opf_case10000_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case10192_epigrids.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case10480_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case118_ieee.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case1354_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case13659_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case14_ieee.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   2.012102e+09     2.033012e+03     1.250214e+09     2.012100e+09   
  2.20e-09
 * time: 1.694551351173339e9
     1   2.012086e+09     1.874465e+03     1.263486e+09     2.012084e+09   
  2.07e-09
 * time: 1.694551360508793e9
     2   2.012015e+09     1.706206e+03     1.275733e+09     2.012013e+09   
  2.06e-09
 * time: 1.694551369565971e9
     3   2.012015e+09     1.703877e+03     1.275867e+09     2.012013e+09   
  2.14e-09
 * time: 1.694551378213167e9
     4   2.012014e+09     1.700637e+03     1.276046e+09     2.012013e+09   
  2.13e-09
 * time: 1.694551387156934e9
     5   2.012012e+09     1.696357e+03     1.276224e+09     2.012010e+09   
  2.12e-09
 * time: 1.694551396139474e9
     6   2.011946e+09     1.685145e+03     1.276804e+09     2.011944e+09   
  2.12e-09
 * time: 1.69455140526511e9
     7   2.011945e+09     1.683385e+03     1.276564e+09     2.011944e+09   
  2.14e-09
 * time: 1.694551414357654e9
     8   2.011844e+09     1.681426e+03     1.282725e+09     2.011842e+09   
  2.16e-09
 * time: 1.694551423013651e9
     9   2.011842e+09     1.658795e+03     1.283589e+09     2.011840e+09   
  2.15e-09
 * time: 1.694551432274365e9
    10   2.011842e+09     1.658778e+03     1.283590e+09     2.011840e+09   
  2.25e-09
 * time: 1.694551440881711e9
    11   2.011842e+09     1.658702e+03     1.283594e+09     2.011840e+09   
  2.17e-09
 * time: 1.694551449511673e9
    12   2.011842e+09     1.658702e+03     1.283595e+09     2.011840e+09   
  2.08e-09
 * time: 1.694551458290871e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   2.012102e+09     2.033012e+03     1.250214e+09     2.012100e+09   
  2.20e-09
 * time: 1.694551475536955e9
     1   2.012086e+09     1.874465e+03     1.263486e+09     2.012084e+09   
  2.07e-09
 * time: 1.694551484641288e9
     2   2.012015e+09     1.706206e+03     1.275733e+09     2.012013e+09   
  2.06e-09
 * time: 1.69455149323672e9
     3   2.012015e+09     1.703877e+03     1.275867e+09     2.012013e+09   
  2.14e-09
 * time: 1.694551501928809e9
     4   2.012014e+09     1.700637e+03     1.276046e+09     2.012013e+09   
  2.13e-09
 * time: 1.694551510583421e9
     5   2.012012e+09     1.696357e+03     1.276224e+09     2.012010e+09   
  2.12e-09
 * time: 1.694551519182805e9
     6   2.011946e+09     1.685145e+03     1.276804e+09     2.011944e+09   
  2.12e-09
 * time: 1.694551527914595e9
     7   2.011945e+09     1.683385e+03     1.276564e+09     2.011944e+09   
  2.14e-09
 * time: 1.694551536523943e9
     8   2.011844e+09     1.681426e+03     1.282725e+09     2.011842e+09   
  2.16e-09
 * time: 1.694551545243567e9
     9   2.011842e+09     1.658795e+03     1.283589e+09     2.011840e+09   
  2.15e-09
 * time: 1.694551553989196e9
    10   2.011842e+09     1.658778e+03     1.283590e+09     2.011840e+09   
  2.25e-09
 * time: 1.694551562788792e9
    11   2.011842e+09     1.658702e+03     1.283594e+09     2.011840e+09   
  2.17e-09
 * time: 1.694551571632357e9
    12   2.011842e+09     1.658702e+03     1.283595e+09     2.011840e+09   
  2.08e-09
 * time: 1.694551580345357e9
file = "/tmp/jl_uYfBkN/pglib_opf_case162_ieee_dtc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case179_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case1803_snem.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case1888_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case19402_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case1951_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case197_snem.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2000_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case200_activ.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case20758_epigrids.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2312_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2383wp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case240_pserc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case24464_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case24_ieee_rts.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.695164e+05     6.479813e+04     1.140983e+04     1.047185e+05   
  1.35e-03
 * time: 1.694552219420979e9
     1   1.684678e+05     6.375099e+04     1.147264e+04     1.047170e+05   
  1.30e-03
 * time: 1.694552308689372e9
     2   1.684558e+05     6.374118e+04     7.276177e+04     1.047148e+05   
  1.32e-03
 * time: 1.694552397733272e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.695164e+05     6.479813e+04     1.140983e+04     1.047185e+05   
  1.35e-03
 * time: 1.694552561546691e9
     1   1.684678e+05     6.375099e+04     1.147264e+04     1.047170e+05   
  1.30e-03
 * time: 1.694552647499474e9
     2   1.684558e+05     6.374118e+04     7.276177e+04     1.047148e+05   
  1.32e-03
 * time: 1.694552733473528e9
file = "/tmp/jl_uYfBkN/pglib_opf_case2736sp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2737sop_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2742_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2746wop_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2746wp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2848_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2853_sdet.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2868_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case2869_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case30000_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case300_ieee.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case3012wp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case3022_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case30_as.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.056925e+03     7.806065e+02     4.884467e+01     1.906542e+02   
  1.14e-01
 * time: 1.694553311850938e9
     1   8.602058e+02     7.662686e+02     1.603886e+02     8.161370e+01   
  1.63e-02
 * time: 1.694553378479725e9
     2   8.394578e+02     7.720930e+02     8.849103e+02     4.738040e+01   
  2.60e-02
 * time: 1.694553444166668e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.056925e+03     7.806065e+02     4.884467e+01     1.906542e+02   
  1.14e-01
 * time: 1.694553574006003e9
     1   8.602058e+02     7.662686e+02     1.603886e+02     8.161370e+01   
  1.63e-02
 * time: 1.694553640691955e9
     2   8.394578e+02     7.720930e+02     8.849103e+02     4.738040e+01   
  2.60e-02
 * time: 1.694553707400057e9
file = "/tmp/jl_uYfBkN/pglib_opf_case30_ieee.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.507290e+08     4.896501e+03     1.177824e+08     1.507241e+08   
  2.26e-08
 * time: 1.694554310157951e9
     1   1.507202e+08     4.294365e+03     1.187496e+08     1.507159e+08   
  2.19e-08
 * time: 1.694554377685525e9
     2   1.507200e+08     4.244052e+03     1.188671e+08     1.507158e+08   
  2.25e-08
 * time: 1.694554447051473e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.507290e+08     4.896501e+03     1.177824e+08     1.507241e+08   
  2.26e-08
 * time: 1.69455458014626e9
     1   1.507202e+08     4.294365e+03     1.187496e+08     1.507159e+08   
  2.19e-08
 * time: 1.694554648987874e9
     2   1.507200e+08     4.244052e+03     1.188671e+08     1.507158e+08   
  2.25e-08
 * time: 1.694554717092388e9
file = "/tmp/jl_uYfBkN/pglib_opf_case3120sp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case3375wp_k.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case3970_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case39_epri.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.285861e+09     8.494204e+04     2.752219e+09     1.285776e+09   
  3.86e-07
 * time: 1.694555691449041e9
     1   1.285789e+09     7.207084e+04     2.750361e+09     1.285717e+09   
  3.64e-07
 * time: 1.694555801007309e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   1.285861e+09     8.494204e+04     2.752219e+09     1.285776e+09   
  3.86e-07
 * time: 1.694556015389051e9
     1   1.285789e+09     7.207084e+04     2.750361e+09     1.285717e+09   
  3.64e-07
 * time: 1.69455612372625e9
file = "/tmp/jl_uYfBkN/pglib_opf_case3_lmbd.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.694556125842666e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.694556125932157e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694556126025898e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694556126128065e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694556126231014e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.69455612632921e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.694556126428674e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.69455612652825e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.69455612662937e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.694556126728878e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.69455612682594e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.694556126935998e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694556127052807e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694556127164171e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694556127276062e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.694556127393018e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.694556127514902e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694556127574605e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -1.332793e+05    2.012000e+05     1.452933e+03     3.337912e+05   
  6.55e+00
 * time: 1.694556127753846e9
     1   3.871159e+03     1.384170e+04     1.848863e+03     9.912224e+03   
  6.55e-01
 * time: 1.694556127848501e9
     2   4.987679e+03     1.025073e+04     4.608072e+02     5.168011e+03   
  1.03e+00
 * time: 1.694556127942642e9
     3   5.334802e+03     9.051079e+03     5.889282e+02     3.685359e+03   
  3.43e-01
 * time: 1.694556128038683e9
     4   5.509669e+03     8.156831e+03     5.818415e+02     2.622187e+03   
  2.84e-01
 * time: 1.694556128135297e9
     5   5.602026e+03     7.589242e+03     6.207632e+02     1.968189e+03   
  2.21e-01
 * time: 1.694556128233084e9
     6   5.655170e+03     7.200137e+03     6.569763e+02     1.529936e+03   
  1.78e-01
 * time: 1.694556128557199e9
     7   5.689681e+03     6.906897e+03     6.817978e+02     1.205104e+03   
  1.46e-01
 * time: 1.694556128645423e9
     8   5.713434e+03     6.680810e+03     6.964722e+02     9.575828e+02   
  1.20e-01
 * time: 1.694556128736753e9
     9   5.729766e+03     6.517382e+03     7.044579e+02     7.796863e+02   
  9.98e-02
 * time: 1.694556128827776e9
    10   5.750961e+03     6.289108e+03     1.247805e+04     5.324717e+02   
  8.41e-02
 * time: 1.694556128921209e9
    11   5.736646e+03     6.273828e+03     2.371472e+05     5.109277e+02   
  4.28e-01
 * time: 1.694556129018078e9
    12   5.730447e+03     6.273790e+03     2.608217e+05     5.108370e+02   
  5.28e-01
 * time: 1.694556129136964e9
    13   5.730414e+03     6.273714e+03     2.069287e+05     5.106566e+02   
  5.28e-01
 * time: 1.694556129253138e9
    14   5.730417e+03     6.273637e+03     1.730478e+05     5.104764e+02   
  5.28e-01
 * time: 1.694556129365548e9
    15   5.730433e+03     6.273627e+03     1.696171e+05     5.104538e+02   
  5.28e-01
 * time: 1.694556129482332e9
    16   5.730435e+03     6.273626e+03     1.691994e+05     5.104510e+02   
  5.28e-01
 * time: 1.694556129608404e9
    17   5.730436e+03     6.273626e+03     1.691980e+05     5.104510e+02   
  5.28e-01
 * time: 1.694556129669395e9
file = "/tmp/jl_uYfBkN/pglib_opf_case4020_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case4601_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case4619_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case4661_sdet.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case4837_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case4917_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case500_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case5658_epigrids.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case57_ieee.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case588_sdet.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case5_pjm.m"
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694556143487885e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694556143995311e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694556144493295e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694556144982258e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.69455614575671e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694556146229231e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694556146733382e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694556147231688e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694556147722834e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694556148208947e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.69455614905185e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.694556149562492e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694556150060261e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694556150550039e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694556151037443e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694556151796647e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694556152305606e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694556152801187e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694556153291413e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694556153779456e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694556154537659e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694556155049866e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694556155547884e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694556156038263e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694556156528334e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694556157273745e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694556157788454e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694556158287067e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694556158779011e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694556159266823e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694556159746225e9
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694556160670602e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694556161172382e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694556161667472e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694556162433687e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.69455616290633e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694556163422544e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694556163915589e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694556164409307e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694556165301225e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694556165775984e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.694556166283489e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.69455616678161e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694556167273807e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.69455616806881e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694556168534801e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694556169046122e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694556169543585e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.694556170032579e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694556170521378e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694556171329138e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694556171839411e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694556172338008e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694556172827881e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694556173313543e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694556174111107e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.69455617462771e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694556175126496e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694556175617079e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694556176106632e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694556176911122e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.694556177095952e9
file = "/tmp/jl_uYfBkN/pglib_opf_case60_c.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case6468_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case6470_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case6495_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case6515_rte.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case7336_epigrids.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case73_ieee_rts.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case78484_epigrids.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case793_goc.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case8387_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case89_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case9241_pegase.m"
file = "/tmp/jl_uYfBkN/pglib_opf_case9591_goc.m"
7×23 DataFrame
 Row │ case                    vars   cons   optimization  optimization_mod
elb ⋯
     │ String                  Int64  Int64  Float64       Float64         
    ⋯
─────┼─────────────────────────────────────────────────────────────────────
─────
   1 │ pglib_opf_case3_lmbd.m    118    169      6.01272               8.78
9e- ⋯
   2 │ pglib_opf_case3_lmbd.m    266    315     49.0382                0.00
014
   3 │ pglib_opf_case3_lmbd.m    236    348     29.3351                0.00
014
   4 │ pglib_opf_case3_lmbd.m    236    348     45.0046                0.00
013
   5 │ pglib_opf_case3_lmbd.m    282    401     94.9359                0.00
016 ⋯
   6 │ pglib_opf_case3_lmbd.m     24     28      0.132264              8.00
09e
   7 │ pglib_opf_case3_lmbd.m     44     53      0.532753              8.47
6e-
                                                              19 columns om
itted
```



```julia
pretty_table(timing_data)
```

```
┌────────────────────────┬───────┬───────┬──────────────┬──────────────────
───────┬───────────────────────────┬───────────────────┬────────────┬──────
───────────┬───────────────────┬───────────┬───────────┬───────────────────
───┬────────────────────────┬────────────────┬───────────┬─────────────────
─────┬────────────────────────┬────────────────┬─────────┬─────────────────
─┬────────────────────┬────────────┐
│                   case │  vars │  cons │ optimization │ optimization_mode
lbuild │ optimization_wcompilation │ optimization_cost │       jump │ jump_
modelbuild │ jump_wcompilation │ jump_cost │ nlpmodels │ nlpmodels_modelbui
ld │ nlpmodels_wcompilation │ nlpmodels_cost │ nonconvex │ nonconvex_modelb
uild │ nonconvex_wcompilation │ nonconvex_cost │   optim │ optim_modelbuild
 │ optim_wcompilation │ optim_cost │
│                 String │ Int64 │ Int64 │      Float64 │                 F
loat64 │                   Float64 │           Float64 │    Float64 │      
   Float64 │           Float64 │   Float64 │   Float64 │              Float
64 │                Float64 │        Float64 │   Float64 │              Flo
at64 │                Float64 │        Float64 │ Float64 │          Float64
 │            Float64 │    Float64 │
├────────────────────────┼───────┼───────┼──────────────┼──────────────────
───────┼───────────────────────────┼───────────────────┼────────────┼──────
───────────┼───────────────────┼───────────┼───────────┼───────────────────
───┼────────────────────────┼────────────────┼───────────┼─────────────────
─────┼────────────────────────┼────────────────┼─────────┼─────────────────
─┼────────────────────┼────────────┤
│ pglib_opf_case3_lmbd.m │   118 │   169 │      6.01272 │                8.
789e-5 │                   10.8063 │           2178.08 │  0.0269048 │      
  0.120043 │         0.0627451 │   2178.08 │ 0.0771195 │             0.1443
84 │              0.0786554 │        2178.08 │   3.79818 │              1.8
3201 │                67.2726 │        2178.08 │ 122.054 │        0.0888073
 │             124.61 │     1658.7 │
│ pglib_opf_case3_lmbd.m │   266 │   315 │      49.0382 │             0.000
141649 │                   54.2288 │           63352.2 │  0.0593161 │      
0.00497223 │         0.0604514 │   63352.2 │  0.295706 │             0.1684
09 │               0.209865 │        63352.2 │   9.62823 │              11.
5519 │                328.495 │        63352.2 │  335.74 │        0.0914028
 │            342.204 │    63741.2 │
│ pglib_opf_case3_lmbd.m │   236 │   348 │      29.3351 │             0.000
145958 │                   29.4144 │           803.127 │   0.041446 │      
 0.0241766 │         0.0421334 │   803.127 │  0.271649 │             0.1668
87 │               0.136806 │        803.127 │   8.88744 │                1
2.86 │                350.238 │        803.127 │ 263.233 │         0.109558
 │            264.488 │    772.093 │
│ pglib_opf_case3_lmbd.m │   236 │   348 │      45.0046 │             0.000
139569 │                   44.9352 │           8208.52 │  0.0587552 │      
0.00655186 │         0.0596448 │   8208.52 │  0.218223 │             0.1686
69 │               0.218164 │        8208.52 │    9.9183 │              13.
2328 │                356.066 │        8208.52 │ 270.041 │        0.0307662
 │            269.721 │    4244.05 │
│ pglib_opf_case3_lmbd.m │   282 │   401 │      94.9359 │             0.000
167699 │                   95.9438 │         1.38416e5 │   0.104633 │      
0.00627794 │           0.10591 │ 1.38416e5 │  0.319533 │             0.1902
15 │               0.320811 │      1.38416e5 │   14.8008 │              19.
8072 │                 523.72 │      1.38416e5 │ 322.719 │         0.172859
 │            331.058 │    72070.8 │
│ pglib_opf_case3_lmbd.m │    24 │    28 │     0.132264 │               8.0
009e-5 │                   0.14002 │           5812.64 │ 0.00958647 │      
0.00290089 │         0.0101603 │   5812.64 │ 0.0192393 │            0.01733
78 │              0.0208487 │        5812.64 │  0.907668 │            0.021
2946 │               0.658419 │        5812.64 │ 2.09478 │      0.000591096
 │             1.9004 │    6273.63 │
│ pglib_opf_case3_lmbd.m │    44 │    53 │     0.532753 │                8.
476e-5 │                   0.75677 │           17551.9 │  0.0188171 │      
0.00231284 │         0.0192927 │   17551.9 │  0.038473 │            0.02638
26 │              0.0398252 │        17551.9 │   1.18848 │            0.088
3582 │                1.23693 │        17551.9 │ 17.3497 │      0.000595826
 │            17.3517 │    77.9548 │
└────────────────────────┴───────┴───────┴──────────────┴──────────────────
───────┴───────────────────────────┴───────────────────┴────────────┴──────
───────────┴───────────────────┴───────────┴───────────┴───────────────────
───┴────────────────────────┴────────────────┴───────────┴─────────────────
─────┴────────────────────────┴────────────────┴─────────┴─────────────────
─┴────────────────────┴────────────┘
```


