
PRINT_LEVEL = 0

# This is a soft upper limit to the time of each optimization.
# If times go above this, they will halt early
MAX_CPU_TIME = 100.0

# Maximum number of variables in an optimization problem for the benchmark
# Anything with more variables is rejected and not run
# This is for testing. 100 is a good size for running a test of changes
# Should be set to typemax(Int) to run the whole benchmark
SIZE_LIMIT = 1000


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


test_u0 = [0.062436387733897314, 1.0711076238965598, 0.0, 1.066509799068872, -0.023231313776594726, 1.0879315976617783, -0.033094993289919016, 1.0999999581285527, 0.07121718642320936, 1.094374845084077, 0.4228458440068076, -3.7102746662566277, 1.8046846767604458e-8, -0.44810504067067086, 8.80063717152151, -0.0, 0.8709675496332583, 3.6803022758556523, -0.0, 4.618897246588245, -1.1691336178031877, 1.3748418519024668, 0.9623014391707738, -1.3174990482204871, -2.3850868109149004, 0.1445158405684026, 2.813869610747349, 0.8151138880859179, 1.9869253829584679, 3.768252275480421, 3.9998421778156934, 0.03553108302190666, 1.177155791026922, -1.3025310027752557, -0.9598988325635542, 1.3193604239530325, 2.399997991458022, -0.003103523654225171, -2.7920689620650667, -0.6047898784636468, -1.9771521474512397, -3.7071711426024025, -3.9623014391707136, 0.3313990482205271]
test_obj = 16236.704322376236
test_cons = [0.0, 2.5424107263916085e-14, -1.0835776720341528e-13, -6.039613253960852e-14, 0.0, 0.0, 0.0, -1.7075230118734908e-13, -3.9968028886505635e-14, 1.532107773982716e-13, 0.0, 6.661338147750939e-16, -1.7763568394002505e-15, 0.0, 8.881784197001252e-16, 4.440892098500626e-16, 0.0, 4.440892098500626e-16, -1.7763568394002505e-15, -8.881784197001252e-16, -4.440892098500626e-16, 2.220446049250313e-16, 0.0, 1.7763568394002505e-15, 0.0, 6.8833827526759706e-15, -8.992806499463768e-15, 3.9968028886505635e-14, -7.105427357601002e-15, 0.0, 0.0, 7.327471962526033e-15, -7.105427357601002e-15, 2.842170943040401e-14, -7.105427357601002e-15, -0.033094993289919016, 0.00986367951332429, -0.062436387733897314, 0.07121718642320936, 0.008780798689312044, 0.09444850019980408, 3.2570635340201743, 2.661827801892032, 5.709523923775402, 8.58227283683798, 18.147597689108025, 15.999999905294098, 3.0822827695389314, 2.6621176970504, 5.759999990861611, 8.161419886019171, 17.65224849471505, 15.80965802401578]


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


optimization_test_res = test_optimization_prob(dataset, test_u0)


@assert optimization_test_res[1] == test_obj


@assert optimization_test_res[2] == test_cons



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


objective, reconstructed_u0, cons_vals, lcons, ucons = test_mtk_prob(dataset, test_u0)


@assert isapprox(objective, test_obj)


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


# Test all constraint values
@assert all(isapprox.(lcons, cons_vals, atol=1e-12) .|| (lcons .<= cons_vals .<= ucons) .|| isapprox.(cons_vals, ucons, atol=1e-12))




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


jump_test_res = test_jump_prob(dataset, test_u0)


@assert jump_test_res[1] ≈ test_obj


@assert sort(abs.(jump_test_res[2])) ≈ sort(abs.(test_cons))


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


nlpmodels_test_res = test_nlpmodels_prob(dataset, test_u0)


@assert nlpmodels_test_res[1] == test_obj


@assert nlpmodels_test_res[2] == test_cons


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


nonconvex_test_res = test_nonconvex_prob(dataset, test_u0)


@assert nonconvex_test_res[1] ≈ test_obj


@assert sort(abs.(nonconvex_test_res[2])) ≈ sort(abs.(test_cons))


println(sort(abs.(nonconvex_test_res[2])))


println(sort(abs.(test_cons)))


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


optim_test_res = test_optim_prob(dataset, test_u0)


@assert optim_test_res[1] == test_obj


@assert optim_test_res[2] == test_cons


file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
dataset = load_and_setup_data(file_name);


model, res = solve_opf_optimization(dataset);
res


model, res = solve_opf_jump(dataset);
res


model, res = solve_opf_nlpmodels(dataset);
res


model, res = solve_opf_nonconvex(dataset);
res


model, res = solve_opf_optim(dataset);
res


file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case3_lmbd.m"
dataset = load_and_setup_data(file_name);


model, res = solve_opf_optimization(dataset);
res


model, res = solve_opf_jump(dataset);
res


model, res = solve_opf_nlpmodels(dataset);
res


model, res = solve_opf_nonconvex(dataset);
res


model, res = solve_opf_optim(dataset);
res


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


timing_data = multidata_multisolver_benchmark(test_datasets)


io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, timing_data; backend = Val(:html))
# show(io, "text/html", pretty_table(timing_data; backend = Val(:html)))
println(io, "```")
Text(String(take!(io)))


using LibGit2
tmpdir = Base.Filesystem.mktempdir()
LibGit2.clone("https://github.com/power-grid-lib/pglib-opf", tmpdir)
benchmarkfiles = readdir(tmpdir)
benchmarkfiles = benchmarkfiles[endswith(".m").(benchmarkfiles)]
benchmark_datasets = joinpath.((tmpdir,),benchmarkfiles)


timing_data = multidata_multisolver_benchmark(benchmark_datasets)


io = IOBuffer()
println(io, "```@raw html")
pretty_table(io, timing_data; backend = Val(:html))
# show(io, "text/html", pretty_table(timing_data; backend = Val(:html)))
println(io, "```")
Text(String(take!(io)))


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

