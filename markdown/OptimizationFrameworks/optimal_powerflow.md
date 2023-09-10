---
author: "Chris Rackauckas"
title: "Optimal Powerflow Nonlinear Optimization Benchmark"
---


## Data Load

```julia
import PowerModels
time_data_start = time()

file_name = "../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m"
const data = PowerModels.parse_file(file_name)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)
const ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

data_load_time = time() - time_data_start
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
11.099168062210083
```





## Optimization.jl

Constraint optimization implementation reference: https://github.com/SciML/Optimization.jl/blob/master/lib/OptimizationMOI/test/runtests.jl
Other AD libraries can be considered: https://docs.sciml.ai/dev/modules/Optimization/API/optimization_function/
However ForwardDiff is the only one that is compatible with constraint functions

```julia
import Optimization
import OptimizationMOI
import ModelingToolkit
import Ipopt

function solve_opf_optimization(file_name)
    time_model_start = time()

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
    function opf_objective(x, param)
        #start = time()
        cost = 0.0
        for (i,gen) in ref[:gen]
            pg = x[var_lookup["pg_$(i)"]]
            cost += gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
        end
        #total_callback_time += time() - start
        return cost
    end

    function opf_constraints(ret, x, param)
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
        ret .= [
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
    model_constraints = length(opf_constraints(ret, var_init, ref))
    println("variables: $(model_variables), $(length(var_lb)), $(length(var_ub))")
    println("constraints: $(model_constraints), $(length(con_lbs)), $(length(con_ubs))")


    optprob = Optimization.OptimizationFunction(opf_objective, Optimization.AutoModelingToolkit(true, true); cons=opf_constraints)
    prob = Optimization.OptimizationProblem(optprob, var_init; lb=var_lb, ub=var_ub, lcons=con_lbs, ucons=con_ubs)

    model_build_time = time() - time_model_start


    time_solve_start = time()

    sol = Optimization.solve(prob, Ipopt.Optimizer())
    cost = sol.minimum
    feasible = (sol.retcode == Optimization.SciMLBase.ReturnCode.Success)
    #println(sol.u) # solution vector

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

solve_opf_optimization("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
```

```
variables: 44, 44, 44
constraints: 53, 53, 53

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

Number of nonzeros in equality constraint Jacobian...:      155
Number of nonzeros in inequality constraint Jacobian.:       36
Number of nonzeros in Lagrangian Hessian.............:      240

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
  19  1.7551891e+04 6.10e-14 6.53e-10  -5.7 5.20e-07    -  1.00e+00 1.00e+0
0h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_p
r  ls
  20  1.7551891e+04 6.34e-12 3.03e-07  -8.6 3.52e-05    -  1.00e+00 1.00e+0
0f  1
  21  1.7551891e+04 2.84e-14 1.83e-12  -8.6 3.34e-08    -  1.00e+00 1.00e+0
0h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.3879727096486931e+02    1.7551890838594773e+0
4
Dual infeasibility......:   1.8295589429970638e-12    7.3182357719882553e-1
1
Constraint violation....:   2.8421709430404007e-14    2.8421709430404007e-1
4
Variable bound violation:   2.9463905093507492e-08    2.9463905093507492e-0
8
Complementarity.........:   2.5059076302144557e-09    1.0023630520857823e-0
7
Overall NLP error.......:   2.5059076302144557e-09    1.0023630520857823e-0
7


Number of objective function evaluations             = 28
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 28
Number of inequality constraint evaluations          = 28
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 1.675

EXIT: Optimal Solution Found.

Summary
   case........: ../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf
_case5_pjm.m
   variables...: 44
   constraints.: 53
   feasible....: true
   cost........: 17552
   total time..: 62.66626310348511
     data time.: 11.099168062210083
     build time: 2.456537961959839
     solve time: 39.946211099624634

Dict{String, Any} with 9 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_total"  => 62.6663
  "time_build"  => 2.45654
  "time_solve"  => 39.9462
  "time_data"   => 11.0992
  "feasible"    => true
```





## JuMP.jl

Implementation reference: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/ac-opf.jl
Only the built-in AD library is supported

```julia
import PowerModels
import Ipopt
import JuMP

function solve_opf_jump(file_name)



    time_model_start = time()

    model = JuMP.Model(Ipopt.Optimizer)
    #JuMP.set_optimizer_attribute(model, "print_level", 0)

    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)

    JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])

    JuMP.@objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i,bus) in ref[:ref_buses]
        JuMP.@constraint(model, va[i] == 0)
    end

    for (i,bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        JuMP.@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        )

        JuMP.@constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        )
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
        JuMP.@NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        JuMP.@NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )

        # To side of the branch flow
        JuMP.@NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        JuMP.@NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )

        # Voltage angle difference limit
        JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])

        # Apparent power limit, from side and to side
        JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    model_variables = JuMP.num_variables(model)

    # for consistency with other solvers, skip the variable bounds in the constraint count
    non_nl_constraints = sum(JuMP.num_constraints(model, ft, st) for (ft, st) in JuMP.list_of_constraint_types(model) if ft != JuMP.VariableRef)
    model_constraints = JuMP.num_nonlinear_constraints(model) + non_nl_constraints

    model_build_time = time() - time_model_start


    time_solve_start = time()

    JuMP.optimize!(model)
    cost = JuMP.objective_value(model)
    feasible = (JuMP.termination_status(model) == JuMP.LOCALLY_SOLVED)

    solve_time = time() - time_solve_start
    total_time = time() - time_data_start

    nlp_block = JuMP.MOI.get(model, JuMP.MOI.NLPBlock())
    total_callback_time =
        nlp_block.evaluator.eval_objective_timer +
        nlp_block.evaluator.eval_objective_gradient_timer +
        nlp_block.evaluator.eval_constraint_timer +
        nlp_block.evaluator.eval_constraint_jacobian_timer +
        nlp_block.evaluator.eval_hessian_lagrangian_timer

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
    println("      callbacks: $(total_callback_time)")
    println("")
    println("   callbacks time:")
    println("   * obj.....: $(nlp_block.evaluator.eval_objective_timer)")
    println("   * grad....: $(nlp_block.evaluator.eval_objective_gradient_timer)")
    println("   * cons....: $(nlp_block.evaluator.eval_constraint_timer)")
    println("   * jac.....: $(nlp_block.evaluator.eval_constraint_jacobian_timer)")
    println("   * hesslag.: $(nlp_block.evaluator.eval_hessian_lagrangian_timer)")
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
        "time_callbacks" => total_callback_time,
    )
end

solve_opf_jump("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
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
Total seconds in IPOPT                               = 0.829

EXIT: Optimal Solution Found.

Summary
   case........: ../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf
_case5_pjm.m
   variables...: 44
   constraints.: 53
   feasible....: true
   cost........: 17552
   total time..: 70.48021507263184
     data time.: 11.099168062210083
     build time: 2.7447099685668945
     solve time: 1.4593870639801025
      callbacks: 0.8135578632354736

   callbacks time:
   * obj.....: 0.0
   * grad....: 0.0
   * cons....: 0.0005714893341064453
   * jac.....: 0.00012540817260742188
   * hesslag.: 0.8128609657287598

Dict{String, Any} with 10 entries:
  "time_callbacks" => 0.813558
  "cost"           => 17551.9
  "variables"      => 44
  "constraints"    => 53
  "case"           => "../../benchmarks/OptimizationFrameworks/opf_data/pgl
ib_o…
  "time_total"     => 70.4802
  "time_build"     => 2.74471
  "time_solve"     => 1.45939
  "time_data"      => 11.0992
  "feasible"       => true
```





## NLPModels.jl

Implementation reference: https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/tutorial/
Other AD libraries can be considered: https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/

```julia
import PowerModels
import Symbolics
import ADNLPModels
import NLPModelsIpopt

function solve_opf_nlpmodels(file_name)



    time_model_start = time()


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
    println("variables: $(model_variables), $(length(var_lb)), $(length(var_ub))")
    println("constraints: $(model_constraints), $(length(con_lbs)), $(length(con_ubs))")

    nlp = ADNLPModels.ADNLPModel!(opf_objective, var_init, var_lb, var_ub, opf_constraints!, con_lbs, con_ubs, backend = :optimized)

    model_build_time = time() - time_model_start


    time_solve_start = time()

    output = NLPModelsIpopt.ipopt(nlp)
    cost = output.objective
    feasible = (output.primal_feas <= 1e-6)
    
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

solve_opf_nlpmodels("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
```

```
variables: 44, 44, 44
constraints: 53, 53, 53
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
Total seconds in IPOPT                               = 3.027

EXIT: Optimal Solution Found.

Summary
   case........: ../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf
_case5_pjm.m
   variables...: 44
   constraints.: 53
   feasible....: true
   cost........: 17552
   total time..: 114.29637908935547
     data time.: 11.099168062210083
     build time: 27.481008052825928
     solve time: 3.028304100036621

Dict{String, Any} with 9 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_total"  => 114.296
  "time_build"  => 27.481
  "time_solve"  => 3.0283
  "time_data"   => 11.0992
  "feasible"    => true
```





## Nonconvex

Implementation reference: https://julianonconvex.github.io/Nonconvex.jl/stable/problem/
Currently does not converge due to an upstream issue with the AD backend Zygote: https://github.com/JuliaNonconvex/Nonconvex.jl/issues/130

```julia
import PowerModels
import Nonconvex
Nonconvex.@load Ipopt


function solve_opf_nonconvex(file_name)



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

    model_variables = Nonconvex.NonconvexCore.getnvars(model)
    model_constraints = Nonconvex.NonconvexCore.getnconstraints(model)
    println("variables: $(model_variables)")
    println("constraints: $(model_constraints)")

    model_build_time = time() - time_model_start


    time_solve_start = time()

    result = Nonconvex.optimize(
        model,
        IpoptAlg(),
        NonconvexCore.getinit(model);
        options = IpoptOptions(; first_order=false, symbolic=true, sparse=true),
    )

    cost = result.minimum
    feasible = result.status == 0 # just guessing this is correct for Ipopt

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

solve_opf_nonconvex("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
```

```
variables: 44
constraints: 59
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
Total seconds in IPOPT                               = 1.324

EXIT: Optimal Solution Found.

Summary
   case........: ../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf
_case5_pjm.m
   variables...: 44
   constraints.: 59
   feasible....: true
   cost........: 17552
   total time..: 250.75755405426025
     data time.: 11.099168062210083
     build time: 15.090951204299927
     solve time: 107.14694285392761

Dict{String, Any} with 9 entries:
  "cost"        => 17551.9
  "variables"   => 44
  "constraints" => 59
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_total"  => 250.758
  "time_build"  => 15.091
  "time_solve"  => 107.147
  "time_data"   => 11.0992
  "feasible"    => true
```





## Optim.jl

Implementation reference: https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/
Currently does not converge to a feasible point, root cause in unclear
`debug/optim-debug.jl` can be used to confirm it will converge if given a suitable starting point

```julia
import PowerModels
import Optim

function solve_opf_optim(file_name)



    time_model_start = time()

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

    model_variables = length(var_init)
    model_constraints = length(opf_constraints(zeros(length(con_lbs)), var_init))
    println("variables: $(model_variables), $(length(var_lb)), $(length(var_ub))")
    println("constraints: $(model_constraints), $(length(con_lbs)), $(length(con_ubs))")

    df = Optim.TwiceDifferentiable(opf_objective, var_init)
    dfc = Optim.TwiceDifferentiableConstraints(opf_constraints, var_lb, var_ub, con_lbs, con_ubs)

    model_build_time = time() - time_model_start

    time_solve_start = time()

    options = Optim.Options(show_trace=true)

    # NOTE: had to change initial guess to be an interior point, otherwise getting NaN values
    res = Optim.optimize(df, dfc, var_init, Optim.IPNewton(), options)
    #res = Optim.optimize(df, dfc, var_init, Optim.LBFGS(), options) #  StackOverflowError:
    #res = Optim.optimize(df, dfc, var_init, Optim.NelderMead(), options) #  StackOverflowError:
    display(res)

    sol = res.minimizer
    cost = res.minimum

    solve_time = time() - time_solve_start
    total_time = time() - time_data_start


    # NOTE: confirmed these constraint violations can be eliminated
    # if a better starting point is used
    sol_eval = opf_constraints(zeros(length(con_lbs)), sol)
    vio_lb = [max(v,0) for v in (con_lbs .- sol_eval)]
    vio_ub = [max(v,0) for v in (sol_eval .- con_ubs)]
    const_vio = vio_lb .+ vio_ub
    #println(const_vio)
    println("total constraint violation: $(sum(const_vio))")
    constraint_tol = 1e-6
    feasible = (sum(const_vio) <= constraint_tol)

    if !feasible
        @warn "Optim optimize failed to satify the problem constraints"
    end

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

solve_opf_optim("../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf_case5_pjm.m")
```

```
variables: 44, 44, 44
constraints: 53, 53, 53
Iter     Lagrangian value Function value   Gradient norm    |==constr.|    
  μ
     0   -6.806002e+16    1.635500e+04     9.027326e+15     1.196696e+16   
  7.89e+14
 * time: 1.694314202841343e9
     1   -5.653618e+16    1.239424e+04     2.048354e+16     3.866281e+16   
  2.59e+14
 * time: 1.694314207440195e9
     2   -6.312953e+16    9.504854e+03     2.472419e+16     5.295854e+16   
  1.58e+14
 * time: 1.694314208063424e9
     3   -6.963967e+16    6.629948e+03     3.379787e+16     6.480692e+16   
  8.46e+13
 * time: 1.694314208703659e9
     4   -8.019084e+16    4.225763e+03     5.796137e+16     7.583884e+16   
  9.24e+13
 * time: 1.694314209338802e9
     5   -9.003775e+16    3.427203e+03     8.262621e+16     8.352453e+16   
  1.59e+14
 * time: 1.694314209997605e9
     6   -1.036563e+17    2.990479e+03     1.206177e+17     9.343366e+16   
  2.74e+14
 * time: 1.694314210624568e9
     7   -1.201133e+17    2.937273e+03     1.748921e+17     1.107859e+17   
  2.63e+14
 * time: 1.694314211257162e9
     8   -1.298488e+17    2.891923e+03     2.008420e+17     1.222013e+17   
  2.27e+14
 * time: 1.694314211888668e9
     9   -1.359597e+17    2.835253e+03     2.136709e+17     1.295591e+17   
  2.01e+14
 * time: 1.694314212522087e9
    10   -1.405817e+17    2.759768e+03     2.223574e+17     1.349031e+17   
  1.91e+14
 * time: 1.69431421317736e9
    11   -1.437112e+17    2.658988e+03     2.290012e+17     1.392006e+17   
  1.65e+14
 * time: 1.69431421380636e9
    12   -1.464692e+17    2.516357e+03     2.368001e+17     1.428086e+17   
  1.51e+14
 * time: 1.694314214438094e9
    13   -1.488184e+17    2.322116e+03     2.482098e+17     1.457502e+17   
  1.49e+14
 * time: 1.694314215073489e9
    14   -1.515591e+17    2.184809e+03     2.565629e+17     1.484211e+17   
  1.76e+14
 * time: 1.694314215730161e9
    15   -1.544486e+17    2.042027e+03     2.516699e+17     1.517197e+17   
  1.83e+14
 * time: 1.694314216358377e9
    16   -1.564462e+17    1.971423e+03     2.431387e+17     1.541470e+17   
  1.75e+14
 * time: 1.694314216987373e9
    17   -1.571498e+17    1.897328e+03     2.080075e+17     1.558652e+17   
  1.14e+14
 * time: 1.69431421761806e9
    18   -1.585181e+17    1.779351e+03     1.905178e+17     1.576816e+17   
  1.01e+14
 * time: 1.694314218269663e9
    19   -1.602408e+17    1.613097e+03     1.793373e+17     1.599373e+17   
  8.05e+13
 * time: 1.694314218896763e9
    20   -1.619481e+17    1.438560e+03     1.854662e+17     1.620090e+17   
  4.61e+13
 * time: 1.694314219523778e9
    21   -1.633753e+17    1.273718e+03     1.923713e+17     1.635305e+17   
  2.40e+13
 * time: 1.694314220152588e9
    22   -1.647239e+17    1.079019e+03     1.841617e+17     1.650091e+17   
  2.16e+13
 * time: 1.694314220783663e9
    23   -1.661138e+17    8.558663e+02     1.739263e+17     1.665270e+17   
  1.86e+13
 * time: 1.694314221434707e9
    24   -1.676187e+17    6.038961e+02     1.598350e+17     1.681490e+17   
  1.51e+13
 * time: 1.694314222058784e9
    25   -1.686664e+17    4.324137e+02     1.559112e+17     1.691788e+17   
  1.09e+13
 * time: 1.694314222684021e9
    26   -1.694824e+17    2.958285e+02     1.532171e+17     1.699581e+17   
  7.96e+12
 * time: 1.694314223313278e9
    27   -1.701523e+17    1.844768e+02     1.519071e+17     1.705673e+17   
  5.51e+12
 * time: 1.694314223962049e9
    28   -1.706362e+17    1.110139e+02     1.568218e+17     1.709533e+17   
  3.47e+12
 * time: 1.694314224584526e9
    29   -1.708669e+17    7.795483e+01     1.709948e+17     1.710819e+17   
  2.09e+12
 * time: 1.694314225207715e9
    30   -1.707499e+17    7.795483e+01     1.390829e+17     1.710819e+17   
  3.23e+12
 * time: 1.69431422543055e9
 * Status: success (objective increased between iterations)

 * Candidate solution
    Final objective value:     7.795483e+01

 * Found with
    Algorithm:     Interior Point Newton

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 4.00e+03 ≰ 1.0e-08

 * Work counters
    Seconds run:   23  (vs limit Inf)
    Iterations:    30
    f(x) calls:    86
    ∇f(x) calls:   86

total constraint violation: 227.84314955347247

Summary
   case........: ../../benchmarks/OptimizationFrameworks/opf_data/pglib_opf
_case5_pjm.m
   variables...: 44
   constraints.: 53
   feasible....: false
   cost........: 78
   total time..: 291.71532917022705
     data time.: 11.099168062210083
     build time: 2.619673013687134
     solve time: 31.341515064239502

Dict{String, Any} with 9 entries:
  "cost"        => 77.9548
  "variables"   => 44
  "constraints" => 53
  "case"        => "../../benchmarks/OptimizationFrameworks/opf_data/pglib_
opf_…
  "time_total"  => 291.715
  "time_build"  => 2.61967
  "time_solve"  => 31.3415
  "time_data"   => 11.0992
  "feasible"    => false
```





## CASADI

Implementation reference: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/ac-opf.jl

CASADI Segfaults so removed for now.

```
import PowerModels
import PythonCall
import CondaPkg
CondaPkg.add("casadi")

function solve_opf_casadi(file_name)


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