
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using ModelingToolkit
using Multibody
using Multibody: Robot6DOF
using JuliaSimCompiler
using OrdinaryDiffEq
using CairoMakie
using Printf

time_instantiate = @elapsed @named robot = Robot6DOF()
robot = complete(robot)
time_simplify = @elapsed ssys = structural_simplify(IRSystem(robot))
time_prob = @elapsed prob = ODEProblem(ssys, [
    robot.mechanics.r1.phi => deg2rad(-60)
    robot.mechanics.r2.phi => deg2rad(20)
    robot.mechanics.r3.phi => deg2rad(90)
    robot.mechanics.r4.phi => deg2rad(0)
    robot.mechanics.r5.phi => deg2rad(-110)
    robot.mechanics.r6.phi => deg2rad(0)
    robot.axis1.motor.Jmotor.phi => deg2rad(-60) * (-105) 
    robot.axis2.motor.Jmotor.phi => deg2rad(20) * (210)
    robot.axis3.motor.Jmotor.phi => deg2rad(90) * (60)
], (0.0, 4.0))
time_solve = @elapsed sol = solve(prob, Rodas5P(autodiff=false)); # With autodiff=true this takes over 150 seconds

tv = 0:0.01:4
time_extract_data = @elapsed data = sol(tv, idxs=[
    robot.pathPlanning.controlBus.axisControlBus1.angle
    robot.pathPlanning.controlBus.axisControlBus2.angle
    robot.pathPlanning.controlBus.axisControlBus3.angle
    robot.pathPlanning.controlBus.axisControlBus4.angle
    robot.pathPlanning.controlBus.axisControlBus5.angle
    robot.pathPlanning.controlBus.axisControlBus6.angle
    robot.mechanics.r1.phi
    robot.mechanics.r2.phi
    robot.mechanics.r3.phi
    robot.mechanics.r4.phi
    robot.mechanics.r5.phi
    robot.mechanics.r6.phi
    robot.axis1.controller.feedback1.output.u
    robot.axis2.controller.feedback1.output.u
    robot.axis3.controller.feedback1.output.u
    robot.axis4.controller.feedback1.output.u
    robot.axis5.controller.feedback1.output.u
    robot.axis6.controller.feedback1.output.u
]);

labels = ["Instantiate", "Simplify", "Problem creation", "Solve", "Extract data"]
timings = [time_instantiate, time_simplify, time_prob, time_solve, time_extract_data]
f = Figure()
xs = 1:length(labels)
points = Makie.Point.(xs .- 0.25, timings .+ 1)
anns = [@sprintf("%3.3g", t) for t in timings]
barplot(f[1,1], timings, axis=(; title="Timings", xticks = (xs, labels), limits = (nothing, (0, maximum(timings)*1.2 + 2))))

annotations!(f[1,1], anns, points)
barplot(f[2,1], timings, axis=(; title="Log Timings", xticks = (xs, labels), yscale=log10))
f


using OMJulia
mod = OMJulia.OMCSession();
OMJulia.sendExpression(mod, "getVersion()")

@show "Start OpenModelica Timings"

om_total = @elapsed begin
    om_build = @elapsed ModelicaSystem(mod, modelName="Modelica.Mechanics.MultiBody.Examples.Systems.RobotR3.FullRobot", library="Modelica")
    om_sim = @elapsed res = sendExpression(mod, "simulate(Modelica.Mechanics.MultiBody.Examples.Systems.RobotR3.FullRobot)")
end
@assert res["messages"][1:11] == "LOG_SUCCESS"

OMJulia.quit(mod)


colors = Makie.wong_colors()
labels = ["Build", "Simulate", "Total"]
julia_build = time_instantiate + time_simplify + time_prob
julia_sim = time_solve + time_extract_data
julia_total = julia_build + julia_sim
dymola_build = sum([5.075, 3.912, 4.024])/3
dymola_total = sum([5.267, 4.112, 4.255])/3
dymola_sim = dymola_total - dymola_build

data = [
    julia_build julia_sim julia_total
    om_build om_sim om_total
    dymola_build dymola_sim dymola_total
]

xs = repeat(1:length(labels), inner=3)
group = repeat([1,2,3], outer=3)
fig = Figure()
barplot(fig[1,1], xs, vec(data), dodge=group, color=colors[group], axis=(; title="Timings", xticks = ([1,2,3], labels)))

# Legend
legendentries = ["Julia", "OpenModelica", "Dymola"]
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(legendentries)]
title = "Contestants"

Legend(fig[1,2], elements, legendentries, title)
fig


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

