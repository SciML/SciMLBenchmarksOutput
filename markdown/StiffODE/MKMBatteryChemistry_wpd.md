---
author: "Chris Rackauckas"
title: "Microkinetic Model (MKM) Battery Chemistry Work-Precision Diagrams"
---


# MKM Battery Chemistry

This benchmark tests a microkinetic model (MKM) for battery chemistry representing oxygen reduction reaction (ORR) kinetics. The model includes 13 state variables representing different chemical species and intermediate states in the electrochemical reaction mechanism. The system exhibits extreme stiffness due to the wide range of time scales in the chemical reactions, from fast electron transfer steps to slower mass transport processes.

The model consists of:
- 13 chemical species concentrations (x₁ to x₁₃)
- Complex reaction kinetics with exponential temperature dependencies
- Multiple reaction pathways for oxygen reduction
- Adsorption/desorption processes on electrode surfaces
- Electrochemical charge transfer reactions

This represents a realistic electrochemical system with multiple time scales and nonlinear reaction kinetics, making it an excellent test for stiff ODE solvers.

```julia
using OrdinaryDiffEq, DiffEqDevTools, Sundials, Plots, ODEInterfaceDiffEq, LSODA, LinearSolve
using ProfileSVG, BenchmarkTools, Profile
gr() # gr(fmt=:png)
using LinearAlgebra, StaticArrays, RecursiveFactorization

ff = begin
        ff = ((var"##MTIIPVar#15413", var"##MTKArg#15409", var"##MTKArg#15410", var"##MTKArg#15411")->begin
                    @inbounds begin
                            begin
                                let (x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₁, x₁₂, x₁₃, α₁, α₂, α₃, α₄, α₅, α₆, α₇, α₈, α₉, α₁₀, α₁₁, α₁₂, α₁₃, α₁₄, α₁₅, α₁₆, α₁₇, α₁₈, α₁₉, α₂₀, α₂₁, α₂₂, α₂₃, α₂₄, α₂₅, α₂₆, α₂₇, α₂₈, α₂₉, α₃₀, α₃₁, α₃₂, t) = (var"##MTKArg#15409"[1], var"##MTKArg#15409"[2], var"##MTKArg#15409"[3], var"##MTKArg#15409"[4], var"##MTKArg#15409"[5], var"##MTKArg#15409"[6], var"##MTKArg#15409"[7], var"##MTKArg#15409"[8], var"##MTKArg#15409"[9], var"##MTKArg#15409"[10], var"##MTKArg#15409"[11], var"##MTKArg#15409"[12], var"##MTKArg#15409"[13], var"##MTKArg#15410"[1], var"##MTKArg#15410"[2], var"##MTKArg#15410"[3], var"##MTKArg#15410"[4], var"##MTKArg#15410"[5], var"##MTKArg#15410"[6], var"##MTKArg#15410"[7], var"##MTKArg#15410"[8], var"##MTKArg#15410"[9], var"##MTKArg#15410"[10], var"##MTKArg#15410"[11], var"##MTKArg#15410"[12], var"##MTKArg#15410"[13], var"##MTKArg#15410"[14], var"##MTKArg#15410"[15], var"##MTKArg#15410"[16], var"##MTKArg#15410"[17], var"##MTKArg#15410"[18], var"##MTKArg#15410"[19], var"##MTKArg#15410"[20], var"##MTKArg#15410"[21], var"##MTKArg#15410"[22], var"##MTKArg#15410"[23], var"##MTKArg#15410"[24], var"##MTKArg#15410"[25], var"##MTKArg#15410"[26], var"##MTKArg#15410"[27], var"##MTKArg#15410"[28], var"##MTKArg#15410"[29], var"##MTKArg#15410"[30], var"##MTKArg#15410"[31], var"##MTKArg#15410"[32], var"##MTKArg#15411")
                                    var"##MTIIPVar#15413"[1] = 0.0
                                    var"##MTIIPVar#15413"[2] = ((((identity(0.0) + ((α₁₄ * (x₁ / 0.5)) * (x₁₀ / 0.5) - ((α₁₄ / exp((((identity(0) + 1.0 * ((α₁ / 2477.572) * true)) + -1.0 * ((α₂ / 2477.572) * true)) + 1.0 * ((α₁₀ / 2477.572) * true)) + -3.697891137634378)) * true) * (x₂ / 0.5))) - ((α₁₅ * (x₂ / 0.5)) * (x₁₁ / 0.5) - (((α₁₅ / exp(((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₄ / 0.5)) * (x₇ / 0.5))) - ((α₁₉ * (x₂ / 0.5)) * (x₁₂ / 0.5) - ((α₁₉ / exp((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₃ / 0.5))) + (α₂₀ * (x₃ / 0.5) - (((α₂₀ / exp((((identity(0) + -1.0 * ((α₂ / 2477.572) * true)) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₂ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[3] = (((((((identity(0.0) - ((α₁₆ * (x₃ / 0.5)) * (x₁₁ / 0.5) - (((α₁₆ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₇ / 0.5))) + ((α₁₉ * (x₂ / 0.5)) * (x₁₂ / 0.5) - ((α₁₉ / exp((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₃ / 0.5))) - (α₂₀ * (x₃ / 0.5) - (((α₂₀ / exp((((identity(0) + -1.0 * ((α₂ / 2477.572) * true)) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₂ / 0.5)) * (x₁₂ / 0.5))) - ((α₂₁ * (x₃ / 0.5)) * (x₁₂ / 0.5) - (((α₂₁ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₄ / 0.5)) * (x₁₃ / 0.5))) + ((α₂₂ * (x₄ / 0.5)) * (x₁₃ / 0.5) - (((α₂₂ / exp(((((identity(0) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₃ / 0.5)) * (x₁₂ / 0.5))) - ((α₃₁ * (x₃ / 0.5)) * (x₁₂ / 0.5) - ((α₃₁ / exp((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₆ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₆ / 0.5))) + (α₃₂ * (x₆ / 0.5) - (((α₃₂ / exp((((identity(0) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₃ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[4] = (((((identity(0.0) + ((α₁₅ * (x₂ / 0.5)) * (x₁₁ / 0.5) - (((α₁₅ / exp(((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₄ / 0.5)) * (x₇ / 0.5))) + ((α₂₁ * (x₃ / 0.5)) * (x₁₂ / 0.5) - (((α₂₁ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₄ / 0.5)) * (x₁₃ / 0.5))) - ((α₂₂ * (x₄ / 0.5)) * (x₁₃ / 0.5) - (((α₂₂ / exp(((((identity(0) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₃ / 0.5)) * (x₁₂ / 0.5))) - ((α₂₃ * (x₄ / 0.5)) * (x₁₂ / 0.5) - ((α₂₃ / exp((((identity(0) + 1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₅ / 0.5))) + (α₂₄ * (x₅ / 0.5) - (((α₂₄ / exp((((identity(0) + -1.0 * ((α₄ / 2477.572) * true)) + 1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₄ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[5] = ((((((identity(0.0) + ((α₁₆ * (x₃ / 0.5)) * (x₁₁ / 0.5) - (((α₁₆ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₇ / 0.5))) + ((α₁₇ * (x₆ / 0.5)) * (x₁₁ / 0.5) - (((α₁₇ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₈ / 0.5))) + ((α₂₃ * (x₄ / 0.5)) * (x₁₂ / 0.5) - ((α₂₃ / exp((((identity(0) + 1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₅ / 0.5))) - (α₂₄ * (x₅ / 0.5) - (((α₂₄ / exp((((identity(0) + -1.0 * ((α₄ / 2477.572) * true)) + 1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₄ / 0.5)) * (x₁₂ / 0.5))) - ((α₂₅ * (x₅ / 0.5)) * (x₁₂ / 0.5) - (((α₂₅ / exp(((((identity(0) + 1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₁₀ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₁₀ / 0.5)) * (x₁₃ / 0.5))) + ((α₂₆ * (x₁₀ / 0.5)) * (x₁₃ / 0.5) - (((α₂₆ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₁₀ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₅ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[6] = ((((identity(0.0) - ((α₁₇ * (x₆ / 0.5)) * (x₁₁ / 0.5) - (((α₁₇ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₈ / 0.5))) - (α₁₈ * (x₆ / 0.5) - (((α₁₈ / exp((((identity(0) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₉ / 2477.572) * true)) + -1.0 * ((α₁₀ / 2477.572) * true)) + 3.697891137634378)) * true) * (x₁₀ / 0.5)) * (x₉ / 0.5))) + ((α₃₁ * (x₃ / 0.5)) * (x₁₂ / 0.5) - ((α₃₁ / exp((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₆ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₆ / 0.5))) - (α₃₂ * (x₆ / 0.5) - (((α₃₂ / exp((((identity(0) + -1.0 * ((α₃ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₃ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[7] = ((((identity(0.0) + ((α₁₅ * (x₂ / 0.5)) * (x₁₁ / 0.5) - (((α₁₅ / exp(((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₄ / 0.5)) * (x₇ / 0.5))) + ((α₁₆ * (x₃ / 0.5)) * (x₁₁ / 0.5) - (((α₁₆ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₇ / 0.5))) - ((α₂₇ * (x₇ / 0.5)) * (x₁₂ / 0.5) - ((α₂₇ / exp((((identity(0) + 1.0 * ((α₇ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₈ / 0.5))) + (α₂₈ * (x₈ / 0.5) - (((α₂₈ / exp((((identity(0) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₈ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₇ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[8] = (((((identity(0.0) + ((α₁₇ * (x₆ / 0.5)) * (x₁₁ / 0.5) - (((α₁₇ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₈ / 0.5))) + ((α₂₇ * (x₇ / 0.5)) * (x₁₂ / 0.5) - ((α₂₇ / exp((((identity(0) + 1.0 * ((α₇ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -3.697891137634378)) * false) * (x₈ / 0.5))) - (α₂₈ * (x₈ / 0.5) - (((α₂₈ / exp((((identity(0) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₈ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 3.697891137634378)) * false) * (x₇ / 0.5)) * (x₁₂ / 0.5))) - ((α₂₉ * (x₈ / 0.5)) * (x₁₂ / 0.5) - (((α₂₉ / exp(((((identity(0) + 1.0 * ((α₈ / 2477.572) * true)) + -1.0 * ((α₁₁ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₁₃ / 0.5)) * (x₁₁ / 0.5))) + ((α₃₀ * (x₁₃ / 0.5)) * (x₁₁ / 0.5) - (((α₃₀ / exp(((((identity(0) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₈ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[9] = 0.0
                                    var"##MTIIPVar#15413"[10] = ((((identity(0.0) - ((α₁₄ * (x₁ / 0.5)) * (x₁₀ / 0.5) - ((α₁₄ / exp((((identity(0) + 1.0 * ((α₁ / 2477.572) * true)) + -1.0 * ((α₂ / 2477.572) * true)) + 1.0 * ((α₁₀ / 2477.572) * true)) + -3.697891137634378)) * true) * (x₂ / 0.5))) + (α₁₈ * (x₆ / 0.5) - (((α₁₈ / exp((((identity(0) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₉ / 2477.572) * true)) + -1.0 * ((α₁₀ / 2477.572) * true)) + 3.697891137634378)) * true) * (x₁₀ / 0.5)) * (x₉ / 0.5))) + ((α₂₅ * (x₅ / 0.5)) * (x₁₂ / 0.5) - (((α₂₅ / exp(((((identity(0) + 1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₁₀ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₁₀ / 0.5)) * (x₁₃ / 0.5))) - ((α₂₆ * (x₁₀ / 0.5)) * (x₁₃ / 0.5) - (((α₂₆ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₁₀ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₅ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[11] = (((((identity(0.0) - ((α₁₅ * (x₂ / 0.5)) * (x₁₁ / 0.5) - (((α₁₅ / exp(((((identity(0) + 1.0 * ((α₂ / 2477.572) * true)) + -1.0 * ((α₄ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₄ / 0.5)) * (x₇ / 0.5))) - ((α₁₆ * (x₃ / 0.5)) * (x₁₁ / 0.5) - (((α₁₆ / exp(((((identity(0) + 1.0 * ((α₃ / 2477.572) * true)) + -1.0 * ((α₅ / 2477.572) * true)) + -1.0 * ((α₇ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₇ / 0.5))) - ((α₁₇ * (x₆ / 0.5)) * (x₁₁ / 0.5) - (((α₁₇ / exp(((((identity(0) + -1.0 * ((α₅ / 2477.572) * true)) + 1.0 * ((α₆ / 2477.572) * true)) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + 0.0)) * true) * (x₅ / 0.5)) * (x₈ / 0.5))) + ((α₂₉ * (x₈ / 0.5)) * (x₁₂ / 0.5) - (((α₂₉ / exp(((((identity(0) + 1.0 * ((α₈ / 2477.572) * true)) + -1.0 * ((α₁₁ / 2477.572) * true)) + 1.0 * ((α₁₂ / 2477.572) * true)) + -1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₁₃ / 0.5)) * (x₁₁ / 0.5))) - ((α₃₀ * (x₁₃ / 0.5)) * (x₁₁ / 0.5) - (((α₃₀ / exp(((((identity(0) + -1.0 * ((α₈ / 2477.572) * true)) + 1.0 * ((α₁₁ / 2477.572) * true)) + -1.0 * ((α₁₂ / 2477.572) * true)) + 1.0 * ((α₁₃ / 2477.572) * true)) + 0.0)) * false) * (x₈ / 0.5)) * (x₁₂ / 0.5))) * 0.5
                                    var"##MTIIPVar#15413"[12] = 0.0
                                    var"##MTIIPVar#15413"[13] = 0.0
                                end
                            end
                        end
                    nothing
                end)
        tgrad = nothing
        jac = ((var"##MTIIPVar#15420", var"##MTKArg#15416", var"##MTKArg#15417", var"##MTKArg#15418")->begin
                    @inbounds begin
                            begin
                                let (x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₁, x₁₂, x₁₃, α₁, α₂, α₃, α₄, α₅, α₆, α₇, α₈, α₉, α₁₀, α₁₁, α₁₂, α₁₃, α₁₄, α₁₅, α₁₆, α₁₇, α₁₈, α₁₉, α₂₀, α₂₁, α₂₂, α₂₃, α₂₄, α₂₅, α₂₆, α₂₇, α₂₈, α₂₉, α₃₀, α₃₁, α₃₂, t) = (var"##MTKArg#15416"[1], var"##MTKArg#15416"[2], var"##MTKArg#15416"[3], var"##MTKArg#15416"[4], var"##MTKArg#15416"[5], var"##MTKArg#15416"[6], var"##MTKArg#15416"[7], var"##MTKArg#15416"[8], var"##MTKArg#15416"[9], var"##MTKArg#15416"[10], var"##MTKArg#15416"[11], var"##MTKArg#15416"[12], var"##MTKArg#15416"[13], var"##MTKArg#15417"[1], var"##MTKArg#15417"[2], var"##MTKArg#15417"[3], var"##MTKArg#15417"[4], var"##MTKArg#15417"[5], var"##MTKArg#15417"[6], var"##MTKArg#15417"[7], var"##MTKArg#15417"[8], var"##MTKArg#15417"[9], var"##MTKArg#15417"[10], var"##MTKArg#15417"[11], var"##MTKArg#15417"[12], var"##MTKArg#15417"[13], var"##MTKArg#15417"[14], var"##MTKArg#15417"[15], var"##MTKArg#15417"[16], var"##MTKArg#15417"[17], var"##MTKArg#15417"[18], var"##MTKArg#15417"[19], var"##MTKArg#15417"[20], var"##MTKArg#15417"[21], var"##MTKArg#15417"[22], var"##MTKArg#15417"[23], var"##MTKArg#15417"[24], var"##MTKArg#15417"[25], var"##MTKArg#15417"[26], var"##MTKArg#15417"[27], var"##MTKArg#15417"[28], var"##MTKArg#15417"[29], var"##MTKArg#15417"[30], var"##MTKArg#15417"[31], var"##MTKArg#15417"[32], var"##MTKArg#15418")
                                    var"##MTIIPVar#15420"[1] = 0.0
                                    var"##MTIIPVar#15420"[2] = 2.0 * x₁₀ * α₁₄
                                    var"##MTIIPVar#15420"[3] = 0.0
                                    var"##MTIIPVar#15420"[4] = 0.0
                                    var"##MTIIPVar#15420"[5] = 0.0
                                    var"##MTIIPVar#15420"[6] = 0.0
                                    var"##MTIIPVar#15420"[7] = 0.0
                                    var"##MTIIPVar#15420"[8] = 0.0
                                    var"##MTIIPVar#15420"[9] = 0.0
                                    var"##MTIIPVar#15420"[10] = -2.0 * x₁₀ * α₁₄
                                    var"##MTIIPVar#15420"[11] = 0.0
                                    var"##MTIIPVar#15420"[12] = 0.0
                                    var"##MTIIPVar#15420"[13] = 0.0
                                    var"##MTIIPVar#15420"[14] = 0.0
                                    var"##MTIIPVar#15420"[15] = 0.5 * (-4.0 * x₁₁ * α₁₅ + -4.0 * x₁₂ * α₁₉ + -2.0 * exp(-3.697891137634378 + -0.00040362096439578745α₂ + 0.00040362096439578745 * (α₁ + α₁₀)) ^ -1 * α₁₄)
                                    var"##MTIIPVar#15420"[16] = 2.0 * x₁₂ * α₁₉
                                    var"##MTIIPVar#15420"[17] = 2.0 * x₁₁ * α₁₅
                                    var"##MTIIPVar#15420"[18] = 0.0
                                    var"##MTIIPVar#15420"[19] = 0.0
                                    var"##MTIIPVar#15420"[20] = 2.0 * x₁₁ * α₁₅
                                    var"##MTIIPVar#15420"[21] = 0.0
                                    var"##MTIIPVar#15420"[22] = 0.0
                                    var"##MTIIPVar#15420"[23] = exp(-3.697891137634378 + -0.00040362096439578745α₂ + 0.00040362096439578745 * (α₁ + α₁₀)) ^ -1 * α₁₄
                                    var"##MTIIPVar#15420"[24] = -2.0 * x₁₁ * α₁₅
                                    var"##MTIIPVar#15420"[25] = 0.0
                                    var"##MTIIPVar#15420"[26] = 0.0
                                    var"##MTIIPVar#15420"[27] = 0.0
                                    var"##MTIIPVar#15420"[28] = α₂₀
                                    var"##MTIIPVar#15420"[29] = 0.5 * (-2.0α₂₀ + -4.0 * x₁₁ * α₁₆ + -4.0 * x₁₂ * (α₂₁ + α₃₁))
                                    var"##MTIIPVar#15420"[30] = 2.0 * x₁₂ * α₂₁
                                    var"##MTIIPVar#15420"[31] = 2.0 * x₁₁ * α₁₆
                                    var"##MTIIPVar#15420"[32] = 2.0 * x₁₂ * α₃₁
                                    var"##MTIIPVar#15420"[33] = 2.0 * x₁₁ * α₁₆
                                    var"##MTIIPVar#15420"[34] = 0.0
                                    var"##MTIIPVar#15420"[35] = 0.0
                                    var"##MTIIPVar#15420"[36] = 0.0
                                    var"##MTIIPVar#15420"[37] = -2.0 * x₁₁ * α₁₆
                                    var"##MTIIPVar#15420"[38] = 0.0
                                    var"##MTIIPVar#15420"[39] = 0.0
                                    var"##MTIIPVar#15420"[40] = 0.0
                                    var"##MTIIPVar#15420"[41] = 2.0 * x₇ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[42] = 2.0 * x₁₃ * α₂₂
                                    var"##MTIIPVar#15420"[43] = 0.5 * (-4.0 * x₁₂ * α₂₃ + -4.0 * x₁₃ * α₂₂ + -4.0 * x₇ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1)
                                    var"##MTIIPVar#15420"[44] = 2.0 * x₁₂ * α₂₃
                                    var"##MTIIPVar#15420"[45] = 0.0
                                    var"##MTIIPVar#15420"[46] = -2.0 * x₇ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[47] = 0.0
                                    var"##MTIIPVar#15420"[48] = 0.0
                                    var"##MTIIPVar#15420"[49] = 0.0
                                    var"##MTIIPVar#15420"[50] = 2.0 * x₇ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[51] = 0.0
                                    var"##MTIIPVar#15420"[52] = 0.0
                                    var"##MTIIPVar#15420"[53] = 0.0
                                    var"##MTIIPVar#15420"[54] = 0.0
                                    var"##MTIIPVar#15420"[55] = 2.0 * x₇ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[56] = α₂₄
                                    var"##MTIIPVar#15420"[57] = 0.5 * (-2.0α₂₄ + -4.0 * x₁₂ * α₂₅ + -4.0 * x₇ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1 + -4.0 * x₈ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1)
                                    var"##MTIIPVar#15420"[58] = 2.0 * x₈ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1
                                    var"##MTIIPVar#15420"[59] = -2.0 * x₇ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[60] = -2.0 * x₈ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1
                                    var"##MTIIPVar#15420"[61] = 0.0
                                    var"##MTIIPVar#15420"[62] = 2.0 * x₁₂ * α₂₅
                                    var"##MTIIPVar#15420"[63] = 0.5 * (4.0 * x₇ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1 + 4.0 * x₈ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1)
                                    var"##MTIIPVar#15420"[64] = 0.0
                                    var"##MTIIPVar#15420"[65] = 0.0
                                    var"##MTIIPVar#15420"[66] = 0.0
                                    var"##MTIIPVar#15420"[67] = 0.0
                                    var"##MTIIPVar#15420"[68] = α₃₂
                                    var"##MTIIPVar#15420"[69] = 0.0
                                    var"##MTIIPVar#15420"[70] = 2.0 * x₁₁ * α₁₇
                                    var"##MTIIPVar#15420"[71] = 0.5 * (-2.0 * (α₁₈ + α₃₂) + -4.0 * x₁₁ * α₁₇)
                                    var"##MTIIPVar#15420"[72] = 0.0
                                    var"##MTIIPVar#15420"[73] = 2.0 * x₁₁ * α₁₇
                                    var"##MTIIPVar#15420"[74] = 0.0
                                    var"##MTIIPVar#15420"[75] = α₁₈
                                    var"##MTIIPVar#15420"[76] = -2.0 * x₁₁ * α₁₇
                                    var"##MTIIPVar#15420"[77] = 0.0
                                    var"##MTIIPVar#15420"[78] = 0.0
                                    var"##MTIIPVar#15420"[79] = 0.0
                                    var"##MTIIPVar#15420"[80] = 2.0 * x₄ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[81] = 2.0 * x₅ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[82] = -2.0 * x₄ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[83] = -2.0 * x₅ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1
                                    var"##MTIIPVar#15420"[84] = 0.0
                                    var"##MTIIPVar#15420"[85] = 0.5 * (-4.0 * x₁₂ * α₂₇ + -4.0 * x₄ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1 + -4.0 * x₅ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1)
                                    var"##MTIIPVar#15420"[86] = 2.0 * x₁₂ * α₂₇
                                    var"##MTIIPVar#15420"[87] = 0.0
                                    var"##MTIIPVar#15420"[88] = 0.0
                                    var"##MTIIPVar#15420"[89] = 0.5 * (4.0 * x₄ * α₁₅ * exp(0.00040362096439578745 * (α₁₁ + α₂) + -0.00040362096439578745 * (α₄ + α₇)) ^ -1 + 4.0 * x₅ * α₁₆ * exp(0.00040362096439578745 * (α₁₁ + α₃) + -0.00040362096439578745 * (α₅ + α₇)) ^ -1)
                                    var"##MTIIPVar#15420"[90] = 0.0
                                    var"##MTIIPVar#15420"[91] = 0.0
                                    var"##MTIIPVar#15420"[92] = 0.0
                                    var"##MTIIPVar#15420"[93] = 0.0
                                    var"##MTIIPVar#15420"[94] = 0.0
                                    var"##MTIIPVar#15420"[95] = 0.0
                                    var"##MTIIPVar#15420"[96] = -2.0 * x₅ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1
                                    var"##MTIIPVar#15420"[97] = 2.0 * x₅ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1
                                    var"##MTIIPVar#15420"[98] = α₂₈
                                    var"##MTIIPVar#15420"[99] = 0.5 * (-2.0α₂₈ + -4.0 * x₁₂ * α₂₉ + -4.0 * x₅ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1)
                                    var"##MTIIPVar#15420"[100] = 0.0
                                    var"##MTIIPVar#15420"[101] = 0.0
                                    var"##MTIIPVar#15420"[102] = 0.5 * (4.0 * x₁₂ * α₂₉ + 4.0 * x₅ * α₁₇ * exp(0.00040362096439578745 * (α₁₁ + α₆) + -0.00040362096439578745 * (α₅ + α₈)) ^ -1)
                                    var"##MTIIPVar#15420"[103] = 0.0
                                    var"##MTIIPVar#15420"[104] = 0.0
                                    var"##MTIIPVar#15420"[105] = 0.0
                                    var"##MTIIPVar#15420"[106] = 0.0
                                    var"##MTIIPVar#15420"[107] = 0.0
                                    var"##MTIIPVar#15420"[108] = 0.0
                                    var"##MTIIPVar#15420"[109] = 0.0
                                    var"##MTIIPVar#15420"[110] = 2.0 * x₁₀ * exp(3.697891137634378 + 0.00040362096439578745α₆ + -0.00040362096439578745 * (α₁₀ + α₉)) ^ -1 * α₁₈
                                    var"##MTIIPVar#15420"[111] = 0.0
                                    var"##MTIIPVar#15420"[112] = 0.0
                                    var"##MTIIPVar#15420"[113] = 0.0
                                    var"##MTIIPVar#15420"[114] = -2.0 * x₁₀ * exp(3.697891137634378 + 0.00040362096439578745α₆ + -0.00040362096439578745 * (α₁₀ + α₉)) ^ -1 * α₁₈
                                    var"##MTIIPVar#15420"[115] = 0.0
                                    var"##MTIIPVar#15420"[116] = 0.0
                                    var"##MTIIPVar#15420"[117] = 0.0
                                    var"##MTIIPVar#15420"[118] = 0.0
                                    var"##MTIIPVar#15420"[119] = 2.0 * x₁ * α₁₄
                                    var"##MTIIPVar#15420"[120] = 0.0
                                    var"##MTIIPVar#15420"[121] = 0.0
                                    var"##MTIIPVar#15420"[122] = 2.0 * x₁₃ * α₂₆
                                    var"##MTIIPVar#15420"[123] = 2.0 * x₉ * exp(3.697891137634378 + 0.00040362096439578745α₆ + -0.00040362096439578745 * (α₁₀ + α₉)) ^ -1 * α₁₈
                                    var"##MTIIPVar#15420"[124] = 0.0
                                    var"##MTIIPVar#15420"[125] = 0.0
                                    var"##MTIIPVar#15420"[126] = 0.0
                                    var"##MTIIPVar#15420"[127] = 0.5 * (-4.0 * x₁ * α₁₄ + -4.0 * x₁₃ * α₂₆ + -4.0 * x₉ * exp(3.697891137634378 + 0.00040362096439578745α₆ + -0.00040362096439578745 * (α₁₀ + α₉)) ^ -1 * α₁₈)
                                    var"##MTIIPVar#15420"[128] = 0.0
                                    var"##MTIIPVar#15420"[129] = 0.0
                                    var"##MTIIPVar#15420"[130] = 0.0
                                    var"##MTIIPVar#15420"[131] = 0.0
                                    var"##MTIIPVar#15420"[132] = -2.0 * x₂ * α₁₅
                                    var"##MTIIPVar#15420"[133] = -2.0 * x₃ * α₁₆
                                    var"##MTIIPVar#15420"[134] = 2.0 * x₂ * α₁₅
                                    var"##MTIIPVar#15420"[135] = 0.5 * (4.0 * x₃ * α₁₆ + 4.0 * x₆ * α₁₇)
                                    var"##MTIIPVar#15420"[136] = -2.0 * x₆ * α₁₇
                                    var"##MTIIPVar#15420"[137] = 0.5 * (4.0 * x₂ * α₁₅ + 4.0 * x₃ * α₁₆)
                                    var"##MTIIPVar#15420"[138] = 0.5 * (4.0 * x₁₃ * α₃₀ + 4.0 * x₆ * α₁₇)
                                    var"##MTIIPVar#15420"[139] = 0.0
                                    var"##MTIIPVar#15420"[140] = 0.0
                                    var"##MTIIPVar#15420"[141] = 0.5 * (-4.0 * x₁₃ * α₃₀ + -4.0 * x₂ * α₁₅ + -4.0 * x₃ * α₁₆ + -4.0 * x₆ * α₁₇)
                                    var"##MTIIPVar#15420"[142] = 0.0
                                    var"##MTIIPVar#15420"[143] = 0.0
                                    var"##MTIIPVar#15420"[144] = 0.0
                                    var"##MTIIPVar#15420"[145] = -2.0 * x₂ * α₁₉
                                    var"##MTIIPVar#15420"[146] = 0.5 * (4.0 * x₂ * α₁₉ + -4.0 * x₃ * (α₂₁ + α₃₁))
                                    var"##MTIIPVar#15420"[147] = 0.5 * (4.0 * x₃ * α₂₁ + -4.0 * x₄ * α₂₃)
                                    var"##MTIIPVar#15420"[148] = 0.5 * (4.0 * x₄ * α₂₃ + -4.0 * x₅ * α₂₅)
                                    var"##MTIIPVar#15420"[149] = 2.0 * x₃ * α₃₁
                                    var"##MTIIPVar#15420"[150] = -2.0 * x₇ * α₂₇
                                    var"##MTIIPVar#15420"[151] = 0.5 * (4.0 * x₇ * α₂₇ + -4.0 * x₈ * α₂₉)
                                    var"##MTIIPVar#15420"[152] = 0.0
                                    var"##MTIIPVar#15420"[153] = 2.0 * x₅ * α₂₅
                                    var"##MTIIPVar#15420"[154] = 2.0 * x₈ * α₂₉
                                    var"##MTIIPVar#15420"[155] = 0.0
                                    var"##MTIIPVar#15420"[156] = 0.0
                                    var"##MTIIPVar#15420"[157] = 0.0
                                    var"##MTIIPVar#15420"[158] = 0.0
                                    var"##MTIIPVar#15420"[159] = 2.0 * x₄ * α₂₂
                                    var"##MTIIPVar#15420"[160] = -2.0 * x₄ * α₂₂
                                    var"##MTIIPVar#15420"[161] = 2.0 * x₁₀ * α₂₆
                                    var"##MTIIPVar#15420"[162] = 0.0
                                    var"##MTIIPVar#15420"[163] = 0.0
                                    var"##MTIIPVar#15420"[164] = 2.0 * x₁₁ * α₃₀
                                    var"##MTIIPVar#15420"[165] = 0.0
                                    var"##MTIIPVar#15420"[166] = -2.0 * x₁₀ * α₂₆
                                    var"##MTIIPVar#15420"[167] = -2.0 * x₁₁ * α₃₀
                                    var"##MTIIPVar#15420"[168] = 0.0
                                    var"##MTIIPVar#15420"[169] = 0.0
                                end
                            end
                        end
                    nothing
                end)
        M = UniformScaling{Bool}(true)
        ODEFunction{true}(ff, jac = jac, tgrad = tgrad, mass_matrix = M, jac_prototype = nothing, syms = [:x₁, :x₂, :x₃, :x₄, :x₅, :x₆, :x₇, :x₈, :x₉, :x₁₀, :x₁₁, :x₁₂, :x₁₃])
    end
u0 = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.81e-6, 55500.0]
tspan = (0.0, 1.0e-5)
p = [153400.0, 134110.0, 116740.0, -9648.3, -14472.0, 147620.0, 51136.0, 19296.6, 179458.0, 0.0, 0.0, 0.0, 0.0, 1.0e8, 47320.473243810055, 3.4311035379976365e6, 3.4311035379976365e6, 1.6603867185365406, 5.52486e14, 2.3763009937807666e12, 5.52486e14, 8.308386964362935e-9, 5.52486e14, 7.882700332950447e13, 5.52486e14, 1.900083786848033e17, 5.52486e14, 2.0709841124207065e8, 5.52486e14, 2.2895136995798813e11, 5.52486e14, 3.010694854288576e19]
prob = ODEProblem(ff, u0, tspan, p)

# Reference solution using a robust stiff solver
sol = solve(prob, Rodas5P(), abstol=1e-20, reltol=1e-14)
test_sol = TestSolution(sol)
abstols = 1.0 ./ 10.0 .^ (4:11)
reltols = 1.0 ./ 10.0 .^ (1:8);
```


```julia
plot(sol, vars=[8], xscale = :log10, tspan=(1e-12, 1e-5))
```

![](figures/MKMBatteryChemistry_wpd_2_1.png)

```julia
plot(sol, vars=[11], xscale = :log10, tspan=(1e-12, 1e-5))
```

![](figures/MKMBatteryChemistry_wpd_3_1.png)



## High Tolerances

This is the speed when you just want the answer.

```julia
abstols = 1.0 ./ 10.0 .^ (5:8)
reltols = 1.0 ./ 10.0 .^ (1:4);
setups = [Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>rodas()),
          Dict(:alg=>radau()),
          Dict(:alg=>lsoda()),
          Dict(:alg=>RadauIIA5()),
          ]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_4_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;dense = false,verbose = false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_5_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_6_1.png)

```julia
setups = [Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>Kvaerno3()),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>KenCarp3()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>lsoda()),
          Dict(:alg=>radau())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_7_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;dense = false,verbose = false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_8_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_9_1.png)

```julia
setups = [Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>KenCarp5()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>KenCarp3()),
          Dict(:alg=>ARKODE(order=5)),
          Dict(:alg=>ARKODE()),
          Dict(:alg=>ARKODE(order=3))]
names = ["Rosenbrock23" "KenCarp5" "KenCarp4" "KenCarp3" "ARKODE5" "ARKODE4" "ARKODE3"]
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      names=names,save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_10_1.png)

```julia
setups = [Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>ImplicitEulerExtrapolation()),
          Dict(:alg=>ImplicitEulerBarycentricExtrapolation()),
          Dict(:alg=>ImplicitHairerWannerExtrapolation()),
          Dict(:alg=>ABDF2()),
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
]
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5))
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_11_1.png)

```julia
setups = [Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>ImplicitEulerExtrapolation(linsolve = RFLUFactorization())),
          Dict(:alg=>ImplicitEulerBarycentricExtrapolation(linsolve = RFLUFactorization())),
          Dict(:alg=>ImplicitHairerWannerExtrapolation(linsolve = RFLUFactorization())),
          Dict(:alg=>ABDF2()),
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
]
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5))
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_12_1.png)



### Low Tolerances

This is the speed at lower tolerances, measuring what's good when accuracy is needed.

```julia
abstols = 1.0 ./ 10.0 .^ (7:13)
reltols = 1.0 ./ 10.0 .^ (4:10)

setups = [
          Dict(:alg=>FBDF()),
          Dict(:alg=>QNDF()),
          Dict(:alg=>Rodas4P()),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>ddebdf()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P()),
          Dict(:alg=>rodas()),
          Dict(:alg=>radau()),
          Dict(:alg=>lsoda())
          ]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_13_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      dense=false,appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_14_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_15_1.png)

```julia
setups = [Dict(:alg=>GRK4A()),
          Dict(:alg=>Rodas5()),
          Dict(:alg=>Kvaerno4()),
          Dict(:alg=>Kvaerno5()),
          Dict(:alg=>CVODE_BDF()),
          Dict(:alg=>KenCarp4()),
          Dict(:alg=>KenCarp5()),
          Dict(:alg=>Rodas4()),
          Dict(:alg=>Rodas5P()),
          Dict(:alg=>radau()),
          Dict(:alg=>ImplicitEulerExtrapolation(min_order = 3)),
          Dict(:alg=>ImplicitEulerBarycentricExtrapolation()),
          Dict(:alg=>ImplicitHairerWannerExtrapolation()),
          ]
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_16_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      dense=false,appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_17_1.png)

```julia
wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)
```

![](figures/MKMBatteryChemistry_wpd_18_1.png)



Multithreading benchmarks with Parallel Extrapolation Methods

```julia
#Setting BLAS to one thread to measure gains
LinearAlgebra.BLAS.set_num_threads(1)

abstols = 1.0 ./ 10.0 .^ (11:13)
reltols = 1.0 ./ 10.0 .^ (8:10)

setups = [
            Dict(:alg=>CVODE_BDF()),
            Dict(:alg=>KenCarp4()),
            Dict(:alg=>Rodas4()),
            Dict(:alg=>Rodas5()),
            Dict(:alg=>Rodas5P()),
            Dict(:alg=>QNDF()),
            Dict(:alg=>lsoda()),
            Dict(:alg=>radau()),
            Dict(:alg=>seulex()),
            Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = OrdinaryDiffEq.PolyesterThreads())),
            Dict(:alg=>ImplicitEulerExtrapolation(min_order = 5, init_order = 3,threading = false)),
            Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = OrdinaryDiffEq.PolyesterThreads())),
            Dict(:alg=>ImplicitEulerBarycentricExtrapolation(min_order = 5, threading = false)),
            Dict(:alg=>ImplicitHairerWannerExtrapolation(threading = OrdinaryDiffEq.PolyesterThreads())),
            Dict(:alg=>ImplicitHairerWannerExtrapolation(threading = false)),
            ]

solnames = ["CVODE_BDF","KenCarp4","Rodas4","Rodas5","Rodas5P","QNDF","lsoda","radau","seulex","ImplEulerExtpl (threaded)", "ImplEulerExtpl (non-threaded)",
            "ImplEulerBaryExtpl (threaded)","ImplEulerBaryExtpl (non-threaded)","ImplHWExtpl (threaded)","ImplHWExtpl (non-threaded)"]

wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                    names = solnames,save_everystep=false,appxsol=test_sol,maxiters=Int(1e5),numruns=10)

plot(wp, title = "Implicit Methods: MKM Battery Chemistry",legend=:outertopleft,size = (1000,500),
     xticks = 10.0 .^ (-15:1:1),
     yticks = 10.0 .^ (-6:0.3:5),
     bottom_margin= 5Plots.mm)
```

![](figures/MKMBatteryChemistry_wpd_19_1.png)



### Conclusion

This MKM battery chemistry benchmark tests stiff ODE solvers on a realistic electrochemical system with complex microkinetic reaction mechanisms. The extreme stiffness arising from multiple time scales in the chemical reactions creates significant challenges for adaptive time stepping algorithms, making it an excellent test for robust stiff solvers.


## Appendix

These benchmarks are a part of the SciMLBenchmarks.jl repository, found at: [https://github.com/SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this benchmark, do the following commands:

```
using SciMLBenchmarks
SciMLBenchmarks.weave_file("benchmarks/StiffODE","MKMBatteryChemistry_wpd.jmd")
```

Computer Information:

```
Julia Version 1.10.10
Commit 95f30e51f41 (2025-06-27 09:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 128 × AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 1 default, 0 interactive, 1 GC (on 128 virtual cores)
Environment:
  JULIA_CPU_THREADS = 128
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/5b300254-1738-4989-ae0a-f4d2d937f953

```

Package Information:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/StiffODE/Project.toml`
  [2169fc97] AlgebraicMultigrid v1.1.0
  [6e4b80f9] BenchmarkTools v1.6.0
  [f3b72e0c] DiffEqDevTools v2.48.0
⌃ [5b8099bc] DomainSets v0.7.15
  [5a33fad7] GeometricIntegratorsDiffEq v0.2.5
  [40713840] IncompleteLU v0.2.1
  [7f56f5a3] LSODA v0.7.5
⌃ [7ed4a6bd] LinearSolve v3.18.2
⌃ [94925ecb] MethodOfLines v0.11.8
⌅ [961ee093] ModelingToolkit v9.80.6
  [09606e27] ODEInterfaceDiffEq v3.13.4
⌃ [1dea7af3] OrdinaryDiffEq v6.98.0
⌃ [65888b18] ParameterizedFunctions v5.17.2
⌃ [91a5bcdd] Plots v1.40.14
  [132c30aa] ProfileSVG v0.2.2
  [f2c3362d] RecursiveFactorization v0.2.23
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [90137ffa] StaticArrays v1.9.13
⌃ [c3572dad] Sundials v4.28.0
⌃ [0c5d862f] Symbolics v6.43.0
  [a759f4b9] TimerOutputs v0.5.29
  [37e2e46d] LinearAlgebra
  [2f01184e] SparseArrays v1.10.0
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

And the full manifest:

```
Status `/cache/build/exclusive-amdci3-0/julialang/scimlbenchmarks-dot-jl/benchmarks/StiffODE/Manifest.toml`
⌃ [47edcb42] ADTypes v1.15.0
  [a4c015fc] ANSIColoredPrinters v0.0.1
  [621f4979] AbstractFFTs v1.5.0
  [1520ce14] AbstractTrees v0.4.5
  [7d9f7c33] Accessors v0.1.42
  [79e6a3ab] Adapt v4.3.0
  [2169fc97] AlgebraicMultigrid v1.1.0
  [66dad0bd] AliasTables v1.1.3
⌅ [4c88cf16] Aqua v0.5.6
  [ec485272] ArnoldiMethod v0.4.0
⌃ [4fba245c] ArrayInterface v7.19.0
⌃ [4c555306] ArrayLayouts v1.11.1
  [13072b0f] AxisAlgorithms v1.1.0
  [aae01518] BandedMatrices v1.9.4
  [6e4b80f9] BenchmarkTools v1.6.0
  [e2ed5e7c] Bijections v0.2.2
  [d1d4a3ce] BitFlags v0.1.9
  [62783981] BitTwiddlingConvenienceFunctions v0.1.6
⌃ [8e7c35d0] BlockArrays v1.6.3
⌃ [70df07ce] BracketingNonlinearSolve v1.3.0
  [fa961155] CEnum v0.5.0
⌃ [2a0fbf3d] CPUSummary v0.2.6
⌃ [d360d2e6] ChainRulesCore v1.25.2
  [fb6a15b2] CloseOpenIntervals v0.1.13
  [944b1d66] CodecZlib v0.7.8
⌃ [35d6a980] ColorSchemes v3.29.0
⌅ [3da002f7] ColorTypes v0.11.5
⌃ [c3611d14] ColorVectorSpace v0.10.0
⌅ [5ae59095] Colors v0.12.11
  [861a8166] Combinatorics v1.0.3
  [a80b9123] CommonMark v0.9.1
  [38540f10] CommonSolve v0.2.4
  [bbf7d656] CommonSubexpressions v0.3.1
  [f70d9fcc] CommonWorldInvalidations v1.0.0
  [a09551c4] CompactBasisFunctions v0.2.14
⌃ [34da2185] Compat v4.16.0
  [b152e2b5] CompositeTypes v0.1.4
  [a33af91c] CompositionsBase v0.1.2
  [2569d6c7] ConcreteStructs v0.2.3
  [f0e56b4a] ConcurrentUtilities v2.5.0
  [8f4d0f93] Conda v1.10.2
  [187b0558] ConstructionBase v1.6.0
⌃ [7ae1f121] ContinuumArrays v0.19.2
  [d38c429a] Contour v0.6.3
  [adafc99b] CpuId v0.3.1
  [a8cc5b0e] Crayons v4.1.1
⌃ [717857b8] DSP v0.7.10
  [9a962f9c] DataAPI v1.16.0
⌅ [864edb3b] DataStructures v0.18.22
  [e2d170a0] DataValueInterfaces v1.0.0
  [55939f99] DecFP v1.4.1
  [8bb1440f] DelimitedFiles v1.9.1
⌃ [2b5f629d] DiffEqBase v6.176.0
⌃ [459566f4] DiffEqCallbacks v4.8.0
  [f3b72e0c] DiffEqDevTools v2.48.0
  [77a26b50] DiffEqNoiseProcess v5.24.1
  [163ba53b] DiffResults v1.1.0
  [b552c78f] DiffRules v1.15.1
⌅ [a0c0ee7d] DifferentiationInterface v0.6.54
⌃ [8d63f2c5] DispatchDoctor v0.4.20
  [b4f34e82] Distances v0.10.12
  [31c24e10] Distributions v0.25.120
  [ffbed154] DocStringExtensions v0.9.5
⌅ [e30172f5] Documenter v0.27.25
⌃ [5b8099bc] DomainSets v0.7.15
⌃ [7c1d4256] DynamicPolynomials v0.6.2
⌃ [06fc5a27] DynamicQuantities v1.8.0
  [4e289a0a] EnumX v1.0.5
⌃ [f151be2c] EnzymeCore v0.8.12
  [460bff9d] ExceptionUnwrapping v0.1.11
  [d4d017d3] ExponentialUtilities v1.27.0
  [e2ba6199] ExprTools v0.1.10
  [55351af7] ExproniconLite v0.10.14
⌃ [c87230d0] FFMPEG v0.4.2
⌃ [7a1cc6ca] FFTW v1.9.0
  [7034ab61] FastBroadcast v0.3.5
  [9aa1b823] FastClosures v0.3.2
  [442a2c76] FastGaussQuadrature v1.0.2
  [a4df4552] FastPower v1.1.3
  [057dd010] FastTransforms v0.17.0
  [5789e2e9] FileIO v1.17.0
⌃ [1a297f60] FillArrays v1.13.0
⌃ [64ca27bc] FindFirstFunctions v1.4.1
⌃ [6a86dc24] FiniteDiff v2.27.0
  [53c48c17] FixedPointNumbers v0.8.5
  [08572546] FlameGraphs v1.1.0
  [1fa38f19] Format v1.3.7
⌅ [f6369f11] ForwardDiff v0.10.38
  [069b7b12] FunctionWrappers v1.1.3
  [77dc65aa] FunctionWrappersWrappers v0.1.3
  [d9f16b24] Functors v0.5.2
  [46192b85] GPUArraysCore v0.2.0
  [28b8d3ca] GR v0.73.17
  [a8297547] GenericFFT v0.1.6
⌃ [14197337] GenericLinearAlgebra v0.3.17
  [c145ed77] GenericSchur v0.5.5
⌅ [9a0b12b7] GeometricBase v0.2.6
⌅ [c85262ba] GeometricEquations v0.2.1
⌅ [dcce2d33] GeometricIntegrators v0.9.2
  [5a33fad7] GeometricIntegratorsDiffEq v0.2.5
⌃ [d7ba0133] Git v1.4.0
  [c27321d9] Glob v1.3.1
⌃ [86223c79] Graphs v1.13.0
  [42e2da0e] Grisu v1.0.2
⌅ [f67ccb44] HDF5 v0.16.16
  [cd3eb016] HTTP v1.10.17
  [eafb193a] Highlights v0.5.3
  [3e5b6fbb] HostCPUFeatures v0.1.17
  [34004b35] HypergeometricFunctions v0.3.28
⌃ [7073ff75] IJulia v1.29.0
  [b5f81e59] IOCapture v0.2.5
  [615f187c] IfElse v0.1.1
  [40713840] IncompleteLU v0.2.1
  [9b13fd28] IndirectArrays v1.0.0
  [4858937d] InfiniteArrays v0.15.6
  [e1ba4f0e] Infinities v0.1.11
  [d25df0c9] Inflate v0.1.5
⌃ [18e54dd8] IntegerMathUtils v0.1.2
⌃ [a98d9a8b] Interpolations v0.16.1
  [8197267c] IntervalSets v0.7.11
  [3587e190] InverseFunctions v0.1.17
  [92d709cd] IrrationalConstants v0.2.4
  [c8e1da08] IterTools v1.10.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [1019f520] JLFzf v0.1.11
⌃ [692b3bcd] JLLWrappers v1.7.0
  [682c06a0] JSON v0.21.4
  [ae98c720] Jieko v0.2.1
⌃ [98e50ef6] JuliaFormatter v2.1.2
⌅ [70703baa] JuliaSyntax v0.4.10
⌃ [ccbc3e58] JumpProcesses v9.16.1
  [ba0b0d4f] Krylov v0.10.1
  [7f56f5a3] LSODA v0.7.5
  [b964fa9f] LaTeXStrings v1.4.0
⌃ [23fbe1c1] Latexify v0.16.8
  [10f19ff3] LayoutPointers v0.1.17
⌃ [5078a376] LazyArrays v2.6.1
⌃ [1d6d02ad] LeftChildRightSiblingTrees v0.2.0
  [87fe0de2] LineSearch v0.1.4
  [d3d80556] LineSearches v7.4.0
⌃ [7ed4a6bd] LinearSolve v3.18.2
  [2ab3a3ac] LogExpFunctions v0.3.29
  [e6f89c97] LoggingExtras v1.1.0
  [bdcacae8] LoopVectorization v0.12.172
  [d8e11817] MLStyle v0.4.17
  [3da0fdf6] MPIPreferences v0.1.11
  [1914dd2f] MacroTools v0.5.16
  [d125e4d3] ManualMemory v0.1.8
  [bb5d69b7] MaybeInplace v0.1.4
  [739be429] MbedTLS v1.1.9
  [442fdcdd] Measures v0.3.2
⌃ [94925ecb] MethodOfLines v0.11.8
  [e1d29d7a] Missings v1.2.0
⌅ [961ee093] ModelingToolkit v9.80.6
⌃ [2e0e35c7] Moshi v0.3.6
  [46d2c3a1] MuladdMacro v0.2.4
⌃ [102ac46a] MultivariatePolynomials v0.5.9
  [ffc61752] Mustache v1.0.21
⌃ [d8a4904e] MutableArithmetics v1.6.4
  [d41bc354] NLSolversBase v7.10.0
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v1.1.3
⌃ [8913a72c] NonlinearSolve v4.9.0
⌃ [be0214bd] NonlinearSolveBase v1.12.0
⌃ [5959db7a] NonlinearSolveFirstOrder v1.5.0
⌃ [9a2c21bd] NonlinearSolveQuasiNewton v1.6.0
⌃ [26075421] NonlinearSolveSpectralMethods v1.2.0
  [54ca160b] ODEInterface v0.5.0
  [09606e27] ODEInterfaceDiffEq v3.13.4
  [6fe1bfb0] OffsetArrays v1.17.0
  [4d8831e6] OpenSSL v1.5.0
  [429524aa] Optim v1.13.2
  [bac558e1] OrderedCollections v1.8.1
⌃ [1dea7af3] OrdinaryDiffEq v6.98.0
⌃ [89bda076] OrdinaryDiffEqAdamsBashforthMoulton v1.2.0
⌃ [6ad6398a] OrdinaryDiffEqBDF v1.6.0
⌃ [bbf590c4] OrdinaryDiffEqCore v1.26.1
⌃ [50262376] OrdinaryDiffEqDefault v1.4.0
⌃ [4302a76b] OrdinaryDiffEqDifferentiation v1.10.0
⌃ [9286f039] OrdinaryDiffEqExplicitRK v1.1.0
⌃ [e0540318] OrdinaryDiffEqExponentialRK v1.4.0
⌃ [becaefa8] OrdinaryDiffEqExtrapolation v1.5.0
⌃ [5960d6e9] OrdinaryDiffEqFIRK v1.12.0
⌃ [101fe9f7] OrdinaryDiffEqFeagin v1.1.0
⌃ [d3585ca7] OrdinaryDiffEqFunctionMap v1.1.1
⌃ [d28bc4f8] OrdinaryDiffEqHighOrderRK v1.1.0
⌃ [9f002381] OrdinaryDiffEqIMEXMultistep v1.3.0
⌃ [521117fe] OrdinaryDiffEqLinear v1.3.0
⌃ [1344f307] OrdinaryDiffEqLowOrderRK v1.2.0
⌃ [b0944070] OrdinaryDiffEqLowStorageRK v1.3.0
⌃ [127b3ac7] OrdinaryDiffEqNonlinearSolve v1.10.0
⌃ [c9986a66] OrdinaryDiffEqNordsieck v1.1.0
⌃ [5dd0a6cf] OrdinaryDiffEqPDIRK v1.3.1
⌃ [5b33eab2] OrdinaryDiffEqPRK v1.1.0
⌃ [04162be5] OrdinaryDiffEqQPRK v1.1.0
⌃ [af6ede74] OrdinaryDiffEqRKN v1.1.0
⌃ [43230ef6] OrdinaryDiffEqRosenbrock v1.11.0
⌃ [2d112036] OrdinaryDiffEqSDIRK v1.3.0
⌃ [669c94d9] OrdinaryDiffEqSSPRK v1.3.0
⌃ [e3e12d00] OrdinaryDiffEqStabilizedIRK v1.3.0
⌃ [358294b1] OrdinaryDiffEqStabilizedRK v1.1.0
⌃ [fa646aed] OrdinaryDiffEqSymplecticRK v1.3.0
⌃ [b1df2697] OrdinaryDiffEqTsit5 v1.1.0
⌃ [79d7bb75] OrdinaryDiffEqVerner v1.2.0
⌃ [a7812802] PDEBase v0.1.17
  [90014a1f] PDMats v0.11.35
⌃ [65888b18] ParameterizedFunctions v5.17.2
  [d96e819e] Parameters v0.12.3
  [69de0a69] Parsers v2.8.3
  [ccf2f8ad] PlotThemes v3.3.0
  [995b91a9] PlotUtils v1.4.3
⌃ [91a5bcdd] Plots v1.40.14
⌃ [e409e4f3] PoissonRandom v0.4.4
  [f517fe37] Polyester v0.7.18
  [1d0040c9] PolyesterWeave v0.2.2
⌅ [f27b6e38] Polynomials v3.2.13
  [85a6dd25] PositiveFactorizations v0.2.4
⌃ [d236fae5] PreallocationTools v0.4.27
⌅ [aea7be01] PrecompileTools v1.2.1
⌃ [21216c6a] Preferences v1.4.3
⌅ [08abe8d2] PrettyTables v2.4.0
  [27ebfcd6] Primes v0.5.7
  [132c30aa] ProfileSVG v0.2.2
⌃ [92933f4c] ProgressMeter v1.10.4
  [43287f4e] PtrArrays v1.3.0
  [1fd47b50] QuadGK v2.11.2
  [a08977f5] QuadratureRules v0.1.6
  [c4ea9172] QuasiArrays v0.12.0
  [74087812] Random123 v1.7.1
  [e6cf234a] RandomNumbers v1.6.0
  [c84ed2f1] Ratios v0.4.5
  [3cdcf5f2] RecipesBase v1.3.4
  [01d81517] RecipesPipeline v0.6.12
  [807425ed] RecurrenceRelationships v0.2.0
⌃ [731186ca] RecursiveArrayTools v3.33.0
  [f2c3362d] RecursiveFactorization v0.2.23
  [189a3867] Reexport v1.2.2
  [05181044] RelocatableFolders v1.0.1
  [ae029012] Requires v1.3.1
  [ae5879a3] ResettableStacks v1.1.1
  [79098fc4] Rmath v0.8.0
  [47965b36] RootedTrees v2.23.1
⌅ [fb486d5c] RungeKutta v0.4.6
  [7e49a35a] RuntimeGeneratedFunctions v0.5.15
⌃ [9dfe8606] SCCNonlinearSolve v1.3.0
  [94e857df] SIMDTypes v0.1.0
  [476501e8] SLEEFPirates v0.6.43
⌃ [0bca4576] SciMLBase v2.102.1
  [31c91b34] SciMLBenchmarks v0.1.3
⌃ [19f34311] SciMLJacobianOperators v0.1.6
⌃ [c0aeaf25] SciMLOperators v1.3.1
  [53ae85a6] SciMLStructures v1.7.0
  [6c6a2e73] Scratch v1.3.0
  [efcf1570] Setfield v1.1.2
  [992d4aef] Showoff v1.0.3
  [777ac1f9] SimpleBufferStream v1.2.0
⌃ [727e6d20] SimpleNonlinearSolve v2.5.0
⌅ [36b790f5] SimpleSolvers v0.2.4
⌃ [699a6c99] SimpleTraits v0.9.4
  [ce78b400] SimpleUnPack v1.1.0
  [b85f4697] SoftGlobalScope v1.1.0
⌃ [a2af1166] SortingAlgorithms v1.2.1
  [0a514795] SparseMatrixColorings v0.4.21
  [276daf66] SpecialFunctions v2.5.1
  [860ef19b] StableRNGs v1.0.3
  [aedffcd0] Static v1.2.0
  [0d7ed370] StaticArrayInterface v1.8.0
⌃ [90137ffa] StaticArrays v1.9.13
  [1e83bf80] StaticArraysCore v1.4.3
  [82ae8749] StatsAPI v1.7.1
⌃ [2913bbd2] StatsBase v0.34.5
  [4c63d2b9] StatsFuns v1.5.0
⌃ [7792a7ef] StrideArraysCore v0.5.7
  [69024149] StringEncodings v0.3.7
  [892a3eda] StringManipulation v0.4.1
  [09ab397b] StructArrays v0.7.1
⌃ [c3572dad] Sundials v4.28.0
⌃ [2efcf032] SymbolicIndexingInterface v0.3.41
⌃ [19f23fe9] SymbolicLimits v0.2.2
⌃ [d1185830] SymbolicUtils v3.29.0
⌃ [0c5d862f] Symbolics v6.43.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.1
⌃ [ed4db957] TaskLocalValues v0.1.2
  [62fd8b95] TensorCore v0.1.1
  [8ea1fca8] TermInterface v2.0.0
  [1c621080] TestItems v1.0.0
  [8290d209] ThreadingUtilities v0.5.5
  [a759f4b9] TimerOutputs v0.5.29
  [c751599d] ToeplitzMatrices v0.8.5
  [3bb67fe8] TranscodingStreams v0.11.3
  [d5829a12] TriangularSolve v0.2.1
⌃ [410a4b4d] Tricks v0.1.10
  [781d530d] TruncatedStacktraces v1.4.0
⌃ [5c2747f8] URIs v1.6.0
  [3a884ed6] UnPack v1.0.2
  [1cfade01] UnicodeFun v0.4.1
⌅ [1986cc42] Unitful v1.23.1
  [45397f5d] UnitfulLatexify v1.7.0
  [a7c27f48] Unityper v0.1.6
  [41fe7b60] Unzip v0.2.0
⌃ [3d5dd08c] VectorizationBase v0.21.71
  [81def892] VersionParsing v1.3.0
  [897b6980] WeakValueDicts v0.1.0
  [44d3d7a6] Weave v0.10.12
  [efce3f68] WoodburyMatrices v1.0.0
  [ddb6d928] YAML v0.4.14
⌃ [c2297ded] ZMQ v1.4.1
  [6e34b625] Bzip2_jll v1.0.9+0
  [83423d85] Cairo_jll v1.18.5+0
  [ee1fde0b] Dbus_jll v1.16.2+0
  [47200ebd] DecFP_jll v2.0.3+1
  [2702e6a9] EpollShim_jll v0.0.20230411+1
⌃ [2e619515] Expat_jll v2.6.5+0
⌅ [b22a6f82] FFMPEG_jll v4.4.4+1
  [f5851436] FFTW_jll v3.3.11+0
  [34b6f7d7] FastTransforms_jll v0.6.3+0
⌃ [a3f928ae] Fontconfig_jll v2.16.0+0
  [d7e528f0] FreeType2_jll v2.13.4+0
  [559328eb] FriBidi_jll v1.0.17+0
  [0656b61e] GLFW_jll v3.4.0+2
  [d2c73de3] GR_jll v0.73.17+0
  [b0724c58] GettextRuntime_jll v0.22.4+0
⌃ [f8c6e375] Git_jll v2.50.0+0
⌃ [7746bdde] Glib_jll v2.84.3+0
  [3b182d85] Graphite2_jll v1.3.15+0
  [0234f1f7] HDF5_jll v1.14.6+0
  [2e76f6c2] HarfBuzz_jll v8.5.1+0
⌃ [e33a78d0] Hwloc_jll v2.12.1+0
⌃ [1d5cc7b8] IntelOpenMP_jll v2025.0.4+0
⌃ [aacddb02] JpegTurbo_jll v3.1.1+0
⌃ [c1c5ebd0] LAME_jll v3.100.2+0
  [88015f11] LERC_jll v4.0.1+0
  [1d63c593] LLVMOpenMP_jll v18.1.8+0
  [aae0fff6] LSODA_jll v0.1.2+0
  [dd4b983a] LZO_jll v2.10.3+0
  [e9f186c6] Libffi_jll v3.4.7+0
  [7e76a0d4] Libglvnd_jll v1.7.1+1
  [94ce4f54] Libiconv_jll v1.18.0+0
⌃ [4b2f31a3] Libmount_jll v2.41.0+0
  [89763e89] Libtiff_jll v4.7.1+0
⌃ [38a345b3] Libuuid_jll v2.41.0+0
⌃ [856f044c] MKL_jll v2025.0.1+1
  [7cb0a576] MPICH_jll v4.3.1+0
  [f1f71cc9] MPItrampoline_jll v5.5.4+0
  [9237b28f] MicrosoftMPI_jll v10.1.4+3
⌃ [c771fb93] ODEInterface_jll v0.0.1+0
⌃ [e7412a2a] Ogg_jll v1.3.5+1
  [fe0851c0] OpenMPI_jll v5.0.8+0
⌃ [9bd350c2] OpenSSH_jll v10.0.1+0
⌃ [458c3c95] OpenSSL_jll v3.5.0+0
  [efe28fd5] OpenSpecFun_jll v0.5.6+0
⌃ [91d4177d] Opus_jll v1.3.3+0
⌃ [36c8627f] Pango_jll v1.56.3+0
⌅ [30392449] Pixman_jll v0.44.2+0
  [c0090381] Qt6Base_jll v6.8.2+1
  [629bc702] Qt6Declarative_jll v6.8.2+1
  [ce943373] Qt6ShaderTools_jll v6.8.2+1
  [e99dba38] Qt6Wayland_jll v6.8.2+1
  [f50d1b31] Rmath_jll v0.5.1+0
⌅ [fb77eaff] Sundials_jll v5.2.2+0
  [a44049a8] Vulkan_Loader_jll v1.3.243+0
⌃ [a2964d1f] Wayland_jll v1.23.1+1
⌅ [02c8fc9c] XML2_jll v2.13.6+1
  [ffd25f8a] XZ_jll v5.8.1+0
  [f67eecfb] Xorg_libICE_jll v1.1.2+0
  [c834827a] Xorg_libSM_jll v1.2.6+0
  [4f6342f7] Xorg_libX11_jll v1.8.12+0
  [0c0b7dd1] Xorg_libXau_jll v1.0.13+0
  [935fb764] Xorg_libXcursor_jll v1.2.4+0
  [a3789734] Xorg_libXdmcp_jll v1.1.6+0
  [1082639a] Xorg_libXext_jll v1.3.7+0
  [d091e8ba] Xorg_libXfixes_jll v6.0.1+0
  [a51aa0fd] Xorg_libXi_jll v1.8.3+0
  [d1454406] Xorg_libXinerama_jll v1.1.6+0
  [ec84b674] Xorg_libXrandr_jll v1.5.5+0
  [ea2f1a96] Xorg_libXrender_jll v0.9.12+0
  [c7cfdc94] Xorg_libxcb_jll v1.17.1+0
  [cc61e674] Xorg_libxkbfile_jll v1.1.3+0
  [e920d4aa] Xorg_xcb_util_cursor_jll v0.1.5+0
  [12413925] Xorg_xcb_util_image_jll v0.4.1+0
  [2def613f] Xorg_xcb_util_jll v0.4.1+0
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.1+0
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.10+0
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.2+0
  [35661453] Xorg_xkbcomp_jll v1.4.7+0
  [33bec58e] Xorg_xkeyboard_config_jll v2.44.0+0
  [c5fb5394] Xorg_xtrans_jll v1.6.0+0
  [8f1865be] ZeroMQ_jll v4.3.6+0
  [3161d3a3] Zstd_jll v1.5.7+1
  [35ca27e7] eudev_jll v3.2.14+0
  [214eeab7] fzf_jll v0.61.1+0
⌃ [477f73a3] libaec_jll v1.1.3+0
⌃ [a4ae2306] libaom_jll v3.11.0+0
⌅ [0ac62f75] libass_jll v0.15.2+0
  [1183f4f0] libdecor_jll v0.2.2+0
  [2db6ffa8] libevdev_jll v1.13.4+0
⌃ [f638f0a6] libfdk_aac_jll v2.0.3+0
  [36db933b] libinput_jll v1.28.1+0
⌃ [b53b4c65] libpng_jll v1.6.49+0
  [a9144af2] libsodium_jll v1.0.21+0
⌃ [f27f6e37] libvorbis_jll v1.3.7+2
  [009596ad] mtdev_jll v1.1.7+0
  [1317d2d5] oneTBB_jll v2022.0.0+0
⌅ [1270edf5] x264_jll v2021.5.5+0
⌅ [dfaa095f] x265_jll v3.5.0+0
  [d8fb68d0] xkbcommon_jll v1.9.2+0
  [0dad84c5] ArgTools v1.1.1
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8ba89e20] Distributed
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.10.0
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays v1.10.0
  [10745b16] Statistics v1.10.0
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll v1.1.1+0
  [781609d7] GMP_jll v6.2.1+6
  [deac9b47] LibCURL_jll v8.4.0+0
  [e37daf67] LibGit2_jll v1.6.4+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [3a97d323] MPFR_jll v4.2.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.2+1
  [14a3606d] MozillaCACerts_jll v2023.1.10
  [4536629a] OpenBLAS_jll v0.3.23+4
  [05823500] OpenLibm_jll v0.8.1+4
  [efcefdf7] PCRE2_jll v10.42.0+1
  [bea87d4a] SuiteSparse_jll v7.2.1+1
  [83775a58] Zlib_jll v1.2.13+1
  [8e850b90] libblastrampoline_jll v5.11.0+0
  [8e850ede] nghttp2_jll v1.52.0+1
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated -m`
Warning The project dependencies or compat requirements have changed since the manifest was last resolved. It is recommended to `Pkg.resolve()` or consider `Pkg.update()` if necessary.
```

