
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


plot(sol, vars=[8], xscale = :log10, tspan=(1e-12, 1e-5))


plot(sol, vars=[11], xscale = :log10, tspan=(1e-12, 1e-5))


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


wp = WorkPrecisionSet(prob,abstols,reltols,setups;dense = false,verbose = false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)


wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)


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


wp = WorkPrecisionSet(prob,abstols,reltols,setups;dense = false,verbose = false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)


wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)


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


wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      dense=false,appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)


wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)


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


wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      dense=false,appxsol=test_sol,maxiters=Int(1e5),error_estimate=:l2,numruns=10)
plot(wp)


wp = WorkPrecisionSet(prob,abstols,reltols,setups; verbose=false,
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2,numruns=10)
plot(wp)


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


using SciMLBenchmarks
SciMLBenchmarks.bench_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

