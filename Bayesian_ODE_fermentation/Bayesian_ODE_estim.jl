md"""
---
title: "Example of classical and Bayesian parameter estimation for ODE models"
author: "David Métivier"
date: last-modified
format:
  html:
    toc: true
engine: julia
julia:
  exeflags: ["--threads=auto"]
---
"""

md"""
This notebook demonstrates parameter estimation for a system of ordinary differential equations (ODEs) using both classical optimization and Bayesian inference. 
The differential equation ecosystem uses SciML packages: DifferentialEquations.jl (OrdinaryDiffEq.jl), DiffEqParamEstim.jl, and Optimization.jl.
The Bayesian inference is performed using Turing.jl.

The example focuses on a toy fermentation process model, estimating parameters from data.

I was inspired by this Tutorial: [https://turinglang.org/docs/tutorials/bayesian-differential-equations/](https://turinglang.org/docs/tutorials/bayesian-differential-equations/)
"""

using Markdown #src
using CSV, DataFrames, DataFramesMeta, Random
using StatsPlots, LaTeXStrings
using GLM # Linear Model
using ComponentArrays: ComponentArray, ComponentVector
Random.seed!(2)
default(fontfamily="Computer Modern", linewidth=1, label=nothing)

md"""
# Loading Data and Preliminary Analysis
"""

md"""
# Chargement et affichage des donnees
"""

data = CSV.read("batch.txt", DataFrame; normalizenames=true, delim=',')
data[:, :N] = data[:, :N] / 1000
data[:, :dCO2dt] = data[:, :dCO2dt] / 100
Nbdata = nrow(data) # nombre de données

begin
    pB = @df data scatter(:X, :N, xlabel="X", ylabel="N", c=:red)
    pS = @df data scatter(:E, :S, xlabel="E", ylabel="S", c=:red)
    plot(pB, pS)
end

md"""
## Linear Regression: Estimating k₁ and k₂

Use linear regression to estimate parameters k₁ and k₂ from the data.
"""

fit_1 = lm(@formula(-N ~ X), data)
c1_est, k1_est = coef(fit_1)
println("Estimated k₁ = ", k1_est)
println("R² for k₁ fit = ", r2(fit_1))

begin
    @df data scatter(:X, :N, xlabel="X", ylabel="N", c=:red, label="data")
    @df data plot!(:X, -k1_est * :X .- c1_est, c=:red, label="fit")
end

fit_2 = lm(@formula(-S ~ E), data)
c2_est, k2_est = coef(fit_2)
println("Estimated k₂ = ", k2_est)
println("R² for k₂ fit = ", r2(fit_2))

begin
    @df data scatter(:E, :S, xlabel="E", ylabel="S", c=:red, label="data")
    @df data plot!(:E, -k2_est * :E .- c2_est, c=:red, label="fit")
end

md"""
# Parameter Estimation with Differential Equations

Estimate model parameters by fitting an ODE model to the data.
"""

using Optimization, ForwardDiff, OptimizationOptimJL
using OrdinaryDiffEq
using DiffEqParamEstim

md"""
## ODE Model Definition

Define the fermentation process as a system of differential equations.
"""

function fermenteur!(du, u, p, t)
    ## Model parameters.
    k₁, k₂, μ₁max, μ₂max, KN, KE, KS = p

    ## Current state.
    X, N, E, S = u

    ## Calculation of μ₁(S)
    μ₁ = μ₁max * N / (KN + N)

    ## Calculation of μ₂(E,S)
    μ₂ = μ₂max * S / (KS + S) * KE / (KE + E)

    ## Evaluate differential equations.
    du[1] = μ₁ * X # dX
    du[2] = -k₁ * μ₁ * X # dN
    du[3] = μ₂ * X # dE
    du[4] = -k₂ * μ₂ * X # dS

    return nothing
end

# Problem Definition
tspan = (0., 90.)
x0_data_val = Matrix(data[1:1, [:X, :N, :E, :S]]) |> vec
StartList_1 = ComponentVector(k₁=0.01, k₂=2, μ₁max=1.2, μ₂max=1.2, KN=1.6, KE=12, KS=0.03)

prob = ODEProblem(fermenteur!, x0_data_val, tspan, StartList_1)

md"""
## Parameter Estimation using classical Optimization
"""

ts_obs = data[:, :time]
data_obs = Matrix(data[:, [:X, :N, :E, :S]]) |> permutedims

# Cost function
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(ts_obs, data_obs),
    Optimization.AutoForwardDiff(),
    maxiters=10000, verbose=false)
lower = 0.001 * ones(length(StartList_1))
upper = 20 * ones(length(StartList_1)
)

# Optimization problem
optprob = Optimization.OptimizationProblem(cost_function, StartList_1; lb=lower, ub=upper
)

# Solve with different algorithms
result_neldermead = solve(optprob, NelderMead())
result_bfgs = solve(optprob, BFGS())

println("BFGS objective: ", result_bfgs.objective)
println("NelderMead objective: ", result_neldermead.objective)

md"""
Plotting the fitted model
"""
so_bfgs = solve(prob, p=result_bfgs, reltol=1e-6)
so_neldermead = solve(prob, p=result_neldermead, reltol=1e-6)

begin
    scatter(data[:, :time], Matrix(data[:, [:X, :N, :E, :S]]))
    plot!(so_bfgs, c=permutedims(1:4), label=permutedims(String.([:X, :N, :E, :S])))
    plot!(so_neldermead, c=permutedims(1:4), linestyle=:dash, label=:none)
end

md"""
# Bayesian Inference with Turing.jl

Estimate parameters and uncertainty using Bayesian methods.
"""

using Turing, LinearAlgebra, Distributions

md"""
## Bayesian Model Definition
"""

all_param = [:σX, :σN, :σE, :σS, :k₁, :k₂, :μ₁max, :μ₂max, :KN, :KE, :KS]
all_param_Latex = [L"\sigma_X", L"\sigma_N", L"\sigma_E", L"\sigma_S", L"k_1", L"k_2", L"\mu_{1,max}", L"\mu_{2,max}", L"K_N", L"K_E", L"K_S"]

ode_param = all_param[5:end]
ode_param_Latex = all_param_Latex[5:end]

# Model definition
@model function fitVin(data, prob, ts)
    ## Priors for noise and parameters
    σ ~ filldist(InverseGamma(3, 2), 4)

    k₁ ~ truncated(Normal(prob.p[1], 0.02); lower=0.)
    k₂ ~ truncated(Normal(prob.p[2], 0.2); lower=0)
    μ₁max ~ truncated(Normal(prob.p[3], 0.2); lower=0)
    μ₂max ~ truncated(Normal(prob.p[4], 0.2); lower=0)
    KN ~ truncated(Normal(prob.p[5], 0.2); lower=0)
    KE ~ truncated(Normal(prob.p[6], 1); lower=0)
    KS ~ truncated(Normal(prob.p[7], 0.01); lower=0)

    ## Simulate ODE with sampled parameters
    p = [k₁, k₂, μ₁max, μ₂max, KN, KE, KS]
    prob_samp = remake(prob; p=p)
    predicted = solve(prob_samp, Tsit5(); saveat=ts, abstol=1e-6)

    ## Likelihood: compare model to data
    for i in eachindex(predicted)
        data[:, i] ~ MvNormal(predicted[i], Diagonal(σ) .^ 2)
    end

    return nothing
end

probT = ODEProblem(fermenteur!, x0_data_val, tspan, [0.06, 2., 0.2, 0.94, 0.01, 19.7, 0.76])
obs = Matrix(data[:, [:X, :N, :E, :S]]) |> permutedims
model = fitVin(obs, probT, data[:, :time])

#-
## using Turing.DynamicPPL
## DynamicPPL.DebugUtils.model_warntype(model)

md"""
## Sampling from the Prior and Posterior
"""

@time "Sampling from the Prior" sample_prior = sample(model, Prior(), 1000; progress=false)
@time "Sampling from the Posterior" sample_posterior = sample(model, NUTS(), 1000; progress=false)

md"""
## Results analysis
"""

md"""
### Posterior Analysis
"""
plot(sample_posterior)

#-

begin
    plt = [stephist(sample_posterior[key], label=ode_param_Latex[i]) for (i, key) in enumerate(ode_param)]
    [stephist!(plt[i], sample_prior[key], label="Prior") for (i, key) in enumerate(ode_param)]
    [vline!(plt[i], [result_bfgs[i]]) for (i, k) in enumerate(ode_param)]
    plt_σ = [stephist(sample_posterior[key], label=string(all_param[i])) for (i, key) in enumerate([Symbol("σ[1]"), Symbol("σ[2]"), Symbol("σ[3]"), Symbol("σ[4]")])]
    [stephist!(plt_σ[i], sample_prior[key], label="Prior") for (i, key) in enumerate([Symbol("σ[1]"), Symbol("σ[2]"), Symbol("σ[3]"), Symbol("σ[4]")])]
    plot(plt..., plt_σ...)
end

md"""
### Parameter Correlations
"""

heatmap(cor(Array(sample_posterior)), yflip=true, xticks=(1:11, all_param_Latex), yticks=(1:11, all_param_Latex), clims=(-1, 1), color=:balance)

md"""
### Posterior Predictive Check

Simulate data from the posterior and compare to observed data.
"""

Ys = map(eachrow(Array(sample_posterior[ode_param]))) do ps
    sol_p = Array(solve(prob; p=ps, saveat=0.1, reltol=1e-6))
end |> stack
ts = 0:0.1:90
styles = [:auto :dot :auto :dash]
vars = String.([:X, :N, :E, :S])
begin
    plt = plot(title="Data retrodiction")
    for (i, label) in enumerate(vars)
        errorline!(plt, ts, Ys[i, :, :], label=label, c=i, groupcolor=i, errortype=:percentile, percentiles=[0, 100], fillalpha=0.8)
    end
    plot!(so_bfgs, c=:black, s=:dash, label=:none, linewidth=2)
    [scatter!(data[:, :time], (data[:, Symbol(label)]), c=:red, label=:none) for label in vars]
    plt
end

md"""
## Handling Missing Observations

You can fit the model even if some variables are not observed (e.g., missing N).
"""

@model function fitVin_Missing(data, prob, ts)
    ## Prior distributions.
    σ ~ filldist(InverseGamma(3, 2), 3)

    k₁ ~ truncated(Normal(prob.p[1], 0.02); lower=0.)
    k₂ ~ truncated(Normal(prob.p[2], 0.2); lower=0)
    μ₁max ~ truncated(Normal(prob.p[3], 0.2); lower=0)
    μ₂max ~ truncated(Normal(prob.p[4], 0.2); lower=0)
    KN ~ truncated(Normal(prob.p[5], 0.2); lower=0)
    KE ~ truncated(Normal(prob.p[6], 1); lower=0)
    KS ~ truncated(Normal(prob.p[7], 0.01); lower=0)

    ## Simulate ODE model. 
    p = [k₁, k₂, μ₁max, μ₂max, KN, KE, KS]
    prob_samp = remake(prob; p=p)
    predicted = solve(prob_samp, Tsit5(); saveat=ts, abstol=1e-6, save_idxs=1:3)

    ## Observations (N is missing, so only X, E, S are used).
    for i in eachindex(predicted)
        data[:, i] ~ MvNormal(predicted[i], Diagonal(σ) .^ 2)
    end

    return nothing
end

md"""
## Sampling with Missing Observations
"""
obs_missing = Matrix(data[:, [:X, :E, :S]]) |> permutedims
model_missing = fitVin_Missing(obs_missing, probT, data[:, :time])

@time "Sampling Posterior with N missing" sample_posterior_missing = sample(model_missing, NUTS(), 1000; progress=false)
@time "Sampling Prior with N missing" sample_prior_missing = sample(model_missing, Prior(), 1000; progress=false)

md"""
## Posterior Analysis with Missing Observations
"""

begin
    plt_missing = [stephist(sample_posterior_missing[key], label=ode_param_Latex[i]) for (i, key) in enumerate(ode_param)]
    [stephist!(plt_missing[i], sample_prior_missing[key], label="Prior") for (i, key) in enumerate(ode_param)]
    [vline!(plt_missing[i], [result_bfgs[i]]) for (i, k) in enumerate(ode_param)]
    plt_σ = [stephist(sample_posterior_missing[key], label=string(all_param[i])) for (i, key) in enumerate([Symbol("σ[1]"), Symbol("σ[2]"), Symbol("σ[3]")])]
    [stephist!(plt_σ[i], sample_prior_missing[key], label="Prior") for (i, key) in enumerate([Symbol("σ[1]"), Symbol("σ[2]"), Symbol("σ[3]")])]
    plot(plt_missing..., plt_σ...)
end

md"""
Correlation plot with missing observations
"""
heatmap(cor(Array(sample_posterior_missing)), yflip=true, xticks=(1:10, all_param_Latex[[1:3; 5:11]]), yticks=(1:10, all_param_Latex[[1:3; 5:11]]), clims=(-1, 1), color=:balance)

md"""
Retrodiction plot for missing observations
"""

Ys_missing = map(eachrow(Array(sample_posterior_missing[ode_param]))) do ps
    Array(solve(probT; p=ps, saveat=0.1, reltol=1e-6))
end |> stack

begin
    plt = plot(title="Data retrodiction (missing observations of N)", legend=false)
    for (i, label) in enumerate(vars)
        errorline!(plt, ts, Ys_missing[i, :, :], label=label, c=i, groupcolor=i, errortype=:percentile, percentiles=[0, 100], fillalpha=0.8)
    end
    [scatter!(data[:, :time], data[:, Symbol(label)], c=:red, label=:none) for label in vars]
    plt
end

md"""
# Julia, Computer and Packages settings
"""
#| code-fold: true
using InteractiveUtils
InteractiveUtils.versioninfo()

import Pkg;
Pkg.status();
