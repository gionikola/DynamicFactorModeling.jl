using DynamicFactorModeling
using Random
using Statistics

# One factor shared by three series, with different intercepts and loadings.
model = HDFM(
    nlevels=1, nvar=3, nfactors=[1], fassign=ones(Int, 3, 1),
    flags=[1], varlags=[1, 1, 1],
    varcoefs=[0.0 1.0; 1.0 0.8; -1.0 1.2],
    varlagcoefs=fill(0.2, 3, 1),
    fcoefs=[reshape([0.7], 1, 1)], fvars=[[1.0]],
    varvars=fill(0.3, 3),
)
ss = convertHDFMtoSS(model)
# Simulation and estimation both use stationary initialization.
y, _, states = simulateSSModel(MersenneTwister(42), 100, ss)
settings = DFMStruct(factorlags=1, errorlags=1, ndraws=500, burnin=500)
fit = KN1LevelEstimator(MersenneTwister(43), y, settings)
coefficient_means = reshape(vec(fit.means.B), 2, 3)'
println("Posterior mean intercepts and loadings (one series per row):")
display(coefficient_means)
println("Correlation of the estimated and simulated factor: ",
        round(cor(fit.means.F[:,1], states[:,2]); digits=3))
println("Long-run factor shares in the simulation model:")
display(variance_decomposition(model).factors)
# A single short chain demonstrates the workflow; use several longer chains
# and inspect mixing before drawing substantive conclusions.
