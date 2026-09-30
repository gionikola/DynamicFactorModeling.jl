using DynamicFactorModeling
using Random

# All six series share a global factor. Two groups share separate local factors.
assignments = [1 1; 1 1; 1 1; 1 2; 1 2; 1 2]
model = HDFM(
    nlevels=2, nvar=6, nfactors=[1, 2], fassign=assignments,
    flags=[1, 1], varlags=zeros(Int, 6),
    varcoefs=[0.0 1.0 0.8; 0.5 0.7 1.2; -0.5 1.3 0.6;
              0.0 0.9 1.1; 1.0 1.2 0.7; -1.0 0.8 1.3],
    varlagcoefs=zeros(6, 0),
    fcoefs=[reshape([0.7], 1, 1), reshape([0.4, 0.2], 2, 1)],
    fvars=[[1.0], [1.0, 1.0]], varvars=fill(0.2, 6),
)
ss = convertHDFMtoSS(model)
y, _, _ = simulateSSModel(MersenneTwister(12), 80, ss)
settings = HDFMStruct(nlevels=2, nfactors=[1,2], factorassign=assignments,
    factorlags=[1,1], errorlags=zeros(Int,6), ndraws=100, burnin=100)
fit = KN2LevelEstimator(MersenneTwister(13), y, settings)
println("Posterior mean intercept, global loading, and group loading:")
display(reshape(vec(fit.means.B), 3, 6)')
pca = PCA2LevelEstimator(y, settings)
println("PCA factor dimensions: ", size(pca.factors))
# For a small dataset the direct Gaussian sampler targets the same model:
# direct_fit = OW2LevelEstimator(MersenneTwister(14), y, settings)
# Increase the number of draws and compare multiple chains for an analysis.
