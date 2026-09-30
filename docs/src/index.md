# Getting started

DynamicFactorModeling fits models in which a few shared, unobserved time series
explain movements in many observed series. Those shared series are called
**factors**. Each observed series has an intercept, factor loadings (multipliers),
and its own error process.

Use Julia 1.10 or later. In a local checkout, run `using Pkg`, then
`Pkg.activate(".")` and `Pkg.instantiate()`. Load the package with
`using DynamicFactorModeling`.

## Simulate and fit one factor

```@example start
using DynamicFactorModeling, Random
model = HDFM(
    nlevels=1, nvar=3, nfactors=[1], fassign=ones(Int, 3, 1),
    flags=[1], varlags=[1, 1, 1],
    varcoefs=[0.0 1.0; 1.0 0.8; -1.0 1.2],
    varlagcoefs=fill(0.2, 3, 1),
    fcoefs=[reshape([0.7], 1, 1)], fvars=[[1.0]],
    varvars=fill(0.3, 3),
)
ss = convertHDFMtoSS(model)
y, _, states = simulateSSModel(MersenneTwister(42), 60, ss)
settings = DFMStruct(factorlags=1, errorlags=1, ndraws=20, burnin=20)
fit = KN1LevelEstimator(MersenneTwister(43), y, settings)
(size(y), size(fit.F), size(fit.means.F))
```

This short run demonstrates the API. It is too short to establish convergence.
For an analysis, run several longer chains and compare their factor and parameter
traces. `ndraws` is the number kept; `burnin` adds discarded iterations.

```@example start
# Rows are series; columns are intercept and loading.
coefficient_means = reshape(vec(fit.means.B), 2, 3)'
```

A factor's contribution is its loading times its value within the same draw.
The fitted signal is the intercept plus all factor contributions, excluding the
error. To estimate its posterior mean, form the signal within each draw, then
average. Factors and loadings are dependent, so multiplying their separate
posterior means generally gives a different answer. For this single-factor fit:

```@example start
signal_mean = zeros(size(y))
for draw in axes(fit.B, 1)
    coefficients = reshape(fit.B[draw, :], 2, size(y, 2))'
    signal_mean .+= coefficients[:, 1]' .+
                   fit.F[:, draw] * coefficients[:, 2]'
end
signal_mean ./= size(fit.B, 1)
size(signal_mean)
```

## A quick PCA fit

```@example start
pca = PCA1LevelEstimator(y)
maximum(abs, y - (pca.intercepts' .+ pca.factors * pca.loadings' + pca.residuals))
```

PCA is deterministic and estimates no AR parameters. Columns are centered but
not rescaled; standardize them beforehand if that is appropriate for your data.
PCA and Bayesian factors use different sign and scale conventions. Factor
contributions and reconstructed signals are useful comparisons across methods.

## Variance shares

```@example start
# Exact long-run shares implied by the simulation model.
variance_decomposition(model)
```

For fitted data, use
`variance_decomposition(y, pca.factors, pca.loadings; intercepts=pca.intercepts)`.
This reports factor shares, a residual share, and the combined contribution of
cross-covariances. Those contributions sum to one. Individual shares need not
lie between zero and one when components are correlated.

The [model guide](@ref "Model and estimation") explains the likelihood and
result layout. The [API reference](@ref "API reference") lists all public functions.
See [Checks and limitations](@ref) for independent test evidence and a hierarchical
example whose individual components still disagree across chains.
