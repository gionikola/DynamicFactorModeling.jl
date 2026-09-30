# DynamicFactorModeling.jl

Simulate and estimate linear Gaussian dynamic factor models in Julia. A few
unobserved time series (the **factors**) explain shared movements in a panel;
each observed series also has an error that can depend on its past values.

The package provides:

- Single-factor and hierarchical Bayesian estimation, with state-space or
  direct Gaussian factor sampling.
- Single-factor and two-stage hierarchical PCA estimates.
- Model simulation, Kalman filtering and smoothing, and conditional state draws.
- Sample and long-run variance decompositions.

Rows are observations in time order; columns are series. An autoregressive (AR)
order is the number of past values a process uses. Models allow different AR
orders across levels and series, including order zero. The Bayesian estimators
require complete, finite data; the state-space tools also accept `missing` values.

## Installation

Use Julia 1.10 or later. From a local checkout:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using DynamicFactorModeling
```

To install from GitHub, use
`Pkg.add(url="https://github.com/gionikola/DynamicFactorModeling.jl")`.

## A small example

```julia
using DynamicFactorModeling, Random

model = HDFM(
    nlevels=1, nvar=3, nfactors=[1], fassign=ones(Int, 3, 1),
    flags=[1], varlags=[1, 1, 1],
    varcoefs=[0.0 1.0; 1.0 0.8; -1.0 1.2],
    varlagcoefs=fill(0.2, 3, 1),
    fcoefs=[reshape([0.7], 1, 1)], fvars=[[1.0]],
    varvars=fill(0.3, 3),
)
state_space = convertHDFMtoSS(model)
y, _, states = simulateSSModel(MersenneTwister(42), 100, state_space)

settings = DFMStruct(factorlags=1, errorlags=1, ndraws=500, burnin=500)
fit = KN1LevelEstimator(MersenneTwister(43), y, settings)
factor_mean = fit.means.F[:, 1]
coefficient_means = reshape(vec(fit.means.B), 2, 3)'

pca = PCA1LevelEstimator(y)
long_run_shares = variance_decomposition(model)
```

`ndraws` counts retained draws, in addition to `burnin`. The example is a
workflow demonstration, not a convergence guarantee. Run several chains and
inspect their traces and agreement before interpreting posterior summaries.

## Choosing an estimator

| Function | Result | Use |
|---|---|---|
| `KN1LevelEstimator` | Posterior draws | One common factor |
| `KN2LevelEstimator` | Posterior draws | Two factor levels |
| `KNHierarchicalEstimator` | Posterior draws | Any declared number of levels |
| `OW1LevelEstimator`, `OW2LevelEstimator` | Posterior draws | Direct Gaussian factor sampling for small problems |
| `PCA1LevelEstimator` | Deterministic fit | Fast rank-one approximation |
| `PCA2LevelEstimator` | Deterministic fit | Global PCA, then PCA within each residual group |

The Bayesian methods share one model: unit factor innovation variances,
explicit coefficient and variance priors, and stationary initial distributions
for factors and errors. Set `initial=:zero` for fixed zero presample values.
The samplers estimate intercepts on the original data and include joint
location and scale updates to help explore the posterior. PCA is a deterministic
approximation; it does not estimate AR parameters or produce posterior draws.

KN uses state-space calculations; OW uses dense matrices spanning the dates.
KN is generally preferable for longer time series, while direct factor draws
can be useful for many series observed over a short period. Individual factors
may still be weakly identified, especially when their assignments and dynamics
are similar.

Read the [model guide](docs/src/model.md), [state-space guide](docs/src/state_space.md),
and [upgrade notes](docs/src/migration.md). Runnable scripts are in [examples](examples).
The [method audit](docs/src/method_audit.md) maps the algorithms to the inspected
sources and distinguishes them from individual papers' experimental settings.

## Validation and limitations

Tests compare the estimators with independent Gaussian calculations, numerical
posterior integration, and external state-space references. A calibration screen
on 64 small one-factor datasets passed its declared checks. These checks support
the tested models and settings; each fitted dataset still needs sampling diagnostics.

In a small hierarchical example, four chains still disagree about individual
factor contributions after 16,000 retained draws. Combined signals agree more
closely. Both larger panels in the paired pilot pass, but one paired example
does not establish general performance. The [checks guide](docs/src/validation.md)
summarizes the evidence, including the remaining sampling-precision limits.

## Development

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/single_factor.jl
julia --project=. examples/hierarchical.jl
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

[CONTRIBUTING.md](CONTRIBUTING.md) describes the code layout, testing conventions,
and release checks. Normal package tests use the included small reference fixtures.
The [checks guide](docs/src/validation.md) also gives the command for extended
sampler and diagnostic tests. Research-study programs and generated outputs are
not distributed with the package.

This project is [MIT licensed](LICENSE).
