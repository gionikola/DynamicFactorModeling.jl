# Upgrading from the unfinished 0.1 code

Version 0.2 replaces incorrect numerical routines and makes the model assumptions
explicit. Results from the old implementation should be recomputed.

- Julia 1.10 or newer is required. Runtime dependencies are reduced to
  Distributions and Julia's LinearAlgebra, Random, and Statistics libraries.
- `SSModel`, `HDFMStruct`, and result types are exported. Model fields use concrete
  numeric arrays, and constructors check dimensions and values. Struct fields
  cannot be reassigned; construct a new specification when changing settings.
- All random APIs accept an RNG as their first argument or as `rng=`. A fixed
  seed is reproducible within the same Julia and dependency environment; no
  bit-for-bit guarantee is made across versions.
- State-space initialization describes time zero. Simulation includes fresh
  innovations in the first returned row and defaults to a stationary draw.
- HDFM states use the compact block ordering described in the state-space guide.
  Do not reuse hard-coded column indices from old experiments. Converted models
  use zero-column exogenous matrices.
- Bayesian estimators default to the full stationary initial likelihood,
  including the required AR Metropolis-Hastings correction. `initial=:zero`
  selects fixed-zero presample factors and errors. Intercepts are estimated on
  the original data; inputs are no longer silently demeaned.
- Every coefficient/variance update uses that series' own previous state.
  AR transformations use all lags, and variance draws use the newly drawn AR
  coefficients. Metropolis rejection correctly retains the previous value;
  bounded rejection sampling under `initial=:zero` raises an error on exhaustion.
- `ndraws` means retained draws, plus `burnin` discarded iterations. `regress`
  keeps its separate `iter` convention: total iterations including burn-in.
- Bayesian samplers now add joint location and scale updates by default
  (`mixing_moves=:location_scale`). They preserve the posterior and change seeded
  draws. Use `:location` for the location update alone or `:none` to reproduce
  the corrected sampler before these additions, in the same software environment.
  This option controls only the extra moves in the 0.2 implementation.
  `scale_step=0.1` controls the fixed log-scale proposal size.
- `means.F` is always time × factor. Hierarchical AR draws are packed process
  by process without padding. See the result-layout table in the model guide.
- PCA returns `PCAResults`, with factors, loadings, intercepts, and residuals.
  These estimates are deterministic; PCA does not estimate AR parameters or
  produce posterior draws.
- `vardecomp2level` fixes the old residual-sign error but retains the explicitly
  documented sum-of-marginal-variances normalization. `variance_decomposition`
  provides sample covariance accounting and exact stationary model shares.
- Old archived implementations and plotting experiments are removed from the
  working tree. Git history retains them; maintained examples and automated
  tests replace them.

The KN/OW estimator names describe their factor samplers under the common
likelihood in the model guide. They do not imply the same initialization or
priors as every published version of those algorithms.
