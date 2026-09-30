"""
    KNHierarchicalEstimator(data, specification::HDFMStruct; kwargs...)
    KNHierarchicalEstimator(rng, data, specification::HDFMStruct; kwargs...)

Sample factors and parameters in a model with at most one assigned factor per
level for each observed series. Any positive number of levels is supported.
Factors are ordered by level, then by their number within that level. Factor
lag orders can differ across levels, and error lag orders across series.
An assignment of zero excludes that level's factor from the series.

All innovations are independent Gaussian variables. Each factor's innovation
variance is fixed at one. Initial factors and errors follow their stationary
distributions by default, and every supplied observation enters the likelihood.
Data must be finite and
complete; intercepts are estimated without automatically centering the data.

Keyword arguments:
- `beta_prior_variance=100.0`: independent zero-mean Normal prior variance for
  intercepts and loadings. A vector `[intercept, level1, ...]` sets separate
  prior variances for each coefficient type.
- `ar_prior_variance=1.0`: independent zero-mean Normal prior variance for AR
  coefficients, restricted to stationary vectors if `stationary=true`. A vector
  with one entry per lag, up to the largest model order, sets lag-specific
  variances; shorter processes use its leading entries.
- `variance_shape=2.0`, `variance_scale=1.0`: inverse-gamma prior for each error
  innovation variance, with density proportional to `v^(-shape-1) * exp(-scale/v)`.
- `stationary=true`: restrict factor and error AR coefficients to stationarity.
- `initial=:stationary`: include the stationary initial-observation likelihood
  and use Metropolis-Hastings AR updates. `:zero` instead fixes all presample
  factors and errors to zero and uses direct Gaussian AR updates.
- `max_attempts=10000`: rejection limit for stationary AR draws under `initial=:zero`.
  Stationary-initial-likelihood AR updates make one proposal per sweep and
  correctly retain the previous value when it is rejected.
- `sign_anchors=nothing`: one series index per factor; each must load on that
  factor. The default is the first assigned series. Its loading is kept positive
  by flipping the factor and all its loadings together.
- `factor_sampler=:state_space`: use forward filtering and backward sampling;
  `:precision` instead draws from the full Gaussian precision matrix. Both
  target the same posterior. `:sequential_precision` draws each factor path
  given the other current paths, as in the multi-factor OW extension.
  Dense precision options suit shorter time series; the state-space dimension
  instead grows with the number of series and their AR orders.
- `rng=Random.default_rng()`: random generator, also accepted as the first argument.
- `initial_factors=nothing`: optional finite `dates × total_factors` matrix for
  starting the chain. The default uses a sequential PCA initialization. Supply
  different matrices to check sensitivity to starting values across chains.
  This changes only the starting point; factor paths are still sampled normally.

`ndraws` retained iterations follow `burnin` discarded iterations. Returns
[`DFMResults`](@ref), with factors of size `(dates, sum(nfactors), ndraws)`.
Sign and scale conventions do not make factors with indistinguishable loading
patterns identifiable. Assess mixing and identification for the supplied model;
a finite chain does not by itself establish convergence.
"""
function KNHierarchicalEstimator(rng::AbstractRNG, data::AbstractMatrix,
                                specification::HDFMStruct; kwargs...)
    specification.nlevels == length(specification.nfactors) ||
        throw(DimensionMismatch("nlevels must equal the length of nfactors"))
    return _estimate_dynamic_factors(
        rng, data, specification.nfactors, specification.factorassign,
        specification.factorlags, specification.errorlags,
        specification.ndraws, specification.burnin; kwargs...)
end

KNHierarchicalEstimator(data::AbstractMatrix, specification::HDFMStruct;
                        rng::AbstractRNG=Random.default_rng(), kwargs...) =
    KNHierarchicalEstimator(rng, data, specification; kwargs...)

"""
    KN2LevelEstimator(data, specification::HDFMStruct; kwargs...)
    KN2LevelEstimator(rng, data, specification::HDFMStruct; kwargs...)

Two-level version of [`KNHierarchicalEstimator`](@ref). The specification must
have exactly two levels. Each series loads on at most one factor at each level.
"""
function KN2LevelEstimator(rng::AbstractRNG, data::AbstractMatrix,
                           specification::HDFMStruct; kwargs...)
    specification.nlevels == 2 || throw(ArgumentError("KN2LevelEstimator requires two levels"))
    return KNHierarchicalEstimator(rng, data, specification; kwargs...)
end

KN2LevelEstimator(data::AbstractMatrix, specification::HDFMStruct;
                  rng::AbstractRNG=Random.default_rng(), kwargs...) =
    KN2LevelEstimator(rng, data, specification; kwargs...)

"""
    OW2LevelEstimator(data, specification::HDFMStruct; kwargs...)
    OW2LevelEstimator(rng, data, specification::HDFMStruct; kwargs...)

Two-level model using sequential Gaussian precision draws: each factor path
is drawn given the other current factors. This targets the same posterior as
[`KN2LevelEstimator`](@ref), using the stationary initial likelihood by default.
Each dense factor matrix has `dates` rows and columns. Set
`factor_sampler=:precision` to draw all factors jointly instead (a dense matrix
with `dates * sum(nfactors)` rows). Priors follow the documented package settings.
"""
function OW2LevelEstimator(rng::AbstractRNG, data::AbstractMatrix,
                           specification::HDFMStruct; kwargs...)
    return KN2LevelEstimator(rng, data, specification; factor_sampler=:sequential_precision, kwargs...)
end

OW2LevelEstimator(data::AbstractMatrix, specification::HDFMStruct;
                  rng::AbstractRNG=Random.default_rng(), kwargs...) =
    OW2LevelEstimator(rng, data, specification; kwargs...)
