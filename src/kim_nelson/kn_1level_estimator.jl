"""
    KN1LevelEstimator(data, specification::DFMStruct; kwargs...)
    KN1LevelEstimator(rng, data, specification::DFMStruct; kwargs...)

Sample the posterior of a single-factor model with Gaussian innovations.
Rows of `data` are dates and columns are series. Intercepts are estimated on
the supplied data; the estimator does not center or standardize it.

Factor innovations have variance one. By default the initial factors and errors
follow their stationary distributions; `initial=:zero` instead fixes presample
values at zero. By default, the first series' loading is made positive
by changing its factor's sign and all corresponding loadings together.
Use `sign_anchors` to choose a different anchoring series.

See [`KNHierarchicalEstimator`](@ref) for priors and keyword arguments.
`results.F` has size `(dates, ndraws)`; the other draws follow [`DFMResults`](@ref).
"""
function KN1LevelEstimator(rng::AbstractRNG, data::AbstractMatrix,
                           specification::DFMStruct; kwargs...)
    return _estimate_single_factor(rng, data, specification; kwargs...)
end

KN1LevelEstimator(data::AbstractMatrix, specification::DFMStruct;
                  rng::AbstractRNG=Random.default_rng(), kwargs...) =
    KN1LevelEstimator(rng, data, specification; kwargs...)

"""
    OW1LevelEstimator(data, specification::DFMStruct; kwargs...)
    OW1LevelEstimator(rng, data, specification::DFMStruct; kwargs...)

Fit the same model as [`KN1LevelEstimator`](@ref), drawing the whole factor
path from its Gaussian precision matrix instead of a state-space sampler.
With the default `initial=:stationary`, parameter updates include the initial
stationary density and its Metropolis-Hastings correction as in Otrok–Whiteman.
The default proper priors are documented in [`KNHierarchicalEstimator`](@ref).
It forms a dense `dates × dates`
matrix, so the state-space method is preferable for long series.
"""
function OW1LevelEstimator(rng::AbstractRNG, data::AbstractMatrix,
                           specification::DFMStruct; kwargs...)
    return _estimate_single_factor(rng, data, specification;
                                   factor_sampler=:precision, kwargs...)
end

OW1LevelEstimator(data::AbstractMatrix, specification::DFMStruct;
                  rng::AbstractRNG=Random.default_rng(), kwargs...) =
    OW1LevelEstimator(rng, data, specification; kwargs...)

function _estimate_single_factor(rng, data, specification; kwargs...)
    specification.factorlags >= 0 || throw(ArgumentError("factorlags must be nonnegative"))
    specification.errorlags >= 0 || throw(ArgumentError("errorlags must be nonnegative"))
    nseries = size(data, 2)
    result = _estimate_dynamic_factors(
        rng, data, [1], ones(Int, nseries, 1), [specification.factorlags],
        fill(specification.errorlags, nseries), specification.ndraws,
        specification.burnin; kwargs...)

    # Keep the original single-factor shape, without a singleton factor axis.
    factors = dropdims(result.F; dims=2)
    return DFMResults(factors, result.B, result.S, result.P, result.P2, result.means)
end
