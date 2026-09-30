module SBCSampler

# Small adapter for six-date, two-series calibration checks.
# Including this file neither fits a model nor generates observations.
using Random: AbstractRNG
using DynamicFactorModeling: DFMStruct, DFMResults, KN1LevelEstimator

include(joinpath(@__DIR__, "..", "reference", "sbc_reference.jl"))

export sample_chain, target_names, target_values, target_trajectories

function checked_data(data)
    data isa AbstractMatrix{<:Real} || throw(ArgumentError("data must be a real matrix"))
    size(data) == (6, 2) || throw(DimensionMismatch("the proposed screen requires six dates and two series"))
    values = Matrix{Float64}(data)
    all(isfinite, values) || throw(ArgumentError("data must be finite in Float64"))
    return values
end

"""Names of the nine primary targets, in their fixed column order."""
function target_names(order)
    order = SBCReference.ar_order(order)
    return (:intercept_1, :anchor_loading, :loading_2, :error_variance_1,
        order == 1 ? :factor_ar : :final_factor,
        order == 1 ? :error_ar_2 : :final_contribution_2,
        :first_factor, :marginal_loglikelihood, :conditional_loglikelihood)
end

"""
    sample_chain(rng, data; order, ndraws, burnin, initial_factors)

Call the public KN1 estimator with the default-prior model. The caller
supplies the RNG, AR order (0 or 1), lengths, and starting path. Pass a six-by-one
matrix for the path, or explicitly pass `nothing` for the public PCA start.
Never pass the generating factor truth as the starting path.

Return only the public `DFMResults`. Target extraction is a separate call so its
independent likelihood calculations can be timed separately from fitting. This
adapter chooses no seeds, run lengths, ranks, or stopping rules.
"""
function sample_chain(rng::AbstractRNG, data; order, ndraws, burnin, initial_factors)
    order = SBCReference.ar_order(order)
    values = checked_data(data)
    ndraws isa Integer && ndraws > 0 || throw(ArgumentError("ndraws must be a positive integer"))
    burnin isa Integer && burnin >= 0 || throw(ArgumentError("burnin must be a nonnegative integer"))
    settings = DFMStruct(; factorlags=order, errorlags=order, ndraws, burnin)
    return KN1LevelEstimator(rng, values, settings;
        beta_prior_variance=100.0, ar_prior_variance=1.0,
        variance_shape=2.0, variance_scale=1.0,
        initial=:stationary, stationary=true, sign_anchors=[1],
        factor_sampler=:state_space, mixing_moves=:location_scale, scale_step=0.1,
        max_attempts=10_000, initial_factors)
end

"""
    target_values(data, parameters, factors)

Evaluate the same nine quantities for a posterior draw, generating truth, or
prior-only control. `parameters` has the fields used by `SBCReference`, and
`factors` is the six-element path. Return `(names, values)`. Inputs are neither
aligned to truth nor sign-folded here; the generator and public estimator apply
their fixed common sign convention before this evaluation. A negative anchor
loading is rejected; an exactly zero anchor is permitted.

The eighth quantity integrates out the factor path while holding intercepts
and other parameters fixed. The ninth is the observation likelihood conditional
on `factors`. Both use the independent reference calculations, including the
stationary initial density.
"""
function target_values(data, parameters, factors)
    values = checked_data(data)
    parameters = SBCReference.checked_parameters(parameters)
    length(parameters.intercepts) == 2 || throw(DimensionMismatch("two series-specific parameter entries are required"))
    parameters.loadings[1] < 0 && throw(ArgumentError(
        "anchor loading must be nonnegative; jointly sign-fold the factor path and all loadings before target evaluation"))
    factors = SBCReference.finite_vector(factors, "factor path")
    length(factors) == 6 || throw(DimensionMismatch("the factor path must have six dates"))
    order = parameters.order
    targets = [parameters.intercepts[1], parameters.loadings[1],
        parameters.loadings[2], parameters.error_variances[1],
        order == 1 ? parameters.factor_ar : factors[end],
        order == 1 ? parameters.error_ar[2] : parameters.loadings[2] * factors[end],
        factors[1], SBCReference.marginal_loglikelihood(values, parameters),
        SBCReference.conditional_loglikelihood(values, parameters, factors)]
    all(isfinite, targets) || throw(ArgumentError("a target is not representable in Float64"))
    return (; names=target_names(order), values=targets)
end

"""
    target_trajectories(data, result::DFMResults; order)

Extract all retained draws in an `ndraws × 9` matrix, with columns named by
`names`. Evaluate both independent likelihood quantities at every retained draw.
`endpoint` is a copy of the final row for independent-chain ranks;
all rows remain available for diagnostics. No posterior means or thinning enter
the extraction, and the public result is unchanged.
"""
function target_trajectories(data, result::DFMResults; order)
    order = SBCReference.ar_order(order)
    values = checked_data(data)
    ndraws = size(result.B, 1)
    ndraws > 0 || throw(ArgumentError("at least one retained draw is required"))
    expected_sizes = ((:B, (ndraws, 4)), (:S, (ndraws, 2)),
        (:P, (ndraws, order)), (:P2, (ndraws, 2 * order)), (:F, (6, ndraws)))
    for (name, expected) in expected_sizes
        size(getproperty(result, name)) == expected ||
            throw(DimensionMismatch("$name must have size $expected for the proposed screen"))
    end

    names = target_names(order)
    draws = Matrix{Float64}(undef, ndraws, length(names))
    for draw in 1:ndraws
        # Public B columns are a1, b1, a2, b2. Error AR columns follow series.
        parameters = (; order,
            intercepts=result.B[draw, [1, 3]], loadings=result.B[draw, [2, 4]],
            error_variances=result.S[draw, :],
            factor_ar=order == 1 ? result.P[draw, 1] : 0.0,
            error_ar=order == 1 ? result.P2[draw, :] : zeros(2))
        targets = target_values(values, parameters, result.F[:, draw])
        draws[draw, :] = targets.values
    end
    return (; names, draws, endpoint=copy(draws[end, :]))
end

end # module
