module GaussianLocationReference

using LinearAlgebra

export ar_path_covariance, gaussian_location_posterior,
       conditional_intercepts, conditional_factors, joint_logdensity,
       alternating_gibbs_rate

# This reference deliberately uses no DynamicFactorModeling functions. It builds
# the finite-sample AR covariances first, then an ordinary Gaussian regression.

function finite_matrix(value, name)
    value isa AbstractMatrix{<:Real} || throw(ArgumentError("$name must be a real matrix"))
    result = Matrix{Float64}(value)
    all(isfinite, result) || throw(ArgumentError("$name must be finite in Float64"))
    return result
end

function finite_vector(value, name)
    value isa AbstractVector{<:Real} || throw(ArgumentError("$name must be a real vector"))
    result = Vector{Float64}(value)
    all(isfinite, result) || throw(ArgumentError("$name must be finite in Float64"))
    return result
end

function positive_variances(value, count, name)
    values = value isa Real ? fill(Float64(value), count) : finite_vector(value, name)
    length(values) == count || throw(DimensionMismatch("$name must have $count entries"))
    all(x -> isfinite(x) && x > 0, values) ||
        throw(ArgumentError("$name must be finite and positive"))
    return values
end

"""
    ar_path_covariance(coefficients, dates; innovation_variance=1, initial=:stationary)

Covariance of `[x[1], ..., x[dates]]` for an AR process with zero mean.
`:stationary` solves the Yule–Walker equations for autocovariances, then uses
their recurrence. `:zero` constructs the causal map from independent innovations
with every presample value fixed at zero. AR(0) is supported in both cases.

This dense reference is intended for small validation problems, not estimation.
"""
function ar_path_covariance(coefficients, dates::Integer;
                            innovation_variance=1.0, initial=:stationary)
    dates > 0 || throw(ArgumentError("dates must be positive"))
    initial in (:stationary, :zero) ||
        throw(ArgumentError("initial must be :stationary or :zero"))
    phi = finite_vector(coefficients, "AR coefficients")
    variance = only(positive_variances(innovation_variance, 1, "innovation_variance"))
    order = length(phi)
    order == 0 && return variance .* Matrix{Float64}(I, dates, dates)

    covariance = if initial == :zero
        # Column j is the response to a unit innovation at date j.
        responses = zeros(dates)
        responses[1] = 1.0
        for lag in 1:(dates - 1)
            responses[lag + 1] = sum(phi[j] * responses[lag - j + 1]
                                      for j in 1:min(order, lag))
        end
        innovation_map = [i >= j ? responses[i - j + 1] : 0.0
                          for i in 1:dates, j in 1:dates]
        variance .* (innovation_map * innovation_map')
    else
        companion = zeros(order, order)
        companion[1, :] = phi
        for row in 2:order
            companion[row, row - 1] = 1.0
        end
        maximum(abs, eigvals(companion)) < 1 ||
            throw(ArgumentError("stationary initialization requires a stable AR process"))

        # gamma(h) - sum_j phi[j] gamma(abs(h-j)) = variance * (h == 0).
        equations = Matrix{Float64}(I, order + 1, order + 1)
        for lag in 0:order, j in 1:order
            equations[lag + 1, abs(lag - j) + 1] -= phi[j]
        end
        gamma = zeros(max(dates, order + 1))
        gamma[1:(order + 1)] = equations \ [variance; zeros(order)]
        for lag in (order + 1):(dates - 1)
            gamma[lag + 1] = sum(phi[j] * gamma[lag - j + 1] for j in 1:order)
        end
        [gamma[abs(i - j) + 1] for i in 1:dates, j in 1:dates]
    end
    all(isfinite, covariance) || throw(ArgumentError("AR covariance overflowed; rescale the model"))
    cholesky(Symmetric(covariance))
    return covariance
end

"""
    gaussian_location_posterior(data, loadings, factor_ar, error_ar, error_variances;
                                initial=:stationary, intercept_prior_variance=100.0)

Exact joint Gaussian posterior of intercepts and factor paths, conditional on
known loadings, AR coefficients, and innovation variances. Data are `T × N`,
loadings are `N × K`, and AR inputs contain one coefficient vector per process.
Factor innovation variances are one. Intercepts have independent zero-mean
Gaussian priors with the supplied scalar or length-`N` variance.

The vector ordering is `[intercepts; vec(factors)]`, where factors are `T × K`:
all dates of factor 1, then all dates of factor 2, and so on. Fixed loadings set
factor signs; no sign alignment or anchor transformation is applied.

Returns the joint `mean`, `covariance`, `precision`, and `information` vector;
the finite-path prior/error covariances; and reusable conditional blocks.
Only complete finite data and nonsingular innovation variances are supported.
"""
function gaussian_location_posterior(data, loadings, factor_ar, error_ar,
                                     error_variances; initial=:stationary,
                                     intercept_prior_variance=100.0)
    y = finite_matrix(data, "data")
    lambda = finite_matrix(loadings, "loadings")
    nobs, nseries = size(y)
    nseries == size(lambda, 1) || throw(DimensionMismatch("loadings must have one row per series"))
    nfactors = size(lambda, 2)
    nobs > 0 && nseries > 0 && nfactors > 0 ||
        throw(ArgumentError("data and loadings must have positive dimensions"))
    length(factor_ar) == nfactors || throw(DimensionMismatch("one AR vector is required per factor"))
    length(error_ar) == nseries || throw(DimensionMismatch("one AR vector is required per series"))
    variances = positive_variances(error_variances, nseries, "error_variances")
    prior_variances = positive_variances(intercept_prior_variance, nseries,
                                         "intercept_prior_variance")
    factor_covariances = [ar_path_covariance(phi, nobs; initial) for phi in factor_ar]
    error_covariances = [ar_path_covariance(error_ar[i], nobs;
                            innovation_variance=variances[i], initial) for i in 1:nseries]
    identity = Matrix{Float64}(I, nobs, nobs)
    factor_precisions = [Matrix(cholesky(Symmetric(covariance)) \ identity)
                         for covariance in factor_covariances]
    error_precisions = [Matrix(cholesky(Symmetric(covariance)) \ identity)
                        for covariance in error_covariances]
    intercept_indices = 1:nseries
    factor_indices = (nseries + 1):(nseries + nobs * nfactors)
    factor_ranges = [(nseries + (k - 1) * nobs + 1):(nseries + k * nobs)
                     for k in 1:nfactors]
    dimension = nseries + nobs * nfactors
    precision = zeros(dimension, dimension)
    information = zeros(dimension)
    precision[intercept_indices, intercept_indices] = Diagonal(1 ./ prior_variances)
    for k in 1:nfactors
        precision[factor_ranges[k], factor_ranges[k]] = factor_precisions[k]
    end

    # y_i = 1*a_i + sum_k lambda[i,k]*f_k + error_i.
    # Accumulate each series' X' W_i X and X' W_i y without a large design matrix.
    ones_path = ones(nobs)
    for i in 1:nseries
        weight = error_precisions[i]
        weighted_ones, weighted_data = weight * ones_path, weight * y[:, i]
        precision[i, i] += sum(weighted_ones)
        information[i] += sum(weighted_data)
        for k in 1:nfactors
            indices = factor_ranges[k]
            cross = lambda[i, k] .* weighted_ones
            precision[indices, i] += cross
            precision[i, indices] += cross
            information[indices] += lambda[i, k] .* weighted_data
            for l in 1:nfactors
                precision[indices, factor_ranges[l]] += lambda[i, k] * lambda[i, l] .* weight
            end
        end
    end
    precision = Matrix(Symmetric(precision))
    decomposition = cholesky(Symmetric(precision))
    posterior_mean = decomposition \ information
    covariance = Matrix(Symmetric(decomposition \ Matrix{Float64}(I, dimension, dimension)))
    intercept_precision = precision[intercept_indices, intercept_indices]
    factor_precision = precision[factor_indices, factor_indices]
    cross_precision = precision[intercept_indices, factor_indices]
    intercept_decomposition = cholesky(Symmetric(intercept_precision))
    factor_decomposition = cholesky(Symmetric(factor_precision))
    intercept_gain = -(intercept_decomposition \ cross_precision)
    factor_gain = -(factor_decomposition \ cross_precision')
    intercept_conditional_covariance = Matrix(intercept_decomposition \ Matrix{Float64}(I, nseries, nseries))
    factor_conditional_covariance = Matrix(factor_decomposition \ Matrix{Float64}(I, nobs * nfactors, nobs * nfactors))
    return (; mean=posterior_mean, covariance, precision, information,
            nobs, nseries, nfactors, intercept_indices, factor_indices, factor_ranges,
            data=y, loadings=lambda, initial, intercept_prior_variances=prior_variances,
            factor_prior_covariances=factor_covariances,
            factor_prior_precisions=factor_precisions, error_covariances, error_precisions,
            intercept_precision, factor_precision, cross_precision,
            intercept_gain, factor_gain, intercept_conditional_covariance,
            factor_conditional_covariance)
end

"""Return the Gaussian intercept conditional given a `T × K` factor matrix."""
function conditional_intercepts(reference, factors)
    f = finite_matrix(factors, "factors")
    size(f) == (reference.nobs, reference.nfactors) ||
        throw(DimensionMismatch("factors must have size (dates, factors)"))
    a, paths = reference.intercept_indices, reference.factor_indices
    mean = reference.mean[a] + reference.intercept_gain * (vec(f) - reference.mean[paths])
    return (; mean, covariance=reference.intercept_conditional_covariance,
            gain=reference.intercept_gain)
end

"""
Return the Gaussian factor conditional given the intercept vector. `mean` and
`covariance` use the factor-major vector ordering; `factors` reshapes the mean.
"""
function conditional_factors(reference, intercepts)
    values = finite_vector(intercepts, "intercepts")
    length(values) == reference.nseries || throw(DimensionMismatch("one intercept is required per series"))
    a, paths = reference.intercept_indices, reference.factor_indices
    mean = reference.mean[paths] + reference.factor_gain * (values - reference.mean[a])
    return (; mean, factors=reshape(mean, reference.nobs, reference.nfactors),
            covariance=reference.factor_conditional_covariance, gain=reference.factor_gain)
end

"""
Unnormalized log posterior evaluated as separate intercept prior, factor prior,
and observation densities. Constants independent of intercepts/factors are omitted.
"""
function joint_logdensity(reference, intercepts, factors)
    values = finite_vector(intercepts, "intercepts")
    paths = finite_matrix(factors, "factors")
    length(values) == reference.nseries || throw(DimensionMismatch("one intercept is required per series"))
    size(paths) == (reference.nobs, reference.nfactors) ||
        throw(DimensionMismatch("factors must have size (dates, factors)"))
    logdensity = -0.5 * sum(abs2.(values) ./ reference.intercept_prior_variances)
    for k in 1:reference.nfactors
        logdensity -= 0.5 * dot(paths[:, k], reference.factor_prior_precisions[k] * paths[:, k])
    end
    residual = reference.data .- values' .- paths * reference.loadings'
    for i in 1:reference.nseries
        logdensity -= 0.5 * dot(residual[:, i], reference.error_precisions[i] * residual[:, i])
    end
    return logdensity
end

"""
    alternating_gibbs_rate(reference)

Exact slowest linear mean-contraction factor for alternating **joint** draws
`a | F` then `F | a`, with loadings, AR parameters, and variances held fixed.
It is the square of the largest canonical correlation between these two blocks.
The returned `rate` is per full sweep; near one means slow forgetting of starts.
`slow_mode_iact = (1+rate)/(1-rate)` is the integrated autocorrelation time of
the corresponding stationary Gaussian eigenmode, not of every model parameter.
`intercept_mode_weights` and `factor_mode_weights` project centered draws onto
these scalar modes. The `*_mean_direction` fields instead describe the right
eigenvectors along which conditional mean errors contract.

If the precision blocks are `A, C, B`, the factor mean transition is
`B⁻¹ C' A⁻¹ C`. Its nonzero eigenvalues equal the squared singular values of
`L_A⁻¹ C L_B⁻ᵀ`, using Cholesky factors. This also supplies the block canonical
correlations without forming a possibly asymmetric transition eigendecomposition.
See Roberts and Sahu (1997), https://doi.org/10.1111/1467-9868.00070, for Gaussian
Gibbs transition analysis. The formula here follows directly from the two
Gaussian conditional means. It does not describe the full unknown-parameter
sampler or a sequential update of individual factors.
"""
function alternating_gibbs_rate(reference)
    a_root = cholesky(Symmetric(reference.intercept_precision)).L
    f_root = cholesky(Symmetric(reference.factor_precision)).L
    coupling = (a_root \ reference.cross_precision) / f_root'
    decomposition = svd(coupling; full=false)
    correlations = decomposition.S
    rate = maximum(abs2, correlations)
    0 <= rate < 1 || throw(ArgumentError("Gaussian Gibbs rate is outside [0,1); rescale the reference"))
    intercept_direction = a_root' \ decomposition.U[:, 1]
    factor_direction = f_root' \ decomposition.V[:, 1]
    intercept_weights = a_root * decomposition.U[:, 1]
    factor_weights = f_root * decomposition.V[:, 1]
    intercept_direction ./= norm(intercept_direction)
    factor_direction ./= norm(factor_direction)
    intercept_weights ./= norm(intercept_weights)
    factor_weights ./= norm(factor_weights)
    return (; rate, canonical_correlations=correlations,
            slow_mode_iact=(1 + rate) / (1 - rate),
            intercept_mean_direction=intercept_direction,
            factor_mean_direction=reshape(factor_direction, reference.nobs, reference.nfactors),
            intercept_mode_weights=intercept_weights,
            factor_mode_weights=reshape(factor_weights, reference.nobs, reference.nfactors))
end

end
