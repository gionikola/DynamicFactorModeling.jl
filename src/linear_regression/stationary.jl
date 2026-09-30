# Covariance of consecutive observations from an AR process with unit innovation
# variance. Stationarity makes the covariance depend only on the time separation.
function _stationary_ar_covariance(coefficients::AbstractVector{<:Real}, n::Integer)
    n >= 0 || throw(ArgumentError("covariance size must be nonnegative"))
    isstationary(coefficients) || throw(ArgumentError("stationary initialization requires stable AR coefficients"))
    p = length(coefficients)
    p == 0 && return Matrix{Float64}(I, n, n)
    transition = zeros(p, p)
    transition[1,:] = coefficients
    for lag in 2:p
        transition[lag,lag-1] = 1
    end
    innovation = zeros(p,p)
    innovation[1,1] = 1
    covariance = _stationary_covariance(transition, innovation)
    autocovariances = zeros(n)
    cross_covariance = covariance[:,1]
    for lag in 0:(n-1)
        autocovariances[lag+1] = cross_covariance[1]
        cross_covariance = transition * cross_covariance
    end
    return [autocovariances[abs(i-j)+1] for i in 1:n, j in 1:n]
end

# Whiten the first min(p,T) observations with their stationary covariance.
# Later rows use ordinary AR innovations. In particular, the early rows are
# not replaced by zero-padded innovations when the likelihood is stationary.
function _stationary_ar_whiten(values, coefficients)
    n = size(values,1)
    p = length(coefficients)
    result = Float64.(values)
    for lag in 1:min(p, n-1)
        result[(lag+1):end,:] -= coefficients[lag] * values[1:(end-lag),:]
    end
    initial_count = min(p,n)
    if initial_count > 0
        covariance = _stationary_ar_covariance(coefficients, initial_count)
        root = cholesky(Symmetric(covariance)).L
        result[1:initial_count,:] = root \ values[1:initial_count,:]
    end
    return result
end

# Log density of the initial observations, omitting constants independent of
# the AR coefficients. The determinant factor is essential to the MH ratio.
function _stationary_initial_logdensity(series, coefficients, variance)
    initial_count = min(length(coefficients), length(series))
    initial_count == 0 && return 0.0
    covariance = _stationary_ar_covariance(coefficients, initial_count)
    root = cholesky(Symmetric(covariance)).L
    standardized = root \ series[1:initial_count]
    return -sum(log, diag(root)) - sum(abs2, standardized) / (2variance)
end

# Independence Metropolis-Hastings step from the Gaussian transition-regression
# posterior. That proposal cancels the transition likelihood and normal prior;
# the acceptance ratio is just the initial stationary density ratio.
function _draw_stationary_ar(rng, series, old_coefficients, variance;
                             prior_variance=1.0)
    p = length(old_coefficients)
    prior_variances = _regression_prior_variances(prior_variance,p,"ar_prior_variance")
    p == 0 && return Float64[]
    isstationary(old_coefficients) ||
        throw(ArgumentError("current AR coefficients must be stationary"))
    if length(series) > p
        rows = (p+1):length(series)
        regressors = _regression_lags(series, p)[rows,:]
        candidate = draw_coefficients(rng, series[rows], regressors, variance;
                                      prior_variance)
    else
        # No transition likelihood remains when the entire sample is initial.
        candidate = sqrt.(prior_variances) .* randn(rng,p)
    end
    # One rejected proposal retains the old value. Drawing repeatedly until a
    # proposal is accepted would change the transition kernel and its target.
    isstationary(candidate) || return copy(old_coefficients)
    log_ratio = _stationary_initial_logdensity(series, candidate, variance) -
                _stationary_initial_logdensity(series, old_coefficients, variance)
    return log(rand(rng)) < min(0.0, log_ratio) ? candidate : copy(old_coefficients)
end

function _draw_stationary_regression(rng, y, x, phi, variance;
                                     beta_prior_variance, ar_prior_variance,
                                     variance_shape, variance_scale)
    transformed_y = vec(_stationary_ar_whiten(reshape(y,:,1), phi))
    transformed_x = _stationary_ar_whiten(x, phi)
    beta = draw_coefficients(rng, transformed_y, transformed_x, variance;
                             prior_variance=beta_prior_variance)
    residuals = y - x * beta
    new_phi = _draw_stationary_ar(rng, residuals, phi, variance;
                                  prior_variance=ar_prior_variance)
    # Use the NEW AR coefficients in the full-sample error sum of squares.
    innovations = vec(_stationary_ar_whiten(reshape(residuals,:,1), new_phi))
    new_variance = draw_error_variance(rng, innovations, zeros(length(y),0), Float64[];
                                       prior_shape=variance_shape, prior_scale=variance_scale)
    return beta, new_phi, new_variance
end
