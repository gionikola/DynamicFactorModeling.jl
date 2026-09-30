# Shared sampler. Every update uses the same chosen initial-condition likelihood.
function _estimate_dynamic_factors(
    rng, data, factor_counts, assignments, level_orders, error_orders, ndraws, burnin;
    beta_prior_variance=100.0, ar_prior_variance=1.0,
    variance_shape::Real=2.0, variance_scale::Real=1.0,
    stationary::Bool=true, max_attempts::Integer=10000,
    sign_anchors=nothing, factor_sampler::Symbol=:state_space,
    initial::Symbol=:stationary, initial_factors=nothing)

    _validate_estimator_inputs(data, factor_counts, assignments, level_orders,
                               error_orders, ndraws, burnin)
    for (name, value) in (("variance_shape", variance_shape),
                          ("variance_scale", variance_scale))
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))
    factor_sampler in (:state_space, :precision, :sequential_precision) ||
        throw(ArgumentError("factor_sampler must be :state_space, :precision, or :sequential_precision"))
    initial in (:stationary, :zero) ||
        throw(ArgumentError("initial must be :stationary or :zero"))
    initial == :stationary && !stationary &&
        throw(ArgumentError("stationary initialization requires stationary=true"))

    y = Matrix{Float64}(data)
    all(isfinite, y) || throw(ArgumentError("data must be finite in Float64"))
    nobs, nseries = size(y)
    nlevels = length(factor_counts)
    nfactors = sum(factor_counts)
    nregressors = nlevels + 1
    beta_variances = _regression_prior_variances(beta_prior_variance,nregressors,
                                                "beta_prior_variance")
    max_ar_order = max(maximum(level_orders),maximum(error_orders))
    ar_variances = _regression_prior_variances(ar_prior_variance,max_ar_order,
                                              "ar_prior_variance")
    level_offsets = cumsum([0; factor_counts[1:end-1]])
    factor_indices = zeros(Int, nseries, nlevels)
    for series in 1:nseries, level in 1:nlevels
        if assignments[series, level] != 0
            factor_indices[series, level] = assignments[series, level] + level_offsets[level]
        end
    end
    factor_orders = [level_orders[level] for level in 1:nlevels
                     for _ in 1:factor_counts[level]]
    anchors = _factor_sign_anchors(factor_indices, nfactors, sign_anchors)

    if initial_factors === nothing
        factors = _initialize_factors(y, factor_indices, nfactors)
    else
        initial_factors isa AbstractMatrix{<:Real} ||
            throw(ArgumentError("initial_factors must be a real matrix with dates in rows"))
        size(initial_factors) == (nobs,nfactors) ||
            throw(DimensionMismatch("initial_factors must have size (dates, total factors)"))
        factors = Matrix{Float64}(initial_factors)
        all(isfinite,factors) || throw(ArgumentError("initial_factors must be finite in Float64"))
    end
    coefficients = zeros(nseries, nregressors)
    error_variances = ones(nseries)
    factor_ar = [zeros(order) for order in factor_orders]
    error_ar = [zeros(order) for order in error_orders]

    factor_draws = zeros(nobs, nfactors, ndraws)
    coefficient_draws = zeros(ndraws, nseries * nregressors)
    variance_draws = zeros(ndraws, nseries)
    factor_ar_draws = zeros(ndraws, sum(factor_orders))
    error_ar_draws = zeros(ndraws, sum(error_orders))

    for iteration in 1:(burnin + ndraws)
        # Given factors, every observed series is a regression with AR errors.
        for series in 1:nseries
            levels = findall(!iszero, factor_indices[series, :])
            coefficient_columns = [1; levels .+ 1]
            regressors = hcat(ones(nobs), factors[:, factor_indices[series, levels]])
            beta, phi, variance = draw_parameters(
                rng, y[:, series], regressors, error_ar[series], error_variances[series];
                initial, beta_prior_variance=beta_variances[coefficient_columns],
                ar_prior_variance=ar_variances[1:error_orders[series]],
                variance_shape, variance_scale, stationary, max_attempts)
            coefficients[series, coefficient_columns] = beta
            error_ar[series] = phi
            error_variances[series] = variance
        end

        # Factor innovation variances stay one; only their AR coefficients change.
        for factor in 1:nfactors
            if initial == :stationary
                factor_ar[factor] = _draw_stationary_ar(
                    rng, factors[:,factor], factor_ar[factor], 1.0;
                    prior_variance=ar_variances[1:factor_orders[factor]])
            else
                regressors = _zero_padded_lags(factors[:, factor], factor_orders[factor])
                factor_ar[factor] = draw_coefficients(
                    rng, factors[:, factor], regressors, 1.0;
                    prior_variance=ar_variances[1:factor_orders[factor]], stationary, max_attempts)
            end
        end

        loadings = zeros(nseries, nfactors)
        for series in 1:nseries, level in 1:nlevels
            factor = factor_indices[series, level]
            if factor != 0
                loadings[series, factor] = coefficients[series, level + 1]
            end
        end
        centered_data = y .- coefficients[:, 1]'
        if factor_sampler == :state_space
            factors = _draw_factors_state_space(
                rng, centered_data, loadings, factor_ar, error_ar, error_variances; initial)
        elseif factor_sampler == :precision
            factors = _draw_factors_precision(
                rng, centered_data, loadings, factor_ar, error_ar, error_variances; initial)
        else
            factors = _draw_factors_sequential_precision(
                rng, centered_data, factors, loadings, factor_ar, error_ar,
                error_variances; initial)
        end

        # This joint sign change leaves fitted values and symmetric priors unchanged.
        _identify_factor_signs!(factors, coefficients, factor_indices, anchors)
        if iteration > burnin
            draw = iteration - burnin
            factor_draws[:, :, draw] = factors
            coefficient_draws[draw, :] = vec(permutedims(coefficients))
            variance_draws[draw, :] = error_variances
            factor_ar_draws[draw, :] = reduce(vcat, factor_ar; init=Float64[])
            error_ar_draws[draw, :] = reduce(vcat, error_ar; init=Float64[])
        end
    end

    means = DFMMeans(dropdims(mean(factor_draws; dims=3); dims=3),
                     mean(coefficient_draws; dims=1), mean(variance_draws; dims=1),
                     mean(factor_ar_draws; dims=1), mean(error_ar_draws; dims=1))
    return DFMResults(factor_draws, coefficient_draws, variance_draws,
                      factor_ar_draws, error_ar_draws, means)
end

function _validate_estimator_inputs(data, factor_counts, assignments, level_orders,
                                    error_orders, ndraws, burnin)
    nobs, nseries = size(data)
    nobs > 0 && nseries > 0 || throw(ArgumentError("data must have at least one row and column"))
    all(value -> value isa Real && isfinite(value), data) ||
        throw(ArgumentError("estimation requires finite, complete, real-valued data"))
    ndraws > 0 || throw(ArgumentError("ndraws must be positive"))
    burnin >= 0 || throw(ArgumentError("burnin must be nonnegative"))
    nlevels = length(factor_counts)
    nlevels > 0 && all(>(0), factor_counts) ||
        throw(ArgumentError("every level must have at least one factor"))
    size(assignments) == (nseries, nlevels) ||
        throw(DimensionMismatch("factorassign must have one row per series and one column per level"))
    length(level_orders) == nlevels ||
        throw(DimensionMismatch("factorlags must have one entry per level"))
    length(error_orders) == nseries ||
        throw(DimensionMismatch("errorlags must have one entry per series"))
    all(>=(0), level_orders) && all(>=(0), error_orders) ||
        throw(ArgumentError("lag orders must be nonnegative"))
    for level in 1:nlevels
        all(index -> 0 <= index <= factor_counts[level], assignments[:, level]) ||
            throw(ArgumentError("factor assignment is outside the declared factor range"))
        for factor in 1:factor_counts[level]
            any(==(factor), assignments[:, level]) ||
                throw(ArgumentError("every declared factor must be assigned to at least one series"))
        end
    end
    return nothing
end

function _factor_sign_anchors(factor_indices, nfactors, requested)
    if requested === nothing
        return [findfirst(==(factor), factor_indices)[1] for factor in 1:nfactors]
    end
    requested isa AbstractVector && length(requested) == nfactors ||
        throw(ArgumentError("sign_anchors must contain one series index per factor"))
    anchors = Int[]
    for (factor, series) in enumerate(requested)
        series isa Integer && 1 <= series <= size(factor_indices, 1) ||
            throw(ArgumentError("sign anchors must be valid integer series indices"))
        factor in factor_indices[series, :] ||
            throw(ArgumentError("each sign anchor must load on its factor"))
        push!(anchors, series)
    end
    return anchors
end

function _initialize_factors(data, factor_indices, nfactors)
    nobs = size(data, 1)
    factors = zeros(nobs, nfactors)
    residual = data .- mean(data; dims=1)
    for factor in 1:nfactors
        series = findall(row -> factor in factor_indices[row, :], axes(data, 2))
        decomposition = svd(residual[:, series]; full=false)
        factors[:, factor] = sqrt(nobs) .* decomposition.U[:, 1]
        loadings = residual[:, series]' * factors[:, factor] / nobs
        residual[:, series] -= factors[:, factor] * loadings'
    end
    return factors
end

function _zero_padded_lags(series, order)
    nobs = length(series)
    regressors = zeros(nobs, order)
    for lag in 1:min(order, nobs - 1)
        regressors[(lag + 1):end, lag] = series[1:(end - lag)]
    end
    return regressors
end

function _identify_factor_signs!(factors, coefficients, factor_indices, anchors)
    for (factor, anchor) in enumerate(anchors)
        anchor_level = findfirst(==(factor), factor_indices[anchor, :])
        if coefficients[anchor, anchor_level + 1] < 0
            factors[:, factor] *= -1
            for series in axes(factor_indices, 1), level in axes(factor_indices, 2)
                if factor_indices[series, level] == factor
                    coefficients[series, level + 1] *= -1
                end
            end
        end
    end
    return nothing
end

# Blocks are [f_t, f_{t-1}, ...] for each factor, followed by
# [e_t, e_{t-1}, ...] for each series. AR(0) still needs its current value.
function _factor_state_space(loadings, factor_ar, error_ar, error_variances)
    nseries, nfactors = size(loadings)
    blocks = vcat(factor_ar, error_ar)
    widths = max.(1, length.(blocks))
    starts = cumsum([1; widths[1:end-1]])
    nstates = sum(widths)
    transition = zeros(nstates, nstates)
    innovation_covariance = zeros(nstates, nstates)
    variances = vcat(ones(nfactors), error_variances)
    for (block, coefficients) in enumerate(blocks)
        first = starts[block]
        order = length(coefficients)
        transition[first, first:(first + order - 1)] = coefficients
        for lag in 1:(widths[block] - 1)
            transition[first + lag, first + lag - 1] = 1.0
        end
        innovation_covariance[first, first] = variances[block]
    end
    measurement = zeros(nseries, nstates)
    for factor in 1:nfactors
        measurement[:, starts[factor]] = loadings[:, factor]
    end
    for series in 1:nseries
        measurement[series, starts[nfactors + series]] = 1.0
    end
    model = SSModel(measurement, zeros(nseries, 0), transition, zeros(nstates),
                    zeros(nseries, nseries), innovation_covariance, zeros(0, 0))
    return model, starts[1:nfactors]
end

function _draw_factors_state_space(rng, data, loadings, factor_ar, error_ar, error_variances;
                                   initial=:zero)
    model, factor_columns = _factor_state_space(loadings, factor_ar, error_ar, error_variances)
    nstates = size(model.F, 1)
    if initial == :stationary
        # Initialize each independent AR block under the same stability rule
        # used by the parameter and precision samplers. A numerical unit-root
        # margin must not grow merely because unrelated states were added.
        covariance = zeros(nstates,nstates)
        first = 1
        for (coefficients, variance) in zip(vcat(factor_ar,error_ar),
                                            vcat(ones(length(factor_ar)),error_variances))
            width = max(1,length(coefficients))
            indices = first:(first+width-1)
            covariance[indices,indices] = variance * _stationary_ar_covariance(coefficients,width)
            first += width
        end
        states = KNFactorSampler(rng, data, model;
                                initial_mean=zeros(nstates), initial_cov=covariance)
    elseif initial == :zero
        states = KNFactorSampler(rng, data, model;
                                initial_mean=zeros(nstates), initial_cov=zeros(nstates, nstates))
    else
        throw(ArgumentError("initial must be :stationary or :zero"))
    end
    return states[:, factor_columns]
end

# W*x is the vector of AR innovations, with x_t = 0 before the sample.
function _ar_innovation_matrix(coefficients, nobs; initial=:zero)
    if initial == :stationary
        return _stationary_ar_whiten(Matrix{Float64}(I,nobs,nobs), coefficients)
    end
    initial == :zero || throw(ArgumentError("initial must be :stationary or :zero"))
    transform = Matrix{Float64}(I, nobs, nobs)
    for lag in 1:min(length(coefficients), nobs - 1)
        for time in (lag + 1):nobs
            transform[time, time - lag] = -coefficients[lag]
        end
    end
    return transform
end

# Factors are stacked by column: every date of factor 1, then factor 2, etc.
# Completing the Gaussian square gives precision J and linear term h; the
# conditional mean is J\h and covariance is inv(J).
function _factor_precision(data, loadings, factor_ar, error_ar, error_variances;
                            initial=:zero)
    nobs, nseries = size(data)
    nfactors = size(loadings, 2)
    precision = zeros(nobs * nfactors, nobs * nfactors)
    linear_term = zeros(nobs * nfactors)
    ranges = [(1 + (factor - 1) * nobs):(factor * nobs) for factor in 1:nfactors]
    for factor in 1:nfactors
        transform = _ar_innovation_matrix(factor_ar[factor], nobs; initial)
        precision[ranges[factor], ranges[factor]] = transform' * transform
    end
    for series in 1:nseries
        transform = _ar_innovation_matrix(error_ar[series], nobs; initial)
        error_precision = (transform' * transform) / error_variances[series]
        weighted_data = error_precision * data[:, series]
        active_factors = findall(!iszero, loadings[series, :])
        for factor in active_factors
            linear_term[ranges[factor]] += loadings[series, factor] * weighted_data
            for other in active_factors
                precision[ranges[factor], ranges[other]] +=
                    (loadings[series, factor] * loadings[series, other]) * error_precision
            end
        end
    end
    return Symmetric(precision), linear_term
end

function _draw_factors_precision(rng, data, loadings, factor_ar, error_ar, error_variances;
                                 initial=:zero)
    precision, linear_term = _factor_precision(data, loadings, factor_ar, error_ar, error_variances; initial)
    decomposition = cholesky(precision)
    mean_factor = decomposition \ linear_term
    draw = mean_factor + decomposition.U \ randn(rng, length(mean_factor))
    return reshape(draw, size(data, 1), size(loadings, 2))
end

# Otrok-Whiteman's multi-factor extension conditions on the other factors and
# draws one whole factor path at a time. Every later draw uses the newly updated
# paths from earlier in this sweep. This is a Gibbs sweep, not independent joint
# sampling; the separate :precision option draws all factor paths together.
function _draw_factors_sequential_precision(rng, data, previous_factors, loadings,
                                            factor_ar, error_ar, error_variances;
                                            initial=:stationary)
    factors = copy(previous_factors)
    for factor in axes(factors,2)
        partial_data = copy(data)
        for other in axes(factors,2)
            if other != factor
                partial_data -= factors[:,other] * loadings[:,other]'
            end
        end
        factors[:,factor] = vec(_draw_factors_precision(rng, partial_data,
            loadings[:,factor:factor], factor_ar[factor:factor], error_ar,
            error_variances; initial))
    end
    return factors
end
