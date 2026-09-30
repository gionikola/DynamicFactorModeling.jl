module SBCReference

# Independent one-factor AR(0)/AR(1) calibration reference.
# Including this file generates no parameters, observations, or posterior draws.
using Random
using LinearAlgebra
using Distributions: Normal, InverseGamma, truncated

export draw_prior_parameters, generate_case, fold_factor_signs,
       stationary_ar_covariance, observation_moments, marginal_loglikelihood,
       conditional_loglikelihood, randomized_rank

function positive_count(value, name)
    value isa Integer && value > 0 || throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end

function finite_number(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$name must be finite in Float64"))
    return converted
end

function finite_vector(value, name)
    value isa AbstractVector{<:Real} || throw(ArgumentError("$name must be a real vector"))
    result = Float64.(value)
    all(isfinite, result) || throw(ArgumentError("$name must be finite in Float64"))
    return result
end

function ar_order(order)
    order isa Integer && order in (0, 1) || throw(ArgumentError("order must be 0 or 1"))
    return Int(order)
end

# Return checked copies so reference calculations never mutate a caller's truth.
function checked_parameters(parameters)
    names = (:order, :intercepts, :loadings, :error_variances, :factor_ar, :error_ar)
    parameters isa NamedTuple && all(name -> haskey(parameters, name), names) ||
        throw(ArgumentError("parameters must contain order, intercepts, loadings, error_variances, factor_ar, and error_ar"))
    order = ar_order(parameters.order)
    intercepts = finite_vector(parameters.intercepts, "intercepts")
    loadings = finite_vector(parameters.loadings, "loadings")
    error_variances = finite_vector(parameters.error_variances, "error variances")
    error_ar = finite_vector(parameters.error_ar, "error AR coefficients")
    factor_ar = finite_number(parameters.factor_ar, "factor AR coefficient")
    series = positive_count(length(intercepts), "number of series")
    all(length(values) == series for values in (loadings, error_variances, error_ar)) ||
        throw(DimensionMismatch("all series-specific parameter vectors must have the same length"))
    all(>(0), error_variances) || throw(ArgumentError("error innovation variances must be positive"))
    abs(factor_ar) < 1 && all(value -> abs(value) < 1, error_ar) ||
        throw(ArgumentError("AR coefficients must lie strictly between -1 and 1"))
    order == 1 || (iszero(factor_ar) && all(iszero, error_ar)) ||
        throw(ArgumentError("AR(0) parameters must have zero AR coefficients"))
    return (; order, intercepts, loadings, error_variances, factor_ar, error_ar)
end

"""
    draw_prior_parameters(rng; order=1, series=2)

Draw the package's default one-factor priors before sign identification:
independent intercepts/loadings with variance 100, error innovation variances
InverseGamma(2,1), and independent standard-normal AR(1) coefficients truncated
to (-1,1). AR(0) uses zero coefficients. Factor innovation variance is one.
An explicit RNG is required; no difficult prior draws are filtered out.
"""
function draw_prior_parameters(rng::AbstractRNG; order=1, series=2)
    order = ar_order(order)
    series = positive_count(series, "series")
    intercepts = 10 .* randn(rng, series)
    loadings = 10 .* randn(rng, series)
    error_variances = rand(rng, InverseGamma(2.0, 1.0), series)
    stationary_prior = truncated(Normal(0.0, 1.0), -1.0, 1.0)
    factor_ar = order == 1 ? rand(rng, stationary_prior) : 0.0
    error_ar = order == 1 ? rand(rng, stationary_prior, series) : zeros(series)
    return checked_parameters((; order, intercepts, loadings, error_variances, factor_ar, error_ar))
end

"""
    stationary_ar_covariance(coefficient, dates; innovation_variance=1)

Analytic AR(1) covariance, variance * coefficient^abs(t-u)/(1-coefficient^2).
A zero coefficient gives the AR(0) covariance. No package state-space or
whitening code is used. An unrepresentable covariance is an error, not a reason
to truncate the prior or silently add numerical noise.
"""
function stationary_ar_covariance(coefficient, dates; innovation_variance=1.0)
    dates = positive_count(dates, "dates")
    coefficient = finite_number(coefficient, "AR coefficient")
    variance = finite_number(innovation_variance, "innovation variance")
    abs(coefficient) < 1 || throw(ArgumentError("the AR coefficient must lie strictly between -1 and 1"))
    variance > 0 || throw(ArgumentError("innovation variance must be positive"))
    marginal_variance = variance / ((1 - coefficient) * (1 + coefficient))
    isfinite(marginal_variance) || throw(ArgumentError("stationary covariance overflows Float64"))
    return [marginal_variance * coefficient^abs(t-u) for t in 1:dates, u in 1:dates]
end

"""
    fold_factor_signs(parameters, factors, initial_factor, innovations; anchor=1)

Return copies with a nonnegative anchor loading. A negative anchor flips every
loading, the whole factor path, its initial state, and its innovations together.
An exactly zero anchor does not flip, matching the public estimator. Intercepts,
AR coefficients, and errors are unchanged. No factor is rescaled or recentered.
"""
function fold_factor_signs(parameters, factors, initial_factor, innovations; anchor=1)
    parameters = checked_parameters(parameters)
    anchor isa Integer && 1 <= anchor <= length(parameters.loadings) ||
        throw(ArgumentError("anchor must identify an observed series"))
    factors = finite_vector(factors, "factors")
    innovations = finite_vector(innovations, "factor innovations")
    length(factors) == length(innovations) || throw(DimensionMismatch("factor and innovation lengths differ"))
    initial_factor = finite_number(initial_factor, "initial factor")
    flipped = parameters.loadings[anchor] < 0
    if flipped
        parameters.loadings .*= -1
        factors .*= -1
        initial_factor *= -1
        innovations .*= -1
    end
    return (; parameters, factors, initial_factor, innovations, flipped)
end

"""
    generate_case(rng; order=1, dates=6, series=2, anchor=1)

Draw parameters, a stationary factor path, independent stationary errors, and
observations under the default priors. Return the sign-identified truth,
observations (dates by series), innovations, and presample states. `factors` is
a vector because this reference has one factor. Sign folding preserves both
observations and the AR recursions. No parameter or dataset is redrawn to avoid
a weak loading, a near-unit root, or an extreme variance.
"""
function generate_case(rng::AbstractRNG; order=1, dates=6, series=2, anchor=1)
    dates = positive_count(dates, "dates")
    series = positive_count(series, "series")
    anchor isa Integer && 1 <= anchor <= series || throw(ArgumentError("anchor must identify an observed series"))
    parameters = draw_prior_parameters(rng; order, series)
    phi = parameters.factor_ar
    initial_factor = randn(rng) / sqrt((1-phi) * (1+phi))
    factor_innovations = randn(rng, dates)
    factors = zeros(dates)
    previous = initial_factor
    for t in 1:dates
        factors[t] = phi * previous + factor_innovations[t]
        previous = factors[t]
    end
    errors, error_innovations = zeros(dates, series), zeros(dates, series)
    initial_errors = zeros(series)
    for i in 1:series
        coefficient = parameters.error_ar[i]
        deviation = sqrt(parameters.error_variances[i])
        previous = deviation * randn(rng) / sqrt((1-coefficient) * (1+coefficient))
        initial_errors[i] = previous
        error_innovations[:, i] = deviation .* randn(rng, dates)
        for t in 1:dates
            errors[t, i] = coefficient * previous + error_innovations[t, i]
            previous = errors[t, i]
        end
    end
    folded = fold_factor_signs(parameters, factors, initial_factor, factor_innovations; anchor)
    signal = folded.factors * folded.parameters.loadings' .+ folded.parameters.intercepts'
    data = signal + errors
    all(isfinite, data) && all(isfinite, initial_errors) ||
        throw(ArgumentError("generated case is not representable in Float64; record this failure without replacing its seed"))
    return (; parameters=folded.parameters, factors=folded.factors, errors, signal, data,
        initial_factor=folded.initial_factor, initial_errors,
        factor_innovations=folded.innovations, error_innovations,
        sign_flipped=folded.flipped, sign_anchor=Int(anchor))
end

"""
    observation_moments(parameters, dates)

Mean and covariance after integrating out the factor path, holding intercepts
and all other parameters fixed. Ordering is all dates of series 1, then all
dates of series 2, etc., matching vec(data). The intercept prior is not integrated.
Covariance block (i,j) is b_i*b_j*K_factor plus the error covariance when i=j.
"""
function observation_moments(parameters, dates)
    parameters = checked_parameters(parameters)
    dates = positive_count(dates, "dates")
    series = length(parameters.intercepts)
    factor_covariance = stationary_ar_covariance(parameters.factor_ar, dates)
    covariance = zeros(dates * series, dates * series)
    for i in 1:series, j in 1:series
        rows, columns = ((i-1)*dates+1):(i*dates), ((j-1)*dates+1):(j*dates)
        covariance[rows, columns] = parameters.loadings[i] * parameters.loadings[j] .* factor_covariance
        if i == j
            covariance[rows, columns] += stationary_ar_covariance(parameters.error_ar[i], dates;
                innovation_variance=parameters.error_variances[i])
        end
    end
    all(isfinite, covariance) || throw(ArgumentError("observation covariance overflows Float64"))
    return (; mean=repeat(parameters.intercepts; inner=dates), covariance)
end

"""
    marginal_loglikelihood(data, parameters)

Factor-integrated Gaussian observation log likelihood. This is a data-dependent
SBC quantity, not the conditional likelihood given a sampled factor path and not
the prior predictive density that also integrates parameters. Only analytic AR
covariances and a dense Cholesky factorization are used.
"""
function marginal_loglikelihood(data, parameters)
    data isa AbstractMatrix{<:Real} || throw(ArgumentError("data must be a real matrix"))
    parameters = checked_parameters(parameters)
    size(data, 2) == length(parameters.intercepts) || throw(DimensionMismatch("data columns must match the number of series"))
    values = Matrix{Float64}(data)
    all(isfinite, values) || throw(ArgumentError("data must be finite in Float64"))
    moments = observation_moments(parameters, size(values, 1))
    root = cholesky(Symmetric(moments.covariance)).L
    standardized = root \ (vec(values) - moments.mean)
    value = -0.5 * (length(values) * log(2pi) + 2sum(log, diag(root)) + sum(abs2, standardized))
    isfinite(value) || throw(ArgumentError("Gaussian log likelihood overflows Float64"))
    return value
end

"""
    conditional_loglikelihood(data, parameters, factors)

Gaussian observation log likelihood conditional on the given factor path and
parameters. Each residual series follows its stationary AR(0) or AR(1) law,
including the first residual's stationary variance. The presample errors are
integrated out; no factor density or parameter prior enters this likelihood.
The factor AR coefficient therefore has no effect when the path is fixed.

Only the explicit residual recursion and stationary initial density are used.
Jointly flipping the factor path and all loadings leaves the result unchanged.
"""
function conditional_loglikelihood(data, parameters, factors)
    data isa AbstractMatrix{<:Real} || throw(ArgumentError("data must be a real matrix"))
    parameters = checked_parameters(parameters)
    dates = positive_count(size(data, 1), "dates")
    size(data, 2) == length(parameters.intercepts) ||
        throw(DimensionMismatch("data columns must match the number of series"))
    values = Matrix{Float64}(data)
    all(isfinite, values) || throw(ArgumentError("data must be finite in Float64"))
    factors = finite_vector(factors, "factor path")
    length(factors) == dates || throw(DimensionMismatch("factor path must match the number of dates"))

    value = 0.0
    for i in axes(values, 2)
        residuals = values[:, i] .- parameters.intercepts[i] .- parameters.loadings[i] .* factors
        all(isfinite, residuals) || throw(ArgumentError("observation residuals overflow Float64"))
        coefficient = parameters.error_ar[i]
        variance = parameters.error_variances[i]
        deviation = sqrt(variance)
        stationary_weight = (1 - coefficient) * (1 + coefficient)
        quadratic = abs2(residuals[1] * sqrt(stationary_weight) / deviation)
        for t in 2:dates
            innovation = residuals[t] - coefficient * residuals[t-1]
            quadratic += abs2(innovation / deviation)
        end
        value += -0.5 * (dates * (log(2pi) + log(variance)) -
            log(stationary_weight) + quadratic)
    end
    isfinite(value) || throw(ArgumentError("conditional Gaussian log likelihood overflows Float64"))
    return value
end

"""
    randomized_rank(rng, truth, draws)

Count draws below truth, then choose uniformly among tied positions, including
the truth's position. Valid ranks are 0:length(draws). Exact ties are randomized;
nearby unequal values are not rounded into ties. Nonfinite quantities are errors.
With no ties (including an empty draw vector), no random number is consumed.
"""
function randomized_rank(rng::AbstractRNG, truth::Real, draws::AbstractVector{<:Real})
    isfinite(truth) && all(isfinite, draws) || throw(ArgumentError("rank quantities must be finite"))
    below = count(value -> value < truth, draws)
    equal = count(==(truth), draws)
    return below + (equal == 0 ? 0 : rand(rng, 0:equal))
end

end
