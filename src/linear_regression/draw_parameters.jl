"""
    draw_parameters([rng], Y, X, σ2; beta_prior_variance=1000,
                    variance_shape=0.5, variance_scale=0.05)

Take one Gibbs step for regression with independent normal errors. Draw `β`
given `σ2`, then draw a new variance given `β`. Return `(β, σ2)`.
The independent priors are a zero-mean normal for each coefficient, with
`beta_prior_variance` a shared scalar or a vector of coefficient variances, and
`σ2 ~ InverseGamma(variance_shape, variance_scale)`.
"""
function draw_parameters(rng::AbstractRNG, Y, X, σ2::Real;
                         beta_prior_variance=1000.0,
                         variance_shape::Real=0.5, variance_scale::Real=0.05)
    β = draw_coefficients(rng, Y, X, σ2; prior_variance=beta_prior_variance)
    variance = draw_error_variance(rng, Y, X, β;
                                  prior_shape=variance_shape, prior_scale=variance_scale)
    return β, variance
end

draw_parameters(Y, X, σ2::Real; rng::AbstractRNG=Random.default_rng(), kwargs...) =
    draw_parameters(rng, Y, X, σ2; kwargs...)

# Each column holds an unmodified lag of the original series. Zero padding
# describes the explicitly fixed presample values in the :zero model.
function _regression_lags(values::AbstractVector{<:Real}, p::Integer)
    lags = zeros(length(values), p)
    for lag in 1:p
        for t in (lag + 1):length(values)
            lags[t, lag] = values[t - lag]
        end
    end
    return lags
end

function _regression_rows(n::Integer, p::Integer, initial::Symbol)
    p >= 0 || throw(ArgumentError("the number of lags must be nonnegative"))
    initial in (:conditional, :zero) ||
        throw(ArgumentError("initial must be :conditional or :zero"))
    initial == :conditional && p >= n &&
        throw(ArgumentError("conditional regression needs more observations than lags"))
    return initial == :conditional ? ((p + 1):n) : (1:n)
end

"""
    draw_parameters([rng], Y, X, ϕ, σ2; initial=:conditional,
                    stationary=true, beta_prior_variance=1000,
                    ar_prior_variance=1000, variance_shape=0.5,
                    variance_scale=0.05, max_attempts=10000)

Take one Gibbs step for `Y = X*β + error`, where the error follows an AR process
with lag coefficients `ϕ` and innovation variance `σ2`. Return `(β, ϕ, σ2)`.
The new variance is conditional on both newly drawn coefficient vectors.

With `initial=:conditional`, the likelihood uses times `p+1:T`, where
`p = length(ϕ)`. With `initial=:zero`, it uses all observations and fixes
presample errors to zero. With `initial=:stationary`, include the stationary
distribution of the first `min(p,T)` errors. This requires `stationary=true`;
the AR update is a Metropolis-Hastings step correcting for that initial density.
A rejected proposal keeps the old AR coefficients. Empty `ϕ` gives independent
errors under all three conventions.

Priors are independent zero-mean normals for `β` and `ϕ`, with variances set by
`beta_prior_variance` and `ar_prior_variance`. Each accepts a shared scalar or a
positive vector with one entry per coefficient. The variance prior is
`σ2 ~ InverseGamma(variance_shape, variance_scale)`. `stationary=true` truncates
the AR prior to stable coefficients by rejection; see [`draw_coefficients`](@ref).
Pass an RNG as the first argument or with `rng=`. Inputs are never modified.
"""
function draw_parameters(rng::AbstractRNG, Y, X,
                         ϕ::AbstractVector{<:Real}, σ2::Real;
                         initial::Symbol=:conditional, stationary::Bool=true,
                         beta_prior_variance=1000.0, ar_prior_variance=1000.0,
                         variance_shape::Real=0.5, variance_scale::Real=0.05,
                         max_attempts::Integer=10000)
    y, x = _regression_data(Y, X)
    all(isfinite, ϕ) || throw(ArgumentError("autoregressive coefficients must be finite"))
    stationary && !isstationary(ϕ) && throw(ArgumentError("initial ϕ must be stationary when stationary=true"))
    _regression_prior_variances(ar_prior_variance, length(ϕ), "ar_prior_variance")
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))
    if initial == :stationary
        stationary || throw(ArgumentError("initial=:stationary requires stationary=true"))
        result = _draw_stationary_regression(rng, y, x, ϕ, σ2;
            beta_prior_variance, ar_prior_variance, variance_shape, variance_scale)
        return X isa AbstractVector ? only(result[1]) : result[1], result[2], result[3]
    end
    p = length(ϕ)
    rows = _regression_rows(length(y), p, initial)

    # Apply (1 - ϕ₁L - ... - ϕₚLᵖ) to both sides. Always lag the original
    # arrays: successively lagging an already transformed column is incorrect.
    y_star = y - _regression_lags(y, p) * ϕ
    x_star = copy(x)
    for j in axes(x, 2)
        x_star[:, j] -= _regression_lags(x[:, j], p) * ϕ
    end
    β = draw_coefficients(rng, y_star[rows], x_star[rows, :], σ2;
                          prior_variance=beta_prior_variance, max_attempts=max_attempts)

    residuals = y - x * β
    lagged_residuals = _regression_lags(residuals, p)[rows, :]
    new_ϕ = draw_coefficients(rng, residuals[rows], lagged_residuals, σ2;
                              prior_variance=ar_prior_variance,
                              stationary=stationary, max_attempts=max_attempts)
    variance = draw_error_variance(rng, residuals[rows], lagged_residuals, new_ϕ;
                                  prior_shape=variance_shape, prior_scale=variance_scale)
    return X isa AbstractVector ? only(β) : β, new_ϕ, variance
end

draw_parameters(Y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}}, X,
                ϕ::AbstractVector{<:Real}, σ2::Real;
                rng::AbstractRNG=Random.default_rng(), kwargs...) =
    draw_parameters(rng, Y, X, ϕ, σ2; kwargs...)
