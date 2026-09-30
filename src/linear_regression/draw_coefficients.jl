# Use one internal representation for vector and single-column responses.
function _regression_data(Y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                          X::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}})
    Y isa AbstractMatrix && size(Y, 2) != 1 &&
        throw(DimensionMismatch("Y must be a vector or a single-column matrix"))
    y = vec(Float64.(Y))
    x = X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X)
    length(y) == size(x, 1) || throw(DimensionMismatch("Y and X must have the same number of rows"))
    isempty(y) && throw(ArgumentError("at least one observation is required"))
    all(isfinite, y) && all(isfinite, x) || throw(ArgumentError("Y and X must contain finite values"))
    return y, x
end

function _positive_regression_parameter(value::Real, name::AbstractString)
    isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    converted = Float64(value)
    isfinite(converted) && converted > 0 ||
        throw(ArgumentError("$name must be finite and positive in Float64"))
    return converted
end

function _regression_prior_variances(value, k::Integer, name::AbstractString)
    if value isa Real
        return fill(_positive_regression_parameter(value, name), k)
    end
    value isa AbstractVector{<:Real} ||
        throw(ArgumentError("$name must be a positive real scalar or vector"))
    length(value) == k || throw(DimensionMismatch("$name must have one entry per coefficient"))
    return [_positive_regression_parameter(entry, name) for entry in value]
end

"""
    isstationary(ϕ)

Return whether the autoregression with lag coefficients `ϕ` is stable: every
eigenvalue of its companion matrix has absolute value below one. Coefficients
are ordered from lag 1 onward. An empty vector denotes white noise and returns
`true`. A scalar is treated as an AR(1) coefficient. For higher orders, roots
within `100 * length(ϕ) * eps(Float64)` of the unit circle are treated as unstable
to avoid accepting a unit root because of eigenvalue roundoff.
"""
function isstationary(ϕ::AbstractVector{<:Real})
    all(isfinite, ϕ) || throw(ArgumentError("autoregressive coefficients must be finite"))
    p = length(ϕ)
    p == 0 && return true
    p == 1 && return abs(ϕ[1]) < 1
    companion = zeros(p, p)
    companion[1, :] = ϕ
    for j in 2:p
        companion[j, j - 1] = 1
    end
    tolerance = 100 * p * eps(Float64)
    return all(abs.(eigvals(companion)) .< 1 - tolerance)
end

isstationary(ϕ::Real) = isstationary([ϕ])

"""
    draw_coefficients([rng], Y, X, σ2; prior_mean=0, prior_variance=1000,
                      stationary=false, max_attempts=10000)

Draw regression coefficients given the error variance in `Y = X * β + error`.
Rows are observations; supply a column of ones in `X` for an intercept. The
independent normal prior has mean `prior_mean` (a scalar or coefficient vector).
`prior_variance` is one positive variance shared by all coefficients, or a
positive vector with one variance per coefficient. The prior is independent
of `σ2`.

If `stationary=true`, all columns of `X` must represent consecutive lags of an
autoregression without an intercept. Draws outside the stationary region are
rejected, which truncates the normal prior to that region. Failure to accept a
draw within `max_attempts` raises an error; no coefficient clipping is used.

A vector `X` returns a scalar; a matrix `X` returns a vector. `Y` may be a vector
or a single-column matrix. Pass an RNG as the first argument or with `rng=`.
"""
function draw_coefficients(rng::AbstractRNG,
                           Y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                           X::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                           σ2::Real;
                           prior_mean=0.0, prior_variance=1000.0,
                           stationary::Bool=false, max_attempts::Integer=10000)
    y, x = _regression_data(Y, X)
    variance = _positive_regression_parameter(σ2, "σ2")
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))
    k = size(x, 2)
    prior_variances = _regression_prior_variances(prior_variance, k, "prior_variance")
    prior_mean isa Real || prior_mean isa AbstractVector{<:Real} ||
        throw(ArgumentError("prior_mean must be a real scalar or vector"))
    mean0 = prior_mean isa Real ? fill(Float64(prior_mean), k) : Float64.(prior_mean)
    length(mean0) == k || throw(DimensionMismatch("prior_mean must have one entry per coefficient"))
    all(isfinite, mean0) || throw(ArgumentError("prior_mean must be finite"))
    k == 0 && return Float64[]

    # Add the normal prior as k weighted observations. QR then gives a factor
    # R with R'R = X'X / σ² + Diagonal(1 ./ prior_variances), without squaring
    # the condition number of X. The prior also supports rank-deficient designs.
    prior_standard_deviations = sqrt.(prior_variances)
    augmented_x = vcat(x / sqrt(variance), Diagonal(1 ./ prior_standard_deviations))
    augmented_y = vcat(y / sqrt(variance), mean0 ./ prior_standard_deviations)
    factor = qr(augmented_x)
    mean1 = factor \ augmented_y
    precision_root = UpperTriangular(factor.R)
    for attempt in 1:max_attempts
        β = mean1 + precision_root \ randn(rng, k)
        if !stationary || isstationary(β)
            return X isa AbstractVector ? only(β) : β
        end
    end
    error("No stationary coefficient draw after $max_attempts attempts. Check the data and prior, or increase max_attempts.")
end

draw_coefficients(Y, X, σ2; rng::AbstractRNG=Random.default_rng(), kwargs...) =
    draw_coefficients(rng, Y, X, σ2; kwargs...)
