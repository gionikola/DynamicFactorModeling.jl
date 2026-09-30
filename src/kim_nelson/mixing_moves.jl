# Posterior-preserving updates along directions that keep fitted signals fixed.
# Both moves use the same full factor law as the parameter and path updates.
function _mixing_ar_whiten(values, coefficients, initial)
    if initial == :stationary
        return _stationary_ar_whiten(values, coefficients)
    end
    innovations = copy(values)
    for lag in 1:min(length(coefficients), size(values, 1) - 1)
        innovations[(lag + 1):end, :] -= coefficients[lag] * values[1:(end - lag), :]
    end
    return innovations
end

function _mixing_factor_energy(values, coefficients, initial)
    whitened = _mixing_ar_whiten(reshape(values, :, 1), coefficients, initial)
    return sum(abs2, whitened)
end

# expm1 avoids cancellation for small proposals. A zero energy stays zero even
# when the proposed scale is so large that its exponential is not representable.
_mixing_energy_change(energy, log_multiplier) =
    iszero(energy) ? 0.0 : energy * expm1(log_multiplier)

"""
    _shift_factor_locations!([rng], factors, intercepts, loadings, factor_ar;
                            initial=:stationary, intercept_prior_variance=100.0)

Gibbs update of the shared factor levels and series intercepts.
`factors` is a dates-by-factors Float64 matrix, `intercepts` a Float64 vector,
and `loadings` a series-by-factors matrix. Put zeros in `loadings` for excluded
factors. `factor_ar[k]` contains factor k's AR coefficients, in lag order.
Factors have innovation variance one. Intercepts have independent zero-mean
normal priors; their variance is a positive scalar or one value per series.

Draw and return a factor shift `delta`, then update
`factors += delta'` and `intercepts -= loadings * delta`. Every fitted signal
and observation residual stays unchanged. The factor and intercept priors do
change, and are included in the conditional draw. `initial` selects the full
stationary AR density or the density conditional on zero presample factors.

Why this is a Gibbs update: write each path as `f[t,k] = g[t,k] + m[k]`, with
`m[k] = f[end,k]` and `g[end,k] = 0`, and write `b = intercepts + loadings*m`.
The map from the original paths/intercepts to `(g[1:end-1,:], m, b)` is
invertible with absolute Jacobian one. Given `(g,b)` and the other parameters,
the observation likelihood is constant in m. If Q[k] is factor k's full path
precision and D is the diagonal intercept prior covariance, completing the
Gaussian square in `delta = m_new - m` gives

    precision = Diagonal([ones' * Q[k] * ones]) + loadings' * inv(D) * loadings
    score = loadings' * inv(D) * intercepts - [ones' * Q[k] * factors[:,k]]
    delta ~ MvNormal(inv(precision) * score, inv(precision))

The implementation uses AR whitening and Cholesky solves instead of forming
path precision matrices or inverses. This updates a conditional distribution;
it is not an unconstrained random shift or a recentering at an estimated mean.
"""
function _shift_factor_locations!(rng::AbstractRNG, factors::AbstractMatrix,
                                 intercepts::AbstractVector, loadings::AbstractMatrix,
                                 factor_ar::AbstractVector;
                                 initial::Symbol=:stationary,
                                 intercept_prior_variance=100.0)
    eltype(factors) === Float64 && eltype(intercepts) === Float64 ||
        throw(ArgumentError("factors and intercepts must store Float64 values"))
    initial in (:stationary, :zero) ||
        throw(ArgumentError("initial must be :stationary or :zero"))
    dates, nfactors = size(factors)
    dates > 0 && nfactors > 0 && !isempty(intercepts) ||
        throw(ArgumentError("at least one date, factor, and series are required"))
    size(loadings) == (length(intercepts), nfactors) ||
        throw(DimensionMismatch("loadings must have one row per series and one column per factor"))
    length(factor_ar) == nfactors ||
        throw(DimensionMismatch("factor_ar must have one coefficient vector per factor"))
    all(coefficients -> coefficients isa AbstractVector{<:Real}, factor_ar) ||
        throw(ArgumentError("each factor_ar entry must be a real coefficient vector"))
    loadings isa AbstractMatrix{<:Real} || throw(ArgumentError("loadings must be real"))
    L = Matrix{Float64}(loadings)
    coefficients = [Vector{Float64}(value) for value in factor_ar]
    all(isfinite, factors) && all(isfinite, intercepts) && all(isfinite, L) &&
        all(value -> all(isfinite, value), coefficients) ||
        throw(ArgumentError("inputs must be finite in Float64"))
    variances = _regression_prior_variances(
        intercept_prior_variance, length(intercepts), "intercept_prior_variance")

    prior_precision = zeros(nfactors)
    prior_score = zeros(nfactors)
    for factor in 1:nfactors
        values = hcat(ones(dates), factors[:, factor])
        whitened = _mixing_ar_whiten(values, coefficients[factor], initial)
        prior_precision[factor] = sum(abs2, whitened[:, 1])
        prior_score[factor] = -dot(whitened[:, 1], whitened[:, 2])
    end

    weighted_loadings = L ./ sqrt.(variances)
    precision = Diagonal(prior_precision) + weighted_loadings' * weighted_loadings
    score = prior_score + weighted_loadings' * (intercepts ./ sqrt.(variances))
    all(isfinite, precision) && all(isfinite, score) ||
        throw(ArgumentError("location conditional is not finite in Float64"))
    root = cholesky(Symmetric(precision))
    delta = root \ score + root.U \ randn(rng, nfactors)

    # Compute both new arrays before mutating either input.
    new_factors = factors .+ delta'
    new_intercepts = intercepts - L * delta
    all(isfinite, new_factors) && all(isfinite, new_intercepts) ||
        throw(ArgumentError("location draw is not finite in Float64"))
    factors .= new_factors
    intercepts .= new_intercepts
    return delta
end

_shift_factor_locations!(factors, intercepts, loadings, factor_ar;
                        rng::AbstractRNG=Random.default_rng(), kwargs...) =
    _shift_factor_locations!(rng, factors, intercepts, loadings, factor_ar; kwargs...)


"""
    _rescale_factors!([rng], factors, loadings, factor_ar; active_loadings,
                     initial=:stationary, loading_prior_variance=100.0,
                     step_size=0.1)

Metropolis update of each factor's scale and its free loadings.
`factors` is a dates-by-factors Float64 matrix and `loadings` a series-by-factors
Float64 matrix. `factor_ar[k]` contains lag coefficients. Factor innovation
variances are fixed at one. Free loadings have independent zero-mean normal
priors, whose variances are a positive scalar or a matrix shaped like loadings.
Only active entries of that variance matrix are used.

The Boolean mask `active_loadings` identifies free parameters. Supply it from
the model structure: a free loading currently equal to zero still counts as a
parameter. Entries outside the mask must be zero and remain zero.

For each factor, draw `eta = step_size * randn(rng)` and propose
`f_new = exp(eta)*f`, `b_new = exp(-eta)*b` for its active loadings. This keeps
every factor contribution, loading sign, and fitted signal unchanged. A fixed
positive-anchor sign convention is therefore preserved. The step
size is fixed, with no adaptation. The full stationary or zero-presample factor
prior and the loading priors determine the acceptance probability.

For T dates and n free loadings, the augmented map `(f,b,eta)` to
`(exp(eta)f,exp(-eta)b,-eta)` is its own inverse. Its Jacobian is block triangular,
with determinant magnitude `exp((T-n)*eta)`. The symmetric normal proposal
density cancels with its reverse. Therefore the log acceptance ratio is

    -(exp(2eta)-1) * (f'Qf) / 2
    -(exp(-2eta)-1) * sum(b.^2 ./ prior_variances) / 2 + (T-n)*eta

Q is the FULL factor path precision, including the stationary initial density
when requested. AR coefficients and all innovation variances stay fixed.
This is an augmented-variable Metropolis step. The general acceptance identity
is equation (2.2) in Green and Hastie, "Reversible jump MCMC":
https://people.maths.bris.ac.uk/~mapjg/papers/GreenHastie.pdf.

Return vectors `accepted`, `log_scale`, and `log_ratio`, one entry per factor.
The latter two describe proposals, including rejected ones. Proposals that
overflow or map a nonzero value to floating-point zero are rejected with a
log ratio of `-Inf`. This move alone does not explore a full posterior: path
directions and factor-loading products stay fixed. It is intended as an extra
step alongside valid parameter and factor updates; improved mixing is a
hypothesis to test, not an assumption of this implementation.
"""
function _rescale_factors!(rng::AbstractRNG, factors::AbstractMatrix,
                         loadings::AbstractMatrix, factor_ar::AbstractVector;
                         active_loadings::AbstractMatrix{Bool},
                         initial::Symbol=:stationary,
                         loading_prior_variance=100.0, step_size::Real=0.1)
    eltype(factors) === Float64 && eltype(loadings) === Float64 ||
        throw(ArgumentError("factors and loadings must store Float64 values"))
    dates, count = size(factors)
    dates > 0 && count > 0 && size(loadings, 1) > 0 ||
        throw(ArgumentError("at least one date, factor, and series are required"))
    size(loadings, 2) == count && length(factor_ar) == count ||
        throw(DimensionMismatch("one loading column and AR vector are required per factor"))
    size(active_loadings) == size(loadings) ||
        throw(DimensionMismatch("active_loadings must have the same shape as loadings"))
    all(iszero, loadings[.!active_loadings]) ||
        throw(ArgumentError("loadings outside the active mask must be zero"))
    initial in (:stationary, :zero) ||
        throw(ArgumentError("initial must be :stationary or :zero"))
    all(value -> value isa AbstractVector{<:Real}, factor_ar) ||
        throw(ArgumentError("factor_ar must contain real coefficient vectors"))
    coefficients = [Vector{Float64}(value) for value in factor_ar]
    all(isfinite, factors) && all(isfinite, loadings) &&
        all(value -> all(isfinite, value), coefficients) ||
        throw(ArgumentError("factors, loadings, and AR coefficients must be finite"))
    step = _positive_regression_parameter(step_size, "step_size")
    variances = if loading_prior_variance isa Real
        value = _positive_regression_parameter(
            loading_prior_variance, "loading_prior_variance")
        fill(value, size(loadings))
    elseif loading_prior_variance isa AbstractMatrix{<:Real}
        size(loading_prior_variance) == size(loadings) ||
            throw(DimensionMismatch("loading prior variances must have the same shape as loadings"))
        Matrix{Float64}(loading_prior_variance)
    else
        throw(ArgumentError("loading_prior_variance must be a scalar or real matrix"))
    end
    all(value -> isfinite(value) && value > 0, variances[active_loadings]) ||
        throw(ArgumentError("active loading prior variances must be finite and positive"))

    active = [findall(active_loadings[:, factor]) for factor in 1:count]
    factor_energies = [_mixing_factor_energy(factors[:, k], coefficients[k], initial) for k in 1:count]
    loading_energies = [sum(abs2, loadings[active[k], k] ./ sqrt.(variances[active[k], k]))
                        for k in 1:count]
    all(isfinite, factor_energies) && all(isfinite, loading_energies) ||
        throw(ArgumentError("current prior energies must be finite in Float64"))

    accepted, log_scale, log_ratio = falses(count), zeros(count), zeros(count)
    for factor in 1:count
        eta = step * randn(rng)
        log_uniform = log(rand(rng))
        log_scale[factor] = eta
        if !isfinite(eta)
            log_ratio[factor] = -Inf
            continue
        end
        ratio = -_mixing_energy_change(factor_energies[factor], 2 * eta) / 2 -
                _mixing_energy_change(loading_energies[factor], -2 * eta) / 2 +
                (dates - length(active[factor])) * eta
        # Overflow in a Gaussian energy gives a proposal of zero probability.
        isnan(ratio) && (ratio = -Inf)
        log_ratio[factor] = ratio
        log_uniform < min(0.0, ratio) || continue
        new_factor = exp(eta) .* factors[:, factor]
        new_loadings = exp(-eta) .* loadings[active[factor], factor]
        representable = all(isfinite, new_factor) && all(isfinite, new_loadings) &&
            all(iszero(factors[t, factor]) || !iszero(new_factor[t]) for t in 1:dates) &&
            all(iszero(loadings[row, factor]) || !iszero(new_loadings[j])
                for (j, row) in enumerate(active[factor]))
        if !representable
            log_ratio[factor] = -Inf
            continue
        end
        factors[:, factor] = new_factor
        loadings[active[factor], factor] = new_loadings
        accepted[factor] = true
    end
    return (; accepted, log_scale, log_ratio)
end

_rescale_factors!(factors, loadings, factor_ar;
                 rng::AbstractRNG=Random.default_rng(), kwargs...) =
    _rescale_factors!(rng, factors, loadings, factor_ar; kwargs...)
