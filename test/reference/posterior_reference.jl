module PosteriorReference

using LinearAlgebra, Statistics

# Gaussian quadrature from the symmetric Legendre recurrence. The sum of the
# returned weights is upper-lower; no package posterior code is called here.
function gauss_legendre(order, lower, upper)
    off_diagonal = [j / sqrt(4j^2 - 1) for j in 1:(order - 1)]
    decomposition = eigen(SymTridiagonal(zeros(order), off_diagonal))
    nodes = (lower + upper) / 2 .+ (upper - lower) / 2 .* decomposition.values
    weights = (upper - lower) .* decomposition.vectors[1, :].^2
    return nodes, weights
end

const statistics = (:intercept, :loading, :loading_squared, :variance,
                    :precision, :factor, :factor_squared, :fitted_factor,
                    :factor_ar_squared, :error_ar_squared,
                    :factor_ar_loading_squared, :error_ar_variance)

coefficient_variances(value) = value isa Real ? (value, value) : (value[1], value[2])

"""
Calculate a one-date, one-series DFM posterior without Gibbs sampling.

The priors are a ~ N(0,v_a), b ~ N(0,v_b), s ~ IG(shape,scale). Factor and error innovation
variances are 1 and s. Integrating the intercept and factor analytically gives
    y | b,s,ψ,φ ~ N(0, v_a + b² V_f + s V_e),
where V_f=V_e=1 for AR(0) or zero presample initialization. Stationary AR(1)
instead has V_f=1/(1-ψ²), V_e=1/(1-φ²). The positive loading is the result of
jointly folding the signs of b and f, exactly as in the public estimator.

Only b, log(s), and (for AR(1)) the magnitudes of ψ and φ require quadrature.
Normalizing constants cancel in posterior ratios. Limits cover nine loading
prior standard deviations and a wide log-variance range. Higher quadrature
orders must agree before this is used as a posterior oracle.
"""
function one_date(y; beta_prior_variance, variance_shape, variance_scale,
                  ar_prior_variance=0.1, initial=:zero, ar_order=0,
                  order=120, ar_quadrature_order=40,
                  loading_upper=nothing,
                  variance_upper=variance_scale * exp(9))
    ar_order in (0, 1) || throw(ArgumentError("the oracle supports AR(0) and AR(1)"))
    initial in (:zero, :stationary) || throw(ArgumentError("unsupported initial condition"))
    v, loading_variance = coefficient_variances(beta_prior_variance)
    loading_upper === nothing && (loading_upper = 9sqrt(loading_variance))
    loading_nodes, loading_weights = gauss_legendre(order, 0.0, loading_upper)
    log_variances, variance_weights = gauss_legendre(
        order, log(variance_scale) - 10, log(variance_upper))
    variances = exp.(log_variances)
    # Changing from s to log(s) supplies a factor s, so the exponent is -shape.
    variance_weights .*= exp.(-variance_shape .* log_variances .- variance_scale ./ variances)
    loading_weights .*= exp.(-loading_nodes.^2 / (2loading_variance))

    if ar_order == 0
        ar_nodes, ar_weights = [0.0], [1.0]
    else
        ar_nodes, ar_weights = gauss_legendre(ar_quadrature_order, 0.0, 1.0)
        ar_weights .*= exp.(-ar_nodes.^2 / (2ar_prior_variance))
        if initial == :zero
            # With one date and zero presample values, the AR posterior is
            # its prior. All requested AR statistics depend only on its
            # second moment, so this integral factors out exactly.
            prior_mass = sum(ar_weights)
            prior_second = sum(ar_weights .* ar_nodes.^2) / prior_mass
            ar_nodes, ar_weights = [sqrt(prior_second)], [prior_mass]
        end
    end
    process_variances = initial == :stationary ? 1 ./ (1 .- ar_nodes.^2) : ones(length(ar_nodes))

    mass = 0.0
    totals = zeros(length(statistics))
    for (j, b) in enumerate(loading_nodes), (k, s) in enumerate(variances)
        base_weight = loading_weights[j] * variance_weights[k]
        for u in eachindex(ar_nodes), z in eachindex(ar_nodes)
            vf, ve = process_variances[u], process_variances[z]
            d = v + b^2 * vf + s * ve
            weight = base_weight * ar_weights[u] * ar_weights[z] * exp(-y^2 / (2d)) / sqrt(d)
            a_mean = v * y / d
            f_mean = b * vf * y / d
            f_second = vf * (v + s * ve) / d + f_mean^2
            values = (a_mean, b, b^2, s, 1 / s, f_mean, f_second,
                      b * f_mean, ar_nodes[u]^2, ar_nodes[z]^2,
                      ar_nodes[u]^2 * b^2, ar_nodes[z]^2 * s)
            mass += weight
            for i in eachindex(totals)
                totals[i] += weight * values[i]
            end
        end
    end
    return (mass=mass, moments=NamedTuple{statistics}(Tuple(totals / mass)))
end

# Quantiles are obtained from new integrals up to each proposed threshold,
# rather than from the discrete quadrature nodes' cumulative weights.
function posterior_quantile(y, probability, parameter; kwargs...)
    parameter in (:loading, :variance) || throw(ArgumentError("unknown parameter"))
    reference = one_date(y; kwargs...)
    loading_variance = coefficient_variances(kwargs[:beta_prior_variance])[2]
    lower, upper = 0.0, parameter == :loading ? 9sqrt(loading_variance) :
                                              kwargs[:variance_scale] * exp(9)
    for _ in 1:45
        midpoint = (lower + upper) / 2
        partial = parameter == :loading ? one_date(y; loading_upper=midpoint, kwargs...) :
                                          one_date(y; variance_upper=midpoint, kwargs...)
        if partial.mass / reference.mass < probability
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return (lower + upper) / 2
end

const hierarchical_statistics = (:intercept, :loading1, :loading2,
    :loading1_squared, :loading2_squared, :variance, :precision, :factor1,
    :factor2, :factor1_squared, :factor2_squared, :factor_cross,
    :contribution1, :contribution2, :loading_cross)

# One observed series can load on two factors belonging to different levels.
# Integrating the intercept and both AR(0) factors gives
# y | b1,b2,s ~ N(0, v_a+b1²+b2²+s). This checks the actual hierarchical joint
# posterior even though the data alone cannot identify the two factors.
function two_factors(y; beta_prior_variance, variance_shape, variance_scale,
                     loading_order=70, variance_order=120)
    va, v1, v2 = beta_prior_variance
    b1_nodes, b1_weights = gauss_legendre(loading_order, 0.0, 9sqrt(v1))
    b2_nodes, b2_weights = gauss_legendre(loading_order, 0.0, 9sqrt(v2))
    log_variances, s_weights = gauss_legendre(
        variance_order, log(variance_scale) - 10, log(variance_scale) + 9)
    variances = exp.(log_variances)
    b1_weights .*= exp.(-b1_nodes.^2 / (2v1))
    b2_weights .*= exp.(-b2_nodes.^2 / (2v2))
    s_weights .*= exp.(-variance_shape .* log_variances .- variance_scale ./ variances)
    mass, totals = 0.0, zeros(length(hierarchical_statistics))
    for (i, b1) in enumerate(b1_nodes), (j, b2) in enumerate(b2_nodes), (k, s) in enumerate(variances)
        d = va + b1^2 + b2^2 + s
        weight = b1_weights[i] * b2_weights[j] * s_weights[k] * exp(-y^2 / (2d)) / sqrt(d)
        f1, f2 = b1 * y / d, b2 * y / d
        values = (va * y / d, b1, b2, b1^2, b2^2, s, 1 / s, f1, f2,
                  1 - b1^2 / d + f1^2, 1 - b2^2 / d + f2^2,
                  -b1 * b2 / d + f1 * f2, b1 * f1, b2 * f2, b1 * b2)
        mass += weight
        for t in eachindex(totals)
            totals[t] += weight * values[t]
        end
    end
    return (mass=mass, moments=NamedTuple{hierarchical_statistics}(Tuple(totals / mass)))
end

const two_date_statistics = (:intercept, :loading, :loading_squared, :variance,
    :precision, :factor1, :factor2, :contribution1, :contribution2,
    :contribution_cross, :factor_ar, :error_ar, :factor_ar_squared,
    :error_ar_squared, :ar_cross, :factor_ar_loading_squared, :error_ar_variance)

# For two dates, the AR(1) covariance is known in closed form:
# zero presample: [1 ψ; ψ 1+ψ²]; stationary: [1 ψ; ψ 1]/(1-ψ²).
# After integrating a and the complete factor path, y has covariance
# D = v_a*ones(2,2) + b²*C_f + s*C_e. Every inverse and determinant below is
# the scalar 2×2 formula, independent of state-space and AR-whitening code.
function two_dates(y; beta_prior_variance, variance_shape, variance_scale,
                   ar_prior_variance, initial, order=100, ar_quadrature_order=40)
    initial in (:zero, :stationary) || throw(ArgumentError("unsupported initial condition"))
    va, vb = coefficient_variances(beta_prior_variance)
    b_nodes, b_weights = gauss_legendre(order, 0.0, 9sqrt(vb))
    logs, s_weights = gauss_legendre(order, log(variance_scale) - 10, log(variance_scale) + 9)
    variances = exp.(logs)
    b_weights .*= exp.(-b_nodes.^2 / (2vb))
    s_weights .*= exp.(-variance_shape .* logs .- variance_scale ./ variances)
    ar_nodes, ar_weights = gauss_legendre(ar_quadrature_order, -1.0, 1.0)
    ar_weights .*= exp.(-ar_nodes.^2 / (2ar_prior_variance))
    mass, totals = 0.0, zeros(length(two_date_statistics))
    y1, y2 = y
    for (i, b) in enumerate(b_nodes), (j, s) in enumerate(variances)
        for (k, ψ) in enumerate(ar_nodes), (l, φ) in enumerate(ar_nodes)
            if initial == :stationary
                cf11, ce11 = 1 / (1 - ψ^2), 1 / (1 - φ^2)
                cf12, cf22 = ψ * cf11, cf11
                ce12, ce22 = φ * ce11, ce11
            else
                cf11, cf12, cf22 = 1.0, ψ, 1 + ψ^2
                ce11, ce12, ce22 = 1.0, φ, 1 + φ^2
            end
            d11 = va + b^2 * cf11 + s * ce11
            d12 = va + b^2 * cf12 + s * ce12
            d22 = va + b^2 * cf22 + s * ce22
            determinant = d11 * d22 - d12^2
            u1, u2 = (d22 * y1 - d12 * y2) / determinant, (d11 * y2 - d12 * y1) / determinant
            weight = b_weights[i] * s_weights[j] * ar_weights[k] * ar_weights[l] *
                     exp(-(y1 * u1 + y2 * u2) / 2) / sqrt(determinant)
            f1, f2 = b * (cf11 * u1 + cf12 * u2), b * (cf12 * u1 + cf22 * u2)
            # Conditional factor covariance C_f - b²*C_f*inv(D)*C_f.
            factor_covariance = cf12 - b^2 *
                (cf11 * (d22 * cf12 - d12 * cf22) +
                 cf12 * (d11 * cf22 - d12 * cf12)) / determinant
            values = (va * (u1 + u2), b, b^2, s, 1 / s, f1, f2,
                      b * f1, b * f2, b^2 * (factor_covariance + f1 * f2),
                      ψ, φ, ψ^2, φ^2, ψ * φ, ψ * b^2, φ * s)
            mass += weight
            for t in eachindex(totals)
                totals[t] += weight * values[t]
            end
        end
    end
    return (mass=mass, moments=NamedTuple{two_date_statistics}(Tuple(totals / mass)))
end

function batch_summary(chains; batch_size=100)
    n, nchains = size(chains)
    nbatches = div(n, batch_size)
    nbatches >= 20 || throw(ArgumentError("use at least 20 batches per chain"))
    n == batch_size * nbatches || throw(ArgumentError("draw count must be a multiple of batch_size"))
    batches = [mean(view(chains, (batch_size * (b - 1) + 1):(batch_size * b), c))
               for b in 1:nbatches, c in 1:nchains]
    # Independent chains and batches longer than typical autocorrelation times
    # give a Monte Carlo standard error that accounts for serial dependence.
    chain_mcse = [std(view(batches, :, c)) / sqrt(nbatches) for c in 1:nchains]
    pooled_mcse = sqrt(sum(abs2, chain_mcse)) / nchains
    return (mean=mean(chains), mcse=pooled_mcse,
            chain_means=vec(mean(chains; dims=1)), chain_mcse=chain_mcse)
end

end
