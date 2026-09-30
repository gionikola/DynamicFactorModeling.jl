"""
    variance_decomposition(data, factors, loadings; intercepts=zeros(nseries))

Account for each series' sample variance. `factors` is time × factor and
`loadings` is series × factor. Return a named tuple with `factors` (series ×
factor variance shares), `residual` (residual variance shares), `covariance`
(the combined cross-covariance share), and `total` (observed variances).

Each factor share is `var(loading * factor) / var(data)`. The residual is
`data - intercepts' - factors * loadings'`. Shares plus `covariance` sum to
one. Correlation can make the covariance share negative, or an individual
share exceed one. Constant observed series have undefined shares and raise
an error. Constant offsets, including `intercepts`, do not affect the shares.
These are sample accounting quantities, not causal effects.
"""
function variance_decomposition(data::AbstractMatrix{<:Real},
                                factors::AbstractMatrix{<:Real},
                                loadings::AbstractMatrix{<:Real};
                                intercepts=zeros(size(data, 2)))
    total, contributions, residual_variances =
        _sample_variance_components(data, factors, loadings, intercepts)
    all(>(0), total) ||
        throw(ArgumentError("observed variances must be positive and representable in Float64"))
    factor_shares = contributions ./ total
    residual_shares = residual_variances ./ total
    covariance_share = 1 .- vec(sum(factor_shares; dims=2)) .- residual_shares
    all(isfinite, factor_shares) && all(isfinite, residual_shares) && all(isfinite, covariance_share) ||
        throw(ArgumentError("components are too large relative to observed variance for finite shares"))
    return (factors=factor_shares, residual=residual_shares,
            covariance=covariance_share, total=total)
end

# Normalize before squaring or summing, so representable variances do not fail
# merely because an intermediate sum of squares overflows.
function _component_variance(values)
    all(isfinite, values) || throw(ArgumentError("variance calculation overflowed; rescale the inputs"))
    scale = maximum(abs, values)
    iszero(scale) && return 0.0
    standard_deviation = scale * std(values / scale)
    variance = standard_deviation^2
    isfinite(variance) || throw(ArgumentError("variance calculation overflowed; rescale the inputs"))
    return variance
end

function _sample_variance_components(data, factors, loadings, intercepts)
    T, N = size(data)
    T >= 2 && N > 0 || throw(ArgumentError("need at least two observations and one series"))
    size(factors, 1) == T || throw(DimensionMismatch("factor and data row counts must match"))
    size(loadings) == (N, size(factors, 2)) || throw(DimensionMismatch("loadings must be series × factor"))
    intercepts isa AbstractVector{<:Real} || throw(ArgumentError("intercepts must be a real vector"))
    length(intercepts) == N || throw(DimensionMismatch("intercepts must have one value per series"))
    # Promote before differences and squares: integer arithmetic can overflow.
    data = Matrix{Float64}(data)
    factors = Matrix{Float64}(factors)
    loadings = Matrix{Float64}(loadings)
    intercepts = Vector{Float64}(intercepts)
    all(isfinite, data) && all(isfinite, factors) && all(isfinite, loadings) &&
        all(isfinite, intercepts) || throw(ArgumentError("all inputs must be finite"))

    # Variance is unaffected by constant offsets, including the intercepts.
    # Remove them before multiplying or subtracting fitted components. Otherwise
    # a large intercept can erase genuine residual variation through roundoff.
    differences = data .- data[1:1,:]
    factor_differences = factors .- factors[1:1,:]
    residuals = copy(differences)
    total = [_component_variance(differences[:, i]) for i in 1:N]
    contributions = zeros(N, size(factors, 2))
    for i in 1:N, j in axes(factors, 2)
        iszero(loadings[i,j]) && continue
        # Form the actual component first. Computing loading² * var(factor)
        # separately can overflow and underflow even when their product is finite.
        component = loadings[i, j] .* factor_differences[:, j]
        contributions[i, j] = _component_variance(component)
        residuals[:, i] -= component
    end
    residual_variances = [_component_variance(residuals[:, i]) for i in 1:N]
    return total, contributions, residual_variances
end

"""
    vardecomp2level(data, factors, betas, factorassign)

Return normalized marginal variance contributions of global and group
factors (series × 2). `factors` contains the global factor followed by group
factors; `betas` has rows `[intercept, global loading, group loading]`.
`factorassign` contains global assignment 1 and group numbers (zero allowed).

The denominator is the sum of the two component variances and the residual
variance. This treats components as uncorrelated and deliberately excludes
sample cross-covariances. Use [`variance_decomposition`](@ref) for an exact
sample-variance account that retains their combined contribution. Unlike that
account, this normalization can be defined for constant data if varying
components cancel; at least one component must have positive variance.
"""
function vardecomp2level(data::AbstractMatrix{<:Real}, factors::AbstractMatrix{<:Real},
                         betas::AbstractMatrix{<:Real}, factorassign::AbstractMatrix{<:Integer})
    N = size(data, 2)
    size(betas) == (N, 3) || throw(DimensionMismatch("betas must be nseries × 3"))
    all(isfinite, betas) || throw(ArgumentError("betas must be finite"))
    size(factorassign) == (N, 2) || throw(DimensionMismatch("factorassign must be nseries × 2"))
    size(factors, 2) >= 1 || throw(ArgumentError("factors must include the global factor"))
    all(==(1), factorassign[:,1]) || throw(ArgumentError("global factor assignment must be 1"))
    loadings = zeros(N, size(factors, 2))
    loadings[:,1] = betas[:,2]
    for i in 1:N
        group = factorassign[i,2]
        0 <= group < size(factors, 2) || throw(ArgumentError("group assignment is outside the factor range"))
        group == 0 || (loadings[i,1+group] = betas[i,3])
    end
    _, contributions, residual_variances =
        _sample_variance_components(data, factors, loadings, betas[:,1])
    result = zeros(N, 2)
    for i in 1:N
        scale = max(maximum(contributions[i, :]), residual_variances[i])
        scale > 0 || throw(ArgumentError("normalized shares need at least one varying component"))
        denominator = sum(contributions[i, :] / scale) + residual_variances[i] / scale
        result[i,1] = (contributions[i,1] / scale) / denominator
        group = factorassign[i,2]
        group == 0 || (result[i,2] = (contributions[i,1+group] / scale) / denominator)
    end
    return result
end

"""
    variance_decomposition(model::HDFM)

Compute long-run variance shares from a stationary model's parameters. Return
`(factors, residual, covariance, total)` with the same layout as the data-based
method. Factors are ordered by level, then factor within level. Because model
innovations are independent, the covariance share is zero and factor and error
shares add to one. `total` contains each series' unconditional variance.
"""
function variance_decomposition(model::HDFM)
    state_space = convertHDFMtoSS(model)
    _, covariance = _initial_distribution(state_space, nothing, nothing)
    contributions = zeros(model.nvar, sum(model.nfactors))
    first_state, factor_column = 2, 1
    for level in 1:model.nlevels
        for factor in 1:model.nfactors[level]
            factor_sd = sqrt(covariance[first_state,first_state])
            contributions[:,factor_column] = (state_space.H[:,first_state] .* factor_sd).^2
            first_state += max(1, model.flags[level])
            factor_column += 1
        end
    end
    residual = zeros(model.nvar)
    for series in 1:model.nvar
        residual[series] = covariance[first_state,first_state]
        first_state += max(1, model.varlags[series])
    end
    total = vec(sum(contributions; dims=2)) + residual
    all(isfinite, total) || throw(ArgumentError("model variances overflowed; rescale the model"))
    all(>(0), total) || throw(ArgumentError("variance shares are undefined for deterministic series"))
    return (factors=contributions ./ total, residual=residual ./ total,
            covariance=zeros(model.nvar), total=total)
end
