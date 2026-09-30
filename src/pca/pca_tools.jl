# All public data matrices use time in rows and series in columns.
function _pca_data(data::AbstractMatrix{<:Real})
    size(data, 1) >= 2 || throw(ArgumentError("PCA needs at least two observations"))
    size(data, 2) > 0 || throw(ArgumentError("PCA needs at least one series"))
    y = Matrix{Float64}(data)
    all(isfinite, y) || throw(ArgumentError("PCA data must be finite in Float64"))
    return y
end

# Subtract a reference row first so constant columns center to exactly zero,
# even when their decimal value is not represented exactly in floating point.
function _center_pca_data(y)
    deviations = y .- y[1:1,:]
    all(isfinite, deviations) ||
        throw(ArgumentError("PCA centering overflowed; rescale the data"))
    scales = maximum(abs, deviations; dims=1)
    divisors = ifelse.(iszero.(scales), 1.0, scales)
    normalized = deviations ./ divisors
    offset = mean(normalized; dims=1)
    centered = (normalized .- offset) .* scales
    intercepts = vec(y[1:1,:] + scales .* offset)
    all(isfinite, centered) && all(isfinite, intercepts) ||
        throw(ArgumentError("PCA centering overflowed; rescale the data"))
    return centered, intercepts
end

"""
    firstComponentFactor(data)

Return `(factor, loadings)` for the best rank-one approximation of centered
`data`. The factor has mean zero and `sum(abs2, factor) / T == 1`; loadings
carry the scale. Its largest absolute loading is positive. Constant data
return zero factors and loadings. Columns are centered but not standardized.
"""
function firstComponentFactor(data::AbstractMatrix{<:Real})
    y = _pca_data(data)
    centered, _ = _center_pca_data(y)
    T, N = size(y)
    scale = maximum(abs, centered)
    if iszero(scale)
        return zeros(T), zeros(N)
    end
    # Singular values can exceed Float64 even when every entry and each
    # loading is representable. Work at unit scale and restore it afterward.
    decomposition = svd(centered / scale; full=false)
    factor = sqrt(T) .* decomposition.U[:,1]
    # Regress each original column on the factor. Scaling columns separately
    # preserves a small loading even if that column underflowed in the SVD.
    loadings = zeros(N)
    for series in 1:N
        column = centered[:, series]
        column_scale = maximum(abs, column)
        if column_scale > 0
            loadings[series] = (dot(column / column_scale, factor) / T) * column_scale
        end
    end
    all(isfinite, loadings) || throw(ArgumentError("PCA loadings overflowed; rescale the data"))
    if loadings[argmax(abs.(loadings))] < 0
        factor .*= -1
        loadings .*= -1
    end
    return factor, loadings
end

"""
    PCA1LevelEstimator(data[, settings::DFMStruct])

Fit one principal component to centered data. Return [`PCAResults`](@ref).
This is a deterministic least-squares factor estimate, with no AR estimation
or posterior draws. An optional `DFMStruct` is accepted for convenience;
its AR orders and MCMC settings do not affect PCA.
"""
function PCA1LevelEstimator(data::AbstractMatrix{<:Real})
    y = _pca_data(data)
    centered, intercepts = _center_pca_data(y)
    factor, loading = firstComponentFactor(y)
    factors, loadings = reshape(factor, :, 1), reshape(loading, :, 1)
    residuals = centered - factors * loadings'
    all(isfinite, residuals) || throw(ArgumentError("PCA fitted values overflowed; rescale the data"))
    return PCAResults(factors, loadings, intercepts, residuals)
end
PCA1LevelEstimator(data::AbstractMatrix{<:Real}, ::DFMStruct) = PCA1LevelEstimator(data)

"""
    PCA2LevelEstimator(data, settings::HDFMStruct)

Fit a global principal component, subtract its fitted contribution, and fit
one principal component to each group's remaining data. Requires two levels
and one global factor assigned to every series. A group assignment of zero
means no group factor. Return [`PCAResults`](@ref), global factor first.

This sequential PCA approximation is deterministic. It is neither a joint
hierarchical likelihood fit nor a Bayesian estimator. AR orders and MCMC
settings do not affect the result. Columns are centered, not standardized.
"""
function PCA2LevelEstimator(data::AbstractMatrix{<:Real}, settings::HDFMStruct)
    y = _pca_data(data)
    # Recheck array fields, which may have been edited since construction.
    settings = HDFMStruct(settings.nlevels, settings.nfactors, settings.factorassign,
                          settings.factorlags, settings.errorlags,
                          settings.ndraws, settings.burnin)
    settings.nlevels == 2 && settings.nfactors[1] == 1 ||
        throw(ArgumentError("two-level PCA requires one global factor and one group level"))
    size(settings.factorassign, 1) == size(y, 2) ||
        throw(DimensionMismatch("factorassign must have one row per data series"))
    all(==(1), settings.factorassign[:,1]) ||
        throw(ArgumentError("the global factor must be assigned to every series"))
    T, N = size(y)
    centered, intercepts = _center_pca_data(y)
    factors = zeros(T, 1 + settings.nfactors[2])
    loadings = zeros(N, size(factors, 2))
    factors[:,1], loadings[:,1] = firstComponentFactor(y)
    residuals = centered - factors[:,1] * loadings[:,1]'
    for group in 1:settings.nfactors[2]
        series = findall(==(group), settings.factorassign[:,2])
        isempty(series) && throw(ArgumentError("each group must contain at least one series"))
        factor, loading = firstComponentFactor(residuals[:,series])
        factors[:,1+group] = factor
        loadings[series,1+group] = loading
        residuals[:,series] .-= factor * loading'
    end
    all(isfinite, residuals) || throw(ArgumentError("PCA fitted values overflowed; rescale the data"))
    return PCAResults(factors, loadings, intercepts, residuals)
end
