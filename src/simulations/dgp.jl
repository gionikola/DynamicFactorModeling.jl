# Scale covariance matrices to correlations before checking their numerical
# rank. This preserves small-variance coordinates and avoids overflow when a
# valid covariance is near the largest representable Float64 value.
function _scaled_covariance(Σ::AbstractMatrix, name::AbstractString="covariance")
    size(Σ, 1) == size(Σ, 2) || throw(DimensionMismatch("$name must be square"))
    all(x -> x isa Real && isfinite(x), Σ) ||
        throw(ArgumentError("$name must contain finite real numbers"))
    covariance = Matrix{Float64}(Σ)
    all(isfinite, covariance) || throw(ArgumentError("$name must be finite in Float64"))
    variances = diag(covariance)
    all(>=(0), variances) || throw(ArgumentError("$name must have nonnegative diagonal entries"))
    for index in axes(covariance, 1)
        if iszero(variances[index]) &&
           !(all(iszero, covariance[index, :]) && all(iszero, covariance[:, index]))
            throw(ArgumentError("$name cannot correlate a zero-variance coordinate with another coordinate"))
        end
    end
    active = findall(>(0), variances)
    scales = sqrt.(variances[active])
    # Divide in two steps: forming scales * scales' can underflow or overflow.
    correlation = covariance[active, active] ./ scales ./ scales'
    all(isfinite, correlation) || throw(ArgumentError("$name must be positive semidefinite"))
    relative_tolerance = 100 * eps(Float64) * max(1, length(active))
    all(x -> abs(x) <= 1 + relative_tolerance, correlation) ||
        throw(ArgumentError("$name must be positive semidefinite"))
    tolerance = relative_tolerance * (isempty(active) ? 0.0 : opnorm(correlation, Inf))
    isapprox(correlation, correlation'; atol=tolerance, rtol=0) ||
        throw(ArgumentError("$name must be symmetric"))
    decomposition = eigen(Symmetric((correlation + correlation') / 2))
    all(λ -> λ >= -tolerance, decomposition.values) ||
        throw(ArgumentError("$name must be positive semidefinite"))
    return active, scales, decomposition, tolerance
end

function _covariance_root(Σ::AbstractMatrix, name::AbstractString="covariance")
    active, scales, decomposition, tolerance = _scaled_covariance(Σ, name)
    keep = decomposition.values .> tolerance
    root = zeros(size(Σ, 1), count(keep))
    root[active, :] = scales .* (decomposition.vectors[:, keep] * Diagonal(sqrt.(decomposition.values[keep])))
    return root
end

function _root_covariance(root)
    covariance = root * root'
    all(isfinite, covariance) || throw(ArgumentError("state covariance overflowed; rescale the model"))
    return covariance
end

function _finite_vector(values, n::Integer, name::AbstractString)
    values isa AbstractVector || throw(ArgumentError("$name must be a vector"))
    length(values) == n || throw(DimensionMismatch("$name must have length $n"))
    all(x -> x isa Real && isfinite(x), values) ||
        throw(ArgumentError("$name must contain finite real numbers"))
    converted = Vector{Float64}(values)
    all(isfinite, converted) || throw(ArgumentError("$name must be finite in Float64"))
    return converted
end

"""
    mvn([rng], μ, Σ[, n]; rng=Random.default_rng())

Draw a normal vector with mean `μ` and covariance `Σ`. Singular covariance
matrices are allowed. Scalar arguments produce a scalar draw. With `n`, return
an `n × length(μ)` matrix (one column for scalar arguments).
"""
function mvn(rng::AbstractRNG, μ::AbstractVector, Σ::AbstractMatrix)
    mean = _finite_vector(μ, size(Σ, 1), "mean")
    root = _covariance_root(Σ)
    return mean + root * randn(rng, size(root, 2))
end

function mvn(rng::AbstractRNG, μ::Real, variance::Real)
    isfinite(μ) || throw(ArgumentError("mean must be finite"))
    isfinite(variance) && variance >= 0 ||
        throw(ArgumentError("variance must be finite and nonnegative"))
    mean, converted_variance = Float64(μ), Float64(variance)
    isfinite(mean) && isfinite(converted_variance) ||
        throw(ArgumentError("mean and variance must be finite in Float64"))
    return mean + sqrt(converted_variance) * randn(rng)
end

function mvn(rng::AbstractRNG, μ::AbstractVector, Σ::AbstractMatrix, n::Integer)
    n >= 0 || throw(ArgumentError("number of draws must be nonnegative"))
    mean = _finite_vector(μ, size(Σ, 1), "mean")
    root = _covariance_root(Σ)
    return Matrix((mean .+ root * randn(rng, size(root, 2), n))')
end

function mvn(rng::AbstractRNG, μ::Real, variance::Real, n::Integer)
    n >= 0 || throw(ArgumentError("number of draws must be nonnegative"))
    isfinite(μ) || throw(ArgumentError("mean must be finite"))
    isfinite(variance) && variance >= 0 ||
        throw(ArgumentError("variance must be finite and nonnegative"))
    mean, converted_variance = Float64(μ), Float64(variance)
    isfinite(mean) && isfinite(converted_variance) ||
        throw(ArgumentError("mean and variance must be finite in Float64"))
    return mean .+ sqrt(converted_variance) .* randn(rng, n, 1)
end

mvn(μ::Union{Real,AbstractVector}, Σ::Union{Real,AbstractMatrix}; rng::AbstractRNG=Random.default_rng()) = mvn(rng, μ, Σ)
mvn(μ::Union{Real,AbstractVector}, Σ::Union{Real,AbstractMatrix}, n::Integer; rng::AbstractRNG=Random.default_rng()) = mvn(rng, μ, Σ, n)

"""
    Γinv([rng], ν, θ; rng=Random.default_rng())

Draw `θ / X`, where `X` is chi-squared with `ν` degrees of freedom. This is
`InverseGamma(ν / 2, θ / 2)`, with positive finite `ν` and `θ`.
"""
function Γinv(rng::AbstractRNG, ν::Real, θ::Real)
    isfinite(ν) && ν > 0 || throw(ArgumentError("degrees of freedom must be positive and finite"))
    isfinite(θ) && θ > 0 || throw(ArgumentError("scale must be positive and finite"))
    degrees, scale = Float64(ν), Float64(θ)
    isfinite(degrees) && degrees > 0 && isfinite(scale) && scale > 0 ||
        throw(ArgumentError("degrees of freedom and scale must be positive and finite in Float64"))
    return scale / rand(rng, Chisq(degrees))
end
Γinv(ν::Real, θ::Real; rng::AbstractRNG=Random.default_rng()) = Γinv(rng, ν, θ)

function validate_ssmodel(model::SSModel)
    H, A, F, μ, R, Q, Z = model.H, model.A, model.F, model.μ, model.R, model.Q, model.Z
    n, m = size(H)
    n > 0 && m > 0 || throw(ArgumentError("a state-space model needs observations and states"))
    size(F) == (m, m) || throw(DimensionMismatch("F must be $m × $m"))
    length(μ) == m || throw(DimensionMismatch("μ must have length $m"))
    size(R) == (n, n) || throw(DimensionMismatch("R must be $n × $n"))
    size(Q) == (m, m) || throw(DimensionMismatch("Q must be $m × $m"))
    size(A, 1) == n || throw(DimensionMismatch("A must have $n rows"))
    size(Z) == (size(A, 2), size(A, 2)) ||
        throw(DimensionMismatch("Z must match the number of columns of A"))
    for (name, values) in (("H", H), ("A", A), ("F", F), ("μ", μ))
        all(isfinite, values) || throw(ArgumentError("$name must contain finite numbers"))
    end
    for (name, covariance) in (("R", R), ("Q", Q), ("Z", Z))
        _scaled_covariance(covariance, name)
    end
    return nothing
end

function validate_hdfm(model::HDFM)
    L, n = model.nlevels, model.nvar
    L > 0 && n > 0 || throw(ArgumentError("nlevels and nvar must be positive"))
    length(model.nfactors) == L || throw(DimensionMismatch("nfactors must have length nlevels"))
    length(model.flags) == L || throw(DimensionMismatch("flags must have length nlevels"))
    length(model.varlags) == n || throw(DimensionMismatch("varlags must have length nvar"))
    size(model.fassign) == (n, L) || throw(DimensionMismatch("fassign must be nvar × nlevels"))
    size(model.varcoefs) == (n, L + 1) || throw(DimensionMismatch("varcoefs must be nvar × (nlevels + 1)"))
    all(>(0), model.nfactors) || throw(ArgumentError("nfactors must be positive"))
    all(>=(0), model.flags) && all(>=(0), model.varlags) ||
        throw(ArgumentError("lag orders must be nonnegative"))
    size(model.varlagcoefs, 1) == n && size(model.varlagcoefs, 2) >= maximum(model.varlags) ||
        throw(DimensionMismatch("varlagcoefs needs nvar rows and at least maximum(varlags) columns"))
    length(model.fcoefs) == L && length(model.fvars) == L ||
        throw(DimensionMismatch("fcoefs and fvars must have length nlevels"))
    length(model.varvars) == n || throw(DimensionMismatch("varvars must have length nvar"))
    for level in 1:L
        all(a -> 0 <= a <= model.nfactors[level], model.fassign[:, level]) ||
            throw(ArgumentError("factor assignments must be between zero and nfactors[level]"))
        size(model.fcoefs[level]) == (model.nfactors[level], model.flags[level]) ||
            throw(DimensionMismatch("fcoefs[level] must be nfactors[level] × flags[level]"))
        length(model.fvars[level]) == model.nfactors[level] ||
            throw(DimensionMismatch("fvars[level] must have nfactors[level] entries"))
    end
    for values in (model.varcoefs, model.varlagcoefs, model.fcoefs...)
        all(x -> x isa Real && isfinite(x), values) ||
            throw(ArgumentError("coefficients must contain finite real numbers"))
    end
    for values in (model.varvars, model.fvars...)
        all(x -> x isa Real && isfinite(x) && x >= 0, values) ||
            throw(ArgumentError("innovation variances must be finite and nonnegative"))
    end
    return nothing
end

"""
    createSSforHDFM(hdfm)

Return `(H, A, F, μ, R, Q, Z)` for a hierarchical factor model. Each series is
its intercept plus its assigned factor contributions and an independent AR
error. Factor innovations are independent of one another and of series errors.
An assignment of zero omits that level's contribution.

The state contains a constant equal to one, then one block per factor (by level
and factor), then one block per series error. Each block holds its current value
followed by its lags, with length `max(1, lag_order)`. Zero lag orders represent
white noise. Thus unequal lag orders need no padding. `R`, `A`, and `Z` are zero.
"""
function createSSforHDFM(hdfm::HDFM)
    validate_hdfm(hdfm)
    n = hdfm.nvar
    state_count = 1 + sum(hdfm.nfactors .* max.(1, hdfm.flags)) + sum(max.(1, hdfm.varlags))
    H = zeros(n, state_count)
    F = zeros(state_count, state_count)
    μ = zeros(state_count)
    Q = zeros(state_count, state_count)

    # Reset the constant to one on every transition; it has no uncertainty.
    H[:, 1] = hdfm.varcoefs[:, 1]
    μ[1] = 1
    first_state = 2
    for level in 1:hdfm.nlevels
        order = hdfm.flags[level]
        for factor in 1:hdfm.nfactors[level]
            for series in 1:n
                if hdfm.fassign[series, level] == factor
                    H[series, first_state] = hdfm.varcoefs[series, level + 1]
                end
            end
            _fill_ar_block!(F, Q, first_state, hdfm.fcoefs[level][factor, :], hdfm.fvars[level][factor])
            first_state += max(1, order)
        end
    end
    for series in 1:n
        order = hdfm.varlags[series]
        H[series, first_state] = 1
        coefficients = hdfm.varlagcoefs[series, 1:order]
        _fill_ar_block!(F, Q, first_state, coefficients, hdfm.varvars[series])
        first_state += max(1, order)
    end
    return H, zeros(n, 0), F, μ, zeros(n, n), Q, zeros(0, 0)
end

function _fill_ar_block!(F, Q, first_state, coefficients, variance)
    order = length(coefficients)
    F[first_state, first_state:(first_state + order - 1)] = coefficients
    for lag in 1:(order - 1)
        F[first_state + lag, first_state + lag - 1] = 1
    end
    Q[first_state, first_state] = variance
    return nothing
end

"""
    convertHDFMtoSS(hdfm)

Convert a hierarchical factor model to an [`SSModel`](@ref). See
[`createSSforHDFM`](@ref) for the state ordering.
"""
convertHDFMtoSS(hdfm::HDFM) = SSModel(createSSforHDFM(hdfm)...)

# Sum Q + F*Q*F' + ... by doubling the number of terms at each step.
# This avoids constructing a state_count² × state_count² linear system.
function _stationary_covariance(F, Q)
    covariance = copy(Q)
    transition = copy(F)
    for iteration in 1:100
        addition = transition * covariance * transition'
        updated = Matrix(Symmetric(covariance + addition))
        all(isfinite, updated) || throw(ArgumentError("stationary covariance overflowed"))
        active = findall(>(0), diag(updated))
        isempty(active) && return updated
        scales = sqrt.(diag(updated)[active])
        scaled_addition = addition[active, active] ./ scales ./ scales'
        scaled_transition = transition[active, active] ./ scales .* scales'
        contraction = all(isfinite, scaled_transition) ? opnorm(scaled_transition, 2) : Inf
        # Bound the omitted geometric tail in units of each state's variance.
        # A global norm could hide a small, slowly converging state variance.
        if contraction < 1 && opnorm(scaled_addition, Inf) / (1 - contraction^2) <= 1e-12
            return updated
        end
        covariance = updated
        transition = transition * transition
    end
    throw(ArgumentError("stationary covariance did not converge; provide initial_cov"))
end

function _initial_distribution(model::SSModel, initial_mean, initial_cov)
    m = length(model.μ)
    if initial_mean === nothing || initial_cov === nothing
        # A tiny margin keeps an exact unit root from being rounded below one.
        margin = m == 1 ? 0.0 : 100 * m * eps(Float64)
        maximum(abs, eigvals(model.F)) < 1 - margin ||
            throw(ArgumentError("default initialization requires numerically stable F; provide initial_mean and initial_cov"))
    end
    mean = initial_mean === nothing ? (I - model.F) \ model.μ :
        _finite_vector(initial_mean, m, "initial_mean")
    all(isfinite, mean) || throw(ArgumentError("initial mean overflowed; rescale the model"))
    covariance = initial_cov === nothing ? _stationary_covariance(model.F, model.Q) : initial_cov
    covariance isa AbstractMatrix || throw(ArgumentError("initial_cov must be a matrix"))
    size(covariance) == (m, m) || throw(DimensionMismatch("initial_cov must be $m × $m"))
    root = _covariance_root(covariance, "initial_cov")
    return mean, _root_covariance(root)
end

"""
    simulateSSModel([rng], num_obs, model; initial_mean=nothing,
                    initial_cov=nothing, initial_state=nothing,
                    rng=Random.default_rng())

Simulate `βₜ = μ + F*βₜ₋₁ + vₜ` and `yₜ = H*βₜ + A*zₜ + eₜ`, with
independent Gaussian innovations having covariances `Q`, `Z`, and `R`.
Return `(data_y, data_z, data_β)`, with observations in rows.

By default `β₀` is drawn from the stationary distribution, requiring all
eigenvalues of `F` to have magnitude below one (with a small numerical margin).
Supply `initial_mean` and
`initial_cov` for another distribution, or `initial_state` for a fixed `β₀`.
Every returned row, including the first, includes a transition and fresh noise.
"""
function simulateSSModel(rng::AbstractRNG, num_obs::Integer, model::SSModel;
                         initial_mean=nothing, initial_cov=nothing, initial_state=nothing)
    num_obs >= 0 || throw(ArgumentError("num_obs must be nonnegative"))
    validate_ssmodel(model)
    if initial_state === nothing
        mean, covariance = _initial_distribution(model, initial_mean, initial_cov)
        state = mvn(rng, mean, covariance)
    else
        initial_mean === nothing && initial_cov === nothing ||
            throw(ArgumentError("initial_state cannot be combined with initial_mean or initial_cov"))
        state = _finite_vector(initial_state, length(model.μ), "initial_state")
    end
    process_root = _covariance_root(model.Q)
    observation_root = _covariance_root(model.R)
    exogenous_root = _covariance_root(model.Z)
    data_y = zeros(num_obs, size(model.H, 1))
    data_z = zeros(num_obs, size(model.A, 2))
    data_β = zeros(num_obs, length(model.μ))
    for t in 1:num_obs
        state = model.μ + model.F * state + process_root * randn(rng, size(process_root, 2))
        exogenous = exogenous_root * randn(rng, size(exogenous_root, 2))
        measurement_noise = observation_root * randn(rng, size(observation_root, 2))
        observation = model.H * state + model.A * exogenous + measurement_noise
        all(isfinite, state) && all(isfinite, observation) ||
            throw(ArgumentError("simulated state or observation overflowed at row $t; rescale the model"))
        data_y[t, :] = observation
        data_z[t, :] = exogenous
        data_β[t, :] = state
    end
    return data_y, data_z, data_β
end

simulateSSModel(num_obs::Integer, model::SSModel; rng::AbstractRNG=Random.default_rng(), kwargs...) =
    simulateSSModel(rng, num_obs, model; kwargs...)
