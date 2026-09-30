# A covariance root has one column for each independent standard-normal
# disturbance. Keep these roots throughout filtering so exact constraints do
# not acquire spurious uncertainty from subtracting covariance matrices.
function _compress_covariance_root(root)
    all(isfinite, root) || throw(ArgumentError("state uncertainty overflowed; rescale the model"))
    m = size(root, 1)
    size(root, 2) == 0 && return zeros(m, 0)
    scales = [norm(view(root, row, :)) for row in 1:m]
    all(isfinite, scales) || throw(ArgumentError("state uncertainty overflowed; rescale the model"))
    active = findall(>(0), scales)
    isempty(active) && return zeros(m, 0)
    standardized = root[active, :] ./ scales[active]
    decomposition = svd(standardized; full=false)
    tolerance = 100 * eps(Float64) * max(size(standardized)...) * maximum(decomposition.S)
    keep = decomposition.S .> tolerance
    compressed = zeros(m, count(keep))
    compressed[active, :] = scales[active] .* (decomposition.U[:, keep] .* decomposition.S[keep]')
    return compressed
end

# If x = mean + B*ξ and the observed residual is C*ξ, with ξ ~ N(0,I),
# an SVD of C splits ξ into observed directions and independent unobserved
# directions. The latter directly give the conditional covariance root.
function _joint_conditioning_parameters(state_map, observation_map)
    m, n = size(state_map, 1), size(observation_map, 1)
    n == 0 && return zeros(m, 0), _compress_covariance_root(state_map), zeros(0, 0)
    all(isfinite, observation_map) || throw(ArgumentError("observation uncertainty overflowed; rescale the model"))
    size(observation_map, 2) == 0 && return zeros(m, n), zeros(m, 0), Matrix{Float64}(I, n, n)
    scales = [norm(view(observation_map, row, :)) for row in 1:n]
    all(isfinite, scales) || throw(ArgumentError("observation uncertainty overflowed; rescale the model"))
    scales[iszero.(scales)] .= 1
    standardized = observation_map ./ scales
    decomposition = svd(standardized; full=true)
    tolerance = 100 * eps(Float64) * max(size(standardized)...) * maximum(decomposition.S)
    rank = count(>(tolerance), decomposition.S)
    observed_directions = decomposition.V[:, 1:rank]
    # Solve in the same equilibrated units used to determine rank. This
    # preserves independent observations whose physical units differ greatly.
    inverse_observations = (decomposition.U[:, 1:rank]' ./ decomposition.S[1:rank]) ./ scales'
    gain = state_map * observed_directions * inverse_observations
    conditional_root = _compress_covariance_root(state_map * decomposition.V[:, (rank + 1):end])
    constraints = decomposition.U[:, (rank + 1):end] ./ scales
    for column in axes(constraints, 2)
        constraints[:, column] ./= norm(constraints[:, column])
    end
    return gain, conditional_root, constraints
end

function _conditioning_parameters(state_root, loading, noise_root)
    state_map = hcat(state_root, zeros(size(state_root, 1), size(noise_root, 2)))
    observation_map = hcat(loading * state_root, noise_root)
    return _joint_conditioning_parameters(state_map, observation_map)
end

# A constrained recursion can amplify rounding in otherwise consistent data.
# Recheck a problematic prefix jointly, before declaring it impossible.
# The regular path remains linear in the number of observation times.
function _recover_observation_prefix(data_y, offsets, model, initial_mean, initial_root,
                                     process_root, measurement_root, last_time; whole_path=false)
    initial_count, process_count = size(initial_root, 2), size(process_root, 2)
    measurement_count = size(measurement_root, 2)
    noise_count = initial_count + last_time * (process_count + measurement_count)
    observed_count = count(!ismissing, data_y[1:last_time, :])
    output_count = length(initial_mean) * (whole_path ? last_time : 1)
    max(noise_count^2, observed_count^2, observed_count * noise_count,
        output_count * noise_count, output_count * observed_count) <= 4_000_000 ||
        throw(ArgumentError("exact constraints are numerically ill-conditioned; rescale the model"))
    state_map = zeros(length(initial_mean), noise_count)
    state_map[:, 1:initial_count] = initial_root
    observation_map = zeros(observed_count, noise_count)
    observed_values, predicted_values = zeros(observed_count), zeros(observed_count)
    prediction_magnitude = zeros(observed_count)
    history_map = whole_path ? zeros(output_count, noise_count) : zeros(0, 0)
    history_mean = whole_path ? zeros(output_count) : Float64[]
    mean = copy(initial_mean)
    next_row = 1
    for t in 1:last_time
        mean_magnitude = abs.(model.μ) + abs.(model.F) * abs.(mean)
        mean = model.μ + model.F * mean
        state_map = model.F * state_map
        first_noise = initial_count + (t - 1) * (process_count + measurement_count) + 1
        state_map[:, first_noise:(first_noise + process_count - 1)] = process_root
        observed = findall(!ismissing, data_y[t, :])
        rows = next_row:(next_row + length(observed) - 1)
        observation_map[rows, :] = model.H[observed, :] * state_map
        noise_columns = (first_noise + process_count):(first_noise + process_count + measurement_count - 1)
        observation_map[rows, noise_columns] = measurement_root[observed, :]
        observed_values[rows] = Float64.(data_y[t, observed])
        predicted_values[rows] = model.H[observed, :] * mean + offsets[t, observed]
        prediction_magnitude[rows] = abs.(model.H[observed, :]) * mean_magnitude + abs.(offsets[t, observed])
        if whole_path
            state_rows = ((t - 1) * length(mean) + 1):(t * length(mean))
            history_map[state_rows, :] = state_map
            history_mean[state_rows] = mean
        end
        next_row += length(observed)
    end
    if whole_path
        state_map, mean = history_map, history_mean
    end
    gain, root, constraints = _joint_conditioning_parameters(state_map, observation_map)
    residual = observed_values - predicted_values
    discrepancy, tolerance, _ = _constraint_discrepancies(constraints, observed_values, predicted_values; prediction_magnitude)
    all(discrepancy .<= tolerance) ||
        throw(ArgumentError("observations through row $last_time violate a deterministic model constraint"))
    conditioned_mean = mean + gain * residual
    all(isfinite, conditioned_mean) || throw(ArgumentError("conditioned state overflowed; rescale the model"))
    return conditioned_mean, root
end

# Assess each exact equation in its own units. An unrelated large observation
# must not hide a violation in a small or deterministic coordinate.
function _constraint_discrepancies(constraints, observed, predicted; prediction_magnitude=abs.(predicted))
    discrepancy = abs.(constraints' * (observed - predicted))
    magnitude = abs.(constraints') * (abs.(observed) + prediction_magnitude)
    tolerance = (1000 * eps(Float64) * max(1, length(observed))) .* magnitude
    all(isfinite, discrepancy) && all(isfinite, tolerance) && all(isfinite, magnitude) ||
        throw(ArgumentError("deterministic constraint check overflowed; rescale the model"))
    return discrepancy, tolerance, magnitude
end

function _observation_offsets(data_y::AbstractMatrix, model::SSModel, data_z)
    T, n = size(data_y)
    n == size(model.H, 1) || throw(DimensionMismatch("data_y columns must match H rows"))
    all(x -> ismissing(x) || (x isa Real && isfinite(x) && isfinite(Float64(x))), data_y) ||
        throw(ArgumentError("data_y must contain real numbers finite in Float64, or missing"))
    if data_z === nothing
        all(iszero, model.A) ||
            throw(ArgumentError("provide data_z when A is nonzero"))
        return zeros(T, n)
    end
    data_z isa AbstractMatrix || throw(ArgumentError("data_z must be a matrix"))
    size(data_z) == (T, size(model.A, 2)) ||
        throw(DimensionMismatch("data_z must have $T rows and size(A, 2) columns"))
    all(x -> x isa Real && isfinite(x), data_z) ||
        throw(ArgumentError("data_z must contain finite real numbers"))
    regressors = Matrix{Float64}(data_z)
    all(isfinite, regressors) || throw(ArgumentError("data_z must be finite in Float64"))
    offsets = regressors * model.A'
    all(isfinite, offsets) || throw(ArgumentError("A * data_z overflowed; rescale the inputs"))
    return offsets
end

function _kalman_filter(data_y::AbstractMatrix, model::SSModel;
                        data_z=nothing, initial_mean=nothing, initial_cov=nothing)
    validate_ssmodel(model)
    offsets = _observation_offsets(data_y, model, data_z)
    mean, covariance = _initial_distribution(model, initial_mean, initial_cov)
    root = _covariance_root(covariance)
    initial_state_mean, initial_state_root = copy(mean), copy(root)
    process_root = _covariance_root(model.Q)
    measurement_root = _covariance_root(model.R)
    T, n = size(data_y)
    m = length(mean)
    predicted_y = zeros(T, n)
    predicted_states = zeros(T, m)
    filtered_states = zeros(T, m)
    predicted_covariances = Vector{Matrix{Float64}}(undef, T)
    filtered_covariances = Vector{Matrix{Float64}}(undef, T)
    filtered_roots = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        predicted_mean_magnitude = abs.(model.μ) + abs.(model.F) * abs.(mean)
        predicted_mean = model.μ + model.F * mean
        predicted_root = _compress_covariance_root(hcat(model.F * root, process_root))
        predicted_observation = model.H * predicted_mean + offsets[t, :]
        all(isfinite, predicted_mean) && all(isfinite, predicted_observation) ||
            throw(ArgumentError("state or observation prediction overflowed at row $t; rescale the model"))
        predicted_y[t, :] = predicted_observation
        predicted_states[t, :] = predicted_mean
        predicted_covariances[t] = _root_covariance(predicted_root)

        observed = findall(x -> !ismissing(x), data_y[t, :])
        if isempty(observed)
            mean, root = predicted_mean, predicted_root
        else
            loading = model.H[observed, :]
            noise_root = _covariance_root(model.R[observed, observed])
            innovation = Float64.(data_y[t, observed]) - predicted_observation[observed]
            gain, root, constraints = _conditioning_parameters(predicted_root, loading, noise_root)

            # Conditioning must not silently ignore impossible exact observations.
            observed_values = Float64.(data_y[t, observed])
            prediction_magnitude = abs.(loading) * predicted_mean_magnitude + abs.(offsets[t, observed])
            discrepancy, tolerance, magnitude = _constraint_discrepancies(
                constraints, observed_values, predicted_observation[observed]; prediction_magnitude)
            if any(discrepancy .> tolerance)
                all(discrepancy .<= sqrt(eps(Float64)) .* magnitude) ||
                    throw(ArgumentError("observation at row $t violates a deterministic model constraint"))
                mean, root = _recover_observation_prefix(data_y, offsets, model,
                    initial_state_mean, initial_state_root, process_root, measurement_root, t)
            else
                mean = predicted_mean + gain * innovation
            end
            all(isfinite, mean) || throw(ArgumentError("filtered state overflowed at row $t; rescale the inputs"))
        end
        filtered_states[t, :] = mean
        filtered_roots[t] = root
        filtered_covariances[t] = _root_covariance(root)
    end
    return (; predicted_y, predicted_states, filtered_states, filtered_roots,
             predicted_covariances, filtered_covariances, offsets, process_root,
             initial_state_mean, initial_state_root, measurement_root)
end

"""
    kalmanFilter(data_y, model; data_z=nothing, initial_mean=nothing, initial_cov=nothing)

Filter a linear Gaussian state-space model. Rows are times and columns are
series. `missing` observations are skipped. Supply the observed exogenous
variables as `data_z` whenever `model.A` is nonzero.

Return `(predicted_y, filtered_states, predicted_covariances, filtered_covariances)`.
`predicted_y[t, :]` uses observations strictly before `t`; the filtered states
and covariances also use row `t`. Covariances are vectors of matrices, one per row.

`initial_mean` and `initial_cov` describe `β₀`, before the first transition.
Each omitted value uses its stationary counterpart, requiring stable `F`.
Singular covariances and exact observations are supported; observations that
contradict a deterministic equation raise an error.
"""
function kalmanFilter(data_y::AbstractMatrix, model::SSModel; kwargs...)
    result = _kalman_filter(data_y, model; kwargs...)
    return result.predicted_y, result.filtered_states,
           result.predicted_covariances, result.filtered_covariances
end

# Parameters of p(β_t | β_{t+1}, y_1,...,y_t), using the same independent
# disturbance calculation as the observation update.
function _backward_parameters(filtered_root, process_root, model)
    gain, conditional_root, _ = _conditioning_parameters(filtered_root, model.F, process_root)
    return gain, conditional_root
end

# Exact transitions can make backward inversion numerically unstable even
# when the whole observation problem is well conditioned. Track how rounding
# would accumulate through the actual gains, in common state units. This is
# independent of observation values and permits cancellation within each gain.
function _backward_distributions(result, model)
    T, m = size(result.filtered_states)
    gains = Vector{Matrix{Float64}}(undef, max(0, T - 1))
    roots = Vector{Matrix{Float64}}(undef, max(0, T - 1))
    T < 2 && return gains, roots, false
    scales = sqrt.([maximum(P[i, i] for P in result.predicted_covariances) for i in 1:m])
    scales[iszero.(scales)] .= 1
    rounding_covariance = Matrix{Float64}(I, m, m)
    amplification_limit = inv(100 * m * sqrt(eps(Float64)))
    for t in (T - 1):-1:1
        gains[t], roots[t] = _backward_parameters(result.filtered_roots[t], result.process_root, model)
        scaled_gain = gains[t] ./ scales .* scales'
        rounding_covariance = Matrix{Float64}(I, m, m) +
            scaled_gain * rounding_covariance * scaled_gain'
        if !all(isfinite, rounding_covariance) || maximum(diag(rounding_covariance)) > amplification_limit^2
            return gains, roots, true
        end
    end
    return gains, roots, false
end

function _joint_smoothed_distribution(data_y, model, result)
    return _recover_observation_prefix(data_y, result.offsets, model,
        result.initial_state_mean, result.initial_state_root, result.process_root,
        result.measurement_root, size(data_y, 1); whole_path=true)
end

function _smoothed_observations(states, model, offsets)
    observations = states * model.H' + offsets
    all(isfinite, observations) || throw(ArgumentError("smoothed observations overflowed; rescale the model"))
    return observations
end

"""
    kalmanSmoother(data_y, model; data_z=nothing, initial_mean=nothing, initial_cov=nothing)

Use all observations to estimate each state with the Rauch–Tung–Striebel
smoother. Return `(smoothed_y, smoothed_states, smoothed_covariances)`.
`smoothed_y` is the fitted signal `H*E[βₜ | data] + A*zₜ`, excluding measurement
noise. Initialization, missing values, and singular covariances follow
[`kalmanFilter`](@ref).

If deterministic backward transitions amplify rounding excessively, a joint
Gaussian calculation replaces the recursion. That exceptional calculation is
limited to four million entries in each required dense matrix; larger cases
raise an error asking for rescaling rather than return an unreliable result.
"""
function kalmanSmoother(data_y::AbstractMatrix, model::SSModel; kwargs...)
    result = _kalman_filter(data_y, model; kwargs...)
    states = copy(result.filtered_states)
    covariances = copy(result.filtered_covariances)
    roots = copy(result.filtered_roots)
    T, m = size(states)
    if all(ismissing, data_y)
        return _smoothed_observations(states, model, result.offsets), states, covariances
    end
    gains, conditional_roots, needs_joint_calculation = _backward_distributions(result, model)
    if needs_joint_calculation
        mean, root = _joint_smoothed_distribution(data_y, model, result)
        states = Matrix(reshape(mean, m, T)')
        for t in 1:T
            rows = ((t - 1) * m + 1):(t * m)
            covariances[t] = _root_covariance(root[rows, :])
        end
        return _smoothed_observations(states, model, result.offsets), states, covariances
    end
    for t in (T - 1):-1:1
        gain, conditional_root = gains[t], conditional_roots[t]
        states[t, :] += gain * (states[t + 1, :] - result.predicted_states[t + 1, :])
        all(isfinite, states[t, :]) || throw(ArgumentError("smoothed state overflowed at row $t; rescale the model"))
        roots[t] = _compress_covariance_root(hcat(conditional_root, gain * roots[t + 1]))
        covariances[t] = _root_covariance(roots[t])
    end
    smoothed_y = _smoothed_observations(states, model, result.offsets)
    return smoothed_y, states, covariances
end

"""
    KNFactorSampler([rng], data_y, model; rng=Random.default_rng(), kwargs...)

Draw one complete state path from its Gaussian distribution conditional on all
observations. The forward filter and backward sampler retain dependence across
times, including exact companion-state lag relationships. Returns a matrix with
times in rows and states in columns. Keywords are those of [`kalmanFilter`](@ref).
The numerical fallback and size limit are those of [`kalmanSmoother`](@ref).
"""
function KNFactorSampler(rng::AbstractRNG, data_y::AbstractMatrix, model::SSModel; kwargs...)
    result = _kalman_filter(data_y, model; kwargs...)
    T = size(data_y, 1)
    states = zeros(T, length(model.μ))
    T == 0 && return states
    if all(ismissing, data_y)
        state = result.initial_state_mean + result.initial_state_root *
            randn(rng, size(result.initial_state_root, 2))
        for t in 1:T
            state = model.μ + model.F * state + result.process_root * randn(rng, size(result.process_root, 2))
            all(isfinite, state) || throw(ArgumentError("sampled state overflowed at row $t; rescale the model"))
            states[t, :] = state
        end
        return states
    end
    gains, conditional_roots, needs_joint_calculation = _backward_distributions(result, model)
    if needs_joint_calculation
        mean, root = _joint_smoothed_distribution(data_y, model, result)
        draw = mean + root * randn(rng, size(root, 2))
        all(isfinite, draw) || throw(ArgumentError("sampled state overflowed; rescale the model"))
        return Matrix(reshape(draw, length(model.μ), T)')
    end
    terminal_root = result.filtered_roots[T]
    states[T, :] = result.filtered_states[T, :] + terminal_root * randn(rng, size(terminal_root, 2))
    all(isfinite, states[T, :]) || throw(ArgumentError("sampled state overflowed at row $T; rescale the model"))
    for t in (T - 1):-1:1
        gain, conditional_root = gains[t], conditional_roots[t]
        mean = result.filtered_states[t, :] +
               gain * (states[t + 1, :] - result.predicted_states[t + 1, :])
        all(isfinite, mean) || throw(ArgumentError("conditional state mean overflowed at row $t; rescale the model"))
        states[t, :] = mean + conditional_root * randn(rng, size(conditional_root, 2))
        all(isfinite, states[t, :]) || throw(ArgumentError("sampled state overflowed at row $t; rescale the model"))
    end
    return states
end

KNFactorSampler(data_y::AbstractMatrix, model::SSModel; rng::AbstractRNG=Random.default_rng(), kwargs...) =
    KNFactorSampler(rng, data_y, model; kwargs...)
