module SimulationScenarios

using LinearAlgebra
using Random

export scenarios, generate_case, stationary_ar_covariance

# These are fixed parameter settings for repeated-data experiments. They are
# not draws from an estimation prior, so the study is not simulation-based
# calibration. No package simulator or state-space helper is used here.

function scenario(name, title, dates, nfactors, assignments, factor_ar, error_ar,
                  intercepts, loadings, error_variances;
                  initial=:stationary, estimator_kind=:single, notes="")
    return (; name, title, dates, nlevels=length(nfactors), nfactors, assignments,
            factor_ar, error_ar, intercepts, loadings, error_variances,
            initial, estimator_kind, notes)
end

"""
    scenarios()

Return six small, fixed-truth experiments. Factor columns are ordered by level,
then factor within level. Loadings have one column per factor; assignments use
factor numbers within each level. Every factor has innovation variance one and
a positive loading on its first assigned series.
"""
function scenarios()
    single_assignments = ones(Int, 4, 1)

    independent = scenario(
        :ar0, "Independent factor and errors", 40, [1], single_assignments,
        [Float64[]], [Float64[] for _ in 1:4],
        [0.8, -0.6, 0.4, -1.1], reshape([1.2, 0.8, -1.0, 1.4], 4, 1),
        [0.3, 0.45, 0.35, 0.25])

    strong = scenario(
        :strong_ar1, "Strong shared AR(1) signal", 60, [1], copy(single_assignments),
        [[0.7]], [[0.2], [-0.15], [0.4], [0.1]],
        [0.5, -1.0, 1.25, 0.1], reshape([1.3, 0.9, -1.1, 1.5], 4, 1),
        [0.2, 0.35, 0.25, 0.3])

    second_order = scenario(
        :mixed_ar2, "AR(2) factor and errors, zero presample", 48,
        [1], copy(single_assignments), [[0.7, -0.25]],
        [[0.4, -0.15], [-0.15, 0.1], [0.25, 0.05], [0.5, -0.2]],
        [-0.7, 0.4, 1.1, -0.1], reshape([1.1, 0.7, -0.9, 1.25], 4, 1),
        [0.35, 0.5, 0.3, 0.4]; initial=:zero,
        notes="Both generation and estimation must use fixed zero presample values.")

    weak = scenario(
        :weak_persistent, "Weak, persistent shared signal", 40,
        [1], copy(single_assignments), [[0.9]], [[0.5], [0.65], [0.25], [-0.2]],
        [0.3, -0.5, 0.8, -0.2], reshape([0.25, 0.3, -0.2, 0.35], 4, 1),
        [0.8, 1.0, 1.2, 0.9];
        notes="A short, noisy sample can leave the factor and its persistence weakly determined.")

    two_assignments = [1 1; 1 1; 1 1; 1 0; 1 2; 1 2; 1 2; 1 2]
    two_loadings = [
        0.9   0.8   0.0
        1.2  -0.6   0.0
       -0.7   0.9   0.0
        0.8   0.0   0.0
       -1.0   0.0   0.9
        1.1   0.0   0.7
        0.85  0.0  -0.8
       -0.9   0.0   1.0
    ]
    two_levels = scenario(
        :two_level, "Global and two group factors", 48, [1, 2], two_assignments,
        [[0.65], Float64[], Float64[]],
        [Float64[], [0.3], Float64[], [-0.2], [0.25], Float64[], [0.4, -0.1], Float64[]],
        [0.5, -0.75, 1.0, 0.0, -0.5, 0.25, 0.8, -0.3], two_loadings,
        [0.3, 0.4, 0.35, 0.3, 0.45, 0.35, 0.4, 0.3]; estimator_kind=:two_level,
        notes="The fourth series has no group factor; error AR orders differ across series.")

    three_assignments = [1 1 1; 1 1 1; 1 1 2; 1 1 2; 1 2 3; 1 2 3; 1 2 4; 1 2 4]
    three_loadings = [
        0.9   0.75   0.0   0.8   0.0   0.0   0.0
        1.1  -0.65   0.0  -0.65  0.0   0.0   0.0
       -0.8   0.85   0.0   0.0   0.7   0.0   0.0
        1.0   0.6    0.0   0.0   0.9   0.0   0.0
        0.85  0.0    0.8   0.0   0.0   0.85  0.0
       -0.9   0.0    0.65  0.0   0.0  -0.7   0.0
        1.15  0.0   -0.75  0.0   0.0   0.0   0.8
        0.7   0.0    0.9   0.0   0.0   0.0  -0.6
    ]
    three_levels = scenario(
        :three_level, "Global, regional, and local factors", 40,
        [1, 2, 4], three_assignments,
        [[0.75], [0.4], [-0.2], [0.15], [0.3], [-0.25], [0.5]],
        [Float64[] for _ in 1:8],
        [0.4, -0.6, 1.0, -0.2, 0.8, -1.0, 0.2, 0.5], three_loadings,
        [0.35, 0.3, 0.45, 0.4, 0.3, 0.4, 0.35, 0.5]; estimator_kind=:hierarchical,
        notes="Each local factor has only two indicators. Individual factors and loadings may be weakly identified; shared fitted signals are the primary recovery target.")

    return [independent, strong, second_order, weak, two_levels, three_levels]
end

"""
    stationary_ar_covariance(coefficients, innovation_variance=1)

Covariance of the `p` presample values `[x[0], x[-1], ..., x[1-p]]` of an AR(p).
Solve the small Yule–Walker equations for autocovariances directly, independently
of the package's companion-state covariance calculation.
"""
function stationary_ar_covariance(coefficients, innovation_variance=1.0)
    p = length(coefficients)
    p == 0 && return zeros(0, 0)
    equations = Matrix{Float64}(I, p + 1, p + 1)
    for lag in 0:p, j in 1:p
        equations[lag + 1, abs(lag - j) + 1] -= coefficients[j]
    end
    autocovariances = equations \ [innovation_variance; zeros(p)]
    covariance = [autocovariances[abs(i - j) + 1] for i in 1:p, j in 1:p]
    # A factorization also detects an invalid stationary covariance if a
    # scenario is edited to contain a nonstationary process.
    cholesky(Symmetric(covariance))
    return covariance
end

function simulate_ar(rng, coefficients, innovation_variance, dates, initial)
    p = length(coefficients)
    presample = if initial == :stationary && p > 0
        covariance = stationary_ar_covariance(coefficients, innovation_variance)
        cholesky(Symmetric(covariance)).L * randn(rng, p)
    elseif initial in (:stationary, :zero)
        zeros(p)
    else
        throw(ArgumentError("initial must be :stationary or :zero"))
    end
    history = copy(presample)
    values = zeros(dates)
    innovations = sqrt(innovation_variance) .* randn(rng, dates)
    for time in 1:dates
        values[time] = dot(coefficients, history) + innovations[time]
        for lag in p:-1:2
            history[lag] = history[lag - 1]
        end
        p == 0 || (history[1] = values[time])
    end
    return values, innovations, presample
end

"""
    generate_case(spec, rng)

Generate one dataset and its latent truth using only this module's AR recursion.
`signal` includes the intercept and factor contributions, but excludes errors.
`B` has one row per series: intercept, then one loading per level. `P` and `P2`
contain factor and error AR coefficients in process order, without padding;
`S` contains error innovation variances. Presample vectors are in reverse time
order, starting at time zero. All factor innovation variances equal one.
"""
function generate_case(spec, rng::AbstractRNG)
    dates, nseries = spec.dates, size(spec.loadings, 1)
    nfactors = sum(spec.nfactors)
    factors, errors = zeros(dates, nfactors), zeros(dates, nseries)
    factor_innovations, error_innovations = similar(factors), similar(errors)
    initial_factors, initial_errors = Vector{Float64}[], Vector{Float64}[]
    for factor in 1:nfactors
        values, innovations, presample = simulate_ar(
            rng, spec.factor_ar[factor], 1.0, dates, spec.initial)
        factors[:, factor] = values
        factor_innovations[:, factor] = innovations
        push!(initial_factors, presample)
    end
    for series in 1:nseries
        values, innovations, presample = simulate_ar(
            rng, spec.error_ar[series], spec.error_variances[series], dates, spec.initial)
        errors[:, series] = values
        error_innovations[:, series] = innovations
        push!(initial_errors, presample)
    end

    factor_orders, error_orders = length.(spec.factor_ar), length.(spec.error_ar)
    level_orders, sign_anchors = Int[], Int[]
    B = zeros(nseries, spec.nlevels + 1)
    B[:, 1] = spec.intercepts
    offset = 0
    for level in 1:spec.nlevels
        indices = (offset + 1):(offset + spec.nfactors[level])
        orders = unique(factor_orders[indices])
        length(orders) == 1 || throw(ArgumentError("factor AR orders must agree within a level"))
        push!(level_orders, only(orders))
        for local_factor in 1:spec.nfactors[level]
            anchor = findfirst(==(local_factor), spec.assignments[:, level])
            anchor === nothing && throw(ArgumentError("every scenario factor needs an assigned series"))
            spec.loadings[anchor, offset + local_factor] > 0 ||
                throw(ArgumentError("true anchor loadings must be positive"))
            push!(sign_anchors, anchor)
        end
        for series in 1:nseries
            local_factor = spec.assignments[series, level]
            if local_factor != 0
                B[series, level + 1] = spec.loadings[series, offset + local_factor]
            end
        end
        offset += spec.nfactors[level]
    end
    signal = factors * spec.loadings' .+ spec.intercepts'
    data = signal + errors
    P = reduce(vcat, spec.factor_ar; init=Float64[])
    P2 = reduce(vcat, spec.error_ar; init=Float64[])
    return (; spec, data, factors, errors, signal, B, P, P2, S=copy(spec.error_variances),
            sign_anchors, factor_orders, error_orders, level_orders,
            factor_innovations, error_innovations, initial_factors, initial_errors)
end

end
