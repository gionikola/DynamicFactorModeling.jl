# Independent whole-trajectory oracle: condition the underlying independent
# Gaussian innovations with one dense SVD, without Kalman recursions.
using DynamicFactorModeling
using LinearAlgebra
using Random
using Test

function independent_state_reference(model, y, initial_mean, initial_root, process_root, measurement_root)
    T, n = size(y)
    m, q = size(process_root)
    initial_count = size(initial_root, 2)
    state_map = zeros(T * m, initial_count + T * q)
    state_mean = zeros(T * m)
    prior_mean = copy(initial_mean)
    for t in 1:T
        rows = ((t - 1) * m + 1):(t * m)
        prior_mean = model.μ + model.F * prior_mean
        state_mean[rows] = prior_mean
        state_map[rows, 1:initial_count] = model.F^t * initial_root
        for s in 1:t
            columns = (initial_count + (s - 1) * q + 1):(initial_count + s * q)
            state_map[rows, columns] = model.F^(t - s) * process_root
        end
    end
    measurement = kron(Matrix{Float64}(I, T, T), model.H)
    observation_map = hcat(measurement * state_map,
                           kron(Matrix{Float64}(I, T, T), measurement_root))
    full_state_map = hcat(state_map, zeros(T * m, T * size(measurement_root, 2)))
    values = vec(permutedims(y))
    used = findall(!ismissing, values)
    observation_map = observation_map[used, :]
    residual = Float64.(values[used]) - (measurement * state_mean)[used]
    if isempty(observation_map)
        return state_mean, full_state_map * full_state_map', 1.0
    end
    decomposition = svd(observation_map; full=true)
    threshold = eps(Float64) * max(size(observation_map)...) * maximum(decomposition.S)
    rank = count(>(threshold), decomposition.S)
    noise_mean = decomposition.V[:, 1:rank] *
        ((decomposition.U[:, 1:rank]' * residual) ./ decomposition.S[1:rank])
    mean = state_mean + full_state_map * noise_mean
    root = full_state_map * decomposition.V[:, (rank + 1):end]
    condition = rank == 0 ? 1.0 : decomposition.S[1] / decomposition.S[rank]
    return mean, root * root', condition
end

function check_singular_case(model, y, initial_mean, initial_root, process_root, measurement_root; seed=1)
    mean, covariance, condition = independent_state_reference(
        model, y, initial_mean, initial_root, process_root, measurement_root)
    kwargs = (; initial_mean, initial_cov=initial_root * initial_root')
    _, smoothed, covariances = kalmanSmoother(y, model; kwargs...)
    m = length(initial_mean)
    # The dense reference can itself lose precision when its nonzero singular
    # values differ greatly. Report that condition number rather than claiming
    # a fixed decimal accuracy for an ill-conditioned exact-observation model.
    scale = max(1.0, maximum(abs, mean))
    mean_tolerance = max(1e-8 * scale, eps(Float64) * condition * scale)
    mean_error = maximum(abs, vec(permutedims(smoothed)) - mean)
    if mean_error > mean_tolerance
        println("Mean audit mismatch: seed=$seed condition=$condition error=$mean_error")
    end
    @test mean_error <= mean_tolerance
    covariance_error = 0.0
    for t in axes(y, 1)
        rows = ((t - 1) * m + 1):(t * m)
        covariance_error = max(covariance_error, maximum(abs, covariances[t] - covariance[rows, rows]))
        if !isapprox(covariances[t], covariance[rows, rows]; atol=1e-8, rtol=1e-8)
            println("Covariance audit mismatch: seed=$seed time=$t condition=$condition error=$covariance_error")
        end
        @test covariances[t] ≈ covariance[rows, rows] atol=1e-8 rtol=1e-8
    end
    draw = KNFactorSampler(MersenneTwister(seed), y, model; kwargs...)
    if iszero(model.R)
        fitted = draw * model.H'
        for index in eachindex(y)
            ismissing(y[index]) || @test fitted[index] ≈ y[index] atol=mean_tolerance rtol=1e-8
        end
    end
    return (; condition, mean_error, covariance_error, mean_tolerance)
end

function check_frozen_singular_regressions()
    cases = include(joinpath(@__DIR__, "singular_drift_fixtures.jl"))
    results = []
    @testset "Exact-observation rounding regression cases" begin
        for case in cases
            model = SSModel(case.H, case.A, case.F, case.μ, case.R, case.Q, case.Z)
            push!(results, check_singular_case(model, case.y, case.mean0,
                zeros(length(case.mean0), 0), case.qroot, case.rroot; seed=case.case))
        end
    end
    return results
end

function singular_reference_case(case; seed=28092026)
    # Separate per-case RNGs keep later models unchanged when a sampler
    # implementation changes how many random numbers it consumes.
    rng = MersenneTwister(seed + case)
    m, n, T = mod(case, 6) + 1, mod(case ÷ 6, 5) + 1, mod(case ÷ 30, 9) + 1
    H, F = randn(rng, n, m), randn(rng, m, m)
    F *= 0.8 / maximum(abs, eigvals(F))
    process_root = randn(rng, m, mod(case, m + 1))
    measurement_root = randn(rng, n, mod(case ÷ 7, n + 1))
    initial_root = randn(rng, m, mod(case ÷ 11, m + 1))
    mean0 = randn(rng, m)
    model = SSModel(H, zeros(n, 0), F, randn(rng, m),
        measurement_root * measurement_root', process_root * process_root', zeros(0, 0))
    # Generate data directly from the supplied independent disturbances.
    # The package's simulator is deliberately not part of this oracle.
    state = mean0 + initial_root * randn(rng, size(initial_root, 2))
    y = Matrix{Union{Missing, Float64}}(undef, T, n)
    for t in 1:T
        state = model.μ + F * state + process_root * randn(rng, size(process_root, 2))
        y[t, :] = H * state + measurement_root * randn(rng, size(measurement_root, 2))
    end
    if case % 3 == 1
        y[rand(rng, T, n) .< 0.2] .= missing
    end
    return (; model, y, mean0, initial_root, process_root, measurement_root)
end

function validate_singular_models(model_count=1500; seed=28092026)
    results = []
    @testset "Independent singular-state audit ($model_count models)" begin
        for case in 1:model_count
            (; model, y, mean0, initial_root, process_root, measurement_root) = singular_reference_case(case; seed)
            push!(results, check_singular_case(model, y, mean0, initial_root,
                process_root, measurement_root; seed=seed + case + model_count))
        end
    end
    append!(results, check_frozen_singular_regressions())
    println("Largest reference condition number: ", maximum(r.condition for r in results))
    println("Largest absolute state-mean difference: ", maximum(r.mean_error for r in results))
    println("Largest absolute covariance difference: ", maximum(r.covariance_error for r in results))
    println("Cases requiring condition-aware mean tolerance: ", count(r -> r.condition > 1e-8 / eps(Float64), results))
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) ? validate_singular_models() : validate_singular_models(parse(Int, only(ARGS)))
end
