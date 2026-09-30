include(joinpath(@__DIR__, "reference", "singular_state_space.jl"))
check_frozen_singular_regressions()

@testset "Covariance scales and numerical input validation" begin
    for variance in (nextfloat(0.0), 1e-300, 1e-18, 1.0, 1e300, 1e308)
        root = DynamicFactorModeling._covariance_root(fill(variance, 1, 1))
        @test only(root) ≈ sqrt(variance) rtol=1e-12
        @test isfinite(only(DynamicFactorModeling.mvn(MersenneTwister(20), [0.0], fill(variance, 1, 1))))
    end
    slow = SSModel(Matrix{Float64}(I, 2, 2), zeros(2, 0), Diagonal([0.9999, 0.0]), zeros(2),
        zeros(2, 2), Diagonal([1e-24, 1e12]), zeros(0, 0))
    _, covariance = DynamicFactorModeling._initial_distribution(slow, nothing, nothing)
    @test covariance[1, 1] ≈ 1e-24 / (1 - 0.9999^2) rtol=1e-10
    @test covariance[2, 2] ≈ 1e12 rtol=1e-12
    huge = SSModel(ones(1, 1), zeros(1, 0), fill(0.5, 1, 1), [0.0],
        zeros(1, 1), fill(1e308, 1, 1), zeros(0, 0))
    @test DynamicFactorModeling._initial_distribution(huge, nothing, nothing)[2][1] ≈ 1e308 / 0.75

    small = SSModel(ones(1, 1), ones(1, 1), zeros(1, 1), [0.0],
        ones(1, 1), ones(1, 1), ones(1, 1))
    @test_throws ArgumentError kalmanFilter(reshape([big"1e1000"], 1, 1), small; data_z=zeros(1, 1))
    @test_throws ArgumentError kalmanFilter(zeros(1, 1), small; data_z=reshape([big"1e1000"], 1, 1))
    @test_throws ArgumentError DynamicFactorModeling.Γinv(MersenneTwister(1), 1.0, big"1e-1000")

    # Posterior covariance depends on model matrices and missingness, not the
    # observations' values or state/observation intercepts.
    H, F = Matrix{Float64}(I, 2, 2), 0.5Matrix{Float64}(I, 2, 2)
    Q, R = Matrix(Diagonal([1e-18, 1.0])), Matrix(Diagonal([2e-18, 0.2]))
    model = SSModel(H, zeros(2, 0), F, zeros(2), R, Q, zeros(0, 0))
    shifted = SSModel(H, zeros(2, 0), F, [1e12, -1e12], R, Q, zeros(0, 0))
    data = zeros(5, 2)
    original = kalmanSmoother(data, model)
    translated = kalmanSmoother(fill(2e12, 5, 2), shifted)
    @test original[3] == translated[3]
    @test all(P -> P[1, 1] > 0, original[3])
end

@testset "Deterministic constraints retain their own units" begin
    model = SSModel(Matrix{Float64}(I, 2, 2), zeros(2, 0), zeros(2, 2), zeros(2),
        zeros(2, 2), Diagonal([1e30, 0.0]), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter([1e15 1.0], model)
    @test_throws ArgumentError kalmanFilter([1e15 1e-20], model)
    _, filtered, _, covariance = kalmanFilter([1e15 0.0], model)
    @test filtered[1, 2] == 0.0
    @test covariance[1][2, 2] == 0.0
    @test KNFactorSampler([1e15 0.0], model)[1, 2] == 0.0
end

@testset "Backward deterministic inversion uses joint conditioning" begin
    for case in (201, 1067)
        reference = singular_reference_case(case)
        (; model, y, mean0, initial_root, process_root, measurement_root) = reference
        filtered = DynamicFactorModeling._kalman_filter(y, model;
            initial_mean=mean0, initial_cov=initial_root * initial_root')
        @test DynamicFactorModeling._backward_distributions(filtered, model)[3]
        check_singular_case(model, y, mean0, initial_root, process_root, measurement_root; seed=case)
    end

    # One unknown presample value determines the whole path. Its posterior is
    # scalar normal regression, giving a reference without state recursions.
    F, intercept, T = 0.01, 0.3, 8
    model = SSModel(ones(1, 1), zeros(1, 0), fill(F, 1, 1), [intercept],
        fill(0.2, 1, 1), zeros(1, 1), zeros(0, 0))
    times = 1:T
    loadings = F .^ times
    affine_mean = intercept .* (1 .- loadings) ./ (1 - F)
    data = reshape(affine_mean + [0.1, -0.2, 0.3, -0.1, 0.05, 0.4, -0.2, 0.1], T, 1)
    posterior_variance = inv(1 + sum(abs2, loadings) / 0.2)
    posterior_mean = posterior_variance * dot(loadings, vec(data) - affine_mean) / 0.2
    kwargs = (; initial_mean=[0.0], initial_cov=ones(1, 1))
    _, states, covariance = kalmanSmoother(data, model; kwargs...)
    @test vec(states) ≈ affine_mean + loadings * posterior_mean atol=1e-14 rtol=1e-14
    for t in times
        @test only(covariance[t]) ≈ loadings[t]^2 * posterior_variance rtol=1e-12
    end
    rng = MersenneTwister(701)
    presample_draws = [(KNFactorSampler(rng, data, model; kwargs...)[1] - intercept) / F for _ in 1:3000]
    @test mean(presample_draws) ≈ posterior_mean atol=0.06
    @test var(presample_draws) ≈ posterior_variance rtol=0.07
end

@testset "Unobserved contracting paths retain their prior" begin
    model = SSModel(ones(1, 1), zeros(1, 0), fill(0.01, 1, 1), [0.0],
        zeros(1, 1), zeros(1, 1), zeros(0, 0))
    data = fill(missing, 10, 1)
    kwargs = (; initial_mean=[0.0], initial_cov=ones(1, 1))
    _, states, covariance = kalmanSmoother(data, model; kwargs...)
    @test iszero(states)
    for t in 1:10
        @test only(covariance[t]) ≈ 0.01^(2t) rtol=1e-12
    end
    draw = KNFactorSampler(MersenneTwister(78), data, model; kwargs...)
    @test draw[2:end, :] ≈ 0.01 * draw[1:(end - 1), :] rtol=1e-14
    @test !iszero(draw)
    gain, root, constraints = DynamicFactorModeling._joint_conditioning_parameters(ones(1, 1), zeros(0, 1))
    @test size(gain) == (1, 0)
    @test size(constraints) == (0, 0)
    @test root * root' ≈ ones(1, 1)
end

@testset "Filtering is invariant to state and observation units" begin
    for exponent in (10, 15, 20)
        H = [10.0^-exponent 2 * 10.0^-exponent; 10.0^exponent -10.0^exponent]
        model = SSModel(H, zeros(2, 0), zeros(2, 2), zeros(2), zeros(2, 2),
            Matrix{Float64}(I, 2, 2), zeros(0, 0))
        truth = [1.0, 2.0]
        @test vec(kalmanFilter(reshape(H * truth, 1, 2), model)[2]) ≈ truth atol=1e-12 rtol=1e-12
        # A redundant measurement must not alter the result in mixed units.
        H = vcat(H, H[1:1, :])
        model = SSModel(H, zeros(3, 0), zeros(2, 2), zeros(2), zeros(3, 3),
            Matrix{Float64}(I, 2, 2), zeros(0, 0))
        @test vec(kalmanFilter(reshape(H * truth, 1, 3), model)[2]) ≈ truth atol=1e-12 rtol=1e-12
    end
    for variance in (1e-18, 1e-100)
        covariance = Matrix(Diagonal([1.0, variance]))
        model = SSModel(Matrix{Float64}(I, 2, 2), zeros(2, 0), 0.8Matrix{Float64}(I, 2, 2),
            zeros(2), covariance, covariance, zeros(0, 0))
        data = repeat([1.0 sqrt(variance)], 20, 1)
        _, states, covariances = kalmanSmoother(data, model; initial_mean=zeros(2), initial_cov=covariance)
        @test states[:, 1] ≈ states[:, 2] / sqrt(variance) atol=1e-12 rtol=1e-12
        @test all(P -> isapprox(P[1, 1], P[2, 2] / variance; atol=1e-12, rtol=1e-12), covariances)
    end
end

@testset "Backward calculations reject state overflow" begin
    model = SSModel(ones(1, 1), zeros(1, 0), fill(0.1, 1, 1), [0.0],
        zeros(1, 1), zeros(1, 1), zeros(0, 0))
    data = reshape(Union{Missing, Float64}[missing, 1e308], 2, 1)
    kwargs = (; initial_mean=[0.0], initial_cov=ones(1, 1))
    @test isfinite(kalmanFilter(data, model; kwargs...)[2][2])
    @test_throws ArgumentError kalmanSmoother(data, model; kwargs...)
    @test_throws ArgumentError KNFactorSampler(MersenneTwister(1), data, model; kwargs...)

    # A cancellation can leave a finite prediction but an unrepresentable
    # rounding bound. It must not become an infinite allowance for bad data.
    cancellation = SSModel([1.0 -1.0], zeros(1, 0), Matrix{Float64}(I, 2, 2), zeros(2),
        zeros(1, 1), zeros(2, 2), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter(ones(1, 1), cancellation;
        initial_mean=fill(1e308, 2), initial_cov=zeros(2, 2))

    # Finite matrix entries can still imply a row norm outside Float64.
    # Such uncertainty must not vanish during row normalization.
    transition_overflow = SSModel(Matrix{Float64}(I, 4, 4), zeros(4, 0), fill(1e308, 4, 4),
        zeros(4), zeros(4, 4), zeros(4, 4), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter(fill(missing, 1, 4), transition_overflow;
        initial_mean=zeros(4), initial_cov=Matrix{Float64}(I, 4, 4))
    observation_overflow = SSModel(fill(1e308, 1, 4), zeros(1, 0), Matrix{Float64}(I, 4, 4),
        zeros(4), zeros(1, 1), zeros(4, 4), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter(zeros(1, 1), observation_overflow;
        initial_mean=zeros(4), initial_cov=Matrix{Float64}(I, 4, 4))
end
