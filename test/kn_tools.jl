using LinearAlgebra
using Random
using Statistics

# Independent reference: construct the joint Gaussian distribution of all
# states and observations at once, then condition using dense linear algebra.
function batch_state_posterior(y, model, initial_mean, initial_cov; data_z=zeros(size(y, 1), size(model.A, 2)), through=size(y, 1))
    T = size(y, 1)
    m, n = length(initial_mean), size(y, 2)
    state_mean = zeros(T * m)
    state_covariance = zeros(T * m, T * m)
    previous_mean, previous_covariance = initial_mean, initial_cov
    for t in 1:T
        rows = ((t - 1) * m + 1):(t * m)
        previous_mean = model.μ + model.F * previous_mean
        previous_covariance = model.F * previous_covariance * model.F' + model.Q
        state_mean[rows] = previous_mean
        state_covariance[rows, rows] = previous_covariance
        for s in 1:(t - 1)
            columns = ((s - 1) * m + 1):(s * m)
            state_covariance[rows, columns] = model.F^(t - s) * state_covariance[columns, columns]
            state_covariance[columns, rows] = state_covariance[rows, columns]'
        end
    end
    measurement = kron(Matrix{Float64}(I, T, T), model.H)
    observation_covariance = measurement * state_covariance * measurement' +
                             kron(Matrix{Float64}(I, T, T), model.R)
    values = vec(permutedims(y))
    used = [i for i in 1:(through * n) if !ismissing(values[i])]
    offsets = vec(permutedims(data_z * model.A'))
    residual = Float64.(values[used]) - (measurement * state_mean + offsets)[used]
    cross_covariance = (state_covariance * measurement')[ :, used]
    gain = cross_covariance * pinv(observation_covariance[used, used])
    mean = state_mean + gain * residual
    covariance = state_covariance - gain * cross_covariance'
    return mean, covariance
end

@testset "Kalman filter and smoother match joint Gaussian conditioning" begin
    model = SSModel([1.0 0.3; -0.2 1.0], reshape([0.7, -0.4], 2, 1),
        [0.6 0.2; 0.0 0.4], [0.2, -0.1], [0.5 0.1; 0.1 0.3],
        [0.7 -0.1; -0.1 0.4], ones(1, 1))
    y = [1.0 -0.5; 0.2 0.8; -0.1 0.3; 0.9 0.4]
    z = reshape([0.3, -0.5, 1.0, 0.2], 4, 1)
    mean0, cov0 = [0.1, 0.5], [0.8 0.2; 0.2 0.6]
    kwargs = (; data_z=z, initial_mean=mean0, initial_cov=cov0)
    predicted_y, filtered, predicted_covariances, filtered_covariances = kalmanFilter(y, model; kwargs...)
    fitted, smoothed, smoothed_covariances = kalmanSmoother(y, model; kwargs...)
    mean, covariance = batch_state_posterior(y, model, mean0, cov0; data_z=z)
    @test vec(permutedims(smoothed)) ≈ mean atol=1e-12
    @test fitted ≈ smoothed * model.H' + z * model.A'
    for t in 1:4
        rows = (2t - 1):(2t)
        filter_mean, filter_covariance = batch_state_posterior(y, model, mean0, cov0; data_z=z, through=t)
        @test filtered[t, :] ≈ filter_mean[rows] atol=1e-12
        @test filtered_covariances[t] ≈ filter_covariance[rows, rows] atol=1e-12
        @test smoothed_covariances[t] ≈ covariance[rows, rows] atol=1e-12
        prior_mean = t == 1 ? mean0 : filtered[t - 1, :]
        prior_covariance = t == 1 ? cov0 : filtered_covariances[t - 1]
        @test predicted_y[t, :] ≈ model.H * (model.μ + model.F * prior_mean) + model.A * z[t, :]
        @test predicted_covariances[t] ≈ model.F * prior_covariance * model.F' + model.Q
    end
    @test smoothed[end, :] ≈ filtered[end, :]
    @test fitted[end, :] ≈ model.H * filtered[end, :] + model.A * z[end, :]
    @test_throws ArgumentError kalmanFilter(y, model)
    @test_throws DimensionMismatch kalmanFilter(y, model; data_z=zeros(3, 1))

    incomplete = Matrix{Union{Missing, Float64}}(y)
    incomplete[1, 2] = missing
    incomplete[2, :] .= missing
    incomplete[4, 1] = missing
    _, missing_smoothed, missing_covariances = kalmanSmoother(incomplete, model; kwargs...)
    missing_mean, missing_covariance = batch_state_posterior(incomplete, model, mean0, cov0; data_z=z)
    @test vec(permutedims(missing_smoothed)) ≈ missing_mean atol=1e-12
    for t in 1:4
        rows = (2t - 1):(2t)
        @test missing_covariances[t] ≈ missing_covariance[rows, rows] atol=1e-12
    end
    invalid = copy(y)
    invalid[1, 1] = NaN
    @test_throws ArgumentError kalmanFilter(invalid, model; kwargs...)
end

@testset "Posterior path draws preserve time dependence" begin
    model = SSModel(ones(1, 1), zeros(1, 0), fill(0.7, 1, 1), [0.3],
        fill(0.4, 1, 1), fill(0.8, 1, 1), zeros(0, 0))
    y = reshape([0.3, 1.2, -0.4], 3, 1)
    kwargs = (; initial_mean=[0.2], initial_cov=fill(0.6, 1, 1))
    mean, covariance = batch_state_posterior(y, model, kwargs.initial_mean, kwargs.initial_cov)
    rng = MersenneTwister(205)
    draws = reduce(vcat, [vec(KNFactorSampler(rng, y, model; kwargs...))' for _ in 1:6000])
    @test vec(Statistics.mean(draws; dims=1)) ≈ mean atol=0.025
    @test cov(draws) ≈ covariance atol=0.025
    @test KNFactorSampler(MersenneTwister(25), y, model; kwargs...) ==
          KNFactorSampler(MersenneTwister(25), y, model; kwargs...)
end

@testset "Singular state and observation covariances" begin
    # A two-lag companion state plus a wholly deterministic third state.
    model = SSModel([1.0 0 0], zeros(1, 0), [0.7 -0.1 0; 1 0 0; 0 0 0.3],
        [0.2, 0, 0.4], fill(0.2, 1, 1), Diagonal([0.8, 0, 0]), zeros(0, 0))
    y = reshape([0.3, 1.2, -0.4, 0.7], 4, 1)
    kwargs = (; initial_mean=[0.0, 0.0, 2.0], initial_cov=zeros(3, 3))
    mean, covariance = batch_state_posterior(y, model, kwargs.initial_mean, kwargs.initial_cov)
    _, smoothed, smoothed_covariances = kalmanSmoother(y, model; kwargs...)
    @test vec(permutedims(smoothed)) ≈ mean atol=1e-12
    for t in 1:4
        rows = (3t - 2):(3t)
        @test smoothed_covariances[t] ≈ covariance[rows, rows] atol=1e-12
    end
    for seed in 1:20
        sampled = KNFactorSampler(MersenneTwister(seed), y, model; kwargs...)
        @test sampled[2:end, 2] ≈ sampled[1:end-1, 1] atol=1e-12
        @test sampled[1, 2] ≈ 0 atol=1e-12
        @test sampled[:, 3] ≈ [1.0, 0.7, 0.61, 0.583] atol=1e-12
    end

    # Two redundant exact observations have a singular innovation covariance.
    exact = SSModel(reshape([1.0, 2.0], 2, 1), zeros(2, 0), fill(0.5, 1, 1), [0.0],
        zeros(2, 2), ones(1, 1), zeros(0, 0))
    exact_y = [1.0 2; -0.2 -0.4; 0.7 1.4]
    _, filtered, _, covariances = kalmanFilter(exact_y, exact)
    @test filtered[:, 1] ≈ exact_y[:, 1]
    @test all(P -> norm(P) < 1e-12, covariances)
    @test KNFactorSampler(exact_y, exact)[:, 1] ≈ exact_y[:, 1]
    exact_y[2, 2] = 0.5
    @test_throws ArgumentError kalmanFilter(exact_y, exact)
end

@testset "Initialization and empty data" begin
    stable = SSModel(ones(1, 1), zeros(1, 0), fill(0.5, 1, 1), [1.0],
        ones(1, 1), fill(0.6, 1, 1), zeros(0, 0))
    _, _, prediction_covariances, _ = kalmanFilter(reshape([1.0], 1, 1), stable)
    @test prediction_covariances[1][1] ≈ 0.6 / (1 - 0.5^2)
    @test size(kalmanFilter(zeros(0, 1), stable)[1]) == (0, 1)
    @test size(kalmanSmoother(zeros(0, 1), stable)[2]) == (0, 1)
    @test size(KNFactorSampler(zeros(0, 1), stable)) == (0, 1)
    precise = SSModel(ones(1, 1), zeros(1, 0), zeros(1, 1), [0.0],
        fill(1e-18, 1, 1), ones(1, 1), zeros(0, 0))
    @test kalmanFilter(zeros(1, 1), precise)[4][1][1] ≈ 1e-18 rtol=1e-12
    random_walk = SSModel(ones(1, 1), zeros(1, 0), ones(1, 1), [0.0],
        ones(1, 1), ones(1, 1), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter(zeros(1, 1), random_walk)
    unit_root_F = zeros(5, 5)
    unit_root_F[1, :] .= 0.2
    unit_root_F[2:5, 1:4] = Matrix{Float64}(I, 4, 4)
    unit_root = SSModel(ones(1, 5), zeros(1, 0), unit_root_F, zeros(5),
        ones(1, 1), Matrix{Float64}(I, 5, 5), zeros(0, 0))
    @test_throws ArgumentError kalmanFilter(zeros(1, 1), unit_root)
    @test all(isfinite, kalmanFilter(zeros(1, 1), random_walk;
        initial_mean=[0], initial_cov=zeros(1, 1))[2])
    @test_throws ArgumentError kalmanFilter(zeros(1, 1), stable;
        initial_mean=[0], initial_cov=fill(-1., 1, 1))
end

@testset "Randomized singular Gaussian reference checks" begin
    rng = MersenneTwister(322)
    for seed in 1:60
        m, n, T = 4, 3, 5
        H = randn(rng, n, m)
        F = randn(rng, m, m)
        F *= 0.8 / maximum(abs, eigvals(F))
        qroot, rroot = randn(rng, m, mod(seed, 4) + 1), randn(rng, n, mod(seed, 3))
        model = SSModel(H, zeros(n, 0), F, randn(rng, m), rroot*rroot', qroot*qroot', zeros(0, 0))
        mean0, cov0 = randn(rng, m), zeros(m, m)
        y, _, _ = simulateSSModel(rng, T, model; initial_mean=mean0, initial_cov=cov0)
        _, smoothed, covariances = kalmanSmoother(y, model; initial_mean=mean0, initial_cov=cov0)
        # Condition the independent innovation draws with a full SVD. This
        # reference avoids both Kalman recursions and covariance subtraction.
        q = size(qroot,2)
        state_map = zeros(T*m, T*q)
        state_mean = zeros(T*m)
        prior_mean = mean0
        for t in 1:T
            rows = ((t-1)*m+1):(t*m)
            prior_mean = model.μ + F*prior_mean
            state_mean[rows] = prior_mean
            for s in 1:t
                columns = ((s-1)*q+1):(s*q)
                state_map[rows, columns] = F^(t-s)*qroot
            end
        end
        measurement = kron(Matrix{Float64}(I,T,T),H)
        observation_map = hcat(measurement*state_map,kron(Matrix{Float64}(I,T,T),rroot))
        full_state_map = hcat(state_map, zeros(T*m,T*size(rroot,2)))
        decomposition = svd(observation_map;full=true)
        rank = count(>(1e-12 * maximum(decomposition.S)),decomposition.S)
        posterior_noise_mean = decomposition.V[:,1:rank] *
            ((decomposition.U[:,1:rank]' * (vec(permutedims(y))-measurement*state_mean)) ./ decomposition.S[1:rank])
        mean = state_mean + full_state_map*posterior_noise_mean
        posterior_root = full_state_map*decomposition.V[:,rank+1:end]
        covariance = posterior_root * posterior_root'
        @test vec(permutedims(smoothed)) ≈ mean atol=1e-7
        for t in 1:T
            rows = ((t-1)*m+1):(t*m)
            @test covariances[t] ≈ covariance[rows, rows] atol=1e-7
        end
        draw = KNFactorSampler(rng, y, model; initial_mean=mean0, initial_cov=cov0)
        if iszero(model.R)
            @test draw * H' ≈ y atol=1e-7
        end
    end
end
