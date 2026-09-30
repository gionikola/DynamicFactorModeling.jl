using Test, LinearAlgebra, Statistics, Random
using DynamicFactorModeling

const DFMEstimatorTests = DynamicFactorModeling

# Build a path's response to independent innovations directly by recursion.
# This oracle does not use the estimator's AR transformation or state matrices.
function ar_impulse_response(coefficients, nobs)
    response = zeros(nobs, nobs)
    for time in 1:nobs
        response[time, time] = 1.0
        for lag in 1:min(length(coefficients), time - 1)
            response[time, :] += coefficients[lag] * response[time - lag, :]
        end
    end
    return response
end

function independent_factor_conditional(data, loadings, factor_ar, error_ar, variances)
    nobs, nseries = size(data)
    nfactors = size(loadings, 2)
    prior = zeros(nobs * nfactors, nobs * nfactors)
    noise = zeros(nobs * nseries, nobs * nseries)
    for factor in 1:nfactors
        rows = ((factor - 1) * nobs + 1):(factor * nobs)
        response = ar_impulse_response(factor_ar[factor], nobs)
        prior[rows, rows] = response * response'
    end
    for series in 1:nseries
        rows = ((series - 1) * nobs + 1):(series * nobs)
        response = ar_impulse_response(error_ar[series], nobs)
        noise[rows, rows] = variances[series] * response * response'
    end
    observation = kron(loadings, Matrix{Float64}(I, nobs, nobs))
    covariance = observation * prior * observation' + noise
    gain = (prior * observation') / covariance
    return gain * vec(data), prior - gain * observation * prior
end

@testset "Factor conditional agrees with joint Gaussian conditioning" begin
    data = [0.3 -1.0 0.8; 1.2 0.1 -0.5; 0.0 0.9 0.2; -0.4 0.2 1.1]
    loadings = [1.2 0.2; 0.0 1.0; 0.5 -0.8]
    factor_ar = [[0.55, -0.15], Float64[]]
    error_ar = [[0.4], Float64[], [-0.2, 0.1]]
    variances = [0.5, 0.8, 0.4]
    nobs, nfactors = size(data, 1), size(loadings, 2)
    expected_mean, expected_covariance = independent_factor_conditional(
        data, loadings, factor_ar, error_ar, variances)

    precision, linear_term = DFMEstimatorTests._factor_precision(
        data, loadings, factor_ar, error_ar, variances)
    @test precision \ linear_term ≈ expected_mean atol=1e-12
    @test inv(precision) ≈ expected_covariance atol=1e-12

    model, columns = DFMEstimatorTests._factor_state_space(
        loadings, factor_ar, error_ar, variances)
    nstates = size(model.F, 1)
    _, smoothed, covariance = kalmanSmoother(data, model;
        initial_mean=zeros(nstates), initial_cov=zeros(nstates, nstates))
    @test vec(smoothed[:, columns]) ≈ expected_mean atol=1e-11
    for time in 1:nobs
        indices = [time + (factor - 1) * nobs for factor in 1:nfactors]
        @test covariance[time][columns, columns] ≈ expected_covariance[indices, indices] atol=1e-11
    end

    # Check complete-path draws, including cross-date and cross-factor covariance.
    for sampler in (DFMEstimatorTests._draw_factors_precision,
                    DFMEstimatorTests._draw_factors_state_space)
        rng = MersenneTwister(134)
        draws = hcat([vec(sampler(rng, data, loadings, factor_ar, error_ar, variances))
                      for _ in 1:3500]...)
        @test maximum(abs.(vec(mean(draws; dims=2)) - expected_mean)) < 0.055
        @test maximum(abs.(cov(draws; dims=2) - expected_covariance)) < 0.045
    end
end

@testset "Factor sign changes preserve fitted values" begin
    factors = [1.0 2.0; -0.5 0.3; 0.2 -1.0]
    coefficients = [2.0 -1.0 0.4; -1.0 0.8 -0.5]
    indices = [1 2; 1 2]
    fitted = hcat(ones(3), factors) * coefficients'
    DFMEstimatorTests._identify_factor_signs!(factors, coefficients, indices, [1, 2])
    @test coefficients[1, 2] > 0
    @test coefficients[2, 3] > 0
    @test hcat(ones(3), factors) * coefficients' ≈ fitted
end

@testset "Single-factor samplers" begin
    rng = MersenneTwister(33)
    data = randn(rng, 30, 3) .+ [3.0 -2.0 1.0]
    original = copy(data)
    specification = DFMStruct(2, 2, 8, 4)
    for estimator in (KN1LevelEstimator, OW1LevelEstimator)
        result = estimator(MersenneTwister(71), data, specification)
        repeated = estimator(data, specification; rng=MersenneTwister(71))
        @test result.F == repeated.F
        @test result.B == repeated.B
        all_iterations = estimator(MersenneTwister(71), data, DFMStruct(2, 2, 12, 0))
        @test result.F == all_iterations.F[:, 5:end]
        @test result.B == all_iterations.B[5:end, :]
        @test data == original
        @test size(result.F) == (30, 8)
        @test size(result.B) == (8, 6)
        @test size(result.S) == (8, 3)
        @test size(result.P) == (8, 2)
        @test size(result.P2) == (8, 6)
        @test result.means.F ≈ mean(result.F; dims=2)
        @test result.means.B ≈ mean(result.B; dims=1)
        @test all(isfinite, result.F)
        @test all(>(0), result.S)
        @test all(>(0), result.B[:, 2])
        for draw in 1:8
            @test DFMEstimatorTests.isstationary(result.P[draw, :])
            for series in 1:3
                @test DFMEstimatorTests.isstationary(result.P2[draw, (2series - 1):(2series)])
            end
        end
    end

    # No-lag models and a one-date sample remain well-defined with proper priors.
    for estimator in (KN1LevelEstimator, OW1LevelEstimator)
        result = estimator(MersenneTwister(8), [2.0 4.0], DFMStruct(0, 0, 2, 0))
        @test size(result.F) == (1, 2)
        @test size(result.P) == (2, 0)
        @test size(result.P2) == (2, 0)

        # Even when the AR order exceeds the sample length, the stationary
        # initial density and proper priors define the posterior.
        short = estimator(MersenneTwister(8), [2.0 4.0], DFMStruct(2, 2, 2, 0))
        @test size(short.P) == (2, 2)
        @test all(isfinite, short.F)

        unrestricted = estimator(MersenneTwister(18), [2.0 4.0], DFMStruct(2, 2, 2, 0);
                                  initial=:zero, stationary=false, ar_prior_variance=20.0)
        @test all(isfinite, unrestricted.P)
        anchored = estimator(MersenneTwister(8), data, DFMStruct(0, 0, 2, 0);
                             sign_anchors=[2])
        @test all(>(0), anchored.B[:, 4])
    end
end

@testset "Hierarchy with unequal orders and excluded loadings" begin
    data = randn(MersenneTwister(31), 15, 6)
    assignments = [1 1; 1 1; 1 2; 1 2; 0 2; 0 0]
    specification = HDFMStruct(2, [1, 2], assignments, [2, 0], [0, 1, 2, 0, 1, 0], 5, 2)
    for estimator in (KN2LevelEstimator, OW2LevelEstimator)
        result = estimator(MersenneTwister(7), data, specification)
        @test size(result.F) == (15, 3, 5)
        @test size(result.B) == (5, 18)
        @test size(result.P) == (5, 2)
        @test size(result.P2) == (5, 4)
        @test size(result.means.F) == (15, 3)
        @test result.means.F ≈ dropdims(mean(result.F; dims=3); dims=3)
        @test all(>(0), result.B[:, [2, 3, 9]])
        @test all(iszero, result.B[:, [14, 17, 18]])
        @test all(>(0), result.S)
    end

    three_levels = HDFMStruct(3, [1, 1, 1], [1 0 0; 1 1 0; 1 1 1],
                              [0, 1, 2], [0, 1, 0], 3, 1)
    result = KNHierarchicalEstimator(MersenneTwister(13), data[:, 1:3], three_levels)
    @test size(result.F) == (15, 3, 3)
    @test size(result.P) == (3, 3)
    @test all(iszero, result.B[:, [3, 4, 8]])
    @test_throws ArgumentError KN2LevelEstimator(data[:, 1:3], three_levels)
end

@testset "Estimator input validation" begin
    specification = DFMStruct(1, 1, 2, 0)
    data = ones(3, 2)
    @test_throws ArgumentError KN1LevelEstimator(zeros(0, 2), specification)
    @test_throws ArgumentError KN1LevelEstimator([1.0 NaN; 2.0 3.0], specification)
    @test_throws ArgumentError KN1LevelEstimator([1.0 missing; 2.0 3.0], specification)
    @test_throws ArgumentError KN1LevelEstimator(data, specification; factor_sampler=:invalid)
    @test_throws ArgumentError KN1LevelEstimator(data, specification; beta_prior_variance=0)
    @test_throws ArgumentError KN1LevelEstimator(data, specification; variance_scale=Inf)
    @test_throws ArgumentError KN1LevelEstimator(data, specification; max_attempts=0)
    @test_throws ArgumentError KN1LevelEstimator(data, specification; sign_anchors=[3])
    @test_throws ArgumentError KN1LevelEstimator(data, specification; sign_anchors=[1, 2])

    hierarchy = HDFMStruct(2, [1, 2], [1 1; 1 2], [0, 0], [0, 0], 1, 0)
    @test_throws ArgumentError KN2LevelEstimator(data, hierarchy; sign_anchors=[1, 2, 1])

    # Settings contain mutable arrays. Estimation must recheck them rather
    # than rely only on the constructor's earlier validation.
    hierarchy.factorassign[2, 2] = 1
    @test_throws ArgumentError OW2LevelEstimator(data, hierarchy)
end

@testset "Documented sampler overrides and prior-vector dimensions" begin
    data = [0.3 -0.2; -0.1 0.4]
    single = DFMStruct(0, 0, 2, 0)
    state_draws = KN1LevelEstimator(MersenneTwister(13), data, single)
    overridden = OW1LevelEstimator(MersenneTwister(13), data, single;
                                  factor_sampler=:state_space)
    @test overridden.F == state_draws.F
    @test overridden.B == state_draws.B

    hierarchy = HDFMStruct(2, [1, 1], [1 0; 0 1], [0, 0], [0, 0], 2, 0)
    priors = (beta_prior_variance=[0.3, 0.7, 1.8], ar_prior_variance=Float64[])
    joint_draws = KN2LevelEstimator(MersenneTwister(13), data, hierarchy;
                                   priors..., factor_sampler=:precision)
    overridden = OW2LevelEstimator(MersenneTwister(13), data, hierarchy;
                                  priors..., factor_sampler=:precision)
    @test overridden.F == joint_draws.F
    @test overridden.B == joint_draws.B
    @test all(iszero, overridden.B[:, [3, 5]])
    @test all(>(0), overridden.B[:, [2, 6]])

    # Vectors are indexed by the declared level and lag, not by each series'
    # number of active loadings or by the total number of factors.
    @test_throws DimensionMismatch KN2LevelEstimator(data, hierarchy;
                                                     beta_prior_variance=[0.3, 0.7])
    lagged = DFMStruct(2, 1, 1, 0)
    @test_throws DimensionMismatch KN1LevelEstimator(data, lagged;
                                                     ar_prior_variance=[1.0])
end

@testset "Simulated common signal is recovered" begin
    rng = MersenneTwister(525)
    nobs = 70
    factor = zeros(nobs)
    for time in 1:nobs
        factor[time] = (time > 1 ? 0.6factor[time - 1] : 0.0) + randn(rng)
    end
    loadings = [1.3, 0.9, -1.1, 1.6, 0.7]
    intercepts = [4.0, -3.0, 1.0, 2.0, -1.0]
    data = factor * loadings' .+ intercepts' .+ 0.25randn(rng, nobs, length(loadings))
    result = KN1LevelEstimator(MersenneTwister(190), data, DFMStruct(1, 0, 70, 70))
    @test cor(factor, vec(result.means.F)) > 0.95
    fitted = zeros(size(data))
    for draw in axes(result.F, 2), series in axes(data, 2)
        fitted[:, series] += result.B[draw, 2series - 1] .+
                             result.B[draw, 2series] .* result.F[:, draw]
    end
    fitted ./= size(result.F, 2)
    @test sqrt(mean(abs2, fitted - data)) < 0.4
    # This catches inadvertent fitting of centered data while returning zero intercepts.
    @test maximum(abs.(vec(mean(fitted; dims=1)) - vec(mean(data; dims=1)))) < 0.25
end

@testset "Independent factor starting points" begin
    data = randn(MersenneTwister(153), 12, 3)
    settings = DFMStruct(1, 0, 4, 2)
    start = randn(MersenneTwister(154), 12, 1)
    original = copy(start)
    pca_start = DFMEstimatorTests._initialize_factors(data, ones(Int, 3, 1), 1)
    for estimator in (KN1LevelEstimator, OW1LevelEstimator)
        result = estimator(MersenneTwister(155), data, settings; initial_factors=start)
        repeated = estimator(MersenneTwister(155), data, settings; initial_factors=start)
        @test result.F == repeated.F
        @test start == original
        @test all(isfinite, result.F)

        # Supplying the default path consumes no extra randomness and follows
        # exactly the same updates as an ordinary call.
        default = estimator(MersenneTwister(156), data, settings)
        explicit = estimator(MersenneTwister(156), data, settings; initial_factors=pca_start)
        for field in (:F, :B, :S, :P, :P2)
            @test getproperty(explicit, field) == getproperty(default, field)
        end

        # A user path really controls initialization; it must not be ignored.
        one_sweep = DFMStruct(1, 0, 1, 0)
        supplied = estimator(MersenneTwister(157), data, one_sweep; initial_factors=start)
        zero_start = estimator(MersenneTwister(157), data, one_sweep; initial_factors=zeros(12, 1))
        @test supplied.B != zero_start.B

        @test_throws DimensionMismatch estimator(data, settings; initial_factors=zeros(12, 2))
        @test_throws DimensionMismatch estimator(data, settings; initial_factors=zeros(11, 1))
        @test_throws ArgumentError estimator(data, settings; initial_factors=zeros(12))
        @test_throws ArgumentError estimator(data, settings; initial_factors=fill(NaN, 12, 1))
        @test_throws ArgumentError estimator(data, settings; initial_factors=fill(Inf, 12, 1))
        @test_throws ArgumentError estimator(data, settings; initial_factors=fill(missing, 12, 1))
        @test_throws ArgumentError estimator(data, settings; initial_factors=zeros(ComplexF64, 12, 1))
        @test_throws ArgumentError estimator(data, settings; initial_factors=fill(big"1e1000", 12, 1))
    end
end

@testset "Hierarchical starting paths follow factor ordering" begin
    data = randn(MersenneTwister(159), 10, 4)
    assignments = [1 1; 1 1; 1 2; 1 2]
    two_levels = HDFMStruct(2, [1, 2], assignments, [1, 0], [0, 1, 0, 1], 3, 1)
    three_levels = HDFMStruct(3, [1, 1, 1], [1 0 0; 1 1 0; 1 1 1; 1 1 1],
                              [0, 1, 2], [0, 1, 0, 1], 3, 1)
    storage = randn(MersenneTwister(160), 10, 5)
    original = copy(storage)
    start = view(storage, :, 2:4)

    for (estimator, settings) in ((KN2LevelEstimator, two_levels),
                                   (OW2LevelEstimator, two_levels),
                                   (KNHierarchicalEstimator, three_levels))
        result = estimator(MersenneTwister(161), data, settings; initial_factors=start)
        @test size(result.F) == (10, 3, 3)
        @test all(isfinite, result.F)
        @test storage == original
        @test_throws DimensionMismatch estimator(data, settings; initial_factors=zeros(10, 2))
        @test_throws DimensionMismatch estimator(data, settings; initial_factors=zeros(3, 10))

        # Match the default initializer's level-first ordering, including
        # unassigned levels and factors with different lag orders.
        indices = copy(settings.factorassign)
        offsets = cumsum([0; settings.nfactors[1:(end - 1)]])
        for level in 1:settings.nlevels, series in axes(indices, 1)
            indices[series, level] == 0 || (indices[series, level] += offsets[level])
        end
        pca_start = DFMEstimatorTests._initialize_factors(data, indices, 3)
        default = estimator(MersenneTwister(162), data, settings)
        explicit = estimator(MersenneTwister(162), data, settings; initial_factors=pca_start)
        for field in (:F, :B, :S, :P, :P2)
            @test getproperty(explicit, field) == getproperty(default, field)
        end
    end
end
