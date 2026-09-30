using Random, LinearAlgebra, Statistics, Distributions

@testset "Normal coefficient posterior" begin
    dfm = DynamicFactorModeling
    y = [1.0, 0.0, 2.0, -1.0]
    x = [1.0 2.0; 1.0 0.0; 1.0 -1.0; 1.0 1.0]
    σ2, prior_variance = 1.7, 2.5
    prior_mean = [0.2, -0.4]
    covariance = inv(x' * x / σ2 + I / prior_variance)
    center = covariance * (x' * y / σ2 + prior_mean / prior_variance)
    rng = MersenneTwister(612)
    draws = zeros(16000, 2)
    for i in axes(draws, 1)
        draws[i, :] = dfm.draw_coefficients(rng, y, x, σ2; prior_mean, prior_variance)
    end
    @test vec(mean(draws; dims=1)) ≈ center atol=0.02
    @test cov(draws) ≈ covariance atol=0.018

    for Y in (y, reshape(y, :, 1), view(y, :)), X in (x, view(x, :, :))
        @test length(dfm.draw_coefficients(rng, Y, X, σ2)) == 2
    end
    @test dfm.draw_coefficients(MersenneTwister(1), y, x, σ2) ==
          dfm.draw_coefficients(y, x, σ2; rng=MersenneTwister(1))
    @test dfm.draw_coefficients(MersenneTwister(2), y, x[:, 1], σ2) ==
          only(dfm.draw_coefficients(MersenneTwister(2), y, x[:, 1:1], σ2))
    @test dfm.draw_coefficients(rng, [1, 2, 3], [1, 1, 1], 1) isa Float64
    @test isempty(dfm.draw_coefficients(rng, y, zeros(4, 0), 1))
    # A proper normal prior also supports a rank-deficient design.
    @test all(isfinite, dfm.draw_coefficients(rng, y, ones(4, 2), 1))
    # Forming X'X loses the prior precision at this scale. The fitted sum
    # remains identified even though the separate coefficients are not.
    large_design_draw = dfm.draw_coefficients(rng, fill(2e8, 4), fill(1e8, 4, 2), 1)
    @test all(isfinite, large_design_draw)
    @test sum(large_design_draw) ≈ 2 atol=1e-6

    @test_throws DimensionMismatch dfm.draw_coefficients(y, ones(3, 2), 1)
    @test_throws DimensionMismatch dfm.draw_coefficients(ones(4, 2), x, 1)
    @test_throws DimensionMismatch dfm.draw_coefficients(y, x, 1; prior_mean=[1])
    @test_throws ArgumentError dfm.draw_coefficients(Float64[], zeros(0, 1), 1)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 0)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, Inf)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_variance=-1)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_mean=NaN)
    @test_throws ArgumentError dfm.draw_coefficients([NaN, 2, 3, 4], x, 1)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; max_attempts=0)
    @test_throws ArgumentError dfm.draw_coefficients(y, x, big"1e-1000")
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_variance=big"1e1000")
    @test_throws DimensionMismatch dfm.draw_coefficients(y, x, 1; prior_variance=[1.0])
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_variance=[1.0, 0.0])
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_variance=[1.0, Inf])
    @test_throws ArgumentError dfm.draw_coefficients(y, x, 1; prior_variance=[1, 2im])
end

@testset "Different prior variances for individual coefficients" begin
    dfm = DynamicFactorModeling
    y = [1.0, 0.0, 2.0, -1.0]
    x = [1.0 2.0; 1.0 0.0; 1.0 -1.0; 1.0 1.0]
    prior_mean, prior_variance, σ2 = [0.2, -0.4], [0.1, 4.0], 1.7
    covariance = inv(x' * x / σ2 + Diagonal(1 ./ prior_variance))
    center = covariance * (x' * y / σ2 + prior_mean ./ prior_variance)
    rng = MersenneTwister(614)
    draws = zeros(16000, 2)
    for i in axes(draws, 1)
        draws[i, :] = dfm.draw_coefficients(rng, y, x, σ2; prior_mean, prior_variance)
    end
    @test vec(mean(draws; dims=1)) ≈ center atol=0.02
    @test cov(draws) ≈ covariance atol=0.018
    @test dfm.draw_coefficients(MersenneTwister(4), y, x, σ2; prior_variance=3) ==
          dfm.draw_coefficients(MersenneTwister(4), y, x, σ2; prior_variance=[3, 3])
    @test dfm.draw_coefficients(MersenneTwister(4), y, x[:, 1], σ2; prior_variance=3) ==
          dfm.draw_coefficients(MersenneTwister(4), y, x[:, 1], σ2; prior_variance=[3])
    @test isempty(dfm.draw_coefficients(rng, y, zeros(4, 0), 1; prior_variance=Float64[]))
    for initial in (:zero, :stationary)
        result = dfm.draw_parameters(rng, y, x, [0.4, -0.2], 1;
            initial, beta_prior_variance=[10.0, 2.0], ar_prior_variance=[1.0, 0.5])
        @test all(isfinite, result[1])
        @test dfm.isstationary(result[2])
        @test result[3] > 0
    end
    @test_throws DimensionMismatch dfm.draw_parameters(y, x, [0.4, -0.2], 1;
        initial=:stationary, ar_prior_variance=[1.0])
    @test_throws ArgumentError dfm.draw_parameters(y, x, [0.4], 1;
        initial=:stationary, max_attempts=0)
    prior_only = dfm.draw_parameters(rng, [1.0], ones(1, 1), zeros(2), 1;
        initial=:stationary, beta_prior_variance=[2.0], ar_prior_variance=[1.0, 0.5])
    @test dfm.isstationary(prior_only[2])
end

@testset "Stationary coefficient draws" begin
    dfm = DynamicFactorModeling
    @test dfm.isstationary(Float64[])
    @test dfm.isstationary(0.5)
    @test !dfm.isstationary(1.0)
    @test !dfm.isstationary(-1.0)
    @test dfm.isstationary([1.5, -0.75])
    @test !dfm.isstationary([0.0, 1.1])
    @test !dfm.isstationary([0.9, 0.2])
    @test !dfm.isstationary(fill(0.2, 5))
    @test dfm.isstationary(fill(0.199, 5))
    @test_throws ArgumentError dfm.isstationary([NaN])

    # Zero regressors leave the prior unchanged; an AR(1) stationary draw
    # therefore has the independently known truncated-normal distribution.
    rng = MersenneTwister(702)
    reference = truncated(Normal(1.3, sqrt(0.7)), -1, 1)
    draws = [dfm.draw_coefficients(rng, [0.0], [0.0], 1;
             prior_mean=1.3, prior_variance=0.7, stationary=true) for _ in 1:12000]
    @test all(abs.(draws) .< 1)
    @test mean(draws) ≈ mean(reference) atol=0.02
    @test var(draws) ≈ var(reference) atol=0.02
    @test_throws ErrorException dfm.draw_coefficients(rng, [0.0], [0.0], 1;
        prior_mean=100.0, prior_variance=1e-8, stationary=true, max_attempts=1)
end

@testset "Error variance posterior" begin
    dfm = DynamicFactorModeling
    y = [1.0, 3.0, -1.0, 4.0, 0.0, 2.0, -2.0, 1.0]
    x = hcat(ones(8), collect(1.0:8.0))
    β = [1.0, -0.1]
    shape0, scale0 = 2.0, 1.3
    shape1 = shape0 + length(y) / 2
    scale1 = scale0 + sum(abs2, y - x * β) / 2
    rng = MersenneTwister(715)
    draws = [dfm.draw_error_variance(rng, y, x, β; prior_shape=shape0,
             prior_scale=scale0) for _ in 1:25000]
    # The precision has gamma moments, which stay well behaved for this test.
    @test mean(1 ./ draws) ≈ shape1 / scale1 rtol=0.018
    @test var(1 ./ draws) ≈ shape1 / scale1^2 rtol=0.035
    @test mean(draws) ≈ scale1 / (shape1 - 1) rtol=0.025
    @test dfm.draw_error_variance(MersenneTwister(1), y, x, β) ==
          dfm.draw_error_variance(reshape(y, :, 1), x, β; rng=MersenneTwister(1))
    @test dfm.draw_error_variance(MersenneTwister(2), y, x[:, 1], 1.0) ==
          dfm.draw_error_variance(MersenneTwister(2), y, x[:, 1:1], [1.0])
    @test dfm.draw_error_variance(y, zeros(8, 0), Float64[]) > 0
    @test_throws DimensionMismatch dfm.draw_error_variance(y, x, [1.0])
    @test_throws DimensionMismatch dfm.draw_error_variance(y, x[:, 1], [1.0, 2.0])
    @test_throws ArgumentError dfm.draw_error_variance(y, x, [Inf, 1])
    @test_throws ArgumentError dfm.draw_error_variance(y, x, β; prior_shape=0)
    @test_throws ArgumentError dfm.draw_error_variance(y, x, β; prior_scale=0)
end

@testset "Gibbs step uses all AR lags and the new residuals" begin
    dfm = DynamicFactorModeling
    y = [1.0, -2.0, 4.0, 0.5, 2.0, -0.3]
    x = [1.0 0.2; 1.0 -1.0; 1.0 2.0; 1.0 3.0; 1.0 0.5; 1.0 -0.7]
    old_ϕ = [0.4, -0.25]
    ycopy, xcopy, ϕcopy = copy(y), copy(x), copy(old_ϕ)
    for initial in (:conditional, :zero)
        # Explicit AR(2) transformations provide an independent small oracle.
        lag1_y, lag2_y = [0.0; y[1:5]], [0.0; 0.0; y[1:4]]
        lag1_x = vcat(zeros(1, 2), x[1:5, :])
        lag2_x = vcat(zeros(2, 2), x[1:4, :])
        rows = initial == :zero ? (1:6) : (3:6)
        ystar = (y - 0.4lag1_y + 0.25lag2_y)[rows]
        xstar = (x - 0.4lag1_x + 0.25lag2_x)[rows, :]
        oracle_rng = MersenneTwister(814)
        β = dfm.draw_coefficients(oracle_rng, ystar, xstar, 0.8; prior_variance=5)
        errors = y - x * β
        lag_errors = hcat([0.0; errors[1:5]], [0.0; 0.0; errors[1:4]])[rows, :]
        ϕ = dfm.draw_coefficients(oracle_rng, errors[rows], lag_errors, 0.8;
                                  prior_variance=0.5, stationary=true)
        variance = dfm.draw_error_variance(oracle_rng, errors[rows], lag_errors, ϕ;
                                           prior_shape=2, prior_scale=1)
        result = dfm.draw_parameters(MersenneTwister(814), y, x, old_ϕ, 0.8;
            initial, beta_prior_variance=5, ar_prior_variance=0.5,
            variance_shape=2, variance_scale=1)
        @test result[1] ≈ β atol=1e-12
        @test result[2] ≈ ϕ atol=1e-12
        @test result[3] ≈ variance atol=1e-12
        @test y == ycopy && x == xcopy && old_ϕ == ϕcopy
    end

    # Empty AR coefficients must reproduce the independent-error Gibbs step.
    β0, v0 = dfm.draw_parameters(MersenneTwister(22), y, x, 1.0)
    β, ϕ, v = dfm.draw_parameters(MersenneTwister(22), y, x, Float64[], 1.0)
    @test β == β0 && isempty(ϕ) && v == v0
    scalar_step = dfm.draw_parameters(MersenneTwister(2), y, x[:, 1], 1.0)
    @test scalar_step[1] isa Float64
    @test scalar_step == dfm.draw_parameters(y, x[:, 1], 1.0; rng=MersenneTwister(2))
    @test dfm.draw_parameters(MersenneTwister(1), y, x, old_ϕ, 1) ==
          dfm.draw_parameters(y, x, old_ϕ, 1; rng=MersenneTwister(1))
    @test dfm.draw_parameters(MersenneTwister(1), y, x[:, 1], old_ϕ, 1)[1] isa Float64
    @test length(dfm.draw_parameters(MersenneTwister(1), [1.0], ones(1, 1),
        zeros(2), 1; initial=:zero, ar_prior_variance=0.1)[2]) == 2
    @test_throws ArgumentError dfm.draw_parameters(y, x, zeros(6), 1)
    @test_throws ArgumentError dfm.draw_parameters(y, x, old_ϕ, 1; initial=:invalid)
    @test_throws ArgumentError dfm.draw_parameters(y, x, [1.1], 1)
    @test_throws ArgumentError dfm.draw_parameters(y, x, [NaN], 1; stationary=false)
end
