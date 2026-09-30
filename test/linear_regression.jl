using Random, LinearAlgebra, Statistics

@testset "Regression sample layout and validation" begin
    dfm = DynamicFactorModeling
    rng = MersenneTwister(901)
    y = randn(rng, 30)
    x = hcat(ones(30), randn(rng, 30))
    for p in (0, 1, 2)
        draws = dfm.regress(MersenneTwister(902), y, x, p, 20, 5)
        @test size(draws[1]) == (15, 2)
        @test size(draws[2]) == (15, p)
        @test size(draws[3]) == (15,)
        @test all(>(0), draws[3])
        @test all(dfm.isstationary(view(draws[2], i, :)) for i in axes(draws[2], 1))
        @test draws == dfm.regress(reshape(y, :, 1), x, p, 20, 5; rng=MersenneTwister(902))
        untrimmed = dfm.regress(MersenneTwister(902), y, x, p, 20, 0)
        @test draws[1] == untrimmed[1][6:end, :]
        @test draws[2] == untrimmed[2][6:end, :]
        @test draws[3] == untrimmed[3][6:end]
    end
    @test size(dfm.regress(rng, y, x[:, 1], 0, 1, 0)[1]) == (1, 1)
    @test size(dfm.regress(rng, [1.0], ones(1, 1), 2, 2, 0;
        initial=:zero, ar_prior_variance=0.1)[2]) == (2, 2)
    @test_throws ArgumentError dfm.regress(y, x, -1, 10, 1)
    @test_throws ArgumentError dfm.regress(y, x, 30, 10, 1)
    @test_throws ArgumentError dfm.regress(y, x, 1, 0, 0)
    @test_throws ArgumentError dfm.regress(y, x, 1, 10, -1)
    @test_throws ArgumentError dfm.regress(y, x, 1, 10, 10)
    @test_throws DimensionMismatch dfm.regress(y, ones(29, 2), 1, 10, 1)
end

@testset "Regression chain agrees with an integrated posterior" begin
    dfm = DynamicFactorModeling
    y = [1.0, 2.0, -1.0, 0.5, 1.5]
    x = ones(length(y))
    prior_variance, shape, scale = 2.0, 2.0, 1.0

    # Integrating σ² out gives p(β|y) proportional to
    # exp(-β² / (2v₀)) * (b₀ + sum((y-β)²)/2)^(-a₀-n/2).
    # Direct numerical integration is independent of the Gibbs implementation.
    grid = collect(-8.0:0.002:8.0)
    conditional_scales = [scale + sum(abs2, y .- β) / 2 for β in grid]
    weights = exp.(-grid.^2 / (2prior_variance)) .* conditional_scales.^(-shape - length(y)/2)
    weights /= sum(weights)
    expected_β = sum(weights .* grid)
    expected_βvariance = sum(weights .* (grid .- expected_β).^2)
    expected_σ2 = sum(weights .* conditional_scales) / (shape + length(y)/2 - 1)

    β, ϕ, σ2 = dfm.regress(MersenneTwister(915), y, x, 0, 18000, 3000;
        beta_prior_variance=prior_variance, variance_shape=shape, variance_scale=scale)
    @test isempty(ϕ)
    @test mean(β) ≈ expected_β atol=0.025
    @test var(β) ≈ expected_βvariance rtol=0.07
    @test mean(σ2) ≈ expected_σ2 rtol=0.04
end

@testset "Recover a regression with AR(2) errors" begin
    dfm = DynamicFactorModeling
    rng = MersenneTwister(921)
    n = 600
    x = hcat(ones(n), randn(rng, n))
    true_β, true_ϕ, true_σ2 = [0.4, -1.2], [0.55, -0.2], 0.4
    errors = sqrt(true_σ2) * randn(rng, n)
    for t in 1:n
        for lag in 1:min(2, t - 1)
            errors[t] += true_ϕ[lag] * errors[t - lag]
        end
    end
    y = x * true_β + errors
    β, ϕ, σ2 = dfm.regress(MersenneTwister(922), y, x, 2, 3000, 500;
        initial=:zero, beta_prior_variance=100, ar_prior_variance=1,
        variance_shape=2, variance_scale=1)
    @test vec(mean(β; dims=1)) ≈ true_β atol=0.12
    @test vec(mean(ϕ; dims=1)) ≈ true_ϕ atol=0.1
    @test mean(σ2) ≈ true_σ2 atol=0.08
end
