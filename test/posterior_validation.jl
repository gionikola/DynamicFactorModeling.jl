using Test, Random, Statistics, Distributions
using DynamicFactorModeling
include(joinpath(@__DIR__, "reference", "posterior_reference.jl"))

# These tests compare complete estimator chains with direct integration of a
# deliberately small posterior. They do not call individual conditional samplers
# to construct the reference, and include correlated-chain Monte Carlo errors.
function one_date_chain_statistics(result)
    a, b, s = result.B[:, 1], result.B[:, 2], result.S[:, 1]
    f = vec(result.F)
    ψ2 = size(result.P, 2) == 0 ? zeros(length(s)) : result.P[:, 1].^2
    φ2 = size(result.P2, 2) == 0 ? zeros(length(s)) : result.P2[:, 1].^2
    return (intercept=a, loading=b, loading_squared=b.^2, variance=s,
            precision=1 ./ s, factor=f, factor_squared=f.^2, fitted_factor=b .* f,
            factor_ar_squared=ψ2, error_ar_squared=φ2,
            factor_ar_loading_squared=ψ2 .* b.^2, error_ar_variance=φ2 .* s)
end

function check_posterior_mean(chains, expected; integration_error=5e-5)
    summary = PosteriorReference.batch_summary(chains)
    # Fail explicitly if the chains are too imprecise to test the target.
    @test summary.mcse < 0.035 * (1 + abs(expected))
    @test abs(summary.mean - expected) <= 6summary.mcse + integration_error
    for i in eachindex(summary.chain_means)
        @test abs(summary.chain_means[i] - expected) <= 7summary.chain_mcse[i] + integration_error
    end
    return summary
end

@testset "Independent posterior quadrature converges" begin
    nodes, weights = PosteriorReference.gauss_legendre(12, -1, 1)
    @test sum(weights) ≈ 2 atol=1e-14
    @test sum(weights .* nodes.^2) ≈ 2 / 3 atol=1e-14
    @test sum(weights .* nodes.^8) ≈ 2 / 9 atol=1e-14
    for (y, prior) in ((1.2, (beta_prior_variance=0.5, variance_shape=4.0, variance_scale=1.0)),
                       (-0.8, (beta_prior_variance=[2.0, 0.4], variance_shape=6.0, variance_scale=3.0)))
        coarse = PosteriorReference.one_date(y; prior..., order=120)
        fine = PosteriorReference.one_date(y; prior..., order=160)
        @test coarse.mass ≈ fine.mass rtol=1e-8
        for name in PosteriorReference.statistics
            @test coarse.moments[name] ≈ fine.moments[name] atol=1e-8
        end
        # For one date with zero presample values, the AR likelihood contains
        # no lagged data. Its posterior must stay exactly at its normal prior
        # restricted to (-1,1), independently of the other unknowns.
        ar_variance = 0.3
        with_lags = PosteriorReference.one_date(y; prior..., ar_order=1,
            ar_prior_variance=ar_variance, initial=:zero)
        expected_ar_second = var(truncated(Normal(0, sqrt(ar_variance)), -1, 1))
        @test with_lags.moments.factor_ar_squared ≈ expected_ar_second atol=1e-12
        @test with_lags.moments.error_ar_squared ≈ expected_ar_second atol=1e-12
        @test with_lags.moments.loading ≈ fine.moments.loading atol=1e-8
        @test with_lags.moments.factor_ar_loading_squared ≈
              expected_ar_second * fine.moments.loading_squared atol=1e-8
        @test with_lags.moments.error_ar_variance ≈
              expected_ar_second * fine.moments.variance atol=1e-8
    end
end

@testset "Full Gibbs posterior agrees with direct integration" begin
    seeds = (412, 815, 1237)
    # Different coefficient and variance priors, and both signs of y, exercise
    # prior dependence and the joint sign folding of factor and loading.
    cases = ((1.2, 0, (beta_prior_variance=0.5, variance_shape=4.0, variance_scale=1.0)),
             (-0.8, 1, (beta_prior_variance=[2.0, 0.4], variance_shape=6.0, variance_scale=3.0)))
    for (y, order, prior) in cases
        reference = PosteriorReference.one_date(y; prior..., ar_order=order,
                                                ar_prior_variance=0.3, initial=:zero)
        loading_median = PosteriorReference.posterior_quantile(y, 0.5, :loading; prior...)
        variance_quartile = PosteriorReference.posterior_quantile(y, 0.75, :variance; prior...)
        for estimator in (KN1LevelEstimator, OW1LevelEstimator)
            outputs = [estimator(MersenneTwister(seed), reshape([y], 1, 1),
                       DFMStruct(order, order, 6000, 1000); prior...,
                       ar_prior_variance=0.3, initial=:zero) for seed in seeds]
            samples = one_date_chain_statistics.(outputs)
            for output in outputs
                @test all(>(0), output.B[:, 2])
            end
            for name in PosteriorReference.statistics
                order == 0 && name in (:factor_ar_squared, :error_ar_squared,
                    :factor_ar_loading_squared, :error_ar_variance) && continue
                chains = hcat([sample[name] for sample in samples]...)
                check_posterior_mean(chains, reference.moments[name])
            end
            # Testing CDFs at independently integrated quantiles avoids treating
            # the autocorrelated empirical quantile as an ordinary sample mean.
            check_posterior_mean(hcat([s.loading .<= loading_median for s in samples]...), 0.5)
            check_posterior_mean(hcat([s.variance .<= variance_quartile for s in samples]...), 0.75)
            if order > 0
                check_posterior_mean(hcat([output.P[:, 1] for output in outputs]...), 0.0)
                check_posterior_mean(hcat([output.P2[:, 1] for output in outputs]...), 0.0)
            end
        end
    end
end

@testset "Stationary full posterior includes the initial density" begin
    y = 0.8
    prior = (beta_prior_variance=0.5, variance_shape=4.0, variance_scale=1.0,
             ar_prior_variance=0.3)
    coarse = PosteriorReference.one_date(y; prior..., ar_order=1, initial=:stationary,
                                         order=120, ar_quadrature_order=50)
    reference = PosteriorReference.one_date(y; prior..., ar_order=1, initial=:stationary,
                                            order=160, ar_quadrature_order=80)
    @test coarse.mass ≈ reference.mass rtol=2e-5
    checked_statistics = filter(!=(:factor_squared), PosteriorReference.statistics)
    for name in checked_statistics
        @test coarse.moments[name] ≈ reference.moments[name] atol=2e-5
    end
    # Unlike zero initialization, the one-date stationary likelihood depends
    # on both AR coefficients through their unconditional process variances.
    ar_prior_second = var(truncated(Normal(0, sqrt(prior.ar_prior_variance)), -1, 1))
    @test abs(reference.moments.factor_ar_squared - ar_prior_second) > 0.01
    @test abs(reference.moments.error_ar_squared - ar_prior_second) > 0.01

    for estimator in (KN1LevelEstimator, OW1LevelEstimator)
        outputs = [estimator(MersenneTwister(seed), reshape([y], 1, 1),
                   DFMStruct(1, 1, 8000, 1500); prior..., initial=:stationary)
                   for seed in (219, 518, 1103)]
        samples = one_date_chain_statistics.(outputs)
        for name in checked_statistics
            chains = hcat([sample[name] for sample in samples]...)
            check_posterior_mean(chains, reference.moments[name]; integration_error=5e-5)
        end
        # The sign symmetry of each AR coefficient is retained after sign
        # identification of the loading. It must not be folded to positive ARs.
        check_posterior_mean(hcat([output.P[:, 1] for output in outputs]...), 0.0)
        check_posterior_mean(hcat([output.P2[:, 1] for output in outputs]...), 0.0)
    end
    # E[f²] exists here, but its variance need not: AR priors allow arbitrarily
    # near-unit roots. A normal-theory Monte Carlo error for sample f² would
    # therefore be unjustified, so this statistic is excluded above.
end

@testset "Full two-factor hierarchical posterior agrees with integration" begin
    y = 0.5
    prior = (beta_prior_variance=[0.3, 0.4, 0.9], variance_shape=4.0, variance_scale=1.0)
    coarse = PosteriorReference.two_factors(y; prior..., loading_order=70, variance_order=120)
    reference = PosteriorReference.two_factors(y; prior..., loading_order=100, variance_order=160)
    @test coarse.mass ≈ reference.mass rtol=1e-8
    for name in PosteriorReference.hierarchical_statistics
        @test coarse.moments[name] ≈ reference.moments[name] atol=1e-8
    end
    @test reference.moments.factor_cross < -0.04
    specification = HDFMStruct(2, [1, 1], [1 1], [0, 0], [0], 6000, 1000)
    # This checks all three factor-update routes: simultaneous state-space,
    # simultaneous precision, and Otrok–Whiteman sequential precision.
    for sampler in (:state_space, :precision, :sequential_precision)
        outputs = [KN2LevelEstimator(MersenneTwister(seed), reshape([y], 1, 1),
                   specification; prior..., factor_sampler=sampler, initial=:stationary)
                   for seed in (97, 804, 1159)]
        samples = map(outputs) do output
            a, b1, b2 = output.B[:, 1], output.B[:, 2], output.B[:, 3]
            s, f1, f2 = output.S[:, 1], vec(output.F[:, 1, :]), vec(output.F[:, 2, :])
            (intercept=a, loading1=b1, loading2=b2,
             loading1_squared=b1.^2, loading2_squared=b2.^2,
             variance=s, precision=1 ./ s, factor1=f1, factor2=f2,
             factor1_squared=f1.^2, factor2_squared=f2.^2,
             factor_cross=f1 .* f2, contribution1=b1 .* f1,
             contribution2=b2 .* f2, loading_cross=b1 .* b2)
        end
        for output in outputs
            @test all(>(0), output.B[:, 2:3])
        end
        for name in PosteriorReference.hierarchical_statistics
            chains = hcat([sample[name] for sample in samples]...)
            check_posterior_mean(chains, reference.moments[name])
        end
    end
end

@testset "Two-date full posterior checks transitions and initial density" begin
    y = [1.2, -0.9]
    prior = (beta_prior_variance=[0.2, 0.6], variance_shape=4.0, variance_scale=1.0,
             ar_prior_variance=0.2)
    for initial in (:zero, :stationary)
        coarse = PosteriorReference.two_dates(y; prior..., initial,
                                              order=120, ar_quadrature_order=45)
        reference = PosteriorReference.two_dates(y; prior..., initial,
                                                 order=160, ar_quadrature_order=70)
        @test coarse.mass ≈ reference.mass rtol=2e-5
        for name in PosteriorReference.two_date_statistics
            @test coarse.moments[name] ≈ reference.moments[name] atol=2e-5
        end
        # Opposite-signed observations give nonzero signed AR posterior means.
        # This directly checks that the transition likelihood enters AR updates.
        @test reference.moments.factor_ar < -0.05
        @test reference.moments.error_ar < -0.05
        for estimator in (KN1LevelEstimator, OW1LevelEstimator)
            outputs = [estimator(MersenneTwister(seed), reshape(y, 2, 1),
                       DFMStruct(1, 1, 10000, 2000); prior..., initial)
                       for seed in (117, 829, 1306)]
            samples = map(outputs) do output
                a, b, s = output.B[:, 1], output.B[:, 2], output.S[:, 1]
                f1, f2 = output.F[1, :], output.F[2, :]
                ψ, φ = output.P[:, 1], output.P2[:, 1]
                (intercept=a, loading=b, loading_squared=b.^2, variance=s,
                 precision=1 ./ s, factor1=f1, factor2=f2,
                 contribution1=b .* f1, contribution2=b .* f2,
                 contribution_cross=b.^2 .* f1 .* f2,
                 factor_ar=ψ, error_ar=φ, factor_ar_squared=ψ.^2,
                 error_ar_squared=φ.^2, ar_cross=ψ .* φ,
                 factor_ar_loading_squared=ψ .* b.^2, error_ar_variance=φ .* s)
            end
            for name in PosteriorReference.two_date_statistics
                chains = hcat([sample[name] for sample in samples]...)
                check_posterior_mean(chains, reference.moments[name]; integration_error=5e-5)
            end
        end
    end
end
