using Test
using LinearAlgebra
using Random

if !isdefined(@__MODULE__, :GaussianLocationReference)
    include(joinpath(@__DIR__, "reference", "gaussian_location_reference.jl"))
end
using .GaussianLocationReference

@testset "Independent AR path covariance" begin
    for initial in (:zero, :stationary), dates in (1, 7)
        @test ar_path_covariance(Float64[], dates; innovation_variance=0.4, initial) ≈
              0.4 .* Matrix{Float64}(I, dates, dates)
        for phi in (-0.6, 0.4)
            expected = [0.7 * phi^abs(i - j) / (1 - phi^2) *
                        (initial == :zero ? 1 - phi^(2 * min(i, j)) : 1)
                        for i in 1:dates, j in 1:dates]
            @test ar_path_covariance([phi], dates; innovation_variance=0.7, initial) ≈ expected
        end
    end

    # Closed-form stationary AR(2) variance, including a sample shorter than p.
    phi1, phi2, variance = 0.55, -0.2, 0.7
    gamma0 = variance * (1 - phi2) /
             ((1 + phi2) * ((1 - phi2)^2 - phi1^2))
    gamma1 = phi1 * gamma0 / (1 - phi2)
    gamma2 = phi1 * gamma1 + phi2 * gamma0
    expected = [gamma0 gamma1 gamma2; gamma1 gamma0 gamma1; gamma2 gamma1 gamma0]
    @test ar_path_covariance([phi1, phi2], 3; innovation_variance=variance) ≈ expected
    @test only(ar_path_covariance([phi1, phi2], 1; innovation_variance=variance)) ≈ gamma0

    # With zero presample values, innovation residuals are triangular equations.
    # Their independently formed precision must invert the reference covariance.
    dates = 6
    coefficients = [0.55, -0.2, 0.1]
    residual_map = Matrix{Float64}(I, dates, dates)
    for t in 1:dates, lag in 1:min(length(coefficients), t - 1)
        residual_map[t, t - lag] -= coefficients[lag]
    end
    covariance = ar_path_covariance(coefficients, dates; innovation_variance=variance, initial=:zero)
    @test covariance * (residual_map' * residual_map / variance) ≈ I atol=2e-14
    @test_throws ArgumentError ar_path_covariance([1.1], 3)
    @test_throws ArgumentError ar_path_covariance([1.0], 3)
    @test_throws ArgumentError ar_path_covariance([0.5], 0)
    @test_throws ArgumentError ar_path_covariance([0.5], 3; initial=:unknown)
    @test_throws ArgumentError ar_path_covariance([NaN], 3)
    @test_throws ArgumentError ar_path_covariance([0.5], 3; innovation_variance=0)
    @test isposdef(ar_path_covariance([1.1], 3; initial=:zero))
end

@testset "Analytic intercept and factor posterior" begin
    reference = gaussian_location_posterior(fill(5.0, 1, 1), fill(2.0, 1, 1),
        [Float64[]], [Float64[]], [3.0]; intercept_prior_variance=4.0)
    @test reference.mean ≈ [20 / 11, 10 / 11]
    @test reference.covariance ≈ [28 / 11 -8 / 11; -8 / 11 7 / 11]
    @test reference.precision ≈ [7 / 12 2 / 3; 2 / 3 7 / 3]
    intercepts = conditional_intercepts(reference, fill(0.2, 1, 1))
    factors = conditional_factors(reference, [0.2])
    @test only(intercepts.mean) ≈ (4 / 7) * (5 - 2 * 0.2)
    @test only(intercepts.covariance) ≈ 12 / 7
    @test only(factors.mean) ≈ (2 / 7) * (5 - 0.2)
    @test only(factors.covariance) ≈ 3 / 7
    @test factors.factors == reshape(factors.mean, 1, 1)
    rate = alternating_gibbs_rate(reference)
    @test rate.rate ≈ 16 / 49
    @test rate.slow_mode_iact ≈ 65 / 33

    independent = gaussian_location_posterior(ones(3, 2), zeros(2, 2),
        [[0.4], Float64[]], [Float64[], [0.2]], [0.5, 0.8])
    @test alternating_gibbs_rate(independent).rate == 0
    @test alternating_gibbs_rate(independent).slow_mode_iact == 1

    # Stationary AR(1) density: the first value has its unconditional variance,
    # followed by ordinary innovation densities. This checks the initial law
    # without using any covariance or precision from the reference itself.
    data = reshape([0.1, -0.4, 0.6], 3, 1)
    paths = reshape([0.5, 0.1, -0.3], 3, 1)
    intercept = [0.2]
    factor_phi, error_phi, error_variance = 0.65, -0.2, 0.4
    stationary = gaussian_location_posterior(data, fill(1.3, 1, 1),
        [[factor_phi]], [[error_phi]], [error_variance]; intercept_prior_variance=3.0)
    residual = vec(data .- intercept' .- 1.3 .* paths)
    direct_density = -intercept[1]^2 / (2 * 3.0) -
        paths[1]^2 * (1 - factor_phi^2) / 2 -
        residual[1]^2 * (1 - error_phi^2) / (2 * error_variance)
    for t in 2:3
        direct_density -= (paths[t] - factor_phi * paths[t - 1])^2 / 2
        direct_density -= (residual[t] - error_phi * residual[t - 1])^2 / (2 * error_variance)
    end
    @test joint_logdensity(stationary, intercept, paths) ≈ direct_density
end

function zero_presample_logdensity(data, loadings, factor_ar, error_ar,
                                   error_variances, prior_variances, intercepts, factors)
    # Evaluate actual innovations; this test does not use the reference matrices.
    result = -0.5 * sum(abs2.(intercepts) ./ prior_variances)
    for (paths, coefficients, variances) in (
        (factors, factor_ar, ones(length(factor_ar))),
        (data .- intercepts' .- factors * loadings', error_ar, error_variances))
        for column in axes(paths, 2), t in axes(paths, 1)
            innovation = paths[t, column]
            for lag in 1:min(length(coefficients[column]), t - 1)
                innovation -= coefficients[column][lag] * paths[t - lag, column]
            end
            result -= 0.5 * innovation^2 / variances[column]
        end
    end
    return result
end

@testset "Joint density, conditioning, and Gaussian Gibbs coupling" begin
    rng = MersenneTwister(67125)
    dates, nseries, nfactors = 5, 3, 2
    data = randn(rng, dates, nseries)
    loadings = [1.2 0.0; -0.7 0.8; 0.0 1.1]
    factor_ar = [[0.6, -0.15], Float64[]]
    error_ar = [Float64[], [-0.3], [0.3, 0.1]]
    variances = [0.4, 0.6, 0.3]
    prior_variances = [3.0, 7.0, 9.0]
    for initial in (:zero, :stationary)
        reference = gaussian_location_posterior(data, loadings, factor_ar, error_ar,
            variances; initial, intercept_prior_variance=prior_variances)
        a, f = reference.intercept_indices, reference.factor_indices
        @test reference.precision * reference.covariance ≈ I atol=1e-12
        @test reference.precision * reference.mean ≈ reference.information atol=1e-12
        @test reference.factor_ranges == [4:8, 9:13]

        # Integrating out the factors yields a different Gaussian regression
        # for the intercepts, with correlated observations across series.
        factor_covariance = zeros(dates * nfactors, dates * nfactors)
        error_covariance = zeros(dates * nseries, dates * nseries)
        for k in 1:nfactors
            indices = ((k - 1) * dates + 1):(k * dates)
            factor_covariance[indices, indices] = reference.factor_prior_covariances[k]
        end
        for i in 1:nseries
            indices = ((i - 1) * dates + 1):(i * dates)
            error_covariance[indices, indices] = reference.error_covariances[i]
        end
        factor_design = kron(loadings, Matrix{Float64}(I, dates, dates))
        intercept_design = kron(Matrix{Float64}(I, nseries, nseries), ones(dates))
        marginal_error = error_covariance + factor_design * factor_covariance * factor_design'
        marginal_precision = Diagonal(1 ./ prior_variances) +
                             intercept_design' * (marginal_error \ intercept_design)
        marginal_information = intercept_design' * (marginal_error \ vec(data))
        @test reference.mean[a] ≈ marginal_precision \ marginal_information
        @test reference.covariance[a, a] ≈ inv(marginal_precision)
        intercepts, factors = randn(rng, nseries), randn(rng, dates, nfactors)
        point = [intercepts; vec(factors)]
        density = joint_logdensity(reference, intercepts, factors)
        zero_density = joint_logdensity(reference, zeros(nseries), zeros(dates, nfactors))
        @test density - zero_density ≈ dot(reference.information, point) -
              dot(point, reference.precision * point) / 2 atol=1e-11
        if initial == :zero
            @test density ≈ zero_presample_logdensity(data, loadings, factor_ar,
                error_ar, variances, prior_variances, intercepts, factors) atol=1e-11
        end

        # Condition the independently returned joint covariance, rather than
        # repeating the precision-block formulas used by the reference helper.
        covariance = reference.covariance
        a_gain = covariance[a, f] / covariance[f, f]
        f_gain = covariance[f, a] / covariance[a, a]
        a_conditional = conditional_intercepts(reference, factors)
        f_conditional = conditional_factors(reference, intercepts)
        @test a_conditional.mean ≈ reference.mean[a] + a_gain * (vec(factors) - reference.mean[f])
        @test f_conditional.mean ≈ reference.mean[f] + f_gain * (intercepts - reference.mean[a])
        @test a_conditional.covariance ≈ covariance[a, a] - a_gain * covariance[f, a]
        @test f_conditional.covariance ≈ covariance[f, f] - f_gain * covariance[a, f]

        # The transition matrix and marginal-covariance canonical correlations
        # are two different checks of the precision-whitened SVD rate formula.
        rate = alternating_gibbs_rate(reference)
        transition = reference.factor_gain * reference.intercept_gain
        @test rate.rate ≈ maximum(abs, eigvals(transition))
        a_root, f_root = cholesky(Symmetric(covariance[a, a])).L,
                         cholesky(Symmetric(covariance[f, f])).L
        covariance_coupling = (a_root \ covariance[a, f]) / f_root'
        @test rate.canonical_correlations ≈ svdvals(covariance_coupling)
        @test transition * vec(rate.factor_mean_direction) ≈
              rate.rate .* vec(rate.factor_mean_direction)
        @test reference.intercept_gain * reference.factor_gain * rate.intercept_mean_direction ≈
              rate.rate .* rate.intercept_mean_direction
        @test transition' * vec(rate.factor_mode_weights) ≈ rate.rate .* vec(rate.factor_mode_weights)
        @test (reference.intercept_gain * reference.factor_gain)' * rate.intercept_mode_weights ≈
              rate.rate .* rate.intercept_mode_weights
        weights = vec(rate.factor_mode_weights)
        @test dot(weights, transition * covariance[f, f] * weights) /
              dot(weights, covariance[f, f] * weights) ≈ rate.rate

        # A full stationary Gibbs sweep must preserve the known joint covariance.
        na, nf = length(a), length(f)
        sweep = [zeros(na, na) reference.intercept_gain;
                 zeros(nf, na) transition]
        a_noise, f_noise = a_conditional.covariance, f_conditional.covariance
        noise = [a_noise a_noise * reference.factor_gain';
                 reference.factor_gain * a_noise f_noise + reference.factor_gain * a_noise * reference.factor_gain']
        @test sweep * covariance * sweep' + noise ≈ covariance atol=2e-12
    end

    @test_throws DimensionMismatch gaussian_location_posterior(data, zeros(4, 2), factor_ar, error_ar, variances)
    @test_throws DimensionMismatch gaussian_location_posterior(data, loadings, factor_ar[1:1], error_ar, variances)
    @test_throws ArgumentError gaussian_location_posterior(data, loadings, factor_ar, error_ar, variances;
                                                         intercept_prior_variance=-1)
    reference = gaussian_location_posterior(data, loadings, factor_ar, error_ar, variances)
    @test_throws DimensionMismatch conditional_intercepts(reference, zeros(4, 2))
    @test_throws DimensionMismatch conditional_factors(reference, zeros(2))
end
