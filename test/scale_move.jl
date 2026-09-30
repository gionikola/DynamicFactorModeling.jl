# Run separately when no simulation campaign is active:
# julia --threads=1 --project=. test/scale_move.jl
# Append --fast to run only small algebra/seeded-proposal tests, skipping the
# 20,000 independent prior-invariance trials while a campaign is running.
using Test, Random, Statistics, LinearAlgebra, Distributions
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :ScaleMove)
    include(joinpath(@__DIR__, "support", "scale_move.jl"))
end
if !isdefined(@__MODULE__, :GaussianLocationReference)
    include(joinpath(@__DIR__, "reference", "gaussian_location_reference.jl"))
end
using .ScaleMove
using .GaussianLocationReference: ar_path_covariance

function scale_test_logprior(path, loadings, covariance, variances)
    return -dot(path, covariance \ path) / 2 - sum(loadings.^2 ./ variances) / 2
end

@testset "Scale acceptance agrees with independent prior densities" begin
    mask = Bool[1 0; 1 1; 0 1]
    original_loadings = [0.0 0.0; -0.5 0.8; 0.0 -0.3]
    prior_variances = [0.4 0.0; 1.5 2.0; 0.0 0.7]
    coefficients = [[0.6], [0.7, -0.2]]
    for initial in (:stationary, :zero), dates in (1, 2, 5), seed in (31, 98, 117)
        factors = reshape(sin.(collect(1.0:(2 * dates))), dates, 2)
        loadings = copy(original_loadings)
        original_factors = copy(factors)
        original_signal = factors * loadings'
        result = rescale_factors!(MersenneTwister(seed), factors, loadings, coefficients;
            active_loadings=mask, initial, loading_prior_variance=prior_variances, step_size=0.25)
        rng = MersenneTwister(seed)
        expected_factors, expected_loadings = copy(original_factors), copy(original_loadings)
        for factor in 1:2
            eta, uniform = 0.25 * randn(rng), rand(rng)
            rows = findall(mask[:, factor])
            covariance = ar_path_covariance(coefficients[factor], dates; initial)
            old_f, old_b = original_factors[:, factor], original_loadings[rows, factor]
            new_f, new_b = exp(eta) .* old_f, exp(-eta) .* old_b
            variances = prior_variances[rows, factor]
            prior_change = scale_test_logprior(new_f, new_b, covariance, variances) -
                           scale_test_logprior(old_f, old_b, covariance, variances)
            expected_ratio = prior_change + (dates - length(rows)) * eta
            @test result.log_scale[factor] == eta
            @test result.log_ratio[factor] ≈ expected_ratio atol=1e-12 rtol=1e-12
            @test result.accepted[factor] == (log(uniform) < min(0.0, expected_ratio))
            if result.accepted[factor]
                expected_factors[:, factor] = new_f
                expected_loadings[rows, factor] = new_b
            end

            # The reverse transformation returns the original state and its
            # prior/Jacobian log ratio is the negative forward log ratio.
            @test exp(-eta) .* new_f ≈ old_f atol=1e-14
            @test exp(eta) .* new_b ≈ old_b atol=1e-14
            reverse_ratio = scale_test_logprior(old_f, old_b, covariance, variances) -
                            scale_test_logprior(new_f, new_b, covariance, variances) -
                            (dates - length(rows)) * eta
            @test reverse_ratio ≈ -result.log_ratio[factor] atol=1e-12
        end
        @test factors ≈ expected_factors atol=1e-14
        @test loadings ≈ expected_loadings atol=1e-14
        @test factors * loadings' ≈ original_signal atol=1e-14
        @test all(iszero, loadings[.!mask])
        @test sign.(loadings) == sign.(original_loadings)
        @test loadings[1, 1] == 0.0 # A free zero STILL counted in the Jacobian above.
    end
end

@testset "Augmented forward and reverse scale Jacobians" begin
    for dates in (1, 3, 6), active in (0, 2, 4), eta in (-0.7, 0.0, 0.4)
        factor = collect(1.0:dates)
        loading = collect(1.0:active)
        dimension = dates + active + 1
        derivative = zeros(dimension, dimension)
        derivative[1:dates, 1:dates] = exp(eta) .* Matrix{Float64}(I, dates, dates)
        derivative[1:dates, end] = exp(eta) .* factor
        rows = (dates + 1):(dates + active)
        derivative[rows, rows] = exp(-eta) .* Matrix{Float64}(I, active, active)
        derivative[rows, end] = -exp(-eta) .* loading
        derivative[end, end] = -1.0 # eta_reverse = -eta.
        @test logabsdet(derivative)[1] ≈ (dates - active) * eta atol=1e-14
    end
end

if !("--fast" in ARGS)
@testset "One-step invariance of a known Gaussian-prior distribution" begin
    # Each trial STARTS from an independent exact prior draw. Applying one MH
    # step must preserve its distribution. These are independent trials, not a
    # scale-only chain: that chain would leave products/directions fixed and
    # could not explore the full target from a single starting point.
    trials, dates = 10_000, 3
    variances = [0.4, 1.7]
    mask = trues(2, 1)
    for initial in (:zero, :stationary)
        coefficients = [[0.6]]
        covariance = ar_path_covariance(coefficients[1], dates; initial)
        prior_root = cholesky(Symmetric(covariance)).L
        scale = sqrt(covariance[1, 1])
        rng, move_rng = MersenneTwister(911), MersenneTwister(1327)
        old_statistics, new_statistics = zeros(trials, 5), zeros(trials, 5)
        accepted = 0
        normal_quantile, square_quantile = quantile(Normal(), 0.75), quantile(Chisq(1), 0.75)
        statistics_of(f, b) = [f[1]^2 / covariance[1, 1], b[1]^2 / variances[1],
            b[2]^2 / variances[2], f[1] / scale <= normal_quantile,
            b[1]^2 / variances[1] <= square_quantile]
        for trial in 1:trials
            factors = reshape(prior_root * randn(rng, dates), dates, 1)
            loadings = reshape(sqrt.(variances) .* randn(rng, 2), 2, 1)
            # Conditioning the first loading to be positive is the package's
            # sign convention. Its normalizing constant cancels in the ratio.
            loadings[1, 1] = abs(loadings[1, 1])
            old_statistics[trial, :] = statistics_of(factors, loadings)
            result = rescale_factors!(move_rng, factors, loadings, coefficients;
                active_loadings=mask, initial, loading_prior_variance=reshape(variances, 2, 1),
                step_size=0.35)
            accepted += only(result.accepted)
            new_statistics[trial, :] = statistics_of(factors, loadings)
        end
        @test 0 < accepted < trials
        for (column, expected) in enumerate((1.0, 1.0, 1.0, 0.75, 0.75))
            values = new_statistics[:, column]
            changes = values - old_statistics[:, column]
            @test abs(mean(values) - expected) <= 6 * std(values) / sqrt(trials)
            @test abs(mean(changes)) <= 6 * std(changes) / sqrt(trials) + 1e-12
        end
    end
end
end

@testset "Scale proposal validation and fixed entries" begin
    factors, loadings, coefficients = ones(2, 1), ones(1, 1), [[0.5]]
    for value in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError rescale_factors!(factors, loadings, coefficients;
            active_loadings=trues(1, 1), step_size=value)
        @test_throws ArgumentError rescale_factors!(factors, loadings, coefficients;
            active_loadings=trues(1, 1), loading_prior_variance=value)
    end
    @test_throws ArgumentError rescale_factors!(factors, loadings, coefficients;
        active_loadings=falses(1, 1))
    @test_throws DimensionMismatch rescale_factors!(factors, loadings, coefficients;
        active_loadings=trues(2, 1))
    @test_throws DimensionMismatch rescale_factors!(factors, loadings, coefficients;
        active_loadings=trues(1, 1), loading_prior_variance=ones(1, 2))
    @test_throws ArgumentError rescale_factors!(factors, loadings, [[1.0]];
        active_loadings=trues(1, 1), initial=:stationary)
    @test_throws ArgumentError rescale_factors!(factors, loadings, coefficients;
        active_loadings=trues(1, 1), initial=:conditional)
    @test_throws ArgumentError rescale_factors!(factors, loadings, [[NaN]];
        active_loadings=trues(1, 1))
    @test factors == ones(2, 1)
    @test loadings == ones(1, 1)

    # Scalar and equal matrix priors give identical proposals and decisions.
    first_f, second_f, first_b, second_b = copy(factors), copy(factors), copy(loadings), copy(loadings)
    first = rescale_factors!(MersenneTwister(87), first_f, first_b, coefficients;
        active_loadings=trues(1, 1), loading_prior_variance=3.0)
    second = rescale_factors!(second_f, second_b, coefficients;
        rng=MersenneTwister(87), active_loadings=trues(1, 1), loading_prior_variance=fill(3.0, 1, 1))
    @test first == second
    @test first_f == second_f && first_b == second_b
    @test ScaleMove.energy_change(0.0, 1000.0) == 0.0
end
