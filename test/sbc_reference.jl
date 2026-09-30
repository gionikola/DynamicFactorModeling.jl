# Only unrelated test seeds appear here; this file is not an SBC campaign and
# uses no reserved study seed.
module SBCReferenceChecks

using Test
using Random
using LinearAlgebra
include(joinpath(@__DIR__, "reference", "sbc_reference.jl"))
const Reference = SBCReference

parameters(; order=0, intercepts=[0.0], loadings=[2.0], error_variances=[3.0],
    factor_ar=0.0, error_ar=zeros(length(intercepts))) =
    (; order, intercepts, loadings, error_variances, factor_ar, error_ar)

@testset "Analytic stationary AR covariance" begin
    @test Reference.stationary_ar_covariance(0.0, 3) == Matrix{Float64}(I, 3, 3)
    @test Reference.stationary_ar_covariance(0.0, 2; innovation_variance=3.0) == [3.0 0.0; 0.0 3.0]
    expected = [4.0 -2.0 1.0; -2.0 4.0 -2.0; 1.0 -2.0 4.0]
    @test Reference.stationary_ar_covariance(-0.5, 3; innovation_variance=3.0) == expected
    @test Reference.stationary_ar_covariance(0.5, 3) ≈ (4/3) .* [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]
    @test Reference.stationary_ar_covariance(0.5, 1)[1, 1] == 4/3
    for coefficient in (-1.0, 1.0, Inf, NaN)
        @test_throws ArgumentError Reference.stationary_ar_covariance(coefficient, 2)
    end
    @test_throws ArgumentError Reference.stationary_ar_covariance(0.1, 0)
    @test_throws ArgumentError Reference.stationary_ar_covariance(0.1, 2; innovation_variance=0.0)
    @test_throws ArgumentError Reference.stationary_ar_covariance(0.9, 2; innovation_variance=floatmax(Float64))
end

@testset "Independent Gaussian density and column-major packing" begin
    # One-date AR(0): the intercept is fixed, so variance is b²+v=7,
    # not 107 (which would incorrectly integrate its default prior).
    p = parameters(intercepts=[0.25])
    @test Reference.observation_moments(p, 1).covariance == reshape([7.0], 1, 1)
    expected = -0.5 * log(2pi * 7) - (1.5-0.25)^2 / 14
    @test Reference.marginal_loglikelihood(reshape([1.5], 1, 1), p) ≈ expected atol=1e-14
    # Two-date AR(1), opposite factor/error autocorrelations:
    # C=[28/3 2/3; 2/3 28/3], det(C)=260/3, y'C^-1*y=37/65.
    p_ar = parameters(order=1, factor_ar=0.5, error_ar=[-0.5])
    @test Reference.observation_moments(p_ar, 2).covariance ≈ [28/3 2/3; 2/3 28/3]
    expected_ar = -log(2pi) - 0.5log(260/3) - 0.5 * (37/65)
    @test Reference.marginal_loglikelihood(reshape([1.0, -2.0], 2, 1), p_ar) ≈ expected_ar atol=1e-14
    # Two series use one shared factor. Negative cross-series covariance is
    # intentional; all dates of the first series precede those of the second.
    p_two = parameters(intercepts=[1.0, -2.0], loadings=[2.0, -1.0], error_variances=[3.0, 4.0])
    moments = Reference.observation_moments(p_two, 2)
    @test moments.mean == [1.0, 1.0, -2.0, -2.0]
    @test moments.covariance == [7.0 0.0 -2.0 0.0; 0.0 7.0 0.0 -2.0;
                                -2.0 0.0 5.0 0.0; 0.0 -2.0 0.0 5.0]
    # Each date has 2×2 covariance [7 -2; -2 5], determinant 31.
    # Residual rows [1,2] and [-1,3] have inverse quadratics 41/31,56/31.
    data = [2.0 0.0; 0.0 1.0]
    expected_two = -2log(2pi) - log(31) - 0.5 * (97/31)
    @test Reference.marginal_loglikelihood(data, p_two) ≈ expected_two atol=1e-14
    reversed = parameters(intercepts=reverse(p_two.intercepts), loadings=reverse(p_two.loadings),
        error_variances=reverse(p_two.error_variances))
    @test Reference.marginal_loglikelihood(data[:, 2:-1:1], reversed) ≈ expected_two atol=1e-14
    # Independent exact-rational fixture, checked by Fraction elimination and
    # a decomposition into even/odd time contrasts. No reference helper is used
    # to obtain these expected entries, determinant, or inverse quadratic.
    mixed = parameters(order=1, intercepts=[1.0, -2.0], loadings=[2.0, -1.0],
        error_variances=[0.5, 1.5], factor_ar=0.5, error_ar=[-0.25, 0.75])
    expected_covariance = [88/15 38/15 -8/3 -4/3;
                          38/15 88/15 -4/3 -8/3;
                          -8/3 -4/3 100/21 68/21;
                          -4/3 -8/3 68/21 100/21]
    @test Reference.observation_moments(mixed, 2).covariance ≈ expected_covariance atol=2e-15
    # Series-major residual is [1/5, -1/10, -2/5, 3/10].
    mixed_data = [1.2 -2.4; 0.9 -1.7]
    exact_log_density = -2log(2pi) - 0.5log(53248/315) - 61281/665600
    @test Reference.marginal_loglikelihood(mixed_data, mixed) ≈ exact_log_density atol=1e-14
    @test exact_log_density ≈ -6.332894385592469 atol=1e-14
end

@testset "Conditional observation density given the factor path" begin
    # AR(0) residuals are [1/4, -3/4], with innovation variance 3.
    # Their inverse quadratic is (1/16 + 9/16)/3 = 5/24.
    p = parameters(intercepts=[0.25])
    data = reshape([1.5, -1.0], 2, 1)
    factors = [0.5, -0.25]
    expected = -log(2pi) - log(3) - 5/48
    @test Reference.conditional_loglikelihood(data, p, factors) ≈ expected atol=1e-14

    # Independent exact-rational AR(1) fixture. Residual columns are
    # [-4/5, 2/5] and [1/10, 1/20]. The block-diagonal error covariance
    # has determinant 48/35, and the inverse quadratic is 77/60.
    mixed = parameters(order=1, intercepts=[1.0, -2.0], loadings=[2.0, -1.0],
        error_variances=[0.5, 1.5], factor_ar=0.5, error_ar=[-0.25, 0.75])
    mixed_data = [1.2 -2.4; 0.9 -1.7]
    exact_log_density = -2log(2pi) - 0.5log(48/35) - 77/120
    actual = Reference.conditional_loglikelihood(mixed_data, mixed, factors)
    @test actual ≈ exact_log_density atol=1e-14
    @test exact_log_density ≈ -4.475347274194596 atol=1e-14
    # Conditioning on F removes its AR density, but not its contribution to y.
    other_factor_ar = merge(mixed, (; factor_ar=-0.9))
    @test Reference.conditional_loglikelihood(mixed_data, other_factor_ar, factors) == actual
    @test Reference.conditional_loglikelihood(mixed_data, mixed, zeros(2)) != actual
    flipped = merge(mixed, (; loadings=-mixed.loadings))
    @test Reference.conditional_loglikelihood(mixed_data, flipped, -factors) == actual
    reversed = merge(mixed, (; intercepts=reverse(mixed.intercepts),
        loadings=reverse(mixed.loadings), error_variances=reverse(mixed.error_variances),
        error_ar=reverse(mixed.error_ar)))
    @test Reference.conditional_loglikelihood(mixed_data[:, 2:-1:1], reversed, factors) ≈ actual atol=1e-14

    # One date still needs the stationary initial density: variance 3/(1-.5²)=4.
    one_date = parameters(order=1, factor_ar=0.8, error_ar=[0.5])
    @test Reference.conditional_loglikelihood(ones(1, 1), one_date, [0.0]) ≈
        -0.5log(8pi) - 1/8 atol=1e-14
    # A large finite variance is valid; avoid forming 2π times that variance.
    large_variance = parameters(error_variances=[floatmax(Float64)])
    @test Reference.conditional_loglikelihood(zeros(1, 1), large_variance, [0.0]) ≈
        -0.5 * (log(2pi) + log(floatmax(Float64))) atol=1e-14

    @test_throws DimensionMismatch Reference.conditional_loglikelihood(zeros(2, 2), p, factors)
    @test_throws DimensionMismatch Reference.conditional_loglikelihood(data, p, factors[1:1])
    @test_throws ArgumentError Reference.conditional_loglikelihood(zeros(0, 1), p, Float64[])
    @test_throws ArgumentError Reference.conditional_loglikelihood(vec(data), p, factors)
    @test_throws ArgumentError Reference.conditional_loglikelihood(data, p, reshape(factors, 2, 1))
    for invalid in (NaN, Inf)
        @test_throws ArgumentError Reference.conditional_loglikelihood(reshape([invalid, 0.0], 2, 1), p, factors)
        @test_throws ArgumentError Reference.conditional_loglikelihood(data, p, [invalid, 0.0])
    end
    @test_throws ArgumentError Reference.conditional_loglikelihood(data,
        parameters(order=1, error_ar=[1.0]), factors)
    @test_throws ArgumentError Reference.conditional_loglikelihood(data,
        parameters(error_variances=[0.0]), factors)
    @test_throws ArgumentError Reference.conditional_loglikelihood(zeros(1, 1),
        parameters(loadings=[floatmax(Float64)]), [2.0])
    @test_throws ArgumentError Reference.conditional_loglikelihood(reshape([floatmax(Float64)], 1, 1),
        p, [0.0])
end

@testset "Joint sign folding preserves signal, dynamics, and marginal density" begin
    p = parameters(order=1, intercepts=[0.2, -0.3], loadings=[-2.0, 3.0],
        error_variances=[0.5, 0.7], factor_ar=0.5, error_ar=[0.2, -0.3])
    initial, innovations, factors = 2.0, [0.0, 2.0, -0.5], [1.0, 2.5, 0.75]
    old_signal = factors * p.loadings' .+ p.intercepts'
    folded = Reference.fold_factor_signs(p, factors, initial, innovations)
    @test folded.flipped && folded.parameters.loadings == [2.0, -3.0]
    @test folded.factors == -factors && folded.initial_factor == -initial
    @test folded.innovations == -innovations
    @test folded.factors * folded.parameters.loadings' .+ folded.parameters.intercepts' == old_signal
    @test folded.factors ≈ p.factor_ar .* [folded.initial_factor; folded.factors[1:end-1]] + folded.innovations
    @test folded.parameters.factor_ar == p.factor_ar && folded.parameters.error_ar == p.error_ar
    @test Reference.marginal_loglikelihood(old_signal, p) ≈ Reference.marginal_loglikelihood(old_signal, folded.parameters)
    @test Reference.conditional_loglikelihood(old_signal, p, factors) ==
        Reference.conditional_loglikelihood(old_signal, folded.parameters, folded.factors)
    @test p.loadings == [-2.0, 3.0] && factors == [1.0, 2.5, 0.75] # Inputs are unchanged.
    @test !Reference.fold_factor_signs(p, factors, initial, innovations; anchor=2).flipped
    zero_anchor = merge(p, (; loadings=[0.0, -3.0]))
    @test !Reference.fold_factor_signs(zero_anchor, factors, initial, innovations).flipped
    @test_throws ArgumentError Reference.fold_factor_signs(p, factors, initial, innovations; anchor=3)
    @test_throws DimensionMismatch Reference.fold_factor_signs(p, factors, initial, innovations[1:2])
end

@testset "Default-prior generator uses explicit, reproducible streams" begin
    # No empirical second-moment tests: stationary AR-prior variances and the
    # IG(2,1) second moment are infinite. Verify law construction and recursions.
    for order in (0, 1)
        seed = 920_100 + order
        first = Reference.generate_case(MersenneTwister(seed); order)
        second = Reference.generate_case(MersenneTwister(seed); order)
        @test isequal(first, second)
        @test size(first.data) == (6, 2) && length(first.factors) == 6
        @test first.parameters.loadings[1] >= 0
        @test all(>(0), first.parameters.error_variances)
        @test abs(first.parameters.factor_ar) < 1 && all(x -> abs(x) < 1, first.parameters.error_ar)
        @test first.data == first.signal + first.errors
        @test first.factors ≈ first.parameters.factor_ar .* [first.initial_factor; first.factors[1:end-1]] + first.factor_innovations
        for i in 1:2
            @test first.errors[:, i] ≈ first.parameters.error_ar[i] .* [first.initial_errors[i]; first.errors[1:end-1, i]] + first.error_innovations[:, i]
        end
        @test isfinite(Reference.marginal_loglikelihood(first.data, first.parameters))
        if order == 0
            @test first.factors == first.factor_innovations
        end
        @test Reference.generate_case(MersenneTwister(seed+10); order).data != first.data
    end
    rng, copy_rng = MersenneTwister(920_201), MersenneTwister(920_201)
    sampled = Reference.draw_prior_parameters(rng; order=0, series=2)
    @test sampled.intercepts == 10 .* randn(copy_rng, 2)
    @test sampled.loadings == 10 .* randn(copy_rng, 2)
    @test sampled.factor_ar == 0 && sampled.error_ar == zeros(2)
    @test_throws ArgumentError Reference.generate_case(MersenneTwister(1); order=2)
    @test_throws ArgumentError Reference.generate_case(MersenneTwister(1); series=0)
    @test_throws ArgumentError Reference.generate_case(MersenneTwister(1); dates=0)
end

@testset "Randomized ranks and invalid quantities" begin
    rng, untouched = MersenneTwister(920_301), MersenneTwister(920_301)
    @test Reference.randomized_rank(rng, 2.0, [1.0, 3.0]) == 1
    @test Reference.randomized_rank(rng, 2.0, Float64[]) == 0
    @test rand(rng) == rand(untouched)
    for seed in 920_310:920_320
        rank_rng, uniform_rng = MersenneTwister(seed), MersenneTwister(seed)
        @test Reference.randomized_rank(rank_rng, 2.0, [1.0, 2.0, 2.0, 4.0]) == 1 + rand(uniform_rng, 0:2)
    end
    @test Reference.randomized_rank(MersenneTwister(1), 0.0, [prevfloat(0.0), nextfloat(0.0)]) == 1
    @test_throws ArgumentError Reference.randomized_rank(MersenneTwister(1), NaN, [0.0])
    @test_throws ArgumentError Reference.randomized_rank(MersenneTwister(1), 0.0, [Inf])
    @test_throws DimensionMismatch Reference.marginal_loglikelihood(zeros(2, 2), parameters())
    @test_throws ArgumentError Reference.marginal_loglikelihood(zeros(0, 1), parameters())
    @test_throws ArgumentError Reference.marginal_loglikelihood(reshape([NaN], 1, 1), parameters())
    @test_throws ArgumentError Reference.observation_moments(parameters(error_variances=[-1.0]), 2)
    @test_throws ArgumentError Reference.observation_moments(parameters(order=0, factor_ar=0.2), 2)
    @test_throws ArgumentError Reference.observation_moments(parameters(order=1, factor_ar=1.0), 2)
    @test_throws DimensionMismatch Reference.observation_moments(parameters(loadings=[1.0, 2.0]), 2)
end

end
