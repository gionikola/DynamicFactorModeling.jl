# Run separately: julia --threads=1 --project=test test/location_posterior.jl
# Append --variant=location_scale to check the additional scale move.
# These checks use public estimators and one RNG for every update.
# The former --public flag is accepted for command compatibility.

function location_test_variant(arguments)
    options = filter(!=("--public"), arguments)
    isempty(options) && return :location
    length(options) == 1 && startswith(only(options), "--variant=") ||
        throw(ArgumentError("use --variant=location or --variant=location_scale, optionally --public"))
    variant = Symbol(split(only(options), '='; limit=2)[2])
    variant in (:location, :location_scale) ||
        throw(ArgumentError("variant must be location or location_scale"))
    return variant
end
const LOCATION_TEST_VARIANT = location_test_variant(ARGS)
const LOCATION_TEST_LABEL = "Public " *
    (LOCATION_TEST_VARIANT == :location ? "location move" : "location and scale moves")

using Test, Random, Statistics, LinearAlgebra, DynamicFactorModeling
BLAS.set_num_threads(1)

for (name, directory, file) in ((:PosteriorReference, "reference", "posterior_reference.jl"),
                               (:SimulationDiagnostics, "support", "simulation_diagnostics.jl"))
    isdefined(@__MODULE__, name) || include(joinpath(@__DIR__, directory, file))
end

function location_posterior_case(observations, order, initial)
    spec = (nlevels=1, nfactors=[1], assignments=ones(Int, 1, 1),
            initial=initial, estimator_kind=:single)
    return (spec=spec, data=reshape(observations, :, 1),
            factor_orders=[order], error_orders=[order])
end

function location_posterior_chains(case, method, prior; burnin)
    return map((317, 826, 1309, 1841)) do seed
        estimator = method == "KN1" ? KN1LevelEstimator : OW1LevelEstimator
        settings = DFMStruct(only(case.factor_orders), only(case.error_orders), 8000, burnin)
        result = estimator(MersenneTwister(seed), case.data, settings;
            mixing_moves=LOCATION_TEST_VARIANT, initial=case.spec.initial, prior...)
        @test all(>(0), result.B[:, 2])
        result
    end
end

function check_location_posterior(chains, expected; integration_error=5e-5)
    batches = PosteriorReference.batch_summary(chains)
    diagnostic = SimulationDiagnostics.chain_diagnostics(chains)
    # Rank ESS checks mixing; the MCSE for the mean uses the original values.
    # The batch estimate provides a second treatment of serial dependence.
    mcse = max(batches.mcse, diagnostic.mcse_mean)
    @test isfinite(mcse)
    @test diagnostic.rhat < 1.02
    @test diagnostic.bulk_ess >= 400
    @test mcse < 0.035 * (1 + abs(expected))
    @test abs(batches.mean - expected) <= 6 * mcse + integration_error
    for chain in eachindex(batches.chain_means)
        @test abs(batches.chain_means[chain] - expected) <=
              7 * batches.chain_mcse[chain] + integration_error
    end
end

@testset "$LOCATION_TEST_LABEL in full AR(0) sampler agree with integration" begin
    y = 1.2
    prior = (beta_prior_variance=0.5, variance_shape=4.0, variance_scale=1.0)
    coarse = PosteriorReference.one_date(y; prior..., order=120)
    reference = PosteriorReference.one_date(y; prior..., order=160)
    names = (:intercept, :loading, :loading_squared, :variance, :precision,
             :factor, :factor_squared, :fitted_factor)
    @test coarse.mass ≈ reference.mass rtol=1e-8
    for name in names
        @test coarse.moments[name] ≈ reference.moments[name] atol=1e-8
    end
    median_loading = PosteriorReference.posterior_quantile(y, 0.5, :loading; prior...)
    upper_quartile_variance = PosteriorReference.posterior_quantile(y, 0.75, :variance; prior...)
    case = location_posterior_case([y], 0, :zero)
    results = location_posterior_chains(case, "KN1", prior; burnin=1500)
    samples = map(results) do result
        a, b, s, f = result.B[:, 1], result.B[:, 2], result.S[:, 1], vec(result.F)
        (intercept=a, loading=b, loading_squared=b.^2, variance=s, precision=1 ./ s,
         factor=f, factor_squared=f.^2, fitted_factor=b .* f)
    end
    for name in names
        check_location_posterior(hcat([sample[name] for sample in samples]...),
                                 reference.moments[name])
    end
    # CDF probabilities at independently integrated quantiles are means of
    # indicators, so their MCSE can account for serial dependence directly.
    check_location_posterior(hcat([sample.loading .<= median_loading for sample in samples]...), 0.5)
    check_location_posterior(hcat([sample.variance .<= upper_quartile_variance for sample in samples]...), 0.75)
end

@testset "$LOCATION_TEST_LABEL in full stationary AR(1) sampler agree with integration" begin
    y = [1.2, -0.9]
    prior = (beta_prior_variance=[0.2, 0.6], variance_shape=4.0,
             variance_scale=1.0, ar_prior_variance=0.2)
    coarse = PosteriorReference.two_dates(y; prior..., initial=:stationary,
                                         order=120, ar_quadrature_order=45)
    reference = PosteriorReference.two_dates(y; prior..., initial=:stationary,
                                            order=160, ar_quadrature_order=70)
    @test coarse.mass ≈ reference.mass rtol=2e-5
    for name in PosteriorReference.two_date_statistics
        @test coarse.moments[name] ≈ reference.moments[name] atol=2e-5
    end
    case = location_posterior_case(y, 1, :stationary)
    results = location_posterior_chains(case, "OW1", prior; burnin=2000)
    samples = map(results) do result
        a, b, s = result.B[:, 1], result.B[:, 2], result.S[:, 1]
        first, second = result.F[1, :], result.F[2, :]
        factor_ar, error_ar = result.P[:, 1], result.P2[:, 1]
        (intercept=a, loading=b, loading_squared=b.^2, variance=s, precision=1 ./ s,
         factor1=first, factor2=second, contribution1=b .* first, contribution2=b .* second,
         contribution_cross=b.^2 .* first .* second, factor_ar=factor_ar, error_ar=error_ar,
         factor_ar_squared=factor_ar.^2, error_ar_squared=error_ar.^2,
         ar_cross=factor_ar .* error_ar, factor_ar_loading_squared=factor_ar .* b.^2,
         error_ar_variance=error_ar .* s)
    end
    # The contribution cross moment checks the joint factor path and loading.
    # Unscaled squared factors are deliberately excluded: a stationary AR prior
    # can give them infinite sampling variance near a unit root.
    for name in PosteriorReference.two_date_statistics
        check_location_posterior(hcat([sample[name] for sample in samples]...),
                                 reference.moments[name])
    end
end
