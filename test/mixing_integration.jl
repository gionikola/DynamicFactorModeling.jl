# Public integration checks against the sampler saved before adding mixing moves.
# Run alone with julia --project=. test/mixing_integration.jl.
module MixingIntegrationChecks

using Test, Random, Statistics, LinearAlgebra, DynamicFactorModeling
using DynamicFactorModeling: draw_parameters, draw_coefficients,
    _validate_estimator_inputs, _regression_prior_variances, _factor_sign_anchors,
    _initialize_factors, _zero_padded_lags, _draw_stationary_ar,
    _draw_factors_state_space, _draw_factors_precision, _draw_factors_sequential_precision
const DFM = DynamicFactorModeling

# Capture the old sampler's state immediately before its sign convention.
# The hook adds no draws and otherwise calls the original identification step.
const PRE_FOLD = Ref{Any}(nothing)
function _identify_factor_signs!(factors, coefficients, indices, anchors)
    PRE_FOLD[] = (factors=copy(factors), coefficients=copy(coefficients),
                  indices=copy(indices), anchors=copy(anchors))
    DFM._identify_factor_signs!(factors, coefficients, indices, anchors)
end

# Load only the frozen outer sampler. Its regression and factor conditionals
# remain unchanged, and its complete update order supplies the RNG reference.
snapshot_path = joinpath(@__DIR__, "reference", "dfm_sampler_before_mixing.jl.txt")
snapshot = first(split(read(snapshot_path, String), "\nfunction _validate_estimator_inputs"; limit=2))
include_string(@__MODULE__, snapshot, snapshot_path)

function original_sampler(rng, data, settings; kwargs...)
    if settings isa DFMStruct
        result = _estimate_dynamic_factors(rng, data, [1], ones(Int, size(data, 2), 1),
            [settings.factorlags], fill(settings.errorlags, size(data, 2)),
            settings.ndraws, settings.burnin; kwargs...)
        return DFMResults(dropdims(result.F; dims=2), result.B, result.S, result.P,
                          result.P2, result.means)
    end
    return _estimate_dynamic_factors(rng, data, settings.nfactors, settings.factorassign,
        settings.factorlags, settings.errorlags, settings.ndraws, settings.burnin; kwargs...)
end

function compare_results(actual, expected)
    for field in (:F, :B, :S, :P, :P2)
        @test getproperty(actual, field) == getproperty(expected, field)
        @test getproperty(actual.means, field) == getproperty(expected.means, field)
    end
end

function cases(draws, burnin)
    single = DFMStruct(2, 1, draws, burnin)
    two = HDFMStruct(2, [1, 2], [1 1; 1 1; 1 2; 0 2; 0 0],
                     [2, 0], [0, 1, 2, 0, 0], draws, burnin)
    three = HDFMStruct(3, [1, 1, 1], [1 1 1; 1 1 0; 1 0 0; 0 0 0; 1 1 1],
                       [0, 1, 2], [0, 1, 2, 0, 0], draws, burnin)
    return ((KN1LevelEstimator, single, :state_space, [2]),
            (OW1LevelEstimator, single, :precision, [2]),
            (KN2LevelEstimator, two, :state_space, [2, 2, 4]),
            (OW2LevelEstimator, two, :sequential_precision, [2, 2, 4]),
            (KNHierarchicalEstimator, three, :state_space, [3, 2, 5]))
end

function options(settings, engine, anchors, initial, stationary)
    levels = settings isa DFMStruct ? 1 : settings.nlevels
    factors = settings isa DFMStruct ? 1 : sum(settings.nfactors)
    return (; factor_sampler=engine, sign_anchors=anchors, initial, stationary,
        beta_prior_variance=[3.0; [0.4, 0.9, 1.1][1:levels]],
        ar_prior_variance=stationary ? [0.6, 0.2] : [8.0, 3.0],
        variance_shape=3.2, variance_scale=0.7,
        initial_factors=randn(MersenneTwister(871), 3, factors))
end

@testset "Disabling mixing moves preserves all original draws and RNG states" begin
    data = randn(MersenneTwister(812), 3, 5)
    for (estimator, settings, default_engine, anchors) in cases(3, 2),
        (initial, stationary) in ((:stationary, true), (:zero, true), (:zero, false)),
        engine in unique((default_engine, :precision))
        kwargs = options(settings, engine, anchors, initial, stationary)
        reference_rng, public_rng, keyword_rng = [MersenneTwister(913) for _ in 1:3]
        expected = original_sampler(reference_rng, data, settings; kwargs...)
        actual = estimator(public_rng, data, settings; mixing_moves=:none, kwargs...)
        keyword = estimator(data, settings; rng=keyword_rng, mixing_moves=:none, kwargs...)
        compare_results(actual, expected)
        compare_results(keyword, expected)
        expected_next = rand(reference_rng, 8)
        @test rand(public_rng, 8) == expected_next
        @test rand(keyword_rng, 8) == expected_next
    end
end

@testset "Public mixing moves follow factor draws and precede sign identification" begin
    data = randn(MersenneTwister(814), 3, 5)
    modes = (:location, :location_scale)
    observed_unstable_zero_factor = false
    observed_accepted_scale = false
    for (estimator, settings, engine, anchors) in cases(1, 0),
        (initial, stationary) in ((:stationary, true), (:zero, true), (:zero, false)), mode in modes
        kwargs = options(settings, engine, anchors, initial, stationary)
        reference_rng, public_rng = MersenneTwister(915), MersenneTwister(915)
        baseline = original_sampler(reference_rng, data, settings; kwargs...)
        state = PRE_FOLD[]
        factors, coefficients, indices = state.factors, state.coefficients, state.indices
        loadings = zeros(size(data, 2), size(factors, 2))
        active = falses(size(loadings))
        prior_variances = ones(size(loadings))
        for series in axes(indices, 1), level in axes(indices, 2)
            factor = indices[series, level]
            iszero(factor) && continue
            loadings[series, factor] = coefficients[series, level + 1]
            active[series, factor] = true
            prior_variances[series, factor] = kwargs.beta_prior_variance[level + 1]
        end
        factor_orders = settings isa DFMStruct ? [settings.factorlags] :
            [settings.factorlags[level] for level in 1:settings.nlevels
             for _ in 1:settings.nfactors[level]]
        factor_ar = Vector{Float64}[]
        offset = 0
        for order in factor_orders
            push!(factor_ar, vec(baseline.P[:, (offset + 1):(offset + order)]))
            offset += order
        end
        if initial == :zero && !stationary
            observed_unstable_zero_factor |= any(phi -> !DFM.isstationary(phi), factor_ar)
        end
        original_signal = factors * loadings' .+ coefficients[:, 1]'
        DFM._shift_factor_locations!(reference_rng, factors, view(coefficients, :, 1),
            loadings, factor_ar; initial, intercept_prior_variance=kwargs.beta_prior_variance[1])
        if mode == :location_scale
            proposal = DFM._rescale_factors!(reference_rng, factors, loadings, factor_ar;
                active_loadings=active, loading_prior_variance=prior_variances,
                initial, step_size=0.17)
            observed_accepted_scale |= any(proposal.accepted)
        end
        @test factors * loadings' .+ coefficients[:, 1]' ≈ original_signal atol=1e-10
        @test all(iszero, loadings[.!active])
        for series in axes(indices, 1), level in axes(indices, 2)
            factor = indices[series, level]
            iszero(factor) || (coefficients[series, level + 1] = loadings[series, factor])
        end
        DFM._identify_factor_signs!(factors, coefficients, indices, state.anchors)
        actual = estimator(public_rng, data, settings; mixing_moves=mode, scale_step=0.17, kwargs...)
        @test vec(actual.F) == vec(factors)
        @test vec(actual.B) == vec(permutedims(coefficients))
        @test actual.means.F == factors
        @test actual.means.B == actual.B
        for field in (:S, :P, :P2)
            @test getproperty(actual, field) == getproperty(baseline, field)
        end
        @test rand(public_rng, 8) == rand(reference_rng, 8)
    end
    @test observed_unstable_zero_factor
    @test observed_accepted_scale
end

@testset "Mixing mode validation happens before drawing" begin
    data, settings = ones(2, 2), DFMStruct(0, 0, 1, 0)
    for kwargs in ((mixing_moves=:invalid,), (scale_step=0.0,), (scale_step=NaN,),
                   (scale_step=Inf,), (mixing_moves=:none, scale_step=-1.0))
        rng, untouched = MersenneTwister(916), MersenneTwister(916)
        @test_throws ArgumentError KN1LevelEstimator(rng, data, settings; kwargs...)
        @test rand(rng, 8) == rand(untouched, 8)
    end
end

end
