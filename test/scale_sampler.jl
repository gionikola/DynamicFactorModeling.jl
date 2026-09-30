module ScaleSamplerChecks

using Test
using Random
using LinearAlgebra
using Statistics
using DynamicFactorModeling

include(joinpath(@__DIR__, "support", "location_move.jl"))
include(joinpath(@__DIR__, "support", "diagnostic_sampler.jl"))
include(joinpath(@__DIR__, "reference", "simulation_scenarios.jl"))
using .DiagnosticSampler
using .SimulationScenarios

BLAS.set_num_threads(1)

# A deterministic RNG fixture for the scale proposal only. Zero is a valid
# uniform output and forces acceptance of every finite log ratio, ensuring
# that the tests exercise actual mutations rather than accidental rejections.
mutable struct ScaleProposalRNG <: AbstractRNG
    normal_value::Float64
    uniform_value::Float64
    normal_calls::Int
    uniform_calls::Int
end
ScaleProposalRNG(value=0.5, uniform=0.0) = ScaleProposalRNG(value, uniform, 0, 0)
Random.randn(rng::ScaleProposalRNG) = (rng.normal_calls += 1; rng.normal_value)
Random.rand(rng::ScaleProposalRNG) = (rng.uniform_calls += 1; rng.uniform_value)

function unpack_draw(case, result)
    spec = case.spec
    coefficients = permutedims(reshape(result.B[1, :], spec.nlevels + 1, :))
    factors = ndims(result.F) == 2 ? reshape(copy(result.F[:, 1]), :, 1) : copy(result.F[:, :, 1])
    loadings = zeros(size(case.data, 2), sum(spec.nfactors))
    active = falses(size(loadings))
    offset = 0
    for level in 1:spec.nlevels
        for series in axes(loadings, 1)
            local_factor = spec.assignments[series, level]
            iszero(local_factor) && continue
            factor = offset + local_factor
            active[series, factor] = true
            loadings[series, factor] = coefficients[series, level + 1]
        end
        offset += spec.nfactors[level]
    end
    factor_ar, first_column = Vector{Float64}[], 1
    for order in case.factor_orders
        push!(factor_ar, vec(result.P[1, first_column:(first_column + order - 1)]))
        first_column += order
    end
    return (; factors, coefficients, loadings, active, factor_ar)
end

function loading_variances(spec, beta_prior)
    column_variances = [beta_prior[level + 1] for level in 1:spec.nlevels
                        for _ in 1:spec.nfactors[level]]
    return repeat(column_variances', size(spec.loadings, 1), 1)
end

function check_trace_against_stored_draw(output, case)
    draw = unpack_draw(case, output.result)
    trace = Dict(zip(output.trace_names, output.trace[end, :]))
    signal = draw.factors * draw.loadings' .+ draw.coefficients[:, 1]'
    for series in axes(signal, 2)
        @test trace["intercept_$series"] == draw.coefficients[series, 1]
        @test trace["signal_mean_$series"] ≈ mean(signal[:, series]) atol=2e-12
    end
    for factor in axes(draw.factors, 2)
        values = draw.factors[:, factor]
        @test trace["factor_mean_$factor"] ≈ mean(values) atol=2e-12
        @test trace["factor_rms_$factor"] ≈ std(values; corrected=false) atol=2e-12
        @test trace["loading_norm_$factor"] ≈ norm(draw.loadings[:, factor]) atol=2e-12
        @test trace["anchor_loading_$factor"] == draw.loadings[case.sign_anchors[factor], factor]
        @test trace["anchor_loading_$factor"] >= 0
    end
    @test all(iszero, draw.loadings[.!draw.active])
    for series in axes(case.spec.assignments, 1), level in axes(case.spec.assignments, 2)
        if iszero(case.spec.assignments[series, level])
            @test draw.coefficients[series, level + 1] == 0
        end
    end
end

@testset "Scale move is placed after the ordinary and location updates" begin
    observed_sign_flip = false
    for scenario_index in (2, 3, 5, 6)
        spec = merge(scenarios()[scenario_index], (; dates=8))
        case = generate_case(spec, MersenneTwister(420 + scenario_index))
        methods = spec.estimator_kind == :single ? ("KN1", "OW1") :
                  spec.estimator_kind == :two_level ? ("KN2", "OW2") : ("KNHierarchical",)
        beta_prior = [3.0; [0.2 + 0.3 * level for level in 1:spec.nlevels]]
        start = randn(MersenneTwister(831 + scenario_index), size(case.factors))
        for method in methods
            plain_rng, moved_rng = MersenneTwister(512), MersenneTwister(512)
            plain_location_rng, moved_location_rng = MersenneTwister(921), MersenneTwister(921)
            scale_rng = ScaleProposalRNG()
            plain = sample_case(case, method, plain_rng; ndraws=1, burnin=0,
                initial_factors=start, variant=:location, move_rng=plain_location_rng,
                beta_prior_variance=beta_prior)
            moved = sample_case(case, method, moved_rng; ndraws=1, burnin=0,
                initial_factors=start, variant=:location_scale, move_rng=moved_location_rng,
                scale_rng, beta_prior_variance=beta_prior)
            expected = unpack_draw(case, plain.result)
            actual = unpack_draw(case, moved.result)
            before_signal = expected.factors * expected.loadings' .+ expected.coefficients[:, 1]'
            # Positive scaling commutes with the public sign-folding map. Thus
            # applying it to the already folded location-only result must equal
            # applying it inside the sampler before sign folding and storage.
            proposal = DiagnosticSampler.ScaleMove.rescale_factors!(ScaleProposalRNG(),
                expected.factors, expected.loadings, expected.factor_ar;
                active_loadings=expected.active, initial=spec.initial,
                loading_prior_variance=loading_variances(spec, beta_prior))
            @test all(proposal.accepted)
            @test actual.factors ≈ expected.factors atol=2e-12
            @test actual.loadings ≈ expected.loadings atol=2e-12
            @test actual.coefficients[:, 1] == expected.coefficients[:, 1]
            @test actual.factors * actual.loadings' .+ actual.coefficients[:, 1]' ≈ before_signal atol=2e-12
            @test plain.result.S == moved.result.S
            @test plain.result.P == moved.result.P
            @test plain.result.P2 == moved.result.P2
            @test plain.sign_flips == moved.sign_flips
            @test rand(plain_rng, 8) == rand(moved_rng, 8)
            @test rand(plain_location_rng, 8) == rand(moved_location_rng, 8)
            @test scale_rng.normal_calls == sum(spec.nfactors)
            @test scale_rng.uniform_calls == sum(spec.nfactors)
            ordinary_events(output) = Dict(key=>value for (key, value) in output.events
                                           if first(key) in ("factor", "error"))
            @test ordinary_events(plain) == ordinary_events(moved)
            observed_sign_flip |= any(>(0), plain.sign_flips)
            check_trace_against_stored_draw(moved, case)
        end
    end
    @test observed_sign_flip
end

@testset "Baseline and location arms never consume the scale RNG" begin
    spec = merge(scenarios()[2], (; dates=8))
    case = generate_case(spec, MersenneTwister(255))
    for variant in (:baseline, :location), method in ("KN1", "OW1")
        untouched_rng = ScaleProposalRNG()
        first_run = sample_case(case, method, MersenneTwister(144); ndraws=2, burnin=1,
            variant, move_rng=MersenneTwister(188), scale_rng=untouched_rng)
        second_run = sample_case(case, method, MersenneTwister(144); ndraws=2, burnin=1,
            variant, move_rng=MersenneTwister(188), scale_rng=MersenneTwister(999))
        for field in (:F, :B, :S, :P, :P2)
            @test getproperty(first_run.result, field) == getproperty(second_run.result, field)
        end
        @test first_run.trace == second_run.trace
        @test first_run.events == second_run.events
        @test untouched_rng.normal_calls == 0
        @test untouched_rng.uniform_calls == 0
    end
end

end
