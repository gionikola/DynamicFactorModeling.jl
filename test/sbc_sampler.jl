# Deterministic extraction checks only: no fits, generated datasets, or study RNGs.
module SBCSamplerChecks

using Test
using Statistics
using DynamicFactorModeling: DFMMeans, DFMResults
include(joinpath(@__DIR__, "support", "sbc_sampler.jl"))
const Sampler = SBCSampler
const Reference = Sampler.SBCReference

# Deliberately unrelated stored means detect accidental extraction from means.
function saved_result(F, B, S, P, P2)
    means = DFMMeans(fill(999.0, 6, 1), fill(999.0, 1, 4), fill(999.0, 1, 2),
        fill(999.0, 1, size(P, 2)), fill(999.0, 1, size(P2, 2)))
    return DFMResults(Float64.(F), Float64.(B), Float64.(S), Float64.(P), Float64.(P2), means)
end

const DATA = [1.0 -2.0; 0.5 3.0; -1.0 0.25; 2.0 -0.5; -3.0 1.0; 0.75 -1.5]

@testset "AR(0) packing, draw-wise products, and fixed endpoint" begin
    factors = [1.0 -2.0 3.0; 0.5 1.0 -0.5; 2.0 0.0 1.0;
               -1.0 3.0 2.0; 0.0 -1.0 4.0; 2.0 4.0 -1.0]
    coefficients = [10.0 1.0 100.0 -3.0; 20.0 2.0 200.0 2.0; 30.0 4.0 300.0 -4.0]
    variances = [1.0 4.0; 2.0 5.0; 3.0 6.0]
    result = saved_result(factors, coefficients, variances, zeros(3, 0), zeros(3, 0))
    before = deepcopy(result)
    targets = Sampler.target_trajectories(DATA, result; order=0)
    @test targets.names == (:intercept_1, :anchor_loading, :loading_2, :error_variance_1,
        :final_factor, :final_contribution_2, :first_factor, :marginal_loglikelihood,
        :conditional_loglikelihood)
    @test size(targets.draws) == (3, 9)
    @test targets.draws[:, 1:7] == [10.0 1.0 -3.0 1.0 2.0 -6.0 1.0;
        20.0 2.0 2.0 2.0 4.0 8.0 -2.0; 30.0 4.0 -4.0 3.0 -1.0 4.0 3.0]
    @test mean(targets.draws[:, 6]) == 2.0
    @test mean(targets.draws[:, 6]) != mean(coefficients[:, 4]) * mean(factors[end, :])
    @test targets.endpoint == targets.draws[end, :]
    @test targets.endpoint[6] == 4.0 # Final product, not its posterior mean.

    expected_parameters = [
        (; order=0, intercepts=[10.0, 100.0], loadings=[1.0, -3.0],
            error_variances=[1.0, 4.0], factor_ar=0.0, error_ar=[0.0, 0.0]),
        (; order=0, intercepts=[20.0, 200.0], loadings=[2.0, 2.0],
            error_variances=[2.0, 5.0], factor_ar=0.0, error_ar=[0.0, 0.0]),
        (; order=0, intercepts=[30.0, 300.0], loadings=[4.0, -4.0],
            error_variances=[3.0, 6.0], factor_ar=0.0, error_ar=[0.0, 0.0])]
    for draw in 1:3
        expected = Reference.marginal_loglikelihood(DATA, expected_parameters[draw])
        @test targets.draws[draw, 8] ≈ expected
        parameters = expected_parameters[draw]
        residuals = DATA .- parameters.intercepts' .- factors[:, draw] * parameters.loadings'
        expected_conditional = -0.5 * (12 * log(2pi) +
            6 * sum(log, parameters.error_variances) +
            sum(residuals.^2 ./ parameters.error_variances'))
        @test targets.draws[draw, 9] ≈ expected_conditional
        evaluated = Sampler.target_values(DATA, expected_parameters[draw], factors[:, draw])
        @test evaluated.names == targets.names
        @test evaluated.values == targets.draws[draw, :]
    end
    @test length(unique(targets.draws[:, 8])) == 3
    targets.endpoint[1] = -999.0
    @test targets.draws[end, 1] == 30.0 # Endpoint does not alias retained draws.
    for name in (:F, :B, :S, :P, :P2)
        @test getproperty(result, name) == getproperty(before, name)
    end
end

@testset "AR(1) process ordering and independent likelihood at every draw" begin
    factors = [1.0 -3.0 5.0; 2.0 -2.0 4.0; 3.0 -1.0 3.0;
               4.0 0.0 2.0; 5.0 1.0 1.0; 9.0 8.0 7.0]
    coefficients = [0.1 1.1 -0.5 -1.2; 0.2 0.9 1.5 0.4; -0.4 2.2 1.3 -0.8]
    variances = [0.3 0.7; 0.5 0.9; 0.8 1.1]
    factor_ar = reshape([0.2, -0.35, 0.6], 3, 1)
    error_ar = [0.1 -0.2; -0.3 0.45; 0.55 -0.65]
    result = saved_result(factors, coefficients, variances, factor_ar, error_ar)
    targets = Sampler.target_trajectories(DATA, result; order=1)
    @test targets.names == (:intercept_1, :anchor_loading, :loading_2, :error_variance_1,
        :factor_ar, :error_ar_2, :first_factor, :marginal_loglikelihood,
        :conditional_loglikelihood)
    @test size(targets.draws) == (3, 9)
    @test targets.draws[:, 1:7] == [0.1 1.1 -1.2 0.3 0.2 -0.2 1.0;
        0.2 0.9 0.4 0.5 -0.35 0.45 -3.0; -0.4 2.2 -0.8 0.8 0.6 -0.65 5.0]
    @test targets.endpoint == targets.draws[3, :]
    expected_parameters = [
        (; order=1, intercepts=[0.1, -0.5], loadings=[1.1, -1.2],
            error_variances=[0.3, 0.7], factor_ar=0.2, error_ar=[0.1, -0.2]),
        (; order=1, intercepts=[0.2, 1.5], loadings=[0.9, 0.4],
            error_variances=[0.5, 0.9], factor_ar=-0.35, error_ar=[-0.3, 0.45]),
        (; order=1, intercepts=[-0.4, 1.3], loadings=[2.2, -0.8],
            error_variances=[0.8, 1.1], factor_ar=0.6, error_ar=[0.55, -0.65])]
    for draw in 1:3
        @test targets.draws[draw, 8] ≈ Reference.marginal_loglikelihood(DATA, expected_parameters[draw])
        @test targets.draws[draw, 9] ≈ Reference.conditional_loglikelihood(
            DATA, expected_parameters[draw], factors[:, draw])
    end
    # Integrating F out means changing its sampled path cannot change this target.
    changed_path = saved_result(factors .+ 100, coefficients, variances, factor_ar, error_ar)
    shifted = Sampler.target_trajectories(DATA, changed_path; order=1)
    @test shifted.draws[:, 8] == targets.draws[:, 8]
    @test all(shifted.draws[:, 9] .!= targets.draws[:, 9])
    @test shifted.draws[:, 7] == targets.draws[:, 7] .+ 100
    @test length(unique(targets.draws[:, 8])) == 3
    # A bad interior draw must not be hidden by evaluating only the endpoint.
    invalid = deepcopy(result)
    invalid.P[2, 1] = 1.0
    @test_throws ArgumentError Sampler.target_trajectories(DATA, invalid; order=1)
end

@testset "Canonical sign input is checked without folding" begin
    parameters = (; order=0, intercepts=[0.1, -0.2], loadings=[-1.0, 2.0],
        error_variances=[0.5, 0.75], factor_ar=0.0, error_ar=[0.0, 0.0])
    factors = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
    @test_throws ArgumentError Sampler.target_values(DATA, parameters, factors)
    @test parameters.loadings == [-1.0, 2.0]
    @test factors == [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
    for anchor in (0.0, -0.0)
        zero_anchor = merge(parameters, (; loadings=[anchor, 2.0]))
        targets = Sampler.target_values(DATA, zero_anchor, factors)
        @test iszero(targets.values[2])
        @test targets.values[3] == 2.0
        @test targets.values[6] == -12.0
        @test targets.values[7] == 1.0
        @test isfinite(targets.values[8])
        @test isfinite(targets.values[9])
    end
end

@testset "Dimension and value errors remain visible" begin
    result = saved_result(ones(6, 2), [0.0 1.0 0.0 -2.0; 1.0 2.0 1.0 -3.0],
        ones(2, 2), zeros(2, 0), zeros(2, 0))
    @test_throws ArgumentError Sampler.target_names(2)
    @test_throws ArgumentError Sampler.target_names(0.5)
    @test_throws DimensionMismatch Sampler.target_trajectories(DATA[1:5, :], result; order=0)
    @test_throws DimensionMismatch Sampler.target_trajectories(DATA, result; order=1)
    bad_data = copy(DATA)
    bad_data[3, 2] = NaN
    @test_throws ArgumentError Sampler.target_trajectories(bad_data, result; order=0)
    empty = saved_result(zeros(6, 0), zeros(0, 4), zeros(0, 2), zeros(0, 0), zeros(0, 0))
    @test_throws ArgumentError Sampler.target_trajectories(DATA, empty; order=0)
    wrong_factors = saved_result(ones(6, 1, 2), result.B, result.S, result.P, result.P2)
    @test_throws DimensionMismatch Sampler.target_trajectories(DATA, wrong_factors; order=0)
    wrong_variances = saved_result(result.F, result.B, ones(2, 1), result.P, result.P2)
    @test_throws DimensionMismatch Sampler.target_trajectories(DATA, wrong_variances; order=0)
    invalid = deepcopy(result)
    invalid.B[1, 3] = Inf # Series-2 intercept is needed by the likelihood.
    @test_throws ArgumentError Sampler.target_trajectories(DATA, invalid; order=0)
    invalid = deepcopy(result)
    invalid.S[1, 2] = 0.0 # Series-2 variance is needed even though it is not ranked.
    @test_throws ArgumentError Sampler.target_trajectories(DATA, invalid; order=0)
end

end # module
