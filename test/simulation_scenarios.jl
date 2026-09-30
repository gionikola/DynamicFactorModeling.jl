using Test, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "reference", "simulation_scenarios.jl"))
using .SimulationScenarios

@testset "Independent simulation initial distributions" begin
    @test stationary_ar_covariance(Float64[]) == zeros(0, 0)
    for coefficient in (-0.6, 0.0, 0.8), variance in (0.3, 1.0)
        expected = variance / (1 - coefficient^2)
        @test only(stationary_ar_covariance([coefficient], variance)) ≈ expected rtol=2e-15
    end

    # Closed-form AR(2) autocovariances, independent of the generator's linear
    # Yule–Walker solve. Check the first generated dates, not only presamples.
    phi, variance = [0.5, -0.2], 0.7
    gamma0 = variance * (1 - phi[2]) /
             ((1 + phi[2]) * ((1 - phi[2])^2 - phi[1]^2))
    gamma1 = phi[1] * gamma0 / (1 - phi[2])
    expected = [gamma0 gamma1; gamma1 gamma0]
    @test stationary_ar_covariance(phi, variance) ≈ expected rtol=2e-15
    rng = MersenneTwister(476)
    first_dates = hcat([SimulationScenarios.simulate_ar(rng, phi, variance, 2,
                                                       :stationary)[1] for _ in 1:12000]...)
    @test maximum(abs, vec(mean(first_dates; dims=2))) < 0.035
    @test maximum(abs, cov(first_dates; dims=2) - expected) < 0.04

    values, innovations, presample = SimulationScenarios.simulate_ar(
        MersenneTwister(72), phi, variance, 12, :zero)
    operator = Matrix{Float64}(I, 12, 12)
    for t in 1:12, lag in 1:min(t - 1, 2)
        operator[t, t - lag] = -phi[lag]
    end
    @test values ≈ operator \ innovations atol=1e-14
    @test presample == zeros(2)
    @test_throws ArgumentError SimulationScenarios.simulate_ar(rng, phi, variance, 2, :invalid)
end

@testset "Simulation truth and hierarchy mappings" begin
    specifications = scenarios()
    @test length(specifications) == 6
    @test length(unique(spec.name for spec in specifications)) == 6
    @test Set(spec.initial for spec in specifications) == Set((:stationary, :zero))
    for (index, spec) in enumerate(specifications)
        sample = generate_case(spec, MersenneTwister(500 + index))
        repeated = generate_case(spec, MersenneTwister(500 + index))
        nseries, nfactors = size(spec.loadings)
        @test sample.data == repeated.data
        @test sample.factors == repeated.factors
        @test size(sample.data) == (spec.dates, nseries)
        @test size(sample.factors) == (spec.dates, nfactors)
        @test size(sample.B) == (nseries, spec.nlevels + 1)
        @test length(sample.P) == sum(length, spec.factor_ar)
        @test length(sample.P2) == sum(length, spec.error_ar)
        @test sample.S == spec.error_variances
        @test sample.data == sample.signal + sample.errors
        @test sample.B[:, 1] == spec.intercepts
        @test all(isfinite, sample.data)

        # Reconstruct through the compact level coefficients, independently of
        # the generator's multiplication by the dense factor-loading matrix.
        reconstructed = repeat(spec.intercepts', spec.dates, 1)
        offset = 0
        for level in 1:spec.nlevels
            for series in 1:nseries
                local_factor = spec.assignments[series, level]
                if local_factor == 0
                    @test sample.B[series, level + 1] == 0
                else
                    factor = offset + local_factor
                    reconstructed[:, series] += sample.B[series, level + 1] .* sample.factors[:, factor]
                end
            end
            offset += spec.nfactors[level]
        end
        @test reconstructed ≈ sample.signal atol=1e-14
        for factor in 1:nfactors
            @test spec.loadings[sample.sign_anchors[factor], factor] > 0
        end
        spec.estimator_kind == :single && @test length(unique(sample.error_orders)) == 1
        if spec.initial == :zero
            @test all(all(iszero, history) for history in sample.initial_factors)
            @test all(all(iszero, history) for history in sample.initial_errors)
            @test sample.factors[1, :] == sample.factor_innovations[1, :]
            @test sample.errors[1, :] == sample.error_innovations[1, :]
        end
    end

    two = generate_case(only(filter(spec -> spec.name == :two_level, specifications)), MersenneTwister(8))
    @test two.sign_anchors == [1, 1, 5]
    @test two.level_orders == [1, 0]
    @test two.B[4, :] == [0.0, 0.8, 0.0]
    @test two.P == [0.65]
    @test two.P2 == [0.3, -0.2, 0.25, 0.4, -0.1]

    three = generate_case(only(filter(spec -> spec.name == :three_level, specifications)), MersenneTwister(9))
    @test three.sign_anchors == [1, 1, 5, 1, 3, 5, 7]
    @test three.level_orders == [1, 1, 1]
    @test three.B[7, :] == [0.2, 1.15, -0.75, 0.8]
    @test three.P == [0.75, 0.4, -0.2, 0.15, 0.3, -0.25, 0.5]
    @test isempty(three.P2)
end
