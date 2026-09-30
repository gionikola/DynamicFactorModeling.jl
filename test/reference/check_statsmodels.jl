# Run directly with an optional path to independently generated fixtures:
# julia --project=. test/reference/check_statsmodels.jl /tmp/many_cases.jl
using DynamicFactorModeling
using Test

function check_statsmodels_references(fixtures_path=joinpath(@__DIR__, "statsmodels_fixtures.jl"))
    fixtures = include(fixtures_path)
    @testset "statsmodels filter/smoother and SciPy stationary covariance" begin
        for reference in fixtures
            model = SSModel(reference.H, reference.A, reference.F, reference.μ,
                            reference.R, reference.Q, reference.Z)
            kwargs = (; data_z=reference.z, initial_mean=reference.initial_mean,
                        initial_cov=reference.initial_cov)
            predicted_y, filtered, predicted_cov, filtered_cov = kalmanFilter(reference.y, model; kwargs...)
            _, smoothed, smoothed_cov = kalmanSmoother(reference.y, model; kwargs...)
            @test predicted_y ≈ reference.predicted_y atol=1e-9 rtol=1e-9
            @test filtered ≈ reference.filtered atol=1e-9 rtol=1e-9
            @test smoothed ≈ reference.smoothed atol=1e-9 rtol=1e-9
            for t in axes(reference.y, 1)
                @test predicted_cov[t] ≈ reference.predicted_cov[:, :, t] atol=1e-9 rtol=1e-9
                @test filtered_cov[t] ≈ reference.filtered_cov[:, :, t] atol=1e-9 rtol=1e-9
                @test smoothed_cov[t] ≈ reference.smoothed_cov[:, :, t] atol=1e-9 rtol=1e-9
            end
            _, stationary_cov = DynamicFactorModeling._initial_distribution(model, nothing, nothing)
            @test stationary_cov ≈ reference.stationary_cov atol=1e-9 rtol=1e-9
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) ? check_statsmodels_references() : check_statsmodels_references(only(ARGS))
end
