# Run from an instantiated test environment: julia --project=test test/run_extended.jl
using Test

# Separate processes keep each reference module and its constants isolated.
# These are bounded correctness checks, not repeated-data research campaigns.
checks = [
    ("simulation_diagnostics.jl", String[]),
    ("gaussian_location_reference.jl", String[]),
    ("sbc_reference.jl", String[]),
    ("sbc_sampler.jl", String[]),
    ("location_move.jl", String[]),
    ("diagnostic_sampler.jl", String[]),
    ("scale_move.jl", String[]),
    ("scale_sampler.jl", String[]),
    ("contribution_diagnostics.jl", String[]),
    ("location_posterior.jl", String[]),
    ("location_posterior.jl", ["--variant=location_scale"]),
]

@testset "Extended reference and sampler checks" begin
    for (file, arguments) in checks
        @testset "$file $(join(arguments, ' '))" begin
            script = joinpath(@__DIR__, file)
            command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $arguments`
            @test success(command)
        end
    end
end
