using Test, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "support", "location_move.jl"))
include(joinpath(@__DIR__, "support", "diagnostic_sampler.jl"))

@testset "Location- and scale-invariant contribution diagnostics" begin
    factors = [1.0 -2.0 0.5; 2.0 1.0 -0.4; 5.0 4.0 1.1]
    loadings = [1.2 0.8 0.0; -0.7 -0.4 0.0; 0.9 0.0 1.1]
    mask = .!iszero.(loadings)
    levels = [1,2,2]
    coefficients = hcat([0.3,-0.2,0.8],loadings)
    function features(f, b, l)
        names, values = DiagnosticSampler.trace_features(f,b,l,
            [Float64[] for _ in 1:3],[Float64[] for _ in 1:3],[1,1,3];
            active_loadings=mask,factor_levels=levels)
        return Dict(zip(names,values))
    end
    original = features(factors,coefficients,loadings)
    centered = factors .- mean(factors;dims=1)
    contributions = [centered[:,k]*loadings[:,k]' for k in 1:3]
    linear = [-1.0,0.0,1.0]/sqrt(2)
    quadratic = [1.0,-2.0,1.0]/sqrt(6)
    for k in 1:3
        @test original["factor_normalized_linear_$k"] ≈ dot(linear,centered[:,k])/norm(centered[:,k])
        @test original["factor_normalized_quadratic_$k"] ≈ dot(quadratic,centered[:,k])/norm(centered[:,k])
        @test original["contribution_linear_$k"] ≈ sum(linear[t]*contributions[k][t,i] for t in 1:3,i in 1:3)/sqrt(3) atol=2e-13
        @test original["contribution_quadratic_$k"] ≈ sum(quadratic[t]*contributions[k][t,i] for t in 1:3,i in 1:3)/sqrt(3) atol=2e-13
    end
    @test original["contribution_overlap_1_2"] ≈ sum(contributions[1].*contributions[2])/3
    @test original["contribution_pair_rms_1_2"] ≈ sqrt(sum(abs2,contributions[1]+contributions[2])/3)
    @test original["contribution_level_rms_2"] ≈ sqrt(sum(abs2,contributions[2]+contributions[3])/3)
    @test original["contribution_total_rms"] ≈ sqrt(sum(abs2,centered*loadings')/3)
    @test !haskey(original,"contribution_overlap_2_3")

    shifts, scales = [2.0,-0.5,1.3],[0.3,2.0,1.5]
    shifted_b = copy(coefficients)
    shifted_b[:,1] -= loadings*shifts
    transformed = features((factors .+ shifts').*scales',shifted_b,loadings./scales')
    for (name,value) in original
        if startswith(name,"contribution_") || startswith(name,"factor_normalized_")
            @test transformed[name] ≈ value atol=2e-13
        end
    end
    signs = [-1.0,1.0,-1.0]
    folded = features(factors.*signs',coefficients,loadings.*signs')
    for (name,value) in original
        startswith(name,"contribution_") && (@test folded[name] ≈ value atol=2e-13)
    end

    # Perfectly constant paths have undefined normalized shape. Preserve that
    # fact instead of manufacturing a passing zero-valued shape diagnostic.
    constant = features(ones(3,3),coefficients,loadings)
    @test isnan(constant["factor_normalized_linear_1"])
    @test isnan(constant["factor_normalized_quadratic_1"])

    # The structural mask, rather than a current zero loading, controls which
    # overlap summaries exist at every sweep.
    zero_loading = copy(loadings)
    zero_loading[:,2] .= 0
    at_zero = features(factors,coefficients,zero_loading)
    @test haskey(at_zero,"contribution_overlap_1_2")
    @test at_zero["contribution_overlap_1_2"] == 0
end
