using DynamicFactorModeling
using LinearAlgebra
using Random
using Statistics
using Test

@testset "DynamicFactorModeling.jl" begin
    include("common_types.jl")
    include("simulation_scenarios.jl")
    include("distribution_functions.jl")
    include("parameter_draws.jl")
    include("hdfm_ss_conversion.jl")
    include("simulate_ss.jl")
    include("linear_regression.jl")
    include("pca.jl")
    include("kn_tools.jl")
    include("reference/check_statsmodels.jl")
    check_statsmodels_references()
    include("state_space_adversarial.jl")
    include("kim_nelson/estimators.jl")
    include("mixing_integration.jl")
    include("stationary_likelihood.jl")
    include("posterior_validation.jl")
end
