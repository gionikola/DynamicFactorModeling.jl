using DynamicFactorModeling
using Test

@testset "DynamicFactorModeling.jl" begin

    include("distribution_functions.jl")
    include("parameter_draws.jl")
    include("hdfm_ss_conversion.jl")
    include("simulate_ss.jl")
    include("linear_regression.jl")
    include("pca.jl")

end
