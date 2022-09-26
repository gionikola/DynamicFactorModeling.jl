using DynamicFactorModeling
using Test

@testset "DynamicFactorModeling.jl" begin
    
    Y = rand(100)
    X = rand(100,2)
    σ2 = 2.0
    β = draw_coefficients(Y, X, σ2)
    @test typeof(β) == Vector{Float64}
    @test size(β)[1] == 2

end
