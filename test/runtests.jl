using DynamicFactorModeling
using Test

@testset "DynamicFactorModeling.jl" begin

    μ = [0.0, 0.0]
    Σ = [1.0 0.0; 0.0 1.0]
    X = mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (2,)

    μ = [0, 0]
    Σ = [1 0; 0 1]
    X = mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (2,)

    μ = 0.0
    Σ = 1.0
    X = mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()

    μ = 0
    Σ = 1
    X = mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()

    Y = rand(100)
    X = rand(100, 2)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)

    σ2 = DynamicFactorModeling.draw_error_variance(Y, X, β)

    @test typeof(σ2) == Float64
    @test size(σ2) == ()

end
