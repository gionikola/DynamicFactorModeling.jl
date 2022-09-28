
@testset "draw β coefficients" begin

    Y = rand(100)
    X = rand(100, 2)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)

    Y = rand(100)[:, :]
    X = rand(100, 2)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)

    Y = rand(100)
    X = rand(100)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Float64
    @test size(β) == ()

end

@testset "draw σ2" begin

    Y = rand(100)
    X = rand(100, 2)
    β = [1.0, 1.0]

    σ2 = DynamicFactorModeling.draw_error_variance(Y, X, β)

    @test typeof(σ2) == Float64
    @test size(σ2) == ()

    Y = rand(100)[:, :]

    σ2 = DynamicFactorModeling.draw_error_variance(Y, X, β)

    @test typeof(σ2) == Float64
    @test size(σ2) == ()

end 