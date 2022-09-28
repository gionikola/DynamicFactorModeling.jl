
@testset "Draw β coefficients" begin

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

    Y = rand(100)[:, :]
    X = rand(100)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Float64
    @test size(β) == ()

    Y = rand(100)
    X = rand(100)
    σ2 = 2.0
    β = DynamicFactorModeling.draw_coefficients(Y, X, σ2)

    @test typeof(β) == Float64
    @test size(β) == ()

end

@testset "Draw σ2" begin

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

    X = rand(100)

    σ2 = DynamicFactorModeling.draw_error_variance(Y, X, β)

    @test typeof(σ2) == Float64
    @test size(σ2) == ()


    Y = rand(100)
    X = rand(100)
    β = [1.0]

    σ2 = DynamicFactorModeling.draw_error_variance(Y, X, β)

    @test typeof(σ2) == Float64
    @test size(σ2) == ()

end

@testset "Draw regression parameters" begin

    Y = rand(100)[:, :]
    X = rand(100, 2)
    σ2 = 1.0
    β, σ2 = DynamicFactorModeling.draw_parameters(Y, X, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)
    @test typeof(σ2) == Float64

    Y = rand(100)
    X = rand(100, 2)
    σ2 = 1.0
    β, σ2 = DynamicFactorModeling.draw_parameters(Y, X, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)
    @test typeof(σ2) == Float64

    Y = rand(100)[:, :]
    X = rand(100)
    σ2 = 1.0
    β, σ2 = DynamicFactorModeling.draw_parameters(Y, X, σ2)

    @test typeof(β) == Float64
    @test size(β) == ()
    @test typeof(σ2) == Float64

    Y = rand(100)
    X = rand(100)
    σ2 = 1.0
    β, σ2 = DynamicFactorModeling.draw_parameters(Y, X, σ2)

    @test typeof(β) == Float64
    @test size(β) == ()
    @test typeof(σ2) == Float64

end

@testset "Draw regression w/ autoregressive error parameters" begin

    Y = rand(100)[:, :]
    X = rand(100, 2)
    σ2 = 1.0
    ϕ = [0.5, 0.25]

    β, ϕ, σ2 = DynamicFactorModeling.draw_parameters(Y, X, ϕ, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)
    @test typeof(σ2) == Float64
    @test typeof(ϕ) == Vector{Float64}
    @test size(ϕ) == (2,)

    Y = rand(100)
    X = rand(100, 2)
    σ2 = 1.0
    ϕ = [0.5, 0.25]

    β, ϕ, σ2 = DynamicFactorModeling.draw_parameters(Y, X, ϕ, σ2)

    @test typeof(β) == Vector{Float64}
    @test size(β) == (2,)
    @test typeof(σ2) == Float64
    @test typeof(ϕ) == Vector{Float64}
    @test size(ϕ) == (2,)

end 