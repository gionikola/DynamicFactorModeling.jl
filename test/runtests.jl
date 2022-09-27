using DynamicFactorModeling
using Test

@testset "DynamicFactorModeling.jl" begin

    ##########
    ##########
    ##########
    # μ = Vector{Float64}; Σ = Matrix{Float64}
    μ = [0.0, 0.0]
    Σ = [1.0 0.0
        0.0 1.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0.0, 0.0]
    Σ = [1.0 0.0
        0.0 0.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0.0, 0.0]
    Σ = [0.0 0.0
        0.0 0.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    ##########
    ##########
    ##########
    # μ = Vector{Float64}; Σ = Matrix{Int}
    μ = [0.0, 0.0]
    Σ = [1 0
        0 1]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0.0, 0.0]
    Σ = [1 0
        0 0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0.0, 0.0]
    Σ = [0 0
        0 0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    ##########
    ##########
    ##########
    # μ = Vector{Int}; Σ = Matrix{Float64}
    μ = [0, 0]
    Σ = [1.0 0.0
        0.0 1.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0, 0]
    Σ = [1.0 0.0
        0.0 0.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0, 0]
    Σ = [0.0 0.0
        0.0 0.0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    ##########
    ##########
    ##########
    # μ = Vector{Int}; Σ = Matrix{Int}
    μ = [0, 0]
    Σ = [1 0
        0 1]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0, 0]
    Σ = [1 0
        0 0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    μ = [0, 0]
    Σ = [0 0
        0 0]
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Vector{Float64}
    @test size(X) == (length(μ),)
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, length(μ))

    ##########
    ##########
    ##########
    # μ = Float64; Σ = Float64
    μ = 0.0
    Σ = 1.0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    μ = 0.0
    Σ = 0.0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    ##########
    ##########
    ##########
    # μ = Float64; Σ = Int
    μ = 0.0
    Σ = 1
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    μ = 0.0
    Σ = 0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    ##########
    ##########
    ##########
    # μ = Int; Σ = Float64 
    μ = 0
    Σ = 1.0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    μ = 0
    Σ = 0.0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    ##########
    ##########
    ##########
    # μ = Int; Σ = Int
    μ = 0
    Σ = 1
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    μ = 0
    Σ = 0
    X = DynamicFactorModeling.mvn(μ, Σ)
    @test typeof(X) == Float64
    @test size(X) == ()
    n = 100
    X = DynamicFactorModeling.mvn(μ, Σ, n)
    @test typeof(X) == Matrix{Float64}
    @test size(X) == (n, 1)

    ##########
    ##########
    ##########

    # T = Int; θ = Float64
    σ2 = DynamicFactorModeling.Γinv(100, 1.0)
    @test size(σ2) == ()
    @test typeof(σ2) == Float64

    # T = Int; θ = Int
    σ2 = DynamicFactorModeling.Γinv(100, 1)
    @test size(σ2) == ()
    @test typeof(σ2) == Float64

    ##########
    ##########
    ##########
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
