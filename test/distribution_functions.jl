@testset "mvn float mean vector and float covariance matrix" begin

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

end

@testset "mvn float mean vector and integer covariance matrix" begin

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

end

@testset "mvn integer mean vector and float covariance matrix" begin

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

end

@testset "mvn integer mean vector and integer covariance matrix" begin

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

end

@testset "mvn float mean and float variance" begin

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

end

@testset "mvn float mean and integer variance" begin

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
end


@testset "mvn integer mean and float variance" begin

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

end

@testset "mvn integer mean and integer variance" begin

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

end

@testset "Γinv" begin

    # T = Int; θ = Float64
    σ2 = DynamicFactorModeling.Γinv(100, 1.0)
    @test size(σ2) == ()
    @test typeof(σ2) == Float64

    # T = Int; θ = Int
    σ2 = DynamicFactorModeling.Γinv(100, 1)
    @test size(σ2) == ()
    @test typeof(σ2) == Float64

end