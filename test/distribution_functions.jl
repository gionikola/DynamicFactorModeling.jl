using Random, LinearAlgebra, Statistics

@testset "Normal draws" begin
    dfm = DynamicFactorModeling
    for Tmean in (Int, Float32, Float64), Tcov in (Int, Float32, Float64)
        μ = Tmean[1, -2]
        Σ = Tcov[2 1; 1 3]
        @test dfm.mvn(MersenneTwister(1), μ, Σ) isa Vector{Float64}
        @test size(dfm.mvn(MersenneTwister(1), μ, Σ, 7)) == (7, 2)
        @test size(dfm.mvn(μ, Σ, 0)) == (0, 2)
        @test dfm.mvn(MersenneTwister(1), μ, Σ) == dfm.mvn(μ, Σ; rng=MersenneTwister(1))
        @test dfm.mvn(MersenneTwister(1), μ, Σ, 7) == dfm.mvn(μ, Σ, 7; rng=MersenneTwister(1))
        @test dfm.mvn(MersenneTwister(1), Tmean(2), Tcov(3)) isa Float64
        @test size(dfm.mvn(Tmean(2), Tcov(3), 7)) == (7, 1)
    end

    rng = MersenneTwister(381)
    μ = [1.5, -2.0]
    Σ = [2.0 -0.6; -0.6 0.8]
    draws = dfm.mvn(rng, μ, Σ, 30000)
    @test vec(mean(draws; dims=1)) ≈ μ atol=0.035
    @test cov(draws) ≈ Σ atol=0.055
    scalar_draws = dfm.mvn(rng, 1.5, 2.0, 30000)
    @test mean(scalar_draws) ≈ 1.5 atol=0.035
    @test var(scalar_draws) ≈ 2.0 atol=0.06

    # Singular covariance can have perfect correlation without zero diagonals.
    singular = dfm.mvn(rng, [2.0, -1.0, 5.0], [1 1 0; 1 1 0; 0 0 0], 100)
    @test singular[:, 1] .- 2 ≈ singular[:, 2] .+ 1 atol=1e-12
    @test singular[:, 3] == fill(5.0, 100)
    @test dfm.mvn(rng, μ, zeros(2, 2)) == μ
    @test dfm.mvn(rng, 3.0, 0.0) == 3.0
    @test dfm.mvn(rng, 3.0, 0.0, 4) == fill(3.0, 4, 1)
    @test size(dfm.mvn(rng, Float64[], zeros(0, 0), 4)) == (4, 0)
    # A small variance is still stochastic when another coordinate has a
    # much larger scale; covariance rank must not erase it.
    unequal_scales = dfm.mvn(rng, zeros(2), [1.0 0.0; 0.0 1e-18], 3000)
    @test var(unequal_scales[:, 2]) ≈ 1e-18 rtol=0.1

    @test_throws DimensionMismatch dfm.mvn([0.0, 0.0], ones(3, 3))
    @test_throws DimensionMismatch dfm.mvn([0.0, 0.0], ones(2, 3))
    @test_throws ArgumentError dfm.mvn([0.0, 0.0], [1.0 0.9; 0.0 1.0])
    @test_throws ArgumentError dfm.mvn([0.0, 0.0], [1.0 2.0; 2.0 1.0])
    @test_throws ArgumentError dfm.mvn([NaN], ones(1, 1))
    @test_throws ArgumentError dfm.mvn([0.0], fill(Inf, 1, 1))
    @test_throws ArgumentError dfm.mvn(0.0, -1.0)
    @test_throws ArgumentError dfm.mvn(Inf, 1.0)
    @test_throws ArgumentError dfm.mvn(0.0, NaN)
    @test_throws ArgumentError dfm.mvn(μ, Σ, -1)
    @test_throws ArgumentError dfm.mvn(0.0, 1.0, -1)
end

@testset "Inverse-gamma parameterization" begin
    dfm = DynamicFactorModeling
    rng = MersenneTwister(384)
    ν, θ = 9.5, 4.4
    draws = [dfm.Γinv(rng, ν, θ) for _ in 1:30000]
    @test all(>(0), draws)
    @test mean(draws) ≈ θ / (ν - 2) atol=0.012
    @test var(draws) ≈ 2θ^2 / ((ν - 2)^2 * (ν - 4)) atol=0.018
    @test mean(1 ./ draws) ≈ ν / θ atol=0.035
    @test dfm.Γinv(MersenneTwister(2), ν, θ) == dfm.Γinv(ν, θ; rng=MersenneTwister(2))
    @test dfm.Γinv(10, 2) isa Float64
    @test dfm.Γinv(0.75, 1.0) > 0
    @test_throws ArgumentError dfm.Γinv(0, 1)
    @test_throws ArgumentError dfm.Γinv(-1, 1)
    @test_throws ArgumentError dfm.Γinv(2, 0)
    @test_throws ArgumentError dfm.Γinv(2, -1)
    @test_throws ArgumentError dfm.Γinv(Inf, 1)
    @test_throws ArgumentError dfm.Γinv(2, NaN)
end
