using Random
using Statistics
using LinearAlgebra

@testset "State-space simulation" begin
    model = SSModel(H=[1.0 2.0], A=reshape([3.0], 1, 1),
        F=[0.5 0; 0 0.25], μ=[1.0, -0.5], R=fill(0.4, 1, 1),
        Q=[1.0 0.5; 0.5 2.0], Z=fill(0.6, 1, 1))
    y, z, states = simulateSSModel(MersenneTwister(902), 40000, model)
    @test (y, z, states) == simulateSSModel(MersenneTwister(902), 40000, model)
    @test vec(mean(states; dims=1)) ≈ [2, -2/3] atol=0.04
    expected_covariance = model.Q ./ (1 .- [0.5, 0.25] * [0.5, 0.25]')
    @test cov(states) ≈ expected_covariance atol=0.07
    innovations = states[2:end, :] - states[1:end-1, :] * model.F' .- model.μ'
    @test cov(innovations) ≈ model.Q atol=0.05
    measurement_errors = y - states * model.H' - z * model.A'
    @test var(measurement_errors) ≈ 0.4 atol=0.015
    @test var(z) ≈ 0.6 atol=0.025
    @test abs(cor(vec(measurement_errors), z[:, 1])) < 0.025

    # A fixed initial state is β₀: row one still receives all three noises.
    first_rows = reduce(vcat, [simulateSSModel(MersenneTwister(i), 1, model;
        initial_state=zeros(2))[1] for i in 1:2000])
    @test var(first_rows) ≈ (model.H * model.Q * model.H')[1] + 9 * 0.6 + 0.4 atol=1.3

    deterministic = SSModel([1.0 0], zeros(1, 0), [1.0 1; 0 1], [0.0, 1.0],
        zeros(1, 1), zeros(2, 2), zeros(0, 0))
    yd, zd, xd = simulateSSModel(3, deterministic; initial_state=[2, 3])
    @test xd == [5 4; 9 5; 14 6]
    @test yd[:, 1] == [5, 9, 14]
    @test size(zd) == (3, 0)
    @test_throws ArgumentError simulateSSModel(3, deterministic)
    @test_throws ArgumentError simulateSSModel(3, model; initial_state=zeros(2), initial_cov=zeros(2, 2))
    @test_throws ArgumentError simulateSSModel(-1, model)
    @test_throws DimensionMismatch simulateSSModel(1, model; initial_state=[1])
    @test size.(simulateSSModel(0, model)) == ((0, 1), (0, 1), (0, 2))

    @test_throws ArgumentError SSModel(ones(1, 2), zeros(1, 0), zeros(2, 2), zeros(2),
        ones(1, 1), [1.0 2e-9; 2e-9 1e-18], zeros(0, 0))
    singular = SSModel(Matrix{Float64}(I, 2, 2), zeros(2, 0), zeros(2, 2), [1., -1.],
        zeros(2, 2), [1.0 2; 2 4], zeros(0, 0))
    ys, _, xs = simulateSSModel(MersenneTwister(56), 100, singular)
    @test ys == xs
    @test xs[:, 2] .+ 1 ≈ 2 .* (xs[:, 1] .- 1)
end
