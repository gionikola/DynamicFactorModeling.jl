using DynamicFactorModeling
using Test

@testset "DynamicFactorModeling.jl" begin
    num_obs = 100
    H = [1.0 0.0]
    A = [0.0][:, :]
    F = [0.3 0.5; 1.0 0.0]
    μ = [0.0, 0.0]
    R = [0.0][:, :]
    Q = [1.0 0.0; 0.0 0.0]
    Z = [0.0][:, :]

    @test data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)
    @test data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)
    
end
