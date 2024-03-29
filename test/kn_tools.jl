
@testset "Kim-Nelson Tools" begin

    H = [1.0 0.0; 0.0 1.0]
    A = zeros(2, 2)
    μ = zeros(2)
    F = [0.5 0.0; 0.0 0.5]
    R = zeros(2, 2)
    Q = [1.0 0.0; 0.0 1.0]
    Z = zeros(2, 2)

    ssmodel = DynamicFactorModeling.SSModel(H, A, F, μ, R, Q, Z)

    num_obs = 20

    data_y, data_z, data_β = DynamicFactorModeling.simulateSSModel(num_obs, ssmodel)

    data_filtered_y, data_filtered_β, Pttlag, Ptt = DynamicFactorModeling.kalmanFilter(data_y, ssmodel)

    @test size(data_filtered_y) == size(data_y)
    @test size(data_filtered_β) == size(data_β)

    data_smoothed_y, data_smoothed_β, PtT = DynamicFactorModeling.kalmanSmoother(data_y, ssmodel)

    @test size(data_smoothed_y) == size(data_y)
    @test size(data_filtered_β) == size(data_β)

    β_realized = DynamicFactorModeling.KNFactorSampler(data_y, ssmodel)

    @test size(β_realized) == (num_obs, length(μ))

    Q = [1.0 0.0; 0.0 0.0]

    ssmodel = DynamicFactorModeling.SSModel(H, A, F, μ, R, Q, Z)

    β_realized = DynamicFactorModeling.KNFactorSampler(data_y, ssmodel)

    @test size(β_realized) == (num_obs, length(μ))

end
