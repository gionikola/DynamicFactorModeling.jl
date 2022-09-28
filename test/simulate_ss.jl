@testset "Simulate state space model" begin

    # Measurement Equation:   
    # y_{t} = H β_{t} + A z_{t} + e_{t} 
    # Transition Equation:    
    # β_{t} = μ + F β_{t-1} + v_{t}
    # e_{t} ~ i.i.d.N(0,R)
    # v_{t} ~ i.i.d.N(0,Q)
    # z_{t} ~ i.i.d.N(0,Z)
    # E(e_t v_s') = 0

    H = [1.0 0.0; 0.0 1.0]
    A = zeros(2, 2)
    μ = zeros(2)
    F = [0.5 0.0; 0.0 0.5]
    R = [1.0 0.0; 0.0 1.0]
    Q = [1.0 0.0; 0.0 1.0]
    Z = [1.0 0.0; 0.0 1.0]

    ssmodel = DynamicFactorModeling.SSModel(H, A, F, μ, R, Q, Z)

    num_obs = 100

    data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

    @test typeof(data_y) == Matrix{Float64}
    @test typeof(data_z) == Matrix{Float64}
    @test typeof(data_β) == Matrix{Float64}

    @test size(data_y) == (num_obs, 2)
    @test size(data_z) == (num_obs, 2)
    @test size(data_β) == (num_obs, 2)

end

@testset "Simulate state space model with some degenerate RVs" begin

    # Measurement Equation:   
    # y_{t} = H β_{t} + A z_{t} + e_{t} 
    # Transition Equation:    
    # β_{t} = μ + F β_{t-1} + v_{t}
    # e_{t} ~ i.i.d.N(0,R)
    # v_{t} ~ i.i.d.N(0,Q)
    # z_{t} ~ i.i.d.N(0,Z)
    # E(e_t v_s') = 0

    H = [1.0 0.0; 0.0 1.0]
    A = zeros(2, 2)
    μ = zeros(2)
    F = [0.5 0.0; 0.0 0.5]
    R = [0.0 0.0; 0.0 1.0]
    Q = [0.0 0.0; 0.0 1.0]
    Z = [1.0 0.0; 0.0 0.0]

    ssmodel = DynamicFactorModeling.SSModel(H, A, F, μ, R, Q, Z)

    num_obs = 100

    data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

    @test typeof(data_y) == Matrix{Float64}
    @test typeof(data_z) == Matrix{Float64}
    @test typeof(data_β) == Matrix{Float64}

    @test size(data_y) == (num_obs, 2)
    @test size(data_z) == (num_obs, 2)
    @test size(data_β) == (num_obs, 2)

end

@testset "Simulate state space model with all degenerate RVs" begin

    # Measurement Equation:   
    # y_{t} = H β_{t} + A z_{t} + e_{t} 
    # Transition Equation:    
    # β_{t} = μ + F β_{t-1} + v_{t}
    # e_{t} ~ i.i.d.N(0,R)
    # v_{t} ~ i.i.d.N(0,Q)
    # z_{t} ~ i.i.d.N(0,Z)
    # E(e_t v_s') = 0

    H = [1.0 0.0; 0.0 1.0]
    A = zeros(2, 2)
    μ = zeros(2)
    F = [0.5 0.0; 0.0 0.5]
    R = zeros(2, 2)
    Q = zeros(2, 2)
    Z = zeros(2, 2)

    ssmodel = DynamicFactorModeling.SSModel(H, A, F, μ, R, Q, Z)

    num_obs = 100

    data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

    @test typeof(data_y) == Matrix{Float64}
    @test typeof(data_z) == Matrix{Float64}
    @test typeof(data_β) == Matrix{Float64}

    @test size(data_y) == (num_obs, 2)
    @test size(data_z) == (num_obs, 2)
    @test size(data_β) == (num_obs, 2)

end