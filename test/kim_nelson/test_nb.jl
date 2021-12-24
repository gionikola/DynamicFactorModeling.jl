### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 63f32100-5e77-11ec-2ed0-133148c01f62
begin
    using DynamicFactorModeling
    using ShiftedArrays
end

# ╔═╡ 7594f079-2226-4bef-ae6d-e64d3012f239
using Plots

# ╔═╡ 85dea106-5cc2-4a8e-a729-ad5f05ecccb5
using Statistics

# ╔═╡ e5f0ecb6-5c6a-4a37-95c8-b96e17ed7b84
begin
    num_obs = 100
    γ1 = 1.0
    γ2 = 1.0
    ϕ1 = 0.5
    ϕ2 = 0.1
    ψ1 = 0.5
    ψ2 = 0.5
    σ2_1 = 5.0
    σ2_2 = 5.0
    H = [γ1 1.0 0.0 0.0; γ2 0.0 1.0 0.0]
    A = zeros(2, 4)
    F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
    μ = [0.0, 0.0, 0.0, 0.0]
    R = zeros(2, 2)
    Q = zeros(4, 4)
    Q[1, 1] = 1.0
    Q[2, 2] = σ2_1
    Q[3, 3] = σ2_2
    Z = zeros(4, 4)
end

# ╔═╡ 7c76abd8-7e89-4b07-ae8c-a6d3cabc212c
begin
    # Gather all SS parameters 
    ssmodel = SSModel(H, A, F, μ, R, Q, Z)

    # Simulate common components model in state space form 
    data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)
end 

# ╔═╡ 36da1590-2c8b-4665-9494-a4a615934266
begin
    plot(data_y)
    plot!(data_β[:, 1])
end

# ╔═╡ fc21ee9c-8d6d-47b8-bc58-1ec86e3852f8

function HDFMStateSpaceGibbsSampler(data_y, data_z)

    # Initialize factor 
    factor = rand(size(data_y)[1])

    # Initialize hyperparameters 
    γ1 = 1.0
    γ2 = 1.0
    ϕ1 = 0.5
    ϕ2 = 0.1
    ψ1 = 0.5
    ψ2 = 0.5
    σ2_1 = 5.0
    σ2_2 = 5.0
    H = [γ1 1.0 0.0 0.0; γ2 0.0 1.0 0.0]
    A = zeros(2, 4)
    F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
    μ = [0.0, 0.0, 0.0, 0.0]
    R = zeros(2, 2)
    Q = zeros(4, 4)
    Q[1, 1] = 1.0
    Q[2, 2] = σ2_1
    Q[3, 3] = σ2_2
    Z = zeros(4, 4)

    # Specify the number of errors lags obs eq. 
    error_lag_num = 1

    # Create empty lists 
    H_list = Any[]
    A_list = Any[]
    F_list = Any[]
    μ_list = Any[]
    R_list = Any[]
    Q_list = Any[]
    Z_list = Any[]
    factor_list = Any[]

    for i = 1:50

        #  Update factor estimate 
        factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
        factor = factor[:, 1]

        # Estimate obs eq. and autoreg hyperparameters
        while γ1 < 0
            γ1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(reshape(data_y[:, 1], length(data_y[:, 1]), 1), reshape(factor, length(factor), 1), error_lag_num)
        end
        γ2, σ2_2, ψ2 = autocorrErrorRegGibbsSampler(reshape(data_y[:, 2], length(data_y[:, 2]), 1), reshape(factor, length(factor), 1), error_lag_num)
        γ1 = γ1[1][1]
        σ2_1 = σ2_1[1]
        ψ1 = ψ1[1][1]
        γ2 = γ2[1][1]
        σ2_2 = σ2_2[1]
        ψ2 = ψ2[1][1]

        # Estimate factor autoreg state eq. hyperparmeters
        σ2 = Q[1, 1]
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor
            x_temp = ShiftedArrays.lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor[3:end, :]
        ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
        #ϕ, σ2 = staticLinearGibbsSampler(factor_temp, X)
        #σ2 = σ2[1] 
        ϕ1 = ϕ[1][1]
        ϕ2 = ϕ[1][2]

        # Update hyperparameters 
        H = [γ1 1.0 0.0 0.0; γ2 0.0 1.0 0.0]
        A = zeros(2, 4)
        F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
        μ = [0.0, 0.0, 0.0, 0.0]
        R = zeros(2, 2)
        #Q[1, 1] = σ2[1] 
        Q[2, 2] = σ2_1[1]
        Q[3, 3] = σ2_2[1]
        Z = zeros(4, 4)

        # Add updated parameter estimates to
        # their corresponding sample lists 
        push!(H_list, H)
        push!(A_list, A)
        push!(F_list, F)
        push!(μ_list, μ)
        push!(R_list, R)
        push!(Q_list, Q)
        push!(Z_list, Z)
        push!(factor_list, factor)

        println(i)
    end

    return factor_list, H_list, A_list, F_list, μ_list, R_list, Q_list, Z_list
end

# ╔═╡ a28f4de5-88dd-4835-95c4-aa442a8cd1f9
factor_est, H_est, A_est, F_est, μ_est, R_est, Q_est, Z_est = HDFMStateSpaceGibbsSampler(data_y, data_z)

# ╔═╡ 95964a88-2118-405e-91f6-22e6047c2f41
df_factor = Statistics.mean(factor_est)

# ╔═╡ 45215c22-9fac-49b2-bf59-331fd9273ada
begin
    plot(df_factor)
    plot!(data_β[:, 1])
end 

# ╔═╡ Cell order:
# ╠═63f32100-5e77-11ec-2ed0-133148c01f62
# ╠═e5f0ecb6-5c6a-4a37-95c8-b96e17ed7b84
# ╠═7c76abd8-7e89-4b07-ae8c-a6d3cabc212c
# ╠═7594f079-2226-4bef-ae6d-e64d3012f239
# ╠═36da1590-2c8b-4665-9494-a4a615934266
# ╠═fc21ee9c-8d6d-47b8-bc58-1ec86e3852f8
# ╠═a28f4de5-88dd-4835-95c4-aa442a8cd1f9
# ╠═85dea106-5cc2-4a8e-a729-ad5f05ecccb5
# ╠═95964a88-2118-405e-91f6-22e6047c2f41
# ╠═45215c22-9fac-49b2-bf59-331fd9273ada
