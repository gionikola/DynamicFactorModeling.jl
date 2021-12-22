
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
using Plots

# Number of observations 
num_obs = 100
# Obs equation parameters 
γ1 = 1.0
γ2 = 1.0
γ3 = 1.0
γ4 = 1.0
α1 = 1.0
α2 = 1.0
α3 = 1.0
α4 = 1.0
H = zeros(4, 10)
H[1, 1] = γ1
H[2, 1] = γ2
H[3, 1] = γ3
H[4, 1] = γ4
H[1, 2] = α1
H[2, 2] = α2
H[3, 3] = α3
H[4, 3] = α4
H[1, 4] = 1.0
H[2, 5] = 1.0
H[3, 6] = 1.0
H[4, 7] = 1.0
A = zeros(4, 10)
Z = zeros(10, 10)
# State equation parameters 
ϕ1 = 0.5
ϕ2 = 0.1
ϕ1_12 = 0.5
ϕ2_12 = 0.1
ϕ1_34 = 0.5
ϕ2_34 = 0.1
ψ1 = 0.1
ψ2 = 0.1
ψ3 = 0.1
ψ4 = 0.1
F = zeros(10, 10)
F[1, 1] = ϕ1
F[1, 8] = ϕ2
F[2, 2] = ϕ1_12
F[2, 9] = ϕ2_12
F[3, 3] = ϕ1_34
F[3, 10] = ϕ2_34
F[4, 4] = ψ1
F[5, 5] = ψ2
F[6, 6] = ψ3
F[7, 7] = ψ4
F[8, 1] = 1.0
F[9, 2] = 1.0
F[10, 3] = 1.0
μ = zeros(10)
R = zeros(4, 4)
Q = zeros(10, 10)
σ2_12 = 1.0
σ2_34 = 1.0
σ2_1 = 0.01
σ2_2 = 0.01
σ2_3 = 0.01
σ2_4 = 0.01
Q[1, 1] = 1.0
Q[2, 2] = σ2_12
Q[3, 3] = σ2_34
Q[4, 4] = σ2_1
Q[5, 5] = σ2_2
Q[6, 6] = σ2_3
Q[7, 7] = σ2_4

# Gather all SS parameters 
ssmodel = SSModel(H, A, F, μ, R, Q, Z)

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

# Plot data 
plot(data_β[:, 1:3])

####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
# Create function for full Kim-Nelson algo 
# conditional on true hyperparameter values 
function HDFMSamplerGivenHyperparameters(data_y, data_z, H, A, F, μ, R, Q, Z)

    # Create empty lists 
    factor_list = Any[]

    for i = 1:5000

        println(i)

        # Estimate factors 
        factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
        factor_global = factor[:, 1]
        factor_reg12 = factor[:, 2]
        factor_reg34 = factor[:, 3]

        # Add updated parameter estimates to
        # their corresponding sample lists 
        push!(factor_list, factor)
    end

    return factor_list
end

factor_list = HDFMSamplerGivenHyperparameters(data_y, data_z, H, A, F, μ, R, Q, Z)

factor_est = mean(factor_list)

plot(factor_est[:, 3])
plot!(data_β[:, 3])

####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
# Create function for full algorithm applied 
# to the above Kim-Nelson model.
function HDFMSamplerGivenFactors(data_y, data_z, data_β)

    # Initialize factor 
    factor_global = data_β[:, 1]
    factor_reg12 = data_β[:, 2]
    factor_reg34 = data_β[:, 3]

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

    for i = 1:500

        # Estimate obs eq. and autoreg hyperparameters
        Y = reshape(data_y[:, 1], length(data_y[:, 1]), 1)
        X = [factor_global factor_reg12]
        par1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ1 = par1[1][1]
        α1 = par1[1][2]
        σ2_1 = σ2_1[1]
        ψ1 = ψ1[1][1]
        # --- 
        Y = reshape(data_y[:, 2], length(data_y[:, 2]), 1)
        X = [factor_global factor_reg12]
        par2, σ2_2, ψ2 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ2 = par2[1][1]
        α2 = par2[1][2]
        σ2_2 = σ2_2[1]
        ψ2 = ψ2[1][1]
        # --- 
        Y = reshape(data_y[:, 3], length(data_y[:, 3]), 1)
        X = [factor_global factor_reg34]
        par3, σ2_3, ψ3 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ3 = par3[1][1]
        α3 = par3[1][2]
        σ2_3 = σ2_3[1]
        ψ3 = ψ3[1][1]
        # --- 
        Y = reshape(data_y[:, 4], length(data_y[:, 4]), 1)
        X = [factor_global factor_reg34]
        par4, σ2_4, ψ4 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ4 = par4[1][1]
        α4 = par4[1][2]
        σ2_4 = σ2_4[1]
        ψ4 = ψ4[1][1]

        # Estimate global factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_global
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_global[3:end, :]
        ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
        ϕ1 = ϕ[1][1]
        ϕ2 = ϕ[1][2]

        # Estimate region-12 factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_reg12
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_reg12[3:end, :]
        Y = factor_temp
        ϕ, σ2_12 = staticLinearGibbsSampler(Y, X)
        ϕ1_12 = ϕ[1][1]
        ϕ2_12 = ϕ[1][2]
        σ2_12 = σ2_12[1]

        # Estimate region-12 factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_reg34
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_reg34[3:end, :]
        Y = factor_temp
        ϕ, σ2_34 = staticLinearGibbsSampler(Y, X)
        ϕ1_34 = ϕ[1][1]
        ϕ2_34 = ϕ[1][2]
        σ2_34 = σ2_34[1]

        # Update hyperparameters 
        H = zeros(4, 10)
        H[1, 1] = γ1
        H[2, 1] = γ2
        H[3, 1] = γ3
        H[4, 1] = γ4
        H[1, 2] = α1
        H[2, 2] = α2
        H[3, 3] = α3
        H[4, 3] = α4
        H[1, 4] = 1.0
        H[2, 5] = 1.0
        H[3, 6] = 1.0
        H[4, 7] = 1.0
        A = zeros(4, 10)
        Z = zeros(10, 10)
        # State equation parameters 
        F = zeros(10, 10)
        F[1, 1] = ϕ1
        F[1, 8] = ϕ2
        F[2, 2] = ϕ1_12
        F[2, 9] = ϕ2_12
        F[3, 3] = ϕ1_34
        F[3, 10] = ϕ2_34
        F[4, 4] = ψ1
        F[5, 5] = ψ2
        F[6, 6] = ψ3
        F[7, 7] = ψ4
        F[8, 1] = 1.0
        F[9, 2] = 1.0
        F[10, 3] = 1.0
        μ = zeros(10)
        R = zeros(4, 4)
        Q = zeros(10, 10)
        Q[1, 1] = 1.0
        Q[2, 2] = σ2_12
        Q[3, 3] = σ2_34
        Q[4, 4] = σ2_1
        Q[5, 5] = σ2_2
        Q[6, 6] = σ2_3
        Q[7, 7] = σ2_4

        # Add updated parameter estimates to
        # their corresponding sample lists 
        push!(H_list, H)
        push!(A_list, A)
        push!(F_list, F)
        push!(μ_list, μ)
        push!(R_list, R)
        push!(Q_list, Q)
        push!(Z_list, Z)

        println(i)
    end

    return H_list, A_list, F_list, μ_list, R_list, Q_list, Z_list
end

H_est, A_est, F_est, μ_est, R_est, Q_est, Z_est = HDFMSamplerGivenFactors(data_y, data_z, data_β)

####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
# Create function for full algorithm applied 
# to the above Kim-Nelson model.
function HDFMSampler(data_y, data_z)

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

    # Initialize parameters 

    # Number of observations 
    num_obs = 100
    # Obs equation parameters 
    γ1 = 1.0
    γ2 = 1.0
    γ3 = 1.0
    γ4 = 1.0
    α1 = 1.0
    α2 = 1.0
    α3 = 1.0
    α4 = 1.0
    H = zeros(4, 10)
    H[1, 1] = γ1
    H[2, 1] = γ2
    H[3, 1] = γ3
    H[4, 1] = γ4
    H[1, 2] = α1
    H[2, 2] = α2
    H[3, 3] = α3
    H[4, 3] = α4
    H[1, 4] = 1.0
    H[2, 5] = 1.0
    H[3, 6] = 1.0
    H[4, 7] = 1.0
    A = zeros(4, 10)
    Z = zeros(10, 10)
    # State equation parameters 
    ϕ1 = 0.5
    ϕ2 = 0.1
    ϕ1_12 = 0.5
    ϕ2_12 = 0.1
    ϕ1_34 = 0.5
    ϕ2_34 = 0.1
    ψ1 = 0.1
    ψ2 = 0.1
    ψ3 = 0.1
    ψ4 = 0.1
    F = zeros(10, 10)
    F[1, 1] = ϕ1
    F[1, 8] = ϕ2
    F[2, 2] = ϕ1_12
    F[2, 9] = ϕ2_12
    F[3, 3] = ϕ1_34
    F[3, 10] = ϕ2_34
    F[4, 4] = ψ1
    F[5, 5] = ψ2
    F[6, 6] = ψ3
    F[7, 7] = ψ4
    F[8, 1] = 1.0
    F[9, 2] = 1.0
    F[10, 3] = 1.0
    μ = zeros(10)
    R = zeros(4, 4)
    Q = zeros(10, 10)
    σ2_12 = 1.0
    σ2_34 = 1.0
    σ2_1 = 0.01
    σ2_2 = 0.01
    σ2_3 = 0.01
    σ2_4 = 0.01
    Q[1, 1] = 1.0
    Q[2, 2] = σ2_12
    Q[3, 3] = σ2_34
    Q[4, 4] = σ2_1
    Q[5, 5] = σ2_2
    Q[6, 6] = σ2_3
    Q[7, 7] = σ2_4

    for i = 1:10

        # Estimate factors 
        factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
        factor_global = factor[:, 1]
        factor_reg12 = factor[:, 2]
        factor_reg34 = factor[:, 3]

        # Estimate obs eq. and autoreg hyperparameters
        Y = reshape(data_y[:, 1], length(data_y[:, 1]), 1)
        X = [factor_global factor_reg12]
        par1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ1 = par1[1][1]
        α1 = par1[1][2]
        σ2_1 = σ2_1[1]
        ψ1 = ψ1[1][1]
        # --- 
        Y = reshape(data_y[:, 2], length(data_y[:, 2]), 1)
        X = [factor_global factor_reg12]
        par2, σ2_2, ψ2 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ2 = par2[1][1]
        α2 = par2[1][2]
        σ2_2 = σ2_2[1]
        ψ2 = ψ2[1][1]
        # --- 
        Y = reshape(data_y[:, 3], length(data_y[:, 3]), 1)
        X = [factor_global factor_reg34]
        par3, σ2_3, ψ3 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ3 = par3[1][1]
        α3 = par3[1][2]
        σ2_3 = σ2_3[1]
        ψ3 = ψ3[1][1]
        # --- 
        Y = reshape(data_y[:, 4], length(data_y[:, 4]), 1)
        X = [factor_global factor_reg34]
        par4, σ2_4, ψ4 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num)
        γ4 = par4[1][1]
        α4 = par4[1][2]
        σ2_4 = σ2_4[1]
        ψ4 = ψ4[1][1]

        # Estimate global factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_global
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_global[3:end, :]
        ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
        ϕ1 = ϕ[1][1]
        ϕ2 = ϕ[1][2]

        # Estimate region-12 factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_reg12
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_reg12[3:end, :]
        Y = factor_temp
        ϕ, σ2_12 = staticLinearGibbsSampler(Y, X)
        ϕ1_12 = ϕ[1][1]
        ϕ2_12 = ϕ[1][2]
        σ2_12 = σ2_12[1]

        # Estimate region-12 factor autoreg state eq. hyperparmeters
        σ2 = 1.0
        X = zeros(size(data_y)[1], 2)
        X = X[3:end, :]
        for j = 1:size(X)[2] # iterate over variables in X
            x_temp = factor_reg34
            x_temp = lag(x_temp, j)
            x_temp = x_temp[3:end, :]
            X[:, j] = x_temp
        end
        factor_temp = factor_reg34[3:end, :]
        Y = factor_temp
        ϕ, σ2_34 = staticLinearGibbsSampler(Y, X)
        ϕ1_34 = ϕ[1][1]
        ϕ2_34 = ϕ[1][2]
        σ2_34 = σ2_34[1]

        # Update hyperparameters 
        H = zeros(4, 10)
        H[1, 1] = γ1
        H[2, 1] = γ2
        H[3, 1] = γ3
        H[4, 1] = γ4
        H[1, 2] = α1
        H[2, 2] = α2
        H[3, 3] = α3
        H[4, 3] = α4
        H[1, 4] = 1.0
        H[2, 5] = 1.0
        H[3, 6] = 1.0
        H[4, 7] = 1.0
        A = zeros(4, 10)
        Z = zeros(10, 10)
        # State equation parameters 
        F = zeros(10, 10)
        F[1, 1] = ϕ1
        F[1, 8] = ϕ2
        F[2, 2] = ϕ1_12
        F[2, 9] = ϕ2_12
        F[3, 3] = ϕ1_34
        F[3, 10] = ϕ2_34
        F[4, 4] = ψ1
        F[5, 5] = ψ2
        F[6, 6] = ψ3
        F[7, 7] = ψ4
        F[8, 1] = 1.0
        F[9, 2] = 1.0
        F[10, 3] = 1.0
        μ = zeros(10)
        R = zeros(4, 4)
        Q = zeros(10, 10)
        Q[1, 1] = 1.0
        Q[2, 2] = σ2_12
        Q[3, 3] = σ2_34
        Q[4, 4] = σ2_1
        Q[5, 5] = σ2_2
        Q[6, 6] = σ2_3
        Q[7, 7] = σ2_4

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

factor_list, H_est, A_est, F_est, μ_est, R_est, Q_est, Z_est = HDFMSampler(data_y, data_z)
