####################
####################
####################
####################
####################
using Plots
using DynamicFactorModeling

num_obs = 100
H = [1.0 0.0]
A = [0.0][:, :]
F = [0.3 0.5; 1.0 0.0]
μ = [0.0, 0.0]
R = [0.0][:, :]
Q = [1.0 0.0; 0.0 0.0]
Z = [0.0][:, :]

data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)
plot(data_y)

data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)

plot(data_β[:, 1])
plot!(data_filtered_β[:, 1])

plot(data_filtered_β[:, 1])

plot(data_y)
plot!(data_filtered_y)

factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
plot(factor[:, 1])

####################
####################
####################
####################
####################

rand(InverseGamma(1, 2))



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

# Specify common components model 
# Section 8.2 in Kim & Nelson (1999) 
"""
num_obs = 100
γ1 = 0.5
γ2 = 0.4
ϕ1 = 0.5
ϕ2 = 0.1
ψ1 = 0.5
ψ2 = 0.5
σ2_1 = 2.0
σ2_2 = 3.0
H = [γ1 1.0 0.0 0.0; γ2 1.0 1.0 0.0]
A = zeros(2, 4)
F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
μ = [0.0, 0.0, 0.0, 0.0]
R = zeros(2, 2)
Q = zeros(4, 4)
Q[1, 1] = 1.0
Q[2, 2] = σ2_1
Q[3, 3] = σ2_2
Z = zeros(4, 4)
"""

num_obs = 100
γ1 = 1.0
γ2 = 1.0
ϕ1 = 0.5
ϕ2 = 0.1
ψ1 = 0.5
ψ2 = 0.5
σ2_1 = 0.00000000001
σ2_2 = 0.00000000001
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

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)

# Plot data 
plot(data_y)
plot!(data_β[:, 1])

#####################################################
# Create function for full algorithm applied 
# to the above Kim-Nelson model.
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
        σ2_1 = 0.00000000001
        σ2_2 = 0.00000000001
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

        for i = 1:1000

                #  Update factor estimate 
                factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
                factor = factor[:, 4]

                # Estimate obs eq. and autoreg hyperparameters
                γ1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(reshape(data_y[:, 1], length(data_y[:, 1]), 1), reshape(factor, length(factor), 1), error_lag_num)
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
                        x_temp = lag(x_temp, j)
                        x_temp = x_temp[3:end, :]
                        X[:, j] = x_temp
                end
                factor_temp = factor[3:end, :]
                ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
                ϕ1 = ϕ[1][1]
                ϕ2 = ϕ[1][2]

                # Update hyperparameters 
                H = [γ1 1.0 0.0 0.0; γ2 0.0 1.0 0.0]
                A = zeros(2, 4)
                F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
                μ = [0.0, 0.0, 0.0, 0.0]
                R = zeros(2, 2)
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

factor_est, H_est, A_est, F_est, μ_est, R_est, Q_est, Z_est = HDFMStateSpaceGibbsSampler(data_y, data_z)

plot(mean(factor_est))
plot!(data_β[:, 1])

ci = sqrt(1 / (1 - ψ1^2 - ψ2^2))

plot(mean(factor_est))
plot!(mean(factor_est) .+ ci * 2)
plot!(mean(factor_est) .- ci * 2)
plot!(data_β[:, 1])

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

# Create function for full algorithm applied 
# to the above Kim-Nelson model, assuming that 
# the hyperparameters are known (insert true values). 
function HDFMStateSpaceGibbsSamplerTest(data_y, data_z, H, A, F, μ, R, Q, Z)

        # Initialize factor 
        factor = rand(size(data_y)[1])

        # Specify the number of errors lags obs eq. 
        error_lag_num = 1

        for i = 1:10000
                println(i)
                #  Update factor estimate 
                factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
                factor = factor[:, 1]
        end

        return factor
end

# Specify common components model 
# Section 8.2 in Kim & Nelson (1999) 
num_obs = 100
γ1 = 1.0
γ2 = 1.0
ϕ1 = 0.5
ϕ2 = 0.0
ψ1 = 0.5
ψ2 = 0.5
σ2_1 = 0.00000000001
σ2_2 = 0.00000000001
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

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)

factor_estimate = HDFMStateSpaceGibbsSamplerTest(data_y, data_z, H, A, F, μ, R, Q, Z)

ci = sqrt(Q[1, 1] / (1 - F[1, 1] - F[1, 2]))

plot(data_β[:, 1])
plot!(factor_estimate)
plot!(factor_estimate .+ (2 * ci))
plot!(factor_estimate .- (2 * ci))

plot(factor_estimate)
plot!(data_y[:, 1])
plot!(data_y[:, 2])

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

# Specify common components model 
# Section 8.2 in Kim & Nelson (1999)
test = zeros(100)

for i = 1:100
        test[i] = F_est[i][1, 1]
end

using StatsPlots

# Test `staticLinearGibbsSamplerRestrictedVariance`
factor = data_β[:, 1]
σ2 = 1
X = zeros(size(data_y)[1], 2)
X = X[3:end, :]
for j = 1:size(X)[2] # iterate over variables in X
        x_temp = factor
        x_temp = lag(x_temp, j)
        x_temp = x_temp[3:end, :]
        X[:, j] = x_temp
end
factor_temp = factor[3:end, :]
ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
ϕ1 = ϕ[1][1]
ϕ2 = ϕ[1][2]


# Test `autocorrErrorRegGibbsSampler(Y, X, error_lag_num)`
Y = reshape(data_y[:, 1], length(data_y[:, 1]), 1)
X = reshape(factor, length(factor), 1)
error_lag_num = 1
γ1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(Y, X, error_lag_num) 