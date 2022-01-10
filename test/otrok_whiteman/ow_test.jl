###############
###############
###############

T = 100                                 # Number of periods in the data 
P = 2                                   # Number of lags in the factor equation 
L = 1                                   # Number of AR lags in the observation equation 
N = 2                                   # Total number of series 
K = 1                                   # Total number of factors (only the global factor)

num_obs = T
γ1 = 1.0
γ2 = 1.0
ϕ1 = 0.5
ϕ2 = 0.1
ψ1 = 0.5
ψ2 = 0.25
σ2_1 = 0.1
σ2_2 = 0.1
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

# Gather all SS parameters 
ssmodel = SSModel(H, A, F, μ, R, Q, Z)

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

priors = priorsSET(K, P, L)             # Set model priors 

###############
###############
###############

results = OWSingleFactorEstimator(data_y, priors)

medians = Any[]
quant33 = Any[]
quant66 = Any[]
stds = Any[]

j = 1
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.33))
    push!(quant66, quantile(results.F[i, j, :], 0.66))
    push!(medians, median(results.F[i, j, :]))
end

plot(results.means.F[:, j])
plot(medians)
plot!(quant33)
plot!(quant66)
plot!(results.means.F[:, j] - stds)
plot!(results.means.F[:, j] + stds)

