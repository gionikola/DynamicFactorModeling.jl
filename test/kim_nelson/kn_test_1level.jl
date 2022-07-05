###############
###############
###############

T = 1000                                 # Number of periods in the data 
P = 2                                   # Number of lags in the factor equation 
L = 2                                   # Number of AR lags in the observation equation 
N = 10                                  # Total number of series 
K = 1                                   # Total number of factors (only the global factor)

H = [ones(N) ident(N) zeros(N)]
R = zeros(N, N)
A = zeros(N, 2)
Z = zeros(2, 2)

F = zeros(P + N, P + N)
F[1, 1] = 0.5
F[1, P+N] = 0.1
F[P+N, 1] = 1.0
F[2:(P+N-1), 2:(P+N-1)] = 0.25 * ident(N)

Q = zeros(P + N, P + N)
Q[1, 1] = 1.0
for i in 1:N
    Q[1+i, 1+i] = 2.0
end

μ = zeros(P + N)

num_obs = T


# Gather all SS parameters 
ssmodel = SSModel(H, A, F, μ, R, Q, Z)

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel)

samplerparams = DFMStruct(P, L, 1000, 50)            # Set model priors 

###############
###############
###############

results = KN1LevelEstimator(data_y, samplerparams)

###############
###############
###############

medians = Any[]
quant33 = Any[]
quant66 = Any[]
stds = Any[]

j = 1
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, :], 0.33))
    push!(quant66, quantile(results.F[i, :], 0.66))
    push!(medians, median(results.F[i, :]))
end

plot(data_β[:,1])
plot!(results.means.F[:, j])
#plot(medians)
plot!(quant33)
plot!(quant66)