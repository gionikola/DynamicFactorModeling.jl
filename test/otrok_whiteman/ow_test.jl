###############
###############
###############

T = 100                                 # Number of periods in the data 
#N_country = 3                           # Number of series per country 
#N_regions = 1                           # Number of regions 
#size_reg = 7                            # Number of countries in a region 
P = 2                                   # Number of lags in the factor equation 
L = 2                                   # Number of AR lags in the observation equation 
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

#meanz, results = OWSingleFactorEstimator(transp_dbl(data_y), priors)
#meanz, results = OWSingleFactorEstimator(data_y, priors)
F, B, S = OWSingleFactorEstimator(data_y, priors)
