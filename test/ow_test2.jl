#using DynamicFactorModeling 

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

# Store priors 
T = 100                #   Number of periods in the data
N_country = 2          #   Number of series per country
N_regions = 1          #   Number of regions
size_reg = 2           #   Number of countries in a region
P = 2                  #   Number of lags in the factor equation
L = 1                  #   Number of AR lags in the observable equation (set P-1 for PCA and KF and P for OW)

N = N_regions * size_reg * N_country    # total number of series 
K = 1 + size_reg                       #Only the world + country factors

# Set priors 
priors_dim = priorsSET2(K, P, L, K * L, 2)

# Estimate model
F, B, S, P, P2 = OWTwoFactorEstimator(data_y, priors_dim)