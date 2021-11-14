using Random
using Distributions 
using LinearAlgebra

######################
######################
######################
@doc """
    
    gen_rand_cov(N,μ,σ)

Description:
Generate a random covariance matrix. 

Inputs:
- N = number of variables
- μ = expected covariance 
- σ = expected std. deviation of covariance
"""
function gen_rand_cov(N,μ,σ)
    cov_mat = zeros(N,N) # return object 
    cov_distr = Normal(μ,σ)
    for i in 1:size(cov_mat)[1]
        for j in i:size(cov_mat)[2]
            cov_mat[i,j] = rand(cov_distr) 
            cov_mat[j,i] = cov_mat[i,j]
        end 
    end 
    return cov_mat
end 

######################
######################
######################
@doc """

"""
function univariateAR(obs_num = 1, parameters = [0.0,0.0], error_distribution = Normal())

    lag_num = length(parameters) - 1
    data = convert(Array{Float64,1}, 1:obs_num)
    disturbance = rand(error_distribution, obs_num)

    for i in 1:obs_num
        if i > lag_num 
            data[i] =  parameters[1] + 
                        reverse(parameters[2:length(parameters)])'*data[(i-lag_num):(i-1)]+ 
                        disturbance[i]
        end 
    end 

    return data
end 

######################
######################
######################
@doc """

"""
function vectorAR(obs_num = 1, var_num = 1, intercept = [0.0,0.0], ar_parameters = zeros(2,2), error_distribution = Normal())

    # Compute number of lags 
    lag_num = floor(Int64, length(ar_parameters[1,:])/var_num)
    
    # Generate empty data matrix 
    data = convert(Array{Float64,2}, zeros(obs_num,var_num))

    # Simulate disturbances 
    disturbance = rand(error_distribution, obs_num, var_num)
    
    # Stacked dependent vector 
    vec_dep = zeros(length(ar_parameters[1,:]))
    dist = zeros(length(ar_parameters[1,:]))

    # Stacked intercept vector 
    vec_int = zeros(length(ar_parameters[1,:]))
    for i in 1:var_num
        vec_int[1+((i-1)*lag_num)] = intercept[i] 
    end

    # Simulate 
    for i in 1:obs_num
        if i > lag_num
            for j in 1:var_num
                for k in 1:lag_num
                    vec_dep[lag_num*(j-1)+k] = data[i-k,j]
                    dist[lag_num*(j-1)+k] = disturbance[i-k,j]
                end 
            end 
            vec_dep = vec_int + ar_parameters*vec_dep + dist  
            for j in 1:var_num
                data[i,j] = vec_dep[1+((j-1)*lag_num)]
            end
        end 
    end 

    # Return 
    return data 
end 

######################
######################
######################
@doc """

"""
function instantDFM(num_vars = 4, num_obs = 10000, num_hierarchies = 2, factor_lags = [1,1])

    # Create factor container 
    num_factors = 0
    for i in 1:num_hierarchies
        num_factors += 2^(i-1)
    end 
    factors = zeros(num_obs, num_factors)

end 

######################
######################
######################
@doc """

"""
function firstComponentFactor(data)

    num_obs = size(data)[1]
    centering = I - 1/(num_obs)*ones(num_obs,num_obs) # centering matrix
    data = centering*data # center the data 
    cov_mat = data'*data
    cov_mat_eigen = eigen(cov_mat)
    components = cov_mat_eigen.vectors
    firstComponent = components[:,end] # eigenvalues ordered from smallest to largest
    factor = data*firstComponent

    return factor 
end 

######################
######################
######################
@doc """

    stateSpaceModel(num_obs, H, A, F, μ, R, Q)

Description: 
Generate data from a DGP in state space form.
Measurement Equation:   
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t} 
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t}
    e_{t} ~ i.i.d.N(0,R)
    v_{t} ~ i.i.d.N(0,Q)
    z_{t} ~ i.i.d.N(0,Z)
    E(e_t v_s') = 0

Inputs: 
- num_obs   = number of observations
- H         = measurement eq. state coef. matrix
- A         = measurement eq. exogenous coef. matrix
- F         = state eq. companion matrix
- μ         = state eq. intercept term
- R         = covariance matrix on measurement disturbance
- Q         = covariance matrix on state disturbance
- Z         = covariance matrix on predetermined var vector 
"""
function simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)
    
    data_y  = zeros(num_obs,size(H)[1])
    data_z  = zeros(num_obs,size(Z)[1])

    β0 = inv(I-F)*μ
    y0 = H*β0

    if Z == zeros(size(Z)[1],size(Z)[1])
        z0 = zeros(size(Z)[1])
    else 
        z0 = rand( MvNormal( zeros( size(Z)[1] ), Z ) )
    end 

    data_y[1,:] = y0
    data_z[1,:] = z0

    β_lag       = β0

    for t in 2:num_obs
        v           = rand(MvNormal(0*[1:size(Q)[1]], Q))
        β           = μ + F*β_lag + v
        if Z == zeros(size(Z)[1],size(Z)[1])
            z = zeros(size(Z)[1])
        else 
            z = rand( MvNormal( zeros( size(Z)[1] ), Z ) )
        end 
        e = rand(MvNormal(0*[1:size(R)[1]], R))
        y           = H*β + A*z + e
        data_y[t,:] = y
        data_z[t,:] = z
        β_lag       = β
    end 

    return data_y, data_z 
end 