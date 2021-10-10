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