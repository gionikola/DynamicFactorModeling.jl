using Random; Distributions 

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

@doc """
    
    gen_rand_cov(N,μ,σ)

Description:
Generate a random covariance matrix. 

Inputs:
- N = number of variables
- μ = expected covariance 
- σ = expected std. deviation of covariance
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
        else 
            data[i] = parameters[1] + 
                        reverse(parameters[2:length(parameters)])[1:(i-1)]'*data[1:(i-1)] +
                        disturbance[i]
        end 
    end 

    return data
end 