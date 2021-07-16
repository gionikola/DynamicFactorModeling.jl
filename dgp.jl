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

