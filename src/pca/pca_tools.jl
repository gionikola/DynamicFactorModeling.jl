######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
@doc """
"""
function firstComponentFactor(data)

    num_obs = size(data)[1]
    centering = I - 1 / (num_obs) * ones(num_obs, num_obs) # centering matrix
    data = centering * data # center the data 
    cov_mat = data' * data
    cov_mat_eigen = eigen(cov_mat)
    components = cov_mat_eigen.vectors
    firstComponent = components[:, end] # eigenvalues ordered from smallest to largest
    firstComponent = sqrt(num_obs) * firstComponent
    factor = data * firstComponent / num_obs

    return factor, firstComponent
end;
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################