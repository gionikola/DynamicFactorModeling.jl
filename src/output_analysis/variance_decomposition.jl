@doc """
    vardecomp2level(data::Array{Float64, 2}, factor::Array{Float64}, betas::Array{Float64}, factorassign::Array{Float64})

Description:
Compute the portion of the variation of each observable series that may be attributed to their corresponding/assigned latent factors across all levels. 

Inputs: 
- data = Matrix with each column representing a data series. 
- factor = Matrix containing latent factor estimates.
- betas = Matrix containing observation equation coefficient parameter estimates.
- factorassign = Matrix containing the indeces of factors across all levels (columns) assigned to each observable series (rows). 

Output: 
- vardecomps = Matrix containing the variance contributions of factors across all levels (columns) corresponding to each observable series (rows). 
"""
function vardecomp2level(data, factor, betas, factorassign)

    T, N = size(data)
    T, Nf = size(factor)

    # Record data variances 
    varvars = zeros(N)
    for i in 1:N
        varvars[i] = var(data[:, i])
    end

    # Record factor estimate variances 
    varfacs = zeros(Nf)
    for j in 1:Nf
        varfacs[j] = var(factor[:, j])
    end

    # Compute variance decompositions 
    vardecomps = zeros(N, 2)
    for i in 1:N
        vardecomps[i, 1] = (betas[i, 2]^2 * varfacs[1]) / varvars[i]
        vardecomps[i, 2] = (betas[i, 3]^2 * varfacs[1+factorassign[i, 2]]) / varvars[i]
    end

    return vardecomps
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