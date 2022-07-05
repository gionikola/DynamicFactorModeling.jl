"""
    mvn(μ, Σ)
Description:
Draw from a multivariate normal distribution with mean μ and variance Σ.
Use cholesky decomposition to generate X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix. 
Cov. matrix Σ does not require non-degenerate random variables (nonzero diag.). 
"""
function mvn(μ, Σ)

    nvar  = size(Σ)[1]         # Num. of variables 

    if (0 in diag(Σ)) == false  # No degenerate random vars.
        Q = cholesky(Σ).U       # Upper triang. Cholesky mat.  
        X = Q * randn(length(μ)) + μ    # Multiv. normal vector draw  
    else                        # in case of degenerate random vars.
        keep = Any[] 
        for i in 1:nvar
            if Σ[i,i] != 0
                push!(keep, i)
            end 
        end  
        Σsub = Σ[keep, keep] 
        μsub = μ[keep] 
        Q = cholesky(Σsub).U       # Upper triang. Cholesky mat.  
        Xsub = Q * randn(length(μsub)) + μsub    # Multiv. normal vector draw  
        X = zeros(nvar) 
        j = 1
        for i in 1:nvar
            if i in keep    # If i-th var. is non-degen. 
                X[i] = Xsub[j]
                j = j + 1   
            else
                X[i] = μ[i] # If i-th var. is degen. 
            end 
        end 
    end 

    return X
end 

"""
    mvn_sample(μ, Σ, n)
Description:
Draw `n` observations from a multivariate normal distribution with mean μ and variance Σ.
Use cholesky decomposition to generate `n` draws of X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix. 
Cov. matrix Σ does not require non-degenerate random variables (nonzero diag.). 
"""
function mvn_sample(μ, Σ, n)

    d  = size(Σ)[1]         # Num. of variables  
    datmat = zeros(n, d)    # Empty data matrix 

    # Draw `n` observations 
    for i in 1:n
        datmat[i,:] = mvn(μ, Σ)
    end 

    return datmat 
end 

"""
    Γinv(T,θ)
Description: 
Gamma inverse distribution with T degrees of freedom and scale parameter θ.
"""
function Γinv(T,θ)

    z0 = randn(T)
    z0z0 = z0' * z0

    return σ2 = θ/z0z0
end 