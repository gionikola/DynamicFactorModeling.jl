"""
    mvn(μ, Σ)

Draw from a multivariate normal distribution with mean vector μ and covariance matrix Σ.
Use cholesky decomposition to generate X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix. 
Cov. matrix Σ does not require non-degenerate random variables (nonzero diagonal). 

Inputs:
- μ = mean vector 
- Σ = covariance matrix 

Outputs:
- X::Array{Float64, 1} = observed draw of X ~ N(μ,Σ)
"""
function mvn(μ::Vector{Float64}, Σ::Matrix{Float64})

    nvar  = size(Σ)[1]         # Num. of variables 

    if (0 in diag(Σ)) == false  # No degenerate random vars.
        Q = cholesky(Hermitian(Σ), Val(true), check = false).U       # Upper triang. Cholesky mat.  
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
        Q = cholesky(Hermitian(Σsub), Val(true), check = false).U       # Upper triang. Cholesky mat.  
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

    return X::Vector{Float64}
end

function mvn(μ::Vector{Int}, Σ::Matrix{Int})

    nvar  = size(Σ)[1]         # Num. of variables 

    if (0 in diag(Σ)) == false  # No degenerate random vars.
        Q = cholesky(Hermitian(Σ), Val(true), check = false).U       # Upper triang. Cholesky mat.  
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
        Q = cholesky(Hermitian(Σsub), Val(true), check = false).U       # Upper triang. Cholesky mat.  
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

    return X::Vector{Float64}
end

function mvn(μ::Float64, Σ::Float64)

    if Σ != 0  # No degenerate random vars.     # Upper triang. Cholesky mat.  
        X = sqrt(Σ) * randn() + μ    # Multiv. normal vector draw  
    else # in case of degenerate random vars.
        X = μ
    end

    return X::Float64
end

function mvn(μ::Int, Σ::Int)

    if Σ != 0  # No degenerate random vars.     # Upper triang. Cholesky mat.  
        X = sqrt(Σ) * randn() + μ    # Multiv. normal vector draw  
    else # in case of degenerate random vars.
        X = μ
    end

    return X::Float64
end

"""
    mvn(μ, Σ, n::Int64)

Draw `n` number of observations from a multivariate normal distribution with mean vector μ and covariance matrix Σ.
Use cholesky decomposition to generate `n` draws of X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix. 
Cov. matrix Σ does not require non-degenerate random variables (nonzero diag.). 

Inputs:
- μ = mean vector 
- Σ = covariance matrix 
- n = number of draws 

Output:
- X::Array{Float64, 2} = simulated data matrix composed of n-number of draws of X ~ N(μ,Σ)
"""
function mvn(μ::Vector{Float64}, Σ::Matrix{Float64}, n::Int64)

    d  = size(Σ)[1]         # Num. of variables  
    X = zeros(n, d)    # Empty data matrix 

    # Draw `n` observations 
    for i in 1:n
        X[i,:] = mvn(μ, Σ)
    end 

    return X::Matrix{Float64} 
end

function mvn(μ::Float64, Σ::Float64, n::Int64)

    X = zeros(n, 1)    # Empty data matrix 

    # Draw `n` observations 
    for i in 1:n
        X[i, 1] = mvn(μ, Σ)
    end

    return X::Matrix{Float64}
end

"""
    Γinv(T::Int64, θ::Float64)

Gamma inverse distribution with T degrees of freedom and scale parameter θ.

Inputs:
- T = degrees of freedom 
- θ = scale parameter 

Output:
- σ2 = draw of X ~ Γ_inverse(T,θ)

"""
function Γinv(T,θ)

    z0 = randn(T)
    z0z0 = z0' * z0

    return σ2 = θ/z0z0
end 

"""
    createSSforHDFM(hdfm::HDFM))

Description:
Create state-space form coefficient and variance matrices for an HDFM object.
Measurement Equation:   
- y_{t} = H β_{t} + A z_{t} + e_{t} 
Transition Equation:    
- β_{t} = μ + F β_{t-1} + v_{t}
- e_{t} ~ i.i.d.N(0,R)
- v_{t} ~ i.i.d.N(0,Q)
- z_{t} ~ i.i.d.N(0,Z)
- E(e_{t} v_{s}') = 0

Inputs:
- hdfm::HDFM

Output:
- H = measurement equation state vector coefficient matrix.
- A = measurement equation predetermined vector coefficient matrix. 
- F = state equation companion matrix.
- μ = state equation intercept vector.
- R = measurement equation error covariance matrix. 
- Q = state equation innovation covariance matrix.
- Z = predetermined vector covariance matrix.
"""
function createSSforHDFM(hdfm::HDFM)

    ######################################
    ## Import all HDFM parameters 
    @unpack nlevels, nvar, nfactors, fassign, flags, varlags, varcoefs, varlagcoefs, fcoefs, fvars, varvars = hdfm

    ######################################
    ## Specify observation equation coefficient matrix 

    # Store number of total lag terms in the state vector 
    ntotlags = sum(varlags) + dot(nfactors,flags)           # variable error lags

    # Create empty observation eq. coefficient matrix 
    H = zeros(nvar, 1 + ntotlags)               # intercept + total lags 

    # Fill out observation eq. coefficient matrix 
    for i = 1:nvar
        H[i, 1] = varcoefs[i, 1]
        for j = 1:nlevels
            for k = 1:nfactors[j]
                if k == fassign[i, j]
                    H[i, 1 + sum(nfactors[1:(j-1)]) + k] = varcoefs[i, 1 + j]
                end
            end
        end
    end
    H[:, (1+sum(nfactors)+1):(1+sum(nfactors)+nvar)] = 1.0 .* Matrix(I(nvar))

    ######################################
    ## Specify observation equation error covariance matrix
    R = zeros(nvar, nvar)

    ######################################
    ## Specify state equation companion matrix
    ## and intercept vector 

    # Create empty state equation companion matrix and intercept vector 
    slength = size(H)[2]                 # length of state vector
    F = zeros(slength, slength)
    μ = zeros(slength)

    # Total number of factors 
    ntotfactors = sum(nfactors)

    # Fill out transition eq. companion matrix 
    μ[1,1] = 1.0
    for i = 1:nlevels
        for j = 1:nfactors[i]
            
            rowind = 0
            if i > 1
                rowind = sum(nfactors[1:(i-1)]) + j
            else 
                rowind = j
            end 

            for k = 1:flags[i]
                F[1+rowind, 1+rowind+(ntotfactors+nvar)*(k-1)] = (fcoefs[i])[j, k] # factor autoregressive lag coefficients
            end
        end
    end
    for i = 1:nvar
        for j = 1:varlags[i]
            F[ntotfactors+i, 1+ntotfactors+(nvar)*(j-1)+i] = varlagcoefs[i, j] # obs. eq. error lag coefficients 
        end
    end
    for i = (ntotfactors+nvar+1):(slength)
        for j = 1:(slength-ntotfactors-nvar)
            if i == ntotfactors + nvar + j
                F[i, j] = 1.0
            end 
        end
    end

    ######################################
    ## Specify state equation error covariance matrix

    # Create empty state equation error covariance matrix 
    Q = zeros(slength, slength)

    # Fill out state equation error covariance matrix 
    for i = 1:nlevels
        for j = 1:nfactors[i]
        
            rowind = 0
            if i > 1
                rowind = sum(nfactors[1:(i-1)]) + j
            else
                rowind = j
            end
        
            Q[1+rowind, 1+rowind] = (fvars[i])[j]
        end
    end
    for i = 1:nvar
        Q[1+ntotfactors+i, 1+ntotfactors+i] = varvars[i]
    end

    ######################################
    ## Specify all predetermined variable-related parameters 
    A = zeros(nvar, nvar)
    Z = zeros(nvar, nvar)

    ######################################
    return H, A, F, μ, R, Q, Z
end;

"""
    convertHDFMtoSS(hdfm::HDFM) 

Description:
Converts an `HDFM` object to an `SSModel` object. 

Inputs:
- hdfm::HDFM

Output:
- ssmodel::SSModel 
"""
function convertHDFMtoSS(hdfm::HDFM)

    H, A, F, μ, R, Q, Z = createSSforHDFM(hdfm::HDFM)

    ssmodel = SSModel(H, A, F, μ, R, Q, Z)

    return ssmodel::SSModel
end;

"""
    simulateSSModel(num_obs::Int64, ssmodel::SSModel)

Generate data from a DGP in state space form.
Measurement Equation:   
    y_{t} = H β_{t} + A z_{t} + e_{t} 
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t}
    e_{t} ~ i.i.d.N(0,R)
    v_{t} ~ i.i.d.N(0,Q)
    z_{t} ~ i.i.d.N(0,Z)
    E(e_t v_s') = 0

Inputs: 
- num_obs           = number of observations
- ssmodel::SSModel 

Output:
- data_y = simulated sample of observed vector  
- data_z = simulated sample of exogenous variables
- data_β = simulated sample of state vector 
"""
function simulateSSModel(num_obs::Int64, ssmodel::SSModel)

    @unpack H, A, F, μ, R, Q, Z = ssmodel

    # Convert cov matrices to PSDMat 
    # to allow for simulation using MvNormal()
    # (MvNormal() needs a positive definite cov matrix)

    # Create empty data storage matrices 
    data_y = zeros(num_obs, size(H)[1])
    data_z = zeros(num_obs, size(Z)[1])
    data_β = zeros(num_obs, size(Q)[1])

    # Initialize β and y 
    β0 = inv(I - F) * μ 
    y0 = H * β0

    # Initialize z
    if Z == zeros(size(Z)[1], size(Z)[1])
        z0 = zeros(size(Z)[1])
    else
        #z0 = rand(MvNormal(zeros(size(Z)[1]), Z))
        z0 = mvn(zeros(size(Z)[1]), Z)
    end

    # Save first observations of y and z
    data_y[1, :] = y0
    data_z[1, :] = z0
    data_β[1, :] = β0

    # Initialize β lag for recursion 
    β_lag = β0

    # Recursively generate data
    for t = 2:num_obs
        # Draw transition distrubance 
        if Q == zeros(size(Q)[1], size(Q)[1])
            v = zeros(size(Q)[1])
        else
            #v = rand(MvNormal(zeros(size(Q)[1]), Q))
            v = mvn(zeros(size(Q)[1]), Q)
        end
        # Record new state observation 
        β = μ + F * β_lag + v
        # Draw new z observation
        if Z == zeros(size(Z)[1], size(Z)[1])
            z = zeros(size(Z)[1])
        else
            #z = rand(MvNormal(zeros(size(Z)[1]), Z))
            z = mvn(zeros(size(Z)[1]), Z)
        end
        # Draw measurement distrubance 
        if R == zeros(size(R)[1], size(R)[1])
            e = zeros(size(R)[1])
        else
            #e = rand(MvNormal(zeros(size(R)[1]), R))
            e = mvn(zeros(size(R)[1]), R)
        end
        # Record new measurement observation 
        y = H * β + A * z + e
        # Save generated data 
        data_y[t, :] = y
        data_z[t, :] = z
        data_β[t, :] = β
        # Update β lag for recursion 
        β_lag = β
    end

    # Return data 
    return data_y, data_z, data_β
end;