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
function sim_MvNormal(μ, Σ)

    obs_aug = μ + cholesky(Hermitian(Σ), Val(true), check = false).U' * randn(length(μ), 1)

    return obs_aug
end

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
function sim_MvNormal_alt(μ,Σ)
    
    nvars = size(Σ)[1]

    keep = Any[]

    for i in 1:nvars
        if Σ[i,i] != 0.0
            push!(keep, i)
        end  
    end 

    Σnew = Σ[keep, keep]
    μnew = μ[keep]

    obs = sim_MvNormal(μnew, Σnew)
    
    obs_aug = zeros(nvars) 

    j = 1
    for i in 1:nvars
        if i in keep
            obs_aug[i] = obs[j]
            j = j + 1   
        else
            obs_aug[i] = μ[i] 
        end 
    end 

    return obs_aug
end 
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
    SSModel(H, A, F, μ, R, Q, Z)

Description:
A type object containing all parameters necessary to specify a data-generating process in state-space form. 
Measurement Equation:   
    y_{t} = H β_{t} + A z_{t} + e_{t} 
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t}
    e_{t} ~ i.i.d.N(0,R)
    v_{t} ~ i.i.d.N(0,Q)
    z_{t} ~ i.i.d.N(0,Z)
    E(e_t v_s') = 0

Inputs:
- H = measurement equation state vector coefficient matrix.
- A = measurement equation predetermined vector coefficient matrix. 
- F = state equation companion matrix.
- μ = state equation intercept vector.
- R = measurement equation error covariance matrix. 
- Q = state equation innovation covariance matrix.
- Z = pretermined vector covariance matrix.
"""
@with_kw mutable struct SSModel
    H::Array{Float64,2}   # Measurement equation state vector coefficient matrix 
    A::Array{Float64,2}   # Measurement equation predetermined vector coefficient matrix 
    F::Array{Float64,2}   # State equation companion matrix 
    μ::Array{Float64,1}   # State equation intercept vector 
    R::Array{Float64,2}   # Measurement equation error covariance matrix 
    Q::Array{Float64,2}   # State equation innovation covariance matrix 
    Z::Array{Float64,2}   # Pretermined vector covariance matrix 
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
@doc """

    HDFM(nlevels, nvar, nfactors, fassign, flags, varlags, varcoefs, fcoefs, fvars, varvars) 

Description:
Creates an object of type `HDFM` that contains all parameters necessary to specify a multi-level linear dynamic factor data-generating process.
This is a convenient alternative to specifying an HDFM directly in state-space form. 

Inputs: 
- nlevels = number of levels in the multi-level model structure.
- nvar = number of variables.
- nfactors = number of factors for each level (vector of length `nlevels`). 
- fassign = determines which factor is assigned to which variable for each level (integer matrix of size `nvar` × `nlevels`).
- flags = number of autoregressive lags for factors of each level (factors of the same level are restricted to having the same number of lags; vector of length `nlevels`).
- varlags = number of observed variable error autoregressive lags (vector of length `nvar`).
- varcoefs = vector of coefficients for each variable in the observation equation (length 1+`nlevels`, where first entry represents the intercept). 
- fcoefs = list of `nlevels` number of matrices, for which each row contains vectors of the autoregressive lag coefficients of the corresponding factor. 
- fvars = list of `nlevels` number of vectors, where each entry contains the disturbance variance of the corresponding factors.
- varvars = vector of `nvar` number of entries, where each entry contains the innovation variance of the corresponding variable.
"""
@with_kw mutable struct HDFM
    nlevels::Int64                  # number of levels in the multi-level model structure 
    nvar::Int64                     # number of variables in the dataset 
    nfactors::Array{Int64,1}        # number of factors for each level (vector of length `nlevels`)
    fassign::Array{Int64,2}         # integer matrix of size `nvar` × `nlevels` 
    flags::Array{Int64,1}           # number of autoregressive lags for each factor level (vector of length `nlevels`)
    varlags::Array{Int64,1}         # number of obs. variable error autoregressive lags (vector of length `nvar`)
    varcoefs::Array{Float64,2}      # vector of observation equation coefficients for each variables (length 1+`nlevels`)
    varlagcoefs::Array{Float64,2}   # vector of error lag coefficients for each variable (matrix length `nvar`, width max(varlags))
    fcoefs::Array{Any,1}            # list of `nlevels` number of matrices, where each row contains vectors of autoregressive lag coefficients of corresponding factors
    fvars::Array{Any,1}             # list of `nlevels` number of vectors, where each entry contains the disturbance variance of the corresponding factors
    varvars::Array{Float64,1}       # vector of `nvar` number of entries, where each entry contains innovation variances of corresponding variables
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
######################
@doc """

    createSSforHDFM(hdfm)

Description:
Create state-space form coefficient and variance matrices for an HDFM object.

Inputs:
- hdfm = `HDFM` object with all parameters necessary to specify a multi-level linear dynamic factor data-generating process. 
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
######################
@doc """
    convertHDFMtoSS(hdfm) 

Description:
Converts an `HDFM` object to an `SSModel` object. 

Inputs:
- hdfm = HDFM object containing all HDFM DGP parameters. 

Outputs:
- ssmodel = SSModel object containing all state-space model parameters.
"""
function convertHDFMtoSS(hdfm::HDFM)

    H, A, F, μ, R, Q, Z = createSSforHDFM(hdfm::HDFM)

    ssmodel = SSModel(H, A, F, μ, R, Q, Z)

    return ssmodel
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
######################
@doc """

    simulateSSModel(num_obs, ssmodel::SSModel)

Description: 
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
- num_obs   = number of observations
- ssmodel   = `SSModel` object containing all parameters necessary to specify a data-generating process in state-space form.  
"""
function simulateSSModel(num_obs, ssmodel::SSModel)

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
        z0 = sim_MvNormal_alt(zeros(size(Z)[1]), Z)
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
            v = sim_MvNormal_alt(zeros(size(Q)[1]), Q)
        end
        # Record new state observation 
        β = μ + F * β_lag + v
        # Draw new z observation
        if Z == zeros(size(Z)[1], size(Z)[1])
            z = zeros(size(Z)[1])
        else
            #z = rand(MvNormal(zeros(size(Z)[1]), Z))
            z = sim_MvNormal_alt(zeros(size(Z)[1]), Z)
        end
        # Draw measurement distrubance 
        if R == zeros(size(R)[1], size(R)[1])
            e = zeros(size(R)[1])
        else
            #e = rand(MvNormal(zeros(size(R)[1]), R))
            e = sim_MvNormal_alt(zeros(size(R)[1]), R)
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