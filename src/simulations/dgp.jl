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

    # Convert cov matrices to PSDMat 
    # to allow for simulation using MvNormal()
    # (MvNormal() needs a positive definite cov matrix)

    if isposdef(R) == false
        R = PSDMat(R)
    end
    if isposdef(Q) == false
        Q = PSDMat(Q)
    end
    if isposdef(Z) == false
        Z = PSDMat(Z)
    end

    # Create empty data storage matrices 
    data_y = zeros(num_obs, size(H)[1])
    data_z = zeros(num_obs, size(Z)[1])
    data_β = zeros(num_obs, size(Q)[1])

    # Initialize β and y 
    β0 = inv(I - F) * μ .+ rand()
    y0 = H * β0

    # Initialize z
    if Z == zeros(size(Z)[1], size(Z)[1])
        z0 = zeros(size(Z)[1])
    else
        z0 = rand(MvNormal(zeros(size(Z)[1]), Z))
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
            v = rand(MvNormal(zeros(size(Q)[1]), Q))
        end
        # Record new state observation 
        β = μ + F * β_lag + v
        # Draw new z observation
        if Z == zeros(size(Z)[1], size(Z)[1])
            z = zeros(size(Z)[1])
        else
            z = rand(MvNormal(zeros(size(Z)[1]), Z))
        end
        # Draw measurement distrubance 
        if R == zeros(size(R)[1], size(R)[1])
            e = zeros(size(R)[1])
        else
            e = rand(MvNormal(zeros(size(R)[1]), R))
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
"""

    HDFM(nlevels, nvar, nfactors, fassign, flags, varlags, varcoefs, fcoefs, fvars, varvars) 

Description:
Creates an object of type `HDFM` that contains all parameters necessary to specify a multi-level linear dynamic factor data-generating process. 

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
    varcoefs::Array{Int64,2}        # vector of coefficients for each variables (length 1+`nlevels`)
    fcoefs::Array{Any,1}            # list of `nlevels` number of matrices, where each row contains vectors of autoregressive lag coefficients of corresponding factors
    fvars::Array{Any,1}             # list of `nlevels` number of vectors, where each entry contains the disturbance variance of the corresponding factors
    varvars::Array{Int64,1}         # vector of `nvar` number of entries, where each entry contains innovation variances of corresponding variables
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