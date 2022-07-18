"""
SSModel(
    H::Array{Float64,2}, 
    A::Array{Float64,2}, 
    F::Array{Float64,2}, 
    μ::Array{Float64,1}, 
    R::Array{Float64,2}, 
    Q::Array{Float64,2}, 
    Z::Array{Float64,2}
)

A type object containing all parameters necessary to specify a data-generating process in state-space form. 
Measurement Equation:   
- y_{t} = H β_{t} + A z_{t} + e_{t} 
Transition Equation:    
- β_{t} = μ + F β_{t-1} + v_{t}
- e_{t} ~ i.i.d.N(0,R)
- v_{t} ~ i.i.d.N(0,Q)
- z_{t} ~ i.i.d.N(0,Z)
- E(e_{t} v_{s}') = 0

Inputs:
- H = measurement equation state vector coefficient matrix.
- A = measurement equation predetermined vector coefficient matrix. 
- F = state equation companion matrix.
- μ = state equation intercept vector.
- R = measurement equation error covariance matrix. 
- Q = state equation innovation covariance matrix.
- Z = predetermined vector covariance matrix.
"""
@with_kw mutable struct SSModel
H::Array{Float64,2}  
A::Array{Float64,2}  
F::Array{Float64,2}    
μ::Array{Float64,1}   
R::Array{Float64,2}  
Q::Array{Float64,2}  
Z::Array{Float64,2}  
end;

"""
    HDFM(
        nlevels::Int64                   
        nvar::Int64                     
        nfactors::Array{Int64,1}        
        fassign::Array{Int64,2}          
        flags::Array{Int64,1}         
        varlags::Array{Int64,1}        
        varcoefs::Array{Any,2}          
        varlagcoefs::Array{Any,2}    
        fcoefs::Array{Any,1}           
        fvars::Array{Any,1}             
        varvars::Array{Any,1}  
    ) 

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
    nlevels::Int64                   
    nvar::Int64                     
    nfactors::Array{Int64,1}        
    fassign::Array{Int64,2}          
    flags::Array{Int64,1}         
    varlags::Array{Int64,1}        
    varcoefs::Array{Any,2}          
    varlagcoefs::Array{Any,2}    
    fcoefs::Array{Any,1}           
    fvars::Array{Any,1}             
    varvars::Array{Any,1}         
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
    DFMStruct(factorlags::Int64, errorlags::Int64, ndraws::Int64, burnin::Int64)

Description:
1-level DFM lag structure specification and MCMC sample size for Bayesian estimation. 

Inputs:
- factorlags = Number of lags in the autoregressive specification of the latent factors. 
- errorlags = Number of lags in the autoregressive specification of the observable variable idiosyncratic errors.
- ndraws = Number of MCMC draws used for posterior distributions.
- burnin = Number of initial MCMC draws discarded. 
"""
@with_kw mutable struct DFMStruct
    factorlags::Int64
    errorlags::Int64
    ndraws::Int64
    burnin::Int64
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
    HDFMStruct(nlevels::Int64, nfactors::Array{Int64,1}, factorassign::Array{Int64,2}, factorlags::Array{Int64,1}, errorlags::Array{Int64,1}, ndraws::Int64, burnin::Int64)

Description:
Multi-level/hierarchical DFM (HDFM) level, factor assignment, and lag structure specification, and MCMC sample size for Bayesian estimation. 

Inputs:
- nlevels = Number of levels in the HDFM specification. 
- nvars = Number of observable variables in the HDFM specification. 
- nfactors = Number of factor per level in the HDFM specification. 
- factorassign = Factors assigned to each variable across all levels. 
- factorlags = Number of lags in the autoregressive specification of the latent factors. 
- errorlags = Number of lags in the autoregressive specification of the observable variable idiosyncratic errors.
- ndraws = Number of MCMC draws used for posterior distributions.
- burnin = Number of initial MCMC draws discarded. 
"""
@with_kw mutable struct HDFMStruct
    nlevels::Int64                  # number of levels in the multi-level model structure 
    nfactors::Array{Int64,1}        # number of factors for each level (vector of length `nlevels`)
    factorassign::Array{Int64,2}         # integer matrix of size `nvar` × `nlevels` 
    factorlags::Array{Int64,1}           # number of autoregressive lags for each factor level (vector of length `nlevels`)
    errorlags::Array{Int64,1}         # number of obs. variable error autoregressive lags (vector of length `nvar`)
    ndraws::Int64
    burnin::Int64
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
    DFMMeans(F::Array{Float64}, B::Array{Float64}, S::Array{Float64}, P::Array{Float64}, P2::Array{Float64})

Description:
HDFM Bayesian estimator-generated latent factor and hyperparameter sample means (expected values). 

Inputs:
- F = MCMC-generated latent factor sample mean.
- B = MCMC-generated observation equation regression coefficient sample means.
- S = MCMC-generated observable variable idiosyncratic error disturbance variance sample means. 
- P = MCMC-generated latent factor autoregressive coefficient sample means. 
- P2 = MCMC-generated idiosyncratic error autoregressive coefficient sample means. 
"""
@with_kw mutable struct DFMMeans
    F::Array{Float64}   # Factor means 
    B::Array{Float64}   # Obs. equation coefficient means 
    S::Array{Float64}   # Idiosyncratic disturbance variance means 
    P::Array{Float64}   # Factor autoregressive coefficient means 
    P2::Array{Float64}  # Idiosyncratic disturbance autoregressive means 
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
    DFMResults(F::Array{Float64}, B::Array{Float64}, S::Array{Float64}, P::Array{Float64}, P2::Array{Float64}, means::DFMMeans)

Description:
HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters. 

Inputs:
- F = MCMC-generated latent factor sample.
- B = MCMC-generated observation equation regression coefficient sample.
- S = MCMC-generated observable variable idiosyncratic error disturbance variance sample. 
- P = MCMC-generated latent factor autoregressive coefficient sample. 
- P2 = MCMC-generated idiosyncratic error autoregressive coefficient sample. 
- means = HDFM Bayesian estimator-generated latent factor and hyperparameter sample means (expected values).
"""
@with_kw mutable struct DFMResults
    F::Array{Float64}   # Factor sample 
    B::Array{Float64}   # Obs. equation coefficient sample 
    S::Array{Float64}   # Idiosyncratic disturbance variance sample 
    P::Array{Float64}   # Factor autoregressive coefficient sample 
    P2::Array{Float64}  # Idiosyncratic disturbance autoregressive sample 
    means::DFMMeans     # Factor and hyperparameter means
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