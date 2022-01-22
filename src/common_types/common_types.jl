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
    DFMStruct(factorlags, errorlags, ndraws, burnin)

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
    HDFMStruct(nlevels, nvars, nfactors, factorassign, factorlags, errorlags, ndraws, burnin)

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
    DFMMeans(F, B, S, P, P2)
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
    DFMResults(F, B, S, P, P2, means)
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