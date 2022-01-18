include("kn_tools.jl")
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
    SamplerParams(factorlags, errorlags)

Description:
Model priors for the Kim-Nelson estimator. 

Inputs:
- factorlags = Number of lags in the factor equation. 
- errorlags = Number of AR lags in the observation equation. 
- ndraws = Number of Monte Carlo draws.
- burnin = Number of initial draws to discard.
"""
@with_kw mutable struct SamplerParams
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
"""
    KNSingleFactorEstimator(data, priorsIN)

Description:
Estimate a single-factor DFM using the Kim-Nelson approach. 

Inputs:
- data = set of observed variables with a hypothesized common trend.
- priorsIN = model priors. 

Outputs:
- B = single-factor DFM coefficient hyperparameter estimates. 
- F = single-factor DFM factor estimate. 
- S = single-factor DFM error variance estimates. 
"""
function KNSingleFactorEstimator(data, SamplerParams)

    # Unpack simulation parameters 
    @unpack factorlags, errorlags, ndraws, burnin = SamplerParams

    # Save total number of Monte Carlo draws 
    totdraws = ndraws + burnin

    # Store data as separate object 
    y = data

    # nvar = number of variables including the variable with missing date
    # nobs = length of data of complete dataset
    nobs, nvar = size(y)

    # Save number of factors to estimate 
    nfact = 1

    # Number of regressors in each observable equation
    # (constant + global factor)
    nreg = 2

    # De-mean data series 
    y = y - repeat(mean(ytemp, dims = 1), nobs, 1)

    # Set up some matricies for storage (optional)
    Xtsave = zeros(nobs, totdraws)                  # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(totdraws, nreg * nvar)           # observable equation regression coefficients
    ssave = zeros(totdraws, nvar)                  # innovation variances
    psave = zeros(totdraws, 1 + factorlags)            # factor autoregressive polynomials
    psave2 = zeros(totdraws, nvar * errorlags)      # factor autoregressive polynomials

    # Initialize global factor 
    factor = mean(y, dims = 2)

    # Begin Monte Carlo Loop
    for dr = 1:totdraws

        println(dr)

        # Create HDFM parameter containers 
        varcoefs = zeros(nvar, 2)
        varlagcoefs = zeros(nvar, errorlags)
        fcoefs = zeros(1 + factorlags)
        fvars = ones(1)
        varvars = zeros(nvar)

        ##################################
        ##################################
        # Draw β, σ2, ϕ

        ## Gather all regressors into `X`
        X = [ones(nobs) factor]

        ## Initialize β, σ2, ϕ
        β = zeros(2)
        σ2 = 0
        ϕ = zeros(1 + errorlags)

        ## Iterate over all data series 
        ## to draw obs. eq. hyperparameters 
        for i = 1:nvar

            ## Save i-th series 
            Y = y[:, i]

            if i == 1
                ind = 0
                while β[2] < 0
                    ind += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                    if ind >= 100
                        factor = -factor
                        X = [ones(nobs) factor]
                    end
                end
            end

            ## Fill out HDFM objects 
            varcoefs[i, :] = β'
            varvars[i] = σ2
            varlagcoefs[i, :] = ϕ'

            ## Save observation eq. hyperparameter draws 
            bsave[dr, ((i-1)*nreg)+1:i*nreg] = β'
            ssave[dr, i] = σ2
            psave2[dr, ((i-1)*arterms)+1:i*arterms] = ϕ'

        end

        ##################################
        ##################################
        # Draw factor lag coefficients 

        ## Create factor regressor matrix 
        X = zeros(nobs, 1 + factorlags)
        X[:, 1] = ones(nobs)
        for j in 1:factorlags
            X[:, 1+j] = lag(factor, j, default = 0.0)
        end
        X = X[(factorlags+1):nobs, :]

        ## Draw ψ
        ψ = linearRegressionSamplerRestrictedVariance(factor, X, σ2)

        ## Fill out HDFM objects 
        fcoefs = ψ

        ## Save new draw of ψ
        psave[dr, (i-1)*(arlag+1)+1:(i-1)*(arlag+1)+(arlag+1)] = ψ

        ##################################
        ##################################
        # Draw global level-1 factor  

        ## Specify hierarchical DFM 
        nlevels = 1
        nvar = nvar
        nfactors = 1
        fassign = ones(nvar)
        flags = factorlags
        varlags = errorlags
        hdfm = HDFM(nlevels, nvar, nfactors, fassign, flags, varlags, varcoefs, varlagcoefs, fcoefs, fvars, varvars)

        ## Construct state space model
        ssmodel = convertHDFMtoSS(hdfm)

        ## Draw global factor 
        factor = KNFactorSampler(y, ssmodel)

        ## Save factor 
        Xtsave[:, dr] = factor

        println(dr)
    end

    # Save resulting samples 
    Xtsave = Xtsave[:, (burnin*nfact)+1:(burnin+ndraws)*nfact]
    bsave = bsave[burnin+1:burnin+ndraws, :]
    ssave = ssave[burnin+1:burnin+ndraws, :]
    psave = psave[burnin+1:burnin+ndraws, :]
    psave2 = psave2[burnin+1:burnin+ndraws, :]

    # Save resulting sample means 
    F = mean(Xtsave, dims = 2)
    B = mean(bsave, dims = 1)
    S = mean(ssave, dims = 1)
    P = mean(psave, dims = 1)
    P2 = mean(psave2, dims = 1)
    means = DFMMeans(F, B, S, P, P2)

    # Gather all results 
    results = DFMResults(Xtsave, bsave, ssave, psave, psave2, means)

    ##################################
    ##################################
    # Return results as single object 
    return results
end
;
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