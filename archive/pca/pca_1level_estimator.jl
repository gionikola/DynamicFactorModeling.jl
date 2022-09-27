include("pca_tools.jl")
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
    PCA1LevelEstimator(data::Array{Float64,2}, dfm::DFMStruct)

Description:
Estimate a single-factor DFM using the Kim-Nelson approach. 

Inputs:
- data = Matrix with each column being a data series. 
- dfm = Model structure specification. 

Outputs:
- results = HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters.
"""
function PCA1LevelEstimator(data::Array{Float64,2}, dfm::DFMStruct)

    # Unpack simulation parameters 
    @unpack factorlags, errorlags, ndraws, burnin = dfm

    # Save total number of Monte Carlo draws 
    totdraws = ndraws + burnin

    # Store data as separate object 
    y = data

    # nvar = number of variables including the variable with missing date
    # nobs = length of data of complete dataset
    nobs, nvar = size(y)
    nvars = nvar 

    # Number of regressors in each observable equation
    # (constant + global factor)
    nreg = 2

    # De-mean data series 
    y = y - repeat(mean(y, dims = 1), nobs, 1)

    # Set up some matricies for storage (optional)
    Xtsave = zeros(nobs, totdraws)                  # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(totdraws, nreg * nvar)           # observable equation regression coefficients
    ssave = zeros(totdraws, nvar)                  # innovation variances
    psave = zeros(totdraws, factorlags)            # factor autoregressive polynomials
    psave2 = zeros(totdraws, nvar * errorlags)      # factor autoregressive polynomials

    # Initialize global factor 
    factor = zeros(nobs, 1)           # Random starting factor series matrix 
    factor[:, 1], component = firstComponentFactor(y)     # Starting global factor = crosssectional mean of obs. series 
    if cor(factor[:, 1], y[:, 1]) < 0
        factor[:, 1] = -factor[:, 1]
    end

    # Begin Monte Carlo Loop
    for dr = 1:totdraws
    
        println(dr)
    
        sigmas = zeros(nvar)
        if dr == 1
            sigmas = ones(nvar)
        else
            sigmas = vec(ssave[dr-1, :])
        end
    
        # Create HDFM parameter containers 
        varcoefs = zeros(nvar, 2)[:, :]
        varlagcoefs = zeros(nvar, errorlags)[:, :]
        fcoefs = Any[]
        push!(fcoefs, zeros(factorlags))
        fvars = Any[]
        push!(fvars, ones(1))
        varvars = zeros(nvar)
    
        ##################################
        ##################################
        # Draw β, σ2, ϕ
    
        ## Gather all regressors into `X`
        X = [ones(nobs) factor]
    
        ## Initialize β, σ2, ϕ
        β = ones(2)
        ϕ = zeros(errorlags)
    
        ## Iterate over all data series 
        ## to draw obs. eq. hyperparameters 
        for i = 1:nvar
        
            ## Save i-th series 
            Y = y[:, i]
        
            ϕold = zeros(errorlags)
            if dr > 1
                ϕold = psave2[dr-1, ((i-1)*errorlags)+1:i*errorlags]
                ϕold = vec(ϕold)
            end
            σ2 = sigmas[i]
        
            if i == 1
                ind = 0
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, σ2, errorlags)
                while β[2] < 0
                    ind += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, σ2, errorlags)
                    if ind >= 100
                        factor = -factor
                        X = [ones(nobs) factor]
                    end
                end
            else
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, σ2, errorlags)
            end
        
            ## Fill out HDFM objects 
            varcoefs[i, :] = β'
            varvars[i] = σ2
            varlagcoefs[i, :] = ϕ'
        
            ## Save observation eq. hyperparameter draws 
            bsave[dr, ((i-1)*nreg)+1:i*nreg] = β'
            ssave[dr, i] = σ2
            psave2[dr, ((i-1)*errorlags)+1:i*errorlags] = ϕ'
        
        end
    
        ##################################
        ##################################
        # Draw factor lag coefficients 
    
        ## Create factor regressor matrix 
        X = zeros(nobs, factorlags)
        for j in 1:factorlags
            X[:, j] = lag(factor, j, default = 0.0)
        end
        X = X[(factorlags+1):nobs, :]
    
        ind = 0
        accept = 0
        ψ = zeros(factorlags)
        while accept == 0
    
            ind += 1
    
            ## Draw ψ
            ψ, discard = linearRegressionSampler(factor[(factorlags+1):nobs, 1], X)
    
            ## Check for stationarity 
            coef = [-reverse(vec(ψ), dims = 1); 1]                      # check stationarity 
            root = roots(Polynomial(reverse(coef)))
            rootmod = abs.(root)
            accept = min(rootmod...) >= 1.01
    
            ## If while loop goes on for too long 
            if ind > 100
                if dr == 1
                    ψ = zeros(factorlags)
                    accept = 1
                else
                    ψ = psave[dr-1, :]
                    coef = [-reverse(vec(ψ), dims = 1); 1]                      # check stationarity 
                    root = roots(Polynomial(reverse(coef)))
                    rootmod = abs.(root)
                    accept = min(rootmod...) >= 1.01
                end
            end
        end
    
        ## Fill out HDFM objects 
        fcoefs = ψ
    
        ## Save new draw of ψ
        psave[dr, :] = ψ'
    
        ##################################
        ##################################
        # Draw factor  
    
        ## Save factor 
        Xtsave[:, dr] = factor
    
        println(dr)
    end

    # Save resulting samples 
    Xtsave = Xtsave[:, (burnin)+1:(burnin+ndraws)]
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