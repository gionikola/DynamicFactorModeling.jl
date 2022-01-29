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
    PCA2LevelEstimator(data::Array{Float64,2}, hdfm::HDFMStruct)

Description:
Estimate a two-level HDFM using the PCA approach.
The latent factors are estimated using PCA, while the hyperparameters are estimated using the Bayesian approach outlined in Kim and Nelson (1999).  

Inputs:
- data = Matrix with each column being a data series. 
- hdfm = Model structure specification. 

Outputs:
- results = HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters.
"""
function PCA2LevelEstimator(data::Array{Float64,2}, hdfm::HDFMStruct)

    # Unpack simulation parameters 
    @unpack nlevels, nfactors, factorassign, factorlags, errorlags, ndraws, burnin = hdfm

    # Save total number of Monte Carlo draws 
    totdraws = ndraws + burnin

    # Store data as separate object 
    y = data

    # nvar = number of variables including the variable with missing date
    # nobs = length of data of complete dataset
    nobs, nvars = size(y)
    nvar = nvars

    # Store factor and parameter counts 
    nfacts = sum(nfactors)       # # of factors 
    factorlags = max(factorlags...)       # autoregressive lags in the dynamic factors 
    errorlags = max(errorlags...)   # number of AR lags to include in each observable equation
    nregs = 1 + nlevels          # # of regressors in each obs. eq. (intercept + factors)

    # Count number of variables each 2nd-level factor loads on
    # using a vector called `fnvars`
    # where the i-th entry represents the i-th 2nd-level factors 
    fnvars = zeros(Int, nfacts - 1)
    for i in 1:(nfacts-1)
        for j in 1:nvars
            if fassign[j, 2] == i
                fnvars[i] = fnvars[i] + 1
            end
        end
    end

    # Track variables assigned to each 2nd-level factor 
    # using a list of vectors called `varassign`
    # where the i-th entry is a vector 
    # that contains the indices of the corresponding variables 
    # in the dataset
    varassign = Any[]
    for i in 1:(nfacts-1) # iterate over level-2 factors 
        push!(varassign, Any[])
        for j in 1:nvars # iterate over variables 
            if fassign[j, 2] == i
                push!(varassign[i], j)
            end
        end
    end

    # De-mean data series 
    y = y - repeat(mean(y, dims = 1), nobs, 1)

    # Set up some matricies for storage (optional)
    Xtsave = zeros(nobs, nfacts, totdraws)              # Keep draw of factor, not all states (others are trivial)
    bsave = zeros(totdraws, nregs * nvars)              # Observable equation regression coefficients
    ssave = zeros(totdraws, nvars)                      # Innovation variances
    psave = zeros(totdraws, nfacts * (1 + factorlags))  # Factor autoregressive polynomials
    psave2 = zeros(totdraws, nvars * errorlags)         # Idiosyncratic disturbance autoregressive polynomials

    # Initialize hyperparameters 
    sigmas = ones(nvars)                    # Idiosyncratic error variance vector 
    psis = zeros(nfacts, 1 + factorlags)               # Factor AR companion matrix 
    betas = zeros(nvars, 1 + nlevels)                # Obs. eq. regression starting coefficient matrix 
    phis = zeros(nvars, errorlags)          # Idiosyncratic error AR companion matrix 

    # Estimate factors 
    factor = zeros(nobs, nfacts)           # Random starting factor series matrix 
    factor[:, 1], component = firstComponentFactor(y)     # Starting global factor = crosssectional mean of obs. series 
    for i in 1:(nfacts-1)              # Set level-2 factors equal to their respective group means
        factor[:, 1+i], component2 = firstComponentFactor(y[:, varassign[i]] - factor[:, 1] * component[varassign[i]]')
        if cor(factor[:, 1+i], y[:, varassign[i][1]] - factor[:, 1]) < 0
            factor[:, 1+i] = -factor[:, 1+i]
        end
    end
    if cor(factor[:, 1], y[:, 1] - y[:, varassign[1][1]]) < 0
        factor[:, 1] = -factor[:, 1]
    end

    # Begin Monte Carlo Loop
    for dr = 1:totdraws
    
        println(dr)
    
        ##################################
        ##################################
        # Draw β, σ2, ϕ
    
        ## Iterate over all data series 
        ## to draw obs. eq. hyperparameters 
        for i = 1:nvar
    
            ϕold = zeros(errorlags)
            if dr > 1
                ϕold = psave2[dr-1, ((i-1)*errorlags)+1:i*errorlags]
                ϕold = vec(ϕold)
            end
            ## Gather all regressors into `X`
            X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i, 2]]]
    
            ## Initialize β, σ2, ϕ
            β = ones(1 + nlevels)
            σ2 = 0
            ϕ = zeros(errorlags)
    
            ## Save i-th series 
            Y = y[:, i]
            ind1 = 0
            ind2 = 0
    
            if i == 1 && i == varassign[factorassign[i, 2]][1]
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                while β[2] < 0 || β[3] < 0
                    if β[2] < 0
                        ind1 += 1
                    else
                        ind1 -= 1
                    end
                    if β[3] < 0
                        ind2 += 1
                    else
                        ind2 -= 1
                    end
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                    println("Factor 1 index: $ind1")
                    println("Factor 2 index: $ind2")
                    if ind1 > 100
                        ind1 = 0
                        factor[:, 1] = -factor[:, 1]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i, 2]]]
                    end
                    if ind2 > 100
                        ind2 = 0
                        factor[:, 1+factorassign[i, 2]] = -factor[:, 1+factorassign[i, 2]]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i, 2]]]
                    end
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                end
            elseif i == 1 && i != varassign[factorassign[i, 2]][1]
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                while β[2] < 0
                    ind1 += 1
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                    println("Factor 1 index: $ind1")
                    if ind1 > 100
                        ind1 = 0
                        factor[:, 1] = -factor[:, 1]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i, 2]]]
                        ## Draw observation eq. hyperparameters 
                        β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                    end
                end
            elseif i != 1 && i == varassign[factorassign[i, 2]][1]
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                while β[3] < 0
                    ind2 += 1
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                    println("Factor 2 index: $ind2")
                    if ind2 > 100
                        ind2 = 0
                        factor[:, 1+factorassign[i, 2]] = -factor[:, 1+factorassign[i, 2]]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i, 2]]]
                        ## Draw observation eq. hyperparameters 
                        β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
                    end
                end
            else
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, ϕold, errorlags)
            end
    
            ## Fill out HDFM objects 
            betas[i, :] = β'
            sigmas[i] = σ2
            phis[i, :] = ϕ'
    
            ## Save observation eq. hyperparameter draws 
            bsave[dr, ((i-1)*nregs)+1:i*nregs] = β'
            ssave[dr, i] = σ2
            psave2[dr, ((i-1)*errorlags)+1:i*errorlags] = ϕ'
        end
    
        ##################################
        ##################################
        # Draw factor lag coefficients 
    
        for i in 1:nfacts
            ## Create factor regressor matrix 
            X = zeros(nobs, 1 + factorlags)
            X[:, 1] = ones(nobs)
            for j in 1:factorlags
                X[:, 1+j] = lag(factor[:, i], j, default = 0.0)
            end
            X = X[(factorlags+1):nobs, :]
    
            ind = 0
            accept = 0
            ψ = zeros(1 + factorlags)
            while accept == 0
    
                ind += 1
    
                ## Draw ψ
                ψ = linearRegressionSamplerRestrictedVariance(factor[(factorlags+1):nobs, i], X, 1.0)
    
                ## Check for stationarity 
                coef = [-reverse(vec(ψ[2:end]), dims = 1); 1]                      # check stationarity 
                root = roots(Polynomial(reverse(coef)))
                rootmod = abs.(root)
                accept = min(rootmod...) >= 1.01
    
                ## If while loop goes on for too long 
                if ind > 100
                    ψ = psave[dr-1, ((i-1)*(1+factorlags)+1):(i*(1+factorlags))]
                    coef = [-reverse(vec(ψ[2:end]), dims = 1); 1]                      # check stationarity 
                    root = roots(Polynomial(reverse(coef)))
                    rootmod = abs.(root)
                    accept = min(rootmod...) >= 1.01
                end
            end
    
            ## Fill out HDFM objects 
            psis[i, :] = ψ'
    
            ## Save new draw of ψ
            psave[dr, ((i-1)*(1+factorlags)+1):(i*(1+factorlags))] = ψ'
        end
    
        ##################################
        ##################################
        # Save factors 
        Xtsave[:, 1, dr] = factor[:, 1]
        for c in 1:(nfacts-1)
            Xtsave[:, 1+c, dr] = factor[:, 1+c]
        end
    
        println(dr)
    end

    ##############################################
    ##############################################
    ## Save Monte Carlo samples 
    Xtsave = Xtsave[:, :, (burnin+1):totdraws]
    bsave = bsave[(burnin+1):totdraws, :]
    ssave = ssave[(burnin+1):totdraws, :]
    psave = psave[(burnin+1):totdraws, :]
    psave2 = psave2[(burnin+1):totdraws, :]

    ##############################################
    ##############################################
    ## Save Monte Carlo means 
    F = mean(Xtsave, dims = 3)
    B = mean(bsave, dims = 1)
    S = mean(ssave, dims = 1)
    P = mean(psave, dims = 1)
    P2 = mean(psave2, dims = 1)

    ##############################################
    ##############################################
    means = DFMMeans(F, B, S, P, P2)
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