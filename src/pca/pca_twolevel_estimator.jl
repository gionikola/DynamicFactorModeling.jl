"""
"""
function firstComponentFactor(data)

    num_obs = size(data)[1]
    centering = I - 1/(num_obs)*ones(num_obs,num_obs) # centering matrix
    data = centering*data # center the data 
    cov_mat = data'*data
    cov_mat_eigen = eigen(cov_mat)
    components = cov_mat_eigen.vectors
    firstComponent = components[:,end] # eigenvalues ordered from smallest to largest
    factor = data*firstComponent

    return factor 
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
"""
@with_kw mutable struct HDFMParams
    nlevels::Int64                  # number of levels in the multi-level model structure 
    nvars::Int64                     # number of variables in the dataset 
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
"""
    KN2LevelEstimator(data, params::HDFMParams)

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
function PCA2LevelEstimator(data::Array{Float64,2}, params::HDFMParams)

    # Unpack simulation parameters 
    @unpack nlevels, nvars, nfactors, factorassign, factorlags, errorlags, ndraws, burnin = params

    # Save total number of Monte Carlo draws 
    totdraws = ndraws + burnin

    # Store data as separate object 
    y = data

    # nvar = number of variables including the variable with missing date
    # nobs = length of data of complete dataset
    nobs, nvar = size(y)

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
    factor[:, 1] = mean(y, dims = 2)     # Starting global factor = crosssectional mean of obs. series 
    for i in 1:(nfacts-1)              # Set level-2 factors equal to their respective group means
        factor[:, 1+i] = mean(y[:, varassign[i]], dims = 2)
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

            ## Gather all regressors into `X`
            X = [ones(nobs) factor[:,1] factor[:,1 + factorassign[i]]]

            ## Initialize β, σ2, ϕ
            β = ones(1+nlevels)
            σ2 = 0
            ϕ = zeros(errorlags)

            ## Save i-th series 
            Y = y[:, i]

            if i == 1 && i == varassign[1+factorassign[i]]
                ind = 0
                ind2 = 0
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                while β[2] < 0 || β[3] < 0
                    ind += 1
                    ind2 += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                    if ind >= 100
                        ind = 0
                        factor[:, 1] = -factor[:, 1]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i]] ]  
                    end
                    if ind2 >= 100
                        ind2 = 0
                        factor[:, 1+factorassign[i]] = -factor[:, 1+factorassign[i]]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i]]]
                    end
                end
            elseif i == 1 && i != varassign[1+factorassign[i]]
                ind = 0
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                while β[2] < 0
                        ind += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                    if ind >= 100
                        ind = 0
                        factor[:, 1] = -factor[:, 1]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i]]]
                    end
                end
            elseif i != 1 && i == varassign[1+factorassign[i]]
                ind2 = 0
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                while β[3] < 0
                    ind2 += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                    if ind2 >= 100
                        ind2 = 0
                        factor[:, 1+factorassign[i]] = -factor[:, 1+factorassign[i]]
                        X = [ones(nobs) factor[:, 1] factor[:, 1+factorassign[i]]]
                    end
                end
            else
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
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
        
            ## Draw ψ
            ψ = linearRegressionSamplerRestrictedVariance(factor[(factorlags+1):nobs, i], X, 1.0)
        
            ## Fill out HDFM objects 
            psis[i, :] = ψ'
        
            ## Save new draw of ψ
            psave[dr, ((i-1)*(1+factorlags)+1):(i*(1+factorlags))] = ψ'
        end

        ############################################
        ## Draw global (level-1) factor 
        ############################################

        yW = similar(y) 

        # Fill out important objects 
        for i = 1:nvars  # Iterate over all observable variable 

            # Save level-2 factor index assigned to obs. variable i 
            nfC = fassign[i,2]

            # Partial out variation in variable i due to intercept + level-2 factor 
            yW[:, i] = y[:, i] - ones(nobs, 1) * betas[i, 1] - factor[:, 1+nfC] * betas[i, 3]

        end

        # Obtain new draw of the global factor 
        fact1 = firstComponentFactor(yW) 

        # Save draw in output object 
        Xtsave[:, 1, dr] = fact1

        # Update factor data matrix to contain
        # new global factor draw 
        factor[:, 1] = fact1

        ############################################
        ## Draw level-2 factors 
        ############################################

        for c = 1:(nfacts-1) # Iterate over level-2 factors 
        
            # Store number of obs. variables 
            # to which level-2 factor number c 
            # gets assigned 
            Size = fnvars[c]
        
            yC = zeros(nobs, Size)
        
            for i = 1:Size # Iterate over all obs. variables assigned to level-2 factor c 
        
                # Partial out variation in variables assigned to level-2 factor c
                # corresponding to variation in intercept + level-1 factor 
                yC[:, i] = y[:, varassign[c][i]] - ones(nobs, 1) * betas[varassign[c][i], 1] - factor[:, 1] * betas[varassign[c][i], 2]
        
            end
        
            # Obtain new draw of level-2 factor c
            fact2 = firstComponentFactor(yC)
        
            # Save draw in output object 
            Xtsave[:, 1+c, dr] = fact2
        
            # Update factor data matrix to contain
            # new level-2 factor c draw 
            factor[:, 1+c] = fact2
        
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
    results = OWResults(Xtsave, bsave, ssave, psave, psave2, means)

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