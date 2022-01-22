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
@doc """
    KNSingleFactorEstimator(data::Array{Float64,2}, dfm::DFMStruct)

Description:
Estimate a single-factor DFM using the Kim-Nelson approach. 

Inputs:
- data = set of observed variables with a hypothesized common trend.
- dfm = model priors. 

Outputs:
- B = single-factor DFM coefficient hyperparameter estimates. 
- F = single-factor DFM factor estimate. 
- S = single-factor DFM error variance estimates. 
"""
function KN1LevelEstimator(data::Array{Float64,2}, dfm::DFMStruct)

    # Unpack simulation parameters 
    @unpack factorlags, errorlags, ndraws, burnin = dfm

    # Save total number of Monte Carlo draws 
    totdraws = ndraws + burnin

    # Store data as separate object 
    y = data

    # nvar = number of variables including the variable with missing date
    # nobs = length of data of complete dataset
    nobs, nvar = size(y)

    # Number of regressors in each observable equation
    # (constant + global factor)
    nreg = 2

    # De-mean data series 
    y = y - repeat(mean(y, dims = 1), nobs, 1)

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
        varcoefs = zeros(nvar, 2)[:, :]
        varlagcoefs = zeros(nvar, errorlags)[:, :]
        fcoefs = Any[]
        push!(fcoefs, zeros(1 + factorlags))
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
        σ2 = 0
        ϕ = zeros(errorlags)

        ## Iterate over all data series 
        ## to draw obs. eq. hyperparameters 
        for i = 1:nvar

            ## Save i-th series 
            Y = y[:, i]

            if i == 1
                ind = 0
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                while β[2] < 0
                    ind += 1
                    ## Draw observation eq. hyperparameters 
                    β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
                    if ind >= 100
                        factor = -factor
                        X = [ones(nobs) factor]
                    end
                end
            else
                β, σ2, ϕ = autocorrErrorLinearRegressionSampler(Y, X, errorlags)
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
        X = zeros(nobs, 1 + factorlags)
        X[:, 1] = ones(nobs)
        for j in 1:factorlags
            X[:, 1+j] = lag(factor, j, default = 0.0)
        end
        X = X[(factorlags+1):nobs, :]

        ## Draw ψ
        ψ = linearRegressionSamplerRestrictedVariance(factor[(factorlags+1):nobs], X, 1.0)

        ## Fill out HDFM objects 
        fcoefs = ψ

        ## Save new draw of ψ
        psave[dr, :] = ψ'

        ##################################
        ##################################
        # Draw factor  

        #draw factor
        #take drawing of World factor 
        sinvf1 = sigbig(ψ[2:end], factorlags, nobs)
        f = zeros(nobs, 1)
        H = sinvf1' * sinvf1

        for i = 1:nvar
            sinv1 = sigbig(vec(varlagcoefs[i, :]), errorlags, nobs)
            H = H + ((varcoefs[i, 2]^2 / varvars[i]) * sinv1' * sinv1)
            f = f + (varcoefs[i, 2] / varvars[i]) * sinv1' * sinv1 * (y[:, i])
        end

        Hinv = inv(H)
        f = Hinv * f
        factor = sim_MvNormal(vec(f), Hinv)

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