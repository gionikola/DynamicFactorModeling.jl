function OW2LevelCoefEstimator(data, prior_hdfm, facts)

    # Unpack two-level HDFM parameters 
    @unpack nlevels, nvar, nfactors, fassign, flags, varlags = prior_hdfm

    # Save data & its size 
    y = data                    # save data in new matrix 
    capt, nvars = size(y)       # nvar = # of variables; capt = # of time periods in complete sample 

    # Specify simulation length 
    ndraws = 1000               # # of Monte Carlo draws 
    burnin = 50                 # # of initial draws to discard; total draws is ndraws + burnin 

    # Store factor and parameter counts 
    nfact = sum(nfactors)       # # of factors 
    arlag = max(flags...)       # autoregressive lags in the dynamic factors 
    arterms = max(varlags...)   # number of AR lags to include in each observable equation
    nreg = 1 + nlevels          # # of regressors in each obs. eq. (intercept + factors)

    # Count number of variables each 2nd-level factor loads on
    # using a vector called `fnvars`
    # where the i-th entry represents the i-th 2nd-level factors 
    fnvars = zeros(Int, nfact - 1)
    for i in 1:(nfact-1)
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
    for i in 1:(nfact-1) # iterate over level-2 factors 
        push!(varassign, Any[])
        for j in 1:nvar # iterate over variables 
            if fassign[j, 2] == i
                push!(varassign[i], j)
            end
        end
    end

    # Load data 
    ytemp = y

    # Express each variable in the dataset 
    # in deviation-from-mean form 
    # (de-mean each variable in the dataset)
    y = ytemp - repeat(mean(ytemp, dims = 1), capt, 1)

    # Set up some matrices for storage
    Xtsave = zeros(capt, nfact, (ndraws + burnin))          # Keep draw of factor, not all states (others are trivial)
    bsave = zeros(ndraws + burnin, nreg * nvar)             # Observable equation regression coefficients
    ssave = zeros(ndraws + burnin, nvar)                    # Innovation variances
    psave = zeros(ndraws + burnin, nfact * arlag)           # Factor autoregressive polynomials
    psave2 = zeros(ndraws + burnin, nvar * arterms)         # Idiosyncratic disturbance autoregressive polynomials

    ##############################################
    ##############################################
    ##### include a necessary procedure #####
    ##### set priors for observable equations in factor model

    # Specify prior mean of factor loading
    # and precision of factor loading 
    b0_ = ones(nreg, 1)                 # prior mean of factor loading
    B0__ = 0.01 * ident(nreg)           # prior precision of factor loading
    B0__[1, 1] = 1.0

    # Specify prior mean of idiosyncratic 
    # disturbance AR lag coefficients
    # and precision of AR lag coefficients 
    r0_ = zeros(arterms, 1)             # prior mean of phi (idiosyncratic AR polynomial)
    phipri = 0.25
    R0__ = phipri * ident(arterms)      # prior precision of phi

    # Specify prior parameters of 
    # innovation variances 
    v0_ = trunc(Int, ceil(capt * 0.05)) # inverted gamma parameters of for innovation variances
    d0_ = (0.25)^2                      # v0 is an integer

    # Specify prior for factor AR lag coefficients 
    # and precision of factor AR lag coefficients 
    prem = 1
    for i = 1:(arlag-1)
        prem = [prem; (1 / 2)^i]
    end
    phiprif = 1 ./ prem
    r0f_ = zeros(arlag)                        # prior mean of phi
    R0f__ = diagrv(ident(arlag), phiprif)
    #R0f__ = phipri * ident(arlag) 

    # Normalize innovation variance for the factor vector 
    # since diagonal variance is set to 1
    sigU = [1; zeros(arlag - 1)]
    for i = 2:nfact
        sigU = [sigU; 1; zeros(arlag - 1)]
    end

    ##############################################
    ##############################################
    ##############################################
    ##############################################

    # Initialize hyperparameters 
    SigE = ones(nvar, 1) * 0.0001           # Idiosyncratic error precision vector 
    phi = zeros(arlag, nfact)               # Factor AR companion matrix 
    bold = zeros(nvar, nreg)                # Obs. eq. regression starting coefficient matrix 
    phimat0 = zeros(arterms, nvar)          # Idiosyncratic error AR companion matrix 

    # Begin Monte Carlo Loop
    # (start iteratively drawing hyperparameters and factors)
    for dr = 1:(ndraws+burnin)

        println(dr)

        ############################################
        ## Draw observation equation hyperparameters 
        ############################################
        for i = 1:nvar # Iterate over all obs. variables 
        
            # Save the index of the factor assigned 
            # to observable variable i 
            nf = fassign[i, 2]
        
            # Create matrix containing all regressors 
            # corresponding to variable i including:
            # (1) an intercept, (2) global factor, (3) level-2 factor 
            xft = [ones(capt, 1) facts[:, 1] facts[:, 1+nf]]
        
            # Update variable i observation equation 
            # hyperparameters, and update corresponding 
            # factor orientation (if appropriate)
            #b1, s21, phi1, facts = ar_LJ(y[:, i], xft, arterms, b0_, B0__, r0_, R0__, v0_, d0_, transp_dbl(bold[i, :]), SigE[i], phimat0[:, i], i, nf, facts, capt, nreg, fnvars[fassign[i, 2]], varassign)
            b1, s21, phi1, facts = ar2(y[:, i], xft, arterms, b0_, B0__, r0_, R0__, v0_, d0_, vec(bold[i, :]), phimat0[:, i], SigE[i], i, nf, facts, capt, varassign)
            bold[i, :] = b1                                     # Update obs. regression coefficients
            phimat0[:, i] = phi1                                # Update idiosyncratic error AR coefficients 
            SigE[i] = s21                                       # Update innovation variance parameter 
            bsave[dr, (((i-1)*nreg)+1):(i*nreg)] = b1           # Save current obs. regression coefficient draw 
            ssave[dr, i] = s21                                  # Save current innovation variance parameter draw
            psave2[dr, (((i-1)*arterms)+1):(i*arterms)] = phi1  # Save current idiosyncratic error AR coefficient draw  
        
        end

        ############################################
        ## Draw factor autoregression coefficients  
        ############################################
        for i = 1:nfact # Iterate over all factors
            j = 1 + (i - 1) * arlag
            phi[:, i] = arfac(facts[:, i], arlag, r0f_, R0f__, phi[:, i], sigU[j, 1], capt)
            psave[dr, ((i-1)*arlag+1):((i-1)*arlag+arlag)] = transp_dbl(phi[:, i])
        end

        println(dr)
    end

    ##############################################
    ##############################################
    ## Save Monte Carlo samples 
    Xtsave = Xtsave[:, :, (burnin+1):(burnin+ndraws)]
    bsave = bsave[(burnin+1):(burnin+ndraws), :]
    ssave = ssave[(burnin+1):(burnin+ndraws), :]
    psave = psave[(burnin+1):(burnin+ndraws), :]
    psave2 = psave2[(burnin+1):(burnin+ndraws), :]

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

    return results
end 