include("ow_tools.jl")
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
    OWSingleFactorEstimator(data, dfm)

Description:
Estimate a single-factor DFM using the Otrok-Whiteman approach.
Both the latent factor and hyperparameters are estimated using the Bayesian approach outlined in Otrok and Whiteman (1998).  

Inputs:
- data = Matrix with each column being a data series. 
- dfm = Model structure specification. 

Outputs:
- results = HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters.
"""
function OW1LevelEstimator(data::Array{Float64,2}, dfm::DFMStruct)

    # nvar = number of variables including the variable with missing date
    # capt = length of data of complete dataset
    y = data
    capt, nvar = size(y)

    ndraws = dfm.ndraws         # number of monte carlo draws
    burnin = dfm.burnin         # number of initial draws to discard, total draws is undrws+burnin

    nfact = 1                   # number of factors to estimate
    arlag = dfm.factorlags      # autoregressive lags in the dynamic factors 
    arterms = dfm.errorlags     # number of AR lags to include in each observable equation
    Size = nvar                 # number of variables each factor loads on
    nreg = 2                    # number of regressors in each observable equation, constant plus K factors       

    ytemp = y
    y = ytemp - repeat(mean(ytemp, dims = 1), capt, 1)

    # set up some matricies for storage (optional)
    Xtsave = zeros(size(y)[1], ((ndraws + burnin) * nfact))    # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(ndraws + burnin, nreg * nvar)          # observable equation regression coefficients
    ssave = zeros(ndraws + burnin, nvar)               # innovation variances
    psave = zeros(ndraws + burnin, nfact * arlag)        # factor autoregressive polynomials
    psave2 = zeros(ndraws + burnin, nvar * arterms)       # factor autoregressive polynomials


    ##### include a necessary procedure #####
    # set priors
    # priors for observable equations in factor model
    # priors for observable equations in factor model
    b0_ = ones(nreg, 1)             # prior mean of factor loading
    B0__ = 0.01 * I(nreg)           # prior precision of factor loading
    B0__[1, 1] = 1.0

    r0_ = zeros(arterms, 1)         # prior mean of phi (idiosyncratic AR polynomial)
    phipri = 0.25
    R0__ = phipri * I(arterms)          # prior precision of phi

    v0_ = ceil(capt * 0.05)    # inverted gamma parameters of for innovation variances
    d0_ = ones(1, 1) * 0.25^2               # v0 is an integer

    # prior for factor AR polynomial
    prem = [1; seqm(1, 0.85, arlag - 1)]
    phiprif = 1 ./ prem
    r0f_ = vec(zeros(arlag, 1))               # prior mean of phi
    R0f__ = diagrv(ident(arlag), phiprif)
    R0f__ = phipri * ident(arlag)

    # Normalize innovation variance for the factor, vector since diagonal,
    # variance set to 1
    sigU = [1; zeros(arlag - 1, 1)]

    for i = 2:nfact
        sigU = [sigU; 1; zeros(arlag - 1, 1)]
    end
    ############################################

    # Initialize and set up factor model
    SigE = ones(nvar, 1) * 0.0001
    phi = zeros(arlag, nfact)
    bold = zeros(nvar, nreg)         # starting value for regression coefficients
    phimat0 = zeros(arterms, nvar)      # observable equation AR coefficients

    facts = randn(capt, nfact)     # random starting factor
    facts[:, 1] = mean(y, dims = 2) |> transp_dbl

    ## Begin Monte Carlo Loop
    for dr = 1:ndraws+burnin

        nf = 1

        for i = 1:nvar

            # call arobs to draw observable coefficients
            xft = [ones(capt, 1) facts[:, 1]]

            b1, s21, phi1, facts = ar(y[:, i], xft, arterms, b0_, B0__, r0_, R0__, v0_, d0_, bold[i, :], SigE[i], phimat0[:, i], i, nf, facts, capt, nreg, Size)
            bold[i, 1:nreg] = b1'
            phimat0[:, i] = phi1
            SigE[i] = s21
            bsave[dr, ((i-1)*nreg)+1:i*nreg] = b1'
            ssave[dr, i] = s21
            psave2[dr, ((i-1)*arterms)+1:i*arterms] = phi1'

        end #end of loop for drawing the coefficients for each observable equation

        # draw factor AR coefficients
        i = 1
        phi = arfac(facts[:, i], arlag, r0f_, R0f__, phi[:, i], sigU[1, 1], capt)
        psave[dr, (i-1)*arlag+1:(i-1)*arlag+arlag] = phi

        #draw factor
        #take drawing of World factor 
        sinvf1 = sigbig(phi, arlag, capt)
        f = zeros(capt, 1)
        H = ((1 / sigU[1]) * sinvf1' * sinvf1)

        for i = 1:nvar
            sinv1 = sigbig(phimat0[:, i], arterms, capt)
            H = H + ((bold[i, 2]^2 / (SigE[i])) * sinv1' * sinv1)
            f = f + (bold[i, 2] / SigE[i]) * sinv1' * sinv1 * (y[:, i])
        end

        Hinv = invpd(H)
        f = Hinv * f
        fact1 = sim_MvNormal(vec(f), Hinv)

        for i = 1:nfact
            Xtsave[:, (((dr-1)*nfact)+i)] = vec(fact1)
            facts[:, 1] = fact1
        end

        println(dr)
    end

    # Save results 
    Xtsave = Xtsave[:, (burnin*nfact)+1:(burnin+ndraws)*nfact]
    bsave = bsave[burnin+1:burnin+ndraws, :]
    ssave = ssave[burnin+1:burnin+ndraws, :]
    psave = psave[burnin+1:burnin+ndraws, :]
    psave2 = psave2[burnin+1:burnin+ndraws, :]

    F = mean(Xtsave, dims = 2)
    B = mean(bsave, dims = 1)
    S = mean(ssave, dims = 1)
    P = mean(psave, dims = 1)
    P2 = mean(psave2, dims = 1)

    means = DFMMeans(F, B, S, P, P2)

    results = DFMResults(Xtsave, bsave, ssave, psave, psave2, means)

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