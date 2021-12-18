
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
function OWSingleFactorEstimator(data, priorsIN)

    # nvar = number of variables including the variable with missing date
    # capt = length of data of complete dataset
    y = data
    capt, nvar = size(y)

    ndraws = 1000          # number of monte carlo draws
    burnin = 50            # number of initial draws to discard, total draws is undrws+burnin

    nfact = priorsIN.K           # number of factors to estimate
    arlag = priorsIN.Plags       # autoregressive lags in the dynamic factors 
    arterms = priorsIN.Llags + 1   # number of AR lags to include in each observable equation
    Size = nvar                 # number of variables each factor loads on
    nreg = 1 + priorsIN.K       # number of regressors in each observable equation, constant plus K factors       
    m = nfact * arlag        # dimension of state vector

    ytemp = y
    y = ytemp - repeat(mean(ytemp, dims = 1), capt, 1)

    # set up some matricies for storage (optional)
    Xtsave = zeros(rows(y), (ndraws + burnin) * nfact)    # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(ndraws + burnin, nreg * nvar)          # observable equation regression coefficients
    ssave = zeros(ndraws + burnin, nvar)               # innovation variances
    psave = zeros(ndraws + burnin, nfact * arlag)        # factor autoregressive polynomials
    psave2 = zeros(ndraws + burnin, nvar * arterms)       # factor autoregressive polynomials
    counter = zeros(nfact, 1)                           # number of nonstationary draws
    metcount = zeros(nfact, 1)                           # number of accepted draws in AR proc


    ##### include a necessary procedure #####
    # set priors
    # priors for observable equations in factor model
    # priors for observable equations in factor model
    b0_ = ones(nreg, 1)             # prior mean of factor loading
    B0__ = 0.01 * I(nreg)           # prior precision of factor loading
    B0__[1, 1] = 1.0

    b0_A = zeros(arlag, 1)              # prior mean of lagged world factor
    B0__A = 0.001 * I(arlag)            # prior precision of lagged world factor in other factors

    r0_ = zeros(arterms, 1)         # prior mean of phi (idiosyncratic AR polynomial)
    phipri = 0.25
    R0__ = phipri * I(arterms)          # prior precision of phi

    r0_v = zeros(arlag, 1)              # prior mean of VAR coefficients (in F)
    phipri = 0.25
    R0__V = phipri * I(arlag)           # prior precision

    r0_v2 = zeros(arlag * 2, 1)         # prior mean of VAR coefficients (in F)
    phipri = 0.25
    R0__v2 = phipri * I(arlag * 2)        # prior precision

    v0_ = ones(1, 1) * ceil(capt * 0.05)    # inverted gamma parameters of for innovation variances
    d0_ = ones(1, 1) * 0.25^2               # v0 is an integer

    # prior for factor AR polynomial
    prem = [1; seqm(1, 0.85, arlag - 1)]
    phiprif = 1 / prem
    r0f_ = zeros(arlag, 1)               # prior mean of phi
    R0f__ = diagrv(I(arlag), phiprif)
    R0f__ = phipri * I(arlag)

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
    facts[:, 1] = mean(y, dims = 2) |> transpose

    ## Begin Monte Carlo Loop
    for dr = 1:ndraws+burnin

        nf = 1

        for i = 1:nvar

            # call arobs to draw observable coefficients
            xft = [ones(capt, 1) facts(:, 1)]

            b1, s21, phi1, facts = ar(y[:, i], xft, arterms, b0_, B0__, r0_, R0__, v0_, d0_, bold[i, :]', SigE[i], phimat0[:, i], i, nf, facts)
            bold[i, 1:nreg] = b1'
            phimat0[:, i] = phi1
            SigE[i] = s21
            bsave[dr, ((i-1)*nreg)+1:i*nreg] = b1'
            ssave[dr, i] = s21
            psave2[dr, ((i-1)*arterms)+1:i*arterms] = phi1'

        end #end of loop for drawing the coefficients for each observable equation

        # draw factor AR coefficients
        i = 1
        phi = arfac(facts[:, i], arlag, r0f_, R0f__, phi[:, i], i, sigU[1, 1])
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
        fact1 = f + chol(Hinv)' * randn(capt, 1)

        for i = 1:nfact
            Xtsave[:, ((dr-1)*nfact)+i] = fact1
            facts[:, 1] = fact1
        end

    end

    # Save results 
    Xtsave  = Xtsave[:, (burnin*nfact)+1:(burnin+ndraws)*nfact]
    bsave   = bsave[burnin+1:burnin+ndraws, :]
    ssave   = ssave[burnin+1:burnin+ndraws, :]
    psave   = psave[burnin+1:burnin+ndraws, :]
    psave2  = psave2[burnin+1:burnin+ndraws, :]

    Xtsort          = sort(Xtsave, dims = 2)
    results.Xt05    = Xtsort[:, trunc(Int, round(0.05 * ndraws))]
    results.Xt16    = Xtsort[:, trunc(Int, round(0.16 * ndraws))]
    results.Xt50    = Xtsort[:, trunc(Int, round(0.50 * ndraws))]
    results.Xt84    = Xtsort[:, trunc(Int, round(0.84 * ndraws))]
    results.Xt95    = Xtsort[:, trunc(Int, round(0.95 * ndraws))]

    results.Xt  = Xtsave
    results.B   = bsave
    results.S   = ssave
    results.P   = psave
    results.P2  = psave2

    meanz.F     = mean(Xtsave, dims = 2)
    meanz.B     = mean(bsave, dims = 1)
    meanz.S     = mean(ssave, dims = 1)
    meanz.P     = mean(psave, dims = 1)
    meanz.P2    = mean(psave2, dims = 1)

    return meanz, results
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
function otrokWhitemanFactorSampler(data_y, data_z, H, A, F, μ, R, Q, Z)

    β_realized = Any[]

    # Return sampled factor series 
    # fot t = 1,...,T 
    return β_realized
end 