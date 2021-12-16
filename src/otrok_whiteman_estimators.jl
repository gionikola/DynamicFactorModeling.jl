function otrokWhitemanFactorSampler(data_y, data_z, H, A, F, μ, R, Q, Z)

    β_realized = Any[]

    # Return sampled factor series 
    # fot t = 1,...,T 
    return β_realized
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

function fac21OW_LJ(data, prior_dim)

    y = data
    capt = size(data)[1]            # length of data of complete dataset 
    nvar = size(data)[2]            # number of variables including the variable with missing date 

    ndraws = 1000                   # number of monte carlo draws 
    burnin = 50                     # number of initial discard, total draws is ndraws+burnin 

    nfact = prior_dim.K             # number of fators to estimate 
    arlag = prior_dim.Plags         # autoregressive lags in the dynamic factors 
    arterms = prior_dim.Llags + 1   # number of AR lags to include in each observable equation 
    Size = 3                        # number of variables each factor loans on 
    nreg = 3                        # number of regressors in each observable equation, constant plus K factors 
    m = nfact * arlag               # dimension of state vector 

    ## load data 
    ytemp = y
    y = ytemp - repeat(mean(ytemp), capt, 1) # repeat(mean(ytemp), capt, 1) stacks mean(ytemp) matrices capt times 

    ## set up some matrices for storage 
    Xtsave = zeros(size(y)[1], nfact, ndraws + burnin) # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(ndraws + burnin, nreg * nvar)      # observable equation regression coefficients
    ssave = zeros(ndraws + burnin, nvar)             # innovation variances
    psave = zeros(ndraws + burnin, nfact * arlag)    # factor autoregressive polynomials
    psave2 = zeros(ndraws + burnin, nvar * arterms)   # factor autoregressive polynomials

    counter = zeros(nfact, 1)                          # number of nonstationary draws
    metcount = zeros(nfact, 1)                          # number of accepted draws in AR proc

    ##################################################
    ##################################################
    # Include a necessary procedure
    # (in the original MATLAB code this calls on priors21.m, a separate script)

    # set priors
    # priors for observable equations in factor model
    b0_ = ones(nreg, 1) # prior mean of factor loading
    B0__ = 0.01 * I(nreg) # prior precision of factor loading
    B0__[1, 1] = 1.0

    b0_A = zeros(arlag, 1) # prior mean of lagged world factor
    B0__A = 0.001*I(arlag) # prior precision of lagged world factor in other factors 

    r0_ = zeros(arterms, 1) # prior mean of phi (idiosyncratic AR polynomial)
    phipri = 0.25 
    R0__ = phipri*I(arterms) # prior precision of phi 

    r0_v = zeros(arlag,1)  # prior mean of VAR coefficients (in F) 
    phipri = 0.25 
    R0__V = phipri*eye(arlag) # prior precision 

    r0_v2 = zeros(arlag*2, 1) # prior mean of VAR coefficients (in F)
    phipri = 0.25 
    R0_v2 = phipri*I(arlag*2) # prior precision 

    v0_ = ones(1,1) * ceil(capt*0.05) # inverted gamma parameters of for innovation variances 
    d0_ = ones(1,1)*0.25^2            # v0 is an integer 

    # prior for factor AR polynomial 
    prem = [1.0]
    for i in 1:(arlag-1) 
        prem = [prem; (0.5)^i]
    end 
    phiprif = 1 ./ prem 
    r0f_ = zeros(arlag, 1) # prior mean of phi 
    R0f__ = diagm(phiprif) # diagonalize `phiprif` 

    # Normalize innovation variance for the factor 
    # vector since diagonal 
    # variance set to 1 
    sigU = [1.0; zeros(arlag-1,1)]

    for i in 2:nfact 
        sigU = [sigU; 1; zeros(arlag-1,1)]
    end 
    ##################################################
    ##################################################

    # Initialize and set up factor model 
    sigE = ones(nvar, 1) .* 0.0001 
    phi = zeros(arlag, nfact)
    bold = zeros(nvar, nreg)
    phimat0 = zeros(arterms, nvar) 
    capt = rows(y)

    facts = rand(capt, nfact) 
    facts[:,1] = mean(y, dims = 2) |> transpose |> copy

    # Begin Monte Carlo loop 
    for dr in 1:(ndraws + burnin)

        

    end 

    return meanz, results
end 