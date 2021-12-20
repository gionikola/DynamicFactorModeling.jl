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
    priorsSET2(K, Plags, Llags, KL, COUNTRY)

Description:
Model priors for the Otrok-Whiteman estimator. 

Inputs:
- K = Number of factors.
- Plags = Number of lags in the factor equation. 
- Llags = Number of AR lags in the observation equation. 
- KL = Number of factors × number of AR lags in the obs equation. 
- COUNTRY = Number of countries in a region 
"""
@with_kw mutable struct priorsSET2
    K::Int64
    Plags::Int64
    Llags::Int64
    KL = K*Llags 
    COUNTRY::Int64 
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
function ar_LJ(y, x, p, b0_, B0__, r0_, R0__, v0_, d0_, b0, s20, phi0, xvar, nfc, facts, capt, nreg, Size)

    local xst, yst, b1, phi1

    n = capt
    signmax1 = 1
    signbeta1 = 1
    signmax2 = 1
    signbeta2 = 0

    while signbeta1 + signbeta2 >= 1

        # generation of phi1 
        yp = y[1:p, 1]          # the first p observations 
        xp = x[1:p, :]

        e = y - x * vec(b0)
        e1 = e[(p+1):n, 1]
        ecap = zeros(n, p)

        for j = 1:p
            ecap[:, j] = lag(e, j, default = 0.0)
        end

        ecap = ecap[(p+1):n, :]

        V = invpd(R0__ .+ inv(s20) * (transp_dbl(ecap) * ecap))
        phihat = V * (R0__ * r0_ + inv(s20) * (transp_dbl(ecap) * e1))

        #phi1 = phihat + transp_dbl(cholesky(V)) * randn(p, 1)       # candidate draw 
        phi1 = rand(MvNormal(vec(phihat), PSDMat(Matrix(V))))

        coef = [-reverse(vec(phi1), dims = 1); 1]                      # check stationarity 
        root = roots(Polynomial(coef))
        rootmod = abs.(root)
        accept = min(rootmod...) >= 1.001             # all the roots bigger than 1 

        if accept == 0
            phi1 = phi0
        else
            #=
            ##### THE FOLLOWING CODE HAS POSITIVE-DEFINITENESS ISSUES 
            ##### IN SIGMA1 
            sigma1 =  Hermitian(sigmat(vec(phi1), p))               # numerator of acceptance prob 
            sigroot = cholesky(sigma1)
            p1 = inv(sigroot)'
            ypst = p1 * yp
            xpst = p1 * xp
            d = det(p1' * p1)
            psi1 = (d^(1 / 2)) * exp(-0.5 * (ypst - xpst * b0)' * (ypst - xpst * b0) / s20)
            =#
            sigma1 = Hermitian(sigmat(vec(phi1), p))               # numerator of acceptance prob 
            d = Complex(det(sigma1))
            psi1 = (d^(-0.5)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0)*invpd(sigma1)*(yp-xp*b0))[1])

            #=
            sigma1 = Hermitian(sigmat(phi0, p))
            sigroot = cholesky(sigma1)
            p1 = inv(sigroot)'
            ypst = p1 * yp
            xpst = p1 * xp
            d = det(p1' * p1)
            psi0 = (d^(1 / 2)) * exp(-0.5 * (ypst - xpst * b0)' * (ypst - xpst * b0) / s20)
            =#
            sigma1 = Hermitian(sigmat(vec(phi0), p))               # numerator of acceptance prob 
            d = Complex(det(sigma1))
            psi0 = (d^(1 / 2)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0)*invpd(sigma1)*(yp-xp*b0))[1])

            if psi0 == 0
                accept = 1
            else
                u = rand(1)[1]
                accept = u <= real(psi1) / real(psi0)
            end
            phi1 = real(phi1) * accept + real(phi0) * (1 - accept)
        end

        # generation of beta 
        sigma = Hermitian(sigmat(phi1, p))              # sigma = sigroot' * sigroot 
        sigroot = cholesky(sigma)                       # signber2v = p1' * p1 
        p1 = transp_dbl(inv(sigroot))
        ypst = p1 * yp
        xpst = p1 * xp
        yst = [ypst; gendiff(y, phi1)]
        xst = [xpst; gendiff(x, phi1)]

        V = invpd(B0__ + s20^(-1) * xst' * xst)
        bhat = V * (B0__ * vec(b0) + s20^(-1) * xst' * yst)
        #b1 = bhat + cholesky(V)' * randn(nreg, 1)          # the draw of beta 
        b1 = rand(MvNormal(vec(bhat), PSDMat(Matrix(V))))

        signbeta1 = (b1[2, 1] <= 0.0) * (xvar == 1)
        signmax1 = signmax1 + (1 * signbeta1)
        signbeta2 = (b1[3, 1] <= 0.0) * ( ((xvar-1)/Size) == floor((xvar-1)/Size))
        signmax2 = signmax2 + (1 * signbeta2) 

        if signmax1 >= 100
            facts[:, 1] = (-1) * facts[:, 1]
            x[:, 2] = facts[:, 1]
            signmax1 = 1
        end
        if signmax2 >= 100
            facts[:, nfc] = (-1)*facts[:, nfc]
            x[:,3] = facts[:, nfc]
            signmax2 = 1 
        end 
    end

    # generation of s2 
    nn = n + v0_
    d = d0_ + transp_dbl(yst - xst * b1) * (yst - xst * b1)
    c = rnchisq(nn)
    t2 = c ./ d
    s21 = 1 ./ t2
    b0 = b1
    s20 = s21[1, 1]

    return b0, s20, phi1, facts
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
function OWTwoFactorEstimator(data, prior_dim)

    y = data                    # save data in new matrix 
    capt, nvar = size(y)        # nvar = # of variables; capt = # of time periods in complete sample 

    ndraws = 1000               # # of Monte Carlo draws 
    burnin = 50                 # # of initial draws to discard; total draws is ndraws + burnin 

    nfact = prior_dim.K           # number of factors to estimate 
    arlag = prior_dim.Plags       # autoregressive lags in the dynamic factors 
    arterms = prior_dim.Llags + 1
    Size = 3
    nreg = 3
    m = nfact * arlag

    # Load data 
    ytemp = y

    y = ytemp - repeat(mean(ytemp, dims = 1), capt, 1)

    # Set up some matrices for storage
    Xtsave = zeros(size(y)[1], nfact, (ndraws + burnin))        # just keep draw of factor, not all states (others are trivial)
    bsave = zeros(ndraws + burnin, nreg * nvar)                   # observable equation regression coefficients
    ssave = zeros(ndraws + burnin, nvar)                        # innovation variances
    psave = zeros(ndraws + burnin, nfact * arlag)                 # factor autoregressive polynomials
    psave2 = zeros(ndraws + burnin, nvar * arterms)               # factor autoregressive polynomials

    ##############################################
    ##############################################
    ##### include a necessary procedure #####
    ##### set priors
    ##### priors for observable equations in factor model

    b0_ = ones(nreg, 1)                 # prior mean of factor loading
    B0__ = 0.01 * ident(nreg)             # prior precision of factor loading
    B0__[1, 1] = 1.0

    b0_A = zeros(arlag, 1)               # prior mean of lagged world factor
    B0__A = 0.001 * ident(arlag)           # prior precision of lagged world factor in other factors

    r0_ = zeros(arterms, 1)             # prior mean of phi (idiosyncratic AR polynomial)
    phipri = 0.25
    R0__ = phipri * ident(arterms)        # prior precision of phi

    r0_v = zeros(arlag, 1)               # prior mean of VAR coefficients (in F)
    phipri = 0.25
    R0__V = phipri * ident(arlag)          # prior precision

    r0_v2 = zeros(arlag * 2, 1)             # prior mean of VAR coefficients (in F)
    phipri = 0.25
    R0__v2 = phipri * ident(arlag * 2)        # prior precision

    v0_ = trunc(Int, ceil(capt * 0.05))  # inverted gamma parameters of for innovation variances
    d0_ = (0.25)^2                     # v0 is an integer

    # prior for factor AR polynomial
    prem = 1
    for i = 1:(arlag-1)
        prem = [prem; (1 / 2)^i]
    end
    phiprif = 1 ./ prem
    r0f_ = zeros(arlag)                        # prior mean of phi
    R0f__ = diagrv(ident(arlag), phiprif)

    # Normalize innovation variance for the factor, vector since diagonal,
    # variance set to 1
    sigU = [1; zeros(arlag - 1)]

    for i = 2:nfact
        sigU = [sigU; 1; zeros(arlag - 1)]
    end

    ##############################################
    ##############################################

    # Initialize and set up factor model
    SigE = ones(nvar, 1) * 0.0001
    phi = zeros(arlag, nfact)
    bold = zeros(nvar, nreg)                 # starting value for regression coefficients
    phimat0 = zeros(arterms, nvar)              # observable equation AR coefficients
    capt = size(y)[1]                       # if including idiosyncratic dynamics condition on lags for quasi-differencing

    facts = rand(capt, nfact)                 # random starting factor
    facts[:, 1] = transp_dbl(mean(y, dims = 2))

    # Begin Monte Carlo Loop
    for dr = 1:(ndraws+burnin)

        println(dr)
        
        nf = 2

        for i = 1:nvar

            # call arobs to draw observable coefficients
            xft = [ones(capt, 1) facts[:, 1] facts[:, nf]]

            b1, s21, phi1, facts = ar_LJ(y[:, i], xft, arterms, b0_, B0__, r0_, R0__, v0_, d0_, transp_dbl(bold[i, :]), SigE[i], phimat0[:, i], i, nf, facts, capt, nreg, Size)

            bold[i, 1:nreg] = transp_dbl(b1)
            phimat0[:, i] = phi1
            SigE[i] = s21
            bsave[dr, (((i-1)*nreg)+1):(i*nreg)] = transp_dbl(b1)
            ssave[dr, i] = s21
            psave2[dr, (((i-1)*arterms)+1):(i*arterms)] = transp_dbl(phi1)

            if (i / Size) == floor(i / Size)
                nf = nf + 1
            end

        end # end of loop for drawing the coefficients for each observable equation

        # draw factor AR coeffcicients
        j = 1
        for i = 1:nfact
            phi[:, i] = arfac(facts[:, i], arlag, r0f_, R0f__, phi[:, i], sigU[j, 1], capt)
            psave[dr, ((i-1)*arlag+1):((i-1)*arlag+arlag)] = transp_dbl(phi[:, i])
            j = j + arlag
        end

        # draw factors
        # take drawing of World factor 

        sinvf1 = sigbig(phi[:, 1], arlag, capt)
        f = zeros(capt, 1)
        H = ((1 / sigU[1]) * (transp_dbl(sinvf1) * sinvf1))
        nfC = 2
        for i = 1:nvar

            yW = y[:, i] - ones(capt, 1) * bold[i, 1] - facts[:, nfC] * transp_dbl(bold[i, 3])
            sinv1 = sigbig(phimat0[:, i], arterms, capt)
            H = H + ((bold[i, 2]^2 / (SigE[i])) * (transp_dbl(sinv1) * sinv1))
            f = f + (bold[i, 2] / SigE[i]) * (transp_dbl(sinv1) * sinv1) * yW

            if (i / Size) == floor(i / Size)
                nfC = nfC + 1
            end
        end

        Hinv = invpd(H)
        f = Hinv * f
        #fact1 = f + transp_dbl(cholesky(Hinv))*randn(capt,1);
        fact1 = rand(MvNormal(f, PSDMat(Hinv)))

        Xtsave[:, 1, dr] = fact1
        facts[:, 1] = fact1

        # take drawing of Country factors
        j = 1 + arlag
        for c = 1:(prior_dim.COUNTRY)

            yC = y[:, 1+(c-1)*Size:c*Size] - ones(capt, 1) * transp_dbl(bold[(1+(c-1)*Size):(c*Size), 1]) - facts[:, 1] * transp_dbl(bold[(1+(c-1)*Size):(c*Size), 2])

            phiC = phi[:, 1+c]
            sinvf1 = sigbig(phiC, arlag, capt)
            f = zeros(capt, 1)
            H = ((1 / sigU[j]) * (transp_dbl(sinvf1) * sinvf1))

            for i = 1:Size

                sinv1 = sigbig(phimat0[:, (1+(c-1)*Size+i-1)], arterms, capt)
                H = H + ((bold[1+(c-1)*Size+i-1, 3]^2 / (SigE[1+(c-1)*Size+i-1])) * (transp_dbl(sinv1) * sinv1))
                f = f + (bold[1+(c-1)*Size+i-1, 3] / SigE[1+(c-1)*Size+i-1]) * (transp_dbl(sinv1) * sinv1) * (yC[:, i])

            end
            Hinv = invpd(H)
            f = Hinv * f
            #fact2 = f+transp_dbl(cholesky(Hinv))*randn(capt,1);
            fact2 = rand(MvNormal(f, PSDMat(Hinv)))

            Xtsave[:, 1+c, dr] = fact2
            facts[:, 1+c] = fact2

            j = j + arlag
        end

        println(dr)
    end

    Xtsave  = Xtsave[:, :, (burnin+1):(burnin+ndraws)]
    bsave   = bsave[(burnin+1):(burnin+ndraws), :]
    ssave   = ssave[(burnin+1):(burnin+ndraws), :]
    psave   = psave[(burnin+1):(burnin+ndraws), :]
    psave2  = psave2[(burnin+1):(burnin+ndraws), :]

    F   = mean(Xtsave, dims = 3)
    B   = mean(bsave, dims = 1)
    S   = mean(ssave, dims = 1)
    P   = mean(psave, dims = 1)
    P2  = mean(psave2, dims = 1)

    return F, B, S, P, P2
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