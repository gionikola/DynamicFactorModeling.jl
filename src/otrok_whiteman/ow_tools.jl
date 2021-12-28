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
    ident(p)

Description:
Create dense identity matrix of dimension p × p.

Inputs:
- p = number of columns and rows of identity matrix.

Outputs:
- ident_mat = p × p dense identity matrix with Float64.
"""
function ident(p)
    ident_mat = 1.0 * Matrix(I, p, p)
    return ident_mat
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
    transp_dbl(A)

Description:
Transpose matrix, so that the output type is Array{Float64,2}. 

Inputs:
- A = matrix.

Outputs:
- transp_A = dense transposed matrix A. 
"""
function transp_dbl(A)
    transp_A = 1.0 * Matrix(A')
    return transp_A
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
    transp_int(A)

Description:
Transpose matrix, so that the output type is Array{Int64,2}. 

Inputs:
- A = matrix.

Outputs:
- transp_A = dense transposed matrix A. 
"""
function transp_int(A)
    transp_A = 1 * Matrix(A')
    return transp_A
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
    seqa(a, b, c)

Description:
Create a sequence with `c` # of inputs, starting at `a` with `b` increments.

Inputs:
- a = starting value of the sequence.
- b = increment size.
- c = number of increments. 

Outputs:
- seq = sequence with `c` # of inputs, starting at `a` with `b` increments.

"""
function seqa(a, b, c)

    seq = zeros(c)

    for i = 1:c
        seq[i] = a + b * (i - 1)
    end

    return seq
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
    sigbig(phi, p, capt)

Description:
Create T×T matrix S^(-1) (pg. 1003 of Otrok and Whiteman (1998)). 

Inputs: 
- phi   = Lag coefficients associated with idiosyncratic error autoregression.
- p     = Number of lags.
- capt  = total number of time periods in the sample.

Outputs:
- Si    = T×T matrix S^(-1). 
"""
function sigbig(phi, p, capt)

    rotate = trunc.(Int, seqa(0, 1, capt - p))
    Siinv_upper =  Hermitian(sigmat(phi, p))
    Siinv_upper = [inv(cholesky(Siinv_upper).U)' zeros(p, capt - p)]
    Siinv_lower = [kron((-reverse(phi, dims = 1)'), ones(capt - p, 1)) ones(capt - p, 1) zeros(capt - p, capt - p - 1)]

    for s = 1:(capt-p)
        Siinv_lower[s, :] = circshift(Siinv_lower, (0, rotate[s]))[s, :]
    end

    Si = [Siinv_upper; Siinv_lower]

    return Si
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
    rnchisq(m)

Description:
Draw sample from a chi-square distribution.

Inputs:
- m = degrees of freedom. 

Outputs:
- g = draw from a chi-square distribution with m degrees of freedom. 
"""
function rnchisq(m)

    g = rand(Chisq(m))

    return g
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
    sigmat(phi, p)

Description:
Yield the stationary covariance matrix of the first p errors (pg 1001 of Otrok and Whiteman (1998)).

Inputs:
- phi  = Lag coefficients associated with idiosyncratic error autoregression.
- p    = Number of lags.

Outputs:
- Stationary covariance matrix of the first p errors.  

"""
function sigmat(phi, p)

    r2 = p^2
    i = [ident(p - 1) zeros(p - 1, 1)]
    Pcap = [phi'; i]
    pp = ident(r2) - kron(Pcap, Pcap)
    e1 = [1; zeros(p - 1, 1)]
    sig = inv(pp) * vec(e1 * e1')
    sigma = reshape(sig', p, p)
    sigma = Matrix(sigma)

    return sigma
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
    gendiff(z, phi)

Description:
Difference a series `z` using the lag coefficients provided in `phi`. 

Inputs:
- z = time series. 
- phi = lag coefficient vector. 

Outputs:
- zgdiff = quasi-differenced version of the inputted series `z`. 

"""
function gendiff(z, phi)

    p = size(phi)[1]
    ztrim = z[(p+1):end, :]
    zgdiff = ztrim

    for i = 1:p
        zgdiff = zgdiff - phi[i, 1] * z[(p-i+1):(end-i), :]
    end

    return zgdiff
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
#function arfac(y, p, r0_, R0__, phi0, xvar, sig2, capt)
function arfac(y, p, r0_, R0__, phi0, sig2, capt)

    # Generation of phi 
    yp = y[1:p, 1]          # the first p observations  
    e = y
    e1 = e[p+1:capt, 1]
    ecap = zeros(capt, p)

    for j = 1:p
        ecap[:, j] = lag(e, j, default = 0.0)
    end

    ecap = ecap[p+1:capt, :]

    V = invpd(R0__ + sig2^(-1) * ecap' * ecap)
    phihat = V * (R0__ * r0_ + sig2^(-1) * ecap' * e1)

    phi1 = sim_MvNormal(phihat, V)

    coef = [-reverse(phi1, dims = 1); 1]                # check stationarity 
    root = roots(Polynomial(reverse(coef)))                      # Find lag polynomial roots 

    rootmod = abs.(root)
    accept = min(rootmod...) >= 1.0001                     # all the roots bigger than 1 

    if accept == 0                                      # doesn't pass stationarity 
        phi1 = phi0
    else
        
        sigma1 = Hermitian(sigmat(vec(phi1), p))               # numerator of acceptance prob 
        d = det(sigma1)
        psi1 = (d^(-1/2)) * exp((-0.5 / sig2) * (transp_dbl(yp)*invpd(sigma1)*(yp))[1])
    
        sigma1 = Hermitian(sigmat(vec(phi0), p))               # numerator of acceptance prob 
        d = det(sigma1)
        psi0 = (d^(-1/2)) * exp((-0.5 / sig2) * (transp_dbl(yp)*invpd(sigma1)*(yp))[1])
        
        #=
        sigma1 = sigmat(phi1, p)       # numerator of acceptance prob
        sigma1 = Hermitian(sigma1)
        sigroot = cholesky(sigma1)
        p1 = transp_dbl(inv(sigroot))
        ypst = p1 * yp
        d = det(p1' * p1)
        psi1 = (d^(1 / 2)) * exp(-0.5 * (ypst)' * (ypst) / sig2)
    
        sigma1 = sigmat(phi0, p)       # numerator of acceptance prob
        sigma1 = Hermitian(sigma1) 
        sigroot = cholesky(sigma1)
        p1 = transp_dbl(inv(sigroot))
        ypst = p1 * yp
        d = det(p1' * p1)
        psi0 = (d^(1 / 2)) * exp(-0.5 * (ypst)' * (ypst) / sig2)
        =# 

        if psi0 == 0
            accept = 1
        else
            u = rand(1)
            accept = u[1] <= psi1 / psi0
        end
        phi1 = phi1 * accept + phi0 * (1 - accept)
    end

    return phi1
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
    seqm(a,b,c)

Description:
Prouce a sequence of values.

Inputs:
- a = initial value of sequence 
- b = increment scaling size 
- c = number of values in the sequence

Outputs:
- seq = geometric sequence 
"""
function seqm(a, b, c)

    seq = zeros(c, 1)

    seq[1] = a

    if c > 1

        seq[2] = a * b

        for i = 3:c
            seq[i] = seq[i-1] * b
        end
    end

    return seq
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
    invpd(X)

Description:
Invert matrix X using augmented eigenvalues if X is not positive-definite. 

Inputs:
- X = matrix of interest 

Output:
- X_inv = inverse of input matrix X 
"""
function invpd(X)

    X_inv = similar(X)

    if isposdef(X) == true 
        X_inv = inv(X)
    else
        n, m = size(X)
        U, dd, V = svd(X)
        xchk = U * Diagonal(dd) * V'
        dd = dd .+ 1000 * 2.2204e-16
        di = ones(n, 1) ./ dd
        X_inv = U * Diagonal(di) * V
    end

    return X_inv
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
    diagrv(X,V)

Description: 
Replace the diagonal of matrix X with vector V. 

Inputs: 
- X = matrix of interest 
- V = new diagonal of X 

Outputs:
- X_new = X with new diagonal V 
"""
function diagrv(X, V)

    X_new = X
    X_new[diagind(X_new)] = V

    X_new
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
    ar(y, x, p, b0_, B0__, r0_, R0__, v0_, d0_, b0, s20, phi0, xvar, nfc, facts, capt, nreg, Size)
    
Description:
Generate β_i , ϕ_i, and σ^2_i estimates based on the posterior distributions (eqs. 6-8) on pg. 1002 of Otrok and Whiteman (1998). 

Inputs:
- y = observed dependent variable / data.  
- x = measurement equation regressors (intercept + factor).
- p = number of state eq. lags. 
- b0_ = prior mean of factor loading (factor coefficient in measurement equation). 
- B0__ = prior variance of factor loading (factor coefficient in measurement equation). 
- r0_ = prior mean of phi (idiosyncratic AR polynomial). 
- R0__ = prior precision of phi (idiosyncratic AR polynomial). 
- v0_ = inverted gamma parameter of innovation variances. 
- d0_ = inverted gamma parameter of innovation variances. 
- b0 = old draw of factor loading. 
- s20 = old draw of innovation variances. 
- phi0 = old draw of phi (idiosyncratic AR polynomial). 
- xvar = measurement eq. regressor index. 
- nfc = number of factors. 
- facts = factor draw. 
- capt = total number of time periods / observations in the sample. 
- nreg = number of regressors in each observable equation, constant plus K factors   
- Size = number of variables each factor loads on

Outputs:
- b0 = β coefficient hyperparameter estimates. 
- s20 = σ^2 coefficient hyperparameter estimate. 
- phi1 = ϕ coefficient hyperparameter estimates. 
- facts = factor estimate. 
"""
function ar(y, x, p, b0_, B0__, r0_, R0__, v0_, d0_, b0, s20, phi0, xvar, nfc, facts, capt, nreg, Size)

    local xst, yst, b1, phi1

    n = capt
    signmax1 = 1
    signbeta1 = 1

    while signbeta1 >= 1

        # generation of phi1 
        yp = y[1:p, 1]          # the first p observations 
        xp = x[1:p, :]

        e = y - x * b0
        e1 = e[(p+1):n, 1]
        ecap = zeros(n, p)

        for j = 1:p
            ecap[:, j] = lag(e, j, default = 0.0)
        end

        ecap = ecap[(p+1):n, :]

        V = invpd(R0__ .+ inv(s20) * (transp_dbl(ecap) * ecap))
        phihat = V * (R0__ * r0_ + inv(s20) * (transp_dbl(ecap) * e1))

        phi1 = sim_MvNormal(vec(phihat), V)

        coef = [-reverse(vec(phi1), dims = 1); 1]                      # check stationarity 
        root = roots(Polynomial(reverse(coef)))
        rootmod = abs.(root)
        accept = min(rootmod...) >= 1.0001             # all the roots bigger than 1 

        if accept == 0
            phi1 = phi0
        else
            sigma1 = Hermitian(sigmat(vec(phi1), p))               # numerator of acceptance prob 
            d = det(sigma1)
            psi1 = (d^(0.5)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0)*invpd(sigma1)*(yp-xp*b0))[1])

            sigma1 = Hermitian(sigmat(vec(phi0), p))               # numerator of acceptance prob 
            d = det(sigma1)
            psi0 = (d^(1 / 2)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0)*invpd(sigma1)*(yp-xp*b0))[1])

            if psi0 == 0
                accept = 1
            else
                u = rand(Uniform(0,1))
                accept = u <= psi1 / psi0
            end
            phi1 = phi1 * accept + phi0 * (1 - accept)
        end

        # generation of beta 
        sigma = Hermitian(sigmat(phi1, p))             # sigma = sigroot' * sigroot 
        sigroot = cholesky(sigma)                 # signber2v = p1' * p1 
        p1 = transp_dbl(inv(sigroot))
        ypst = p1 * yp
        xpst = p1 * xp
        yst = [ypst; gendiff(y, phi1)]
        xst = [xpst; gendiff(x, phi1)]

        V = invpd(B0__ + s20^(-1) * xst' * xst)
        bhat = V * (B0__ * b0 + s20^(-1) * xst' * yst)
        b1 = sim_MvNormal(vec(bhat), Matrix(V))

        signbeta1 = (b1[2, 1] <= 0.0) * (xvar == 1)
        signmax1 = signmax1 + (1 * signbeta1)

        if signmax1 >= 100
            facts[:, 1] = (-1) * facts[:, 1]
            x[:, 2] = facts[:, 1]
            signmax1 = 1
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
function ar_LJ(y, x, p, b0_, B0__, r0_, R0__, v0_, d0_, b0, s20, phi0, xvar, nfc, facts, capt, nreg, Size, varassign)

    # Make sure relevant objects are accessible
    # outside of the upcoming for-loops 
    local xst, yst, b1, phi1

    # Save number of observations in the sample 
    n = capt

    # Initialize parameter signs 
    signmax1 = 1.0      # 
    signbeta1 = 1.0     # 
    signmax2 = 1.0      # 
    signbeta2 = 0.0     # 

    testind = 0

    while signbeta1 + signbeta2 >= 1.0 && testind < 10000

        testind = testind + 1

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

        phi1 = sim_MvNormal(vec(phihat), Matrix(V))

        coef = [-reverse(vec(phi1), dims = 1); 1]                      # check stationarity 
        root = roots(Polynomial(reverse(coef)))
        rootmod = abs.(root)
        accept = min(rootmod...) >= 1.0001             # all the roots bigger than 1 

        if accept == 0
            phi1 = phi0
        else
            
            #=
            sigma1 = Hermitian(sigmat(vec(phi1), p))               # numerator of acceptance prob 
            d = det(sigma1)
            psi1 = (d^(-1/2)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0')*invpd(sigma1)*(yp-xp*b0'))[1])
        
            sigma1 = Hermitian(sigmat(vec(phi0), p))               # numerator of acceptance prob 
            d = det(sigma1)
            psi0 = (d^(-1/2)) * exp((-0.5 / s20) * (transp_dbl(yp - xp * b0')*invpd(sigma1)*(yp-xp*b0'))[1])
            =#

            
            sigma1 = sigmat(phi1, p)       # numerator of acceptance prob
            sigma1 = Hermitian(sigma1)
            sigroot = cholesky(sigma1, Val(true)).U
            p1 = transp_dbl(inv(sigroot))
            ypst = p1 * yp
            xpst = p1 * xp
            d = det(p1' * p1)
            psi1 = (d^(1 / 2)) * exp(-0.5 * (ypst - xpst * b0')' * (ypst - xpst * b0') / s20)
        
            sigma1 = sigmat(phi0, p)       # numerator of acceptance prob
            sigma1 = Hermitian(sigma1) 
            sigroot = cholesky(sigma1, Val(true)).U
            p1 = transp_dbl(inv(sigroot))
            ypst = p1 * yp
            xpst = p1 * xp
            d = det(p1' * p1)
            psi0 = (d^(1 / 2)) * exp(-0.5 * (ypst - xpst * b0')' * (ypst - xpst * b0') / s20)
            

            if psi0 == 0
                accept = 1
            else
                u = rand(Uniform(0,1))
                accept = u <= psi1[1,1] / psi0[1,1]
            end
            phi1 = phi1 * accept + phi0 * (1 - accept)
        end
        
        # generation of beta 
        sigma = Hermitian(sigmat(phi1, p))              # sigma = sigroot' * sigroot 
        sigroot = cholesky(sigma).U                       # signber2v = p1' * p1 
        p1 = transp_dbl(inv(sigroot))
        ypst = p1 * yp
        xpst = p1 * xp
        yst = [ypst; gendiff(y, phi1)]
        xst = [xpst; gendiff(x, phi1)]

        V = invpd(B0__ + s20^(-1) * xst' * xst)
        bhat = V * (B0__ * vec(b0) + s20^(-1) * xst' * yst)

        b1 = sim_MvNormal(vec(bhat), Matrix(V))

        signbeta1 = (b1[2, 1] <= 0.0) * (xvar == 1)
        signmax1 = signmax1 + (1 * signbeta1)
        signbeta2 = (b1[3, 1] <= 0.0) * (varassign[nfc][1] == xvar)
        signmax2 = signmax2 + (1 * signbeta2)

        if signmax1 >= 100
            facts[:, 1] = (-1) * facts[:, 1]
            x[:, 2] = facts[:, 1]
            signmax1 = 1
        end
        if signmax2 >= 100
            facts[:, 1 + nfc] = (-1) * facts[:, 1 + nfc]
            x[:, 3] = facts[:, 1 + nfc]
            signmax2 = 1
        end
    end

    # generation of s2 
    nn = n + v0_
    d = d0_ + (transp_dbl(yst - xst * b1)*(yst-xst*b1))[1, 1]
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