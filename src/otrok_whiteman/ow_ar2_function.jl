function ar2(y, x, p, b0_, B0__, r0_, R0__, v0_, d0_, b0, s20, phi0, xvar, nfc, facts, capt, nreg, Size, varassign)

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

    while signbeta1 + signbeta2 >= 1.0 ##&& testind < 10000

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
        accept = min(rootmod...) >= 1.01             # all the roots bigger than 1 

        if accept == 0
            phi1 = phi0
        else
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
                u = rand(Uniform(0, 1))
                accept = u <= psi1[1, 1] / psi0[1, 1]
            end
            phi1 = phi1 * accept + phi0 * (1 - accept)
        end

        # generation of beta 
        sigma = Hermitian(sigmat(phi1, p))              # sigma = sigroot' * sigroot 
        sigroot = cholesky(sigma, Val(true)).U                       # signber2v = p1' * p1 
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
            facts[:, 1+nfc] = (-1) * facts[:, 1+nfc]
            x[:, 3] = facts[:, 1+nfc]
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