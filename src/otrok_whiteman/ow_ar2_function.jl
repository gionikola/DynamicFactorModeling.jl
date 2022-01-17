@doc """

    ar2(y, x, p, βbar, Bbarinv, ϕbar, Vbarinv, υbar, δbar, βold, ϕold, σ2old, varind, lev2ind, factors, numobs, varassign)

Description:
Estimate observed variable hyperparameters conditional on the factors. 
Refer to pg. 1002 of Otrok and Whiteman (1998), or Chib and Greenberg (1994). 

Inputs:
- y             = observed series vector.
- x             = data matrix of regressors in the observation equation (intercept + factors).
- p             = number of idiosyncratic error lags. 
- βbar          = prior mean of factor loading (factor coefficient in measurement equation). 
- Bbarinv       = prior variance of factor loading (factor coefficient in measurement equation). 
- ϕbar          = prior mean of phi (idiosyncratic AR polynomial). 
- Vbarinv       = prior precision of phi (idiosyncratic AR polynomial). 
- υbar          = inverted gamma parameter of innovation variances. 
- δbar          = inverted gamma parameter of innovation variances. 
- βold          = old draw of factor loading. 
- ϕold          = old draw of phi (idiosyncratic AR polynomial). 
- σ2old         = old draw of innovation variances. 
- varind        = measurement eq. regressor index. 
- lev2ind       = level-2 factor index. 
- factors       = factor draw. 
- numobs        = total number of time periods / observations in the sample. 
- varassign     = list containing vector of observed series indeces assigned to each level-2 factor.

Outputs:
- βnew      = β coefficient hyperparameter estimates.
- ϕnew      = ϕ coefficient hyperparameter estimates.  
- σ2new     = σ^2 coefficient hyperparameter estimate. 
- factors   = factor estimates. 
"""

function ar2(y, x, p, βbar, Bbarinv, ϕbar, Vbarinv, υbar, δbar, βold, ϕold, σ2old, varind, lev2ind, factors, numobs, varassign)

    # Guarantee that y is a vector 
    y = vec(y)

    # Define ỹ = first p observations of series vector 
    ỹ = y[1:p]

    # Define x̃ = first p observations of regressor matrix 
    x̃ = x[1:p, :]

    # Create Φ matrix (companion matrix of idiosyncratic error autoregression)
    Φ = [ϕold'; I(p - 1) zeros(p - 1)]

    # Covariance matrix component of the first p errors (Σ in σ2Σ)
    vecΣ = inv(I - Φ ⊗ Φ) * vec([1; zeros(p - 1)] * [1; zeros(p - 1)]')
    Σ = reshape(vecΣ, p, p)
    Σ = Hermitian(Σ) 

    # Compute the Cholesky factor of Σ
    # notation is such that Q Q' = Σ
    # so that Q is the lower factor of Σ
    Q = cholesky(Σ).L

    # Define ỹ∗1 = Q^{-1} ỹ and x̃∗1 = Q^{-1} x̃
    ỹstar1 = inv(Q) * ỹ
    x̃star1 = inv(Q) * x̃

    # Define ỹ∗2 as a (T-p × 1) with t-th row ϕ(L)y 
    # and x̃∗2 as a (T-p × 3) with t-th row ϕ(1)
    ỹstar2 = gendiff(y, ϕold)
    x̃star2 = [gendiff(ones(numobs), ϕold) gendiff(factors[:, 1], ϕold) gendiff(factors[:, lev2ind], ϕold)]

    # Define e = last T-p "observed" idiosyncratic errors 
    e = y - x * βold
    e = e[(p+1):numobs]

    # Define E = [ lag(e,1) lag(e,2) ... lag(e,p) ]
    E = zeros(numobs, p)
    for j = 1:p
        E[:, j] = lag(y - x * βold, j, default = 0.0)
    end
    E = E[(p+1):numobs, :]

    # Define x̃∗ = [x̃∗1 ; x̃∗2] and ỹ∗ = [ỹ∗1 ; ỹ∗2]
    x̃star = [x̃star1; x̃star2]
    ỹstar = [ỹstar1; ỹstar2]

    #############################################
    #############################################
    ## Create objects for posterior densities 

    # Define V̅ and B̅
    Vbar = inv(Vbarinv)
    Bbar = inv(Bbarinv)

    # Define V = V̅ + σ²E'E 
    V = Vbar + σ2old * E' * E

    # Define B = B̅ + ̅σ²x̃∗'x̃∗
    B = Bbar + σ2old * x̃star' * x̃star

    # Define ̂ϕ = inv(V)(V̅ ̅ϕ + σ²E'e)
    ϕhat = inv(V) * (Vbar * ϕbar + σ2old * E' * e)

    #############################################
    #############################################
    ## Draw β

    # Initialize sign of level-1 factor loading 
    # of first series in dataset (must be positive)
    β2sign = false

    # Initialize sign of level-2 factor loading 
    # of first series in corresponding level-2 group
    # (must be positive) 
    β3sign = false

    # Initialize β
    βnew = βold

    # Keep track of wrong sign persistence 
    β2signmax = 0
    β3signmax = 0

    whileind = 0
    while β2sign == false || β3sign == false

        whileind += 1
        println("Series index: $varind")
        println("While loop index: $whileind")
        println("β2sigmax: $β2signmax")
        println("β3signmax: $β3signmax") 

        # Draw new β
        βnew = sim_MvNormal(inv(B) * (Bbar * βbar + inv(σ2old) * x̃star' * ỹstar), inv(B))

        # Check if new β draw satisfies
        # sign identification restrictions 
        if varind == 1
            β2sign = (βnew[2] > 0)
            if β2sign == false
                β2signmax += 1
            end
        else
            β2sign = true
        end
        if varind == varassign[lev2ind][1]
            β3sign = (βnew[3] > 0)
            if β3sign == false
                β3signmax += 1
            end
        else
            β3sign = true
        end

        if β2signmax > 100
        
            # Reflect level-1 factor over x-axis 
            factors[:, 1] = -factors[:, 1]
        
            # Redefine x 
            x = [ones(numobs) factors[:, 1] factors[:, 1+lev2ind]]
        
            # Define x̃ = first p observations of regressor matrix 
            x̃ = x[1:p, :]
        
            # Create Φ matrix (companion matrix of idiosyncratic error autoregression)
            Φ = [ϕold'; I(p - 1) zeros(p - 1)]
        
            # Covariance matrix component of the first p errors (Σ in σ2Σ)
            vecΣ = inv(I - Φ ⊗ Φ) * vec([1; zeros(p - 1)] * [1; zeros(p - 1)]')
            Σ = reshape(vecΣ, p, p)
            Σ = Hermitian(Σ)
        
            # Compute the Cholesky factor of Σ
            # notation is such that Q Q' = Σ
            # so that Q is the lower factor of Σ
            Q = cholesky(Σ).L
        
            # Define ỹ∗1 = Q^{-1} ỹ and x̃∗1 = Q^{-1} x̃
            ỹstar1 = inv(Q) * ỹ
            x̃star1 = inv(Q) * x̃
        
            # Define ỹ∗2 as a (T-p × 1) with t-th row ϕ(L)y 
            # and x̃∗2 as a (T-p × 3) with t-th row ϕ(1)
            ỹstar2 = gendiff(y, ϕold)
            x̃star2 = [gendiff(ones(numobs), ϕold) gendiff(factors[:, 1], ϕold) gendiff(factors[:, lev2ind], ϕold)]
        
            # Define e = last T-p "observed" idiosyncratic errors 
            e = y - x * βold
            e = e[(p+1):numobs]
        
            # Define E = [ lag(e,1) lag(e,2) ... lag(e,p) ]
            E = zeros(numobs, p)
            for j = 1:p
                E[:, j] = lag(y - x * βold, j, default = 0.0)
            end
            E = E[(p+1):numobs, :]
        
            # Define x̃∗ = [x̃∗1 ; x̃∗2] and ỹ∗ = [ỹ∗1 ; ỹ∗2]
            x̃star = [x̃star1; x̃star2]
            ỹstar = [ỹstar1; ỹstar2]
        
            # Reset `β2signmax`
            β2signmax = 0
        end

        if β3signmax > 1000

            # Reflect level-1 factor over x-axis 
            factors[:, 1+lev2ind] = -factors[:, 1+lev2ind]

            # Redefine x 
            x = [ones(numobs) factors[:, 1] factors[:, 1+lev2ind]]

            # Define x̃ = first p observations of regressor matrix 
            x̃ = x[1:p, :]

            # Create Φ matrix (companion matrix of idiosyncratic error autoregression)
            Φ = [ϕold'; I(p - 1) zeros(p - 1)]

            # Covariance matrix component of the first p errors (Σ in σ2Σ)
            vecΣ = inv(I - Φ ⊗ Φ) * vec([1; zeros(p - 1)] * [1; zeros(p - 1)]')
            Σ = reshape(vecΣ, p, p)
            Σ = Hermitian(Σ) 

            # Compute the Cholesky factor of Σ
            # notation is such that Q Q' = Σ
            # so that Q is the lower factor of Σ
            Q = cholesky(Σ).L

            # Define ỹ∗1 = Q^{-1} ỹ and x̃∗1 = Q^{-1} x̃
            ỹstar1 = inv(Q) * ỹ
            x̃star1 = inv(Q) * x̃

            # Define ỹ∗2 as a (T-p × 1) with t-th row ϕ(L)y 
            # and x̃∗2 as a (T-p × 3) with t-th row ϕ(1)
            ỹstar2 = gendiff(y, ϕold)
            x̃star2 = [gendiff(ones(numobs), ϕold) gendiff(factors[:, 1], ϕold) gendiff(factors[:, lev2ind], ϕold)]

            # Define e = last T-p "observed" idiosyncratic errors 
            e = y - x * βold
            e = e[(p+1):numobs]

            # Define E = [ lag(e,1) lag(e,2) ... lag(e,p) ]
            E = zeros(numobs, p)
            for j = 1:p
                E[:, j] = lag(y - x * βold, j, default = 0.0)
            end
            E = E[(p+1):numobs, :]

            # Define x̃∗ = [x̃∗1 ; x̃∗2] and ỹ∗ = [ỹ∗1 ; ỹ∗2]
            x̃star = [x̃star1; x̃star2]
            ỹstar = [ỹstar1; ỹstar2]

            # Reset `β2signmax`
            β3signmax = 0
        end

    end

    #############################################
    #############################################
    ## Draw ϕ

    # Draw new ϕ candidate 
    ϕnew = sim_MvNormal(ϕhat, inv(V))

    # Check for stationary of new ϕ draw 
    coef = [-reverse(ϕnew); 1]
    coef = vec(coef)
    root = roots(Polynomial(reverse(coef)))
    rootmod = abs.(root)
    accept = min(rootmod...) >= 1.001

    # Complete 
    if accept == 0
        ϕnew = ϕold
    else
        # Define Ψ(ϕold) (pg. 1002)
        Ψold = det(sigmat(ϕold, p))^(-1 / 2) * exp(-(1 / (2 * σ2old)) * (ỹ - x̃ * βnew)' * inv(sigmat(ϕold, p)) * (ỹ - x̃ * βnew))
        Ψold = first(Ψold)

        # Define Ψ(ϕnew) (pg. 1002)
        Ψnew = det(sigmat(ϕnew, p))^(-1 / 2) * exp(-(1 / (2 * σ2old)) * (ỹ - x̃ * βnew)' * inv(sigmat(ϕnew, p)) * (ỹ - x̃ * βnew))
        Ψnew = first(Ψnew)

        # Determine acceptance probability 
        if Ψold == 0
            accept = 1
        else
            u = rand()
            accept = (u <= Ψnew / Ψold)
        end

        # Pick new ϕ
        ϕnew = ϕnew * accept + ϕold * (1 - accept)
    end

    #############################################
    #############################################
    ## Draw σ2 

    # Draw new σ2 from inverse gamma distribution 
    #σ2new = rand(InverseGamma((υbar + numobs) / 2, (δbar + (ỹstar - x̃star * βnew)' * (ỹstar - x̃star * βnew)) / 2))
    σ2new = rand(InverseGamma((υbar + numobs) / 2, (δbar + norm(ỹstar - x̃star * βnew)^2) / 2))

    #############################################
    #############################################
    ## Return new hyperparameter draws 
    return βnew, σ2new, ϕnew, factors
end 

