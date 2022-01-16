@doc """

    ar2(y, x, p, βbar, Bbarinv, ϕbar, Vbarinv, υbar, δbar, βold, ϕold, σ2old, varind, lev2ind, factors, numobs, numfactassign, varassign)

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
- numfactassign = number of variables each factor loads on.
- varassign     = list containing vector of observed series indeces assigned to each level-2 factor.

Outputs:
- βnew      = β coefficient hyperparameter estimates.
- ϕnew      = ϕ coefficient hyperparameter estimates.  
- σ2new     = σ^2 coefficient hyperparameter estimate. 
- factors   = factor estimates. 
"""

function ar2(y, x, p, βbar, Bbarinv, ϕbar, Vbarinv, υbar, δbar, βold, ϕold, σ2old, varind, lev2ind, factors, numobs, numfactassign, varassign)



    return βnew, ϕnew, σ2new, factors
end 