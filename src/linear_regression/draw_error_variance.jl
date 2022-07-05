function draw_error_variance(Y,X,β)

    ## Priors
    T0 = 1
    θ0 = 0.1
    
    ## Degrees of freedom parameter posterior
    T = size(X)[1]
    T1 = (T0 + T) 

    ## Scale parameter posterior
    θ1 = θ0 + dot((Y - X * β),(Y - X * β))
    
    ## Return draw of σ2
    return σ2 = Γinv(T1,θ1)
end 