function draw_parameters(Y,X,σ2)

    β = draw_coefficients(Y,X,σ2)
    σ2 = draw_error_variance(Y,X,β)

    return β, σ2
end 

function draw_parameters(Y,X,ϕ,σ2)

        ## Generate X* 
        X_star = similar(X[(1+length(ϕ)):end, :])
        for i = 1:size(X)[2] # iterate over variables in X
            x_temp = X[:, i]
            for p = 1:length(ϕ) # iterate over lag params in ϕ
                x_temp[(1+length(ϕ)):end, :] = x_temp[(1+length(ϕ)):end, :] - ϕ[p] .* lag(x_temp, p)[(1+length(ϕ)):end, :]
            end
            x_temp = x_temp[(1+length(ϕ)):end, :]
            X_star[:, i] = x_temp
        end
    
        ## Generate Y*
        Y_star = similar(Y[(1+length(ϕ)):end, :])
        y_temp = Y
        for p = 1:length(ϕ)
            y_temp = Y
            y_temp = y_temp[(1+length(ϕ)):end, :] - ϕ[p] .* lag(y_temp, p)[(1+length(ϕ)):end, :]
        end
        Y_star = y_temp

        ## Draw β
        β = draw_coefficients(Y_star,X_star,σ2)

        ## Generate error 
        error = Y - X * β

        ## Draw ϕ
        ϕ = draw_coefficients(error[2:end], lag(error)[2:end][:,:], σ2)

        ## Draw σ2 
        σ2 = draw_error_variance(Y_star,X_star,β)

    return β, ϕ, σ2
end 