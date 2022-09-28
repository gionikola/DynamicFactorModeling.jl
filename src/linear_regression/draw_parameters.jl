"""
    draw_parameters() 

Description.
"""
function draw_parameters(Y::Matrix{Float64}, X::Matrix{Float64}, σ2::Float64)

    β = draw_coefficients(Y,X,σ2)
    σ2 = draw_error_variance(Y,X,β)

    return β::Vector{Float64}, σ2::Float64
end 

function draw_parameters(Y::Vector{Float64}, X::Matrix{Float64}, σ2::Float64)

    β = draw_coefficients(Y, X, σ2)
    σ2 = draw_error_variance(Y, X, β)

    return β::Vector{Float64}, σ2::Float64
end 

function draw_parameters(Y::Matrix{Float64}, X::Vector{Float64}, σ2::Float64)

    β = draw_coefficients(Y, X, σ2)
    σ2 = draw_error_variance(Y, X, β)

    return β::Float64, σ2::Float64
end 

function draw_parameters(Y::Vector{Float64}, X::Vector{Float64}, σ2::Float64)

    β = draw_coefficients(Y, X, σ2)
    σ2 = draw_error_variance(Y, X, β)

    return β::Float64, σ2::Float64
end 

"""
    draw_parameters() 

Description.
"""
function draw_parameters(Y::Matrix{Float64}, X::Matrix{Float64}, ϕ::Vector{Float64}, σ2::Float64)

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
    β = draw_coefficients(Y_star, X_star, σ2)

    ## Generate error 
    error = Y - X * β
    errors = zeros(length(error), length(ϕ))
    for p = 1:length(ϕ)
        errors[:, p] = lag(error, p, default = 0.0)
    end
    errors = errors[(1+length(ϕ)):end,:]
    error = error[(1+length(ϕ)):end]

    ## Draw ϕ
    ϕ = draw_coefficients(error, errors, σ2)

    ## Draw σ2 
    σ2 = draw_error_variance(Y_star, X_star, β)

    return β::Vector{Float64}, ϕ::Vector{Float64}, σ2::Float64
end 

function draw_parameters(Y::Vector{Float64}, X::Matrix{Float64}, ϕ::Vector{Float64}, σ2::Float64)

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
    β = draw_coefficients(Y_star, X_star, σ2)

    ## Generate error 
    error = Y - X * β
    errors = zeros(length(error), length(ϕ))
    for p = 1:length(ϕ)
        errors[:, p] = lag(error, p, default=0.0)
    end
    errors = errors[(1+length(ϕ)):end, :]
    error = error[(1+length(ϕ)):end]

    ## Draw ϕ
    ϕ = draw_coefficients(error, errors, σ2)

    ## Draw σ2 
    σ2 = draw_error_variance(Y_star, X_star, β)

    return β::Vector{Float64}, ϕ::Vector{Float64}, σ2::Float64
end 