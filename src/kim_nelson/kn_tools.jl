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
@doc """
    
    kalmanFilter(data, ssmodel)

Description: 
Apply Kalman filter to observed data. 
Measurement Equation:   
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t} .
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t};
    e_{t} ~ i.i.d.N(0,R);
    v_{t} ~ i.i.d.N(0,Q);
    z_{t} ~ i.i.d.N(0,Z);
    E(e_t v_s') = 0.

Inputs: 
- data      = observed data 
- H         = measurement eq. state coef. matrix
- A         = measurement eq. exogenous coef. matrix
- F         = state eq. companion matrix
- μ         = state eq. intercept term
- R         = covariance matrix on measurement disturbance
- Q         = covariance matrix on state disturbance
- Z         = covariance matrix on predetermined var vector 
"""
function kalmanFilter(data_y, ssmodel)

    @unpack H, A, F, μ, R, Q, Z = ssmodel

    # Save number of observations 
    num_obs = size(data_y)[1]

    # Empty filtered data matrices 
    data_filtered_y = similar(data_y)
    data_filtered_β = zeros(num_obs, size(Q)[1])

    # Create empty lists for P_{t}, P_{t|t-1}
    Ptt = Any[]
    Pttlag = Any[]

    # Initialize β_pred and P_pred 
    β_pred_laglag = inv(I - F) * μ
    P_pred_laglag = ones(size(Q)[1], size(Q)[1])

    for t = 1:num_obs
        # Save current obs of y
        y = data_y[t, :]

        # Prediction 
        β_pred_lag = μ + F * β_pred_laglag
        P_pred_lag = F * P_pred_laglag * transpose(F) + Q
        y_pred_lag = H * β_pred_lag
        η_pred_lag = y - y_pred_lag
        f_pred_lag = H * P_pred_lag * transpose(H) + R

        # Save P_{t|t-1}
        push!(Pttlag, P_pred_lag)

        # Updating 
        K = P_pred_lag * transpose(H) * pinv(f_pred_lag)
        β_pred = β_pred_lag + K * η_pred_lag
        P_pred = P_pred_lag - K * H * P_pred_lag

        # Save P_{t|t}
        push!(Ptt, P_pred)

        # Save data 
        data_filtered_y[t, :] = y_pred_lag
        data_filtered_β[t, :] = β_pred

        # Lag the predictions 
        β_pred_laglag = β_pred
        P_pred_laglag = P_pred
    end

    # Returned filtered series 
    # for obs variable and state 
    return data_filtered_y, data_filtered_β, Pttlag, Ptt
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
@doc """
    
    kalmanSmoother(data, ssmodel)

Description: 
Apply Kalman smoother to observed data. 
Measurement Equation:   
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t}.
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t};
    e_{t} ~ i.i.d.N(0,R);
    v_{t} ~ i.i.d.N(0,Q);
    z_{t} ~ i.i.d.N(0,Z);
    E(e_t v_s') = 0.

Inputs: 
- data      = observed data 
- H         = measurement eq. state coef. matrix
- A         = measurement eq. exogenous coef. matrix
- F         = state eq. companion matrix
- μ         = state eq. intercept term
- R         = covariance matrix on measurement disturbance
- Q         = covariance matrix on state disturbance
- Z         = covariance matrix on predetermined var vector 
"""
function kalmanSmoother(data_y, ssmodel)

    @unpack H, A, F, μ, R, Q, Z = ssmodel

    # Save number of observations 
    num_obs = size(data_y)[1]

    # Empty filtered data matrices 
    data_smoothed_y = similar(data_y)
    data_smoothed_β = zeros(num_obs, size(Q)[1])

    # Create empty list for P_{t|T}
    PtT = Any[]

    # Run Kalman filter 
    data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, ssmodel)

    # Initialize β_{t+1|T} (β_{T|T})
    βtflagT = data_filtered_β[end]
    data_smoothed_β[end] = βtflagT

    # Initialize P_{t+1|T} (P_{T|T})
    Ptflag_T = Ptt[end]
    push!(PtT, Ptflag_T)

    # Initialize y_{t|T} (y_{T|T})
    data_smoothed_y[end] = data_filtered_y[end]

    # Run Kalman smoother 
    for i = 1:(num_obs-1)

        # Retrieve β_{t|t}
        βtt = data_filtered_β[end-i]

        # Compute β_{t|T} using β_{t+1|T}, β_{t|t}, P_{t|t}, and P_{t+1|t}
        βtT = βtt +
              Ptt[end-i] * transpose(F) * inv(Pttlag[end-i+1]) *
              (βtflagT - F * βtt - μ)

        # Store β_{t+1|T} in smoothed data 
        data_smoothed_β[end-i] = βtT

        # Set β_{t|T} as new β_{t+1|T} for next iteration 
        βtflagT = βtT

        # Compute P_{t|T} using P_{t|t}, P_{t+1|t}, and P_{t+1|T}
        Pt_T = Ptt[end-i] +
               Ptt[end-i] * transpose(F) * inv(Pttlag[end-i+1]) *
               (Ptflag_T - Pttlag[end-i+1]) *
               transpose(Ptt[end-i] * transpose(F) * inv(Pttlag[end-i+1]))

        # Store P_{t|T}
        push!(PtT, Pt_T)

        # Set P_{t|T} as new P_{t+1|T} for next iteration 
        Ptflag_T = Pt_T

        # Generate y_{t|T} (smoothed obs.)
        ytT = H * βtT

        # Store smoothed obs.
        data_smoothed_y[end-i] = ytT
    end

    # Flip P_{t|T} list 
    PtT = reverse(PtT)

    # Returned filtered series 
    # for obs variable and state 
    return data_smoothed_y, data_smoothed_β, PtT
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
@doc """
    
    KNFactorSampler(data_y, ssmodel)

Description: 
Draw a sample series of dynamic factor from conditional distribution in Ch 8, Kim & Nelson (1999).
Measurement Equation:   
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t}.
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t};
    e_{t} ~ i.i.d.N(0,R);
    v_{t} ~ i.i.d.N(0,Q);
    z_{t} ~ i.i.d.N(0,Z);
    E(e_t v_s') = 0.

Inputs: 
- data      = observed data 
- H         = measurement eq. state coef. matrix
- A         = measurement eq. exogenous coef. matrix
- F         = state eq. companion matrix
- μ         = state eq. intercept term
- R         = covariance matrix on measurement disturbance
- Q         = covariance matrix on state disturbance
- Z         = covariance matrix on predetermined var vector 
"""
function KNFactorSampler(data_y, ssmodel)

    @unpack H, A, F, μ, R, Q, Z = ssmodel

    # Run Kalman filter 
    data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, ssmodel)

    # Create placeholders for factor distr. 
    # mean vector and covariance matrix for all t 
    β_t_mean = Any[]
    β_t_var = Any[]

    # Record number of time periods 
    T = size(data_y)[1]

    # Create empty vector for factor realizations
    β_realized = similar(data_filtered_β)

    # Initialize β_realized 
    push!(β_t_mean, data_filtered_β[T, :])
    push!(β_t_var, Ptt[T])
    β_realized[T, :] = sim_MvNormal_alt(β_t_mean[1], β_t_var[1])

    # Generate `β_t_mean` and `β_t_var`
    # for all time periods 
    if isposdef(Q) == false
        ## IF Q IS SINGULAR 

        # Determine number of rows
        # of state factor used 
        num_use_rows = 0
        for i = 1:size(Q)[1]
            if Q[i, i] != 0
                num_use_rows += 1
            end
        end

        # Create modified F and Q matrices 
        F_star = F[1:num_use_rows, :]
        Q_star = Q[1:num_use_rows, 1:num_use_rows]

        # Iteratively generate conditional draws 
        # of state vector 
        for j = 1:(T-1)

            # β_{t|t,β*_{t+1}}
            β_t_mean_temp = data_filtered_β[T-j, :]
            +Ptt[T-j] * transpose(F_star) * inv(F_star * Ptt[T-j] * transpose(F_star) + Q_star) * (β_realized[T+1-j, :][1:num_use_rows] - μ[1:num_use_rows] - F_star * data_filtered_β[T-j, :])
            push!(β_t_mean, β_t_mean_temp)

            # P_{t|t,β*_{t+1}}
            β_t_var_temp = Ptt[T-j] - Ptt[T-j] * transpose(F_star) * inv(F_star * Ptt[T-j] * transpose(F_star) + Q_star) * F_star * Ptt[T-j]
            push!(β_t_var, β_t_var_temp)

            # Draw new β_t 
            β_realized[T-j, :] = sim_MvNormal_alt(β_t_mean[1+j], β_t_var[1+j])
        end
    else
        ## IF Q IS NOT SINGULAR (redundant, but potentially faster) 
        for j = 1:(T-1)

            # β_{t|t,β_{t+1}}
            β_t_mean_temp = data_filtered_β[T-j, :]
            +Ptt[T-j] * transpose(F) * inv(F * Ptt[T-j] * transpose(F) + Q) * (β_realized[T+1-j, :] - μ - F * data_filtered_β[T-j, :])
            push!(β_t_mean, β_t_mean_temp)

            # P_{t|t,β_{t+1}}
            β_t_var_temp = Ptt[T-j] - Ptt[T-j] * transpose(F) * inv(F * Ptt[T-j] * transpose(F) + Q) * F * Ptt[T-j]
            push!(β_t_var, β_t_var_temp)

            # Draw new β_t 
            β_realized[T-j, :] = sim_MvNormal_alt(β_t_mean[1+j], β_t_var[1+j])
        end
    end

    # Return sampled factor series 
    # fot t = 1,...,T 
    return β_realized
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
@doc """
    
    linearRegressionSampler(Y,X)

Description: 
Estimate β and σ^2 in Y = Xβ + e, e ~ N(0,σ^2 I_T).
Generate samples of β and σ^2. 

Inputs: 
- Y     = Dependent data matrix
- X     = Independent data matrix 
"""
function linearRegressionSampler(Y, X)

    # Save number of obs 
    T = size(X)[1]

    # Initialize σ2 
    σ2 = 1.0

    ##################################
    ##################################
    # Generate new β

    ## Prior parameters in N(β0,Σ0)
    β0 = zeros(size(X)[2])
    Σ0 = Matrix(I, size(β0)[1], size(β0)[1]) .* 1000.0

    ## Posterior parameters in N(β1,Σ1) 
    β1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X) * (inv(Σ0) * β0 + inv(σ2) * transpose(X) * Y)
    β1 = vec(β1)
    Σ1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X)
    Σ1 = Hermitian(Σ1)

    ## Generate new β
    β = sim_MvNormal(β1, Σ1)

    ##################################
    ##################################
    # Update σ2

    ## Prior parameters in IG(ν0/2, δ0/2) 
    ν0 = 0.002
    δ0 = 0.002

    ## Posterior parameters in IG(ν1/2, δ1/2)
    ν1 = ν0 + T
    δ1 = δ0 + norm(Y - X * β)^2

    ## Generate new σ2
    σ2 = rand(InverseGamma(ν1 / 2, δ1 / 2))

    ##################################
    ##################################
    # Return parameters 
    return β, σ2
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
@doc """
    
    linearRegressionSamplerRestrictedVariance(Y, X, σ2)

Description: 
Estimate β and σ^2 in Y = Xβ + e, e ~ N(0,σ^2 I_T),
where σ^2 is restricted to some chosen value. 
Generate samples of β and σ^2. 

Inputs: 
- Y     = Dependent data matrix
- X     = Independent data matrix 
- σ2    = Restricted error variance 
"""
function linearRegressionSamplerRestrictedVariance(Y, X, σ2)

    # Save number of obs 
    T = size(X)[1]

    ##################################
    ##################################
    # Generate new β

    ## Prior parameters in N(β0,Σ0)
    β0 = zeros(size(X)[2])
    Σ0 = Matrix(I, size(β0)[1], size(β0)[1]) .* 1000.0

    ## Posterior parameters in N(β1,Σ1) 
    β1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X) * (inv(Σ0) * β0 + inv(σ2) * transpose(X) * Y)
    β1 = vec(β1)
    Σ1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X)
    Σ1 = Hermitian(Σ1)

    ## Generate new β
    β = sim_MvNormal(β1, Σ1)

    ##################################
    ##################################
    # Return parameters 
    return β
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
@doc """
    
    autocorrErrorLinearRegressionSampler(Y, X, error_lag_num)

Description: 
Estimate β, σ^2, and ϕ in Y = Xβ + e, e = Eϕ + ν, ν_t ~ i.i.d.N(0,σ^2).  
Generate samples of β, σ^2, and ϕ.  
Procedure descried in Section 7.4.2 in Kim & Nelson.

Inputs: 
- Y             = Dependent data matrix
- X             = Independent data matrix 
- error_lag_num = Number of lags in the disturbance DGP 
"""

function autocorrErrorLinearRegressionSampler(Y, X, ϕold, σ2old, error_lag_num)

    # Save number of obs 
    T = size(X)[1]

    # Initialize σ2 
    ϕ = ϕold
    σ2 = σ2old

    ##################################
    ##################################
    # Generate β

    ## Prior parameters in N(b0,A0)
    b0 = zeros(size(X)[2])
    A0 = Matrix(I, size(b0)[1], size(b0)[1]) .* 1000.0

    ## Generate X^⋆ 
    X_star = similar(X[(1+length(ϕ)):end, :])
    for i = 1:size(X)[2] # iterate over variables in X
        x_temp = X[:, i]
        for p = 1:length(ϕ) # iterate over lag params in ϕ
            x_temp[(1+length(ϕ)):end, :] = x_temp[(1+length(ϕ)):end, :] - ϕ[p] .* lag(x_temp, p)[(1+length(ϕ)):end, :]
        end
        x_temp = x_temp[(1+length(ϕ)):end, :]
        X_star[:, i] = x_temp
    end

    ## Generate Y^⋆
    Y_star = similar(Y[(1+length(ϕ)):end, :])
    y_temp = Y
    for p = 1:length(ϕ)
        y_temp = Y
        y_temp = y_temp[(1+length(ϕ)):end, :] - ϕ[p] .* lag(y_temp, p)[(1+length(ϕ)):end, :]
    end
    Y_star = y_temp

    ## Posterior parameters in N(b1,A1) 
    b1 = inv(inv(A0) + inv(σ2) * transpose(X_star) * X_star) * (inv(A0) * b0 + inv(σ2) * transpose(X_star) * Y_star)
    b1 = vec(b1)
    A1 = inv(inv(A0) + inv(σ2) * transpose(X_star) * X_star)
    A1 = Hermitian(A1)

    ## Draw new β
    β = sim_MvNormal(b1, A1)

    ##################################
    ##################################
    # Generate ϕ

    ## Prior parameters in N(c0,B0)
    c0 = zeros(size(ϕ)[1])
    B0 = Matrix(I, size(c0)[1], size(c0)[1]) .* 1000.0

    ## Generate e^⋆ 
    e_star = Y - X * β

    ## Generate E^⋆
    E_star = zeros(T, length(ϕ))
    E_star = E_star[(1+length(ϕ)):end, :]
    for i = 1:(size(E_star)[2]) # iterate over variables in X
        e_temp = lag(e_star, i)
        E_star[:, i] = e_temp[(1+length(ϕ)):end]
    end
    e_star = e_star[(1+length(ϕ)):end]

    ## Posterior parameters in N(c1,B1)
    c1 = inv(inv(B0) + inv(σ2) * transpose(E_star) * E_star) * (inv(B0) * c0 + inv(σ2) * transpose(E_star) * e_star)
    c1 = vec(c1)
    B1 = inv(inv(B0) + inv(σ2) * transpose(E_star) * E_star)
    B1 = Hermitian(B1)

    ## Generate new ϕ
    ###=
    ind = 0
    accept = 0
    ϕ = similar(ϕold)
    while accept == 0

        ind += 1

        ## Draw ψ
        ϕ = sim_MvNormal(c1, B1)

        ## Check for stationarity 
        coef = [-reverse(vec(ϕ), dims = 1); 1]                      # check stationarity 
        root = roots(Polynomial(reverse(coef)))
        rootmod = abs.(root)
        accept = min(rootmod...) >= 1.01

        ## If while loop goes on for too long 
        if ind > 100
            ϕ = ϕold
            coef = [-reverse(vec(ϕ), dims = 1); 1]                      # check stationarity 
            root = roots(Polynomial(reverse(coef)))
            rootmod = abs.(root)
            accept = min(rootmod...) >= 1.01
        end
    end
    ##=#

    #ϕ = sim_MvNormal(c1, B1)


    ##################################
    ##################################
    # Generate σ2

    ## Prior parameters in IG(ν0/2, δ0/2) 
    ν0 = 0.002
    δ0 = 0.002

    ## Posterior parameters in IG(ν1/2, δ1/2)
    ν1 = ν0 + T
    δ1 = δ0 .+ norm(Y_star - X_star * β)^2
    δ1 = δ1[1, 1]

    ## Generate new σ2
    σ2 = rand(InverseGamma(ν1 / 2, δ1 / 2))

    ##################################
    ##################################
    # Return parameters 
    return β, σ2, ϕ
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