using LinearAlgebra
using Random
using Distributions 
using PDMats 
using PDMatsExtras 

######################
######################
######################
@doc """
    
    kalmanFilter(data, H, A, F, μ, R, Q, Z)

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
function kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)

    # Save number of observations 
    num_obs = size(data_y)[1]

    # Empty filtered data matrices 
    data_filtered_y = similar(data_y)
    data_filtered_β = zeros(num_obs, size(Q)[1])

    # Create empty lists for P_{t}, P_{t|t-1}
    Ptt     = Any[] 
    Pttlag  = Any[] 

    # Initialize β_pred and P_pred 
    β_pred_laglag = inv(I - F) * μ
    P_pred_laglag = ones(size(Q)[1],size(Q)[1])

    for t = 1:num_obs
        # Save current obs of z & y
        z = data_z[t,:]
        y = data_y[t,:]

        # Prediction 
        β_pred_lag = μ + F * β_pred_laglag
        P_pred_lag = F * P_pred_laglag * transpose(F) + Q
        y_pred_lag = H * β_pred_lag + A * z
        η_pred_lag = y - y_pred_lag
        f_pred_lag = H * P_pred_lag * transpose(H) + R

        # Save P_{t|t-1}
        push!(Pttlag, P_pred_lag)

        # Updating 
        K = P_pred_lag * transpose(H) * inv(f_pred_lag)
        β_pred = β_pred_lag + K * η_pred_lag
        P_pred = P_pred_lag - K * H * P_pred_lag

        # Save P_{t|t}
        push!(Ptt, P_pred)

        # Save data 
        data_filtered_y[t,:] = y_pred_lag 
        data_filtered_β[t,:] = β_pred 

        # Lag the predictions 
        β_pred_laglag = β_pred
        P_pred_laglag = P_pred
    end

    # Returned filtered series 
    # for obs variable and state 
    return data_filtered_y, data_filtered_β, Pttlag, Ptt 
end 

######################
######################
######################
@doc """
    
    kalmanSmoother(data, H, A, F, μ, R, Q, Z)

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
function kalmanSmoother(data_y, data_z, H, A, F, μ, R, Q, Z)

    # Save number of observations 
    num_obs = size(data_y)[1]

    # Empty filtered data matrices 
    data_smoothed_y = similar(data_y)
    data_smoothed_β = zeros(num_obs, size(Q)[1])

    # Create empty list for P_{t|T}
    PtT = Any[]

    # Run Kalman filter 
    data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)

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
        ytT = H * βtT + A * data_z[end-i]

        # Store smoothed obs.
        data_smoothed_y[end-i] = ytT
    end

    # Flip P_{t|T} list 
    PtT = reverse(PtT)

    # Returned filtered series 
    # for obs variable and state 
    return data_smoothed_y, data_smoothed_β, PtT
end 

######################
######################
######################
@doc """
    
    dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)

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
function dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)

    # Run Kalman filter 
    data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)

    # Format non-positive definite P_{t|t}
    # matrices as PSDMat for sampler 
    for t in 1:size(Ptt)[1] 
        if isposdef(Ptt[t]) == false
            Ptt[t] = PSDMat(Ptt[t])
        end
    end 

    # Create placeholders for factor distr. 
    # mean vector and covariance matrix for all t 
    β_t_mean = Any[]
    β_t_var = Any[]

    # Record number of time periods 
    T = size(data_y)[1]

    # Create empty vector for factor realizations
    β_realized = similar(data_filtered_β)

    # Initialize β_realized 
    push!(β_t_mean, data_filtered_β[T,:])
    push!(β_t_var, Ptt[T])
    β_realized[T,:] = rand(MvNormal(β_t_mean[1], β_t_var[1]))

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
            if isposdef(β_t_var_temp) == false
                β_t_var_temp = PSDMat(β_t_var_temp)
            end
            push!(β_t_var, β_t_var_temp)
        
            # Draw new β_t 
            β_realized[T-j, :] = rand(MvNormal(β_t_mean[j], β_t_var[j]))
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
            if isposdef(β_t_var_temp) == false
                β_t_var_temp = PSDMat(β_t_var_temp)
            end
            push!(β_t_var, β_t_var_temp)
    
            # Draw new β_t 
            β_realized[T-j, :] = rand(MvNormal(β_t_mean[j], β_t_var[j]))
        end
    end 

    # Return sampled factor series 
    # fot t = 1,...,T 
    return β_realized
end 

######################
######################
######################
@doc """
    
    staticLinearGibbsSampler(Y,X)

Description: 
Estimate β and σ^2 in Y = Xβ + e, e ~ N(0,σ^2 I_T).
Generate samples of β and σ^2. 

Inputs: 
- Y     = Dependent data matrix
- X     = Independent data matrix 
"""
function staticLinearGibbsSampler(Y, X)

    # Create parameter lists 
    data_β = Any[]
    data_σ2 = Any[]

    # Save number of obs 
    T = size(X)[1]

    # Initialize σ2 
    σ2 = 1 

    # Apply iterated updating of β and σ^2 
    for j = 1:10000
    
        # Generate new β^j 
        ## Prior parameters in N(β0,Σ0)
        β0 = ones(size(X)[2])
        Σ0 = Matrix(I, size(β0)[1], size(β0)[1])
        ## Posterior parameters in N(β1,Σ1) 
        β1 = transpose(inv(Σ0) + inv(σ2) * transpose(X) * X) * (inv(Σ0) * β0 + inv(σ2) * transpose(X) * Y)
        Σ1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X)
        ## Generate new β
        β = rand(MvNormal(β1, Σ1))
    
        # Record new β^j
        push!(data_β, β)
    
        # Update σ2^j 
        ## Prior parameters in IG(ν0/2, δ0/2) 
        ν0 = 0.002
        δ0 = 0.002
        ## Posterior parameters in IG(ν1/2, δ1/2)
        ν1 = ν0 + T
        δ1 = δ0 + transpose(Y - X * β) * (Y - X * β)
        ## Generate new σ2
        σ2 = rand(InverseGamma(ν1 / 2, δ1 / 2))
    
        # Record new σ2^j
        push!(data_σ2, σ2)
    end

    # Drop first 3000 observations for all parameters 
    data_β = data_β[3000:10000]
    data_σ2 = data_σ2[3000:10000]
    
    # Return parameters 
    return data_β, data_σ2
end 