using LinearAlgebra

######################
######################
######################
@doc """
    
    kalmanFilter(data, H, A, F, μ, R, Q, Z)

Description: 
Apply Kalman filter to observed data. 
Measurement Equation:   
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t} 
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t}
    e_{t} ~ i.i.d.N(0,R)
    v_{t} ~ i.i.d.N(0,Q)
    z_{t} ~ i.i.d.N(0,Z)
    E(e_t v_s') = 0

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
    y_{t} = H_{t} β_{t} + A z_{t} + e_{t} 
Transition Equation:    
    β_{t} = μ + F β_{t-1} + v_{t}
    e_{t} ~ i.i.d.N(0,R)
    v_{t} ~ i.i.d.N(0,Q)
    z_{t} ~ i.i.d.N(0,Z)
    E(e_t v_s') = 0

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

    data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, 
                                                                data_z,     
                                                                H,  
                                                                A,  
                                                                F,  
                                                                μ,  
                                                                R,  
                                                                Q,  
                                                                Z)

    #######################
    ### Kalman Smoother ###
    #######################
    for i = 1:(num_obs-1)

    end

    # Returned filtered series 
    # for obs variable and state 
    return data_smoothed_β
end 

######################
######################
######################