module DiagnosticSampler

using DynamicFactorModeling
using LinearAlgebra
using Random
using Statistics
using ..LocationMove: shift_factor_locations!
include("scale_move.jl")

export sample_case, companion_radius
const DFM = DynamicFactorModeling

# This test harness records diagnostics around the public update order.
# Baseline, location, and location/scale modes are checked draw-for-draw against
# the public estimators.

function stationary_ar_step(rng, series, previous, variance, prior_variance)
    p = length(previous)
    p == 0 && return Float64[], :absent
    prior_variances = DFM._regression_prior_variances(prior_variance,p,"ar_prior_variance")
    if length(series) > p
        rows = (p+1):length(series)
        regressors = DFM._regression_lags(series,p)[rows,:]
        candidate = DFM.draw_coefficients(rng,series[rows],regressors,variance; prior_variance)
    else
        candidate = sqrt.(prior_variances) .* randn(rng,p)
    end
    DFM.isstationary(candidate) || return copy(previous), :unstable
    log_ratio = DFM._stationary_initial_logdensity(series,candidate,variance) -
                DFM._stationary_initial_logdensity(series,previous,variance)
    accepted = log(rand(rng)) < min(0.0,log_ratio)
    return accepted ? (candidate,:accepted) : (copy(previous),:mh_rejected)
end

function regression_step(rng, y, x, phi, variance, initial, beta_prior, ar_prior,
                         variance_shape, variance_scale)
    if initial == :zero
        beta, next_phi, next_variance = DFM.draw_parameters(rng,y,x,phi,variance;
            initial, beta_prior_variance=beta_prior, ar_prior_variance=ar_prior,
            variance_shape, variance_scale, stationary=true, max_attempts=10000)
        return beta,next_phi,next_variance,:direct
    end
    transformed_y = vec(DFM._stationary_ar_whiten(reshape(y,:,1),phi))
    transformed_x = DFM._stationary_ar_whiten(x,phi)
    beta = DFM.draw_coefficients(rng,transformed_y,transformed_x,variance; prior_variance=beta_prior)
    residuals = y - x*beta
    next_phi, status = stationary_ar_step(rng,residuals,phi,variance,ar_prior)
    innovations = vec(DFM._stationary_ar_whiten(reshape(residuals,:,1),next_phi))
    next_variance = DFM.draw_error_variance(rng,innovations,zeros(length(y),0),Float64[];
                                        prior_shape=variance_shape,prior_scale=variance_scale)
    return beta,next_phi,next_variance,status
end

function companion_radius(coefficients)
    isempty(coefficients) && return 0.0
    length(coefficients) == 1 && return abs(only(coefficients))
    p = length(coefficients)
    transition = zeros(p,p)
    transition[1,:] = coefficients
    for lag in 2:p
        transition[lag,lag-1] = 1
    end
    return maximum(abs,eigvals(transition))
end

function trace_features(factors, coefficients, loadings, factor_ar, error_ar, anchors;
                        error_variances=nothing, active_loadings=nothing,
                        factor_levels=nothing)
    names, values = String[], Float64[]
    add(name,value) = (push!(names,name); push!(values,value))
    averages = vec(mean(factors;dims=1))
    centered = factors .- averages'
    dates = size(factors,1)
    time = collect(1.0:dates) .- (dates+1)/2
    linear = iszero(norm(time)) ? zeros(dates) : time ./ norm(time)
    quadratic = time.^2 .- mean(time.^2)
    iszero(norm(quadratic)) || (quadratic ./= norm(quadratic))
    # These fixed contrasts measure path shape without using the true factors.
    # T=1 has no linear contrast; T<=2 has no quadratic contrast, so these are zero.
    error_variances === nothing || length(error_variances) == size(coefficients,1) ||
        throw(DimensionMismatch("error_variances must contain one value per series"))
    active = active_loadings === nothing ? .!iszero.(loadings) : active_loadings
    size(active) == size(loadings) || throw(DimensionMismatch("active loading mask has wrong size"))
    factor_levels === nothing || length(factor_levels) == size(factors,2) ||
        throw(DimensionMismatch("factor_levels must contain one level per factor"))
    contributions = [centered[:,k] * loadings[:,k]' for k in axes(factors,2)]
    for series in axes(coefficients,1)
        add("intercept_$series",coefficients[series,1])
        add("signal_mean_$series",coefficients[series,1] + dot(loadings[series,:],averages))
        error_variances === nothing || add("error_variance_$series",error_variances[series])
    end
    for factor in axes(factors,2)
        factor_rms = sqrt(mean(abs2,centered[:,factor]))
        loading_norm = norm(loadings[:,factor])
        add("factor_mean_$factor",averages[factor])
        add("factor_abs_mean_$factor",abs(averages[factor]))
        add("factor_rms_$factor",factor_rms)
        add("factor_contrast_$factor",factors[end,factor]-factors[1,factor])
        add("factor_linear_contrast_$factor",dot(linear,centered[:,factor]))
        add("factor_quadratic_contrast_$factor",dot(quadratic,centered[:,factor]))
        path_norm = norm(centered[:,factor])
        add("factor_normalized_linear_$factor",iszero(path_norm) ? NaN :
            dot(linear,centered[:,factor])/path_norm)
        add("factor_normalized_quadratic_$factor",iszero(path_norm) ? NaN :
            dot(quadratic,centered[:,factor])/path_norm)
        add("loading_norm_$factor",loading_norm)
        add("contribution_rms_$factor",factor_rms*loading_norm)
        add("anchor_loading_$factor",loadings[anchors[factor],factor])
        # Fixed time and equal-series projections of the actual fitted
        # contribution. These survive both reciprocal scaling and sign folding.
        indicator_projection = vec(sum(contributions[factor];dims=2))/sqrt(size(loadings,1))
        add("contribution_linear_$factor",dot(linear,indicator_projection))
        add("contribution_quadratic_$factor",dot(quadratic,indicator_projection))
        for lag in eachindex(factor_ar[factor])
            add("factor_ar_$(factor)_$lag",factor_ar[factor][lag])
        end
        isempty(factor_ar[factor]) ||
            add("factor_radius_$factor",companion_radius(factor_ar[factor]))
    end
    for left in axes(factors,2), right in (left+1):size(factors,2)
        add("factor_correlation_$(left)_$right",cor(factors[:,left],factors[:,right]))
        # Disjoint supports give an identically zero overlap, so omit those
        # structural constants rather than calling their ESS a diagnostic.
        if any(active[:,left] .& active[:,right])
            add("contribution_overlap_$(left)_$right",
                dot(contributions[left],contributions[right])/dates)
            add("contribution_pair_rms_$(left)_$right",
                norm(contributions[left]+contributions[right])/sqrt(dates))
        end
    end
    if factor_levels !== nothing
        for level in sort(unique(factor_levels))
            level_contribution = sum(contributions[k] for k in eachindex(factor_levels)
                                     if factor_levels[k] == level)
            add("contribution_level_rms_$level",norm(level_contribution)/sqrt(dates))
        end
    end
    add("contribution_total_rms",norm(sum(contributions))/sqrt(dates))
    for series in eachindex(error_ar)
        for lag in eachindex(error_ar[series])
            add("error_ar_$(series)_$lag",error_ar[series][lag])
        end
        isempty(error_ar[series]) ||
            add("error_radius_$series",companion_radius(error_ar[series]))
    end
    return names,values
end

function sample_case(case, method, rng; ndraws, burnin, initial_factors=nothing,
                     variant=:baseline, move_rng=MersenneTwister(1),
                     scale_rng=MersenneTwister(2),
                     beta_prior_variance=100.0, ar_prior_variance=1.0,
                     variance_shape=2.0, variance_scale=1.0)
    variant in (:baseline,:location,:location_scale) ||
        throw(ArgumentError("unknown sampler variant"))
    method in ("KN1","OW1","KN2","OW2","KNHierarchical") ||
        throw(ArgumentError("unknown method"))
    spec = case.spec
    y = case.data
    dates,nseries = size(y)
    nlevels,nfactors = spec.nlevels,sum(spec.nfactors)
    factor_levels = [level for level in 1:nlevels for _ in 1:spec.nfactors[level]]
    offsets = cumsum([0;spec.nfactors[1:end-1]])
    indices = [iszero(spec.assignments[i,l]) ? 0 : spec.assignments[i,l]+offsets[l]
               for i in 1:nseries,l in 1:nlevels]
    anchors = DFM._factor_sign_anchors(indices,nfactors,nothing)
    factors = initial_factors === nothing ? DFM._initialize_factors(y,indices,nfactors) :
        Matrix{Float64}(initial_factors)
    coefficients = zeros(nseries,nlevels+1)
    variances = ones(nseries)
    factor_ar = [zeros(order) for order in case.factor_orders]
    error_ar = [zeros(order) for order in case.error_orders]
    beta_prior = DFM._regression_prior_variances(beta_prior_variance,nlevels+1,"beta_prior_variance")
    active_loadings = falses(nseries,nfactors)
    loading_prior = ones(nseries,nfactors)
    for series in 1:nseries, level in 1:nlevels
        factor = indices[series,level]
        if !iszero(factor)
            active_loadings[series,factor] = true
            loading_prior[series,factor] = beta_prior[level+1]
        end
    end
    max_order = max(maximum(case.factor_orders),maximum(case.error_orders))
    ar_prior = DFM._regression_prior_variances(ar_prior_variance,max_order,"ar_prior_variance")
    factor_draws = zeros(dates,nfactors,ndraws)
    B,S = zeros(ndraws,nseries*(nlevels+1)),zeros(ndraws,nseries)
    P,P2 = zeros(ndraws,sum(case.factor_orders)),zeros(ndraws,sum(case.error_orders))
    trace_names = String[]
    trace = zeros(0,0)
    events = Dict{Tuple{String,Int,Symbol},Int}()
    sign_flips = zeros(Int,nfactors)
    count_event(family,index,status) =
        (events[(family,index,status)] = get(events,(family,index,status),0)+1)

    for iteration in 1:(burnin+ndraws)
        for series in 1:nseries
            levels = findall(!iszero,indices[series,:])
            columns = [1;levels.+1]
            regressors = hcat(ones(dates),factors[:,indices[series,levels]])
            beta,phi,variance,status = regression_step(rng,y[:,series],regressors,
                error_ar[series],variances[series],spec.initial,beta_prior[columns],
                ar_prior[1:case.error_orders[series]],variance_shape,variance_scale)
            coefficients[series,columns] = beta
            error_ar[series],variances[series] = phi,variance
            iteration > burnin && count_event("error",series,status)
        end
        for factor in 1:nfactors
            if spec.initial == :stationary
                factor_ar[factor],status = stationary_ar_step(rng,factors[:,factor],
                    factor_ar[factor],1.0,ar_prior[1:case.factor_orders[factor]])
            else
                regressors = DFM._zero_padded_lags(factors[:,factor],case.factor_orders[factor])
                factor_ar[factor] = DFM.draw_coefficients(rng,factors[:,factor],regressors,1.0;
                    prior_variance=ar_prior[1:case.factor_orders[factor]],
                    stationary=true,max_attempts=10000)
                status = :direct
            end
            iteration > burnin && count_event("factor",factor,status)
        end
        loadings = zeros(nseries,nfactors)
        for series in 1:nseries,level in 1:nlevels
            factor = indices[series,level]
            iszero(factor) || (loadings[series,factor] = coefficients[series,level+1])
        end
        centered_y = y .- coefficients[:,1]'
        factors = if startswith(method,"KN")
            DFM._draw_factors_state_space(rng,centered_y,loadings,factor_ar,error_ar,variances;
                                          initial=spec.initial)
        elseif method == "OW1"
            DFM._draw_factors_precision(rng,centered_y,loadings,factor_ar,error_ar,variances;
                                        initial=spec.initial)
        else
            DFM._draw_factors_sequential_precision(rng,centered_y,factors,loadings,
                factor_ar,error_ar,variances; initial=spec.initial)
        end
        if variant in (:location,:location_scale)
            shift_factor_locations!(move_rng,factors,view(coefficients,:,1),loadings,factor_ar;
                initial=spec.initial,intercept_prior_variance=beta_prior[1])
        end
        if variant == :location_scale
            scale_step = ScaleMove.rescale_factors!(scale_rng,factors,loadings,factor_ar;
                active_loadings,initial=spec.initial,loading_prior_variance=loading_prior)
            # Store the same new loadings that the next sweep and traces use.
            for series in 1:nseries, level in 1:nlevels
                factor = indices[series,level]
                iszero(factor) || (coefficients[series,level+1] = loadings[series,factor])
            end
            for factor in 1:nfactors
                status = scale_step.accepted[factor] ? :accepted : :rejected
                iteration > burnin && count_event("scale",factor,status)
            end
        end
        for factor in 1:nfactors
            flipped = loadings[anchors[factor],factor] < 0
            iteration > burnin && (sign_flips[factor] += flipped)
            flipped && (loadings[:,factor] .*= -1)
        end
        DFM._identify_factor_signs!(factors,coefficients,indices,anchors)
        names,values = trace_features(factors,coefficients,loadings,factor_ar,error_ar,anchors;
                                      error_variances=variances,active_loadings,factor_levels)
        if iteration == 1
            trace_names = names
            trace = zeros(burnin+ndraws,length(names))
        end
        trace[iteration,:] = values
        if iteration > burnin
            draw = iteration-burnin
            factor_draws[:,:,draw] = factors
            B[draw,:] = vec(permutedims(coefficients))
            S[draw,:] = variances
            P[draw,:] = reduce(vcat,factor_ar;init=Float64[])
            P2[draw,:] = reduce(vcat,error_ar;init=Float64[])
        end
    end
    means = DFMMeans(dropdims(mean(factor_draws;dims=3);dims=3),mean(B;dims=1),
                     mean(S;dims=1),mean(P;dims=1),mean(P2;dims=1))
    stored_factors = spec.estimator_kind == :single ? dropdims(factor_draws;dims=2) : factor_draws
    result = DFMResults(stored_factors,B,S,P,P2,means)
    return (; result,trace_names,trace,events,sign_flips)
end

end
