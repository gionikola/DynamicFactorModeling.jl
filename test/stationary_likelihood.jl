using LinearAlgebra, Random, Statistics, Distributions

# Independent Yule-Walker equations for gamma(0),...,gamma(p). The implementation
# uses a companion-state Lyapunov sum instead, so indexing errors do not cancel.
function yule_walker_covariance(phi, n)
    p = length(phi)
    equations = Matrix{Float64}(I,p+1,p+1)
    for lag in 0:p, j in 1:p
        equations[lag+1, abs(lag-j)+1] -= phi[j]
    end
    gamma = equations \ [1.0; zeros(p)]
    while length(gamma) < n
        lag = length(gamma)
        push!(gamma, sum(phi[j] * gamma[lag-j+1] for j in 1:p; init=0.0))
    end
    return [gamma[abs(i-j)+1] for i in 1:n, j in 1:n]
end

# Independent AR(2) density, integrated over its triangular stability region.
# The stationary covariance has a closed form, so this reference uses neither
# the package's covariance solver nor its whitening and regression functions.
function ar2_posterior_moments(series, variance, prior_variances, grid_size)
    total = 0.0
    moments = zeros(5)
    for j in 1:grid_size, i in 1:grid_size
        phi2 = -1 + (2j - 1) / grid_size
        width = 1 - phi2
        phi1 = width * (-1 + (2i - 1) / grid_size)
        gamma0 = width / ((1 + phi2) * (width^2 - phi1^2))
        rho = phi1 / width
        log_density = -phi1^2 / (2prior_variances[1]) - phi2^2 / (2prior_variances[2])
        if length(series) == 1
            log_density -= (log(gamma0) + series[1]^2 / (variance * gamma0)) / 2
        else
            determinant = gamma0^2 * (1 - rho^2)
            quadratic = (series[1]^2 + series[2]^2 - 2rho * series[1] * series[2]) /
                        (gamma0 * (1 - rho^2))
            log_density -= (log(determinant) + quadratic / variance) / 2
            for t in 3:length(series)
                innovation = series[t] - phi1 * series[t-1] - phi2 * series[t-2]
                log_density -= innovation^2 / (2variance)
            end
        end
        weight = width * exp(log_density)
        total += weight
        moments += weight .* [phi1, phi2, phi1^2, phi2^2, phi1*phi2]
    end
    return moments / total
end

@testset "Stationary AR(2) updates match independent integration" begin
    variance, prior_variances = 0.7, [0.8,0.3]
    for (case, series) in enumerate(([1.3], [0.9,-0.5,0.3,1.0,-0.2]))
        reference = ar2_posterior_moments(series, variance, prior_variances, 400)
        coarser = ar2_posterior_moments(series, variance, prior_variances, 200)
        integration_error = abs.(reference - coarser)
        @test maximum(integration_error) < 1e-4
        rng = MersenneTwister(8920 + case)
        phi = zeros(2)
        statistics = zeros(60000,5)
        for step in 1:62000
            phi = DynamicFactorModeling._draw_stationary_ar(rng, series, phi, variance;
                prior_variance=prior_variances)
            if step > 2000
                statistics[step-2000,:] = [phi[1],phi[2],phi[1]^2,phi[2]^2,phi[1]*phi[2]]
            end
        end
        # Consecutive batches estimate uncertainty without treating correlated
        # draws as independent. Require both agreement and useful precision.
        batch_means = dropdims(mean(reshape(statistics,200,300,5);dims=1);dims=1)
        standard_errors = vec(std(batch_means;dims=1)) / sqrt(300)
        @test maximum(standard_errors) < 0.01
        @test all(abs.(vec(mean(statistics;dims=1)) - reference) .<
                  6standard_errors + 2integration_error)
    end
end

@testset "Stationary likelihood retains the initial density" begin
    dfm = DynamicFactorModeling
    for phi in (Float64[], [0.8], [0.6,-0.2], [0.4,0.1,-0.1,0.05]), n in (1,2,7)
        expected = yule_walker_covariance(phi,n)
        @test dfm._stationary_ar_covariance(phi,n) ≈ expected atol=1e-11
        transform = dfm._ar_innovation_matrix(phi,n; initial=:stationary)
        @test transform * expected * transform' ≈ Matrix{Float64}(I,n,n) atol=1e-11
        sample = collect(1.0:n) ./ n .- 0.4
        q = min(length(phi),n)
        if q > 0
            reference = MvNormal(zeros(q), 1.3expected[1:q,1:q])
            without_constant = logpdf(reference,sample[1:q]) + q/2*log(2pi*1.3)
            @test dfm._stationary_initial_logdensity(sample,phi,1.3) ≈ without_constant atol=1e-10
        end
    end
    @test_throws ArgumentError dfm._stationary_ar_covariance([1.0],3)

    # At T=1 the AR(1) posterior is NOT its truncated-normal prior: its first
    # value already carries information through stationary variance 1/(1-phi²).
    grid = collect(range(-0.99999,0.99999; length=20001))
    observation, variance, prior_variance = 2.0, 0.7, 0.6
    density = exp.(-grid.^2/(2prior_variance)) .* sqrt.(1 .- grid.^2) .*
              exp.(-(1 .- grid.^2)*observation^2/(2variance))
    expected_square = sum(grid.^2 .* density) / sum(density)
    rng = MersenneTwister(1301)
    phi = [0.0]
    squares = zeros(60000)
    for step in 1:61000
        phi = dfm._draw_stationary_ar(rng,[observation],phi,variance; prior_variance)
        step > 1000 && (squares[step-1000] = only(phi)^2)
    end
    @test mean(squares) ≈ expected_square atol=0.012
    prior = truncated(Normal(0,sqrt(prior_variance)),-1,1)
    @test abs(expected_square - var(prior)) > 0.08

    # The likelihood contribution of a nonzero initial value distinguishes
    # stationary and fixed-zero models even when they use identical AR params.
    @test dfm._ar_innovation_matrix([0.8],1; initial=:stationary)[1,1] ≈ 0.6
    @test dfm._ar_innovation_matrix([0.8],1; initial=:zero)[1,1] == 1
end

@testset "Stationary factor samplers match an independent Gaussian model" begin
    dfm = DynamicFactorModeling
    n, k = 4, 2
    loadings = [1.0 0.3; 0.2 -0.6; 0.0 1.2]
    factor_ar = [[0.7,-0.2], [0.4]]
    error_ar = [[0.3], Float64[], [0.3,0.1,-0.1,0.05,0.02]]
    variances = [0.4,0.7,0.5]
    y = [0.2 1.0 -0.3; -0.4 0.3 0.9; 0.5 -0.1 0.4; 0.7 -0.2 0.1]
    prior = zeros(n*k,n*k)
    noise = zeros(3n,3n)
    for factor in 1:k
        rows = ((factor-1)*n+1):(factor*n)
        prior[rows,rows] = yule_walker_covariance(factor_ar[factor],n)
    end
    for series in 1:3
        rows = ((series-1)*n+1):(series*n)
        noise[rows,rows] = variances[series] * yule_walker_covariance(error_ar[series],n)
    end
    observation = kron(loadings,Matrix{Float64}(I,n,n))
    data_covariance = observation*prior*observation' + noise
    gain = (prior*observation') / data_covariance
    expected_mean = gain*vec(y)
    expected_covariance = prior - gain*observation*prior
    precision, linear = dfm._factor_precision(y,loadings,factor_ar,error_ar,variances; initial=:stationary)
    @test precision \ linear ≈ expected_mean atol=1e-11
    @test inv(precision) ≈ expected_covariance atol=1e-11
    state_model, columns = dfm._factor_state_space(loadings,factor_ar,error_ar,variances)
    _, smoothed, covariance = kalmanSmoother(y,state_model)
    @test vec(smoothed[:,columns]) ≈ expected_mean atol=1e-11
    for t in 1:n
        indices = [t,n+t]
        @test covariance[t][columns,columns] ≈ expected_covariance[indices,indices] atol=1e-10
    end
    for sampler in (dfm._draw_factors_state_space, dfm._draw_factors_precision)
        rng = MersenneTwister(782)
        draws = hcat([vec(sampler(rng,y,loadings,factor_ar,error_ar,variances;
                          initial=:stationary)) for _ in 1:3000]...)
        @test maximum(abs.(vec(mean(draws;dims=2)) - expected_mean)) < 0.06
        @test maximum(abs.(cov(draws;dims=2) - expected_covariance)) < 0.055
    end
    # Start independent replicates in the exact joint posterior. A valid
    # sequential Gibbs sweep must leave that distribution invariant.
    rng = MersenneTwister(1772)
    reference = MvNormal(expected_mean,Symmetric(expected_covariance))
    sequential_draws = zeros(n*k,4000)
    for draw in axes(sequential_draws,2)
        current = reshape(rand(rng,reference),n,k)
        sequential_draws[:,draw] = vec(dfm._draw_factors_sequential_precision(
            rng,y,current,loadings,factor_ar,error_ar,variances; initial=:stationary))
    end
    @test maximum(abs.(vec(mean(sequential_draws;dims=2)) - expected_mean)) < 0.05
    @test maximum(abs.(cov(sequential_draws;dims=2) - expected_covariance)) < 0.05
    @test_throws ArgumentError KN1LevelEstimator(y, DFMStruct(1,1,2,0); stationary=false)
    @test_throws ArgumentError KN1LevelEstimator(y, DFMStruct(1,1,2,0); initial=:invalid)
    @test_throws ArgumentError dfm.draw_parameters(y[:,1],ones(n,1),[0.2],1.0;
        initial=:stationary,stationary=false)

    # Adding an independent error state must not change whether a factor AR
    # coefficient is accepted as stable. Both conditional samplers use the
    # same prior even close to the numerical stationarity boundary.
    near_unit = [1 - 1e-14]
    @test dfm.isstationary(near_unit)
    for sampler in (dfm._draw_factors_state_space, dfm._draw_factors_precision)
        draw = sampler(MersenneTwister(71),reshape([0.2],1,1),ones(1,1),
            [near_unit],[Float64[]],[1.0];initial=:stationary)
        @test all(isfinite,draw)
    end
end
