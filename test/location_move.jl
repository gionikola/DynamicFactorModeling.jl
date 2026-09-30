using Test, Random, LinearAlgebra
if !isdefined(@__MODULE__, :LocationMove)
    include(joinpath(@__DIR__, "support", "location_move.jl"))
end
using .LocationMove
if !isdefined(@__MODULE__, :GaussianLocationReference)
    include(joinpath(@__DIR__, "reference", "gaussian_location_reference.jl"))
end
using .GaussianLocationReference

# Independent closed-form stationary covariances through AR(2), and causal
# impulse responses for the zero-presample model. No package AR helper is used.
function location_test_covariance(coefficients, dates, initial)
    if initial == :zero
        responses = zeros(dates, dates)
        for innovation in 1:dates
            responses[innovation, innovation] = 1.0
            for t in (innovation + 1):dates
                responses[t, innovation] = sum(
                    coefficients[lag] * responses[t - lag, innovation]
                    for lag in 1:min(length(coefficients), t - innovation); init=0.0)
            end
        end
        return responses * responses'
    end
    isempty(coefficients) && return Matrix{Float64}(I, dates, dates)
    if length(coefficients) == 1
        coefficient = only(coefficients)
        return [coefficient^abs(t - s) / (1 - coefficient^2)
                for t in 1:dates, s in 1:dates]
    end
    first, second = coefficients
    covariances = zeros(max(dates, 2))
    covariances[1] = (1 - second) / ((1 + second) * ((1 - second)^2 - first^2))
    covariances[2] = first * covariances[1] / (1 - second)
    for lag in 2:(dates - 1)
        covariances[lag + 1] = first * covariances[lag] + second * covariances[lag - 1]
    end
    return [covariances[abs(t - s) + 1] for t in 1:dates, s in 1:dates]
end

function location_test_system(factors, intercepts, loadings, coefficients, variances, initial)
    dates, count = size(factors)
    precisions = [inv(Symmetric(location_test_covariance(phi, dates, initial)))
                  for phi in coefficients]
    ones_vector = ones(dates)
    diagonal = [dot(ones_vector, Q * ones_vector) for Q in precisions]
    precision = Diagonal(diagonal) + loadings' * Diagonal(1 ./ variances) * loadings
    score = loadings' * (intercepts ./ variances) -
            [dot(ones_vector, precisions[k] * factors[:, k]) for k in 1:count]
    return (; precision, score, precisions)
end

function location_test_logprior(factors, intercepts, precisions, variances)
    return -sum(intercepts .^ 2 ./ variances) / 2 -
           sum(dot(factors[:, k], precisions[k] * factors[:, k]) for k in axes(factors, 2)) / 2
end

@testset "Exact factor-location conditional and signal invariance" begin
    coefficients = [Float64[], [0.85], [0.6, -0.25]]
    loadings = [1.0 0.6 0.0; -0.4 0.0 1.2; 0.0 -0.5 0.0; 0.0 0.0 0.0]
    for initial in (:zero, :stationary), dates in (1, 2, 8),
        prior in (100.0, [0.5, 2.0, 3.0, 10.0])
        factors = reshape(sin.(collect(1.0:(3 * dates))), dates, 3)
        intercepts = [0.3, -0.7, 1.2, 0.1]
        variances = prior isa Real ? fill(prior, 4) : prior
        system = location_test_system(factors, intercepts, loadings, coefficients, variances, initial)
        root = cholesky(Symmetric(system.precision))
        expected = root \ system.score + root.U \ randn(MersenneTwister(118), 3)
        original_factors, original_intercepts = copy(factors), copy(intercepts)
        original_signal = factors * loadings' .+ intercepts'
        delta = shift_factor_locations!(MersenneTwister(118), factors, intercepts,
                                        loadings, coefficients; initial,
                                        intercept_prior_variance=prior)
        @test delta ≈ expected atol=2e-13 rtol=2e-13
        @test factors ≈ original_factors .+ delta'
        @test intercepts ≈ original_intercepts - loadings * delta
        @test factors * loadings' .+ intercepts' ≈ original_signal atol=2e-13
        @test intercepts[4] == original_intercepts[4] # Series with no factor loading.

        # The entire initial-state density is included, including T < AR order.
        old_density = location_test_logprior(original_factors, original_intercepts,
                                            system.precisions, variances)
        new_density = location_test_logprior(factors, intercepts, system.precisions, variances)
        @test new_density - old_density ≈ dot(system.score, delta) -
              dot(delta, system.precision * delta) / 2 atol=2e-12

        # Starting elsewhere on the same likelihood-preserving line and using
        # the same random normal draw must produce the same final state.
        offset = [0.4, -0.8, 0.2]
        other_factors = original_factors .+ offset'
        other_intercepts = original_intercepts - loadings * offset
        other_delta = shift_factor_locations!(other_factors, other_intercepts,
            loadings, coefficients; rng=MersenneTwister(118), initial,
            intercept_prior_variance=prior)
        @test other_delta ≈ delta - offset atol=2e-13
        @test other_factors ≈ factors atol=2e-13
        @test other_intercepts ≈ intercepts atol=2e-13
    end
end

@testset "Joint posterior preserved by the implemented location move" begin
    loadings = [1.2 0.4; -0.7 0.0; 0.0 0.8]
    factor_ar = [[0.88], [0.7, -0.22]]
    error_ar = [[0.5], Float64[], [0.4, -0.15]]
    prior = [100.0, 2.0, 10.0]
    for initial in (:zero, :stationary), dates in (1, 4)
        data = reshape(cos.(collect(1.0:(3 * dates))), dates, 3)
        reference = gaussian_location_posterior(data, loadings, factor_ar, error_ar,
            [0.4, 0.7, 0.9]; initial, intercept_prior_variance=prior)
        dimension = length(reference.mean)
        direction = vcat(-loadings, kron(Matrix{Float64}(I, 2, 2), ones(dates)))
        precision = Symmetric(direction' * reference.precision * direction)
        covariance = inv(precision)

        # Recover the implemented affine map x_new = A*x + noise by keeping
        # the random normal draw fixed and applying the update to each basis
        # vector. This tests the real mutating routine, not just its formula.
        function apply_move(values)
            intercepts = copy(values[1:3])
            factors = reshape(copy(values[4:end]), dates, 2)
            shift_factor_locations!(MersenneTwister(765), factors, intercepts,
                loadings, factor_ar; initial, intercept_prior_variance=prior)
            return [intercepts; vec(factors)]
        end
        noise = apply_move(zeros(dimension))
        identity = Matrix{Float64}(I, dimension, dimension)
        transition = hcat([apply_move(identity[:, column]) - noise
                           for column in 1:dimension]...)
        expected_transition = identity - direction *
            (precision \ (direction' * reference.precision))
        @test transition ≈ expected_transition atol=5e-13
        @test noise ≈ direction * (cholesky(precision).U \ randn(MersenneTwister(765), 2)) atol=5e-13

        # A linear Gaussian update preserves this FULL posterior exactly if
        # its resulting mean and covariance equal the independent reference.
        @test transition * reference.mean ≈ reference.mean atol=2e-12
        @test transition * reference.covariance * transition' +
              direction * covariance * direction' ≈ reference.covariance atol=2e-12
        @test direction' * reference.information ≈ zeros(2) atol=5e-13

        before = sin.(collect(1.0:dimension))
        after = apply_move(before)
        delta = reshape(after[4:end] - before[4:end], dates, 2)[1, :]
        score = direction' * (reference.information - reference.precision * before)
        density_change = joint_logdensity(reference, after[1:3], reshape(after[4:end], dates, 2)) -
                         joint_logdensity(reference, before[1:3], reshape(before[4:end], dates, 2))
        @test density_change ≈ dot(score, delta) - dot(delta, precision * delta) / 2 atol=2e-12
    end
end

@testset "Location-coordinate Jacobian" begin
    loadings = [1.0 0.3; -0.4 0.9; 0.0 0.0]
    series, count = size(loadings)
    for dates in (1, 2, 7)
        # Input [a; vec(F)], output [b; vec(G[1:T-1,:]); m].
        transformation = zeros(series + dates * count, series + dates * count)
        transformation[1:series, 1:series] = Matrix{Float64}(I, series, series)
        for factor in 1:count
            last = series + dates * factor
            transformation[1:series, last] = loadings[:, factor]
            for t in 1:(dates - 1)
                row = series + (dates - 1) * (factor - 1) + t
                transformation[row, series + dates * (factor - 1) + t] = 1
                transformation[row, last] = -1
            end
            transformation[series + (dates - 1) * count + factor, last] = 1
        end
        @test abs(det(transformation)) ≈ 1.0 atol=1e-14
    end
end

@testset "Location move input validation" begin
    factors = reshape([0.2, -0.5], :, 1)
    intercepts, loadings, coefficients = [0.3], ones(1, 1), [[0.6]]
    for invalid in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError shift_factor_locations!(copy(factors), copy(intercepts),
            loadings, coefficients; intercept_prior_variance=invalid)
    end
    @test_throws DimensionMismatch shift_factor_locations!(factors, intercepts, zeros(2, 1), coefficients)
    @test_throws DimensionMismatch shift_factor_locations!(factors, intercepts, loadings, [Float64[], Float64[]])
    @test_throws DimensionMismatch shift_factor_locations!(factors, intercepts, loadings,
        coefficients; intercept_prior_variance=[1.0, 2.0])
    @test_throws ArgumentError shift_factor_locations!(factors, intercepts, loadings, coefficients; initial=:conditional)
    @test_throws ArgumentError shift_factor_locations!(factors, intercepts, loadings, [[1.0]])
    @test_throws ArgumentError shift_factor_locations!(factors, intercepts, loadings, [[NaN]])
    @test_throws ArgumentError shift_factor_locations!(ones(Int, 2, 1), intercepts, loadings, coefficients)
    @test_throws ArgumentError shift_factor_locations!(zeros(0, 1), intercepts, loadings, coefficients)
    # Invalid arguments are rejected before either supplied array is changed.
    @test factors == reshape([0.2, -0.5], :, 1)
    @test intercepts == [0.3]
end
