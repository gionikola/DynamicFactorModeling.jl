"""
    draw_error_variance([rng], Y, X, β; prior_shape=0.5, prior_scale=0.05)

Draw the error variance in `Y = X * β + error`, conditional on `β`. With an
`InverseGamma(prior_shape, prior_scale)` prior, the posterior has shape
`prior_shape + length(Y)/2` and scale `prior_scale + sum(abs2, Y - X*β)/2`.
The inverse-gamma density is proportional to `v^(-shape-1) * exp(-scale/v)`.

`Y` is a vector or a single-column matrix. `X` may be a vector for one
coefficient; otherwise its columns match the entries of `β`. Pass an RNG as
the first argument or with `rng=`.
"""
function draw_error_variance(rng::AbstractRNG,
                             Y::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             X::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                             β::Union{Real,AbstractVector{<:Real}};
                             prior_shape::Real=0.5, prior_scale::Real=0.05)
    y, x = _regression_data(Y, X)
    coefficients = β isa Real ? [Float64(β)] : Float64.(β)
    length(coefficients) == size(x, 2) ||
        throw(DimensionMismatch("β must have one entry per column of X"))
    all(isfinite, coefficients) || throw(ArgumentError("β must be finite"))
    shape = _positive_regression_parameter(prior_shape, "prior_shape")
    scale = _positive_regression_parameter(prior_scale, "prior_scale")
    residuals = y - x * coefficients
    return rand(rng, InverseGamma(shape + length(y) / 2, scale + sum(abs2, residuals) / 2))
end

draw_error_variance(Y, X, β; rng::AbstractRNG=Random.default_rng(), kwargs...) =
    draw_error_variance(rng, Y, X, β; kwargs...)
