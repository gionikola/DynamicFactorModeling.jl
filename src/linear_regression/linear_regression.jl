include("draw_coefficients.jl")
include("draw_error_variance.jl")
include("stationary.jl")
include("draw_parameters.jl")

"""
    regress([rng], Y, X, p, iter, burnin; kwargs...)

Sample a Bayesian regression whose errors have `p` autoregressive lags. `iter`
is the total number of Gibbs steps and `burnin` is the number discarded.
Return coefficient draws, AR coefficient draws, and innovation variance draws,
with sizes `(iter-burnin, size(X,2))`, `(iter-burnin, p)`, and `(iter-burnin,)`.
A vector `X` is treated as a one-column design; no intercept is added.

The chain starts with zero AR coefficients and variance one. See
[`draw_parameters`](@ref) for priors, stationary draws, and the `initial`
convention. Additional keywords are passed to that function. Use `p=0` for
independent errors. Draws from a single chain are correlated; assess convergence
and sensitivity to priors before interpreting them. An RNG may be supplied as
the first argument or with `rng=`.
"""
function regress(rng::AbstractRNG, Y, X, p::Integer, iter::Integer, burnin::Integer;
                 kwargs...)
    y, x = _regression_data(Y, X)
    p >= 0 || throw(ArgumentError("p must be nonnegative"))
    iter > 0 || throw(ArgumentError("iter must be positive"))
    0 <= burnin < iter || throw(ArgumentError("burnin must satisfy 0 ≤ burnin < iter"))
    retained = iter - burnin
    βsave = Matrix{Float64}(undef, retained, size(x, 2))
    ϕsave = Matrix{Float64}(undef, retained, p)
    σ2save = Vector{Float64}(undef, retained)
    ϕ = zeros(p)
    σ2 = 1.0

    for step in 1:iter
        β, ϕ, σ2 = draw_parameters(rng, y, x, ϕ, σ2; kwargs...)
        if step > burnin
            row = step - burnin
            βsave[row, :] = β
            ϕsave[row, :] = ϕ
            σ2save[row] = σ2
        end
    end
    return βsave, ϕsave, σ2save
end

regress(Y, X, p::Integer, iter::Integer, burnin::Integer;
        rng::AbstractRNG=Random.default_rng(), kwargs...) =
    regress(rng, Y, X, p, iter, burnin; kwargs...)
