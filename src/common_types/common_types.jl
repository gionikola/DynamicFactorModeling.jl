"""
    SSModel(; H, A, F, μ, R, Q, Z)

Linear Gaussian state-space model:
`y[t] = H * β[t] + A * z[t] + e[t]`,
`β[t] = μ + F * β[t-1] + v[t]`.

`R`, `Q`, and `Z` are the covariance matrices of mutually independent
measurement errors, state innovations, and simulated regressors. Rows of data
matrices are times. Covariances may be singular. For no regressors use an
`nseries × 0` matrix `A` and a `0 × 0` matrix `Z`.
"""
struct SSModel
    H::Matrix{Float64}
    A::Matrix{Float64}
    F::Matrix{Float64}
    μ::Vector{Float64}
    R::Matrix{Float64}
    Q::Matrix{Float64}
    Z::Matrix{Float64}

    function SSModel(H, A, F, μ, R, Q, Z)
        model = new(Matrix{Float64}(H), Matrix{Float64}(A), Matrix{Float64}(F),
                    Vector{Float64}(μ), Matrix{Float64}(R), Matrix{Float64}(Q),
                    Matrix{Float64}(Z))
        validate_ssmodel(model)
        return model
    end
end
SSModel(; H, A, F, μ, R, Q, Z) = SSModel(H, A, F, μ, R, Q, Z)

"""
    HDFM(; nlevels, nvar, nfactors, fassign, flags, varlags,
           varcoefs, varlagcoefs, fcoefs, fvars, varvars)

Specify a hierarchical dynamic factor model for simulation. Each series has
an intercept, at most one assigned factor per level, and an autoregressive
error. Factors have independent autoregressive dynamics.

- `nfactors[l]` and `flags[l]`: factor count and AR order at level `l`.
- `fassign[i,l]`: factor number within level `l`; zero means no factor.
- `varcoefs[i,:]`: intercept followed by one loading per level.
- `varlags[i]`: error AR order; `varlagcoefs[i,1:varlags[i]]`: coefficients.
- `fcoefs[l]`: `nfactors[l] × flags[l]` matrix of factor AR coefficients.
- `fvars[l]`: factor innovation variances; `varvars`: error innovation variances.

AR order zero is allowed. Entries beyond a series' `varlags` are ignored.
Variances must be nonnegative. Stationarity is checked when a stationary
initial distribution is requested, rather than when constructing the model.
"""
struct HDFM
    nlevels::Int
    nvar::Int
    nfactors::Vector{Int}
    fassign::Matrix{Int}
    flags::Vector{Int}
    varlags::Vector{Int}
    varcoefs::Matrix{Float64}
    varlagcoefs::Matrix{Float64}
    fcoefs::Vector{Matrix{Float64}}
    fvars::Vector{Vector{Float64}}
    varvars::Vector{Float64}

    function HDFM(nlevels, nvar, nfactors, fassign, flags, varlags,
                  varcoefs, varlagcoefs, fcoefs, fvars, varvars)
        model = new(Int(nlevels), Int(nvar), Vector{Int}(nfactors),
                    Matrix{Int}(fassign), Vector{Int}(flags), Vector{Int}(varlags),
                    Matrix{Float64}(varcoefs), Matrix{Float64}(varlagcoefs),
                    [Matrix{Float64}(c) for c in fcoefs],
                    [Vector{Float64}(v) for v in fvars], Vector{Float64}(varvars))
        validate_hdfm(model)
        return model
    end
end
HDFM(; nlevels, nvar, nfactors, fassign, flags, varlags, varcoefs,
       varlagcoefs, fcoefs, fvars, varvars) =
    HDFM(nlevels, nvar, nfactors, fassign, flags, varlags,
         varcoefs, varlagcoefs, fcoefs, fvars, varvars)

"""
    DFMStruct(; factorlags, errorlags, ndraws=1000, burnin=500)

Single-factor estimation settings. AR orders may be zero. `ndraws` is the
number of retained Gibbs draws; `burnin` is the number discarded beforehand.
The estimator performs `burnin + ndraws` iterations.
"""
struct DFMStruct
    factorlags::Int
    errorlags::Int
    ndraws::Int
    burnin::Int

    function DFMStruct(factorlags, errorlags, ndraws, burnin)
        factorlags >= 0 || throw(ArgumentError("factorlags must be nonnegative"))
        errorlags >= 0 || throw(ArgumentError("errorlags must be nonnegative"))
        ndraws > 0 || throw(ArgumentError("ndraws must be positive"))
        burnin >= 0 || throw(ArgumentError("burnin must be nonnegative"))
        return new(Int(factorlags), Int(errorlags), Int(ndraws), Int(burnin))
    end
end
DFMStruct(; factorlags, errorlags, ndraws=1000, burnin=500) =
    DFMStruct(factorlags, errorlags, ndraws, burnin)

"""
    HDFMStruct(; nlevels, nfactors, factorassign, factorlags, errorlags,
                ndraws=1000, burnin=500)

Hierarchical estimation settings. `factorassign[i,l]` is a factor number
within level `l`, or zero for no factor. `factorlags[l]` is the AR order for
that level; `errorlags[i]` is the AR order for series `i`. `ndraws` counts
retained draws and `burnin` counts additional discarded iterations.
"""
struct HDFMStruct
    nlevels::Int
    nfactors::Vector{Int}
    factorassign::Matrix{Int}
    factorlags::Vector{Int}
    errorlags::Vector{Int}
    ndraws::Int
    burnin::Int

    function HDFMStruct(nlevels, nfactors, factorassign, factorlags, errorlags,
                        ndraws, burnin)
        nlevels > 0 || throw(ArgumentError("nlevels must be positive"))
        length(nfactors) == length(factorlags) == nlevels ||
            throw(DimensionMismatch("nfactors and factorlags must have nlevels entries"))
        size(factorassign) == (length(errorlags), nlevels) ||
            throw(DimensionMismatch("factorassign must be nseries × nlevels"))
        !isempty(errorlags) || throw(ArgumentError("at least one series is required"))
        all(>(0), nfactors) || throw(ArgumentError("each level needs at least one factor"))
        all(>=(0), factorlags) && all(>=(0), errorlags) ||
            throw(ArgumentError("AR orders must be nonnegative"))
        for l in 1:nlevels
            all(x -> 0 <= x <= nfactors[l], factorassign[:,l]) ||
                throw(ArgumentError("factor assignment is outside its level's range"))
            for k in 1:nfactors[l]
                any(==(k), factorassign[:,l]) ||
                    throw(ArgumentError("every factor must be assigned to at least one series"))
            end
        end
        ndraws > 0 || throw(ArgumentError("ndraws must be positive"))
        burnin >= 0 || throw(ArgumentError("burnin must be nonnegative"))
        return new(Int(nlevels), Vector{Int}(nfactors), Matrix{Int}(factorassign),
                   Vector{Int}(factorlags), Vector{Int}(errorlags), Int(ndraws), Int(burnin))
    end
end
HDFMStruct(; nlevels, nfactors, factorassign, factorlags, errorlags,
             ndraws=1000, burnin=500) =
    HDFMStruct(nlevels, nfactors, factorassign, factorlags, errorlags, ndraws, burnin)

"""
    DFMMeans(F, B, S, P, P2)

Posterior means: factors `F`, observation coefficients `B`, error innovation
variances `S`, factor AR coefficients `P`, and error AR coefficients `P2`.
Observation coefficients are stored series by series (intercept, then level
loadings). AR coefficients are stored process by process, in lag order.
"""
struct DFMMeans
    F::Array{Float64}
    B::Array{Float64}
    S::Array{Float64}
    P::Array{Float64}
    P2::Array{Float64}
end
DFMMeans(; F, B, S, P, P2) = DFMMeans(F, B, S, P, P2)

"""
    DFMResults(F, B, S, P, P2, means)

Retained Gibbs draws and their [`DFMMeans`](@ref). `F` is time × draw for a
single-factor estimator, and time × factor × draw for a hierarchical one.
`B`, `S`, `P`, and `P2` have draws in rows. Factor order is level first,
then factor within level; AR coefficients omit padding for unused lags.
Samples are Monte Carlo output: assess mixing across several chains before
using posterior summaries.
"""
struct DFMResults
    F::Array{Float64}
    B::Array{Float64}
    S::Array{Float64}
    P::Array{Float64}
    P2::Array{Float64}
    means::DFMMeans
end
DFMResults(; F, B, S, P, P2, means) = DFMResults(F, B, S, P, P2, means)
