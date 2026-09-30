module SimulationDiagnostics

using FFTW
using MCMCDiagnosticTools
using Statistics

export chain_diagnostics

"""
    chain_diagnostics(draws::AbstractMatrix)

Summarize one scalar parameter from equally long chains. Rows are retained draws;
columns are independent chains. Remove warmup before calling this function.
At least two chains and ten draws per chain are required.

The returned fields are:

- `rhat`: the larger of rank-normalized split R-hat and folded split R-hat.
  The first detects differences in location; the second also detects differences
  in spread.
- `bulk_ess`: effective sample size of the rank-normalized split chains.
- `tail_ess`: the smaller effective sample size for the 5% and 95% quantiles.
- `mcse_mean`: Monte Carlo standard error of the pooled mean, using the effective
  sample size of the original values, rather than their ranks.

MCMCDiagnosticTools computes these statistics using Geyer's positive, monotone
autocorrelation sequence. FFT autocovariances and all available split-chain lags
avoid a fixed lag cutoff for slowly mixing chains. A direct autocovariance
calculation handles constant tail indicators if the FFT method is undefined.
With an odd number of draws, the library drops the middle draw when splitting
each chain.

If any original chain is constant, all four results are `NaN`: an unmoving chain
cannot establish precision for a stochastic parameter. Exclude known fixed model
entries before calling this function. Other undefined library results also remain
nonfinite. A small R-hat and adequate effective sample sizes are diagnostic checks,
not a guarantee of convergence or a finite-variance posterior mean.

References:
- Vehtari et al. (2021): https://arxiv.org/abs/1903.08008
- MCMCDiagnosticTools: https://turinglang.org/MCMCDiagnosticTools.jl/stable/
"""
function chain_diagnostics(draws::AbstractMatrix{<:Real})
    n, chains = size(draws)
    n >= 10 || throw(ArgumentError("at least ten retained draws per chain are required"))
    chains >= 2 || throw(ArgumentError("at least two independent chains are required"))
    values = Matrix{Float64}(draws)
    all(isfinite, values) || throw(ArgumentError("draws must be finite Float64 values"))

    if any(chain -> all(==(first(chain)), chain), eachcol(values))
        return (rhat=NaN, bulk_ess=NaN, tail_ess=NaN, mcse_mean=NaN)
    end

    result = _library_diagnostics(values, MCMCDiagnosticTools.FFTAutocovMethod())
    all(isfinite, result) && return result

    # The FFT backend divides each chain's autocovariances by its lag-zero
    # value. A constant split chain, including a 0/1 tail indicator, gives 0/0.
    # The direct backend handles that case without changing the ESS definition.
    return _library_diagnostics(values, MCMCDiagnosticTools.AutocovMethod())
end

function _library_diagnostics(values, autocov_method)
    settings = (split_chains=2, maxlag=size(values, 1) ÷ 2 - 4, autocov_method)
    bulk = MCMCDiagnosticTools.ess_rhat(values; kind=:rank, settings...)
    tail = MCMCDiagnosticTools.ess(values; kind=:tail, tail_prob=0.1, settings...)
    mean_error = MCMCDiagnosticTools.mcse(values; kind=mean, settings...)
    return (rhat=bulk.rhat, bulk_ess=bulk.ess, tail_ess=tail,
            mcse_mean=only(mean_error))
end

end # module
