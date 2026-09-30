# Model and estimation

## Equations

For observed series ``i`` at time ``t``, the model is

```math
y_{it} = a_i + \sum_{l=1}^{L} b_{il} f_{g(i,l),t} + e_{it},
```

where ``g(i,l)`` is the overall index of the assigned factor. Factors are
ordered by level, then by number within level. The assignment fields are
`fassign` in `HDFM` for simulation and `factorassign` in `HDFMStruct` for
estimation. Both count within each level; add the number of factors in earlier
levels to obtain the overall index. An assignment of zero omits that contribution.
Factors and errors each follow their own autoregression, a linear
combination of their earlier values plus a new random innovation:

```math
f_{kt} = \sum_{j=1}^{p_k}\psi_{kj}f_{k,t-j} + u_{kt}, \qquad
e_{it} = \sum_{j=1}^{q_i}\phi_{ij}e_{i,t-j} + v_{it}.
```

All innovations are mutually independent and Gaussian. There are no cross-factor
lag effects. An AR order of zero means independent draws over time. The hierarchy
specifies which series share factors; it does not introduce regression links
between factors at different levels.

## Bayesian likelihood and priors

Estimation defaults to stationary initialization: the factors and errors already
follow their long-run distributions at the start of the sample. The likelihood
includes the first observations and their parameter-dependent stationary
covariance. Simulation uses the same convention by default.

Alternatively, `initial=:zero` fixes all factors and errors before time 1 to
zero. Match that model in simulation with `initial_state=zeros(size(ss.F,1))`.
The deterministic intercept state becomes one at the first transition in either
case. These are different likelihoods; the option applies to every update,
not just the factor sampler.

Factor innovation variances are fixed to one to set the factor scale. Defaults:

| Parameter | Prior |
|---|---|
| Intercepts and active loadings | Independent ``N(0,100)`` |
| Each process's AR coefficients | ``N(0,I)`` restricted to stationary coefficients |
| Each error innovation variance | Inverse gamma with shape 2 and scale 1 |

The inverse-gamma density is proportional to
``x^{-\alpha-1}\exp(-\beta/x)``. The coefficient priors are independent of the
error variance. Estimator keywords allow changing these prior scales. A scalar
`beta_prior_variance` applies to all coefficients; a vector specifies the
intercept followed by the loading at each level. For example, `[1,10,10]`
sets a tighter intercept prior in a two-level model. An `ar_prior_variance`
vector gives variances by lag, up to the largest factor or error order; shorter
processes use its first entries. For example, `[1,0.5,0.25]` progressively
shrinks lag 2 and lag 3 toward zero. All variances must be strictly positive.

Each sweep updates the observation coefficients, error AR coefficients, error
variances, factor AR coefficients, and factor paths given the other quantities.
Coefficients, variances, and factor paths have standard conditional draws.
Under stationary initialization, the AR conditional also includes the initial
observations' density. The sampler proposes AR coefficients from the Gaussian
transition-regression posterior, then accepts or rejects them using that initial
density. An unstable or rejected proposal correctly retains the old coefficients.
This is a Metropolis-Hastings step within Gibbs sampling.

Under `initial=:zero`, that extra density is absent and AR updates are Gaussian
draws restricted to the stable region when `stationary=true`. Setting
`stationary=false` allows unrestricted AR coefficients for this finite sample;
it requires `initial=:zero`. If rejection sampling exceeds
`max_attempts`, it raises an error. Stationarity checks reject roots
indistinguishable from a unit root at floating-point precision.

By default, `mixing_moves=:location_scale` adds two updates after drawing the
factor paths. First, it shifts factor levels and offsets those shifts in the
series intercepts. Second, it proposes a factor rescaling with the inverse
change in its loadings. Both leave the fitted signal unchanged. The first is
an exact conditional draw; the second accepts or rejects its proposal using
the factor and loading priors and the change of variables. The posterior,
including unit factor innovation variances, stays the same.

These updates help when intercepts and factor levels, or loadings and factor
sizes, otherwise move slowly together. `mixing_moves=:location` uses only the
level update; `:none` disables both. `scale_step=0.1` sets the fixed standard
deviation of the log-scale proposal. It is not tuned automatically. The KN/OW
names still describe the factor-path engines; these additional updates are
package extensions. They improve the tested examples but do not guarantee
adequate sampling of every hierarchical decomposition.

The standalone `regress` utility instead defaults to a likelihood conditional
on the first `p` observed rows (`initial=:conditional`). Its `initial=:zero`
option uses every row and zero presample errors, while `initial=:stationary`
uses the full stationary likelihood and Metropolis correction. Its prior
defaults also differ from the DFM estimators; see its docstring when using it
directly.

## Factor samplers

`KN1LevelEstimator`, `KN2LevelEstimator`, and `KNHierarchicalEstimator` use a
Kalman filter followed by backward sampling of the whole state path. Singular
process covariance is expected: lagged states are deterministic shifts.

`OW1LevelEstimator` draws the whole factor path from its Gaussian precision
matrix (the inverse covariance). `OW2LevelEstimator` updates one complete factor
path at a time, conditioning on the current other factors. Each dense matrix has
one row per observation. Storage grows quadratically in sample length and dense
factorization grows cubically. Both approaches target the same posterior as KN.
The state-space approach normally avoids a matrix spanning all dates, apart
from the bounded [numerical fallback](@ref "State-space tools"). Its state
dimension grows with the number of series and their error AR orders. Direct
factor draws can therefore be useful for many series with a short time span.

Setting `factor_sampler=:precision` draws all factors jointly from a larger
dense matrix indexed by time and factor. `:sequential_precision` selects the
per-factor update; `:state_space` selects the joint recursive method. The
stationary initial likelihood is included consistently in all three.

The [method audit](@ref "Method audit") records the source equations, author
code, and algorithm choices. Defaults use proper, explicit priors. They do not
automatically reproduce every empirical prior, normalization, or data treatment
in the referenced applications.

## Signs, scale, and identification

By default, after each sweep the first assigned series' loading for each factor is made
nonnegative by flipping that factor and all its loadings together. This leaves
fitted values unchanged and is valid because coefficient priors are symmetric.
Use `sign_anchors` to choose another assigned series for each factor.
Scale is set by unit factor innovation variances.

These conventions do not ensure that every factor is separately identified.
Factors with identical series assignments and similar dynamics may exchange
roles or admit rotations. Weak anchor loadings can also make summaries unstable.
Use a structure supported by the data, compare multiple chains, and interpret
shared fitted components cautiously when individual factors are ambiguous.

## Starting and checking chains

Each Bayesian chain starts from sequential PCA factor paths by default. Supply
`initial_factors` to choose a different starting path: a finite `T × K` real
matrix, with one column per factor in level-first order. A single-factor fit
still needs a one-column matrix. The estimator copies this input before use.
AR coefficients start at zero and error innovation variances at one.

For example, several independent chains can use different random factor paths
and scales, alongside the default PCA start. Compare their retained draws using
mixing diagnostics before combining summaries. `burnin` counts discarded sweeps;
`ndraws` counts additional retained sweeps. A supplied starting path changes
neither the priors nor the posterior targeted by the sampler.

The `initial` keyword selects the model's presample distribution, as described
above. `initial_factors` selects a numerical starting path over the observed
dates; those factors are updated during sampling.

Check intercepts, factor means, loadings, and individual factor contributions
alongside reconstructed signals. Chains can agree on a shared signal or have
highly correlated factor paths while assigning different contributions to the
factors and intercepts. Neither correlation nor good reconstruction alone
establishes reliable summaries of individual components.

R-hat compares the chains' locations and spreads; values near one indicate
agreement. Effective sample size (ESS) estimates how many independent draws
would provide comparable information. Bulk ESS checks typical values, while
tail ESS checks the ends of the distribution. These diagnostics can reveal poor
sampling, but cannot prove convergence.

For independent single-factor fits `fit1`, `fit2`, `fit3`, and `fit4` with the
same retained draw count, check the first series' loading (`B[:, 2]`) as follows.
Run this from the checkout root, using its test environment and
existing diagnostic helper:

```julia
using Pkg
Pkg.activate("test")
Pkg.develop(path=pwd())
Pkg.instantiate()
include("test/support/simulation_diagnostics.jl")
# Rows are retained draws; columns are independent chains.
loading_draws = hcat(fit1.B[:, 2], fit2.B[:, 2], fit3.B[:, 2], fit4.B[:, 2])
SimulationDiagnostics.chain_diagnostics(loading_draws)
```

Repeat this for other parameters and factor contributions. These optional checks
use test dependencies; they are not part of the package's public API.
See [Checks and limitations](@ref) for the evidence behind these recommendations.

## Settings and result layout

`DFMStruct` describes one factor and a common error AR order.
`HDFMStruct` allows one AR order per level and one per observed series. Every
declared factor must be assigned to at least one series. `KN2LevelEstimator`
requires two levels; `KNHierarchicalEstimator` accepts any positive level count.

For `T` times, `N` series, `K` factors, `L` levels, and `D` retained draws:

| Field | Dimensions | Ordering |
|---|---|---|
| `F` (single factor) | `T × D` | Time, draw |
| `F` (hierarchical) | `T × K × D` | Time, factor, draw |
| `B` | `D × N*(L+1)` | Series; intercept, then one loading per level |
| `S` | `D × N` | Error innovation variances |
| `P` | `D × sum(factor AR orders)` | Factor, then lag; no padding |
| `P2` | `D × sum(error AR orders)` | Series, then lag; no padding |

`means.F` is `T × K` (including `K=1`). Other means have one row, with the same
column ordering as their draws. Unassigned loadings are stored as zero.

Rows of input are consecutive, equally spaced observations. Estimators require
finite, complete numeric data. They do not perform automatic seasonal adjustment,
detrending, standardization, missing-data imputation, or convergence detection.

## References

- [Kim and Nelson (1999), *State-Space Models with Regime Switching*](https://mitpress.mit.edu/9780262112383/state-space-models-with-regime-switching/): state-space methods and Gibbs sampling.
- [Otrok and Whiteman (1998), *Bayesian Leading Indicators*](https://doi.org/10.2307/2527349): Bayesian dynamic factor estimation.
- [Stock and Watson (2016), *Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics*](https://www.princeton.edu/~mwatson/papers/Stock_Watson_HOM_Vol2.pdf): factor models and identification.

The assumptions above define the fitted model. Tests compare its conditional
means and covariances with independent Gaussian calculations, and its complete
posterior in small examples with direct numerical integration.
