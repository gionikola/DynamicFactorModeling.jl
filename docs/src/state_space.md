# State-space tools

An `SSModel` represents

```math
\beta_t = \mu + F\beta_{t-1} + v_t, \qquad
y_t = H\beta_t + Az_t + e_t,
```

with independent Gaussian innovations of covariance `Q` and `R`. `Z` is the
covariance used to simulate `z`. Filtering conditions on supplied `data_z`;
it does not integrate over `Z`.

## Initialization and timing

`initial_mean` and `initial_cov` describe the state **before the first returned
row**, at time zero. A transition occurs before observing row 1. By default,
they are the stationary mean and covariance, which requires all eigenvalues of
`F` to have magnitude less than one. Explicitly provide both to use an unstable
transition, such as a random walk. In simulation, `initial_state` fixes time zero
exactly and cannot be combined with the other initialization keywords.

```@example filtering
using DynamicFactorModeling, Random
ss = SSModel(
    H=ones(1,1), A=zeros(1,0), F=reshape([0.8],1,1), μ=[0.0],
    R=reshape([0.2],1,1), Q=reshape([0.5],1,1), Z=zeros(0,0),
)
y, _, states = simulateSSModel(MersenneTwister(10), 30, ss)
y_missing = Matrix{Union{Missing,Float64}}(y)
y_missing[10:12,1] .= missing
predicted_y, filtered, predicted_cov, filtered_cov = kalmanFilter(y_missing, ss)
fitted_y, smoothed, smoothed_cov = kalmanSmoother(y_missing, ss)
path = KNFactorSampler(MersenneTwister(11), y_missing, ss)
(size(filtered), size(smoothed), size(path))
```

`predicted_y[t,:]` uses observations before time `t`. Filtered states also use
row `t`; smoothed states use the entire sample. Covariances are vectors of
matrices, one for each time. Smoothed fitted observations exclude measurement
noise. A sampled path retains uncertainty and dependence across time; it is
not a sequence of independent draws from smoothed marginals.

## Missing values, regressors, and exact constraints

`missing` measurements are skipped. An entirely missing row performs only the
prediction step. `NaN` and infinities raise an error. Regressors must be finite
and complete; provide `data_z` whenever `A` is nonzero. With no regressors,
use an `nseries × 0` matrix `A` and `0 × 0` matrix `Z`.

`R`, `Q`, and `Z` may be singular. For example, companion states hold exact lags
and have zero innovation variance. The sampler preserves those relationships.
The filter rejects observations that contradict an exact zero-noise equation.
Calculations use `Float64`. The filter and backward sampler carry covariance
roots: matrices that multiply independent standard-normal noise to produce the
state uncertainty. Conditioning separates the observed and unobserved noise
directions using a singular-value decomposition. This avoids subtracting nearly
equal covariance matrices, which can create spurious uncertainty in exact lags.

Rank checks first scale the coordinates, so a small independent variance is not
discarded just because another state has a larger scale. Input covariance
checks remove tiny negative eigenvalues caused by rounding and reject materially
indefinite matrices. Nearly dependent directions still have a numerical rank
threshold; severely ill-conditioned systems may need rescaling. No Float64
implementation can retain arbitrary precision near a deterministic constraint.

When deterministic backward transitions amplify rounding, smoothing and sampling
switch to a joint calculation for the entire path. A constrained filtering step
can likewise recheck its observation prefix jointly. These exceptional dense
calculations are limited to four million entries in each required matrix,
including the observation decomposition and trajectory noise map. Larger
problematic cases raise an error asking for rescaling. An entirely unobserved
path uses its prior directly and does not require this fallback.

The stationary covariance calculation checks convergence in each state's own
units, so a large variance cannot hide a small, slowly converging variance.
See [Checks and limitations](@ref) for independent external-library comparisons,
singular-model checks, and the numerical limits observed in those checks.

## Hierarchical state ordering

`convertHDFMtoSS` creates the following state vector:

1. A deterministic constant equal to one.
2. A block for each factor, by level and then factor number.
3. A block for each series' error.

Each process block holds its current value followed by lags, with length
`max(1, AR_order)`. AR(0) therefore uses one state. The constant is reset by
`μ[1]=1`; it is not a stochastic unit root. Observation errors are represented
inside the state, so `R` is zero in this conversion.

The implementation uses the Gaussian filtering and smoothing identities described
in [Särkkä (2013), *Bayesian Filtering and Smoothing*](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf),
Chapters 4 and 8. Tests also condition a separately assembled joint Gaussian
model, including cross-time covariances, to check these identities independently.
