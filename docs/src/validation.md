# Checks and limitations

The tests check probability distributions and numerical identities using
independent calculations. They support the models and settings covered below;
they cannot prove that every implementation is correct or that every fitted
chain has converged. Check several chains on each dataset before interpreting
individual factors, their contributions, or reconstructed signals.

## Running the maintained checks

From the checkout root, run the normal package tests:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

These include analytic regression and AR checks, state-space references, PCA
and variance accounting, sampler compatibility, and complete small-model
posteriors. The extended suite adds joint-update, diagnostic, Gaussian-reference,
and calibration-target checks, including public sampler posterior comparisons:

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test test/run_extended.jl
```

Both suites use included code and small fixtures; no saved research data are
needed. Independent calculations are in `test/reference`, with supporting
utilities in `test/support`. The diagnostic recipe in
[Model and estimation](@ref) uses that same test environment.

Research-study programs and generated outputs are not included in the package.
The study results below summarize completed work; running the maintained tests
does not rerun those studies. Remote CI results and the final release checks
have not yet been verified for the current revision. Local results do not
establish cross-platform success.

## Independent mathematical checks

**Regression and autoregressions.** Tests compare conditional means and
covariances with Gaussian formulas and inverse-gamma moments. Independently
solved Yule–Walker equations check stationary covariances and whitening through
AR(5), including samples shorter than the AR order. Numerical integration over
the AR(2) stability region checks both short-sample and transition-regression
Metropolis proposals. Stationary and fixed-zero initial conditions are tested
as different likelihoods.

**Complete posteriors.** The reference integrates out intercepts and factor
paths analytically, then integrates remaining parameters by Gaussian quadrature.
It does not call the package's filters, conditional draws, or covariance helpers.
The cases include different priors, one and two factors, and stationary or
zero-initialized AR(1) models with one or two dates. Increasing the integration
order checks reference accuracy. Independently seeded KN and OW chains are
compared through means, cross moments, and quantile probabilities, with errors
estimated from batches of consecutive draws. Statistics without the finite
sampling variance needed for that error estimate are omitted explicitly.

**Joint updates.** Tests check the location and scale moves against independent
prior densities, signal-preservation identities, and small-model posterior
references. Integration checks cover loading storage, exclusions, sign anchors,
initial conditions, and random-number use. A frozen pre-update sampler verifies
that `mixing_moves=:none` preserves its update order and seeded draws in the
same environment. That compatibility fixture is separate from the independent
probability calculations.

**State-space calculations.** Normal tests include 20 fixed models generated
directly by statsmodels 0.14.6 and SciPy 1.15.3. They check forecasts, filtered
and smoothed means and covariances, and stationary covariance. The independent
Python generator and pinned requirements are retained in `test/reference`.
A separate completed 1,000-model comparison passed 21,985 comparisons at
absolute and relative tolerances of `1e-9`.

For singular covariances and exact observations, another reference constructs
the complete Gaussian trajectory and conditions its original disturbances.
Saved difficult cases remain in the normal tests. A completed 1,500-model
stress check found covariance differences up to about `3.6e-12`. In severely
ill-conditioned cases, mean differences reached about `2e-5`; their checks
allowed for the reference's estimated rounding error. This is evidence of a
numerical limit, not uniform accuracy for arbitrary singular models. See
[State-space tools](@ref) for the bounded joint-path fallback and rescaling errors.

## A small calibration screen

Simulation-based calibration draws parameters from the priors, generates data,
and compares the generating values with posterior draws. A completed screen
used **64 datasets**: 32 each for AR(0) and stationary AR(1), with one factor,
two series, and six dates. Generation and fitting used the default priors,
unit factor innovation variance, and the same joint sign convention. The
public KN1 sampler included the default location and scale updates.

Each dataset used eight independent chains with 2,000 warmup and 8,000 retained
sweeps. One final draw per chain supplied eight posterior endpoints. All 512
chains completed; all 576 target checks passed the declared limits of R-hat
at most 1.01 and bulk and tail ESS of at least 400. No cases were discarded.

A rank counts how many endpoints fall below the generating value. Correctly
calibrated ranks are uniform over 0 through 8. All 18 target/order rank
distributions stayed within the fixed cumulative-distribution bands: ±0.240
for one comparison and ±0.321 for the family of 18. The largest discrepancy
was 0.2292, for the factor AR coefficient. With only 32 cases per distribution,
these broad bands have low power to detect smaller errors.
Independent chains also do not eliminate bias from finite sampling runs.

Two deliberate controls were detected by likelihood targets: draws that ignored
the data, and posterior parameters paired with an unrelated prior factor path.
Parameter ranks alone did not detect all these errors. This checks sensitivity
to those particular mistakes, not every possible defect. The screen covers
default-prior one-factor models; it does not validate all hierarchy sizes,
priors, initial laws, or finite-run behavior. See
[Talts et al.](https://arxiv.org/abs/1804.06788) for the calibration method.

## A difficult hierarchical example

Four KNHierarchical chains on the **same 40-date, eight-series, seven-factor
dataset** each used 2,000 warmup and 16,000 retained draws with the default
location and scale updates. Of 1,781 monitored scalar summaries, **14 failed
the declared chain-agreement check**: three loadings, four additional summaries,
and seven centered factor contributions. These summaries overlap; they are not
1,781 independent tests. All failures were R-hat failures (maximum 1.012953);
every bulk and tail ESS exceeded 400. Factor-date and fitted-signal checks passed.

A full-draw follow-up measured the practical disagreement:

| Across-chain posterior means | Smallest–largest | Range relative to the smallest mean |
| --- | ---: | ---: |
| Regional loading on series 1 | 0.6484–0.7215 | 11.3% |
| Local factor 4 contribution size | 1.0526–1.1354 | 7.86% |
| Combined full-signal size | 3.70638–3.70823 | 0.050% |

Here a contribution is its loading times its factor **within each draw**.
Size is the norm of the centered contribution or signal over the original four
series and 40 dates, divided by `sqrt(40)`. It is not a variance share.
Agreement on a norm does not establish agreement on every time path; the
signal-coordinate checks are separate evidence.

The largest selected sign-probability difference was **8.56 percentage points**.
All four chains favored the same sign at that coordinate, but their confidence
was meaningfully different. Later halves did not resolve the movement: the
regional-loading means spanned 0.6865–0.7097 in the first 8,000 draws and
0.5949–0.7566 in the last 8,000. No early draws or failed chains were discarded.

Batch-mean Monte Carlo errors, with fixed batch-size sensitivity checks,
quantified numerical uncertainty within each chain. They cannot bound bias from
unexplored posterior regions. Opposing late movements and the failed R-hat
checks therefore remain relevant. The comparisons are descriptive, not a
multiple-testing-adjusted significance test. Both starting paths and random
streams differed, so this example does not isolate initialization as a cause.

Paired panels with more dates or more series passed their declared checks.
They share one underlying dataset and do not establish repeated-dataset
coverage or general performance. The small-model calibration results likewise
do not resolve the difficult hierarchy. No new formula defect was established.
These are differences between chains fitted to the same data; a small dataset
and broad posterior uncertainty do not by themselves establish that those chains
are sampling reliably.

For practical use, examine individual loadings, contributions, and signals
across chains. Broad uncertainty can coexist with inadequate numerical precision
in a component probability. Do not infer reliable individual factors from good
reconstruction alone. [Vehtari et al.](https://arxiv.org/abs/1903.08008) explain
the rank-based R-hat and effective-sample-size diagnostics used here.
