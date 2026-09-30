# Changelog

## 0.2.0 (unreleased)

The unfinished 0.1 routines were replaced with a consistent Gaussian model,
explicit initial conditions and priors, validated state-space calculations,
and independent posterior tests. See the [upgrade notes](docs/src/migration.md)
for changes to the API, stored results, and assumptions.

- Bayesian estimators now add joint factor/intercept location updates and
  factor/loading scale updates by default. These preserve the model and its
  posterior while improving mixing in the controlled examples.
- `mixing_moves=:location` uses only the location update; `:none` preserves the
  corrected sampler's previous seeded draws in the same software environment.
  `scale_step=0.1` sets the fixed log-scale proposal size.
- Added independent distribution checks, public sampler integration tests,
  and a maintained extended test suite.
- A 64-dataset one-factor calibration screen passed its declared checks.
  A hierarchical comparison retains chain-agreement failures in its smaller
  panel; both expanded panels passed. The [checks and limitations](docs/src/validation.md)
  summarize this evidence and the remaining precision limits.
- Research-study programs and generated outputs are separate from the package.
  Reusable references and bounded checks are maintained under `test`.
