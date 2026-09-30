# Contributing

Keep the statistical model and its initial conditions explicit. A sampling step
must target the conditional distribution of that same model. When correcting
an algorithm, add a test using an analytic answer or an independent construction,
not another copy of the implementation.

## Local checks

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=test -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test test/run_extended.jl
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Documenter's `@example` blocks execute during the build; plain Julia snippets
do not. Do not commit local manifests or generated documentation. CI is configured
to test Julia 1.10 and the current stable release on Linux, and the current stable
release on macOS and Windows.

The [checks guide](docs/src/validation.md) explains the independent posterior
calculations, maintained tests, and remaining sampling limits. The
[method audit](docs/src/method_audit.md) records which primary sources were
actually inspected and how the implementation follows their equations.

## Code map

| Directory | Contents |
|---|---|
| `src/common_types` | Model specifications and result containers |
| `src/simulations` | Gaussian draws, input checks, conversion, simulation |
| `src/linear_regression` | Gaussian regression conditionals and AR errors |
| `src/kim_nelson` | Filtering, smoothing, state draws, and shared Gibbs sampler |
| `src/pca` | Deterministic principal components |
| `src/output_analysis` | Variance accounting |
| `test` | Analytic, simulation, and API checks |
| `test/reference` | Independent calculations, external references, and compatibility fixtures |
| `test/support` | Diagnostic and sampler helpers for extended tests |
| `examples` | Small runnable workflows |

Use descriptive names, short functions, and comments that explain mathematical
choices. Prefer linear solves to explicit inverses. Keep random generators
explicit and avoid hidden input mutation, printing inside sampling loops, or
silent fallbacks when an algorithm fails. Keep numerical tests seeded, and set
Monte Carlo tolerances from sampling uncertainty with enough margin to avoid
flaky tests. Exact constraints deserve direct tests in addition to moment checks.

Extra sampler updates belong in `src/kim_nelson/mixing_moves.jl`; test helpers
should call those kernels rather than maintain a second implementation.
Preserve the `mixing_moves=:none` compatibility tests when changing the sampler.

## Before a release

- Review the complete diff, public API, result layouts, examples and documentation
  for consistency and readability.
- Run the full package tests, examples and documentation build in the supported
  environments. Verify the remote CI results for the release commit separately;
  local success does not establish cross-platform success.
- Distinguish implementation defects from documented numerical and sampling
  limits. Keep failed validation cases visible and scope claims to the evidence.
- Preserve reference fixtures and their provenance. For any new study, specify
  its question, design, cost and stopping rules beforehand; retain its original
  source and outputs separately from the package.
