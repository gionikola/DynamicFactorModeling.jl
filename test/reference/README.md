# Independent references and compatibility fixtures

These small files are part of the tests. They do not require saved research-study
results. See the [checks guide](../../docs/src/validation.md) for test commands
and the limits of the evidence.

- `statsmodels_fixtures.jl` contains 20 filter, smoother, and stationary-covariance
  references from statsmodels 0.14.6 and SciPy 1.15.3, generated with seed 9272026.
  `check_statsmodels.jl` checks them during the normal package tests.
  `generate_statsmodels.py` and `requirements.txt` provide the separate Python
  environment and generator. Regenerate references deliberately; do not edit
  expected numbers to match package output.
- `singular_drift_fixtures.jl` preserves difficult exact-observation cases from
  the September 28, 2026 audit. `singular_state_space.jl` compares them
  with an independent calculation that conditions the complete Gaussian path.
- `dfm_sampler_before_mixing.jl.txt` is an immutable sampler snapshot from before
  the location and scale updates were added. `test/mixing_integration.jl` loads
  only its outer sampling loop to check that `mixing_moves=:none` preserves the
  previous update order, draws, and random-number state. It is a compatibility
  reference, not an independent proof of the posterior calculations.

The maintained sampler fixture preserves the original pre-update loop unchanged.
Its 17,062 bytes have SHA-256:

```text
8349d92a1b6def57274edd61eabb54437e7e244ceedde9615768b82186298afa
```

Keep these fixture bytes unchanged when editing production code. A deliberate
replacement needs its own provenance and an explanation of the changed test
contract.

The other reference modules provide reusable independent calculations:

- `posterior_reference.jl` integrates small-model posteriors without calling
  the package's conditional draws or covariance routines.
- `gaussian_location_reference.jl` constructs the full Gaussian posterior for
  factors and intercepts when the remaining parameters are known.
- `sbc_reference.jl` independently evaluates one-factor Gaussian likelihoods
  and quantities used in calibration checks.
- `simulation_scenarios.jl` supplies declared model fixtures for synthetic tests.

Sampler and diagnostic helpers live in `test/support`. They are test utilities,
not additional public package APIs.
