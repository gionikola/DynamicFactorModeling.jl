# Method audit

This page records the sources inspected for the estimator rewrite. It separates
the sampling algorithms from the particular data, priors, and normalizations
used in published applications. Page numbers below are the printed page numbers.

## Otrok and Whiteman

The inspected full text is the authors' **September 28, 1996 working paper**,
[Bayesian Leading Indicators](https://econwpa.ub.uni-muenchen.de/econ-wp/mac/papers/9610/9610002.pdf).
The [1998 published article](https://doi.org/10.2307/2527349) was identified, but its
final typeset text was not obtained for this audit.

Pages 4–6, equations (5)–(8), include the stationary distribution of the first
error observations. Regression coefficients use whitened initial rows as well as
later AR innovations. The AR conditional contains an additional initial-density
factor, so merely rejecting unstable Gaussian draws is insufficient. Page 6
uses a Metropolis–Hastings correction. Pages 7–9 derive the Gaussian conditional
distribution of the complete factor path, including its stationary initial
density.

The draft's application, pages 10–11, uses coefficient prior variance 1000,
AR prior covariance `I`, and zero inverse-gamma hyperparameters. Its fixed factor
innovation variance is an empirical average of observable-series innovation
variances. These are application settings, not universal requirements of the
derivation, and should not be attributed to the final 1998 version without
checking that version.

## Kose, Otrok, and Whiteman

The inspected full text is the authors' conference
[preprint of International Business Cycles](https://epge.fgv.br/cfrio/articles/CF010.pdf),
not the [final 2003 AER article](https://www.aeaweb.org/articles?id=10.1257/000282803769206278).

Pages 3–5 describe independent world, regional, country, and error AR processes.
Fixed factor innovation variances identify scale; positive selected loadings
identify sign. The sampler updates the world factor, then regional factors,
then country factors, conditioning on the latest other factors. Footnotes 7–9
explicitly describe the AR Metropolis step and discuss recursive Gaussian factor
sampling as an alternative to direct path-covariance calculations.

Page 6 uses AR(3) processes, loading variance 1, and AR prior covariance
`Diagonal([1, 0.5, 0.25])`. It labels the variance prior “Inverted Gamma
(6, 0.001)” without defining the density there. That notation alone is not
enough to assert a match to a software library's shape and scale convention.

## Kim and Nelson's accompanying code

The author's [book site](http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm) links its
[program list](http://econ.korea.ac.kr/~cjkim/MARKOV/prgmlist.htm) to
[SW_GIBS.PRG](http://econ.korea.ac.kr/~cjkim/MARKOV/programs/sw_gibs.prg),
the Chapter 8 linear dynamic-factor example. The actual program was read,
including `GEN_ZT`, `GEN_PSI`, `GEN_LMDA`, and `GEN_PHI`.

It fixes the factor innovation variance to 1, uses normal loading/AR priors
with covariance `I`, and sets the variance-prior hyperparameters to zero.
`GEN_ZT` initializes a Kalman filter with the stationary companion covariance
and uses backward sampling. Its parameter regressions discard early dates and
reject unstable AR proposals without the Otrok–Whiteman initial-density
correction. Consequently, matching that script literally is different from
using the full stationary likelihood consistently in every update.

The example also includes lagged factor loadings in one observation equation.
The package's DFM estimators support contemporaneous loadings and independent
Gaussian AR processes; they do not reproduce that entire application or the
regime-switching model of the 1998 article. The author's code was consulted,
not copied or redistributed.

## Jackson, Kose, Otrok, and Owyang

The full [author-uploaded 2016 chapter](https://www.researchgate.net/publication/315960109_Specification_and_Estimation_of_Bayesian_Dynamic_Factor_Models_A_Monte_Carlo_Analysis_with_an_Application_to_Global_House_Price_Comovement)
was inspected, particularly sections 2.1–2.3 and 3.1, pages 367–374.

This chapter compares the methods using the same parameter conditionals.
It distinguishes sequential factor-by-level sampling from joint state-space
sampling. Pages 369–371 explain why updating an expanded lag state one date at
a time can fail, and why a complete backward path draw is needed. They also
describe both expanded error states and quasi-differenced observations.

Its simulation-estimation priors, page 374, give the intercept variance 1,
loading variance 10, and stationary-truncated AR covariance `I`. The factor
innovation variance is fixed to 1. The variance prior is written
`IG(0.05*T, 0.25^2)`. The package's defaults are explicit configuration choices;
they should not be described as an automatic reproduction of this experiment.

## Consequences for this implementation

For an error series `e` of length `T`, set `m = min(p, T)` and let `C(phi)`
be the stationary covariance of its first `m` values for unit innovation
variance. The initial log-density contribution, omitting constants independent
of `phi`, is

```math
\ell_0(\phi) = -\tfrac12\log|C(\phi)|
              -\frac{e_{1:m}'C(\phi)^{-1}e_{1:m}}{2\sigma^2}.
```

The implementation uses this density in the AR acceptance ratio, whitens the
initial observations for coefficient and variance updates, and initializes
factor sampling with the same stationary law. It proposes once from the
unrestricted Gaussian transition-regression posterior and retains the current
draw when the proposal is unstable. This is a valid Metropolis kernel; the
paper's stationarity-truncated proposal has a different holding probability.
For `T <= p`, all observations belong to the initial block and the Gaussian
proposal is the prior. AR(0) reduces to independent errors.

`KN` selects a joint state-space path draw. `OW1` selects a Gaussian precision
draw; `OW2` selects successive complete factor-path draws conditional on the
other current factors. The optional joint precision sampler is a larger
Gaussian block update with the same posterior. `initial=:zero` instead defines
an explicit fixed-zero presample model; it is not the stationary likelihood.

The default sampler also jointly updates factor levels with intercepts and
factor sizes with loadings. These are package additions, separate from the
published factor-path algorithms above. They preserve the likelihood and
account for the changing priors. The scale Metropolis ratio includes the
transformation's Jacobian. Set `mixing_moves=:none` to omit these additions.
Independent density and small-posterior checks, followed by controlled
multi-chain comparisons, are summarized in [Checks and limitations](@ref).

Tests compare the initial covariance with independently solved Yule–Walker
equations, factor draws with a directly assembled joint Gaussian distribution,
and AR(1) and AR(2) updates with numerical integration. Complete small-model
posteriors are also integrated independently and compared with multiple KN and
OW chains. These checks target the probability distributions, rather than
reproducing a historical random number sequence or assuming the reference
programs are error-free. The [checks guide](@ref "Checks and limitations")
explains the calculations and their limits.

## Original repository

The pre-rewrite code was inspected at Git commit `dfa6efb`, especially
`archive/otrok_whiteman/ow_tools.jl`. Its `sigmat`, `sigbig`, and `arfac`
functions attempted stationary covariance, whitening, and initial-density
correction. The `ar` routine used the wrong determinant exponent, and regression
updates used an old coefficient draw in place of the prior mean. Those routines
are historical evidence of the intended method, not numerical reference
implementations. The retired Otrok author-code URL did not return source code;
no comparison against that unavailable program is claimed.
