# DynamicFactorModeling.jl

[![Build Status](https://github.com/gionikola/DynamicFactorModeling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gionikola/DynamicFactorModeling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gionikola/DynamicFactorModeling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gionikola/DynamicFactorModeling.jl)

## Overview 

This is a Julia package that allows the user to easily construct, simulate, and estimate linear multi-level/hierarchical dynamic factor models (HDFMs) using a variety of Bayesian approaches. 
A wonderful explanation of HDFMs is provided in [[5]](#5). For an example, check out [[4]](#4).

Although the simulation capabalities can come in quite handy, the main value of this package lies in its estimation capabilities.
Three HDFM estimation approaches are offered: 
1. Principal component analysis (PCA) (overviewed in [[1]](#1)) (**AVAILABLE**);
2. Kim-Nelson (KM) state-space approach (introduced in [[2]](#2) and [[3]](#3)) (**IN PROGRESS**);
3. Otrok-Whiteman (OW) approach (introduced in [[5]](#6) and [[3]](#4)) (**IN PROGRESS**).
This package estimates HDFM hyperaparemeters in the same manner (outlined in [[3]](#3)) across all of the above methodologies, despite the original hyperparameter estimation procedure for the OW estimator being based on the Chib-Greenberg linear regression with ARMA(p,q) error estimator -- the former seems to be faster and easier to diagnose.

## Installation

DynamicFactorModeling.jl is still in development and not available through the Julia registry.
Thereofore, you may import the package using the GitHub repo url in the following manner:

```julia
using Pkg
Pkg.add(url = "https://github.com/gionikola/DynamicFactorModeling.jl")
```

## Walkthrough 

### 1. **Specify HDFM** 

Text.

```julia

nlevels = 2

nvar = 9

nfactors = [1, 2]

fassign = [1 1
    1 1
    1 1
    1 1
    1 2
    1 2
    1 2
    1 2
    1 2]

flags = [2, 2]

varlags = [2, 2, 2, 2, 2, 2, 2, 2, 2]

varcoefs = [0.0 1.0 1.0
    0.0 0.5 0.2
    0.0 0.7 0.4
    0.0 0.3 0.5
    0.0 0.5 1.0
    0.0 0.5 0.7
    0.0 0.4 0.5
    0.0 0.5 0.2
    0.0 0.5 0.2]

varlagcoefs = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]

fcoefs = Any[]
fmat = [0.85 -0.3][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.05
    0.2 -0.1]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = [1.0, 1.0]
push!(fvars, fmat)

varvars = 0.5 * ones(nvar);

hdfm = HDFM(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags,
    varcoefs = varcoefs,
    varlagcoefs = varlagcoefs,
    fcoefs = fcoefs,
    fvars = fvars,
    varvars = varvars)

```

### 2. Simulate HDFM 

Text.

```julia

ssmodel = convertHDFMtoSS(hdfm)

num_obs = 100
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel::SSModel)

```

### 3. Estimate HDFM 

Text.

```julia

hdfmpriors = HDFMStruct(nlevels = nlevels,
    nfactors = nfactors,
    factorassign = fassign,
    factorlags = flags,
    errorlags = varlags,
    ndraws = 1000,
    burnin = 50)

results = PCA2LevelEstimator(data_y, hdfmpriors)

```

### 4. Variance decomposition

Text.

```julia

vardecomp = vardecomp2level(datamat, results.means.F, reshape(results.means.B, 3, 50)', fassign)

```

## References 

<a id="1">[1]</a> 
Jackson, L.E., Kose, M.A., Otrok, C. and Owyang, M.T. (2016), "Specification and Estimation of Bayesian Dynamic Factor Models: A Monte Carlo Analysis with an Application to Global House Price Comovement", Dynamic Factor Models (Advances in Econometrics, Vol. 35), Emerald Group Publishing Limited, Bingley, pp. 361-400.

<a id="2">[2]</a> 
Kim, Chang-Jin and Nelson, Charles, (1998), Business Cycle Turning Points, A New Coincident Index, And Tests Of Duration Dependence Based On A Dynamic Factor Model With Regime Switching, The Review of Economics and Statistics, 80, issue 2, p. 188-201.

<a id="3">[3]</a> 
Kim, Chang-Jin and Nelson, Charles, (1999), State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications, vol. 1, 1 ed., The MIT Press.

<a id="4">[4]</a> 
Kose, M. Ayhan, Christopher Otrok, and Charles H. Whiteman. 2003. "International Business Cycles: World, Region, and Country-Specific Factors." American Economic Review, 93 (4): 1216-1239.

<a id="5">[5]</a> 
Moench, Emanuel, Serena Ng, Simon Potter. 2013. Dynamic Hierarchical Factor Models. The Review of Economics and Statistics, 95 (5): 1811–1817.

<a id="6">[6]</a> 
Otrok, Christopher and Whiteman, Charles, (1998), Bayesian Leading Indicators: Measuring and Predicting Economic Conditions in Iowa, International Economic Review, 39, issue 4, p. 997-1014.
