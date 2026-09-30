# API reference

## Models and results

```@docs
SSModel
HDFM
DFMStruct
HDFMStruct
DFMMeans
DFMResults
PCAResults
```

## Simulation and states

```@docs
convertHDFMtoSS
DynamicFactorModeling.createSSforHDFM
simulateSSModel
kalmanFilter
kalmanSmoother
KNFactorSampler
```

## Estimation

```@docs
KN1LevelEstimator
KN2LevelEstimator
KNHierarchicalEstimator
OW1LevelEstimator
OW2LevelEstimator
firstComponentFactor
PCA1LevelEstimator
PCA2LevelEstimator
regress
```

## Variance analysis

```@docs
variance_decomposition
vardecomp2level
```

## Regression and distribution utilities

These lower-level utilities are available through the module prefix.

```@docs
DynamicFactorModeling.draw_coefficients
DynamicFactorModeling.draw_error_variance
DynamicFactorModeling.draw_parameters
DynamicFactorModeling.isstationary
DynamicFactorModeling.mvn
DynamicFactorModeling.Γinv
```
