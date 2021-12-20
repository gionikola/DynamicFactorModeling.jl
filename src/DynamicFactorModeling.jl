module DynamicFactorModeling

######################
# Import packages 
using LinearAlgebra
using Random
using Distributions
using PDMats
using PDMatsExtras
using ShiftedArrays
using Parameters
using Polynomials

######################
# Include all package scripts 
include("simulations/dgp.jl")
include("kim_nelson/kn_estimator.jl")
include("otrok_whiteman/ow_globalfactor_estimator.jl")
include("otrok_whiteman/ow_twofactor_estimator.jl")
include("test_scripts.jl")

######################
# Export package objects 
export simulateStateSpaceModel, SSModelParameters,
    kalmanFilter, kalmanSmoother,
    dynamicFactorGibbsSampler,
    staticLinearGibbsSampler, staticLinearGibbsSamplerRestrictedVariance,
    autocorrErrorRegGibbsSampler,
    priorsSET,
    OWSingleFactorEstimator,
    priorsSET2,
    OWTwoFactorEstimator,
    sayhi
end
