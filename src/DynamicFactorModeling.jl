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

######################
# Export package objects 
export simulateStateSpaceModel, SSModelParameters,
    kalmanFilter, kalmanSmoother,
    dynamicFactorGibbsSampler,
    staticLinearGibbsSampler, staticLinearGibbsSamplerRestrictedVariance,
    autocorrErrorRegGibbsSampler, 
    otrokWhitemanFactorSampler,
    sayhi

######################
# Include all package scripts 
include("dgp.jl")
include("estimators.jl")
include("test_scripts.jl")
include("otrok_whiteman_1998_estimator.jl")

end
