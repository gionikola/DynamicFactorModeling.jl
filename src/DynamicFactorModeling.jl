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
    autocorrErrorRegGibbsSampler

######################
# Include all package scripts 
include("dgp.jl")
include("estimators.jl")

end
