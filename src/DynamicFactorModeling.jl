module DynamicFactorModeling

######################
# Import packages 
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats
using PDMatsExtras
using ShiftedArrays
using Parameters
using Polynomials
using Kronecker

######################
# Include all package scripts 
include("simulations/dgp.jl")
include("kim_nelson/kn_tools.jl")
include("otrok_whiteman/ow_globalfactor_estimator.jl")
include("otrok_whiteman/ow_twolevel_estimator.jl")
include("test_scripts.jl")

######################
# Export package objects 
export SSModel, HDFM, convertHDFMtoSS,
        simulateSSModel,
        kalmanFilter, kalmanSmoother,
        dynamicFactorGibbsSampler, staticLinearGibbsSampler, staticLinearGibbsSamplerRestrictedVariance, autocorrErrorRegGibbsSampler,
        priorsSET, OWSingleFactorEstimator,
        HDFMPriors, OWTwoLevelEstimator,
        sayhi
end
