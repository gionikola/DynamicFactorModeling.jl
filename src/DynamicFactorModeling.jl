module DynamicFactorModeling

######################
# Import packages 
using LinearAlgebra
using Statistics
using Random
using Distributions
using ShiftedArrays
using Parameters
using Polynomials

######################
# Include all package scripts
include("common_types/common_types.jl") 
include("simulations/dgp.jl")
include("linear_regression/linear_regression.jl")
include("kim_nelson/kn_1level_estimator.jl")
include("kim_nelson/kn_2level_estimator.jl")
include("output_analysis/variance_decomposition.jl")

######################
# Export package objects 
export  HDFM, DFMStruct, convertHDFMtoSS, simulateSSModel, DFMStruct,
        KN1LevelEstimator, KN2LevelEstimator,
        vardecomp2level
end
