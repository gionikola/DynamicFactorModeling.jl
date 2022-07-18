module DynamicFactorModeling

######################
# Import packages 
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats
using ShiftedArrays
using Parameters
using Polynomials
using Kronecker

######################
# Include all package scripts 
include("simulations/dgp.jl")
include("common_types/common_types.jl")
include("linear_regression/linear_regression.jl")
include("kim_nelson/kn_1level_estimator.jl")
include("kim_nelson/kn_2level_estimator.jl")
include("output_analysis/variance_decomposition.jl")
include("test_scripts.jl")

######################
# Export package objects 
export SSModel, HDFM, convertHDFMtoSS, simulateSSModel, DFMStruct, HDFMStruct,
        KN1LevelEstimator, KN2LevelEstimator,
        vardecomp2level,
        sayhi
end
