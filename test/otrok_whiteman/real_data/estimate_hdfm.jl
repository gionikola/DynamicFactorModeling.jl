using Plots
using LinearAlgebra
using Statistics
using Random
using Distributions
using PDMats
using PDMatsExtras
using ShiftedArrays
using Parameters
using Polynomials

using DelimitedFiles
using DataFrames


# Import data 
data = readdlm("test/otrok_whiteman/real_data/data.csv", ',')
regstatematch = readdlm("test/otrok_whiteman/real_data/regstatematch.csv", ',')

# Data as matrix 
datamat = Matrix(data[2:end, :])     # remove first row containing col names 
datamat = Float64.(datamat)         # transform entires to Float64
regstatematch = Matrix(regstatematch[2:end, :])
regassign  = Int64.(regstatematch[:,3])

# Estimate HDFM

nlevels = 2

nvar = size(datamat)[2]

nfactors = [1, max(regassign...)]

fassign = ones(Int, nvar, nlevels)
fassign[:,2] = regassign

flags = [3, 3]

varlags = 3 * ones(Int, nvar)

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

F2, B2, S2, P2, P22 = OWTwoLevelEstimator(datamat, hdfmpriors)
