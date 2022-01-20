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
data = readdlm("test/otrok_whiteman/real_data/data_deseasoned.csv", ',')
regstatematch = readdlm("test/otrok_whiteman/real_data/regstatematch.csv", ',')

# Data as matrix 
datamat = Matrix(data[2:end, :])     # remove first row containing col names 
datamat = Float64.(datamat)         # transform entires to Float64
regstatematch = Matrix(regstatematch[2:end, :])
regassign = Int64.(regstatematch[:, 3])

# Estimate HDFM

nlevels = 2

nvar = size(datamat)[2]

nfactors = [1, max(regassign...)]

fassign = ones(Int, nvar, nlevels)
fassign[:, 2] = regassign

flags = [3, 3]

varlags = 3 * ones(Int, nvar)

hdfmpriors = HDFMParams(nlevels = nlevels,
    nvars = nvar,
    nfactors = nfactors,
    factorassign = fassign,
    factorlags = flags,
    errorlags = varlags,
    ndraws = 1000,
    burnin = 50)

results = PCA2LevelEstimator(datamat, hdfmpriors)

medians = Any[]
quant33 = Any[]
quant66 = Any[]
stds = Any[]

j = 1
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.05))
    push!(quant66, quantile(results.F[i, j, :], 0.95))
    push!(medians, median(results.F[i, j, :]))
end

plot(results.means.F[:, j])

vardecomp = vardecomp2level(datamat, results.means.F, reshape(results.means.B, 3, 50)', fassign)

plot(vardecomp[:, 1])
plot!(vardecomp[:, 2])

histogram(vardecomp[:, 1], normalize = :probability)
histogram(vardecomp[:, 2], normalize = :probability)