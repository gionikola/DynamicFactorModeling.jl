
nlevels = 2

nvar = 8

nfactors = [1, 2]

fassign = [1 1
    1 1
    1 1
    1 1
    1 2
    1 2
    1 2
    1 2]

flags = [2, 2]

varlags = [2, 2, 2, 2, 2, 2, 2, 2]

varcoefs = [0.0 1.0 1.0
    0.0 0.5 0.2
    0.0 0.7 0.4
    0.0 0.3 0.5
    0.0 0.5 1.0
    0.0 0.5 0.7
    0.0 0.4 0.5
    0.0 0.5 0.2]

varlagcoefs = [0.5 0.25
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

varvars = 0.1 * ones(nvar);

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

ssmodel = convertHDFMtoSS(hdfm)

num_obs = 100
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel::SSModel)


#flags = [3, 3]

#varlags = [3, 3, 3, 3, 3, 3, 3, 3, 3]

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

results = OW2LevelCoefEstimator(data_y, hdfmpriors, data_β[:, 2:4])
results = OW2LevelFactorEstimator(data_y, hdfmpriors, varcoefs, varlagcoefs, [fcoefs[1]; fcoefs[2]], varvars)


stds = Any[]
quant33 = Any[]
quant66 = Any[]
medians = Any[]

j = 1
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.05))
    push!(quant66, quantile(results.F[i, j, :], 0.95))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
plot!(medians)
plot!(quant33)
plot!(quant66)