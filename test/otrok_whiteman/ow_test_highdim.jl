
nlevels = 2

nvar = 20

nfactors = [1, 2]

fassign = ones(Int, nvar, 2)
fassign[(1+trunc(Int, nvar / 2)):end, 2] = 2 * ones(Int, trunc(Int, nvar / 2))

flags = [2, 2]

varlags = 2 * ones(Int, nvar)

varcoefs = zeros(nvar, 1 + nlevels)
varcoefs[:, 2] = 0.5 * ones(nvar)
varcoefs[1, 2] = 1.0
varcoefs[:, 3] = 0.1 * ones(nvar)
varcoefs[1, 3] = 1.0
varcoefs[1+trunc(Int, nvar / 2), 3] = 1.0

varlagcoefs = ones(nvar, 2)
varlagcoefs[:, 1] = 0.2 * varlagcoefs[:, 1]
varlagcoefs[:, 2] = 0.1 * varlagcoefs[:, 2]

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

varvars = 1 * ones(nvar);

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

results = OWTwoLevelEstimator(data_y, hdfmpriors)

medians = Any[]
quant33 = Any[]
quant66 = Any[]
stds = Any[]

j = 3
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.33))
    push!(quant66, quantile(results.F[i, j, :], 0.66))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
plot!(medians)
plot!(quant33)
plot!(quant66)
plot!(results.means.F[:, j] - stds)
plot!(results.means.F[:, j] + stds)