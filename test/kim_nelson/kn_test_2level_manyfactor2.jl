
nlevels = 2

nvar = 90

nfactors = [1, 3]

fassign = ones(Int, nvar, 2)
fassign[1:30, 2] = ones(Int, 30)
fassign[31:60, 2] = ones(Int, 30)*2
fassign[61:90, 2] = ones(Int, 30)*3

flags = [2, 2]

varlags = 1 * ones(Int, nvar)

varcoefs = zeros(nvar, 1 + nlevels)
varcoefs[:, 2] = 0.5 * ones(nvar)
varcoefs[1, 2] = 1.0
varcoefs[:, 3] = 0.1 * ones(nvar)
varcoefs[1, 3] = 1.0
varcoefs[31, 3] = 1.0
varcoefs[61, 3] = 1.0


varlagcoefs = ones(nvar, 1)[:,:]
varlagcoefs[:, 1] = 0.6 * varlagcoefs[:, 1]

fcoefs = Any[]
fmat = [0.3 0.0][:, :]
push!(fcoefs, fmat)
fmat = [0.6 0.0
    0.3 0.0
    -0.4 0.0][:,:]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = [1.0
        1.0
        1.0]
push!(fvars, fmat)

varvars = 0.4 * ones(nvar);

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

num_obs = 110
data_y, data_z, data_β = simulateSSModel(num_obs, ssmodel::SSModel)


#flags = [3, 3]

#varlags = [3, 3, 3, 3, 3, 3, 3, 3, 3]

hdfmpriors = HDFMStruct(nlevels = nlevels,
    nfactors = nfactors,
    factorassign = fassign,
    factorlags = flags,
    errorlags = varlags,
    ndraws = 1000,
    burnin = 100)

results = KN2LevelEstimator(data_y, hdfmpriors)
#results2 = KN2LevelEstimator(data_y, hdfmpriors)
#results3 = PCA2LevelEstimator(data_y, hdfmpriors)

medians = Any[]
quant33 = Any[]
quant66 = Any[]
stds = Any[]

j = 4
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.1))
    push!(quant66, quantile(results.F[i, j, :], 0.9))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
#plot!(medians)
plot!(quant33)
plot!(quant66)


vardecomp = vardecomp2level(data_y, results.means.F, reshape(results.means.B, 3, nvar)', fassign)
vardecomp2 = vardecomp2level(data_y, data_β[:, 2:5], varcoefs, fassign)


plot(vardecomp[:, 1])
plot!(vardecomp[:, 2])
plot!(vardecomp2[:, 1])
plot(vardecomp2[:, 2])