
nlevels = 2

nvar = 9

nfactors = [1, 2]

fassign = [1 1
    1 1
    1 1
    1 1
    1 2
    1 2
    1 2
    1 2
    1 2]

flags = [2, 2]

varlags = [2, 2, 2, 2, 2, 2, 2, 2, 2]

varcoefs = [0.0 1.0 1.0
    0.0 0.5 0.2
    0.0 0.7 0.4
    0.0 0.3 0.5
    0.0 0.5 1.0
    0.0 0.5 0.7
    0.0 0.4 0.5
    0.0 0.5 0.2
    0.0 0.5 0.3]
varcoefs[:,2] = 3 .* varcoefs[:,2]
varcoefs[:,3] = 1 .* varcoefs[:,3]

varlagcoefs = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]

fcoefs = Any[]
fmat = [0.45 -0.1][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.05
    0.2 -0.1]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = [1.0, 1.0]
push!(fvars, fmat)

varvars = 0.2 * ones(nvar);

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
    burnin = 50)

results = KN2LevelEstimator(data_y, hdfmpriors)

stds = Any[]
quant33 = Any[]
quant66 = Any[]
medians = Any[]

j = 3
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.05))
    push!(quant66, quantile(results.F[i, j, :], 0.95))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
#plot!(medians)
plot!(quant33)
plot!(quant66)

vardecomp = vardecomp2level(data_y, results.means.F, reshape(results.means.B, 3, nvar)', fassign)
vardecomp2 = vardecomp2level(data_y, data_β[:, 2:4], varcoefs, fassign)
plot(vardecomp[:, 1])
plot!(vardecomp[:, 2])
plot!(vardecomp2[:, 1])
plot!(vardecomp2[:, 2])