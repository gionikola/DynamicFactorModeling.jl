
nlevels = 2

nvar = 200

nfactors = [1, 10]

fassign = ones(Int, nvar, 2)
fassign[:, 2] = rand(1:10, 200)

flags = [2, 2]

varlags = 2 * ones(Int, nvar)

varcoefs = zeros(nvar, 1 + nlevels)
varcoefs[:, 2] = rand(nvar)
varcoefs[:, 3] = ones(nvar)


varlagcoefs = ones(nvar, 2)
varlagcoefs[:, 1] = 0.5 * varlagcoefs[:, 1]
varlagcoefs[:, 2] = 0.25 * varlagcoefs[:, 2]

fcoefs = Any[]
fmat = [0.85 -0.3][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.05
    0.2 -0.1
    0.3 -0.1
    0.4 -0.1
    0.5 -0.1
    0.5 0.05
    0.2 -0.1
    0.3 -0.1
    0.4 -0.1
    0.5 -0.1]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = ones(nfactors[2])
push!(fvars, fmat)

varvars = 0.05 * ones(nvar);

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

hdfmpriors = HDFMStruct(nlevels = nlevels,
    nvars = nvar,
    nfactors = nfactors,
    factorassign = fassign,
    factorlags = flags,
    errorlags = varlags,
    ndraws = 1000,
    burnin = 50)

results = PCA2LevelEstimator(data_y, hdfmpriors)

stds = Any[]
quant33 = Any[]
quant66 = Any[]
medians = Any[]

j = 10
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.05))
    push!(quant66, quantile(results.F[i, j, :], 0.95))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
#plot!(medians)
#plot!(quant33)
#plot!(quant66)