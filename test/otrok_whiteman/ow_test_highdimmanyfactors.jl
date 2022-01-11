
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

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

results = OW2LevelFactorEstimator(data_y, hdfmpriors, varcoefs, varlagcoefs, [fcoefs[1]; fcoefs[2]], varvars)