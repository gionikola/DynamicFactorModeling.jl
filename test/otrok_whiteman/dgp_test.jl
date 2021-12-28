nlevels = 3

nvar = 8

nfactors = [1, 2, 4]

fassign = [1 1 1
    1 1 1
    1 1 2
    1 1 2
    1 2 3
    1 2 3
    1 2 4
    1 2 4]

flags = [1, 2, 1]

varlags = [2, 2, 2, 2, 2, 2, 2, 2]

varcoefs = [1.0 0.5 0.5 0.5
    2.0 0.5 0.2 0.5
    3.0 0.7 0.4 0.4
    4.0 0.3 0.5 0.3
    5.0 0.5 0.5 0.1
    6.0 0.5 0.7 0.2
    7.0 0.4 0.5 0.5
    8.0 0.5 0.2 0.5]

varlagcoefs = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]

fcoefs = Any[]
fmat = [0.5][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.25
    0.5 0.25]
push!(fcoefs, fmat)
fmat = [0.5
    0.5
    0.5
    0.5][:, :]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [0.1]
push!(fvars, fmat)
fmat = [1.5, 1.5]
push!(fvars, fmat)
fmat = [1.0, 1.0, 1.0, 1.0]
push!(fvars, fmat)

varvars = 5 .* rand(8) .^ 2;

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

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

##################################################
##################################################
##################################################
##################################################

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

varcoefs = [1.0 0.5 0.5
    2.0 0.5 0.2
    3.0 0.7 0.4
    4.0 0.3 0.5
    5.0 0.5 0.5
    6.0 0.5 0.7
    7.0 0.4 0.5
    8.0 0.5 0.2]

varlagcoefs = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]

fcoefs = Any[]
fmat = [0.5 0.25][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.25
    0.5 0.25]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = [1.0, 1.0]
push!(fvars, fmat)

varvars = 0.0001 * ones(nvar);

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

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

#F, B, S, P, P2 = OWTwoLevelEstimator2(data_y, data_β[:,2:4], hdfmpriors)
F, B, S, P, P2 = OWTwoLevelEstimator(data_y, hdfmpriors)

##################################################
##################################################
##################################################
##################################################

nlevels = 2

nvar = 8

nfactors = [1, 4]

fassign = [1 1
    1 1
    1 2
    1 2
    1 3
    1 3
    1 4
    1 4]

flags = [1, 3]

varlags = [2, 2, 2, 2, 2, 2, 2, 2]

varcoefs = [1.0 0.5 0.5
    2.0 0.5 0.2
    3.0 0.7 0.4
    4.0 0.3 0.5
    5.0 0.5 0.5
    6.0 0.5 0.7
    7.0 0.4 0.5
    8.0 0.5 0.2]

varlagcoefs = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]

fcoefs = Any[]
fmat = [0.5][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.25
    0.5 0.25
    0.5 0.25
    0.5 0.25]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [0.1]
push!(fvars, fmat)
fmat = [1.5, 1.5, 1.5, 1.5]
push!(fvars, fmat)

varvars = 5 .* rand(8) .^ 2;

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

hdfmpriors = HDFMPriors(nlevels = nlevels,
    nvar = nvar,
    nfactors = nfactors,
    fassign = fassign,
    flags = flags,
    varlags = varlags)

F, B, S, P, P2 = OWTwoLevelEstimator(data_y, hdfmpriors)