
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

varcoefs = [1.0 0.5 0.5
    2.0 0.5 0.2
    3.0 0.7 0.4
    4.0 0.3 0.5
    5.0 0.5 0.5
    6.0 0.5 0.7
    7.0 0.4 0.5
    8.0 0.5 0.2
    9.0 0.5 0.2]

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
fmat = [0.5 0.25][:, :]
push!(fcoefs, fmat)
fmat = [0.5 0.25
    0.5 0.25]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [2.0]
push!(fvars, fmat)
fmat = [0.00001, 0.000001]
push!(fvars, fmat)

varvars = 5 .* rand(9) .^ 2;

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

F, B, S, P, P2 = OWTwoLevelEstimator(data_y, hdfmpriors)


using Plots

plot(data_β[:, 2])
plot!(F[:, 1])

plot(data_β[:, 3])
plot!(F[:, 2])

plot(data_β[:, 4])
plot!(F[:, 3])