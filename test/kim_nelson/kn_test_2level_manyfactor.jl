
nlevels = 2

nvar = 20

nfactors = [1, 5]

fassign = ones(Int, nvar, 2)
fassign[1:4, 2] = ones(Int, 4)
fassign[5:8, 2] = 2 * ones(Int, 4)
fassign[9:12, 2] = 3 * ones(Int, 4)
fassign[13:16, 2] = 4 * ones(Int, 4)
fassign[17:20, 2] = 5 * ones(Int, 4)

flags = [2, 2]

varlags = 2 * ones(Int, nvar)

varcoefs = zeros(nvar, 1 + nlevels)
varcoefs[:, 2] = 0.5 * ones(nvar)
varcoefs[1, 2] = 1.0
varcoefs[:, 3] = 0.1 * ones(nvar)
varcoefs[1, 3] = 1.0
varcoefs[5, 3] = 1.0
varcoefs[9, 3] = 1.0
varcoefs[13, 3] = 1.0
varcoefs[17, 3] = 1.0


varlagcoefs = ones(nvar, 2)
varlagcoefs[:, 1] = 0.5 * varlagcoefs[:, 1]
varlagcoefs[:, 2] = 0.25 * varlagcoefs[:, 2]

fcoefs = Any[]
fmat = [0.45 -0.2][:, :]
push!(fcoefs, fmat)
fmat = [0.6 0.00
    0.2 -0.05
    -0.3 0.2
    0.2 -0.1
    -0.4 0.15]
push!(fcoefs, fmat)

fvars = Any[]
fmat = [1.0]
push!(fvars, fmat)
fmat = [1.0, 1.0, 1.0, 1.0, 1.0]
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

num_obs = 100
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

j = 6
for i in 1:size(results.F)[1]
    push!(stds, std(results.F[i, j, :]))
    push!(quant33, quantile(results.F[i, j, :], 0.33))
    push!(quant66, quantile(results.F[i, j, :], 0.66))
    push!(medians, median(results.F[i, j, :]))
end

plot(data_β[:, 1+j])
plot!(results.means.F[:, j])
#plot!(medians)
plot!(quant33)
plot!(quant66)

vardecomp = vardecomp2level(data_y, results.means.F, reshape(results.means.B, 3, nvar)', fassign)
vardecomp2 = vardecomp2level(data_y, data_β[:, 2:7], varcoefs, fassign)

plot(vardecomp[:, 1])
plot!(vardecomp[:, 2])
plot!(vardecomp2[:, 1])
plot!(vardecomp2[:, 2])

medianF = similar(results.means.F)
for j in 1:6
    medians = Any[]
    for i in 1:size(results.F)[1]
        push!(medians, median(results.F[i, j, :]))
    end
    medianF[:, j] = medians
end
medianB = similar(results.means.B)
for i in 1:length(results.means.B)
    medianB[i] = median(results.B[:, i])
end
vardecomp = vardecomp2level(data_y, medianF, reshape(medianB, 3, nvar)', fassign)

for i in 1:300
    coefvec = results.P[270, 2:3]
    coef = [-reverse(vec(coefvec), dims = 1); 1]                      # check stationarity 
    root = roots(Polynomial(reverse(coef)))
    rootmod = abs.(root)
    accept = min(rootmod...) >= 1.01
    accept
    println(accept)
end 

varvec = []
for i in 1:1000
    facvar = var(results.F[:,end,i])
    push!(varvec, facvar)
end 