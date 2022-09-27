

@testset "Create SS for HDFM" begin

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

    varvars = 0.2 * ones(nvar)

    hdfm = HDFM(nlevels=nlevels,
        nvar=nvar,
        nfactors=nfactors,
        fassign=fassign,
        flags=flags,
        varlags=varlags,
        varcoefs=varcoefs,
        varlagcoefs=varlagcoefs,
        fcoefs=fcoefs,
        fvars=fvars,
        varvars=varvars)

    H, A, F, μ, R, Q, Z = DynamicFactorModeling.createSSforHDFM(hdfm)

    @test typeof(H) == Array{Float64,2}
    @test typeof(A) == Array{Float64,2}
    @test typeof(F) == Array{Float64,2}
    @test typeof(μ) == Array{Float64,1}
    @test typeof(R) == Array{Float64,2}
    @test typeof(Q) == Array{Float64,2}
    @test typeof(Z) == Array{Float64,2}

    #=
    @test size(H) == 
    @test size(A) == 
    @test size(F) == 
    @test size(μ) == 
    @test size(R) == 
    @test size(Q) == 
    @test size(Z) == 
    =#

end

@testset "Convert HDFM to SS" begin

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

    varvars = 0.2 * ones(nvar)

    hdfm = HDFM(nlevels=nlevels,
        nvar=nvar,
        nfactors=nfactors,
        fassign=fassign,
        flags=flags,
        varlags=varlags,
        varcoefs=varcoefs,
        varlagcoefs=varlagcoefs,
        fcoefs=fcoefs,
        fvars=fvars,
        varvars=varvars)

    ssmodel = convertHDFMtoSS(hdfm)

    @test typeof(ssmodel) == DynamicFactorModeling.SSModel

end 