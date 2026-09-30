using LinearAlgebra
using Random

@testset "HDFM state-space conversion" begin
    # Factor orders 2 and 0, series-error orders 1, 3, and 0.
    model = HDFM(nlevels=2, nvar=3, nfactors=[1, 2],
        fassign=[1 1; 1 2; 0 1], flags=[2, 0], varlags=[1, 3, 0],
        varcoefs=[1.0 2.0 3.0; -2.0 0.5 -1.0; 4.0 9.0 0.2],
        varlagcoefs=[0.4 0.0 0.0; 0.2 -0.1 0.05; 0.0 0.0 0.0],
        fcoefs=[[0.5 -0.2], zeros(2, 0)], fvars=[[0.7], [0.8, 0.9]],
        varvars=[0.1, 0.2, 0.3])
    ss = convertHDFMtoSS(model)
    @test size(ss.F) == (10, 10)
    @test ss.H == [1 2 0 3 0 1 0 0 0 0;
                   -2 0.5 0 0 -1 0 1 0 0 0;
                   4 0 0 0.2 0 0 0 0 0 1]
    expected_F = zeros(10, 10)
    expected_F[2, 2:3] = [0.5, -0.2]
    expected_F[3, 2] = 1
    expected_F[6, 6] = 0.4
    expected_F[7, 7:9] = [0.2, -0.1, 0.05]
    expected_F[8, 7] = expected_F[9, 8] = 1
    @test ss.F == expected_F
    @test ss.μ == [1; zeros(9)]
    @test ss.Q == Diagonal([0, 0.7, 0, 0.8, 0.9, 0.1, 0.2, 0, 0, 0.3])
    @test iszero(ss.R)
    @test size(ss.A) == (3, 0)
    @test size(ss.Z) == (0, 0)

    # Verify the defining equations directly, independently of state indexing code.
    y, z, states = simulateSSModel(MersenneTwister(804), 50, ss)
    @test states[:, 1] == ones(50)
    @test states[2:end, 3] ≈ states[1:end-1, 2]
    @test states[2:end, 8] ≈ states[1:end-1, 7]
    @test states[2:end, 9] ≈ states[1:end-1, 8]
    @test y[:, 1] ≈ 1 .+ 2states[:, 2] .+ 3states[:, 4] .+ states[:, 6]
    @test y[:, 2] ≈ -2 .+ 0.5states[:, 2] .- states[:, 5] .+ states[:, 7]
    @test y[:, 3] ≈ 4 .+ 0.2states[:, 4] .+ states[:, 10]
    @test size(z) == (50, 0)

    # Array fields remain mutable, so conversion must validate them again.
    model.fassign[1, 1] = 2
    @test_throws ArgumentError convertHDFMtoSS(model)
    model.fassign[1, 1] = 1
    model.varvars[1] = -1
    @test_throws ArgumentError convertHDFMtoSS(model)
end
