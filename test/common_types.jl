@testset "Model settings" begin
    @test DFMStruct(factorlags=0, errorlags=0, ndraws=1, burnin=0).ndraws == 1
    @test_throws ArgumentError DFMStruct(factorlags=-1, errorlags=0)
    @test_throws ArgumentError DFMStruct(factorlags=1, errorlags=1, ndraws=0)
    @test_throws ArgumentError DFMStruct(factorlags=1, errorlags=1, burnin=-1)
    @test_throws DimensionMismatch HDFMStruct(nlevels=2, nfactors=[1],
        factorassign=ones(Int,3,2), factorlags=[1,1], errorlags=[1,1,1])
    @test_throws ArgumentError HDFMStruct(nlevels=1, nfactors=[2],
        factorassign=ones(Int,3,1), factorlags=[1], errorlags=[1,1,1])
    @test_throws ArgumentError HDFMStruct(nlevels=1, nfactors=[1],
        factorassign=fill(2,3,1), factorlags=[1], errorlags=[1,1,1])
end
