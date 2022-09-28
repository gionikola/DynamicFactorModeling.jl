@testset "First principal component" begin

    numobs = 100
    numvar = 3
    X = rand(numobs, numvar)

    a, b = DynamicFactorModeling.firstComponentFactor(X) 

    @test size(a) == (numobs,)
    @test size(b) == (numvar,)

end 