
@testset "Linear regression" begin

    Y = rand(30)[:,:]
    X = rand(30, 2)
    p = 1
    iter = 10
    burnin = 5

    β, ϕ, σ2 = DynamicFactorModeling.regress(Y, X, p, iter, burnin)

    @test size(β) == (iter - burnin, size(X)[2])
    @test size(ϕ) == (iter - burnin, p)
    @test size(σ2) == (iter - burnin,)

    Y = rand(30)
    X = rand(30, 2)
    p = 1
    iter = 10
    burnin = 5

    β, ϕ, σ2 = DynamicFactorModeling.regress(Y, X, p, iter, burnin)

    @test size(β) == (iter - burnin, size(X)[2])
    @test size(ϕ) == (iter - burnin, p)
    @test size(σ2) == (iter - burnin,)

end
