@testset "PCA reconstruction and normalization" begin
    factor = [-2.0, -1, 0, 1, 2]
    loading = [2.0, -3, 1]
    data = [4.0 2.0 -1.0] .+ factor * loading'
    f, b = firstComponentFactor(data)
    @test f * b' ≈ data .- mean(data; dims=1)
    @test mean(f) ≈ 0 atol=1e-15
    @test sum(abs2, f) / length(f) ≈ 1
    @test b[argmax(abs.(b))] > 0
    result = PCA1LevelEstimator(data)
    @test result.intercepts' .+ result.factors * result.loadings' + result.residuals ≈ data
    @test norm(result.residuals) < 1e-12
    @test firstComponentFactor(fill(2.0, 5, 3)) == (zeros(5), zeros(3))
    @test firstComponentFactor(fill(0.1, 7, 2)) == (zeros(7), zeros(2))
    @test_throws ArgumentError firstComponentFactor(zeros(1, 3))
    @test_throws ArgumentError firstComponentFactor(fill(NaN, 3, 2))
    @test_throws ArgumentError firstComponentFactor(zeros(3, 0))
    @test_throws ArgumentError firstComponentFactor(reshape([-1e308,1e308],2,1))

    # Check the minimum rank-one squared error against all singular values.
    y = randn(MersenneTwister(73), 9, 14)
    r = PCA1LevelEstimator(y)
    singular_values = svdvals(y .- mean(y; dims=1))
    @test sum(abs2, r.residuals) ≈ sum(abs2, singular_values[2:end])
end

@testset "Two-stage hierarchical PCA" begin
    # Orthogonal time patterns and disjoint group supports make this exact.
    g = [-1.0, -1, 1, 1, -1, -1, 1, 1]
    a = [-1.0, 1, -1, 1, -1, 1, -1, 1]
    b = [-1.0, -1, -1, -1, 1, 1, 1, 1]
    y = 4g * ones(4)' + a * [1.0,-1,0,0]' + b * [0.0,0,1,-1]'
    settings = HDFMStruct(nlevels=2, nfactors=[1,2], factorassign=[1 1;1 1;1 2;1 2],
                           factorlags=[1,1], errorlags=zeros(Int,4), ndraws=1, burnin=0)
    r = PCA2LevelEstimator(y, settings)
    @test size(r.factors) == (8, 3)
    @test norm(r.residuals) < 1e-12
    @test all(iszero, r.loadings[1:2,3])
    @test all(iszero, r.loadings[3:4,2])
    @test r.intercepts' .+ r.factors * r.loadings' + r.residuals ≈ y
    @test_throws DimensionMismatch PCA2LevelEstimator(y[:,1:3], settings)
    settings.factorassign[1,2] = 3
    @test_throws ArgumentError PCA2LevelEstimator(y, settings)
end

@testset "PCA preserves representable extreme and near-constant scales" begin
    # The leading singular value exceeds Float64, although each loading and
    # fitted observation is finite. Scaling before SVD avoids infinite results.
    pattern = repeat([0.0, 1.0], 4)
    huge = 1e308 .* repeat(pattern, 1, 4)
    result = PCA1LevelEstimator(huge)
    @test all(isfinite, result.factors)
    @test all(isfinite, result.loadings)
    @test all(isfinite, result.residuals)
    reconstructed = (result.intercepts / 1e308)' .+
                    result.factors * (result.loadings / 1e308)' + result.residuals / 1e308
    @test reconstructed ≈ huge / 1e308 atol=1e-14
    @test sum(abs2, result.factors) / length(pattern) ≈ 1

    # Center columns at their own scales; a small column must not disappear
    # just because a different column is many orders of magnitude larger.
    other = repeat([0.0, 0.0, 1.0, 1.0], 2)
    mixed = PCA1LevelEstimator(hcat(1e200 .* pattern, 1e-200 .* other))
    @test mixed.intercepts[2] / 1e-200 ≈ 0.5
    @test mixed.residuals[:, 2] ./ 1e-200 ≈ other .- 0.5 atol=1e-14

    # A tiny perfectly correlated column still belongs to the rank-one fit,
    # even when its scale relative to the largest column underflows to zero.
    correlated = PCA1LevelEstimator(hcat(1e200 .* pattern, 1e-200 .* pattern))
    @test correlated.loadings[2, 1] > 0
    @test correlated.factors[:, 1] .* (correlated.loadings[2, 1] / 1e-200) ≈
          pattern .- 0.5 atol=1e-14
    @test correlated.residuals[:, 2] ./ 1e-200 ≈ zeros(length(pattern)) atol=1e-14

    step = eps(1e10)
    near_constant = reshape(1e10 .+ (0:4) .* step, :, 1)
    f, b = firstComponentFactor(near_constant)
    @test only(b) > 0
    @test vec(f * b') ≈ (-2:2) .* step atol=1e-20
    @test PCA1LevelEstimator(fill(0.1, 7, 2)).residuals == zeros(7, 2)
    @test_throws ArgumentError PCA1LevelEstimator(fill(big"1e1000", 3, 1))
end
