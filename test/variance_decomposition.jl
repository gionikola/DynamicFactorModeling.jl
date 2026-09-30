@testset "Variance accounting" begin
    f = [-1.0, -1, 1, 1]
    g = [-1.0, 1, -1, 1]
    e = [1.0, -1, -1, 1]
    factors = hcat(f, g)
    y = reshape(7 .+ 2f .+ 3g .+ e, :, 1)
    loading = [2.0 3.0]
    account = variance_decomposition(y, factors, loading; intercepts=[7])
    @test account.factors ≈ [4/14 9/14]
    @test account.residual ≈ [1/14]
    @test account.covariance ≈ [0] atol=1e-14
    shares = vardecomp2level(y, factors, [7.0 2.0 3.0], [1 1])
    @test shares ≈ [4/14 9/14]

    # Correlated components require a covariance term, even without residuals.
    correlated = hcat(f, 2f)
    y2 = reshape(3f, :, 1)
    account2 = variance_decomposition(y2, correlated, ones(1, 2))
    @test account2.factors ≈ [1/9 4/9]
    @test account2.covariance ≈ [4/9]
    @test sum(account2.factors) + only(account2.residual) + only(account2.covariance) ≈ 1
    @test_throws ArgumentError variance_decomposition(ones(4,1), factors, loading)
    @test_throws ArgumentError variance_decomposition(fill(0.1,7,1), zeros(7,2), loading)
    @test_throws DimensionMismatch variance_decomposition(y, factors[1:3,:], loading)
    @test_throws ArgumentError vardecomp2level(y, factors, [7.0 2.0 3.0], [1 2])
    integer_loading = reshape([10^10], 1, 1)
    integer_factor = reshape([-1,0,1], 3, 1)
    integer_data = integer_factor * integer_loading'
    @test variance_decomposition(integer_data, integer_factor, integer_loading).factors ≈ ones(1,1)
    @test_throws ArgumentError variance_decomposition(y, factors, loading; intercepts=ones(1,1))
end

@testset "Long-run model variance shares" begin
    model = HDFM(nlevels=1, nvar=2, nfactors=[1], fassign=ones(Int,2,1),
        flags=[1], varlags=[1,0], varcoefs=[4.0 2.0; -7.0 3.0],
        varlagcoefs=reshape([0.5,0.0], 2, 1), fcoefs=[reshape([0.8], 1, 1)],
        fvars=[[1.0]], varvars=[0.75,2.0])
    shares = variance_decomposition(model)
    contributions = [4.0,9.0] ./ (1 - 0.8^2)
    residual = [1.0,2.0]
    @test shares.total ≈ contributions + residual
    @test vec(shares.factors) ≈ contributions ./ (contributions + residual)
    @test vec(sum(shares.factors; dims=2)) + shares.residual ≈ ones(2)
    @test shares.covariance == zeros(2)
end

@testset "Hierarchical variance shares with unequal AR orders" begin
    # Factor order is global, local 1, local 2. Series 3 has no local factor;
    # its finite placeholder loading must be ignored.
    model = HDFM(nlevels=2, nvar=4, nfactors=[1,2],
        fassign=[1 1; 1 2; 1 0; 1 2], flags=[1,2], varlags=[0,1,2,0],
        varcoefs=[3.0 2.0 3.0; -4.0 -1.0 0.5; 10.0 0.75 987.0; 0.0 0.0 -2.0],
        varlagcoefs=[0.0 0.0; 0.5 0.0; 0.2 -0.25; 0.0 0.0],
        fcoefs=[reshape([0.5],1,1), [0.4 -0.2; -0.3 0.1]],
        fvars=[[0.75], [0.8,0.7]], varvars=[2.0,0.75,0.6,1.25])

    # Yule-Walker gives Var(AR2) = s²(1-b) / ((1+b)((1-b)²-a²)).
    # This reference uses only scalar formulas, not package state indices or
    # covariance helpers. AR0 variance is s²; AR1 variance is s²/(1-a²).
    ar2_variance(a, b, innovation_variance) =
        innovation_variance * (1-b) / ((1+b) * ((1-b)^2-a^2))
    global_variance = 0.75 / (1-0.5^2)
    local1_variance = ar2_variance(0.4, -0.2, 0.8)
    local2_variance = ar2_variance(-0.3, 0.1, 0.7)
    error_variances = [2.0, 0.75/(1-0.5^2), ar2_variance(0.2,-0.25,0.6), 1.25]
    factor_variances = [4global_variance 9local1_variance 0.0;
                       global_variance 0.0 0.25local2_variance;
                       0.75^2*global_variance 0.0 0.0;
                       0.0 0.0 4local2_variance]
    level_variances = [4global_variance 9local1_variance;
                      global_variance 0.25local2_variance;
                      0.75^2*global_variance 0.0;
                      0.0 4local2_variance]
    total = [4global_variance + 9local1_variance + error_variances[1],
             global_variance + 0.25local2_variance + error_variances[2],
             0.75^2*global_variance + error_variances[3],
             4local2_variance + error_variances[4]]

    shares = variance_decomposition(model)
    level_shares = hcat(shares.factors[:,1], vec(sum(shares.factors[:,2:3]; dims=2)))
    @test size(shares.factors) == (4,3)
    @test shares.total ≈ total
    @test shares.factors .* shares.total ≈ factor_variances
    @test shares.factors ≈ factor_variances ./ total
    @test level_shares .* shares.total ≈ level_variances
    @test level_shares ≈ level_variances ./ total
    @test shares.residual .* shares.total ≈ error_variances
    @test shares.residual ≈ error_variances ./ total
    @test shares.covariance == zeros(4)
    @test vec(sum(shares.factors; dims=2)) + shares.residual ≈ ones(4)
    @test shares.factors[3,2:3] == zeros(2)

    changed_loading = deepcopy(model)
    changed_loading.varcoefs[3,3] = -12345.0
    @test variance_decomposition(changed_loading) == shares
    changed_padding = deepcopy(model)
    changed_padding.varlagcoefs[2,2] = 9.0
    @test variance_decomposition(changed_padding) == shares
end

@testset "Variance accounting is independent of offsets and factor units" begin
    data = reshape([-1.0, 0.0, 1.0], :, 1)
    no_factors = zeros(3, 0)
    no_loadings = zeros(1, 0)
    baseline = variance_decomposition(data, no_factors, no_loadings)
    large_intercept = variance_decomposition(data, no_factors, no_loadings; intercepts=[1e20])
    @test baseline.residual ≈ [1.0]
    @test large_intercept.residual == baseline.residual
    @test large_intercept.covariance == baseline.covariance
    unused_factor = variance_decomposition(data, 1e308 .* data, zeros(1,1))
    @test unused_factor.factors == zeros(1,1)
    @test unused_factor.residual == baseline.residual

    # Only the product of loading and factor determines a contribution.
    # Their separate squares/variances would overflow and underflow here.
    for scale in (1e-200, 1e200)
        result = variance_decomposition(data, scale .* data, reshape([1 / scale], 1, 1))
        @test result.factors ≈ ones(1, 1)
        @test result.residual ≈ [0.0] atol=1e-28
        @test result.covariance ≈ [0.0] atol=1e-14
    end

    f = [-1.0, -1, 1, 1]
    g = [-1.0, 1, -1, 1]
    factors = hcat(f, f + g)
    loading = [2.0 -1.0]
    residual = 0.4f + 0.3g
    observed = reshape(7 .+ vec(factors * loading') + residual, :, 1)
    account = variance_decomposition(observed, factors, loading; intercepts=[7])
    components = hcat(2f, -(f + g), residual)
    covariance = cov(components)
    total = only(var(observed; dims=1))
    @test vec(account.factors) ≈ diag(covariance)[1:2] / total
    @test account.residual ≈ [covariance[3, 3] / total]
    cross_share = 2 * (covariance[1, 2] + covariance[1, 3] + covariance[2, 3]) / total
    @test account.covariance ≈ [cross_share]
    @test account.covariance[1] < 0
    @test account.factors[1] > 1

    # Marginal-variance normalization can be defined for constant observed
    # data whose varying components cancel. Exact observed-variance shares cannot.
    cancellation = hcat(f, -f)
    @test vardecomp2level(zeros(4, 1), cancellation, [0.0 1.0 1.0], [1 1]) ≈ [0.5 0.5]
    @test_throws ArgumentError variance_decomposition(zeros(4, 1), cancellation, ones(1, 2))
    @test_throws ArgumentError vardecomp2level(zeros(4, 1), zeros(4, 2), zeros(1, 3), [1 1])

    # An unrepresentable ratio must be rejected, rather than returning Inf
    # and a covariance share of -Inf that cannot form a variance account.
    @test_throws ArgumentError variance_decomposition(1e-150 .* data, 1e150 .* data, ones(1, 1))
    @test_throws ArgumentError variance_decomposition(data, data, ones(1, 1); intercepts=[Inf])
end

@testset "Long-run variance calculations respect factor units" begin
    model = HDFM(nlevels=1, nvar=1, nfactors=[1], fassign=ones(Int, 1, 1),
        flags=[0], varlags=[0], varcoefs=[0.0 1e200],
        varlagcoefs=zeros(1, 0), fcoefs=[zeros(1, 0)],
        fvars=[[1e-300]], varvars=[2e100])
    account = variance_decomposition(model)
    @test account.total ≈ [3e100]
    @test account.factors ≈ fill(1 / 3, 1, 1)
    @test account.residual ≈ [2 / 3]

    overflowing = HDFM(nlevels=1, nvar=1, nfactors=[1], fassign=ones(Int, 1, 1),
        flags=[0], varlags=[0], varcoefs=[0.0 1.0],
        varlagcoefs=zeros(1, 0), fcoefs=[zeros(1, 0)],
        fvars=[[1e308]], varvars=[1e308])
    @test_throws ArgumentError variance_decomposition(overflowing)
end
