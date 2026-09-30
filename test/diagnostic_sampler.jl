# Standalone, short instrumented-sampler checks; no convergence study is run.
# julia --project=. test/diagnostic_sampler.jl
using Test, Random, LinearAlgebra, Statistics, DynamicFactorModeling
include(joinpath(@__DIR__, "support", "location_move.jl"))
include(joinpath(@__DIR__, "support", "diagnostic_sampler.jl"))
include(joinpath(@__DIR__, "reference", "simulation_scenarios.jl"))
using .DiagnosticSampler, .SimulationScenarios

BLAS.set_num_threads(1)

function diagnostic_settings(case, ndraws, burnin)
    spec = case.spec
    return spec.estimator_kind == :single ?
        DFMStruct(only(case.factor_orders), only(unique(case.error_orders)), ndraws, burnin) :
        HDFMStruct(spec.nlevels, spec.nfactors, spec.assignments,
                   case.level_orders, case.error_orders, ndraws, burnin)
end

function same_results(left, right)
    for field in (:F, :B, :S, :P, :P2)
        @test getproperty(left, field) == getproperty(right, field)
        @test getproperty(left.means, field) == getproperty(right.means, field)
    end
end

function check_retained_features(output, case, burnin)
    result, spec = output.result, case.spec
    paths = ndims(result.F) == 2 ? reshape(result.F, spec.dates, 1, :) : result.F
    columns = Dict(name => index for (index, name) in enumerate(output.trace_names))
    @test length(columns) == length(output.trace_names)
    for draw in axes(result.B, 1)
        coefficients = permutedims(reshape(result.B[draw,:], spec.nlevels+1, :))
        loadings = zeros(size(case.data,2), sum(spec.nfactors))
        offset = 0
        for level in 1:spec.nlevels
            for series in axes(case.data, 2)
                local_factor = spec.assignments[series,level]
                iszero(local_factor) && continue
                loadings[series, offset+local_factor] = coefficients[series,level+1]
            end
            offset += spec.nfactors[level]
        end
        signal = paths[:,:,draw] * loadings' .+ coefficients[:,1]'
        trace = output.trace[burnin+draw,:]
        for series in axes(case.data, 2)
            @test trace[columns["intercept_$series"]] == coefficients[series,1]
            @test trace[columns["signal_mean_$series"]] ≈ mean(signal[:,series]) atol=1e-13
            @test trace[columns["error_variance_$series"]] == result.S[draw,series]
        end
        for factor in axes(paths, 2)
            values = paths[:,factor,draw]
            @test trace[columns["factor_mean_$factor"]] ≈ sum(values)/length(values)
            @test trace[columns["factor_abs_mean_$factor"]] == abs(trace[columns["factor_mean_$factor"]])
            @test trace[columns["factor_rms_$factor"]] ≈ std(values; corrected=false)
            @test trace[columns["factor_contrast_$factor"]] == values[end]-values[1]
            @test trace[columns["loading_norm_$factor"]] ≈ sqrt(sum(abs2,loadings[:,factor]))
            contribution = (values .- mean(values)) * loadings[:,factor]'
            @test trace[columns["contribution_rms_$factor"]] ≈ sqrt(sum(abs2,contribution)/length(values))
            @test trace[columns["anchor_loading_$factor"]] == loadings[case.sign_anchors[factor],factor]
            @test trace[columns["anchor_loading_$factor"]] >= 0
        end
    end
end

@testset "Instrumented baseline exactly matches public estimators" begin
    estimators = Dict("KN1"=>KN1LevelEstimator, "OW1"=>OW1LevelEstimator,
                      "KN2"=>KN2LevelEstimator, "OW2"=>OW2LevelEstimator,
                      "KNHierarchical"=>KNHierarchicalEstimator)
    ndraws, burnin = 3, 2
    for (scenario_index, spec) in enumerate(scenarios())
        case = generate_case(spec, MersenneTwister(1700+scenario_index))
        methods = spec.estimator_kind == :single ? ["KN1","OW1"] :
                  spec.estimator_kind == :two_level ? ["KN2","OW2"] : ["KNHierarchical"]
        max_order = max(maximum(case.factor_orders), maximum(case.error_orders))
        alternative = (; beta_prior_variance=[7.0; fill(0.4,spec.nlevels)],
                         ar_prior_variance=[1.3/(lag^2) for lag in 1:max_order],
                         variance_shape=3.0, variance_scale=0.8)
        supplied_start = randn(MersenneTwister(3800+scenario_index), size(case.factors))
        for (method_index, method) in enumerate(methods), initial_factors in (nothing,supplied_start),
            priors in (NamedTuple(),alternative)
            seed = 6400+100*scenario_index+method_index
            rng_public, rng_diagnostic = MersenneTwister(seed), MersenneTwister(seed)
            expected = estimators[method](rng_public,case.data,diagnostic_settings(case,ndraws,burnin);
                                           initial=spec.initial, initial_factors,
                                           mixing_moves=:none, priors...)
            move_rng = MersenneTwister(seed+1)
            untouched_move_rng = copy(move_rng)
            output = sample_case(case,method,rng_diagnostic; ndraws,burnin,initial_factors,
                                  variant=:baseline,move_rng,priors...)
            same_results(output.result, expected)
            @test rand(rng_public,8) == rand(rng_diagnostic,8)
            @test rand(move_rng,8) == rand(untouched_move_rng,8)
            @test size(output.trace) == (ndraws+burnin, length(output.trace_names))
            for family in ("factor", "error")
                count = family == "factor" ? sum(spec.nfactors) : size(case.data,2)
                for index in 1:count
                    @test sum(number for ((f,i,_),number) in output.events if f==family && i==index) == ndraws
                end
            end
            @test all(0 .<= output.sign_flips .<= ndraws)
            check_retained_features(output,case,burnin)
        end
    end
end

@testset "Instrumented mixing updates exactly match public estimators" begin
    estimators = Dict("KN1"=>KN1LevelEstimator, "OW1"=>OW1LevelEstimator,
                      "KN2"=>KN2LevelEstimator, "OW2"=>OW2LevelEstimator,
                      "KNHierarchical"=>KNHierarchicalEstimator)
    for (scenario_index, spec) in enumerate(scenarios())
        case = generate_case(spec, MersenneTwister(1750 + scenario_index))
        methods = spec.estimator_kind == :single ? ["KN1", "OW1"] :
                  spec.estimator_kind == :two_level ? ["KN2", "OW2"] : ["KNHierarchical"]
        priors = (; beta_prior_variance=[7.0; fill(0.4, spec.nlevels)],
                    variance_shape=3.0, variance_scale=0.8)
        for method in methods, variant in (:location, :location_scale)
            public_rng, shared_rng = MersenneTwister(6550), MersenneTwister(6550)
            expected = estimators[method](public_rng, case.data, diagnostic_settings(case, 3, 2);
                initial=spec.initial, mixing_moves=variant, priors...)
            # One shared object makes draw order directly comparable with the
            # public API, which deliberately exposes only one RNG.
            actual = sample_case(case, method, shared_rng; ndraws=3, burnin=2,
                variant, move_rng=shared_rng, scale_rng=shared_rng, priors...)
            same_results(actual.result, expected)
            @test rand(public_rng, 8) == rand(shared_rng, 8)
        end
    end
end

@testset "Events exclude warmup and do not change RNG consumption" begin
    spec = scenarios()[6]
    case = generate_case(spec, MersenneTwister(1717))
    retained = sample_case(case,"KNHierarchical",MersenneTwister(619); ndraws=3,burnin=2)
    complete = sample_case(case,"KNHierarchical",MersenneTwister(619); ndraws=5,burnin=0)
    warmup = sample_case(case,"KNHierarchical",MersenneTwister(619); ndraws=2,burnin=0)
    @test retained.trace == complete.trace
    @test retained.result.F == complete.result.F[:,:,3:5]
    for event in union(keys(retained.events),keys(complete.events),keys(warmup.events))
        @test get(retained.events,event,0) == get(complete.events,event,0)-get(warmup.events,event,0)
    end
    @test retained.sign_flips == complete.sign_flips-warmup.sign_flips

    # A one-observation stationary AR step has no regression likelihood. Broad
    # normal proposals exercise both rejection causes as well as acceptance.
    statuses = Set{Symbol}()
    for seed in 1:100
        rng_control, rng_observed = MersenneTwister(seed), MersenneTwister(seed)
        previous = [0.8]
        actual,status = DiagnosticSampler.stationary_ar_step(rng_observed,[2.0],previous,1.0,4.0)
        expected = DynamicFactorModeling._draw_stationary_ar(rng_control,[2.0],previous,1.0;
                                                            prior_variance=4.0)
        @test actual == expected
        @test rand(rng_control,3) == rand(rng_observed,3)
        @test (actual != previous) == (status == :accepted)
        push!(statuses,status)
    end
    @test statuses == Set((:accepted,:unstable,:mh_rejected))
    rng = MersenneTwister(922)
    original = copy(rng)
    @test DiagnosticSampler.stationary_ar_step(rng,[2.0],Float64[],1.0,Float64[]) == (Float64[],:absent)
    @test rand(rng,3) == rand(original,3)
end

@testset "Companion radius and independent feature identities" begin
    @test companion_radius(Float64[]) == 0
    @test companion_radius([-0.7]) == 0.7
    # The AR(2) characteristic roots are 0.5 and -0.2.
    @test companion_radius([0.3,0.1]) ≈ 0.5
    # Complex-conjugate roots with product 0.25 have modulus 0.5.
    @test companion_radius([0.2,-0.25]) ≈ 0.5
    factors = [1.0 -2.0; 2.0 1.0; 5.0 4.0]
    loadings = [0.7 -0.2; 0.0 0.5]
    coefficients = [0.3 0.7 -0.2; -0.6 0.0 0.5]
    names,values = DiagnosticSampler.trace_features(factors,coefficients,loadings,
                                                    [[-0.7],[0.3,0.1]], [Float64[],[0.2]], [1,2];
                                                    error_variances=[0.2,0.8])
    traces = Dict(zip(names,values))
    @test traces["factor_radius_1"] == 0.7
    @test traces["factor_radius_2"] ≈ 0.5
    @test traces["error_radius_2"] == 0.2
    @test !haskey(traces,"error_radius_1")
    @test traces["error_variance_1"] == 0.2
    @test traces["error_variance_2"] == 0.8
    @test traces["factor_correlation_1_2"] ≈ cor(factors[:,1],factors[:,2])
    for factor in 1:2
        # For three dates these fixed orthonormal contrasts have closed forms.
        @test traces["factor_linear_contrast_$factor"] ≈
              dot([-1.0,0.0,1.0]/sqrt(2),factors[:,factor])
        @test traces["factor_quadratic_contrast_$factor"] ≈
              dot([1.0,-2.0,1.0]/sqrt(6),factors[:,factor]) atol=1e-14
        contribution = (factors[:,factor] .- mean(factors[:,factor])) * loadings[:,factor]'
        @test traces["contribution_rms_$factor"] ≈ sqrt(sum(abs2,contribution)/3)
    end
    signal = factors*loadings' .+ coefficients[:,1]'
    @test [traces["signal_mean_$i"] for i in 1:2] ≈ vec(mean(signal;dims=1))

    # A location shift preserves signal means, centered scale, endpoint contrast,
    # correlations, and loadings. Only factor means and intercepts should change.
    shifts = [2.0,-0.7]
    shifted_coefficients = copy(coefficients)
    shifted_coefficients[:,1] -= loadings*shifts
    new_names,new_values = DiagnosticSampler.trace_features(factors .+ shifts',shifted_coefficients,
                          loadings,[[-0.7],[0.3,0.1]],[Float64[],[0.2]],[1,2];
                          error_variances=[0.2,0.8])
    shifted = Dict(zip(new_names,new_values))
    for name in names
        if !startswith(name,"intercept_") && !startswith(name,"factor_mean_") &&
           !startswith(name,"factor_abs_mean_")
            @test shifted[name] ≈ traces[name] atol=1e-14
        end
    end
    for dates in (1,2)
        names,values = DiagnosticSampler.trace_features(ones(dates,1),ones(1,2),ones(1,1),
                                                        [Float64[]],[Float64[]],[1])
        simple = Dict(zip(names,values))
        @test simple["factor_linear_contrast_1"] == 0
        @test simple["factor_quadratic_contrast_1"] == 0
        @test !haskey(simple,"error_variance_1")
    end
    @test_throws DimensionMismatch DiagnosticSampler.trace_features(factors,coefficients,loadings,
                           [[-0.7],[0.3,0.1]],[Float64[],[0.2]],[1,2];error_variances=[0.2])
end

@testset "Location move follows ordinary updates without changing their signal" begin
    for spec in scenarios()[[2,3,5,6]]
        case = generate_case(spec,MersenneTwister(487))
        method = spec.estimator_kind == :single ? "OW1" :
                 spec.estimator_kind == :two_level ? "OW2" : "KNHierarchical"
        priors = (; beta_prior_variance=[3.0; fill(0.2,spec.nlevels)])
        baseline = sample_case(case,method,MersenneTwister(32);ndraws=1,burnin=0,priors...)
        moved = sample_case(case,method,MersenneTwister(32);ndraws=1,burnin=0,
                             variant=:location,move_rng=MersenneTwister(86),priors...)
        left, right = Dict(zip(baseline.trace_names,baseline.trace[1,:])),
                      Dict(zip(moved.trace_names,moved.trace[1,:]))
        for name in baseline.trace_names
            if !startswith(name,"intercept_") && !startswith(name,"factor_mean_") &&
               !startswith(name,"factor_abs_mean_")
                @test left[name] ≈ right[name] atol=1e-12
            end
        end
        @test baseline.result.S == moved.result.S
        @test baseline.result.P == moved.result.P
        @test baseline.result.P2 == moved.result.P2
        @test baseline.events == moved.events
        @test baseline.sign_flips == moved.sign_flips
        check_retained_features(moved,case,0)
    end
end
