var documenterSearchIndex = {"docs":
[{"location":"#DynamicFactorModeling.jl","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.jl","text":"","category":"section"},{"location":"","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.jl","text":"Documentation for DynamicFactorModeling.jl","category":"page"},{"location":"","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.jl","text":"Modules = [DynamicFactorModeling]","category":"page"},{"location":"#DynamicFactorModeling.DFMMeans","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.DFMMeans","text":"DFMMeans(F::Array{Float64}, B::Array{Float64}, S::Array{Float64}, P::Array{Float64}, P2::Array{Float64})\n\nDescription: HDFM Bayesian estimator-generated latent factor and hyperparameter sample means (expected values). \n\nInputs:\n\nF = MCMC-generated latent factor sample mean.\nB = MCMC-generated observation equation regression coefficient sample means.\nS = MCMC-generated observable variable idiosyncratic error disturbance variance sample means. \nP = MCMC-generated latent factor autoregressive coefficient sample means. \nP2 = MCMC-generated idiosyncratic error autoregressive coefficient sample means. \n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.DFMResults","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.DFMResults","text":"DFMResults(F::Array{Float64}, B::Array{Float64}, S::Array{Float64}, P::Array{Float64}, P2::Array{Float64}, means::DFMMeans)\n\nDescription: HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters. \n\nInputs:\n\nF = MCMC-generated latent factor sample.\nB = MCMC-generated observation equation regression coefficient sample.\nS = MCMC-generated observable variable idiosyncratic error disturbance variance sample. \nP = MCMC-generated latent factor autoregressive coefficient sample. \nP2 = MCMC-generated idiosyncratic error autoregressive coefficient sample. \nmeans = HDFM Bayesian estimator-generated latent factor and hyperparameter sample means (expected values).\n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.DFMStruct","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.DFMStruct","text":"DFMStruct(factorlags::Int64, errorlags::Int64, ndraws::Int64, burnin::Int64)\n\nDescription: 1-level DFM lag structure specification and MCMC sample size for Bayesian estimation. \n\nInputs:\n\nfactorlags = Number of lags in the autoregressive specification of the latent factors. \nerrorlags = Number of lags in the autoregressive specification of the observable variable idiosyncratic errors.\nndraws = Number of MCMC draws used for posterior distributions.\nburnin = Number of initial MCMC draws discarded. \n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.HDFM","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.HDFM","text":"HDFM(\n    nlevels::Int64                   \n    nvar::Int64                     \n    nfactors::Array{Int64,1}        \n    fassign::Array{Int64,2}          \n    flags::Array{Int64,1}         \n    varlags::Array{Int64,1}        \n    varcoefs::Array{Any,2}          \n    varlagcoefs::Array{Any,2}    \n    fcoefs::Array{Any,1}           \n    fvars::Array{Any,1}             \n    varvars::Array{Any,1}  \n)\n\nCreates an object of type HDFM that contains all parameters necessary to specify a multi-level linear dynamic factor data-generating process. This is a convenient alternative to specifying an HDFM directly in state-space form. \n\nInputs: \n\nnlevels = number of levels in the multi-level model structure.\nnvar = number of variables.\nnfactors = number of factors for each level (vector of length nlevels). \nfassign = determines which factor is assigned to which variable for each level (integer matrix of size nvar × nlevels).\nflags = number of autoregressive lags for factors of each level (factors of the same level are restricted to having the same number of lags; vector of length nlevels).\nvarlags = number of observed variable error autoregressive lags (vector of length nvar).\nvarcoefs = vector of coefficients for each variable in the observation equation (length 1+nlevels, where first entry represents the intercept). \nfcoefs = list of nlevels number of matrices, for which each row contains vectors of the autoregressive lag coefficients of the corresponding factor. \nfvars = list of nlevels number of vectors, where each entry contains the disturbance variance of the corresponding factors.\nvarvars = vector of nvar number of entries, where each entry contains the innovation variance of the corresponding variable.\n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.HDFMStruct","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.HDFMStruct","text":"HDFMStruct(nlevels::Int64, nfactors::Array{Int64,1}, factorassign::Array{Int64,2}, factorlags::Array{Int64,1}, errorlags::Array{Int64,1}, ndraws::Int64, burnin::Int64)\n\nDescription: Multi-level/hierarchical DFM (HDFM) level, factor assignment, and lag structure specification, and MCMC sample size for Bayesian estimation. \n\nInputs:\n\nnlevels = Number of levels in the HDFM specification. \nnvars = Number of observable variables in the HDFM specification. \nnfactors = Number of factor per level in the HDFM specification. \nfactorassign = Factors assigned to each variable across all levels. \nfactorlags = Number of lags in the autoregressive specification of the latent factors. \nerrorlags = Number of lags in the autoregressive specification of the observable variable idiosyncratic errors.\nndraws = Number of MCMC draws used for posterior distributions.\nburnin = Number of initial MCMC draws discarded. \n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.SSModel","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.SSModel","text":"SSModel(     H::Array{Float64,2},      A::Array{Float64,2},      F::Array{Float64,2},      μ::Array{Float64,1},      R::Array{Float64,2},      Q::Array{Float64,2},      Z::Array{Float64,2} )\n\nA type object containing all parameters necessary to specify a data-generating process in state-space form.  Measurement Equation:   \n\ny{t} = H β{t} + A z{t} + e{t} \n\nTransition Equation:    \n\nβ{t} = μ + F β{t-1} + v_{t}\ne_{t} ~ i.i.d.N(0,R)\nv_{t} ~ i.i.d.N(0,Q)\nz_{t} ~ i.i.d.N(0,Z)\nE(e{t} v{s}') = 0\n\nInputs:\n\nH = measurement equation state vector coefficient matrix.\nA = measurement equation predetermined vector coefficient matrix. \nF = state equation companion matrix.\nμ = state equation intercept vector.\nR = measurement equation error covariance matrix. \nQ = state equation innovation covariance matrix.\nZ = predetermined vector covariance matrix.\n\n\n\n\n\n","category":"type"},{"location":"#DynamicFactorModeling.KN1LevelEstimator-Tuple{Matrix{Float64}, DFMStruct}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.KN1LevelEstimator","text":"KNSingleFactorEstimator(data::Array{Float64,2}, dfm::DFMStruct)\n\nDescription: Estimate a single-factor DFM using the Kim-Nelson approach. \n\nInputs:\n\ndata = Matrix with each column being a data series. \ndfm = Model structure specification. \n\nOutputs:\n\nresults = HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters.\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.KN2LevelEstimator-Tuple{Matrix{Float64}, DynamicFactorModeling.HDFMStruct}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.KN2LevelEstimator","text":"KN2LevelEstimator(data::Array{Float64,2}, hdfm::HDFMStruct)\n\nDescription: Estimate a two-level HDFM using the Kim-Nelson approach.  Both the latent factors and hyperparameters are estimated using the Bayesian approach outlined in Kim and Nelson (1999).   \n\nInputs:\n\ndata = Matrix with each column being a data series. \nhdfm = Model structure specification. \n\nOutputs:\n\nresults = HDMF Bayesian estimator-generated MCMC posterior distribution samples and their means for latent factors and hyperparameters.\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.KNFactorSampler-Tuple{Any, Any}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.KNFactorSampler","text":"KNFactorSampler(data_y, ssmodel)\n\nDescription:  Draw a sample series of dynamic factor from conditional distribution in Ch 8, Kim & Nelson (1999). Measurement Equation:        y{t} = H{t} β{t} + A z{t} + e{t}. Transition Equation:         β{t} = μ + F β{t-1} + v{t};     e{t} ~ i.i.d.N(0,R);     v{t} ~ i.i.d.N(0,Q);     z{t} ~ i.i.d.N(0,Z);     E(et v_s') = 0.\n\nInputs: \n\ndata      = observed data \nH         = measurement eq. state coef. matrix\nA         = measurement eq. exogenous coef. matrix\nF         = state eq. companion matrix\nμ         = state eq. intercept term\nR         = covariance matrix on measurement disturbance\nQ         = covariance matrix on state disturbance\nZ         = covariance matrix on predetermined var vector \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.convertHDFMtoSS-Tuple{HDFM}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.convertHDFMtoSS","text":"convertHDFMtoSS(hdfm::HDFM)\n\nDescription: Converts an HDFM object to an SSModel object. \n\nInputs:\n\nhdfm::HDFM\n\nOutput:\n\nssmodel::SSModel \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.createSSforHDFM-Tuple{HDFM}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.createSSforHDFM","text":"createSSforHDFM(hdfm::HDFM))\n\nDescription: Create state-space form coefficient and variance matrices for an HDFM object. Measurement Equation:   \n\ny{t} = H β{t} + A z{t} + e{t} \n\nTransition Equation:    \n\nβ{t} = μ + F β{t-1} + v_{t}\ne_{t} ~ i.i.d.N(0,R)\nv_{t} ~ i.i.d.N(0,Q)\nz_{t} ~ i.i.d.N(0,Z)\nE(e{t} v{s}') = 0\n\nInputs:\n\nhdfm::HDFM\n\nOutput:\n\nH = measurement equation state vector coefficient matrix.\nA = measurement equation predetermined vector coefficient matrix. \nF = state equation companion matrix.\nμ = state equation intercept vector.\nR = measurement equation error covariance matrix. \nQ = state equation innovation covariance matrix.\nZ = predetermined vector covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.draw_parameters-Tuple{Matrix{Float64}, Matrix{Float64}, Float64}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.draw_parameters","text":"draw_parameters()\n\nDescription.\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.draw_parameters-Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Float64}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.draw_parameters","text":"draw_parameters()\n\nDescription.\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.firstComponentFactor-Tuple{Any}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.firstComponentFactor","text":"\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.kalmanFilter-Tuple{Any, Any}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.kalmanFilter","text":"kalmanFilter(data, ssmodel)\n\nDescription:  Apply Kalman filter to observed data.  Measurement Equation:        y{t} = H{t} β{t} + A z{t} + e{t} . Transition Equation:         β{t} = μ + F β{t-1} + v{t};     e{t} ~ i.i.d.N(0,R);     v{t} ~ i.i.d.N(0,Q);     z{t} ~ i.i.d.N(0,Z);     E(et v_s') = 0.\n\nInputs: \n\ndata      = observed data \nH         = measurement eq. state coef. matrix\nA         = measurement eq. exogenous coef. matrix\nF         = state eq. companion matrix\nμ         = state eq. intercept term\nR         = covariance matrix on measurement disturbance\nQ         = covariance matrix on state disturbance\nZ         = covariance matrix on predetermined var vector \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.kalmanSmoother-Tuple{Any, Any}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.kalmanSmoother","text":"kalmanSmoother(data, ssmodel)\n\nDescription:  Apply Kalman smoother to observed data.  Measurement Equation:        y{t} = H{t} β{t} + A z{t} + e{t}. Transition Equation:         β{t} = μ + F β{t-1} + v{t};     e{t} ~ i.i.d.N(0,R);     v{t} ~ i.i.d.N(0,Q);     z{t} ~ i.i.d.N(0,Z);     E(et v_s') = 0.\n\nInputs: \n\ndata      = observed data \nH         = measurement eq. state coef. matrix\nA         = measurement eq. exogenous coef. matrix\nF         = state eq. companion matrix\nμ         = state eq. intercept term\nR         = covariance matrix on measurement disturbance\nQ         = covariance matrix on state disturbance\nZ         = covariance matrix on predetermined var vector \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.mvn-Tuple{Vector{Float64}, Matrix{Float64}, Int64}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.mvn","text":"mvn(μ, Σ, n::Int64)\n\nDraw n number of observations from a multivariate normal distribution with mean vector μ and covariance matrix Σ. Use cholesky decomposition to generate n draws of X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix.  Cov. matrix Σ does not require non-degenerate random variables (nonzero diag.). \n\nInputs:\n\nμ = mean vector \nΣ = covariance matrix \nn = number of draws \n\nOutput:\n\nX::Array{Float64, 2} = simulated data matrix composed of n-number of draws of X ~ N(μ,Σ)\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.mvn-Tuple{Vector{Float64}, Matrix{Float64}}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.mvn","text":"mvn(μ, Σ)\n\nDraw from a multivariate normal distribution with mean vector μ and covariance matrix Σ. Use cholesky decomposition to generate X = Z Q + μ, where Z is (d × 1) N(0,1) vector, and Q is upper-triangular cholesky matrix.  Cov. matrix Σ does not require non-degenerate random variables (nonzero diagonal). \n\nInputs:\n\nμ = mean vector \nΣ = covariance matrix \n\nOutputs:\n\nX::Array{Float64, 1} = observed draw of X ~ N(μ,Σ)\n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.simulateSSModel-Tuple{Int64, DynamicFactorModeling.SSModel}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.simulateSSModel","text":"simulateSSModel(num_obs::Int64, ssmodel::SSModel)\n\nGenerate data from a DGP in state space form. Measurement Equation:        y{t} = H β{t} + A z{t} + e{t}  Transition Equation:         β{t} = μ + F β{t-1} + v{t}     e{t} ~ i.i.d.N(0,R)     v{t} ~ i.i.d.N(0,Q)     z{t} ~ i.i.d.N(0,Z)     E(et vs') = 0\n\nInputs: \n\nnum_obs           = number of observations\nssmodel::SSModel \n\nOutput:\n\ndata_y = simulated sample of observed vector  \ndata_z = simulated sample of exogenous variables\ndata_β = simulated sample of state vector \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.vardecomp2level-NTuple{4, Any}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.vardecomp2level","text":"vardecomp2level(data::Array{Float64, 2}, factor::Array{Float64}, betas::Array{Float64}, factorassign::Array{Float64})\n\nDescription: Compute the portion of the variation of each observable series that may be attributed to their corresponding/assigned latent factors across all levels. \n\nInputs: \n\ndata = Matrix with each column representing a data series. \nfactor = Matrix containing latent factor estimates.\nbetas = Matrix containing observation equation coefficient parameter estimates.\nfactorassign = Matrix containing the indeces of factors across all levels (columns) assigned to each observable series (rows). \n\nOutput: \n\nvardecomps = Matrix containing the variance contributions of factors across all levels (columns) corresponding to each observable series (rows). \n\n\n\n\n\n","category":"method"},{"location":"#DynamicFactorModeling.Γinv-Tuple{Int64, Float64}","page":"DynamicFactorModeling.jl","title":"DynamicFactorModeling.Γinv","text":"Γinv(T::Int64, θ::Float64)\n\nGamma inverse distribution with T degrees of freedom and scale parameter θ.\n\nInputs:\n\nT = degrees of freedom \nθ = scale parameter \n\nOutput:\n\nσ2 = draw of X ~ Γ_inverse(T,θ)\n\n\n\n\n\n","category":"method"}]
}
