
####################
####################
####################
####################
####################
using StatsBase, Plots;
pyplot();

names = ["Mary", "Mel", "David",
        "John", "Kayley", "Anderson"]
randomName() = rand(names)
X = 3:8
N = 10^6
sampleLengths = [length(randomName()) for _ = 1:N]

bar(X,
        counts(sampleLengths) / N,
        ylims = (0, 0.35),
        xlabel = "Name length",
        ylabel = "Estimated p(x)",
        legend = :none)

####################
####################
####################
####################
####################
using Plots, Measures;
pyplot();

pDiscrete = [0.25, 0.25, 0.5]
xGridD = 0:2

pContinuous(x) = 3 / 4 * (1 - x^2)
xGridC = -1:0.01:1

pContinuous2(x) = x < 0 ? x + 1 : 1 - x

p1 = plot(xGridD, line = :stem, pDiscrete, marker = :circle, c = :blue, ms = 6, msw = 0)
p2 = plot(xGridC, pContinuous.(xGridC), c = :blue)
p3 = plot(xGridC, pContinuous2.(xGridC), c = :blue)

plot(p1, p2, p3,
        layout = (1, 3), legend = false, ylims = (0, 1.1),
        xlabel = "x",
        ylabel = ["Probability" "Density" "Density"],
        size = (1200, 400), margin = 5mm)

####################
####################
####################
####################
####################
using QuadGK

sup = (-1, 1)
f1(x) = 3 / 4 * (1 - x^2)
f2(x) = x < 0 ? x + 1 : 1 - x

expect(f, support) = quadgk((x) -> x * f(x), support...)[1]

println("Mean 1: ", expect(f1, sup))
println("Mean 2: ", expect(f2, sup))

####################
####################
####################
####################
####################
using QuadGK
using Plots

funk(x, μ, σ) = (1 / (σ * sqrt(2 * π))) * exp((-1 / 2) * ((x - μ) / σ)^2)
funk2(x) = funk(x, 0, 1)
funk3(x) = funk(x, 0, 5)
sup = (-10, 10)
expect(f, support) = quadgk((x) -> x * f(x), support...)[1]
println("Mean: ", expect(funk2, sup))

xGrid = -10:0.01:10

plot(xGrid, funk3.(xGrid), c = :blue)

####################
####################
####################
####################
####################
using Plots, LaTeXStrings;
pyplot();

f2(x) = (x < 0 ? x + 1 : 1 - x) * (abs(x) < 1 ? 1 : 0)
a, b = -1.5, 1.5
delta = 0.01

F(x) = sum([f2(u) * delta for u = a:delta:x])

xGrid = a:delta:b
y = [F(u) for u in xGrid]
plot(xGrid, y, c = :blue, xlims = (a, b), ylims = (0, 1),
        xlabel = L"x", ylabel = L"F(x)", legend = :none)

####################
####################
####################
####################
####################
using QuadGK, Plots, LaTeXStrings;
pyplot();

f2(x) = (x < 0 ? x + 1 : 1 - x) * (abs(x) < 1 ? 1 : 0)
a, b = -1.5, 1.5
delta = 0.01

F(x) = quadgk(x -> f2(x), (a, x)...)[1]

xGrid = a:delta:b
y = [F(u) for u in xGrid]
plot(xGrid, y, c = :red, xlims = (a, b), ylims = (0, 1),
        xlabel = L"x", ylabel = L"F(x)", legend = :none)

####################
####################
####################
####################
####################
using Distributions, Plots, LaTeXStrings;
pyplot();

dist = TriangularDist(0, 2, 1)
xGrid = 0:0.01:2
uGrid = 0:0.01:1

p1 = plot(xGrid, pdf.(dist, xGrid), c = :blue,
        xlims = (0, 2), ylims = (0, 1.1),
        xlabel = "x", ylabel = "f(x)")

p2 = plot(xGrid, cdf.(dist, xGrid), c = :blue,
        xlims = (0, 2), ylims = (0, 1),
        xlabel = "x", ylabel = "F(x)")

p3 = plot(uGrid, quantile.(dist, uGrid), c = :blue,
        xlims = (0, 1), ylims = (0, 2),
        xlabel = "u", ylabel = L"F^{-1}(u)")

plot(p1, p2, p3, legend = false, layout = (1, 3), size = (1200, 400))

####################
####################
####################
####################
####################
using Plots

num_obs = 100
H = [1.0 0.0]
A = [0.0][:, :]
F = [0.3 0.5; 1.0 0.0]
μ = [0.0, 0.0]
R = [0.0][:, :]
Q = [1.0 0.0; 0.0 0.0]
Z = [0.0][:, :]

data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)
plot(data_y)

data_filtered_y, data_filtered_β, Pttlag, Ptt = kalmanFilter(data_y, data_z, H, A, F, μ, R, Q, Z)

plot(data_β[:, 1])
plot!(data_filtered_β[:, 1])

plot(data_filtered_β[:, 1])

plot(data_y)
plot!(data_filtered_y)

factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
plot(factor[:, 1])

####################
####################
####################
####################
####################

rand(InverseGamma(1, 2))



####################
####################
####################
####################
####################
using Plots

# Specify common components model 
# Section 8.2 in Kim & Nelson (1999) 
num_obs = 100
γ1 = 0.5
γ2 = 0.4
ϕ1 = 0.5
ϕ2 = 0.1
ψ1 = 0.5
ψ2 = 0.5
σ2_1 = 2.0
σ2_2 = 3.0
H = [γ1 1.0 0.0 0.0; γ2 1.0 1.0 0.0]
A = zeros(2, 4)
F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
μ = [0.0, 0.0, 0.0, 0.0]
R = zeros(2, 2)
Q = zeros(4, 4)
Q[1, 1] = 1.0
Q[2, 2] = σ2_1
Q[3, 3] = σ2_2
Z = zeros(4, 4)

# Simulate common components model in state space form 
data_y, data_z, data_β = simulateStateSpaceModel(num_obs, H, A, F, μ, R, Q, Z)

# Plot data 
plot(data_y)
plot!(data_β[:, 1])

#####################################################
#####################################################

function HDFMStateSpaceGibbsSampler(data_y, data_z)

        # Initialize factor 
        factor = rand(size(data_y)[1])

        # Specify the number of errors lags obs eq. 
        error_lag_num = 1

        for i = 1:1000
                
                print(i) 
                # Estimate obs eq. and autoreg hyperparameters
                γ1, σ2_1, ψ1 = autocorrErrorRegGibbsSampler(reshape(data_y[:, 1], length(data_y[:, 1]), 1), reshape(factor, length(factor), 1), error_lag_num)
                γ2, σ2_2, ψ2 = autocorrErrorRegGibbsSampler(reshape(data_y[:, 2], length(data_y[:, 2]), 1), reshape(factor, length(factor), 1), error_lag_num)
                γ1 = γ1[1][1]
                σ2_1 = σ2_1[1]
                ψ1 = ψ1[1][1] 
                γ2 = γ2[1][1] 
                σ2_2 = σ2_2[1] 
                ψ2 = ψ2[1][1] 

                # Estimate factor autoreg state eq. hyperparmeters
                σ2 = 1
                X = zeros(size(data_y)[1], 2)
                X = X[3:end, :]
                for j = 1:size(X)[2] # iterate over variables in X
                        x_temp = factor
                        x_temp = lag(x_temp, j)
                        x_temp = x_temp[3:end, :]
                        X[:, j] = x_temp
                end
                factor_temp = factor[3:end, :]
                ϕ = staticLinearGibbsSamplerRestrictedVariance(factor_temp, X, σ2)
                ϕ1 = ϕ[1][1] 
                ϕ2 = ϕ[1][2] 
        
                # Update hyperparameters 
                H = [γ1 1.0 0.0 0.0; γ2 1.0 1.0 0.0]
                A = zeros(2, 4)
                F = [ϕ1 0.0 0.0 ϕ2; 0.0 ψ1 0.0 0.0; 0.0 0.0 ψ2 0.0; 1.0 0.0 0.0 0.0]
                μ = [0.0, 0.0, 0.0, 0.0]
                R = zeros(2, 2)
                Q = zeros(4, 4)
                Q[1, 1] = 1.0
                Q[2, 2] = σ2_1[1]
                Q[3, 3] = σ2_2[1] 
                Z = zeros(4, 4)
        
                #  Update factor estimate 
                factor = dynamicFactorGibbsSampler(data_y, data_z, H, A, F, μ, R, Q, Z)
                factor = factor[:,1] 
                print(factor) 
        end

        return factor
end

factor_estimate = HDFMStateSpaceGibbsSampler(data_y, data_z)

factor = rand(size(data_y)[1])

Y = data_y[:, 1]
X = reshape(factor, length(factor), 1)
error_lag_num = 1
autocorrErrorRegGibbsSampler(Y, X, error_lag_num)

Y       = data_y[:, 1]
X       = reshape(factor, length(factor), 1)
σ2      = 1
staticLinearGibbsSamplerRestrictedVariance(Y, X, σ2)