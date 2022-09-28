include("draw_coefficients.jl")
include("draw_error_variance.jl")
include("draw_parameters.jl")

function regress(Y::Matrix{Float64}, X::Matrix{Float64}, p::Int, iter::Int, burnin::Int)

    βsave = zeros(iter, size(X)[2])[:, :]
    ϕsave = zeros(iter, p)[:, :]
    σ2save = zeros(iter)

    βinit = zeros(size(X)[2])
    ϕinit = zeros(p)
    σ2init = 1.0

    β = βinit
    ϕ = ϕinit
    σ2 = σ2init

    for i in 1:iter
        β, ϕ, σ2 = draw_parameters(Y, X, ϕ, σ2)
        βsave[i, :] = β'
        ϕsave[i, :] = ϕ'
        σ2save[i] = σ2
    end

    #Eβ = mean(βsave[(burnin+1):end,:], dims = 1)
    #Eϕ = mean(ϕsave[(burnin+1):end,:], dims = 1)
    #Eσ2 = mean(σ2save[(burnin+1):end])

    #return samples for β, ϕ, σ2
    return βsave[(burnin+1):end, :]::Matrix{Float64}, ϕsave[(burnin+1):end, :]::Matrix{Float64}, σ2save[(burnin+1):end]::Vector{Float64}
end

function regress(Y::Vector{Float64}, X::Matrix{Float64}, p::Int, iter::Int, burnin::Int)

    βsave = zeros(iter, size(X)[2])[:, :]
    ϕsave = zeros(iter, p)[:, :]
    σ2save = zeros(iter)

    βinit = zeros(size(X)[2])
    ϕinit = zeros(p)
    σ2init = 1.0

    β = βinit
    ϕ = ϕinit
    σ2 = σ2init

    for i in 1:iter
        β, ϕ, σ2 = draw_parameters(Y, X, ϕ, σ2)
        βsave[i, :] = β'
        ϕsave[i, :] = ϕ'
        σ2save[i] = σ2
    end

    #Eβ = mean(βsave[(burnin+1):end,:], dims = 1)
    #Eϕ = mean(ϕsave[(burnin+1):end,:], dims = 1)
    #Eσ2 = mean(σ2save[(burnin+1):end])

    #return samples for β, ϕ, σ2
    return βsave[(burnin+1):end, :]::Matrix{Float64}, ϕsave[(burnin+1):end, :]::Matrix{Float64}, σ2save[(burnin+1):end]::Vector{Float64}
end 