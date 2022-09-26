function draw_coefficients(Y::Matrix{Float64}, X::Matrix{Float64}, σ2::Float64)

    ## Prior parameters in N(β0,Σ0)
    β0 = zeros(size(X)[2])
    Σ0 = Matrix(I, size(β0)[1], size(β0)[1]) .* 1000.0

    ## Posterior parameters in N(β1,Σ1) 
    β1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X) * (inv(Σ0) * β0 + inv(σ2) * transpose(X) * Y) |> vec 
    #β1 = vec(β1)
    Σ1 = inv(inv(Σ0) + inv(σ2) * transpose(X) * X) |> Hermitian 
    #Σ1 = Hermitian(Σ1)

    ## Return draw of β
    return β = mvn(β1, Σ1)::Vector{Float64}
end 