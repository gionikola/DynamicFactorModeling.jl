function vardecomp2level(data, factor, betas, factorassign)

    T, N = size(data)
    T, Nf = size(factor)

    # Record data variances 
    varvars = zeros(N)
    for i in 1:N
        varvars[i] = var(data[:, i])
    end

    # Record factor estimate variances 
    varfacs = zeros(Nf)
    for j in 1:Nf
        varfacs[j] = var(factor[:, j])
    end

    # Compute variance decompositions 
    vardecomps = zeros(N, 2)
    for i in 1:N
        vardecomps[i, 1] = (betas[i, 2]^2 * varfacs[1]) / varvars[i]
        vardecomps[i, 2] = (betas[i, 3]^2 * varfacs[1+factorassign[i, 2]]) / varvars[i]
    end

    return vardecomps
end

