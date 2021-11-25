using Plots, Distributions

function stock_price(num_obs=100, p_init=50, μ=0.01, σ=0.1)
    
    return_distr = Normal(μ,σ)
    return_obs   = rand(return_distr, num_obs) 
    price_obs    = zeros(num_obs)
    price_obs[1] = p_init 
    for i in 2:(num_obs) 
        price_obs[i] = price_obs[i-1]*(1+return_obs[i])
    end 
    return price_obs 
end 

function plot_price(stock_price)
    plot(stock_price, xlabel = "Days", ylabel = "Price (USD)", title="Daily Stock Price Chart", label="Stock X", fmt=:png)
end 

function gen_price_plot(num_obs=100, p_init=50, μ=0.01, σ=0.1)
    prices = stock_price(num_obs, p_init, μ, σ)
    plot_price(prices)
end 

function portfolio(num_obs=10000, w=0.5, μ=0.02, σ=0.05, μ1=0.01, σ1=0.01, μ2=0.01, σ2=0.06)
    common = Normal(μ,σ) 
    idio1 = Normal(μ1,σ1)
    idio2 = Normal(μ2,σ2)
    common_ret = rand(common, num_obs)
    return1 = rand(idio1, num_obs) + 0.3*common_ret
    return2 = rand(idio2, num_obs) - 0.6*common_ret
    portfolio_return = w*return1+(1-w)*return2 
    avg_return = mean(portfolio_return)
    volatility = sqrt(var(portfolio_return))
    return avg_return, volatility 
end 

function portfolio_sim(num_obs=100000, μ=0.02, σ=0.05, μ1=0.01, σ1=0.01, μ2=0.01, σ2=0.06)
    volatility = zeros(100)
    avg_return = zeros(100) 
    weight     = zeros(100)
    for i in 1:100
        weight[i] = i/100
        avg_return[i], volatility[i] = portfolio(num_obs, weight[i] , μ, σ, μ1, σ1, μ2, σ2)
    end 
    return volatility, avg_return, weight 
end 

function plot_avg_return(num_obs=100000, μ=0.02, σ=0.05, μ1=0.01, σ1=0.01, μ2=0.01, σ2=0.06)
    volatility, avg_return, weight = portfolio_sim(num_obs, μ, σ, μ1, σ1, μ2, σ2)
    plot(weight, avg_return, xlabel = "Weight of Stock 1 in Portfolio", ylabel = "Expected Rate of Return", title="Expected Portfolio Returns", label="", fmt=:png)
end 

function plot_volatility(num_obs=100000, μ=0.02, σ=0.05, μ1=0.01, σ1=0.01, μ2=0.01, σ2=0.06)
    volatility, avg_return, weight = portfolio_sim(num_obs, μ, σ, μ1, σ1, μ2, σ2)
    plot(weight, volatility, xlabel = "Weight of Stock 1 in Portfolio", ylabel = "Std. Dev of Returns", title="Portfolio Volatility", label="", fmt=:png)
end 
