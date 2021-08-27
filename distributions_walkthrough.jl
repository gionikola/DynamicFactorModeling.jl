```
Starting with a normal distribution.
```

using Random, Distributions

Random.seed!(123) # Setting the seed 

d = Normal()

x = rand(d, 100)

quantiles = quantile.(d, [0.5, 0.95])

y = rand(Normal(1,2), 100)

``` 
Using other distributions.
``` 
