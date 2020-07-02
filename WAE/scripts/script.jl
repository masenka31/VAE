## Generate data
ndat = 500
X = rand(MvNormal([0;0],I),ndat)
x = X[1,:]
y = X[2,:]
# otočená
# x = 2 .* x .+ y
# y = y .+ x
# X = vcat(x',y')
# podkova 
x = 0.5 .* x .+ y .^2
y = 2 .* y
X = vcat(x',y')
# scatter(x,y,aspect_ratio=:equal)

include(scriptsdir("wae_functions.jl"))

dataT = [[x[i] y[i]]' for i in 1:ndat]
data_train = zip(dataT,)

# klasická
s = 10
include(scriptsdir("wasserstein_run.jl"))
include(scriptsdir("wasserstein_run.jl"))
s = 1
include(scriptsdir("wasserstein_run.jl"))
s = 0.01
include(scriptsdir("wasserstein_run.jl"))
include(scriptsdir("wasserstein_run.jl"))

# otočená 
s = 10
include(scriptsdir("otocena.jl"))
include(scriptsdir("otocena.jl"))
s = 1
include(scriptsdir("otocena.jl"))
include(scriptsdir("otocena.jl"))
s = 0.01
include(scriptsdir("otocena.jl"))
include(scriptsdir("otocena.jl"))

# podkova 
s = 10
include(scriptsdir("podkova.jl"))
include(scriptsdir("podkova.jl"))
s = 1
include(scriptsdir("podkova.jl"))
include(scriptsdir("podkova.jl"))
s = 0.1
include(scriptsdir("podkova.jl"))
include(scriptsdir("podkova.jl"))

