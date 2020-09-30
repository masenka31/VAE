using DrWatson
@quickactivate("WAE")
; cd Desktop/VAE/WAE
using IPMeasures
using Plots
# using PlotlyJS
plotlyjs();
using LinearAlgebra
using Distributions
using Flux
using BSON
using DataFrames

ndat = 500
X = rand(MvNormal([3;5],I),ndat)
X = rand(MvNormal([0;0],I),ndat)
x = X[1,:]
y = X[2,:]
x = 2 .* x .+ y
y = y .+ x
x = 0.5 .* x .+ y .^2
y = 2 .* y
X = vcat(x',y')
dataT = [[x[i] y[i]]' for i in 1:ndat]
data_train = zip(dataT,)
scatter(x,y)

include(scriptsdir("wae_functions.jl"))