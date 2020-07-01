"""
using DrWatson
@quickactivate"MixtureVAE")
include(scripts("dirinit_mixtureVAE.jl")
"""

using Plots
using Flux
using Distributions
using LinearAlgebra
plotlyjs();

X = rand(MvNormal([0;0], I), 100)
Y = rand(MvNormal([15;15], I), 100)
# Z = rand(MvNormal([-15;-15], I), 100)
dt = vcat(X', Y')
scatter(dt[:,1],dt[:,2])
ndat = size(dt, 1)
dataT = [dt[i,:] for i in 1:ndat]
opt = ADAM()
α = [1 / comp for i in 1:comp] # no prior information
s = 0.04
b = 1 / s

ts1 = generate_data(100,nz,2,f[1])
ts2 = generate_data(100,nz,2,f[2])
ts3 = generate_data(100,nz,2,f[3])

scatter(dt[:,1],dt[:,2],legend=:bottomright,label="data");
scatter!(ts1[:,1],ts1[:,2],label="f1");
scatter!(ts2[:,1],ts2[:,2],label="f2");
scatter!(ts3[:,1],ts3[:,2],label="f3")

gen = vcat(ts1,ts2,ts3)
scatter(dt[:,1],dt[:,2],zcolor=LI[:,1],legend=:bottomright,label="data")
scatter!(gen[:,1],gen[:,2])