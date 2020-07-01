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
x = X[1,:]
y = X[2,:]
x = 2 .* x .+ y
y = y .+ x
dataT = [[x[i] y[i]]' for i in 1:ndat]
data_train = zip(dataT,)

all_parameters = Dict(
    "nx" => 2,
    "nz" => 2,
    "nh" => [2,4,5],
    "s" => [1, 0.1, 0.05, 0.01],
    "γ" => [0.1, 0.01],
    "n_sample" => [50,100]
)

include(scriptsdir("run_wae.jl"))

@time for j in 1:size(dict_list(all_parameters),1)
    par = dict_list(all_parameters)[j]
    run_me(par,dataT,50) 
end


function reconstruct_data(x)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID))
    return f(zsample)
end