"""
cd("/mnt/lustre/helios-home/maskomic/Desktop/VAE/pulsars")
using DrWatson
@quickactivate("pulsars")
include(scriptsdir("init_pulsars.jl"))
"""

# packages
using DrWatson
using Flux
using Distributions
using LinearAlgebra
using DataFrames
using EvalCurves
using CSV
using BSON
using IPMeasures

# include predefined functions
include(scriptsdir("functions.jl"))
include(scriptsdir("run_me.jl"))

# DATA
pulsars = CSV.read(datadir("datasets","htru2.csv"))
#kdd = CSV.read(datadir("datasets","kdd10_http.csv"))

# manipulation with data 
grouped_label = groupby(pulsars,:label);
normal = grouped_label[(label=0,)];
anomaly = grouped_label[(label=1,)];
dataset = pulsars[:,Not(:label)] |> Array
ndat = size(dataset,1)
normal_count = size(normal,1)
anomaly_count = ndat - normal_count

# only normal data for training - only the first 10000?
train_size = 10000
tmp = normal[:,Not(:label)]
DT = tmp[1:train_size,:] |> Array

# get labels
labels = pulsars[:,:label] 

# training data
dataT = [dataset[i,:] for i in 1:ndat ] 
data_norm = [DT[i,:] for i in 1:train_size] 
# data_train = zip(dataT,)  # data that fits to Flux.train! function 
data_train = zip(data_norm,)

# test dataset
AN = anomaly[:,Not(:label)] 
TT = normal[train_size+1:end,Not(:label)]
test = vcat(AN,TT) |> Array  
test_size = size(test,1)
test_data = [test[i,:] for i in 1:test_size]
test_labels = vcat(ones(anomaly_count),zeros(test_size-anomaly_count))

println("Data imported and ready :)")
