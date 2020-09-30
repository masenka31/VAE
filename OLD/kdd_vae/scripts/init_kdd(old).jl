# init

"""
cd("/mnt/lustre/helios-home/maskomic/Desktop/VAE/kdd_vae")
using DrWatson
@quickactivate("kdd_vae")
include(scriptsdir("init_kdd.jl"))
"""

# inside kdd_vae project 
# packages
using DrWatson
# using Plots
# plotlyjs();
using Flux
using Distributions
using LinearAlgebra
using DataFrames
using DelimitedFiles
using EvalCurves
using CSV
using BSON

# include predefined functions
include(scriptsdir("functions.jl"))
include(scriptsdir("run_me.jl"))

# DATA
kdd = CSV.read(datadir("datasets","kdd10_http.csv"))

# manipulation with data 
grouped_label = groupby(kdd,:label);
normal = grouped_label[(label="normal.",)];
dataset = kdd[!,[:duration,:src_bytes,:dst_bytes]] |> Array # most important collumns  
ndat = size(dataset,1)
normal_count = size(normal,1)
anomaly_count = ndat - normal_count
dataset2 = kdd[:,23:41] |> Array
dataset3 = hcat(dataset,dataset2) # main dataset, 22 entries, all continuous, all data (normal+anomalies)

# only normal data for training - only the first 20000?
dat_norm = normal[1:20000,[:duration,:src_bytes,:dst_bytes]]
dat_norm2 = normal[1:20000,23:41]
DT = hcat(dat_norm,dat_norm2) |> Array 

# training data
dataT = [ dataset3[i,:] for i in 1:ndat] # all data - dataset as vectors in a vector
dataN = [DT[i,:] for i in 1:20000]       # only normal data 
# data_train = zip(dataT,)               # data that fits to Flux.train! function 
data_train = zip(dataN,)

# get labels
labels = zeros(ndat)
labels_rev = ones(ndat)
for i in 1:ndat
    if kdd[i,:label] != "normal."
        labels[i] = 1
        labels_rev[i] = 0
    end
end
# labels_bool = labels |> BitVector # in case we needed labels in bool format  

println("Data imported and ready :)")
