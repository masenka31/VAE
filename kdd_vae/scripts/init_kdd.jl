# init

"""
cd("/mnt/lustre/helios-home/maskomic/Desktop/VAE/kdd_vae")
using DrWatson
@quickactivate("kdd_vae")
include("scripts/init_kdd.jl")
"""

# inside kdd_vae project 
# packages
using DrWatson
using Plots
plotlyjs();
using Flux
using Distributions
using LinearAlgebra
using DataFrames
using DelimitedFiles
using EvalCurves
using CSV
using BSON

# include predefined functions
include("functions.jl")
include("run_me.jl")

# DATA
#pulsars = CSV.read(datadir("datasets","htru2.csv"))
kdd = CSV.read(datadir("datasets","kdd10_http.csv"))

# manipulation with data 
grouped_label = groupby(kdd,:label);
normal = grouped_label[(label="normal.",)];
dataset = kdd[!,[:duration,:src_bytes,:dst_bytes]] |> Array # most important collumns  
ndat = size(dataset,1)
normal_count = size(normal,1)
anomaly_count = ndat - normal_count
dataset2 = kdd[:,23:41] |> Array
dataset3 = hcat(dataset,dataset2) # main dataset, 22 entries, all continuous  

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

# training data
dataT = [ dataset3[i,:] for i in 1:ndat] # dataset as vectors in a vector  
data_train = zip(dataT,)                 # data that fits to Flux.train! function 

println("Data imported and ready :)")
