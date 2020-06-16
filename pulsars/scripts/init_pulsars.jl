"""
cd("/mnt/lustre/helios-home/maskomic/Desktop/VAE/pulsars")
using DrWatson
@quickactivate("pulsars")
include(scriptsdir("init_pulsars.jl"))
"""

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
include(scriptsdir("functions.jl"))
include(scriptsdir("run_me.jl"))

# DATA
pulsars = CSV.read(datadir("datasets","htru2.csv"))
#kdd = CSV.read(datadir("datasets","kdd10_http.csv"))

# manipulation with data 
grouped_label = groupby(pulsars,:label);
normal = grouped_label[(label=0,)];
dataset = pulsars[:,Not(:label)] |> Array
ndat = size(dataset,1)
normal_count = size(normal,1)
anomaly_count = ndat - normal_count

# get labels
labels = pulsars[:,:label] 

# training data
dataT = [dataset[i,:] for i in 1:ndat ] 
data_train = zip(dataT,)  # data that fits to Flux.train! function 

println("Data imported and ready :)")