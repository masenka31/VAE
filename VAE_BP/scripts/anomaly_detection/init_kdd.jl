"""
To start, initialize DrWatson project "VAE_BP":
using DrWatson
@quickactivate("VAE_BP")
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

# DATA IMPORT
kdd = CSV.read(datadir("datasets","kdd10_http.csv"))

# manipulation with data 
grouped_label = groupby(kdd,:label);
normal = grouped_label[(label="normal.",)];
dataset = kdd[!,[:duration,:src_bytes,:dst_bytes]] |> Array
ndat = size(dataset,1)
normal_count = size(normal,1)
anomaly_count = ndat - normal_count
dataset2 = kdd[:,23:41] |> Array
dataset3 = hcat(dataset,dataset2) # main dataset, 22 entries, all continuous, all data (normal+anomalies)

# only normal data for training
dat_norm1 = normal[1:normal_count,[:duration,:src_bytes,:dst_bytes]]
dat_norm2 = normal[1:normal_count,23:41]
dat_norm = hcat(dat_norm1,dat_norm2) |> Array

# only anomaly data
anomaly = filter(:label => x -> x != "normal.",kdd)
adat1 = anomaly[!,[:duration,:src_bytes,:dst_bytes]]
adat2 = anomaly[:,23:41]
dat_anomaly = hcat(adat1,adat2) |> Array

# split training and test data, and anomaly data
train_size = 20000
dataT = [ dataset3[i,:] for i in 1:ndat]                            # all data
dataN_train = [dat_norm[i,:] for i in 1:train_size]                 # only normal data for training (train_size of data)
dataN_test = [dat_norm[i,:] for i in train_size+1:normal_count]     # only normal data for testing
dataN = vcat(dataN_train,dataN_test)                                # all normal data
dataA = [dat_anomaly[i,:] for i in 1:anomaly_count]                 # all anomaly data
data_test = vcat(dataN_test,dataA)                                  # test data normal + anomalies
# data_train = zip(dataT,)                                          # data that fits to Flux.train! function 
# data_train = zip(dataN_train,)
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
test_labels = vcat(zeros(normal_count-train_size),ones(anomaly_count))

println("Data imported and ready :)")
