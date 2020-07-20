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

# DATA
pulsars = CSV.read(datadir("datasets","htru2.csv"))

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
dat_norm_train = tmp[1:train_size,:] |> Array
dat_norm_test = tmp[train_size+1:normal_count,:] |> Array  

# get labels
labels = pulsars[:,:label] 

# training data
dataT = [dataset[i,:] for i in 1:ndat ]                                 # whole dataset 
dataN_train = [dat_norm_train[i,:] for i in 1:train_size]               # train dataset 
dataN_test = [dat_norm_test[i,:] for i in 1:normal_count-train_size]    # test normal
dataN = vcat(dataN_train,dataN_test)
# data_train = zip(dataT,)  # data that fits to Flux.train! function 
data_train = zip(dataN,)
# data_train = zip(dataN_train,)

AN = anomaly[:,Not(:label)] |> Array                                    # anomalies data
dataA = [AN[i,:] for i in 1:anomaly_count]                              # anomalies for loss  

data_test = vcat(dataN_test,dataA)                                      # test dataset 
test_labels = vcat(zeros(normal_count-train_size),ones(anomaly_count))  # test labels 


println("Data imported and ready :)")
