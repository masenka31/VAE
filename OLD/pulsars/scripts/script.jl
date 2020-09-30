all_parameters = Dict(
    "run_id" => 4, #[1,2,3,4],
    "nx" => 8,
    "nz" => [2,5,8],
    "nh" => [2,5,10,30],
    "η"  => [0.01,0.001],
    "opts" => [ADAM, RMSProp],
    "activ" => [σ, swish],
    "s" => [1, 0.1, 10]
)

for j in 1:size(dict_list(all_parameters),1)
    par = dict_list(all_parameters)[j]
    run_exp(par,dataT,50) 
end

res = collect_results(datadir("results"))


L = 20
function rec_prob(x)
    HID = A(x);
    zsample = hcat([rand(MvNormal(μ(HID),logs(HID))) for i in 1:L]...)
    E = hcat([f(zsample[:,i]) for i in 1:L]...)
    1/L*sum([pdf(MvNormal(E[:,i],s*I),x) for i in 1:L])
end

function rec_prob(x)
    HID = A(x);
    zsample = hcat([z(μ(HID), logs(HID)) for i in 1:L]...)
    E = hcat([f(zsample[:,i]) for i in 1:L]...)
    1/L*sum([pdf(MvNormal(E[:,i],s*I),x) for i in 1:L])
end

function rec_loss(x)
    HID = A(x)
    zsample = z(μ(HID), logs(HID))
    0.5 / s * sum((x .- f(zsample)).^2)
end

model = BSON.load(datadir("final/final_model_auc_end=0.832_ep=400_s=1.bson"))
@unpack ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s = model 

LL = loss.(test_data)
RP = rec_prob.(test_data)
RL = rec_loss.(test_data)
fpr,tpr = roccurve(LL,test_labels)
fpr,tpr = roccurve(RP,test_labels)
fpr,tpr = roccurve(RL,test_labels)
auc_test = auc(fpr,tpr)

function encoding(x)
    HID = A(x)
    zs = z(μ(HID),logs(HID))
end

function data_encoding(dataT)
    return hcat([encoding(dataT[i]) for i in 1:size(dataT,1)]...)
end

enc_normal = data_encoding(test_data[1:anomaly_count])
enc_anomaly = data_encoding(test_data[anomaly_count+1:end])