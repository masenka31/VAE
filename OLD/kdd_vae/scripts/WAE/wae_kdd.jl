"""
Check the training epochs!!!
Check what you're saving!
"""

# load data
include(scriptsdir("init_kdd.jl"))
using IPMeasures

"""
model = BSON.load(datadir("KDD_test_WAE_gauss/test_vs_train-2_auc1=0.985_i=30_s=1.bson"))
@unpack i,A,μ,logs,f,opt,fpr,tpr,auc1,s = model
"""

# parameters
nx = 22
nz = 8
nh = 30;
γ = 0.01
# c = 1
s = 1
n_sample = 200
ep = 100

# model 
opt = Flux.ADAM(1e-2)
A, μ, logs = Dense(nx, nh, σ), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz,nh,σ),Dense(nh,nx))
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. *logs)
# kernel = IMQKernel(c)
kernel = GaussianKernel(γ)
ps = Flux.params(A, μ, logs, f)

# loss function 
function loss(x)
    HID = A(x);
    zsample = z(μ(HID),logs(HID))
    sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
    sample_z = hcat([z(μ(HID),logs(HID)) for i in 1:n_sample]...)
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample_rnd,sample_z)
end

x = dataN_train[1]

# training procedure
@time for i in 31:ep
    Flux.train!(loss, ps, data_train, opt)
    L = loss.(dataT)
    global l = loss(x)
    if isnan(l)
        "loss diverged into NaN, training terminated..."
        break
    end
    L_train = loss.(dataN_train)
    LA = loss.(dataA)
    avg_norm = sum(L_train) / train_size
    avg_an = sum(LA) / anomaly_count
    if i%5 == 0
        L = loss.(data_test)
        fpr, tpr = roccurve(L, test_labels)
        auc1 = auc(fpr,tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s)
        safesave(datadir("KDD_test_WAE_gauss", savename("test_vs_train-2", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

