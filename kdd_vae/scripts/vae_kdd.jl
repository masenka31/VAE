"""
It is important to correctly define the model
and be sure that everything is saved correctly.
If you change the model, be sure to change the
dictionary that is being saved at the end!
"""

# data load
include(scriptsdir("init_kdd.jl"))

# parameters
nx = 22
nh = 30
nz = 8
s = 100
ep = 500

# create the neural networks
A, μ, logs = Dense(nx, nh, σ), Dense(nh,nz), Dense(nh,nz)
f = Chain(Dense(nz, nh, σ), Dense(nh, nx));
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ));
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs);
ps = Flux.params(A, μ, logs, f);
opt = ADAM(0.01)

# loss function
function loss(x)
    HID = A(x);
    zsample = z(μ(HID), logs(HID));
    0.5 * sum((x .- f(zsample)).^2) + s * KL(μ(HID), logs(HID))
end

x = dataN_train[1]

# training procedure
@time for i in 1:ep
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
        safesave(datadir("KDD_test", savename("test_vs_train-2", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 