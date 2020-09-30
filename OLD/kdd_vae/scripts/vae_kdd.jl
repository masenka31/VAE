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
nz = 10
s = 100
ep = 400

# create the neural networks
A, μ, logs = Dense(nx, nh, σ), Dense(nh,nz), Dense(nh,nz)
f = Chain(Dense(nz, nh, σ), Dense(nh, nx));
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ));
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs);
ps = Flux.params(A, μ, logs, f);
opt = ADAM(0.005)

# loss function
function loss(x)
    HID = A(x);
    zsample = z(μ(HID), logs(HID));
    0.5 * sum((x .- f(zsample)).^2) + s * KL(μ(HID), logs(HID))
end

rec_K = 50
function rec_loss(x)
    HID = A(x)
    sample = hcat([z(μ(HID), logs(HID)) for i in 1:rec_K]...)
    X = hcat([f(sample[:,i]) for i in 1:rec_K]...)
    Y = vcat([sum((x .- X[:,i]).^2) for i in 1:rec_K]...)
    mean(Y)
end

x = dataN_train[1]

# training procedure
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global l = loss(x)
    if isnan(l)
        "loss diverged into NaN, training terminated..."
        break
    end
    L_train = loss.(dataN)
    LA = loss.(dataA)
    avg_norm = mean(L_train)
    avg_an = mean(LA)
    if i%5 == 0
        L = loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr,tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s)
        safesave(datadir("ALL_DATA", savename("run-hard-4", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

