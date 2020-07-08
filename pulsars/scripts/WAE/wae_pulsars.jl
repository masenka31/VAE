"""
Check the training epochs!!!
"""

# load data
include(scriptsdir("init_pulsars.jl"))

"""
model = BSON.load(datadir("WAE_final/final_wae_auc_end=0.984_c=1_ep=40_n_sample=200_s=1.bson"))
@unpack ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,c,n_sample = model
""" 

# parameters
nx = 8
nz = 4
nh = 30;
# γ = 0.01
c = 1
s = 1
n_sample = 200
ep = 25

# model 
opt = Flux.ADAM(1e-2)
A, μ, logs = Dense(nx, nh, σ), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz,nh,σ),Dense(nh,nx))
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. *logs)
kernel = IMQKernel(c)
ps = Flux.params(A, μ, logs, f)

# loss function 
function loss(x)
    HID = A(x);
    zsample = z(μ(HID),logs(HID))
    sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
    sample_z = hcat([z(μ(HID),logs(HID)) for i in 1:n_sample]...)
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample_rnd,sample_z)
end

# training procedure
l_prev = loss(dataT[1])
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global l = loss(dataT[1])
    if isnan(l)
        "loss diverged into NaN, training terminated..."
        break
    end
    if i == ep
        L = loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr, tpr)
        global FPR_end = fpr
        global TPR_end = tpr
        global auc_end = auc1
    end
    println("Epoch: $i, loss:$l")
end

println("Results: \n maximum AUC = $auc_end")

# save the final model after specified number of epochs
final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,c,n_sample)
safesave(datadir("WAE_final", savename("final_wae", final_model, "bson")), final_model)

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

