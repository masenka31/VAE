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
nh1 = 40
nh2 = 20
nh = 2
nz = 2
s = 1
ep = 400

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

# saving the results
FPR = []
TPR = []
auc_max, k_max = missing, missing
auc_progress = []

# training procedure
l_prev = loss(dataT[1])
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    L = loss.(dataT)
    global l = loss(dataT[1])
    if isnan(l)
        "loss diverged into NaN, training terminated..."
        break
    end
    fpr, tpr = roccurve(L, labels)
    auc1 = auc(fpr, tpr)
    if i > 1
          # this way we save the best model there is throughtout training process 
        if auc1 > maximum(auc_progress)
            global FPR = fpr
            global TPR = tpr
            global auc_max = auc1
            global k_max = i
            global saved_model = @dict(A,μ,logs,f,opt) 
        end
    end
    if i == ep
        global FPR_end = fpr
        global TPR_end = tpr
        global auc_end = auc1
    end
    println("Epoch: $i, loss:$l, auc=$auc1, auc_max=$auc_max")
    global auc_progress = vcat(auc_progress, auc1)
end


println("Results: \n maximum AUC = $auc_max")

# save the final model after specified number of epochs
final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,nh,nz)
safesave(datadir("final", savename("final_model_swish(f)", final_model, "bson")), final_model)

# save the best model during training and the best results
if !isempty(saved_model)
    @unpack A, μ, logs, f, opt = saved_model
    result = @dict(auc_max,k_max,FPR,TPR,auc_progress,A,μ,logs,f,opt,nx,nz,nh,s)
    # result = @dict(auc_max,k_max,FPR,TPR,auc_progress,A,μ,logs,f,opt,nx,nz,nh1,nh2,s)
    safesave(datadir("best", savename("best_model_swish(f)", result, "bson")), result)
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 