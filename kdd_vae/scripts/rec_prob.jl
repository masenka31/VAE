# parameters
nx = 22
nh = 30
nz = 8
s = 0.5
ep = 400

# create the neural networks
A, μ, logs = Dense(nx, nh, σ), Dense(nh, nz), Dense(nh, nz);
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
L = 20
function rec(x)
    HID = A(x);
    zsample = hcat([z(μ(HID), logs(HID)) for i in 1:L]...)
    1 / L * sum([0.5 / s * sum((x .- f(zsample[:,i])).^2) for i in 1:L])
end

RR = rec_prob.(dataT)

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
        if abs(auc_progress[i - 1] - auc1) < 0.000001
            println("no significant change of AUC, training terminated...")
            break
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
final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end)
safesave(datadir("final", savename("final_model", final_model, "bson")), final_model)

# save the best model during training and the best results
if !isempty(saved_model)
    @unpack A, μ, logs, f, opt = saved_model
    result = @dict(auc_max,k_max,FPR,TPR,auc_progress,A,μ,logs,f,opt,nx,nz,nh,s)
    safesave(datadir("best", savename("best_model", result, "bson")), result)
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 