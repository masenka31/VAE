"""
Initialize DrWatson project.

using DrWatson
@quickactivate("components")
"""

include(scriptsdir("funkce.jl"))
include(scriptsdir("KDD/init_kdd.jl"))

"""
model = BSON.load(datadir("KDD/emvae_auc_end=0.978_i=30_s=0.1.bson"))
@unpack i,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s = model
"""

# initialize network
nx = 22
nz = 8
nh = 30
comp = 5
α = [1 / comp for i in 1:comp] # no prior information
s = 1
b = 1 / s
ep = 200

opt = ADAM(0.01)
A = [Dense(nx, nh, σ) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, σ), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))


function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

# training procedure
x = dataN_train[1]
l_prev = loss(x)

global LI = lik_n(dataN_train)                # count the l_ik matrix
global α = sum(LI, dims = 1) / train_size     # count α

@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global LI = lik_n(dataN_train)                # count the l_ik matrix
    global α = sum(LI, dims = 1) / train_size     # count α
    global l = loss(x)
    if isnan(l)
        println("loss diverged into NaN, training terminated...")
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
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,α)
        safesave(datadir("KDD", savename("test_vs_train-2", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    n_vec = sum(round.(LI),dims=1)
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an)), n: $n_vec")
end

# save the final model after specified number of epochs
# final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,c,n_sample)
# safesave(datadir("WAE_final", savename("final_wae_gaussian", final_model, "bson")), final_model)

printstyled("Current experiment completed.\n", bold = true, color = :cyan)
